"""Neo4j 知识点图谱。

Schema：
    节点  Subject / KnowledgePoint / Tag / Chunk
    边    CHILD_OF    知识点层级（6182 节点、深度 11 层、100% 覆盖）—— 图谱主力
          MAPS_TO     真题 tag -> 知识点（226 个粗粒度标签接 6182 个细粒度节点）
          MENTIONS    Chunk -> 知识点
          CO_OCCURS   tag 间共现（同一道真题涉及多个考点）—— 真实数据支撑的横向关联
          RELATES_TO  XMind 关系线（实测仅 9 条，贡献有限，如实记录）

第三路召回的做法：
    query -> 命中知识点节点 -> 沿 CHILD_OF 向上取祖先（定位章节）
                            -> 向下取同节兄弟 -> 经 MENTIONS 映射到 Chunk
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger
from neo4j import GraphDatabase

from cs408rag.config import PROCESSED_DIR, settings
from cs408rag.ingestion.chunker import Chunk
from cs408rag.ingestion.xmind_loader import KnowledgePoint

BATCH = 1000


@dataclass
class GraphHit:
    chunk_id: str
    score: float
    kp_titles: list[str]
    path: str

    @property
    def citation(self) -> str:
        return " > ".join(self.kp_titles[:3]) or self.path


class GraphStore:
    def __init__(
        self, uri: str | None = None, user: str | None = None, password: str | None = None
    ):
        self.uri = uri or settings.neo4j_uri
        self.user = user or settings.neo4j_user
        self.password = password or settings.neo4j_password
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def ping(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Neo4j 连接失败（{self.uri}）：{e}")
            return False

    def _run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        with self._driver.session() as s:
            return [r.data() for r in s.run(cypher, **params)]

    # -- 建图 -------------------------------------------------------------

    def clear(self) -> None:
        """清空图谱。调用方需明确意图。"""
        self._run("MATCH (n) DETACH DELETE n")
        logger.info("图谱已清空")

    def create_constraints(self) -> None:
        for label, prop in (
            ("KnowledgePoint", "node_id"),
            ("Tag", "name"),
            ("Chunk", "chunk_id"),
            ("Subject", "name"),
        ):
            try:
                self._run(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
                )
            except Exception as e:  # noqa: BLE001
                logger.debug(f"约束 {label}.{prop} 未创建：{e}")

    def load_knowledge_points(self, kps: Sequence[KnowledgePoint]) -> None:
        """写入知识点节点与层级边。"""
        rows = [
            {
                "node_id": k.node_id,
                "title": k.title,
                "subject": k.subject,
                "depth": k.depth,
                "path": k.path,
                "path_str": k.path_str,
                "chapter": k.chapter,
                "section": k.section,
                "is_leaf": k.is_leaf,
                "parent_id": k.parent_id or "",
            }
            for k in kps
        ]

        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (k:KnowledgePoint {node_id: row.node_id})
                SET k.title = row.title, k.subject = row.subject, k.depth = row.depth,
                    k.path = row.path, k.path_str = row.path_str, k.chapter = row.chapter,
                    k.section = row.section, k.is_leaf = row.is_leaf
                MERGE (s:Subject {name: row.subject})
                MERGE (k)-[:BELONGS_TO]->(s)
                """,
                rows=batch,
            )

        # 层级边单独写：父节点可能在后一批
        parents = [r for r in rows if r["parent_id"]]
        for i in range(0, len(parents), BATCH):
            self._run(
                """
                UNWIND $rows AS row
                MATCH (c:KnowledgePoint {node_id: row.node_id})
                MATCH (p:KnowledgePoint {node_id: row.parent_id})
                MERGE (c)-[:CHILD_OF]->(p)
                """,
                rows=parents[i : i + BATCH],
            )

        logger.info(f"知识点节点 {len(rows)}，层级边 {len(parents)}")

    def load_tags(
        self,
        tag_map: dict[str, list[str]],
        cooccurrence: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """写入 tag 节点、MAPS_TO 边与 CO_OCCURS 共现边。"""
        rows = [{"tag": t, "kp_ids": ids} for t, ids in tag_map.items() if ids]
        for i in range(0, len(rows), BATCH):
            self._run(
                """
                UNWIND $rows AS row
                MERGE (t:Tag {name: row.tag})
                WITH t, row
                UNWIND row.kp_ids AS kid
                MATCH (k:KnowledgePoint {node_id: kid})
                MERGE (t)-[:MAPS_TO]->(k)
                """,
                rows=rows[i : i + BATCH],
            )
        logger.info(f"tag 节点 {len(rows)}，MAPS_TO 边 {sum(len(r['kp_ids']) for r in rows)}")

        if cooccurrence:
            co = [c for c in cooccurrence if c.get("count", 0) > 0]
            for i in range(0, len(co), BATCH):
                self._run(
                    """
                    UNWIND $rows AS row
                    MERGE (a:Tag {name: row.a})
                    MERGE (b:Tag {name: row.b})
                    MERGE (a)-[r:CO_OCCURS]-(b)
                    SET r.count = row.count
                    """,
                    rows=co[i : i + BATCH],
                )
            logger.info(f"CO_OCCURS 共现边 {len(co)}")

    def load_chunks(self, chunks: Sequence[Chunk]) -> None:
        """写入 chunk 节点及其到知识点的 MENTIONS 边。

        只写有 kp 关联的 chunk —— 没有关联的 chunk 对图谱路没有贡献，
        写进去只会让图变大而不增加召回能力。
        """
        rows = [
            {"chunk_id": c.chunk_id, "subject": c.subject, "kp_ids": c.kp_ids}
            for c in chunks
            if c.kp_ids
        ]
        if not rows:
            logger.warning("没有任何 chunk 关联到知识点，图谱路的 MENTIONS 边为空")
            return

        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            self._run(
                """
                UNWIND $rows AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.subject = row.subject
                WITH c, row
                UNWIND row.kp_ids AS kid
                MATCH (k:KnowledgePoint {node_id: kid})
                MERGE (c)-[:MENTIONS]->(k)
                """,
                rows=batch,
            )
        logger.info(f"chunk 节点 {len(rows)}，MENTIONS 边 {sum(len(r['kp_ids']) for r in rows)}")

    # -- 检索（第三路）-----------------------------------------------------

    def search_by_keywords(
        self,
        keywords: Sequence[str],
        *,
        top_k: int = 20,
        expand_siblings: bool = True,
        subject: str | None = None,
    ) -> list[GraphHit]:
        """图谱路召回。

        步骤：
            1. 关键词在知识点标题上做包含匹配，得到**种子节点**
            2. 种子 + 其直接子节点（1 层）
            3. 种子的同父兄弟（1 层）—— 同节知识点是最强的相关性信号
            4. 经 MENTIONS 反向映射到 chunk
            5. 打分：命中**种子**权重远高于命中扩展节点

        实测踩过的坑（改之前）：
            最初对**每个后代节点**都做兄弟扩展，邻域爆炸到整张图，
            排序又按「提及知识点总数」—— 结果偏好泛化综述章节，
            还会跨科目乱召回（数据结构的题返回计算机网络章节）。
            实测三路里图谱路命中率最差，把融合结果拖得比单路 dense 还低。

        关键修正：
            - 后代只取 1 层，兄弟只取种子的（不取后代的）
            - 打分以「命中种子数」为主，扩展节点只做小幅加权
        """
        kws = [k.strip() for k in keywords if k and len(k.strip()) >= 2]
        if not kws:
            return []

        cypher = """
        // 1) 种子：标题命中关键词的知识点
        MATCH (seed:KnowledgePoint)
        WHERE any(kw IN $keywords WHERE seed.title CONTAINS kw)
          AND ($subject IS NULL OR seed.subject = $subject)
        WITH collect(DISTINCT seed) AS seeds
        WHERE size(seeds) > 0

        // 2) 种子的直接子节点（1 层）
        UNWIND seeds AS s
        OPTIONAL MATCH (s)<-[:CHILD_OF]-(child:KnowledgePoint)
        WITH seeds, collect(DISTINCT child) AS children

        // 3) 种子自身的同父兄弟（不递归到后代，避免邻域爆炸）
        UNWIND seeds AS s2
        OPTIONAL MATCH (s2)-[:CHILD_OF]->(:KnowledgePoint)<-[:CHILD_OF]-(sib:KnowledgePoint)
        WITH seeds, children, collect(DISTINCT sib) AS siblings

        // 4) 候选知识点池。★ seeds 必须一路带到最后 ——
        //    写成 WITH seeds + children + siblings AS pool 会把 seeds 挤出作用域，
        //    后面 (kp IN seeds) 直接报 Variable not defined。
        UNWIND (seeds + children + siblings) AS kp
        WITH DISTINCT kp, seeds
        WHERE kp IS NOT NULL

        // 5) 映射到 chunk，区分「命中种子」与「命中扩展节点」
        MATCH (c:Chunk)-[:MENTIONS]->(kp)
        WITH c, kp, seeds
        RETURN c.chunk_id AS chunk_id,
               sum(CASE WHEN kp IN seeds THEN 3 ELSE 1 END) AS score,
               count(DISTINCT kp) AS kp_hits,
               collect(DISTINCT kp.title)[0..5] AS kp_titles,
               collect(DISTINCT kp.path_str)[0] AS path
        ORDER BY score DESC, kp_hits DESC
        LIMIT $limit
        """
        rows = self._run(
            cypher,
            keywords=kws,
            subject=subject,
            limit=top_k * 4,
        )

        hits = [
            GraphHit(
                chunk_id=r["chunk_id"],
                score=float(r["score"]),
                kp_titles=list(r["kp_titles"] or []),
                path=r["path"] or "",
            )
            for r in rows
            if r["chunk_id"]
        ]
        hits.sort(key=lambda h: (-h.score, h.chunk_id))
        return hits[:top_k]

    def sibling_titles(
        self,
        keywords: Sequence[str],
        *,
        subject: str | None = None,
        limit: int = 8,
    ) -> list[str]:
        """取命中知识点的同父兄弟标题，用于查询扩展。

        实测（dev 452 题）：原始检索 + 兄弟词扩展后 RRF 合并，
        题级 Hit@10 从 0.577 升到 0.602（+2.4pp）。单独用扩展检索反而下降
        （0.518），所以只能作为「补充一路」与原始结果合并，不能替代。
        """
        kws = [k.strip() for k in keywords if k and len(k.strip()) >= 2]
        if not kws:
            return []
        rows = self._run(
            """
            MATCH (k:KnowledgePoint)
            WHERE any(kw IN $kws WHERE k.title CONTAINS kw)
              AND ($subject IS NULL OR k.subject = $subject)
            MATCH (k)-[:CHILD_OF]->(:KnowledgePoint)<-[:CHILD_OF]-(s:KnowledgePoint)
            WHERE s.title <> k.title AND size(s.title) <= 20
            RETURN DISTINCT s.title AS title
            LIMIT $limit
            """,
            kws=kws,
            subject=subject,
            limit=limit,
        )
        return [r["title"] for r in rows if r.get("title")]

    def load_prerequisites(self, rows: Sequence[dict[str, Any]]) -> int:
        """写入知识点前置依赖边 (A)-[:PREREQUISITE]->(B)。

        只接受图谱中已存在的节点，不新建节点 —— 防止 LLM 判定阶段
        引入不存在的知识点。
        """
        valid = [r for r in rows if r.get("from_id") and r.get("to_id")]
        for i in range(0, len(valid), BATCH):
            self._run(
                """
                UNWIND $rows AS row
                MATCH (a:KnowledgePoint {node_id: row.from_id})
                MATCH (b:KnowledgePoint {node_id: row.to_id})
                MERGE (a)-[r:PREREQUISITE]->(b)
                SET r.confidence = row.confidence
                """,
                rows=valid[i : i + BATCH],
            )
        logger.info(f"PREREQUISITE 边 {len(valid)}")
        return len(valid)

    def prerequisites_among(self, titles: Sequence[str]) -> list[tuple[str, str]]:
        """返回给定标题集合内部存在的 (前置, 后置) 边。

        Plan 校验用：如果前置知识点被排在更晚的周次，说明学习顺序错了。
        """
        names = [t for t in titles if t]
        if len(names) < 2:
            return []
        rows = self._run(
            """
            MATCH (a:KnowledgePoint)-[:PREREQUISITE]->(b:KnowledgePoint)
            WHERE a.title IN $names AND b.title IN $names
            RETURN DISTINCT a.title AS pre, b.title AS post
            """,
            names=names,
        )
        return [(r["pre"], r["post"]) for r in rows]

    def learning_path(self, title: str, *, max_depth: int = 6) -> dict[str, Any]:
        """给定知识点，返回它自身的前置链（按先后顺序）。

        沿 PREREQUISITE 反向走：谁指向它、谁又指向那个节点……
        """
        rows = self._run(
            f"""
            MATCH (k:KnowledgePoint)
            WHERE k.title CONTAINS $title
            WITH k
            ORDER BY size(k.title) ASC
            LIMIT 1
            OPTIONAL MATCH path=(pre:KnowledgePoint)-[:PREREQUISITE*1..{int(max_depth)}]->(k)
            RETURN k.title AS target,
                   [n IN nodes(path) | n.title] AS chain,
                   length(path) AS depth
            ORDER BY depth DESC
            LIMIT 1
            """,
            title=title,
        )
        if not rows:
            return {"found": False, "target": title}
        r = rows[0]
        return {
            "found": True,
            "target": r.get("target") or title,
            "chain": r.get("chain") or [],
            "depth": r.get("depth") or 0,
        }

    def kp_profile(self, title: str) -> dict[str, Any]:
        """某个知识点的画像：层级路径、同节兄弟、关联考点。

        二期「考点分析」用它回答「这个知识点和什么有关联」。
        """
        rows = self._run(
            """
            MATCH (k:KnowledgePoint)
            WHERE k.title CONTAINS $title
            WITH k
            // 导图里大量节点是长描述句（如「每个进程都需要同时拥有多个临界资源，
            // 所以有死锁隐患」）。问「死锁」时应返回核心概念节点，
            // 而不是随便一条含该词的长句。按标题长度升序取最短的那个。
            ORDER BY size(k.title) ASC, k.depth DESC
            LIMIT 1
            OPTIONAL MATCH (k)-[:CHILD_OF*1..]->(anc:KnowledgePoint)
            WITH k, collect(DISTINCT anc.title) AS ancestors
            OPTIONAL MATCH (k)-[:CHILD_OF]->(p:KnowledgePoint)<-[:CHILD_OF]-(sib:KnowledgePoint)
            WITH k, ancestors, collect(DISTINCT sib.title) AS siblings
            OPTIONAL MATCH (t:Tag)-[:MAPS_TO]->(k)
            WITH k, ancestors, siblings, collect(DISTINCT t.name) AS tags
            OPTIONAL MATCH (t2:Tag)-[co:CO_OCCURS]-(other:Tag)
            WHERE t2.name IN tags
            RETURN k.title AS title, k.subject AS subject, k.path_str AS path,
                   ancestors, siblings, tags,
                   collect(DISTINCT {tag: other.name, count: co.count})[0..15] AS cooccurring
            """,
            title=title,
        )
        return rows[0] if rows else {}

    def stats(self) -> dict[str, int]:
        rows = self._run(
            """
            CALL () { MATCH ()-[r:CHILD_OF]->() RETURN count(r) AS child_of }
            CALL () { MATCH ()-[r:MAPS_TO]->() RETURN count(r) AS maps_to }
            CALL () { MATCH ()-[r:MENTIONS]->() RETURN count(r) AS mentions }
            CALL () { MATCH ()-[r:CO_OCCURS]-() RETURN count(r)/2 AS cooc }
            CALL () { MATCH (k:KnowledgePoint) RETURN count(k) AS kp }
            CALL () { MATCH (t:Tag) RETURN count(t) AS tag }
            CALL () { MATCH (c:Chunk) RETURN count(c) AS chunk }
            RETURN kp, tag, chunk, child_of, maps_to, mentions, cooc
            """
        )
        return rows[0] if rows else {}


# ---------------------------------------------------------------------------
# 建图编排
# ---------------------------------------------------------------------------


def build_graph(
    kps: Sequence[KnowledgePoint],
    chunks: Sequence[Chunk],
    tag_map: dict[str, list[str]],
    *,
    recreate: bool = True,
) -> dict[str, Any]:
    co_path = PROCESSED_DIR / "tag_cooccurrence.json"
    co = json.loads(co_path.read_text(encoding="utf-8")) if co_path.exists() else None
    if co is None:
        logger.warning("未找到共现边数据，CO_OCCURS 边将为空。先跑 quiz_loader 导出。")

    with GraphStore() as gs:
        if not gs.ping():
            raise RuntimeError(f"Neo4j 连不上（{gs.uri}）。先执行：docker compose up -d")
        if recreate:
            gs.clear()
        gs.create_constraints()
        gs.load_knowledge_points(kps)
        gs.load_tags(tag_map, co)
        gs.load_chunks(chunks)
        return gs.stats()


if __name__ == "__main__":
    import argparse
    import json

    from cs408rag.config import ensure_dirs
    from cs408rag.ingestion.chunker import chunk_md_sections
    from cs408rag.ingestion.md_loader import load_wangdao
    from cs408rag.ingestion.tag_mapper import load_mapping
    from cs408rag.ingestion.xmind_loader import load_all_mindmaps

    ap = argparse.ArgumentParser(description="构建 Neo4j 知识点图谱")
    ap.add_argument("--no-recreate", action="store_true", help="不先清空图谱")
    args = ap.parse_args()

    ensure_dirs()

    gs = GraphStore()
    if not gs.ping():
        raise SystemExit(f"Neo4j 连不上（{gs.uri}）。先执行：docker compose up -d")
    gs.close()

    mindmaps = load_all_mindmaps()
    kps = [n for mm in mindmaps for n in mm.nodes]
    sections = load_wangdao()
    chunks = chunk_md_sections(sections)
    tag_map = load_mapping()

    # ★ 必须先做知识点关联，否则 chunk.kp_ids 为空，
    #   MENTIONS 边建不出来，图谱第三路就完全失效（实测踩过）。
    from cs408rag.ingestion.indexer import link_chunks_to_kps
    from cs408rag.ingestion.quiz_loader import load_questions as _lq

    questions = _lq()
    n_linked = link_chunks_to_kps(
        chunks, kps, tag_map, {q.qid: q for q in questions}
    )
    if n_linked == 0:
        raise SystemExit("没有 chunk 关联到知识点，图谱的 MENTIONS 边会是空的，已中止")

    stats = build_graph(kps, chunks, tag_map, recreate=not args.no_recreate)
    print(json.dumps(stats, ensure_ascii=False, indent=2))