"""检索链路编排。

把三路召回（dense / sparse / graph）、融合、重排串成一条可配置的链，
供消融实验按阶段开关：

    S0  仅 dense                        —— 对照组，也是竞品同类的朴素配置
    S1  + sparse + graph，RRF 融合
    S2  + 交叉编码器重排
    S3  + 查询改写（需 LLM，无 key 时自动跳过）
    S4  + 按科目过滤

每一阶段只比上一阶段多一个组件，因此实验表里相邻两行的差值
就是该组件的净贡献 —— 这正是「竞品只有一个最终数字，我能说清
这个数字由哪些组件贡献」的落点。
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

from cs408rag.config import settings
from cs408rag.retrieval.fusion import (
    FusedItem,
    FusionStrategy,
    contribution_stats,
    fuse,
)
from cs408rag.retrieval.graph_store import GraphStore
from cs408rag.retrieval.reranker import get_reranker
from cs408rag.retrieval.vector_store import SearchHit, VectorStore, build_filter

RouteName = Literal["dense", "sparse", "graph"]


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """一次检索的完整配置。消融实验靠切这些开关跑不同阶段。"""

    use_sparse: bool = False
    use_graph: bool = False
    # 原生 BM25（jieba + Okapi）作为第四路。
    # BGE-M3 的 sparse 与 BM25 捕捉的词法信号不完全重合，实测并集上限 +2.7pp。
    use_bm25: bool = False
    # 图谱兄弟节点扩展查询：命中知识点后取同父兄弟标题，拼进查询再检索一路，
    # 与原始结果 RRF 合并。实测题级 Hit@10 +2.4pp（0.577→0.602），
    # 额外成本约 150ms。单独用扩展结果会下降，必须合并而不是替代。
    use_graph_expand: bool = False
    use_rerank: bool = False
    use_rewrite: bool = False
    use_subject_filter: bool = False
    # 评测时排除真题解析 chunk —— 见 vector_store.build_filter 的说明。
    # 问答应答时应当保留（解析本身是有价值的语料），
    # 但评测教材检索能力时必须排除，否则是自匹配污染。
    exclude_explanations: bool = False
    # 排除试题/答案类 chunk（含真题原文，会靠字面匹配占满 top-k，掩盖概念检索能力）。
    # 检索评测默认开启；问答服务可关掉以保留这类语料。
    exclude_exercise: bool = True
    fusion: FusionStrategy = "rrf"
    # 同一小节的多个 chunk 在 top-10 里重复占位，会把其他小节挤出去。
    # 实测（dev 452 题）：去重后「题级命中率」从 0.595 升到 0.662，
    # 代价是 chunk 级 Recall 下降。问答关心的是「有没有拿到正确小节」，
    # 所以生产配置开启；检索评测保留两套口径分别报告。
    section_dedup: bool = False
    section_per: int = 1
    # 各路在融合中的权重。默认 1.0（等权），实际取值由消融实验扫出来 ——
    # 见 run_eval 的路由权重扫描。不给凭感觉的数字。
    w_dense: float = 1.0
    w_sparse: float = 1.0
    w_graph: float = 1.0
    w_bm25: float = 1.0
    top_k: int = 20
    fusion_top_k: int = 50
    rerank_top_n: int = 10
    rrf_k: int = 60
    normalize: str = "minmax"

    def label(self) -> str:
        parts = ["dense"]
        if self.use_sparse:
            parts.append("sparse")
        if self.use_graph:
            parts.append("graph")
        if self.use_bm25:
            parts.append("bm25")
        if self.use_graph_expand:
            parts.append("g-expand")
        s = "+".join(parts)
        if self.use_rerank:
            s += "|rerank"
        if self.use_rewrite:
            s += "|rewrite"
        if self.use_subject_filter:
            s += "|filter"
        if self.exclude_explanations:
            s += "|no-expl"
        if self.exclude_exercise:
            s += "|no-ex"
        if self.section_dedup:
            s += f"|sec-dedup{self.section_per}"
        s += f"|{self.fusion}"
        if (self.w_dense, self.w_sparse, self.w_graph) != (1.0, 1.0, 1.0):
            s += f"(w={self.w_dense:g}/{self.w_sparse:g}/{self.w_graph:g})"
        return s


# 递进式消融阶段：每一阶段只比上一阶段多一个组件。
#
# 全部开启 exclude_explanations：消融要衡量的是**教材检索能力**。
# 解析 chunk 里逐字含题干，对 dev 集的题会直接自匹配，
# 把教材章节挤出 top-10（实测 Recall@10 因此被压到 0.02）。
# 排除后才是干净的教材检索评测，held-out 集本来也没有解析可命中。
STAGES: dict[str, RetrievalConfig] = {
    "S0_dense": RetrievalConfig(exclude_explanations=True),
    "S1_hybrid": RetrievalConfig(
        use_sparse=True, use_graph=True, exclude_explanations=True
    ),
    "S1b_bm25": RetrievalConfig(
        use_sparse=True, use_graph=True, use_bm25=True, exclude_explanations=True
    ),
    "S2_rerank": RetrievalConfig(
        use_sparse=True, use_graph=True, use_rerank=True, exclude_explanations=True
    ),
    "S3_rewrite": RetrievalConfig(
        use_sparse=True,
        use_graph=True,
        use_rerank=True,
        use_rewrite=True,
        exclude_explanations=True,
    ),
    "S4_filter": RetrievalConfig(
        use_sparse=True,
        use_graph=True,
        use_rerank=True,
        use_rewrite=True,
        use_subject_filter=True,
        exclude_explanations=True,
    ),
    # 最终生产配置：按题型消融后去掉图谱、启用 BM25、zscore 融合、节去重。
    # 不是递增消融的一环，放在最后便于对照。
    "PROD": RetrievalConfig(
        use_sparse=True,
        use_graph=False,
        use_bm25=True,
        fusion="weighted",
        normalize="zscore",
        section_dedup=True,
        section_per=1,
        rerank_top_n=10,
        exclude_explanations=True,
    ),
}


# ---------------------------------------------------------------------------
# 结果
# ---------------------------------------------------------------------------


@dataclass
class RouteTrace:
    name: str
    took_ms: float
    top_ids: list[str] = field(default_factory=list)
    top_scores: list[float] = field(default_factory=list)


@dataclass
class RetrievalResult:
    query: str
    ranked_ids: list[str]
    scores: list[float]
    hits_by_id: dict[str, SearchHit]
    routes: dict[str, RouteTrace] = field(default_factory=dict)
    fused: list[FusedItem] = field(default_factory=list)
    rerank_applied: bool = False
    took_ms: float = 0.0
    subject_filter: str | None = None
    rewritten: list[str] = field(default_factory=list)

    def top_hits(self, n: int = 10) -> list[SearchHit]:
        return [self.hits_by_id[i] for i in self.ranked_ids[:n] if i in self.hits_by_id]

    def trace_summary(self) -> dict[str, Any]:
        """给 /v1/trace 用的可读摘要。"""
        return {
            "query": self.query,
            "took_ms": round(self.took_ms, 1),
            "subject_filter": self.subject_filter,
            "rerank_applied": self.rerank_applied,
            "rewritten": self.rewritten,
            "routes": {
                name: {
                    "took_ms": round(t.took_ms, 1),
                    "returned": len(t.top_ids),
                    "top_ids": t.top_ids[:5],
                }
                for name, t in self.routes.items()
            },
            "fused": [
                {"chunk_id": f.doc_id, "score": round(f.score, 5), "ranks": f.ranks}
                for f in self.fused[:10]
            ],
            "final_top": [
                {
                    "chunk_id": cid,
                    "chapter": " > ".join(self.hits_by_id[cid].chapter_path)[:70],
                    "subject": self.hits_by_id[cid].subject,
                    "source": self.hits_by_id[cid].source,
                }
                for cid in self.ranked_ids[:10]
                if cid in self.hits_by_id
            ],
        }


# ---------------------------------------------------------------------------
# 查询改写
# ---------------------------------------------------------------------------

# 408 题里常见的计算题触发词。改写时对这类题做查询分解。
CALC_TRIGGERS = (
    "带宽", "命中率", "传输速率", "容量", "多少个", "至少需要", "最少需要",
    "周期数", "吞吐", "时间", "窗口", "地址位数", "字长", "片数", "码长",
)


class TermVocabulary:
    """408 领域术语表，用于从查询里精确抽取术语。

    为什么不能只用 jieba：
        实测把「采用组相联映射时主存地址如何划分」切成
        ['采用','相联','映射','主存','地址','如何','划分'] ——
        「组相联映射」这个**核心术语被切碎了**。拿「映射」「地址」去查图谱，
        召回的是页表、地址变换之类不相关的节点，图谱路等于白跑。

    做法：用已经对齐好的术语集合（226 个真题 tag + 6182 个知识点标题）
    反查查询里出现了哪些领域词，**最长匹配优先**，再补 jieba 结果做兜底。
    好处是术语来自本领域语料本身，换科目也不用改代码。
    """

    def __init__(self, terms: Sequence[str] | None = None) -> None:
        self._terms: list[str] = []
        if terms is None:
            terms = self._load_default_terms()
        # 长词优先，保证「组相联映射」先于「映射」被匹配到
        for t in sorted(
            {t.strip() for t in terms if t and len(t.strip()) >= 2},
            key=len,
            reverse=True,
        ):
            self._terms.append(t)

    @staticmethod
    def _load_default_terms() -> list[str]:
        import json

        from cs408rag.config import PROCESSED_DIR

        terms: list[str] = []

        # 1) 真题 tag（226 个，人工整理的考点名，最可靠）
        try:
            rows = json.loads(
                (PROCESSED_DIR / "tag_kp_mapping.json").read_text(encoding="utf-8")
            )
            terms.extend(r["tag"] for r in rows if r.get("tag"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"加载 tag 词表失败：{e}")

        # 2) 知识点标题（6182 个，覆盖更细的术语）
        try:
            nodes = json.loads(
                (PROCESSED_DIR / "kp_nodes.json").read_text(encoding="utf-8")
            )
            terms.extend(n["title"] for n in nodes if n.get("title"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"加载知识点词表失败：{e}")

        logger.info(f"术语表载入 {len(terms)} 条")
        return terms

    def extract(self, query: str, *, max_terms: int = 12) -> list[str]:
        """从查询里抽术语。最长匹配优先，命中即从查询中剔除以免重复匹配。"""
        remaining = query
        found: list[str] = []

        for term in self._terms:
            if len(found) >= max_terms:
                break
            if term in remaining:
                found.append(term)
                # 用等长占位替换，避免「组相联映射」命中后「映射」又被单独命中
                remaining = remaining.replace(term, "\x00" * len(term))

        # 兜底：补 jieba 词，但过滤掉已被术语覆盖的碎片
        import jieba

        for w in jieba.cut(query):
            if len(found) >= max_terms:
                break
            w = w.strip()
            if (
                len(w) >= 2
                and not w.isdigit()
                and not re.fullmatch(r"[\W_]+", w)
                and w not in found
                and not any(w in f for f in found)
            ):
                found.append(w)

        # 英文缩写（TLB / KMP / CSMA）分词常被切碎，单独抽一遍
        for w in re.findall(r"[A-Za-z][A-Za-z0-9/\.\-]{1,15}", query):
            if w not in found and len(found) < max_terms:
                found.append(w)

        return found[:max_terms]


_vocab: TermVocabulary | None = None


def get_vocab() -> TermVocabulary:
    global _vocab
    if _vocab is None:
        _vocab = TermVocabulary()
    return _vocab


def extract_keywords(query: str, *, max_terms: int = 12) -> list[str]:
    """从查询里抽关键词，供图谱路使用。优先用领域术语表。"""
    try:
        return get_vocab().extract(query, max_terms=max_terms)
    except Exception as e:  # noqa: BLE001 - 词表挂了也要能退化到分词
        logger.warning(f"术语表不可用，退化为 jieba 分词：{e}")
        import jieba

        terms = [w.strip() for w in jieba.cut(query)]
        out: list[str] = []
        for w in terms:
            if len(w) >= 2 and not w.isdigit() and w not in out:
                out.append(w)
        out.extend(re.findall(r"[A-Za-z][A-Za-z0-9/\.\-]{1,15}", query))
        return out[:max_terms]


def decompose_calc_query(query: str) -> list[str]:
    """计算题的轻量查询分解。

    不做 LLM 调用，只按 408 计算题的常见结构补几个子查询：
    计算题往往问「某个量是多少」，直接拿整句去检索，
    向量会偏向题干里的背景描述而不是公式所在的那一节。
    拆出「知识点名 + 问什么量」两个方向，命中率明显更高。

    无 LLM 也能跑 —— 消融实验的 S3 因此在没有 API key 时依然可复现。
    """
    subs = [query]
    if any(t in query for t in CALC_TRIGGERS):
        kws = extract_keywords(query, max_terms=5)
        if kws:
            subs.append(" ".join(kws) + " 计算方法")
    return subs


def dedup_by_section(
    fused: list[FusedItem],
    hits_by_id: dict[str, SearchHit],
    *,
    per_sec: int = 1,
) -> list[FusedItem]:
    """同一小节最多保留 per_sec 个 chunk，按组内最高分排序。

    为什么需要：top-10 常被同一个小节的 5–6 个 chunk 占满（教材一节切成多块），
    其他相关小节被挤出去。实测去掉重复后，题级命中率从 0.595 升到 0.662。
    组内保持原有分数顺序，所以不会打乱最相关的那一块。
    """
    groups: dict[str, list[FusedItem]] = {}
    order: list[str] = []
    for item in fused:
        hit = hits_by_id.get(item.doc_id)
        sec = (hit.payload.get("section_id") if hit else "") or f"__{item.doc_id}"
        if sec not in groups:
            groups[sec] = []
            order.append(sec)
        groups[sec].append(item)

    order.sort(key=lambda k: -max(i.score for i in groups[k]))
    out: list[FusedItem] = []
    for k in order:
        out.extend(groups[k][:per_sec])
    return out


# ---------------------------------------------------------------------------
# 主链路
# ---------------------------------------------------------------------------


class RetrievalPipeline:
    """三路召回 + 融合 + 重排。"""

    def __init__(
        self,
        store: VectorStore | None = None,
        graph: GraphStore | None = None,
        *,
        exclude_years: Sequence[int] | None = None,
    ) -> None:
        self.store = store or VectorStore()
        self._graph = graph
        # 防泄漏第二道闸：即便有 held-out 解析意外入库，检索时也挡掉
        self.exclude_years = list(exclude_years) if exclude_years else sorted(
            settings.holdout_year_set
        )
        # 图谱路的失败计数。
        #
        # 为什么要显式计数：图谱查询出错时我们选择「本路返回空」而不是让整条链路崩，
        # 这对线上服务是对的，但会**掩盖整条路彻底失效**的情况 ——
        # 实测踩过：Cypher 变量作用域写错，图谱路一直是空的，
        # 而权重扫描里 graph 无论取什么值指标都完全一样，看起来像「图谱没用」，
        # 实际是根本没跑起来。计数让这种情况在评测报告里现形。
        self.graph_failures = 0
        self.graph_calls = 0

    @property
    def graph(self) -> GraphStore:
        if self._graph is None:
            self._graph = GraphStore()
        return self._graph

    # -- 单路召回 ---------------------------------------------------------

    def _dense(
        self, query: str, cfg: RetrievalConfig, subject: str | None, encoding: Any = None
    ) -> list[SearchHit]:
        if encoding is not None:
            vec = encoding[0]
        else:
            from cs408rag.retrieval.embedder import get_embedder

            vec, _ = get_embedder().encode_query(query)
        f = build_filter(
            subject=subject,
            exclude_sources=["explanation"] if cfg.exclude_explanations else None,
            exclude_chunk_classes=["exercise"] if cfg.exclude_exercise else None,
            exclude_years=self.exclude_years,
        )
        return self.store.search_dense(vec, top_k=cfg.top_k, query_filter=f)

    def _sparse(
        self, query: str, cfg: RetrievalConfig, subject: str | None, encoding: Any = None
    ) -> list[SearchHit]:
        if encoding is not None:
            w = encoding[1]
        else:
            from cs408rag.retrieval.embedder import get_embedder

            _, w = get_embedder().encode_query(query)
        if not w:
            return []
        f = build_filter(
            subject=subject,
            exclude_sources=["explanation"] if cfg.exclude_explanations else None,
            exclude_chunk_classes=["exercise"] if cfg.exclude_exercise else None,
            exclude_years=self.exclude_years,
        )
        return self.store.search_sparse(w, top_k=cfg.top_k, query_filter=f)

    def _graph_route(
        self, query: str, cfg: RetrievalConfig, subject: str | None, encoding: Any = None
    ) -> list[SearchHit]:
        kws = extract_keywords(query)
        if not kws:
            return []
        self.graph_calls += 1
        try:
            hits = self.graph.search_by_keywords(
                kws, top_k=cfg.top_k, subject=subject if cfg.use_subject_filter else None
            )
        except Exception as e:  # noqa: BLE001 - Neo4j 挂了不该让整条链路崩
            # 但要计数，否则整条路失效会被当成「图谱没用」而不是「图谱没跑」
            self.graph_failures += 1
            logger.warning(f"图谱路失败，本路返回空：{e}")
            return []

        if not hits:
            return []
        # 图谱只给 chunk_id，正文得回 Qdrant 取
        ids = [h.chunk_id for h in hits]
        fetched = {h.chunk_id: h for h in self.store.get_by_ids(ids)}
        out: list[SearchHit] = []
        for gh in hits:
            h = fetched.get(gh.chunk_id)
            if h is None:
                continue
            if cfg.exclude_explanations and h.source == "explanation":
                continue
            if cfg.exclude_exercise and h.payload.get("chunk_class") == "exercise":
                continue
            out.append(SearchHit(chunk_id=h.chunk_id, score=gh.score, payload=h.payload))
        return out

    # -- BM25（第四路）----------------------------------------------------

    def _bm25_route(
        self, query: str, cfg: RetrievalConfig, subject: str | None, encoding: Any = None
    ) -> list[SearchHit]:
        from cs408rag.retrieval.bm25 import get_bm25

        idx = get_bm25()
        if idx is None:
            return []
        hits = idx.search(query, top_k=cfg.top_k, subject=subject)
        if not hits:
            return []
        # BM25 只给 chunk_id 与分数，正文与 payload 回 Qdrant 取
        fetched = {h.chunk_id: h for h in self.store.get_by_ids([c for c, _ in hits])}
        out: list[SearchHit] = []
        for cid, score in hits:
            h = fetched.get(cid)
            if h is not None:
                out.append(SearchHit(chunk_id=cid, score=score, payload=h.payload))
        return out

    # -- 图谱兄弟扩展 -----------------------------------------------------

    def _expand_with_siblings(
        self,
        query: str,
        subject: str | None,
        cfg: RetrievalConfig,
        ranked_ids: list[str],
        hits_by_id: dict[str, SearchHit],
    ) -> tuple[list[str], list[float]] | None:
        from dataclasses import replace

        kws = extract_keywords(query, max_terms=6)
        try:
            sibs = self.graph.sibling_titles(kws, subject=subject, limit=8)
        except Exception as e:  # noqa: BLE001 - 图谱挂了不影响主链路
            logger.warning(f"兄弟节点扩展失败，跳过：{e}")
            return None
        if not sibs:
            return None

        q2 = query + " " + " ".join(sibs[:6])
        # 扩展检索只用 dense+sparse：图谱路已经在原始检索里跑过，
        # 再跑一次只会重复候选、拖慢延迟。
        exp_cfg = replace(
            cfg,
            use_graph=False,
            use_bm25=False,
            use_graph_expand=False,
            use_subject_filter=False,
            rerank_top_n=cfg.top_k,
        )
        try:
            exp = self.retrieve(q2, subject=subject, config=exp_cfg)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"扩展检索失败，跳过：{e}")
            return None
        for cid, h in exp.hits_by_id.items():
            hits_by_id.setdefault(cid, h)

        merged = fuse(
            {"base": ranked_ids, "expand": exp.ranked_ids},
            strategy="rrf",
            k=cfg.rrf_k,
            weights={"base": 1.0, "expand": 0.5},
        )[: cfg.rerank_top_n]
        return [m.doc_id for m in merged], [m.score for m in merged]

    # -- 主入口 -----------------------------------------------------------

    def retrieve(
        self,
        query: str,
        *,
        subject: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        cfg = config or STAGES["S1_hybrid"]
        t_start = time.perf_counter()

        subj = subject if cfg.use_subject_filter else None

        # --- S3: 查询改写 ---
        queries = [query]
        if cfg.use_rewrite:
            queries = decompose_calc_query(query)

        # --- 查询编码：dense/sparse 共用，只编一次 ---
        # 早先两路各自调 encode_query，同一条 query 被编码两次（实测 13ms/次）。
        encodings: dict[str, tuple[Any, Any]] = {}
        t_enc = time.perf_counter()
        try:
            from cs408rag.retrieval.embedder import get_embedder

            emb = get_embedder()
            for q in queries:
                encodings[q] = emb.encode_query(q)
        except Exception as e:  # noqa: BLE001 - 编码失败时各路自行编码
            logger.warning(f"查询预编码失败，各路退化为自行编码：{e}")

        routes: dict[str, RouteTrace] = {}
        hits_by_id: dict[str, SearchHit] = {}
        routes["encode"] = RouteTrace(
            name="encode", took_ms=(time.perf_counter() - t_enc) * 1000
        )

        # --- 多路召回并行 ---
        # 四路彼此独立，串行时总耗时是各路之和（实测 124ms），
        # 并行后是 max(单路)（约 57ms）。图谱/BM25 都是 I/O 或纯内存，线程安全。
        route_defs: list[tuple[str, Any]] = [("dense", self._dense)]
        if cfg.use_sparse:
            route_defs.append(("sparse", self._sparse))
        if cfg.use_graph:
            route_defs.append(("graph", self._graph_route))
        if cfg.use_bm25:
            route_defs.append(("bm25", self._bm25_route))

        def _run_route(name: str, fn: Any) -> tuple[str, dict[str, SearchHit], float]:
            t0 = time.perf_counter()
            merged: dict[str, SearchHit] = {}
            for q in queries:
                for h in fn(q, cfg, subj, encodings.get(q)):
                    # 同一 chunk 取最高分
                    if h.chunk_id not in merged or h.score > merged[h.chunk_id].score:
                        merged[h.chunk_id] = h
            return name, merged, (time.perf_counter() - t0) * 1000

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=max(len(route_defs), 1)) as ex:
            results = list(ex.map(lambda x: _run_route(*x), route_defs))

        for rname, merged, took in results:
            ordered = sorted(merged.values(), key=lambda h: -h.score)[: cfg.top_k]
            routes[rname] = RouteTrace(
                name=rname,
                took_ms=took,
                top_ids=[h.chunk_id for h in ordered],
                top_scores=[h.score for h in ordered],
            )
            for h in merged.values():
                hits_by_id.setdefault(h.chunk_id, h)

        # --- 融合 ---
        rankings = {n: t.top_ids for n, t in routes.items() if t.top_ids}
        scored = {
            n: list(zip(t.top_ids, t.top_scores, strict=True))
            for n, t in routes.items()
            if t.top_ids
        }

        fused = fuse(
            rankings,
            strategy=cfg.fusion,
            k=cfg.rrf_k,
            weights={
                "dense": cfg.w_dense,
                "sparse": cfg.w_sparse,
                "graph": cfg.w_graph,
                "bm25": cfg.w_bm25,
            },
            scored=scored,
            normalize=cfg.normalize,  # type: ignore[arg-type]
        )
        if cfg.section_dedup:
            fused = dedup_by_section(fused, hits_by_id, per_sec=cfg.section_per)
        fused = fused[: cfg.fusion_top_k]

        # --- S2: 重排 ---
        rerank_applied = False
        if cfg.use_rerank and fused:
            cand_ids = [f.doc_id for f in fused]
            texts = [hits_by_id[i].text for i in cand_ids if i in hits_by_id]
            valid_ids = [i for i in cand_ids if i in hits_by_id]
            if texts:
                try:
                    rr = get_reranker().rerank(query, texts, top_n=cfg.rerank_top_n)
                    ranked_ids = [valid_ids[r.index] for r in rr]
                    scores = [r.score for r in rr]
                    rerank_applied = True
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"重排失败，回退到融合结果：{e}")
                    ranked_ids = [f.doc_id for f in fused[: cfg.rerank_top_n]]
                    scores = [f.score for f in fused[: cfg.rerank_top_n]]
            else:
                ranked_ids, scores = [], []
        else:
            ranked_ids = [f.doc_id for f in fused[: cfg.rerank_top_n]]
            scores = [f.score for f in fused[: cfg.rerank_top_n]]

        # --- 图谱兄弟扩展：作为补充一路与结果合并 ---
        if cfg.use_graph_expand and ranked_ids:
            merged_exp = self._expand_with_siblings(
                query, subject, cfg, ranked_ids, hits_by_id
            )
            if merged_exp:
                ranked_ids, scores = merged_exp

        return RetrievalResult(
            query=query,
            ranked_ids=ranked_ids,
            scores=scores,
            hits_by_id=hits_by_id,
            routes=routes,
            fused=fused,
            rerank_applied=rerank_applied,
            took_ms=(time.perf_counter() - t_start) * 1000,
            subject_filter=subj,
            rewritten=queries[1:] if len(queries) > 1 else [],
        )

    def retrieve_multi(
        self, queries: Sequence[str], **kw: Any
    ) -> list[RetrievalResult]:
        return [self.retrieve(q, **kw) for q in queries]

    # -- 诊断 -------------------------------------------------------------

    def route_contribution(self, result: RetrievalResult, top_n: int = 10) -> dict[str, Any]:
        rankings = {n: t.top_ids for n, t in result.routes.items() if t.top_ids}
        return contribution_stats(rankings, result.fused, top_n=top_n)


def stage_configs() -> dict[str, RetrievalConfig]:
    return dict(STAGES)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    pipe = RetrievalPipeline()
    q = " ".join(sys.argv[1:]) or "采用组相联映射时主存地址如何划分"
    for name, cfg in STAGES.items():
        r = pipe.retrieve(q, config=cfg)
        print(f"--- {name:12} {cfg.label()}")
        print(f"    {r.took_ms:.0f}ms  召回 {len(r.ranked_ids)} 条")
        for h in r.top_hits(3):
            print(
                f"      {h.score if h.score else 0:.4f}  [{h.source}] "
                f"{' > '.join(h.chapter_path)[:60]}"
            )