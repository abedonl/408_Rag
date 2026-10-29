"""索引构建编排。

把三份数据源汇入 Qdrant：
    教材分块      按章节切分，带 chapter_path
    真题解析分块   ★ held-out 年份的解析被挡住，不入库
    知识点关联    chunk -> kp_ids，来自 tag 映射与标题匹配

两条必须遵守的纪律：
    1. 入库前调用 chunker.assert_no_holdout_leak —— 让泄漏炸出来，而不是悄悄虚高
    2. 写出构建报告（含防泄漏拦截数、失败页数），供 README 引用
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from loguru import logger

from cs408rag.config import REPORTS_DIR, ensure_dirs, settings
from cs408rag.ingestion.chunker import (
    Chunk,
    assert_no_holdout_leak,
    assert_unique_ids,
    chunk_explanations,
    chunk_md_sections,
    chunk_stats,
    export_chunks,
)
from cs408rag.ingestion.md_loader import load_wangdao
from cs408rag.ingestion.quiz_loader import Question, load_questions
from cs408rag.ingestion.tag_mapper import load_mapping
from cs408rag.ingestion.xmind_loader import KnowledgePoint, load_all_mindmaps
from cs408rag.retrieval.embedder import get_embedder
from cs408rag.retrieval.vector_store import VectorStore


@dataclass
class BuildReport:
    questions_total: int = 0
    dev_questions: int = 0
    holdout_questions: int = 0
    holdout_blocked_explanations: int = 0
    kp_nodes: int = 0
    tag_mapping_coverage: float = 0.0
    textbook_sections: int = 0
    textbook_chars: int = 0
    chunks_total: int = 0
    chunks_by_source: dict[str, int] = field(default_factory=dict)
    chunks_by_subject: dict[str, int] = field(default_factory=dict)
    chunks_linked_to_kp: int = 0
    kp_nodes_covered: int = 0
    indexed_points: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 知识点关联
# ---------------------------------------------------------------------------


def link_chunks_to_kps(
    chunks: list[Chunk],
    kps: list[KnowledgePoint],
    tag_map: dict[str, list[str]],
    questions_by_qid: dict[str, Question],
    *,
    max_kp_per_chunk: int = 24,
) -> int:
    """给 chunk 填 kp_ids。

    三条来源各走各的路：
        explanation  用题目的 tags 经映射表拿 kp —— 精确
        mindmap      chunk_mindmap 按 DFS 分区时已给好 kp_ids —— 最精确
        wangdao      用正文子串匹配 kp 标题 —— 近似，必须限流

    王道 chunk 的匹配按「标题越长越具体」优先并截断：
    一个 500 字的教材小节可能提到几十个概念，全挂上去会让
    「题目 -> 知识点 -> chunk」的 ground truth 又大又糊，指标失去区分度。
    """
    from cs408rag.ingestion.tag_mapper import normalize

    title_idx: dict[str, list[str]] = {}
    for kp in kps:
        k = normalize(kp.title)
        if k:
            title_idx.setdefault(k, []).append(kp.node_id)

    # 只对长度 >= 2 的标题做匹配，长的排前面（更具体，优先保留）
    matchable = sorted(
        ((k, ids) for k, ids in title_idx.items() if len(k) >= 2),
        key=lambda kv: -len(kv[0]),
    )

    linked = 0
    for ch in chunks:
        # mindmap chunk 在分区时已带 kp_ids
        if ch.kp_ids:
            linked += 1
            continue

        ids: list[str] = []

        if ch.source == "explanation":
            q = questions_by_qid.get(ch.meta.get("qid", ""))
            if q:
                for tag in q.tags:
                    ids.extend(tag_map.get(tag, []))

        elif ch.source == "wangdao":
            norm_text = normalize(ch.text[:6000])
            for key, node_ids in matchable:
                if key in norm_text:
                    ids.extend(node_ids)
                    if len(ids) >= max_kp_per_chunk * 4:
                        break

        ch.kp_ids = sorted(set(ids))[:max_kp_per_chunk]
        if ch.kp_ids:
            linked += 1

    by_source: dict[str, list[int]] = {}
    for ch in chunks:
        s = by_source.setdefault(ch.source, [0, 0])
        s[0] += 1
        if ch.kp_ids:
            s[1] += 1
    for src, (total, hit) in sorted(by_source.items()):
        logger.info(f"  知识点关联 [{src}]：{hit}/{total} = {hit / max(total, 1):.1%}")

    logger.info(f"知识点关联合计：{linked}/{len(chunks)} 个 chunk 关联到知识点")
    return linked


# ---------------------------------------------------------------------------
# 向量化与入库
# ---------------------------------------------------------------------------


def embed_and_upsert(
    chunks: list[Chunk],
    vs: VectorStore,
    *,
    batch_size: int = 32,
    recreate: bool = True,
) -> int:
    if recreate:
        vs.recreate()
    else:
        vs.ensure()

    embedder = get_embedder()
    texts = [c.text for c in chunks]

    total = 0
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        batch = texts[start:end]
        r = embedder.encode(batch, show_progress=False)
        vs.upsert_chunks(chunks[start:end], r.dense, r.sparse)
        total += len(batch)
        logger.info(f"向量化进度 {total}/{len(texts)}")

    return total


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build(
    *,
    recreate: bool = True,
    include_explanations: bool = True,
    include_textbook: bool = True,
    embed_batch_size: int = 32,
) -> BuildReport:
    ensure_dirs()
    rep = BuildReport()

    # --- 真题 ---
    questions = load_questions()
    rep.questions_total = len(questions)
    holdout = settings.holdout_year_set
    rep.dev_questions = sum(1 for q in questions if q.year not in holdout)
    rep.holdout_questions = sum(1 for q in questions if q.year in holdout)
    questions_by_qid = {q.qid: q for q in questions}

    # --- 知识点图谱 ---
    mindmaps = load_all_mindmaps()
    kps = [n for mm in mindmaps for n in mm.nodes]
    rep.kp_nodes = len(kps)

    # --- tag 映射 ---
    try:
        tag_map = load_mapping()
        rep.tag_mapping_coverage = len(tag_map) / max(len({t for q in questions for t in q.tags}), 1)
    except FileNotFoundError:
        logger.warning("未找到 tag 映射，跳过知识点关联。先跑 tag_mapper。")
        tag_map = {}
        rep.notes.append("tag 映射缺失，chunk 未关联知识点")

    chunks: list[Chunk] = []

    # --- 真题解析分块（含防泄漏）---
    if include_explanations:
        # 解析里的内联 SVG（47 题）落盘并挂到 chunk 上。
        # held-out 年份的图不生成，与解析正文的隔离口径一致。
        from cs408rag.ingestion.exam_figures import extract_exam_figures

        exam_figures = extract_exam_figures(questions, exclude_years=holdout)
        ex_chunks, blocked = chunk_explanations(
            questions, holdout, figures_by_qid=exam_figures
        )
        rep.holdout_blocked_explanations = len(blocked)
        rep.notes.append(
            f"解析图：{len(exam_figures)} 题 / "
            f"{sum(len(v) for v in exam_figures.values())} 张 SVG"
        )
        chunks.extend(ex_chunks)

    # --- ★ 主语料：王道复习指导书正文 ---
    #
    # 为什么不用 naplatte 那批「教材 PDF」：实测它们是 XMind 思维导图的扁平导出
    # （PDF 文本行 84.8% 逐字符命中 XMind 节点标题），没有成段讲解文字，
    # 按页切只能得到 4 个 chunk（每科一个），完全不可用。
    #
    # 思维导图不作为 chunk 语料，只用于建 Neo4j 知识点图谱（节点 + 层级边）。
    # 它的价值在结构与层级关系，不在文本本身。
    if include_textbook:
        md_sections = load_wangdao()
        rep.textbook_sections = len(md_sections)
        rep.textbook_chars = sum(len(s.text) for s in md_sections)
        chunks.extend(chunk_md_sections(md_sections))

    # --- ★ 防泄漏最后一道闸 ---
    assert_no_holdout_leak(chunks, holdout)

    # --- ★ chunk_id 唯一性（撞了会在 Qdrant 静默丢数据）---
    assert_unique_ids(chunks)

    # --- 关联知识点 ---
    rep.chunks_linked_to_kp = link_chunks_to_kps(chunks, kps, tag_map, questions_by_qid)

    # --- 统计 ---
    st = chunk_stats(chunks)
    rep.chunks_total = st["count"]
    rep.chunks_by_source = st["by_source"]
    rep.chunks_by_subject = st["by_subject"]

    export_chunks(chunks, "chunks_all")

    # 图片索引：教材插图 + 真题解析图，合并写一份
    from cs408rag.ingestion.image_index import save_image_index

    save_image_index()

    # --- 向量化入库 ---
    vs = VectorStore()
    if not vs.ping():
        raise RuntimeError(f"Qdrant 连不上（{vs.url}）。先执行：docker compose up -d")
    rep.indexed_points = embed_and_upsert(
        chunks, vs, batch_size=embed_batch_size, recreate=recreate
    )

    # --- 报告 ---
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    p = REPORTS_DIR / "build_report.json"
    p.write_text(json.dumps(rep.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"构建报告 -> {p}")

    return rep


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建 Qdrant 索引")
    ap.add_argument("--no-recreate", action="store_true", help="不重建 collection，增量追加")
    ap.add_argument("--no-explanations", action="store_true", help="不含真题解析（消融用）")
    ap.add_argument("--no-textbook", action="store_true", help="不含王道教材（消融用）")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    report = build(
        recreate=not args.no_recreate,
        include_explanations=not args.no_explanations,
        include_textbook=not args.no_textbook,
        embed_batch_size=args.batch_size,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))