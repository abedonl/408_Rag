"""检索评测集构造。

★ 为什么不是「chunk 级」ground truth

    最初的实现是：题目 tag -> 知识点 -> 所有含该知识点的 chunk。
    实测结果是灾难 —— 每题 GT chunk 数的中位数是 **120**（最大 2175），
    而全集才 5296 个 chunk。这等于说「五分之一的内容都算相关」，
    Recall@10 的理论上限只有 8.3%，指标彻底失去区分度。

    根因：子串匹配会把泛化标题（「树」「进程」「查找」）命中的
    所有章节都算作相关，而教材里这些词遍地都是。

★ 改成「章节级」

    对每道题，找**最覆盖它考点的教材小节**，ground truth 取这一节的全部 chunk。
    一节通常切出 1-5 个 chunk，GT 规模回到可信区间，Recall@k / nDCG@k 才有意义。

    打分用「命中的不同知识点标题数」（按标题长度加权），不是出现次数 ——
    否则一篇反复提到「树」的小节会压过真正讲「树与二叉树遍历」的那一节。

★ 这一步的产物需要人工抽检

    它是整条评测链的上游，映射错了后面所有数字都没意义。
    脚本会输出 samples_for_review.json，供人工核对。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cs408rag.config import EVAL_DIR, ensure_dirs, settings
from cs408rag.ingestion.chunker import Chunk
from cs408rag.ingestion.md_loader import MdSection
from cs408rag.ingestion.quiz_loader import Question
from cs408rag.ingestion.tag_mapper import GENERIC_NODE_TITLES, normalize
from cs408rag.ingestion.xmind_loader import KnowledgePoint

# 至少命中多少个不同知识点标题才认为该小节与题目相关
MIN_MATCH_SCORE = 1
# 每题最多取几个小节作为 ground truth。
#
# 实测取 2 时会引入噪声：一道「最小生成树」的题，第二个命中章节被定位到
# 「5.5 树与二叉树的应用」，与考点无关，纯粹因为共享了某个泛化知识点标题。
# 噪声进 GT 会同时压低各项指标并掩盖真实差异，所以只取分数最高的那一个。
MAX_GT_SECTIONS = 1
# 参与匹配的最短标题长度。1 字标题（如「树」「图」「串」）太泛，会污染打分
MIN_TITLE_LEN = 2
# ★ 文档频率上限：某标题出现在超过这个比例的小节里，就认为它没有区分度
#
# 思维导图里有大量泛化父节点（「概念」「管理」「方法」「算法」「特点」「分类」），
# 它们在教材里遍地都是。实测这些词会把「磁盘调度算法」这类题
# 匹配到几十上百个不相关小节，GT 中位数一度被推到 120。
#
# 用 IDF 思路数据驱动地剔除，比手工维护停用词表更稳：
# 判断标准来自语料本身，换一份教材也自动适配。
#
# 阈值是扫出来的，实测（2137 个小节）：
#   0.02 -> 覆盖 50.6%   0.10 -> 70.3%   0.20 -> 76.1%   关闭 -> 79.2%
# GT 中位数在各档都是 6-7，说明 GT 规模主要由小节切出的 chunk 数决定，
# 不随阈值漂移。
#
# 最终取 0.05（约 107 节，即 df<=107 的术语才算有区分度）：
#   保留 「二叉树遍历」(df=8)、「最小生成树」(13)、「平衡二叉树」(16)、「缓冲区」(76)
#   剔除 「遍历」(120)、「排序」(121)、「队列」(130)、「树」(201)、「查找」(204)
#        「进程」(328)、「地址」(524)、「图」(640)
# 0.10 时「遍历」会被保留，实测导致「二叉树遍历」的题被错定位到
# 「队列在层次遍历中的应用」一节 —— 这正是选 0.05 的原因。
MAX_DOC_FREQ_RATIO = 0.05
# 兜底停用词（泛化到即使 df 未超限也不该参与打分）
GENERIC_TITLES = {
    "概念", "特点", "分类", "定义", "作用", "方法", "管理", "算法", "结构",
    "原理", "组成", "功能", "性质", "条件", "方式", "过程", "情况", "内容",
    "基本概念", "相关概念", "概述", "总结", "应用", "实现", "机制", "技术",
}

# 不作为 ground truth 的小节。
#
# 王道的每章末尾有「本节试题精选」与「答案与解析」两节，里面**直接收录了历年真题**
# 及其解答。若允许它们当 GT，会出现两种情况，都让评测失去意义：
#   1. 检索器靠题面文字精确匹配就命中，测的不是概念理解
#   2. 这类小节包含大量无关题目的答案，作为 GT 噪声很大
# 评测要回答的是「能不能找到这个考点的概念讲解」，所以排除掉。
#
# 注意：这些 chunk 仍然**留在索引里**（它们对标答类问答有用），
# 只是不作为检索评测的正确答案。
#
# ★ 另外排除章节导语类小节（复习提示/小结/导读/前言）。
# 实测：「完全二叉树」这个 tag 的词只出现在第 5 章的「【复习提示】」里
# （导语把整章术语列了一遍），而正文小节标题叫「几种特殊的二叉树」「二叉树的性质」，
# 不含「完全二叉树」这个完整串。于是 GT 被定位到复习提示，检索返回正确小节反而判错。
# 365 题里有 7 道完全指到复习提示，属于纯粹的上游污染。
GT_EXCLUDED_SECTION_PATTERNS = (
    "答案与解析",
    "试题精选",
    "习题",
    "真题",
    "复习提示",
    "考纲内容",
    "本章导学",
    "学习目标",
    "本章重点",
    "知识框架",
    "本章小结",
    "本节小结",
    "知识点总结",
    "导读",
    "前言",
)


@dataclass
class EvalItem:
    qid: str
    year: int
    subject: str
    question: str
    options: dict[str, str]
    answer: str
    tags: list[str]
    # ground truth
    gt_section_ids: list[str] = field(default_factory=list)
    gt_chunks: list[str] = field(default_factory=list)
    match_score: int = 0
    matched_titles: list[str] = field(default_factory=list)
    is_holdout: bool = False

    @property
    def gt_size(self) -> int:
        return len(self.gt_chunks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "year": self.year,
            "subject": self.subject,
            "question": self.question,
            "options": self.options,
            "answer": self.answer,
            "tags": self.tags,
            "gt_section_ids": self.gt_section_ids,
            "gt_chunks": self.gt_chunks,
            "gt_size": self.gt_size,
            "match_score": self.match_score,
            "matched_titles": self.matched_titles,
            "is_holdout": self.is_holdout,
        }


# ---------------------------------------------------------------------------
# 构造
# ---------------------------------------------------------------------------


def _kp_titles_by_tag(
    tag_map: dict[str, list[str]],
    kp_title_by_id: dict[str, str],
) -> dict[str, list[str]]:
    """tag -> 可用于匹配教材小节的术语列表。

    ★ 把 **tag 名本身**放在最前面。

    实测踩过的坑：tag 是人工整理的考点名，本身相当精确
    （如「二叉树遍历」df=8，「最小生成树」df=13），
    但经 tag→知识点 映射展开后变成 ['二叉树','遍历'] 这种泛化组合
    （「遍历」df=120，遍布各章）。精确信号在映射这一步被稀释，
    导致「二叉树遍历」的题被定位到「队列在层次遍历中的应用」一节。

    所以术语池 = tag 原名（高精度） + 映射到的知识点标题（补充召回）。
    """
    out: dict[str, list[str]] = {}
    for tag, ids in tag_map.items():
        titles = [tag]  # ★ 原名优先
        for i in ids:
            t = kp_title_by_id.get(i)
            if not t:
                continue
            nt = normalize(t)
            # 过滤泛化知识点标题：它们作为 GT 术语会把题目定位到
            # 「性能指标」「方式」这类遍地都是的小节。
            # tag 原名不过滤 —— 那是最精确的信号。
            if len(nt) < 2 or nt in GENERIC_TITLES or nt in GENERIC_NODE_TITLES:
                continue
            titles.append(t)
        out[tag] = titles
    return out


# 术语命中**小节标题/路径**时的权重倍数。
#
# 这是 GT 质量的关键修正。最初只用正文子串匹配，噪声极大：
# 「平衡二叉树」的题被定位到「5.2.1 几种特殊的二叉树」（讲完全二叉树的），
# 而真正的「7.3.2 平衡二叉树」反而没被选上 —— 因为术语在正文里
# 到处出现，正文命中几乎不携带定位信息。
#
# 标题命中才是强信号：小节就叫「平衡二叉树」，那就是讲这个考点的。
TITLE_MATCH_BOOST = 4.0

# --- 交叉验证 ---
# 实测对比：加标题加权前，200 题的 Recall@10 只有 0.14，
# 且抽检发现多数「未命中」其实是**检索对了、GT 错了**。
# 这类问题的危险在于：它看起来像检索效果差，会误导整个优化方向。


def _section_label(s: Any) -> str:
    """小节的标题+路径，用于标题级匹配。"""
    return normalize(" ".join([*getattr(s, "path", []), getattr(s, "title", "")]))


def build_eval_set(
    questions: Sequence[Question],
    sections: Sequence[MdSection],
    chunks: Sequence[Chunk],
    kps: Sequence[KnowledgePoint],
    tag_map: dict[str, list[str]],
    *,
    max_gt_sections: int = MAX_GT_SECTIONS,
) -> list[EvalItem]:
    """为每道题定位最相关的教材小节。"""
    kp_title_by_id = {k.node_id: k.title for k in kps}
    tag_titles = _kp_titles_by_tag(tag_map, kp_title_by_id)

    # 小节正文归一化索引。
    #
    # 排除「答案与解析」「试题精选」等小节 —— 它们直接收录历年真题原文，
    # 让检索器靠题面精确匹配就能命中，测不出概念理解能力（见上方常量说明）。
    excluded_sids: set[str] = set()
    sec_norm: dict[str, str] = {}
    sec_label: dict[str, str] = {}
    for s in sections:
        label = " ".join([*getattr(s, "path", []), getattr(s, "title", "")])
        if any(p in label for p in GT_EXCLUDED_SECTION_PATTERNS):
            excluded_sids.add(s.section_id)
            continue
        sec_norm[s.section_id] = normalize(s.text + " " + s.title)
        sec_label[s.section_id] = _section_label(s)

    logger.info(
        f"GT 候选小节：{len(sec_norm)} 个（排除了 {len(excluded_sids)} 个"
        f"「答案与解析/试题精选/习题」类小节，它们含真题原文，不适合作 GT）"
    )

    # 小节 -> 其 chunk（先建，用于后面筛掉没有正文的小节）
    sec2chunks: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        if c.section_id and c.section_id not in excluded_sids:
            sec2chunks[c.section_id].append(c.chunk_id)

    # ★ 只把**确实产出过 chunk** 的小节作为 GT 候选。
    #
    # 踩过的坑：正文为空、只剩标题的小节也会进 sec_norm（因为拼了标题），
    # 于是能被术语匹配上并成为 top-1 小节，但它的 chunk 列表是空的 ——
    # gt_chunks 成了空集，这道题就被算作「无 ground truth」。
    # 实测这会让覆盖率无端少掉十几个百分点，而且是随打分规则变化而漂移的隐形损失。
    no_body_sids = {
        s.section_id
        for s in sections
        if s.section_id and s.section_id not in sec2chunks
    }
    for sid in no_body_sids:
        sec_norm.pop(sid, None)
        sec_label.pop(sid, None)
    if no_body_sids:
        logger.info(f"剔除 {len(no_body_sids)} 个无正文小节（只有标题，无法作为 GT）")

    # --- 候选标题 ---
    all_titles: set[str] = set()
    for titles in tag_titles.values():
        all_titles.update(titles)
    norm_pairs = [(normalize(t), t) for t in all_titles]
    norm_pairs = [(k, t) for k, t in norm_pairs if len(k) >= MIN_TITLE_LEN]

    n_sections = max(len(sec_norm), 1)

    # --- 一次扫描建倒排索引：标题 -> 含有它的小节 ---
    #
    # 别写成「每题 × 每节 × 每标题」的三重循环 —— 那是几十亿次子串比较。
    # 倒排之后，每题只需对它的少量标题取 postings 并累加权重。
    #
    # ★ 只扫**同科目**的小节。实测不加这个约束会出大错：
    #   2009-01 是数据结构题（打印机缓冲区，考队列），
    #   却因为 tag「缓冲区」在操作系统的「系统资源管理功能」一节里
    #   被顺带列举，GT 就被定位到了操作系统章节 —— 与考点毫无关系。
    #   408 四科的术语高度重叠（缓冲区/地址/调度/缓存…），
    #   不按科目隔离，跨科误匹配会大面积污染 GT。
    sections_by_subject: dict[str, list[str]] = defaultdict(list)
    for sec in sections:
        # 必须用 sec_norm 判断：它已经剔除了「答案与解析」类小节和无正文小节。
        # 早先只检查 excluded_sids，漏掉 no_body_sids，后面 sec_norm[sid] 会 KeyError。
        if sec.section_id in sec_norm:
            sections_by_subject[sec.subject].append(sec.section_id)

    postings: dict[str, list[str]] = defaultdict(list)
    title_hits: dict[str, set[str]] = defaultdict(set)
    for key, _raw in norm_pairs:
        for sids in sections_by_subject.values():
            for sid in sids:
                if key in sec_norm[sid]:
                    postings[key].append(sid)
                    if key in sec_label.get(sid, ""):
                        title_hits[key].add(sid)

    max_df = n_sections * MAX_DOC_FREQ_RATIO
    title_info: dict[str, tuple[str, float]] = {}  # norm_key -> (raw, idf)
    dropped: list[tuple[str, int]] = []
    for key, raw in norm_pairs:
        df = len(postings.get(key, ()))
        if raw in GENERIC_TITLES or df > max_df or df == 0:
            dropped.append((raw, df))
            postings.pop(key, None)
            continue
        title_info[key] = (raw, math.log(n_sections / (1 + df)))

    logger.info(
        f"知识点标题：候选 {len(norm_pairs)} 个 → 保留 {len(title_info)} 个，"
        f"剔除泛化/无区分度标题 {len(dropped)} 个（df 上限 {max_df:.0f}/{n_sections} 节）"
    )
    if dropped:
        top = sorted(dropped, key=lambda kv: -kv[1])[:10]
        logger.info("  剔除样例：" + ", ".join(f"{t}(df={d})" for t, d in top))

    holdout = settings.holdout_year_set
    items: list[EvalItem] = []

    for q in questions:
        q_titles: set[str] = set()
        for tag in q.tags:
            for t in tag_titles.get(tag, []):
                k = normalize(t)
                if k in title_info:
                    q_titles.add(k)

        # ★ 只在该题的科目内打分。跨科匹配是主要的 GT 污染源。
        allowed = set(sections_by_subject.get(q.subject, []))
        if not allowed:
            # 科目名对不上（如「组成原理」vs「计算机组成原理」）时退化为全科，
            # 但记下来便于发现映射问题
            allowed = set(sec_norm)

        sec_score: dict[str, float] = defaultdict(float)
        sec_titles: dict[str, list[str]] = defaultdict(list)
        for key in q_titles:
            raw, idf = title_info[key]
            th = title_hits.get(key, set())
            for sid in postings.get(key, ()):
                if sid not in allowed:
                    continue
                # ★ 标题命中远重于正文命中（见 TITLE_MATCH_BOOST 说明）
                w = TITLE_MATCH_BOOST if sid in th else 1.0
                sec_score[sid] += idf * w
                sec_titles[sid].append(raw)

        ranked = sorted(sec_score.items(), key=lambda kv: -kv[1])[:max_gt_sections]
        gt_sections = [sid for sid, _ in ranked]
        gt_chunks = sorted({c for sid in gt_sections for c in sec2chunks.get(sid, [])})

        items.append(
            EvalItem(
                qid=q.qid,
                year=q.year,
                subject=q.subject,
                question=q.stem,
                options=q.options,
                answer=q.answer,
                tags=q.tags,
                gt_section_ids=gt_sections,
                gt_chunks=gt_chunks,
                match_score=round(ranked[0][1], 2) if ranked else 0,
                matched_titles=sorted(sec_titles[gt_sections[0]]) if gt_sections else [],
                is_holdout=q.year in holdout,
            )
        )

    return items


# ---------------------------------------------------------------------------
# 统计与导出
# ---------------------------------------------------------------------------


def eval_stats(items: Sequence[EvalItem]) -> dict[str, Any]:
    import statistics

    usable = [it for it in items if it.gt_size > 0]
    sizes = [it.gt_size for it in usable]
    zero = [it for it in items if it.gt_size == 0]

    def q(p: float) -> float:
        if not sizes:
            return 0.0
        s = sorted(sizes)
        return float(s[min(int(len(s) * p), len(s) - 1)])

    return {
        "total_questions": len(items),
        "usable": len(usable),
        "coverage": round(len(usable) / max(len(items), 1), 4),
        "no_ground_truth": len(zero),
        "no_ground_truth_qids": [it.qid for it in zero][:30],
        "gt_size": {
            "min": min(sizes) if sizes else 0,
            "p25": q(0.25),
            "median": statistics.median(sizes) if sizes else 0,
            "mean": round(statistics.mean(sizes), 2) if sizes else 0,
            "p75": q(0.75),
            "p95": q(0.95),
            "max": max(sizes) if sizes else 0,
        },
        "gt_size_le_10": sum(1 for n in sizes if n <= 10),
        "gt_size_gt_50": sum(1 for n in sizes if n > 50),
        "holdout_usable": sum(1 for it in usable if it.is_holdout),
        "dev_usable": sum(1 for it in usable if not it.is_holdout),
    }


def export_eval_set(
    items: Sequence[MdSection] | Sequence[EvalItem],
    *,
    name: str = "eval_set",
    review_sample: int = 60,
) -> dict[str, str]:
    ensure_dirs()
    rows = [it.to_dict() for it in items]  # type: ignore[union-attr]

    p = EVAL_DIR / f"{name}.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    st = eval_stats(items)  # type: ignore[arg-type]
    sp = EVAL_DIR / f"{name}_stats.json"
    sp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人工抽检样本：分层抽（有 GT 的、无 GT 的、GT 偏大的）
    import random

    rng = random.Random(42)
    usable = [it for it in items if it.gt_size > 0]  # type: ignore[union-attr]
    zero = [it for it in items if it.gt_size == 0]  # type: ignore[union-attr]
    big = sorted(usable, key=lambda x: -x.gt_size)[:10]  # type: ignore[union-attr]

    sample = (
        rng.sample(usable, min(review_sample, len(usable)))
        + rng.sample(zero, min(10, len(zero)))
        + big
    )
    rp = EVAL_DIR / "samples_for_review.json"
    rp.write_text(
        json.dumps(
            [
                {
                    "qid": it.qid,
                    "subject": it.subject,
                    "question": it.question[:120],
                    "tags": it.tags,
                    "matched_titles": it.matched_titles,
                    "gt_size": it.gt_size,
                    "match_score": it.match_score,
                    "问题": "请核对：matched_titles 是否确实对应题干考点？gt_size 是否合理？",
                }
                for it in sample
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"评测集 -> {p}")
    logger.info(f"统计   -> {sp}")
    logger.info(f"人工抽检样本 -> {rp}（{len(sample)} 条）")
    return {"eval_set": str(p), "stats": str(sp), "review": str(rp)}


if __name__ == "__main__":
    from cs408rag.ingestion.chunker import chunk_md_sections
    from cs408rag.ingestion.md_loader import load_wangdao
    from cs408rag.ingestion.quiz_loader import load_questions
    from cs408rag.ingestion.tag_mapper import load_mapping
    from cs408rag.ingestion.xmind_loader import load_all_mindmaps

    ensure_dirs()
    sections = load_wangdao()
    chunks = chunk_md_sections(sections)
    questions = load_questions()
    kps = [n for mm in load_all_mindmaps() for n in mm.nodes]
    tag_map = load_mapping()

    items = build_eval_set(questions, sections, chunks, kps, tag_map)
    out = export_eval_set(items)
    st = eval_stats(items)
    print(json.dumps(st, ensure_ascii=False, indent=2))