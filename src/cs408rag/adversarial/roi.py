"""高性价比知识点：只用真题统计与知识点映射，不用 LLM 猜。

「性价比」是启发式，不是押题：

    yield  考查频次 + 近 5 年出现 + 计算/图表占比 + 年份跨度
    cost   知识点标题数 + 前置链深度（学起来要多久）
    roi    0.7 × yield + 0.3 × (1 - cost)

所有分项都写进 Plan，前端可展开核对；标签 S/A/B/C 只是排序，不是「必考」。
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from cs408rag.agent.exam_analysis import ExamAnalyzer, qid_to_ref
from cs408rag.config import PROCESSED_DIR

# 最近 N 个考试年份。数据集覆盖 2009–2026，latest_year 由数据动态取。
RECENT_YEARS = 5


@lru_cache(maxsize=1)
def load_kp_titles() -> dict[str, list[str]]:
    path = PROCESSED_DIR / "tag_kp_mapping.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r.get("tag")): list(r.get("kp_titles") or []) for r in rows}


@lru_cache(maxsize=1)
def load_prereq_depths() -> dict[str, int]:
    """知识点标题 -> 前置链深度（无前置为 0）。"""
    path = PROCESSED_DIR / "prerequisites.json"
    if not path.exists():
        return {}
    edges = json.loads(path.read_text(encoding="utf-8")).get("edges", [])
    pre_of: dict[str, list[str]] = {}
    for e in edges:
        to_t, from_t = e.get("to_title"), e.get("from_title")
        if to_t and from_t:
            pre_of.setdefault(to_t, []).append(from_t)

    memo: dict[str, int] = {}

    def depth(title: str, stack: set[str]) -> int:
        if title in memo:
            return memo[title]
        if title in stack:
            return 0
        pres = pre_of.get(title) or []
        d = 0 if not pres else 1 + max(depth(p, stack | {title}) for p in pres)
        memo[title] = d
        return d

    for title in list(pre_of):
        depth(title, set())
    return memo


@dataclass
class TagROI:
    tag: str
    subject: str
    frequency: int
    recent_5y: int
    span: int
    types: dict[str, int]
    kp_titles: list[str]
    prereq_depth: int
    yield_score: float
    cost_score: float
    roi: float
    roi_label: str
    evidence: list[str] = field(default_factory=list)
    subtopics: list[dict[str, Any]] = field(default_factory=list)
    must_learn: list[str] = field(default_factory=list)
    later: list[str] = field(default_factory=list)
    why: str = ""
    weight: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "kp": self.kp_titles[0] if self.kp_titles else self.tag,
            "subject": self.subject,
            "frequency": self.frequency,
            "recent_5y": self.recent_5y,
            "span": self.span,
            "types": self.types,
            "kp_titles": self.kp_titles,
            "prereq_depth": self.prereq_depth,
            "yield_score": round(self.yield_score, 4),
            "cost_score": round(self.cost_score, 4),
            "roi": round(self.roi, 4),
            "roi_label": self.roi_label,
            "evidence": self.evidence,
            "subtopics": self.subtopics,
            "must_learn": self.must_learn,
            "later": self.later,
            "why": self.why,
            "weight": self.weight,
        }


def _subject_questions(analyzer: ExamAnalyzer, tag: str, subject: str) -> list[Any]:
    return [q for q in analyzer.by_tag.get(tag, []) if q.subject == subject]


def _types_text(types: dict[str, int]) -> str:
    names = {"concept": "概念", "calc": "计算", "figure": "图表", "cross_subject": "跨科目"}
    parts = [f"{names.get(k, k)} {v}" for k, v in types.items() if v]
    return " / ".join(parts) or "无题型数据"


def _roi_label(roi: float) -> str:
    if roi >= 0.7:
        return "S"
    if roi >= 0.5:
        return "A"
    if roi >= 0.3:
        return "B"
    return "C"


def score_subject_tags(
    analyzer: ExamAnalyzer,
    subject: str,
    *,
    recent_years: int = RECENT_YEARS,
) -> list[TagROI]:
    """给一个科目的全部 tag 打分，按 roi 降序返回。"""
    latest_year = max((q.year for q in analyzer.questions), default=2026)
    recent_from = latest_year - recent_years + 1
    kp_map = load_kp_titles()
    depths = load_prereq_depths()

    raw: list[dict[str, Any]] = []
    for tag in analyzer.by_tag:
        qs = _subject_questions(analyzer, tag, subject)
        if not qs:
            continue
        types = dict(Counter(analyzer.type_of.get(q.qid, "concept") for q in qs))
        recent = sum(1 for q in qs if q.year >= recent_from)
        years = sorted({q.year for q in qs})
        titles = kp_map.get(tag, [])
        prereq_depth = max((depths.get(t, 0) for t in titles), default=0)
        cost_raw = 1.0 + 0.5 * max(0, len(titles) - 1) + 0.25 * prereq_depth
        raw.append(
            {
                "tag": tag,
                "qs": qs,
                "frequency": len(qs),
                "recent": recent,
                "span": len(years),
                "types": types,
                "titles": titles,
                "prereq_depth": prereq_depth,
                "cost_raw": cost_raw,
            }
        )
    if not raw:
        return []

    max_freq = max(r["frequency"] for r in raw) or 1
    max_recent = max(r["recent"] for r in raw) or 1
    max_span = max(r["span"] for r in raw) or 1
    max_cost = max(r["cost_raw"] for r in raw) or 1.0

    out: list[TagROI] = []
    for r in raw:
        freq = r["frequency"]
        calc_fig = r["types"].get("calc", 0) + r["types"].get("figure", 0)
        yield_score = (
            0.45 * freq / max_freq
            + 0.25 * r["recent"] / max_recent
            + 0.15 * calc_fig / freq
            + 0.15 * r["span"] / max_span
        )
        cost_score = r["cost_raw"] / max_cost
        roi = 0.7 * yield_score + 0.3 * (1.0 - cost_score)

        # 证据优先给最近的真题
        recent_qs = sorted(r["qs"], key=lambda q: (-q.year, q.num))[:3]
        evidence = [qid_to_ref(q.qid) for q in recent_qs]

        subtopics: list[dict[str, Any]] = []
        for rel in analyzer.related_tags(r["tag"], top_n=10):
            other = rel["tag"]
            other_qs = _subject_questions(analyzer, other, subject)
            if not other_qs:
                continue
            other_types = dict(
                Counter(analyzer.type_of.get(q.qid, "concept") for q in other_qs)
            )
            subtopics.append(
                {
                    "tag": other,
                    "frequency": len(other_qs),
                    "types": other_types,
                }
            )
        subtopics.sort(key=lambda x: (-x["frequency"], x["tag"]))
        subtopics = subtopics[:8]

        strong = [
            s
            for s in subtopics
            if s["frequency"] >= 2
            or s["types"].get("calc", 0)
            or s["types"].get("figure", 0)
        ]
        must = [s["tag"] for s in strong][:4]
        if not must:
            must = r["titles"][:2] or [r["tag"]]
        # 「可后置」只能是低频且没有计算/图表收益的子项；
        # 高频子项即使没进 must，也不能标成可后置。
        later = [
            s["tag"]
            for s in subtopics
            if s["tag"] not in must
            and s["frequency"] < 2
            and not s["types"].get("calc", 0)
            and not s["types"].get("figure", 0)
        ][:3]

        share = calc_fig / freq
        why = (
            f"共 {freq} 次、近 {recent_years} 年 {r['recent']} 次，"
            f"覆盖 {r['span']} 个年份；{_types_text(r['types'])}"
        )
        if share >= 0.3:
            why += f"；计算/图表占 {share:.0%}，适合先拿模板分"
        elif r["recent"] >= 2:
            why += "；近年仍有出现"

        out.append(
            TagROI(
                tag=r["tag"],
                subject=subject,
                frequency=freq,
                recent_5y=r["recent"],
                span=r["span"],
                types=r["types"],
                kp_titles=r["titles"],
                prereq_depth=r["prereq_depth"],
                yield_score=yield_score,
                cost_score=cost_score,
                roi=roi,
                roi_label=_roi_label(roi),
                evidence=evidence,
                subtopics=subtopics,
                must_learn=must,
                later=later,
                why=why,
            )
        )
    return sorted(out, key=lambda x: (-x.roi, -x.frequency, x.tag))


def select_high_value(
    analyzer: ExamAnalyzer,
    subject: str,
    *,
    remaining_weeks: int,
    boost_tags: Iterable[str] = (),
    top_n: int | None = None,
) -> list[TagROI]:
    """按剩余时间选每科考点。

    - 短周期只看前 10 个，长周期前 20 个
    - 频次前 5 必选（规则 missing_high_freq_kp 的下限）
    - 频次 ≥8 必选
    - 用户自述的具体知识点必选
    """
    rows = score_subject_tags(analyzer, subject)
    if not rows:
        return []
    if top_n is None:
        if remaining_weeks <= 8:
            top_n = 10
        elif remaining_weeks <= 16:
            top_n = 15
        else:
            top_n = 20
    top_n = max(5, min(top_n, len(rows)))

    boost = set(boost_tags)
    mandatory = {r.tag for r in rows if r.frequency >= 8 or r.tag in boost}
    mandatory |= {r.tag for r in sorted(rows, key=lambda x: (-x.frequency, x.tag))[:5]}

    ordered = sorted(
        rows,
        key=lambda r: (r.tag in boost, r.roi, r.frequency),
        reverse=True,
    )
    chosen = ordered[:top_n]
    chosen_tags = {r.tag for r in chosen}
    missing = [r for r in ordered if r.tag in mandatory and r.tag not in chosen_tags]
    if missing:
        # 替换掉非必选里 roi 最低的项
        replaceable = [r for r in chosen if r.tag not in mandatory]
        for r in missing:
            if replaceable:
                drop = min(replaceable, key=lambda x: x.roi)
                replaceable.remove(drop)
                chosen[chosen.index(drop)] = r
            else:
                chosen.append(r)
    # 用户点名的考点必须排在最前，不能被最后的 roi 排序冲掉
    return sorted(
        chosen,
        key=lambda x: (x.tag not in boost, -x.roi, -x.frequency, x.tag),
    )


def subject_coverage(
    analyzer: ExamAnalyzer, subject: str, selected: Iterable[TagROI]
) -> dict[str, Any]:
    """选中的考点覆盖了该科多少 tag 次数（按 tag 计，不是按题去重）。"""
    selected_tags = {r.tag for r in selected}
    total = 0
    covered = 0
    for tag in analyzer.by_tag:
        n = len(_subject_questions(analyzer, tag, subject))
        if not n:
            continue
        total += n
        if tag in selected_tags:
            covered += n
    return {
        "covered": covered,
        "total": total,
        "ratio": round(covered / total, 4) if total else 0.0,
    }


def prereq_order(analyzer: ExamAnalyzer, rows: list[TagROI]) -> list[TagROI]:
    """按前置关系排序选中的考点；没有边的保持 roi 降序。

    只对 tag 的 kp_titles 建边；同一 tag 内部不拆。
    """
    if len(rows) <= 1:
        return rows
    title_to_tag: dict[str, str] = {}
    for r in rows:
        for t in r.kp_titles or [r.tag]:
            title_to_tag.setdefault(t, r.tag)

    path = PROCESSED_DIR / "prerequisites.json"
    edges: list[tuple[str, str]] = []
    if path.exists():
        for e in json.loads(path.read_text(encoding="utf-8")).get("edges", []):
            pre, post = e.get("from_title"), e.get("to_title")
            if pre in title_to_tag and post in title_to_tag:
                a, b = title_to_tag[pre], title_to_tag[post]
                if a != b:
                    edges.append((a, b))

    order = {r.tag: i for i, r in enumerate(rows)}
    # Kahn 拓扑；同层按原 roi 顺序
    incoming: dict[str, set[str]] = {r.tag: set() for r in rows}
    outgoing: dict[str, set[str]] = {r.tag: set() for r in rows}
    for a, b in edges:
        incoming[b].add(a)
        outgoing[a].add(b)

    ready = sorted([t for t, inc in incoming.items() if not inc], key=lambda t: order[t])
    out: list[str] = []
    while ready:
        t = ready.pop(0)
        out.append(t)
        for nxt in sorted(outgoing[t], key=lambda x: order[x]):
            incoming[nxt].discard(t)
            if not incoming[nxt] and nxt not in out and nxt not in ready:
                ready.append(nxt)
        ready.sort(key=lambda x: order[x])
    # 有环或遗漏时按 roi 顺序补齐
    for r in rows:
        if r.tag not in out:
            out.append(r.tag)
    by_tag = {r.tag: r for r in rows}
    return [by_tag[t] for t in out]