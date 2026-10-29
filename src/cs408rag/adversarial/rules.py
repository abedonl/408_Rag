"""Plan 的 6 类可验证规则。

每条挑战都带 evidence 与 locator，便于前端展开和人工核对。
规则全部是纯函数：不调 LLM，结果可复现。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Sequence

from loguru import logger

from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer, normalize_qid
from cs408rag.config import PROCESSED_DIR
from cs408rag.generation.citation import extract_exam_refs

SEVERITIES = ("high", "medium", "low")


@dataclass
class Challenge:
    rule: str
    severity: str
    message: str
    evidence: dict[str, Any] = field(default_factory=dict)
    locator: str = ""
    status: str = "open"  # open | resolved | dropped

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "evidence": self.evidence,
            "locator": self.locator,
            "status": self.status,
        }

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.rule, self.locator, self.message)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _all_evidence(plan: dict[str, Any]) -> list[tuple[str, Sequence[str]]]:
    """(位置, 引用列表)。"""
    out: list[tuple[str, Sequence[str]]] = []
    for i, kp in enumerate(plan.get("priority_kps") or []):
        out.append((f"priority_kps[{i}].evidence", kp.get("evidence") or []))
    for wi, week in enumerate(plan.get("weekly_plan") or []):
        for ti, task in enumerate(week.get("tasks") or []):
            out.append((f"weekly_plan[{wi}].tasks[{ti}].evidence", task.get("evidence") or []))
    return out


def _valid_refs(refs: Sequence[str], analyzer: ExamAnalyzer) -> list[str]:
    valid: list[str] = []
    for ref in extract_exam_refs(" ".join(str(r) for r in refs)):
        if analyzer.question(normalize_qid(ref)) is not None:
            valid.append(ref)
    return valid


def _kp_weeks(plan: dict[str, Any]) -> dict[str, list[int]]:
    """知识点标题 -> 出现的周次（升序）。"""
    out: dict[str, list[int]] = defaultdict(list)
    for week in plan.get("weekly_plan") or []:
        w = int(week.get("week") or 0)
        for task in week.get("tasks") or []:
            for kp in task.get("kps") or []:
                if kp and w not in out[kp]:
                    out[kp].append(w)
    return {k: sorted(v) for k, v in out.items()}


def _subject_hours(plan: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for week in plan.get("weekly_plan") or []:
        for task in week.get("tasks") or []:
            out[str(task.get("subject") or "")] += float(task.get("hours") or 0)
    return dict(out)


def _load_offline_prereqs() -> list[dict[str, Any]]:
    path = PROCESSED_DIR / "prerequisites.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("edges", [])


def _covered_by_plan(tag: str, plan: dict[str, Any]) -> bool:
    for kp in plan.get("priority_kps") or []:
        name = str(kp.get("kp") or "")
        if kp.get("tag") == tag or tag in name or (name and name in tag):
            return True
    for week in plan.get("weekly_plan") or []:
        for task in week.get("tasks") or []:
            for kp in task.get("kps") or []:
                if tag in str(kp) or (kp and str(kp) in tag):
                    return True
    return False


# ---------------------------------------------------------------------------
# 规则
# ---------------------------------------------------------------------------


def _rule_fabricated_ref(
    plan: dict[str, Any], analyzer: ExamAnalyzer
) -> list[Challenge]:
    out: list[Challenge] = []
    for locator, refs in _all_evidence(plan):
        if not refs:
            continue
        for ref in extract_exam_refs(" ".join(str(r) for r in refs)):
            if analyzer.question(normalize_qid(ref)) is None:
                out.append(
                    Challenge(
                        rule="fabricated_ref",
                        severity="high",
                        message=f"真题编号 {ref} 不存在于题库",
                        evidence={"ref": ref},
                        locator=locator,
                    )
                )
    return out


def _rule_time_overrun(plan: dict[str, Any]) -> list[Challenge]:
    goal = plan.get("goal") or {}
    limit = float(goal.get("weekly_hours") or 0)
    if limit <= 0:
        return []
    out: list[Challenge] = []
    for week in plan.get("weekly_plan") or []:
        total = sum(float(t.get("hours") or 0) for t in week.get("tasks") or [])
        ratio = total / limit
        if ratio <= 1.1:
            continue
        out.append(
            Challenge(
                rule="time_overrun",
                severity="high" if ratio > 1.5 else "medium",
                message=f"第 {week.get('week')} 周安排 {total:g} 小时，超过预算 {limit:g} 小时",
                evidence={"week": week.get("week"), "sum": total, "limit": limit},
                locator=f"weekly_plan[week={week.get('week')}]",
            )
        )
    return out


def _rule_missing_high_freq(
    plan: dict[str, Any], analyzer: ExamAnalyzer
) -> list[Challenge]:
    goal = plan.get("goal") or {}
    # 全科模式检查四科；单科模式只查薄弱科目
    subjects = (
        list(goal.get("subjects") or [])
        if goal.get("scope") == "full_408"
        else list(goal.get("weak_subjects") or [])
    )
    out: list[Challenge] = []
    for subject in subjects:
        # 必须按「该科目下」的题数排序。旧实现用 any(q.subject==subject)
        # 会把只在别的科目考过的 tag 也算进来，和 ROI 选点口径不一致。
        counts = []
        for tag in analyzer.by_tag:
            n = sum(1 for q in analyzer.by_tag[tag] if q.subject == subject)
            if n:
                counts.append((n, tag))
        counts.sort(key=lambda x: (-x[0], x[1]))
        tags = [t for _, t in counts[:5]]
        missing = [t for t in tags if not _covered_by_plan(t, plan)]
        for t in missing:
            out.append(
                Challenge(
                    rule="missing_high_freq_kp",
                    severity="high",
                    message=f"薄弱科目「{subject}」的高频考点「{t}」未出现在 Plan 中",
                    evidence={"tag": t, "subject": subject},
                    locator="priority_kps",
                )
            )
    return out


def _rule_prerequisite_order(
    plan: dict[str, Any], graph: Any | None
) -> list[Challenge]:
    weeks = _kp_weeks(plan)
    if len(weeks) < 2:
        return []
    # 出现在多周的标题有歧义，跳过
    names = [k for k, w in weeks.items() if len(w) == 1]
    if len(names) < 2:
        return []

    pairs: list[tuple[str, str]] = []
    if graph is not None:
        try:
            pairs = graph.prerequisites_among(names)
        except Exception as e:  # noqa: BLE001 - 图谱不可用时回退离线
            logger.warning(f"图谱前置查询失败，回退离线数据：{e}")
    if not pairs:
        name_set = set(names)
        for e in _load_offline_prereqs():
            pre, post = e.get("from_title"), e.get("to_title")
            if pre in name_set and post in name_set:
                pairs.append((pre, post))

    out: list[Challenge] = []
    for pre, post in pairs:
        wp, wn = weeks[pre][0], weeks[post][0]
        if wp > wn:
            out.append(
                Challenge(
                    rule="prerequisite_order",
                    severity="high",
                    message=f"「{pre}」是「{post}」的前置，但被安排在更晚的周次",
                    evidence={"pre": pre, "post": post, "pre_week": wp, "post_week": wn},
                    locator=f"weekly_plan[week={wp}/{wn}]",
                )
            )
        elif wp == wn:
            out.append(
                Challenge(
                    rule="prerequisite_order",
                    severity="medium",
                    message=f"「{pre}」与「{post}」有前置关系，被安排在同一周",
                    evidence={"pre": pre, "post": post, "week": wp},
                    locator=f"weekly_plan[week={wp}]",
                )
            )
    return out


def _rule_uncited_priority(
    plan: dict[str, Any], analyzer: ExamAnalyzer
) -> list[Challenge]:
    out: list[Challenge] = []
    task_refs = _valid_refs(
        [r for _, refs in _all_evidence(plan) if _.startswith("weekly_plan") for r in refs],
        analyzer,
    )
    for i, kp in enumerate(plan.get("priority_kps") or []):
        weight = float(kp.get("weight") or 0)
        if weight < 0.15:
            continue
        if _valid_refs(kp.get("evidence") or [], analyzer):
            continue
        out.append(
            Challenge(
                rule="uncited_priority",
                severity="medium" if task_refs else "high",
                message=f"高权重知识点「{kp.get('kp')}」（{weight:g}）没有真实真题引用",
                evidence={"kp": kp.get("kp"), "weight": weight},
                locator=f"priority_kps[{i}].evidence",
            )
        )
    return out


def _rule_weak_subject_underweight(plan: dict[str, Any]) -> list[Challenge]:
    goal = plan.get("goal") or {}
    weak = goal.get("weak_subjects") or []
    hours = _subject_hours(plan)
    total = sum(hours.values())
    if total <= 0:
        return []
    out: list[Challenge] = []
    for subject in weak:
        ratio = hours.get(subject, 0.0) / total
        if ratio >= 0.3:
            continue
        out.append(
            Challenge(
                rule="weak_subject_underweight",
                severity="high" if ratio < 0.2 else "medium",
                message=f"薄弱科目「{subject}」只占 {ratio:.0%} 的学习时间",
                evidence={"subject": subject, "ratio": round(ratio, 3), "total_hours": total},
                locator="weekly_plan",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def review_plan(
    plan: dict[str, Any],
    *,
    analyzer: ExamAnalyzer | None = None,
    graph: Any | None = None,
) -> list[Challenge]:
    """跑全部规则，返回去重后的挑战列表。"""
    analyzer = analyzer or get_analyzer()
    challenges: list[Challenge] = []
    challenges += _rule_fabricated_ref(plan, analyzer)
    challenges += _rule_time_overrun(plan)
    challenges += _rule_missing_high_freq(plan, analyzer)
    challenges += _rule_prerequisite_order(plan, graph)
    challenges += _rule_uncited_priority(plan, analyzer)
    challenges += _rule_weak_subject_underweight(plan)

    seen: set[tuple[str, str, str]] = set()
    out: list[Challenge] = []
    for c in challenges:
        if c.key in seen:
            continue
        seen.add(c.key)
        out.append(c)
    out.sort(key=lambda c: (SEVERITIES.index(c.severity), c.rule))
    return out