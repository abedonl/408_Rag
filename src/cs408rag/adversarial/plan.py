"""备考 Plan 的兼容入口与错误注入。

真正的结构化生成在 `planner.py`（日期/阶段/科目份额/ROI/周任务/经验贴模板）。
这里保留旧函数签名，给单测、数据集和旧调用方用；并负责「注入已知错误」，
用于评测规则检出率。
"""

from __future__ import annotations

import copy
import json
from datetime import date, timedelta
from typing import Any

from cs408rag.adversarial.planner import SUBJECTS, build_detailed_plan
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer
from cs408rag.config import PROCESSED_DIR

# 注入用的错误类型，与规则一一对应
ERROR_TYPES = (
    "fabricated_ref",
    "time_overrun",
    "missing_high_freq_kp",
    "prerequisite_order",
    "uncited_priority",
    "weak_subject_underweight",
)


def build_golden_plan(
    *,
    weak_subject: str = "操作系统",
    weeks: int = 12,
    weekly_hours: int = 10,
    top_n_tags: int = 5,  # 兼容旧签名，新生成器按剩余时间决定考点数
    scope: str = "full_408",
    progress: dict[str, str] | None = None,
    concerns: list[str] | None = None,
) -> dict[str, Any]:
    """用真实题库构造一个「规则上没问题」的详细 Plan。"""
    start = date.today()
    exam = start + timedelta(weeks=max(1, int(weeks)))
    goal = {
        "scope": scope,
        "weak_subjects": [weak_subject] if weak_subject else [],
        "start_date": start.isoformat(),
        "exam_date": exam.isoformat(),
        "daily_hours": max(0.5, float(weekly_hours) / 4),
        "study_days_per_week": 4,
        "target_score": 115,
        "progress": progress or {s: "一轮中" for s in SUBJECTS},
        "concerns": concerns or [],
    }
    return build_detailed_plan(goal, get_analyzer())


# ---------------------------------------------------------------------------
# 注入
# ---------------------------------------------------------------------------


def _subject_top_tag(
    analyzer: ExamAnalyzer, subject: str, plan: dict[str, Any]
) -> str | None:
    counts = []
    for tag in analyzer.by_tag:
        n = sum(1 for q in analyzer.by_tag[tag] if q.subject == subject)
        if n:
            counts.append((n, tag))
    counts.sort(key=lambda x: (-x[0], x[1]))
    return counts[0][1] if counts else None


def _strip_tag(plan: dict[str, Any], tag: str) -> None:
    """删掉一个考点及其所有「包含匹配」的别名。

    规则 `_covered_by_plan` 允许 tag 与知识点标题互相包含；只删完全相等的
    tag 会留下「调度算法」这类别名，注入后仍被判为已覆盖。
    """

    def related(value: Any) -> bool:
        s = str(value or "")
        return bool(s) and (s == tag or tag in s or s in tag)

    for kp in plan.get("priority_kps") or []:
        if related(kp.get("tag")) or related(kp.get("kp")):
            kp["evidence"] = []
    plan["priority_kps"] = [
        kp
        for kp in (plan.get("priority_kps") or [])
        if not related(kp.get("tag")) and not related(kp.get("kp"))
    ]
    for week in plan.get("weekly_plan") or []:
        for task in week.get("tasks") or []:
            task["kps"] = [k for k in (task.get("kps") or []) if not related(k)]
            text = str(task.get("content") or "")
            if tag in text:
                task["content"] = text.replace(tag, "（已移除考点）")


def _inject_prerequisite(plan: dict[str, Any]) -> bool:
    """把一对真实前置边拆到「前置在后、后置在前」的两周。"""
    path = PROCESSED_DIR / "prerequisites.json"
    if not path.exists():
        return False
    edges = json.loads(path.read_text(encoding="utf-8")).get("edges", [])
    if not edges:
        return False
    weeks = plan.get("weekly_plan") or []
    if len(weeks) < 2:
        return False

    # 先清掉所有任务里的候选标题，避免同一标题出现在多周被规则跳过
    candidates = {(e.get("from_title"), e.get("to_title")) for e in edges}
    titles = {t for pair in candidates for t in pair if t}
    for week in weeks:
        for task in week.get("tasks") or []:
            task["kps"] = [k for k in (task.get("kps") or []) if k not in titles]

    first = weeks[0]["tasks"][0]
    last = weeks[-1]["tasks"][0]
    pre, post = next(iter(candidates))
    first["kps"] = list(dict.fromkeys([post, *(first.get("kps") or [])]))
    last["kps"] = list(dict.fromkeys([pre, *(last.get("kps") or [])]))
    return True


def inject(
    plan: dict[str, Any],
    error_type: str,
    *,
    analyzer: ExamAnalyzer | None = None,
) -> dict[str, Any]:
    """在黄金 Plan 上注入一个已知错误，返回新 Plan（不修改原对象）。"""
    analyzer = analyzer or get_analyzer()
    p = copy.deepcopy(plan)

    if error_type == "fabricated_ref":
        kps = p.get("priority_kps") or []
        if kps:
            kps[0]["evidence"] = ["2099#99"]

    elif error_type == "time_overrun":
        weekly = float((p.get("goal") or {}).get("weekly_hours") or 10)
        week = (p.get("weekly_plan") or [{}])[0]
        tasks = week.get("tasks") or []
        if tasks:
            tasks[0]["hours"] = max(weekly * 2, tasks[0].get("hours", 0) + weekly)

    elif error_type == "missing_high_freq_kp":
        goal = p.get("goal") or {}
        subjects = goal.get("weak_subjects") or goal.get("subjects") or ["操作系统"]
        subject = subjects[0]
        tag = _subject_top_tag(analyzer, subject, p)
        if tag:
            _strip_tag(p, tag)

    elif error_type == "prerequisite_order":
        if not _inject_prerequisite(p):
            # 回退：直接塞一对已知标题
            weeks = p.get("weekly_plan") or []
            if len(weeks) >= 2:
                weeks[0]["tasks"][0]["kps"] = ["中断和异常的处理"]
                weeks[-1]["tasks"][0]["kps"] = ["系统调用"]

    elif error_type == "uncited_priority":
        kps = p.get("priority_kps") or []
        if kps:
            kps[0]["weight"] = max(float(kps[0].get("weight") or 0), 0.2)
            kps[0]["evidence"] = []

    elif error_type == "weak_subject_underweight":
        goal = p.get("goal") or {}
        weak = set(goal.get("weak_subjects") or [])
        for week in p.get("weekly_plan") or []:
            for task in week.get("tasks") or []:
                if task.get("subject") in weak:
                    task["hours"] = 0.5

    else:
        raise ValueError(f"未知错误类型：{error_type}")

    return p