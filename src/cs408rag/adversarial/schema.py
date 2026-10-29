"""Plan 的校验、摘要与 diff。

Plan 是普通 dict，不引入 pydantic，避免 LLM 输出多一个字段就整份作废。
校验只检查「规则需要的字段是否存在且类型正确」，其余字段允许扩展。
"""

from __future__ import annotations

import difflib
import json
from typing import Any


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_plan(plan: Any) -> list[str]:
    """返回错误列表；空列表表示可以进入规则审查。"""
    errors: list[str] = []
    if not isinstance(plan, dict):
        return ["plan 必须是 JSON 对象"]

    goal = plan.get("goal")
    remaining_weeks: int | None = None
    if not isinstance(goal, dict):
        errors.append("缺少 goal 对象")
    else:
        weeks = goal.get("weeks", goal.get("remaining_weeks"))
        if not _is_number(weeks) or weeks <= 0:
            errors.append("goal.weeks / goal.remaining_weeks 必须是正数")
        else:
            remaining_weeks = int(weeks)
        if not _is_number(goal.get("weekly_hours")) or goal.get("weekly_hours") <= 0:
            errors.append("goal.weekly_hours 必须是正数")
        if not isinstance(goal.get("weak_subjects"), list):
            errors.append("goal.weak_subjects 必须是数组")
        if goal.get("scope") and goal.get("scope") not in ("full_408", "single_subject"):
            errors.append("goal.scope 只能是 full_408 / single_subject")
        for key in ("daily_hours", "study_days_per_week"):
            if key in goal and (not _is_number(goal.get(key)) or goal.get(key) <= 0):
                errors.append(f"goal.{key} 必须是正数")

    kps = plan.get("priority_kps")
    if not isinstance(kps, list):
        errors.append("priority_kps 必须是数组")
    else:
        for i, kp in enumerate(kps):
            if not isinstance(kp, dict) or not kp.get("kp"):
                errors.append(f"priority_kps[{i}] 缺少 kp")
                continue
            if not _is_number(kp.get("weight")):
                errors.append(f"priority_kps[{i}].weight 必须是数字")
            if not isinstance(kp.get("evidence", []), list):
                errors.append(f"priority_kps[{i}].evidence 必须是数组")

    stages = plan.get("stages")
    stage_names: set[str] = set()
    if stages is not None:
        if not isinstance(stages, list) or not stages:
            errors.append("stages 必须是非空数组")
        else:
            total_weeks = 0
            for i, stage in enumerate(stages):
                if not isinstance(stage, dict):
                    errors.append(f"stages[{i}] 必须是对象")
                    continue
                for key in ("name", "start", "end"):
                    if not stage.get(key):
                        errors.append(f"stages[{i}] 缺少 {key}")
                if not _is_number(stage.get("weeks")) or stage.get("weeks") <= 0:
                    errors.append(f"stages[{i}].weeks 必须是正数")
                else:
                    total_weeks += int(stage["weeks"])
                stage_names.add(str(stage.get("name")))
            if remaining_weeks is not None and total_weeks != remaining_weeks:
                errors.append(
                    f"stages 周数合计 {total_weeks} 与 remaining_weeks {remaining_weeks} 不一致"
                )

    weeks = plan.get("weekly_plan")
    if not isinstance(weeks, list):
        errors.append("weekly_plan 必须是数组")
    else:
        for i, week in enumerate(weeks):
            if not isinstance(week, dict) or not _is_number(week.get("week")):
                errors.append(f"weekly_plan[{i}] 缺少 week")
                continue
            if stage_names and week.get("stage") not in stage_names:
                errors.append(f"weekly_plan[{i}].stage 不在 stages 中")
            tasks = week.get("tasks")
            if not isinstance(tasks, list):
                errors.append(f"weekly_plan[{i}].tasks 必须是数组")
                continue
            for j, task in enumerate(tasks):
                if not isinstance(task, dict):
                    errors.append(f"weekly_plan[{i}].tasks[{j}] 必须是对象")
                    continue
                if not task.get("subject"):
                    errors.append(f"weekly_plan[{i}].tasks[{j}] 缺少 subject")
                if not _is_number(task.get("hours")) or task.get("hours") < 0:
                    errors.append(f"weekly_plan[{i}].tasks[{j}].hours 必须是非负数")
                if stages is not None:
                    for key in ("content", "method", "acceptance"):
                        if not str(task.get(key) or "").strip():
                            errors.append(
                                f"weekly_plan[{i}].tasks[{j}] 缺少 {key}"
                            )

    concerns = (goal or {}).get("concerns") if isinstance(goal, dict) else None
    if isinstance(concerns, list) and concerns:
        actions = plan.get("concern_actions")
        if not isinstance(actions, list):
            errors.append("goal.concerns 非空时 concern_actions 必须是数组")
        else:
            covered = {str(a.get("concern")) for a in actions if isinstance(a, dict)}
            for c in concerns:
                if str(c) not in covered:
                    errors.append(f"用户问题未回应：{c}")

    return errors


def plan_digest(plan: dict[str, Any], *, max_chars: int = 2400) -> str:
    """给 Critic 看的紧凑摘要，优先保留阶段、周任务和用户问题。"""
    goal = plan.get("goal") or {}
    lines = [
        f"目标分 {goal.get('target_score', '?')} | 考试 {goal.get('exam_date', '?')} | "
        f"剩余 {goal.get('remaining_weeks', goal.get('weeks', '?'))} 周 | "
        f"每天 {goal.get('daily_hours', '?')}h × {goal.get('study_days_per_week', '?')} 天 "
        f"= {goal.get('weekly_hours', '?')}h/周 | 范围 {goal.get('scope', '?')} | "
        f"薄弱 {'、'.join(goal.get('weak_subjects') or []) or '无'}",
    ]
    progress = goal.get("progress") or {}
    if progress:
        lines.append("当前进度：" + "、".join(f"{k}{v}" for k, v in progress.items()))

    stages = plan.get("stages") or []
    if stages:
        lines.append("阶段：")
        for s in stages:
            lines.append(
                f"- {s.get('name')} {s.get('start')}~{s.get('end')} "
                f"({s.get('weeks')} 周)：{'、'.join(s.get('focus') or [])}；"
                f"检查点 {str(s.get('checkpoint') or '')[:40]}"
            )

    lines.append("周计划：")
    weeks = plan.get("weekly_plan") or []
    hidden_note = ""
    for week in weeks[:4]:
        tasks = "；".join(
            f"{t.get('subject')} {t.get('hours')}h "
            f"{'、'.join(t.get('kps') or []) or '无考点'}："
            f"{str(t.get('content') or '')[:30]}｜{str(t.get('method') or '')[:25]}"
            f"｜验收 {str(t.get('acceptance') or '')[:20]}"
            for t in (week.get("tasks") or [])
        )
        lines.append(f"- 第 {week.get('week')} 周 [{week.get('stage')}]：{tasks}")
    if len(weeks) > 4:
        # 不写这句，Critic 会把「摘要截断」误判成「后面几周没安排」
        hidden_note = f"（摘要只展示前 4 周，另有 {len(weeks) - 4} 周未展示）"

    actions = plan.get("concern_actions") or []
    if actions:
        lines.append("用户问题与动作：")
        for a in actions:
            lines.append(
                f"- {a.get('concern')} → {str(a.get('action') or '')[:60]}"
                f"（阶段 {a.get('stage')}，检查点 {str(a.get('checkpoint') or '')[:40]}）"
            )

    lines.append("高性价比考点（前 8）：")
    for kp in (plan.get("priority_kps") or [])[:8]:
        refs = "、".join(kp.get("evidence") or []) or "无引用"
        lines.append(
            f"- {kp.get('subject', '')} {kp.get('tag') or kp.get('kp')} "
            f"[{kp.get('roi_label', '?')}] 共 {kp.get('frequency', '?')} 次 / "
            f"近 5 年 {kp.get('recent_5y', '?')} 次；学 {kp.get('must_learn') or '—'}；"
            f"证据 {refs}"
        )

    guide = plan.get("guide") or {}
    if guide.get("overview"):
        lines.append(f"方法建议：{str(guide['overview'])[:160]}")

    text = "\n".join(lines)
    if hidden_note:
        text += "\n" + hidden_note
    if len(text) <= max_chars:
        return text
    if hidden_note:
        # 先保证「还有 N 周未展示」这句话不被截掉，否则 Critic 会误判漏排
        keep = max(0, max_chars - len(hidden_note) - 2)
        return text[:keep] + "…\n" + hidden_note
    return text[: max_chars - 1] + "…"


def plan_diff(old: dict[str, Any] | None, new: dict[str, Any]) -> str:
    """两版 Plan 的文本 diff，用于前端展示修订轨迹。"""
    a = json.dumps(
        old or {}, ensure_ascii=False, indent=2, sort_keys=True
    ).splitlines()
    b = json.dumps(
        new or {}, ensure_ascii=False, indent=2, sort_keys=True
    ).splitlines()
    return "\n".join(
        difflib.unified_diff(
            a, b, fromfile="plan_v_prev", tofile="plan_v_new", lineterm=""
        )
    )