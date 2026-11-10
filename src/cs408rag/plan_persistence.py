"""规划落库：plans 表 + profile_events 同事务。

这层不依赖 Streamlit，Streamlit（plan_ui）和 FastAPI（api/main）共用。
规划是按 client_id 绑定的独立记录，不属于任何会话。
"""

from __future__ import annotations

from typing import Any

from cs408rag.profile.store import get_profile_store


def persist_plan(
    client_id: str,
    result: dict[str, Any],
    goal: dict[str, Any] | None = None,
    *,
    stop_reason: str | None = None,
) -> bool:
    """规划落库 + 画像事件一次事务写入；成功返回 True。

    `result` 是 PlanResult.to_dict() 或 generate_plan 工具的完整产出，
    都要带 `plan` 字段；`planner_fallback` 为真的结果视为不可信，不落库。
    """
    plan_out = result.get("plan") or {}
    if not plan_out or result.get("planner_fallback"):
        return False
    g = goal or result.get("goal") or plan_out.get("goal") or {}
    per_subj: dict[str, list[str]] = {}
    for k in plan_out.get("priority_kps") or []:
        per_subj.setdefault(str(k.get("subject") or ""), []).append(
            str(k.get("tag") or "")
        )
    get_profile_store().save_plan_with_event(
        client_id,
        result,
        "plan",
        {
            "weak_subjects": g.get("weak_subjects") or [],
            "concerns": g.get("concerns") or [],
            "priority_tags": [
                {"subject": s, "tag": t}
                for s, ts in per_subj.items()
                for t in ts[:2]
                if s and t
            ],
            "stop_reason": stop_reason or result.get("stop_reason") or "",
            "version": result.get("version") or 0,
        },
    )
    return True
