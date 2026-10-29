"""对抗 Plan 状态机的状态定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class PlanState(TypedDict, total=False):
    goal: dict[str, Any]
    plan: dict[str, Any]
    # 每个版本一份快照，用于前端展示两版 diff；不参与规则判定
    plan_history: Annotated[list[dict[str, Any]], operator.add]
    version: int
    round: int
    challenges: list[dict[str, Any]]
    # 跨轮累计：已解决的挑战不能因为最后一轮没再出现就消失
    challenge_history: list[dict[str, Any]]
    verdicts: list[dict[str, Any]]
    prev_high: list[str]
    timeline: Annotated[list[dict[str, Any]], operator.add]
    tokens: dict[str, int]
    llm_calls: int
    llm_errors: int
    stop_reason: str
    plan_source: str
    guide_source: str
    use_llm: bool
    max_rounds: int
    started_at: float
    planner_fallback: bool


def initial_state(
    goal: dict[str, Any],
    *,
    use_llm: bool = True,
    max_rounds: int = 2,
) -> PlanState:
    return PlanState(
        goal=goal,
        plan={},
        plan_history=[],
        version=0,
        round=0,
        challenges=[],
        challenge_history=[],
        verdicts=[],
        prev_high=[],
        timeline=[],
        tokens={},
        llm_calls=0,
        llm_errors=0,
        stop_reason="",
        plan_source="deterministic",
        guide_source="template",
        use_llm=use_llm,
        max_rounds=max_rounds,
        started_at=0.0,
        planner_fallback=False,
    )