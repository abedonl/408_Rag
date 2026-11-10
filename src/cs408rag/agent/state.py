"""Agent 状态定义。

状态尽量扁平、可 JSON 序列化 —— 便于写 trace、做评测复现，
也避免把 RetrievalResult 这类重对象塞进图状态导致 checkpoint 失败。

plan-act-observe 循环只带**压缩后的 observation**（摘要 + 截断片段），
不把完整工具数据再复制一份进 state；完整数据留在 `tool_outputs` 里供 answer 用。
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

# 有界循环：最多 3 个工具轮次，总预算 90 秒。
# 规则 planner 优先，LLM planner 最多 1 次（见 nodes.plan）。
MAX_TOOL_ROUNDS = 3
DEFAULT_BUDGET_S = 90.0


class AgentState(TypedDict, total=False):
    # 输入
    question: str
    options: dict[str, str]
    subject: str | None
    qid: str | None
    mode: str  # "quiz" | "qa" | "exam"

    # 路由
    route: str  # concept / calc / figure / exam / cross
    route_reason: str
    topic: str | None

    # 检索
    queries: list[str]
    attempts: int
    contexts: list[dict[str, Any]]
    retrieval: dict[str, Any]

    # 工具输出
    tool_outputs: dict[str, Any]
    assessment: dict[str, Any]

    # plan-act-observe
    plan: dict[str, Any]
    plan_history: Annotated[list[dict[str, Any]], operator.add]
    observations: Annotated[list[dict[str, Any]], operator.add]
    last_result: dict[str, Any]
    tool_rounds: int
    planner_calls: int
    tool_failures: Annotated[list[dict[str, Any]], operator.add]
    stop_reason: str
    started_at: float
    budget_s: float

    # 生成
    answer: str
    choice: str
    llm_error: str
    empty: bool

    # 校验
    verification: dict[str, Any]
    verify_retries: int
    verify_feedback: str

    # 会话（Week 2）：history_text 是渲染好的 summary + 最近几轮，
    # 只在 answer 注入 prompt；完整结构化历史在 SessionStore，不进图状态。
    session_id: str
    history_text: str
    profile: dict[str, Any]

    # 澄清（Week 3 接 HITL；字段先占位，避免后续改 schema）
    needs_clarification: bool
    clarification_options: list[str]

    # 可观测性
    steps: Annotated[list[dict[str, Any]], operator.add]
    tokens: dict[str, int]
    took_ms: float


def initial_state(
    question: str,
    *,
    options: dict[str, str] | None = None,
    subject: str | None = None,
    qid: str | None = None,
    mode: str = "qa",
    budget_s: float = DEFAULT_BUDGET_S,
    session_id: str = "",
    history_text: str = "",
    profile: dict[str, Any] | None = None,
) -> AgentState:
    return AgentState(
        question=question,
        options=options or {},
        subject=subject,
        qid=qid,
        mode=mode,
        route="concept",
        topic=None,
        queries=[question],
        attempts=0,
        contexts=[],
        tool_outputs={},
        assessment={},
        plan={},
        plan_history=[],
        observations=[],
        last_result={},
        tool_rounds=0,
        planner_calls=0,
        tool_failures=[],
        stop_reason="",
        started_at=time.monotonic(),
        budget_s=budget_s,
        answer="",
        choice="",
        empty=False,
        verify_retries=0,
        verify_feedback="",
        session_id=session_id,
        history_text=history_text,
        profile=profile or {},
        needs_clarification=False,
        clarification_options=[],
        steps=[],
        tokens={},
        took_ms=0.0,
    )