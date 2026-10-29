"""Agent 状态定义。

状态尽量扁平、可 JSON 序列化 —— 便于写 trace、做评测复现，
也避免把 RetrievalResult 这类重对象塞进图状态导致 checkpoint 失败。
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


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

    # 检索
    queries: list[str]
    attempts: int
    contexts: list[dict[str, Any]]
    retrieval: dict[str, Any]

    # 工具输出
    tool_outputs: dict[str, Any]
    assessment: dict[str, Any]

    # 生成
    answer: str
    choice: str
    llm_error: str
    empty: bool

    # 校验
    verification: dict[str, Any]

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
) -> AgentState:
    return AgentState(
        question=question,
        options=options or {},
        subject=subject,
        qid=qid,
        mode=mode,
        route="concept",
        queries=[question],
        attempts=0,
        contexts=[],
        tool_outputs={},
        assessment={},
        answer="",
        choice="",
        empty=False,
        steps=[],
        tokens={},
        took_ms=0.0,
    )