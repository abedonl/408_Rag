"""LangGraph 编排与运行入口。

图结构：

    route → retrieve → assess ──(不足)──→ rewrite ──→ retrieve
                         │
                         └──(足够/走工具)──→ act → answer → verify → END

为什么是这条链而不是自由 ReAct：见 nodes.py 顶部说明。
`assess` 的重试上限为 2 次，且计算/图表/考点题直接走 `act`，
不会被检索不足卡住。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger

from cs408rag.agent.nodes import AgentNodes
from cs408rag.agent.state import AgentState, initial_state
from cs408rag.agent.tools import AgentTools
from cs408rag.generation.llm import LLMClient, get_llm


@dataclass
class AgentResult:
    question: str
    answer: str = ""
    choice: str = ""
    route: str = ""
    route_reason: str = ""
    ok: bool = False
    error: str = ""
    empty: bool = False
    contexts: list[dict[str, Any]] = field(default_factory=list)
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)
    took_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "choice": self.choice,
            "route": self.route,
            "route_reason": self.route_reason,
            "ok": self.ok,
            "error": self.error,
            "empty": self.empty,
            "took_ms": round(self.took_ms, 1),
            "contexts": [
                {
                    "index": i,
                    "chunk_id": c.get("chunk_id"),
                    "citation": c.get("citation"),
                    "subject": c.get("subject"),
                    "source": c.get("source"),
                    "text": c.get("text", "")[:300],
                }
                for i, c in enumerate(self.contexts, 1)
            ],
            "tool_outputs": self.tool_outputs,
            "verification": self.verification,
            "steps": self.steps,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
            "n_steps": len(self.steps),
            "n_tool_calls": len(self.tool_calls),
        }


def build_graph(nodes: AgentNodes) -> Any:
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("route", nodes.route)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("assess", nodes.assess)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("act", nodes.act)
    g.add_node("answer", nodes.answer)
    g.add_node("verify", nodes.verify)

    g.set_entry_point("route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges(
        "assess",
        lambda s: "rewrite" if (s.get("assessment") or {}).get("should_rewrite") else "act",
        {"rewrite": "rewrite", "act": "act"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("act", "answer")
    g.add_edge("answer", "verify")
    g.add_edge("verify", END)
    return g.compile()


class AgentRunner:
    """对外入口。一个实例可复用于多次提问（工具与图都复用）。"""

    def __init__(
        self,
        *,
        tools: AgentTools | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.tools = tools or AgentTools()
        self.llm = llm or get_llm()
        self._nodes = AgentNodes(self.tools, self.llm)
        self._graph = build_graph(self._nodes)
        # 工具实例持有 trace 与 current_qid，图状态之外的这些字段不能并发共享。
        # Agent 本身是 LLM 延迟主导的工作负载，串行化不会显著影响吞吐。
        self._lock = Lock()

    @property
    def graph(self) -> Any:
        return self._graph

    def run(
        self,
        question: str,
        *,
        options: dict[str, str] | None = None,
        subject: str | None = None,
        qid: str | None = None,
        mode: str = "qa",
    ) -> AgentResult:
        t0 = time.perf_counter()
        state = initial_state(
            question, options=options, subject=subject, qid=qid, mode=mode
        )
        with self._lock:
            self.tools.reset(current_qid=qid)
            try:
                out: AgentState = self._graph.invoke(state)
            except Exception as e:  # noqa: BLE001 - Agent 失败也要给出可读结果
                logger.exception(f"Agent 执行失败：{e}")
                return AgentResult(
                    question=question,
                    ok=False,
                    error=f"Agent 执行失败：{str(e)[:300]}",
                    took_ms=(time.perf_counter() - t0) * 1000,
                )
            tool_calls = self.tools.trace.to_list()

        verification = out.get("verification") or {}
        ok = bool(out.get("answer")) and not out.get("llm_error")
        if mode == "quiz" and options:
            ok = ok and out.get("choice", "") in options

        return AgentResult(
            question=question,
            answer=out.get("answer", ""),
            choice=out.get("choice", ""),
            route=out.get("route", ""),
            route_reason=out.get("route_reason", ""),
            ok=ok,
            error=out.get("llm_error", ""),
            empty=bool(out.get("empty")),
            contexts=out.get("contexts") or [],
            tool_outputs=out.get("tool_outputs") or {},
            verification=verification,
            steps=out.get("steps") or [],
            tool_calls=tool_calls,
            tokens=out.get("tokens") or {},
            took_ms=(time.perf_counter() - t0) * 1000,
        )


_runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    runner = AgentRunner()
    q = " ".join(sys.argv[1:]) or "死锁这个考点历年考过几次？和哪些知识点关联？"
    r = runner.run(q, mode="qa")
    print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2)[:4000])