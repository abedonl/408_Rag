"""LangGraph 编排与运行入口。

图结构（有界 plan-act-observe）：

    route ──(信息不足)──→ clarify → END（结构化澄清选项，不硬猜）
      │
      └──→ retrieve → assess ──(不足)──→ rewrite ──→ retrieve
                         │
                         └──(足够/走工具)──→ plan ⇄ act ⇄ observe
                                                 │
                                                 └──(完成/预算)──→ answer → verify
                                                                        │
                                                          (校验失败一次) └→ answer → END

上限：最多 3 个工具轮次、总预算 90 秒；`observe` 每次都必须给出终态
或明确的下一步，不存在自由 ReAct 的无限循环。
`assess` 的检索重试上限仍为 2 次，计算/图表/考点题直接走工具路径。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger

from cs408rag.agent.cost import compute_cost_cny
from cs408rag.agent.followup import resolve_followup
from cs408rag.agent.memory import Turn, get_session_store
from cs408rag.agent.nodes import AgentNodes, extract_topic
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
    # 有界循环与可观测性（Week 1）
    rounds: int = 0
    stop_reason: str = ""
    tool_failures: list[dict[str, Any]] = field(default_factory=list)
    plan_history: list[dict[str, Any]] = field(default_factory=list)
    verify_retries: int = 0
    needs_clarification: bool = False
    clarification_options: list[str] = field(default_factory=list)
    # 会话（Week 2）
    session_id: str = ""
    turn_no: int = 0
    resolved_query: str = ""  # 指代消解后的有效问题；与 question 相同则为空
    # 成本（Week 3）
    cost_cny: float = 0.0

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
            "rounds": self.rounds,
            "stop_reason": self.stop_reason,
            "tool_failures": self.tool_failures,
            "plan_history": self.plan_history,
            "verify_retries": self.verify_retries,
            "needs_clarification": self.needs_clarification,
            "clarification_options": self.clarification_options,
            "session_id": self.session_id,
            "turn_no": self.turn_no,
            "resolved_query": self.resolved_query,
            "cost_cny": self.cost_cny,
        }


def _after_assess(state: AgentState) -> str:
    return "rewrite" if (state.get("assessment") or {}).get("should_rewrite") else "plan"


def _after_route(state: AgentState) -> str:
    # 澄清短路：问题缺少必要信息时直接反问，不走检索/工具，不硬猜。
    if state.get("needs_clarification"):
        return "clarify"
    # 话题外：与 408 无关的输入直接给引导回复，不检索不调工具。
    if state.get("route") == "irrelevant":
        return "irrelevant"
    # 中枢路由（规划/画像/打卡）不需要教材检索，直接进 plan-act 循环。
    if state.get("route") == "plan":
        return "plan"
    return "retrieve"


def _after_observe(state: AgentState) -> str:
    return "answer" if state.get("stop_reason") else "plan"


def _after_verify(state: AgentState) -> str:
    # verify 只在失败时写入 feedback；answer 消费后清空，因此这条边最多回一次。
    return "answer" if state.get("verify_feedback") else "end"


def build_graph(nodes: AgentNodes) -> Any:
    from langgraph.graph import END, StateGraph

    g = StateGraph(AgentState)
    g.add_node("route", nodes.route)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("assess", nodes.assess)
    g.add_node("rewrite", nodes.rewrite)
    g.add_node("plan", nodes.plan)
    g.add_node("act", nodes.act)
    g.add_node("observe", nodes.observe)
    g.add_node("answer", nodes.answer)
    g.add_node("verify", nodes.verify)

    g.set_entry_point("route")
    g.add_node("clarify", nodes.clarify)
    g.add_node("irrelevant", nodes.irrelevant)
    g.add_conditional_edges(
        "route",
        _after_route,
        {"clarify": "clarify", "irrelevant": "irrelevant", "retrieve": "retrieve", "plan": "plan"},
    )
    g.add_edge("clarify", END)
    g.add_edge("irrelevant", END)
    g.add_edge("retrieve", "assess")
    g.add_conditional_edges(
        "assess",
        _after_assess,
        {"rewrite": "rewrite", "plan": "plan"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("plan", "act")
    g.add_edge("act", "observe")
    g.add_conditional_edges(
        "observe",
        _after_observe,
        {"plan": "plan", "answer": "answer"},
    )
    g.add_edge("answer", "verify")
    g.add_conditional_edges(
        "verify",
        _after_verify,
        {"answer": "answer", "end": END},
    )
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
        # 会话存储独立于工具实例：tools.reset 每次 run 会清 trace，
        # 历史挂在那里会被顺带清掉。
        self.sessions = get_session_store()
        # 工具实例持有 trace 与 current_qid，图状态之外的这些字段不能并发共享。
        # Agent 本身是 LLM 延迟主导的工作负载，串行化不会显著影响吞吐。
        self._lock = Lock()

    @property
    def graph(self) -> Any:
        return self._graph

    def rehydrate_session(
        self,
        session_id: str,
        history: list[dict],
        *,
        client_id: str | None = None,
    ) -> int:
        """用落库的聊天历史回灌内存会话，返回该会话的总轮数。

        内存会话表进程内 LRU，切回旧会话时它是空的；不回灌的话「这题」
        「那 2021 年呢」这类指代在翻出来的历史里全部失效。

        回灌按 `turn_count` 增量补齐：历史比内存多就追加缺的尾部，
        因此刷新、多标签、或同一会话被别的入口续写后都能对齐；
        历史没变时只做一次长度比较，可安全地每轮调用。
        """
        session = self.sessions.get_or_create(
            session_id, client_id=client_id or ""
        )
        if session.turn_count >= len(history):
            return session.turn_count
        for item in history[session.turn_count:]:
            question = str((item or {}).get("q") or "").strip()
            if not question:
                continue
            d = (item or {}).get("d") or {}
            self.sessions.append_turn(
                session,
                Turn(
                    question=question,
                    answer=str(d.get("answer") or "")[:600],
                    route=str(d.get("route") or ""),
                    qid=_extract_qid(question),
                    topic=extract_topic(question, self.tools),
                ),
            )
        return session.turn_count

    def run(
        self,
        question: str,
        *,
        options: dict[str, str] | None = None,
        subject: str | None = None,
        qid: str | None = None,
        mode: str = "qa",
        session_id: str | None = None,
        profile: dict[str, Any] | None = None,
        client_id: str | None = None,
    ) -> AgentResult:
        t0 = time.perf_counter()

        # -- 会话：取历史 + 指代消解（规则，不调 LLM） --
        session = None
        resolved_query = ""
        effective_q = question
        if session_id:
            session = self.sessions.get_or_create(
                session_id, profile=profile, client_id=client_id or ""
            )
            r = resolve_followup(question, session)
            if r is not None:
                if r.question != question:
                    resolved_query = r.question
                    effective_q = r.question
                qid = qid or r.qid
                if r.note:
                    logger.info(r.note)

        state = initial_state(
            effective_q,
            options=options,
            subject=subject,
            qid=qid,
            mode=mode,
            session_id=session.session_id if session else "",
            history_text=session.render_history() if session else "",
            profile=session.profile if session else None,
        )
        with self._lock:
            self.tools.reset(current_qid=qid)
            # 在锁内绑定用户：写工具（打卡/判分）与画像读取都以它为命名空间，
            # 共享 runner 的并发请求不会串号。
            self.tools.set_client(client_id)
            try:
                out: AgentState = self._graph.invoke(state)
            except Exception as e:  # noqa: BLE001 - Agent 失败也要给出可读结果
                logger.exception(f"Agent 执行失败：{e}")
                return AgentResult(
                    question=question,
                    ok=False,
                    error=f"Agent 执行失败：{str(e)[:300]}",
                    stop_reason="tool_error",
                    took_ms=(time.perf_counter() - t0) * 1000,
                )
            tool_calls = self.tools.trace.to_list()
            tool_failures = self.tools.trace.failure_details()
            # 工具的 data 里 ok=False（如除零）时 ToolResult 仍是 ok=True，
            # 失败由 observe 记到 state.tool_failures；两边合并，避免漏计。
            seen = {
                (f.get("tool"), f.get("round"), f.get("error_type"))
                for f in tool_failures
            }
            for f in out.get("tool_failures") or []:
                key = (f.get("tool"), f.get("round"), f.get("error_type"))
                if key not in seen:
                    tool_failures.append(f)
                    seen.add(key)

        verification = out.get("verification") or {}
        answer = out.get("answer", "")
        ok = bool(answer) and not out.get("llm_error")
        if mode == "quiz" and options:
            ok = ok and out.get("choice", "") in options

        # 图内每个终态都写 stop_reason；异常/短路时兜底，避免空值。
        stop_reason = out.get("stop_reason") or ("answered" if answer else "no_evidence")

        tokens = out.get("tokens") or {}
        cost_cny = compute_cost_cny(
            getattr(self.llm, "model", ""),
            tokens.get("prompt", 0),
            tokens.get("completion", 0),
        )

        # -- 会话回写：记录 Turn，更新 last_qid / last_topic --
        turn_no = 0
        if session is not None:
            topic = out.get("topic") or extract_topic(effective_q, self.tools)
            effective_qid = qid or _extract_qid(effective_q)
            turn = Turn(
                question=question,  # 存用户原始问句，不改写后的
                answer=answer[:600],
                route=out.get("route", ""),
                qid=effective_qid,
                topic=topic,
                tool_names=[t.get("tool", "") for t in tool_calls],
            )
            turn_no = self.sessions.append_turn(session, turn)

        return AgentResult(
            question=question,
            answer=answer,
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
            rounds=int(out.get("tool_rounds", 0)),
            stop_reason=stop_reason,
            tool_failures=tool_failures,
            plan_history=out.get("plan_history") or [],
            verify_retries=int(out.get("verify_retries", 0)),
            needs_clarification=bool(out.get("needs_clarification")),
            clarification_options=out.get("clarification_options") or [],
            session_id=session.session_id if session else "",
            turn_no=turn_no,
            resolved_query=resolved_query,
            cost_cny=cost_cny,
        )


_QID_RE = re.compile(r"\b(20\d{2}-\d{2})\b")


def _extract_qid(text: str) -> str | None:
    """从问题里捞真题编号（如 2009-04），用于会话的 last_qid 追踪。"""
    m = _QID_RE.search(text or "")
    return m.group(1) if m else None


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