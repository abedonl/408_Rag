"""对抗 Plan 的 LangGraph 状态机与运行入口。

流程：plan_gen → rule_guard → llm_critic → (arbitrate) → advance
       advance 决定 converged / max_rounds / stalled / degraded / budget 或继续修订。

防闲聊：角色只能输出 JSON；是否继续由 advance 的条件边决定，Agent 无权要求「再聊聊」。
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from loguru import logger

from cs408rag.adversarial.planner import build_detailed_plan
from cs408rag.adversarial.roles import Critic, Expert, Planner
from cs408rag.adversarial.rules import review_plan
from cs408rag.adversarial.state import PlanState, initial_state
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
LLM_BUDGET = 8
# 结构 Plan 是本地计算；这里主要留给可选的 LLM 文案 + Critic。
# 之前 90s 会把「419 重试 + 文案」正常路径误判成 budget。
TIME_BUDGET_S = 150.0


def _cid(c: dict[str, Any]) -> str:
    """挑战 id：规则 + 位置 + 内容摘要。

    只按位置生成会让同一位置的多条 LLM 挑战共用一个 id，
    Expert 的裁定就会落到错误条目上。
    """
    base = f"{c.get('rule')}@{c.get('locator')}"
    msg = str(c.get("message") or "")
    if not msg:
        return base
    return f"{base}#{hashlib.md5(msg.encode('utf-8')).hexdigest()[:6]}"


def _tok_stats(client: Any | None) -> tuple[int, int]:
    """(prompt, completion) 累计值；假角色没有 llm/stats 时返回 0。"""
    stats = getattr(client, "stats", None)
    if stats is None:
        return (0, 0)
    return (
        int(getattr(stats, "prompt_tokens", 0) or 0),
        int(getattr(stats, "completion_tokens", 0) or 0),
    )


def _tok_delta(before: tuple[int, int], after: tuple[int, int]) -> dict[str, int]:
    return {"prompt": after[0] - before[0], "completion": after[1] - before[1]}


class PlanNodes:
    def __init__(
        self,
        *,
        planner: Planner | None = None,
        critic: Critic | None = None,
        expert: Expert | None = None,
        analyzer: ExamAnalyzer | None = None,
    ) -> None:
        self.analyzer = analyzer or get_analyzer()
        self.planner = planner or Planner(analyzer=self.analyzer)
        self.critic = critic or Critic()
        self.expert = expert or Expert(analyzer=self.analyzer)

    # -- plan_gen ---------------------------------------------------------

    def plan_gen(self, state: PlanState) -> dict[str, Any]:
        t0 = time.perf_counter()
        version = int(state.get("version") or 0) + 1
        use_llm = bool(state.get("use_llm", True))
        prev = state.get("plan") or None
        llm_client = getattr(self.planner, "llm", None)
        guide_client = getattr(self.planner, "guide_llm", None)
        tok_before = _tok_stats(llm_client)
        guide_before = (
            _tok_stats(guide_client) if guide_client is not llm_client else (0, 0)
        )

        if use_llm:
            plan, used_llm = self.planner.generate(
                state["goal"], prev_plan=prev, challenges=state.get("challenges") or []
            )
        else:
            plan = build_detailed_plan(state["goal"], self.analyzer)
            used_llm = False

        meta = plan.get("meta") or {}
        plan_source = meta.get("plan_source") or ("llm_enriched" if used_llm else "deterministic")
        guide_source = meta.get("guide_source") or ("llm" if used_llm else "template")
        tokens = _tok_delta(tok_before, _tok_stats(llm_client))
        if guide_client is not llm_client:
            g = _tok_delta(guide_before, _tok_stats(guide_client))
            tokens = {
                "prompt": tokens["prompt"] + g["prompt"],
                "completion": tokens["completion"] + g["completion"],
            }
        entry = {
            "node": "plan_gen",
            "version": version,
            "used_llm": used_llm,
            "plan_source": plan_source,
            "guide_source": guide_source,
            "tokens": tokens,
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        upd: dict[str, Any] = {
            "plan": plan,
            "plan_history": [plan],
            "version": version,
            "timeline": [entry],
            # 结构由确定性生成器保证；LLM 文案失败不算 degraded
            "planner_fallback": False,
            "plan_source": plan_source,
            "guide_source": guide_source,
        }
        if use_llm and meta.get("guide_attempted"):
            upd["llm_calls"] = int(state.get("llm_calls") or 0) + 1
        return upd

    # -- rule_guard -------------------------------------------------------

    def rule_guard(self, state: PlanState) -> dict[str, Any]:
        t0 = time.perf_counter()
        challenges = []
        for c in review_plan(state["plan"], analyzer=self.analyzer):
            d = c.to_dict()
            d["id"] = _cid(d)
            challenges.append(d)
        entry = {
            "node": "rule_guard",
            "n_challenges": len(challenges),
            "n_high": sum(1 for c in challenges if c["severity"] == "high"),
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return {"challenges": challenges, "timeline": [entry]}

    # -- llm_critic -------------------------------------------------------

    def llm_critic(self, state: PlanState) -> dict[str, Any]:
        if not state.get("use_llm", True):
            return {"timeline": [{"node": "llm_critic", "skipped": True}]}
        t0 = time.perf_counter()
        tok_before = _tok_stats(getattr(self.critic, "llm", None))
        llm_challenges, ok = self.critic.review(
            state["plan"], state.get("challenges") or []
        )
        for c in llm_challenges:
            c["id"] = _cid(c)
            # 规则挑战由 Challenge.to_dict() 自带 status；LLM 挑战没有，
            # 不补的话 advance 里的 open high 判定会永远忽略它们。
            c.setdefault("status", "open")
        combined = list(state.get("challenges") or []) + llm_challenges
        entry = {
            "node": "llm_critic",
            "n_new": len(llm_challenges),
            "ok": ok,
            "tokens": _tok_delta(
                tok_before, _tok_stats(getattr(self.critic, "llm", None))
            ),
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        upd: dict[str, Any] = {
            "challenges": combined,
            "timeline": [entry],
            "llm_calls": int(state.get("llm_calls") or 0) + 1,
        }
        if not ok:
            upd["llm_errors"] = int(state.get("llm_errors") or 0) + 1
        return upd

    # -- arbitrate --------------------------------------------------------

    def route_arbitrate(self, state: PlanState) -> str:
        """只有「规则 high 与 LLM high 同时存在」才需要数据裁定。

        单独一条 LLM high 不构成冲突：没有客观规则与它对立，
        让 Planner 直接处理即可，不必多花一次 Expert 调用。
        """
        challenges = state.get("challenges") or []
        has_rule_high = any(
            c.get("rule") != "llm_critic" and c.get("severity") == "high" for c in challenges
        )
        has_llm_high = any(
            c.get("rule") == "llm_critic" and c.get("severity") == "high" for c in challenges
        )
        return "arbitrate" if (has_rule_high and has_llm_high) else "advance"

    def arbitrate(self, state: PlanState) -> dict[str, Any]:
        t0 = time.perf_counter()
        tok_before = _tok_stats(getattr(self.expert, "llm", None))
        conflicts = [
            c
            for c in state.get("challenges") or []
            if c.get("rule") == "llm_critic" and c.get("severity") == "high"
        ][:2]
        verdicts = self.expert.arbitrate(conflicts) if conflicts else []
        return {
            "verdicts": verdicts,
            "llm_calls": int(state.get("llm_calls") or 0) + (1 if conflicts else 0),
            "timeline": [
                {
                    "node": "arbitrate",
                    "n_conflicts": len(conflicts),
                    "n_verdicts": len(verdicts),
                    "tokens": _tok_delta(
                        tok_before, _tok_stats(getattr(self.expert, "llm", None))
                    ),
                    "took_ms": round((time.perf_counter() - t0) * 1000, 1),
                }
            ],
        }

    # -- advance ----------------------------------------------------------

    def advance(self, state: PlanState) -> dict[str, Any]:
        t0 = time.perf_counter()
        challenges = list(state.get("challenges") or [])
        verdicts = {str(v.get("challenge_id")): v for v in state.get("verdicts") or []}

        for c in challenges:
            v = verdicts.get(str(c.get("id")))
            if not v:
                continue
            action = v.get("action")
            if action == "drop":
                c["status"] = "dropped"
            elif action == "downgrade" and c["severity"] == "high":
                c["severity"] = "medium"
            c["verdict"] = action

        # 跨轮轨迹：本轮不再出现的 open 挑战视为已解决；
        # 已 resolved/dropped 的挑战不允许被重新提出。
        history = [dict(c) for c in state.get("challenge_history") or []]
        by_id = {str(c.get("id")): i for i, c in enumerate(history)}
        closed = {
            cid
            for cid, i in by_id.items()
            if history[i].get("status") in ("resolved", "dropped")
        }
        if closed:
            challenges = [c for c in challenges if str(c.get("id")) not in closed]
        current_ids = {str(c.get("id")) for c in challenges}
        for c in challenges:
            cid = str(c.get("id"))
            if cid in by_id:
                history[by_id[cid]] = {**history[by_id[cid]], **c}
            else:
                by_id[cid] = len(history)
                history.append(dict(c))
        for i, h in enumerate(history):
            cid = str(h.get("id"))
            if cid not in current_ids and h.get("status", "open") == "open":
                history[i] = {**h, "status": "resolved"}

        open_high = [
            c for c in challenges if c["severity"] == "high" and c.get("status", "open") == "open"
        ]
        prev = set(state.get("prev_high") or [])
        cur = {c["id"] for c in open_high}
        new_high = cur - prev
        resolved = prev - cur

        round_no = int(state.get("round") or 0) + 1
        max_rounds = int(state.get("max_rounds") or 0)
        llm_errors = int(state.get("llm_errors") or 0)
        llm_calls = int(state.get("llm_calls") or 0)
        elapsed = time.time() - float(state.get("started_at") or time.time())

        stop = ""
        if state.get("planner_fallback") and round_no == 1:
            # LLM 没真正生成 Plan（配额/空回答），规则兜底虽然无挑战，
            # 也不能标成 converged，否则会掩盖「模型没工作」。
            stop = "degraded"
        elif not open_high:
            stop = "converged"
        elif round_no > max_rounds:
            stop = "max_rounds"
        elif llm_errors >= 2:
            stop = "degraded"
        elif round_no > 1 and not new_high and not resolved:
            stop = "stalled"
        elif llm_calls >= LLM_BUDGET or elapsed > TIME_BUDGET_S:
            stop = "budget"

        entry = {
            "node": "advance",
            "round": round_no,
            "open_high": len(open_high),
            "new_high": len(new_high),
            "resolved": len(resolved),
            "stop_reason": stop,
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        return {
            "challenges": challenges,
            "challenge_history": history,
            "prev_high": sorted(cur),
            "round": round_no,
            "stop_reason": stop,
            "timeline": [entry],
        }

    def should_continue(self, state: PlanState) -> str:
        from langgraph.graph import END

        if state.get("stop_reason"):
            return END
        return "plan_gen"


def build_graph(nodes: PlanNodes) -> Any:
    from langgraph.graph import END, StateGraph

    g = StateGraph(PlanState)
    g.add_node("plan_gen", nodes.plan_gen)
    g.add_node("rule_guard", nodes.rule_guard)
    g.add_node("llm_critic", nodes.llm_critic)
    g.add_node("arbitrate", nodes.arbitrate)
    g.add_node("advance", nodes.advance)

    g.set_entry_point("plan_gen")
    g.add_edge("plan_gen", "rule_guard")
    g.add_edge("rule_guard", "llm_critic")
    g.add_conditional_edges(
        "llm_critic",
        nodes.route_arbitrate,
        {"arbitrate": "arbitrate", "advance": "advance"},
    )
    g.add_edge("arbitrate", "advance")
    g.add_conditional_edges(
        "advance",
        nodes.should_continue,
        {"plan_gen": "plan_gen", END: END},
    )
    return g.compile()


@dataclass
class PlanResult:
    goal: dict[str, Any]
    plan: dict[str, Any] = field(default_factory=dict)
    plan_history: list[dict[str, Any]] = field(default_factory=list)
    challenges: list[dict[str, Any]] = field(default_factory=list)
    challenge_history: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    version: int = 0
    rounds: int = 0
    llm_calls: int = 0
    llm_errors: int = 0
    planner_fallback: bool = False
    plan_source: str = "deterministic"
    guide_source: str = "template"
    role_models: dict[str, str] = field(default_factory=dict)
    tokens: dict[str, int] = field(default_factory=dict)
    error: str = ""
    took_ms: float = 0.0

    @property
    def open_high(self) -> list[dict[str, Any]]:
        return [
            c
            for c in self.challenges
            if c.get("severity") == "high" and c.get("status", "open") == "open"
        ]

    @property
    def llm_challenges(self) -> list[dict[str, Any]]:
        """LLM 主观挑战：辅助信号，未经人工核对。"""
        return [c for c in self.challenges if c.get("rule") == "llm_critic"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "plan": self.plan,
            "plan_history": self.plan_history,
            "challenges": self.challenges,
            "challenge_history": self.challenge_history,
            "timeline": self.timeline,
            "stop_reason": self.stop_reason,
            "open_high": self.open_high,
            "llm_challenges": self.llm_challenges,
            "version": self.version,
            "rounds": self.rounds,
            "llm_calls": self.llm_calls,
            "llm_errors": self.llm_errors,
            "planner_fallback": self.planner_fallback,
            "plan_source": self.plan_source,
            "guide_source": self.guide_source,
            "role_models": self.role_models,
            "tokens": self.tokens,
            "error": self.error,
            "took_ms": round(self.took_ms, 1),
        }


class PlanRunner:
    def __init__(
        self,
        *,
        nodes: PlanNodes | None = None,
        use_llm: bool = True,
        max_rounds: int = 2,
    ) -> None:
        self.nodes = nodes or PlanNodes()
        self.use_llm = use_llm
        # 纯规则模式没有修订能力，只跑一版；否则 max_rounds 决定最多几版
        self.max_rounds = max_rounds if use_llm else 0
        self._graph = build_graph(self.nodes)
        self._lock = Lock()

    # -- 统计 -------------------------------------------------------------

    def _clients(self) -> list[Any]:
        """Planner / Critic / Expert 可能各持一个 LLMClient，token 要合并统计。"""
        seen: set[int] = set()
        out: list[Any] = []
        for role in (self.nodes.planner, self.nodes.critic, self.nodes.expert):
            for attr in ("llm", "guide_llm"):
                client = getattr(role, attr, None)
                if client is None or id(client) in seen:
                    continue
                seen.add(id(client))
                out.append(client)
        return out

    def _stats(self) -> dict[str, int]:
        total = {"calls": 0, "prompt": 0, "completion": 0}
        for client in self._clients():
            stats = getattr(client, "stats", None)
            if stats is None:
                continue
            total["calls"] += int(getattr(stats, "calls", 0) or 0)
            total["prompt"] += int(getattr(stats, "prompt_tokens", 0) or 0)
            total["completion"] += int(getattr(stats, "completion_tokens", 0) or 0)
        return total

    def _role_models(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, role in (
            ("planner", self.nodes.planner),
            ("critic", self.nodes.critic),
            ("expert", self.nodes.expert),
        ):
            client = getattr(role, "llm", None)
            out[name] = str(getattr(client, "model", "") or "")
        return out

    # 供前端在做流式展示时自行计算 token 增量
    def stats_snapshot(self) -> dict[str, int]:
        return self._stats()

    @property
    def role_models(self) -> dict[str, str]:
        return self._role_models()

    # -- 执行 -------------------------------------------------------------

    def _stream(self, state: PlanState) -> Any:
        """从给定状态跑图，逐节点产出完整快照。流式与一次性执行共用同一张图。"""
        state["started_at"] = time.time()
        last: PlanState = state
        with self._lock:
            try:
                # values 模式每次返回累计后的完整状态，不需要手工合并 reducer
                for snapshot in self._graph.stream(state, stream_mode="values"):
                    last = snapshot
                    yield snapshot
            except Exception as e:  # noqa: BLE001
                logger.exception(f"对抗 Plan 执行失败：{e}")
                failed: PlanState = dict(last)  # type: ignore[assignment]
                failed["stop_reason"] = "error"
                failed["error"] = str(e)  # type: ignore[typeddict-unknown-key]
                yield failed

    def iter_run(self, goal: dict[str, Any]) -> Any:
        """全新一轮：从初始状态开始。"""
        state = initial_state(goal, use_llm=self.use_llm, max_rounds=self.max_rounds)
        return self._stream(state)

    def iter_resume(self, prev: dict[str, Any], challenges: list[dict[str, Any]]) -> Any:
        """人工裁定后续修一版：以 prev 快照为起点，把采纳的挑战喂回 plan_gen。

        只多修一版：max_rounds 设为上轮 round，修订后的 advance 把 round 推过线即停。
        prev 只需带 goal/plan/plan_history/version/round/challenge_history——
        其余字段按新一轮初始化。prev_high 缺省为空：本轮 high 全算新增，
        恰好避免「人工修订后立刻被判 stalled」。
        """
        goal = prev.get("goal") or {}
        round_no = int(prev.get("round") or 0)
        state = initial_state(goal, use_llm=self.use_llm, max_rounds=round_no)
        state["plan"] = prev.get("plan") or {}
        state["plan_history"] = list(prev.get("plan_history") or [])
        state["version"] = int(prev.get("version") or 0)
        state["round"] = round_no
        # 采纳的挑战保持 open 喂给 planner；被人工忽略（dropped）的带进来，
        # advance 的 closed 集合会阻止它们被再次提出
        state["challenges"] = [dict(c) for c in challenges]
        state["challenge_history"] = [dict(c) for c in prev.get("challenge_history") or []]
        return self._stream(state)

    def run(self, goal: dict[str, Any]) -> PlanResult:
        t0 = time.perf_counter()
        before = self._stats()
        out: dict[str, Any] = {}
        for snapshot in self.iter_run(goal):
            out = snapshot
        after = self._stats()

        return PlanResult(
            goal=goal,
            plan=out.get("plan") or {},
            plan_history=out.get("plan_history") or [],
            challenges=out.get("challenges") or [],
            challenge_history=out.get("challenge_history") or out.get("challenges") or [],
            timeline=out.get("timeline") or [],
            stop_reason=out.get("stop_reason") or "",
            version=int(out.get("version") or 0),
            rounds=int(out.get("round") or 0),
            llm_calls=int(out.get("llm_calls") or 0),
            llm_errors=int(out.get("llm_errors") or 0),
            planner_fallback=bool(out.get("planner_fallback")),
            plan_source=str(out.get("plan_source") or "deterministic"),
            guide_source=str(out.get("guide_source") or "template"),
            role_models=self._role_models(),
            tokens={
                "prompt": after["prompt"] - before["prompt"],
                "completion": after["completion"] - before["completion"],
            },
            error=str(out.get("error") or ""),
            took_ms=(time.perf_counter() - t0) * 1000,
        )


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    goal = {
        "target_score": 110,
        "weeks": 12,
        "weekly_hours": 10,
        "weak_subjects": ["操作系统"],
    }
    result = PlanRunner(use_llm="--llm" in sys.argv).run(goal)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2)[:3000])