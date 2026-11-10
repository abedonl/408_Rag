"""Agent 各节点的实现。

刻意不用 prebuilt ReAct：本环境的 LLM 长输出约 12 token/s、单次完整问答
50s 以上，ReAct 的自由多轮会把延迟放到不可接受。这里用**规则优先的
有界 plan-act-observe**：每轮只选一个工具，观察结果后再决定下一步，
最多 3 轮、总预算 90 秒；只有规则无法决定时才调一次 LLM planner。

判断「要不要多轮」的原则：
    概念题答案通常在某一节里，多轮只会引入噪声 -> 检索 1 轮就答
    计算/图表/考点题单轮检索按定义答不了 -> 按规则逐个调工具
    planner 的每个决定都写进 plan_history/observations，便于分层评测
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger

from cs408rag.agent.contracts import TOOL_SPECS, ToolSpec
from cs408rag.agent.exam_analysis import RE_CALC_ASK
from cs408rag.agent.state import DEFAULT_BUDGET_S, MAX_TOOL_ROUNDS, AgentState
from cs408rag.agent.tools import AgentTools
from cs408rag.generation.citation import check_citations, detect_predictions, extract_exam_refs
from cs408rag.generation.llm import LLMClient
from cs408rag.generation.prompt import (
    ContextBlock,
    agent_exam_messages,
    agent_qa_messages,
    agent_quiz_messages,
    parse_choice,
)
from cs408rag.retrieval.pipeline import extract_keywords

EXAM_HINTS = (
    "考过", "考查", "考了", "真题", "易考", "重点", "怎么考", "出现过",
    "频次", "考点分析", "关联", "和什么有关", "出过", "历年",
    "先学", "前置", "学习路径", "学习顺序", "一起考", "高频考点",
)
# 中枢意图：规划 / 画像 / 打卡 / 错题。要在 EXAM_HINTS 之前判断——
# 「帮我规划一下复习」里可能带「学习」等词，但意图是生成计划不是查考情。
PLAN_HINTS = (
    "规划", "学习计划", "复习计划", "备考计划", "计划表", "打卡",
    "我的画像", "画像", "错题本", "我的进度", "备考安排", "怎么安排复习",
)
FIGURE_HINTS = ("下图", "如图", "图中", "图所示", "所示图", "配图", "图示")

# 408 四科：用于「话题外」兜底判断——既抽不到考点、也不含科目名/真题编号
# 的输入直接礼貌拒绝，不做无意义检索（否则「今天吃什么」也会翻教材）。
CS408_SUBJECTS = ("数据结构", "组成原理", "操作系统", "计算机网络")
_QID_PATTERN = re.compile(r"\b20\d{2}-\d{2}\b")

# 考点问句的意图。规则 planner 按这些意图排工具顺序，避免「一次全调」。
FREQ_HINTS = ("考过", "考查", "考了", "几次", "频次", "历年", "出现过", "出过", "考情", "高频")
RELATED_HINTS = ("关联", "一起考", "和哪些", "关系", "联系", "共现")
QUESTION_HINTS = ("真题", "原题", "哪几道", "哪些题", "题目", "例子", "举例", "刷题")
LEARNING_HINTS = ("先学", "前置", "学习路径", "学习顺序", "基础", "入门")

# 工具名 -> tool_outputs 的键。answer 的 prompt 依赖这些固定键。
OUTPUT_KEYS = {
    "search_knowledge_graph": "knowledge_graph",
    "read_figure": "figure",
    "calculate": "calculate",
    "get_exam_stats": "exam_stats",
    "get_kp_questions": "kp_questions",
    "get_related_kps": "related",
    "get_learning_path": "learning_path",
    "verify_answer": "verification",
    "search_knowledge_base": "retrieval",
    "generate_plan": "plan_result",
    "read_profile": "profile",
    "record_checkin": "checkin",
    "judge_quiz": "judge",
}

MAX_PLANNER_LLM_CALLS = 1


def _step(node: str, took_ms: float, **detail: Any) -> dict[str, Any]:
    return {"node": node, "took_ms": round(took_ms, 1), "detail": detail}


def _result_status(result: dict[str, Any]) -> tuple[bool, str, str]:
    """统一「工具调用成功」与「工具返回 ok=False」两种失败形态。"""
    ok = bool(result.get("ok"))
    data = result.get("data")
    if ok and isinstance(data, dict) and data.get("ok") is False:
        return False, str(data.get("error_type") or "validation"), str(data.get("error") or "")
    if not ok:
        return False, str(result.get("error_type") or "internal"), str(result.get("error") or "")
    return True, "", ""


def _snippet(data: Any, limit: int = 600) -> str:
    try:
        text = json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        text = repr(data)
    return text[:limit]


def _question_limit(question: str) -> int:
    """从「给 2 道相关真题」里取条数；没写就默认 5，上限 10。"""
    m = re.search(r"(\d+)\s*道", question)
    if not m:
        return 5
    return max(1, min(int(m.group(1)), 10))


def _spec_brief(spec: ToolSpec) -> str:
    fields = spec.args_model.model_fields
    args = ", ".join(
        f"{name}{'' if f.is_required() else '?'}" for name, f in fields.items()
    )
    return f"{spec.desc} args: {args}"


def _parse_json_object(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def extract_topic(question: str, tools: AgentTools) -> str | None:
    """从自然语言里定位考点。

    优先用题库自带的 226 个标签做子串匹配（最长优先）。
    若命中的不是完整标签而是一个**主题词**（如「死锁」只出现在四个
    「死锁xxx」标签里），返回主题词本身，让 ExamAnalyzer 汇总其下全部标签
    —— 直接取频次最高的单个标签会让「死锁考几次」只答 3 次而不是 6 次。
    """
    analyzer = tools.analyzer
    hits = [t for t in analyzer.by_tag if t in question]
    if hits:
        return max(hits, key=len)
    for kw in extract_keywords(question, max_terms=6):
        if analyzer.resolve_tags(kw):
            return kw
    return None


class AgentNodes:
    def __init__(self, tools: AgentTools, llm: LLMClient | None = None) -> None:
        self.tools = tools
        self.llm = llm

    # -- route -----------------------------------------------------------

    def route(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("question", "")
        qid = state.get("qid")

        route, reason = "concept", "默认按概念题处理"
        if any(h in q for h in PLAN_HINTS):
            route, reason = "plan", "意图是规划/画像/打卡，走中枢工具"
        elif any(h in q for h in EXAM_HINTS):
            route, reason = "exam", "问题在问考查情况/关联，走考点分析"
        else:
            qtype = self.tools.analyzer.type_of.get(qid or "", "") if qid else ""
            if qtype == "figure":
                route, reason = "figure", "真题标注为含图题，需要图形解析"
            elif qtype == "calc":
                route, reason = "calc", "真题标注为计算题，需要计算器"
            elif qtype == "cross_subject":
                route, reason = "cross", "跨科目题"
            elif any(h in q for h in FIGURE_HINTS):
                route, reason = "figure", "问句提到图，尝试图形解析"
            elif RE_CALC_ASK.search(q):
                route, reason = "calc", "问句是求值型，尝试计算器"
            elif not qid and self._is_off_topic(q):
                route, reason = "irrelevant", "问题不在 408 范围内，不检索不调工具"

        return {
            "route": route,
            "route_reason": reason,
            "attempts": 0,
            **self._clarify_check(state, route),
            "steps": [
                _step(
                    "route",
                    (time.perf_counter() - t0) * 1000,
                    route=route,
                    reason=reason,
                )
            ],
        }

    def _clarify_check(self, state: AgentState, route: str) -> dict[str, Any]:
        """HITL 澄清检测（Week 3）。宁可多问一句，不硬猜。

        两类触发：
        1. 指代题但没上下文：「这道题选什么」无 qid、无选项、会话也没消解出 qid
        2. 考点类问题但定位不到考点：问考情/关联却抽不出 tag
        """
        q = state.get("question", "")
        qid = state.get("qid")
        options = state.get("options") or {}

        refers_question = any(p in q for p in ("这题", "这道题", "选什么", "下图", "如图"))
        if refers_question and not qid and not any(options.values()):
            return {
                "needs_clarification": True,
                "clarification_options": [
                    "告诉我真题编号（如 2009-04），我可以解析配图并作答",
                    "直接把 A/B/C/D 四个选项的文字发给我",
                ],
            }

        asks_kp = route == "exam" or "考点" in q or "知识点" in q
        if asks_kp and not qid and extract_topic(q, self.tools) is None:
            top = sorted(
                self.tools.analyzer.by_tag, key=lambda t: -len(self.tools.analyzer.by_tag[t])
            )[:5]
            return {
                "needs_clarification": True,
                "clarification_options": [
                    f"我想问的是「{t}」" for t in top
                ] + ["我换个说法，把考点名称写全"],
            }

        return {}

    # -- clarify ---------------------------------------------------------

    def clarify(self, state: AgentState) -> dict[str, Any]:
        """澄清终态：不检索不调 LLM，直接把问题抛回给用户。"""
        t0 = time.perf_counter()
        opts = state.get("clarification_options") or []
        lines = ["这个问题我需要先确认一下："] + [f"{i}. {o}" for i, o in enumerate(opts, 1)]
        return {
            "answer": "\n".join(lines),
            "stop_reason": "clarification",
            "steps": [
                _step(
                    "clarify",
                    (time.perf_counter() - t0) * 1000,
                    n_options=len(opts),
                )
            ],
        }

    # -- irrelevant ------------------------------------------------------

    def _is_off_topic(self, q: str) -> bool:
        """是否与 408 无关：既抽不到考点，也不含科目名/真题编号。

        放在 route 里最后判断，避免误伤概念题（「什么是死锁」里的「死锁」
        会被 extract_topic 命中）。
        """
        if extract_topic(q, self.tools) is not None:
            return False
        if any(s in q for s in CS408_SUBJECTS):
            return False
        if _QID_PATTERN.search(q):
            return False
        return True

    def irrelevant(self, state: AgentState) -> dict[str, Any]:
        """话题外终态：不检索、不调 LLM、不选工具，直接引导回 408。

        这样「今天吃什么」不会翻教材，answer 里也没有考点词，
        会话层的真题推荐自然为空。
        """
        t0 = time.perf_counter()
        return {
            "answer": (
                "我是专注 408 考研（数据结构 / 组成原理 / 操作系统 / 计算机网络）"
                "的备考助手，这个话题不在我的能力范围内。\n\n"
                "你可以问我：考点考情、真题解析、知识点关联、学习路径、"
                "备考规划、打卡与判分等。"
            ),
            "stop_reason": "irrelevant",
            "steps": [_step("irrelevant", (time.perf_counter() - t0) * 1000)],
        }

    # -- retrieve --------------------------------------------------------

    def retrieve(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        # 按题型自适应选择检索路：实测 BM25 对概念/图表题 +1~4pp，
        # 但对计算题略负，所以计算题关闭 BM25；图谱路对所有题型负贡献，默认关。
        from dataclasses import replace

        cfg = self.tools.config
        if state.get("route") == "calc":
            cfg = replace(cfg, use_bm25=False)
        self.tools.config = cfg

        merged: dict[str, dict[str, Any]] = {}
        routes: dict[str, int] = {}
        took_ms = 0.0
        errors: list[str] = []

        for query in state.get("queries", [])[:3]:
            r = self.tools.call(
                "search_knowledge_base",
                query=query,
                subject=state.get("subject"),
                top_k=8,
            )
            if not r.ok:
                errors.append(r.error)
                continue
            took_ms += r.took_ms
            routes = r.data.get("routes") or routes
            for h in r.data.get("hits") or []:
                prev = merged.get(h["chunk_id"])
                if prev is None or (h.get("score") or 0) > (prev.get("score") or 0):
                    merged[h["chunk_id"]] = h

        ordered = sorted(merged.values(), key=lambda h: -(h.get("score") or 0))[:10]
        attempts = state.get("attempts", 0) + 1
        return {
            "contexts": ordered,
            "attempts": attempts,
            "retrieval": {
                "n": len(ordered),
                "routes": routes,
                "queries": state.get("queries", []),
                "errors": errors,
            },
            "steps": [
                _step(
                    "retrieve",
                    (time.perf_counter() - t0) * 1000,
                    attempt=attempts,
                    n_hits=len(ordered),
                    queries=state.get("queries", []),
                )
            ],
        }

    # -- assess ----------------------------------------------------------

    def assess(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        contexts = state.get("contexts") or []
        q = state.get("question", "")
        attempts = state.get("attempts", 1)

        kws = extract_keywords(q, max_terms=6)
        top_text = " ".join(c.get("text", "") for c in contexts[:3])
        covered = sum(1 for k in kws if k in top_text)
        sufficient = len(contexts) >= 3 and (covered >= 1 or len(contexts) >= 6)

        # 计算/图表/考点题的主路径是工具，即使检索弱也继续
        tool_route = state.get("route") in ("calc", "figure", "exam")
        should_rewrite = (not sufficient) and attempts < 2 and not tool_route

        assessment = {
            "sufficient": sufficient,
            "covered_keywords": covered,
            "n_contexts": len(contexts),
            "should_rewrite": should_rewrite,
            "reason": (
                "检索充分"
                if sufficient
                else "检索不足"
                + ("，改写后重试" if should_rewrite else "，但已达重试上限或走工具路径")
            ),
        }
        return {
            "assessment": assessment,
            "steps": [_step("assess", (time.perf_counter() - t0) * 1000, **assessment)],
        }

    # -- rewrite ---------------------------------------------------------

    def rewrite(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("question", "")
        kws = extract_keywords(q, max_terms=6)
        candidates = [q]
        if kws:
            candidates.append(" ".join(kws[:4]) + " 定义 原理")
        if state.get("route") == "calc" and kws:
            candidates.append(" ".join(kws[:4]) + " 计算方法 公式")
        if state.get("route") == "exam":
            topic = self._topic(state)
            if topic:
                candidates.append(f"{topic} 定义 原理")

        uniq: list[str] = []
        for c in candidates:
            if c and c not in uniq:
                uniq.append(c)
        return {
            "queries": uniq[:3],
            "steps": [_step("rewrite", (time.perf_counter() - t0) * 1000, queries=uniq[:3])],
        }

    # -- plan ------------------------------------------------------------

    def plan(self, state: AgentState) -> dict[str, Any]:
        """选下一个动作：规则优先，规则无法决定时最多调一次 LLM planner。"""
        t0 = time.perf_counter()
        round_no = state.get("tool_rounds", 0) + 1
        started = state.get("started_at", time.monotonic())
        budget_s = state.get("budget_s", DEFAULT_BUDGET_S)

        if time.monotonic() - started > budget_s:
            return {
                "plan": {},
                "last_result": {},
                "stop_reason": "budget",
                "steps": [
                    _step("plan", (time.perf_counter() - t0) * 1000, stop="budget")
                ],
            }
        if round_no > MAX_TOOL_ROUNDS:
            return {
                "plan": {},
                "last_result": {},
                "stop_reason": "max_rounds",
                "steps": [
                    _step("plan", (time.perf_counter() - t0) * 1000, stop="max_rounds")
                ],
            }

        action = self._plan_calc(state)
        source = "rule"
        if action is None:
            action = self._rule_plan(state)
        planner_calls = state.get("planner_calls", 0)
        tok: dict[str, int] = {}
        if action is None:
            action, called, tok = self._llm_plan(state)
            source = "llm"
            if called:
                planner_calls += 1

        # planner 的 token 也计入成本追踪
        tokens = dict(state.get("tokens") or {})
        if tok:
            tokens["prompt"] = tokens.get("prompt", 0) + tok.get("prompt", 0)
            tokens["completion"] = tokens.get("completion", 0) + tok.get("completion", 0)

        update: dict[str, Any] = {
            "plan": action or {},
            "planner_calls": planner_calls,
            "tokens": tokens,
            "steps": [
                _step(
                    "plan",
                    (time.perf_counter() - t0) * 1000,
                    round=round_no,
                    source=source,
                    action=(action or {}).get("action"),
                    reason=(action or {}).get("reason", ""),
                )
            ],
        }
        if not action:
            return update

        action["round"] = round_no
        update["plan_history"] = [{**action, "source": source, "round": round_no}]
        kp_tools = {
            "get_exam_stats",
            "get_kp_questions",
            "get_related_kps",
            "get_learning_path",
            "search_knowledge_graph",
        }
        if action["action"] in kp_tools:
            update["topic"] = action.get("args", {}).get("kp_name")
        return update

    def _plan_calc(self, state: AgentState) -> dict[str, Any] | None:
        """计算题：让 LLM 只产出一个算式，再由规则决定调用计算器。

        单独放在这里而不是 `_rule_plan` 里，是因为 `observe` 也会调用
        `_rule_plan` 预判下一步；如果里面有 LLM 调用，观察阶段会偷偷多花一次。
        """
        if state.get("route") != "calc":
            return None
        done = {o.get("tool") for o in (state.get("observations") or [])}
        if "calculate" in done:
            return None
        expr = self._propose_expression(
            state.get("question", ""), state.get("options") or {}
        )
        if not expr:
            return None
        return {
            "action": "calculate",
            "args": {"expr": expr["expr"]},
            "reason": expr.get("why") or "把题目翻译成算式后确定性求值",
        }

    def _rule_plan(self, state: AgentState) -> dict[str, Any] | None:
        """纯规则规划：不调 LLM，observe 可以安全地用它预判下一步。"""
        route = state.get("route", "concept")
        q = state.get("question", "")
        done = {o.get("tool") for o in (state.get("observations") or [])}

        if route == "figure":
            qid = state.get("qid")
            if qid and "read_figure" not in done:
                return {
                    "action": "read_figure",
                    "args": {"question_id": qid, "question": q},
                    "reason": "真题标注为含图题，先做图形解析",
                }
            return None

        if route in ("concept", "cross"):
            if "search_knowledge_graph" not in done:
                kws = extract_keywords(q, max_terms=3)
                if kws:
                    return {
                        "action": "search_knowledge_graph",
                        "args": {"kp_name": kws[0]},
                        "reason": "用图谱定位章节，补充引用路径",
                    }
            return None

        if route == "plan":
            # 中枢路由：先读画像（规划要吃它），再按需生成规划。
            # 打卡/判分这类写操作不进规则 planner——参数（周次/选项字母）从
            # 自然语言里解析不可靠，交给 LLM planner 按契约给参数，或走 UI。
            if "read_profile" not in done:
                return {
                    "action": "read_profile",
                    "args": {},
                    "reason": "先读画像：规划与建议要基于你的提问/作答/打卡记录",
                }
            wants_plan = any(h in q for h in ("规划", "计划", "安排"))
            if wants_plan and "generate_plan" not in done:
                # 轻量抽取明显参数：「每天 6 小时」「目标 120 分」。
                # 抽不到就缺省——工具内部会从画像推断薄弱科目与加权考点。
                args: dict[str, Any] = {}
                m = re.search(r"每天\s*(\d{1,2})\s*(?:小时|h|H)", q)
                if m:
                    args["daily_hours"] = int(m.group(1))
                m = re.search(r"目标\s*(\d{2,3})\s*分?", q)
                if m:
                    args["target_score"] = int(m.group(1))
                weak = [
                    s
                    for s in ("数据结构", "组成原理", "操作系统", "计算机网络")
                    if (f"{s}弱" in q) or (f"弱{s}" in q) or ("薄弱" in q and s in q)
                ]
                if weak:
                    args["weak_subjects"] = weak
                return {
                    "action": "generate_plan",
                    "args": args,
                    "reason": "用画像加权生成个性化规划（缺省参数自动从画像推断）",
                }
            return None

        if route == "exam":
            topic = self._topic(state)
            if not topic:
                return None
            for action in self._exam_tool_order(q):
                if action in done:
                    continue
                args: dict[str, Any] = {"kp_name": topic}
                if action == "get_kp_questions":
                    args["limit"] = _question_limit(q)
                return {
                    "action": action,
                    "args": args,
                    "reason": f"考点「{topic}」的备考问句，按意图依次取证",
                }
        return None

    @staticmethod
    def _exam_tool_order(question: str) -> list[str]:
        """按问句意图排工具，而不是 9 个工具一次全调。

        例：
            「死锁考几次？先学什么？」 -> stats, learning_path
            「死锁和哪些考点一起考？给 2 道相关真题」 -> related, kp_questions
        没有任何意图词时退回 stats/related/kp_questions 的最小完整组合。
        """
        freq = any(h in question for h in FREQ_HINTS)
        related = any(h in question for h in RELATED_HINTS)
        questions = any(h in question for h in QUESTION_HINTS)
        learning = any(h in question for h in LEARNING_HINTS)
        if not any((freq, related, questions, learning)):
            freq = related = questions = True

        order: list[str] = []
        if freq:
            order.append("get_exam_stats")
        if learning:
            order.append("get_learning_path")
        if related:
            order.append("get_related_kps")
        if questions:
            order.append("get_kp_questions")
        return order

    def _llm_plan(
        self, state: AgentState
    ) -> tuple[dict[str, Any] | None, bool, dict[str, int]]:
        """规则无法决定时的一次 LLM 规划。返回 (plan, 是否真的调了 LLM, token 用量)。"""
        if state.get("planner_calls", 0) >= MAX_PLANNER_LLM_CALLS:
            return None, False, {}
        if self.llm is None or not self.llm.available:
            return None, False, {}
        # 概念/跨科目题如果检索和工具都没有任何产出，再调一次 planner 只会
        # 花钱让模型猜；直接交给 answer 走「无资料」分支。
        if (
            state.get("route") in ("concept", "cross")
            and not state.get("contexts")
            and not state.get("observations")
        ):
            return None, False, {}

        catalog = "\n".join(
            f"- {name}: {_spec_brief(spec)}"
            for name, spec in TOOL_SPECS.items()
            if name != "verify_answer"
        )
        contexts = state.get("contexts") or []
        sources = "、".join(
            str(c.get("citation") or "")[:40] for c in contexts[:3]
        ) or "（无）"
        obs = state.get("observations") or []
        obs_text = "\n".join(
            f"- {o.get('tool')}: ok={o.get('ok')} {o.get('summary', '')[:80]}"
            for o in obs[-3:]
        ) or "（尚无工具结果）"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 408 备考 Agent 的规划器。只能从给定工具中选一个，"
                    "输出 JSON：{\"action\": 工具名, \"args\": {...}, \"reason\": \"一句话\"}。"
                    "不要输出 JSON 以外的内容。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"【问题】{state.get('question', '')}\n"
                    f"【路由】{state.get('route', '')}\n"
                    f"【已有资料】{sources}\n"
                    f"【已执行工具】\n{obs_text}\n\n"
                    f"【可用工具】\n{catalog}\n\n"
                    "选一个最能推进回答的工具；若已有足够信息，输出 "
                    '{"action": null, "args": {}, "reason": "信息已足够"}。'
                ),
            },
        ]
        try:
            resp = self.llm.chat(messages, max_tokens=400)
        except Exception as e:  # noqa: BLE001 - planner 失败不阻断主流程
            logger.warning(f"LLM planner 调用失败：{e}")
            return None, True, {}
        if not resp.ok:
            return None, True, {}

        tok = {"prompt": resp.prompt_tokens, "completion": resp.completion_tokens}
        obj = _parse_json_object(resp.text)
        if not obj or not obj.get("action"):
            return None, True, tok
        name = str(obj["action"])
        spec: ToolSpec | None = TOOL_SPECS.get(name)
        if spec is None or name == "verify_answer":
            logger.warning(f"LLM planner 选择了白名单外/不可用工具：{name}")
            return None, True, tok
        try:
            args = spec.args_model.model_validate(obj.get("args") or {}).model_dump()
        except Exception as e:  # noqa: BLE001 - 参数不合契约就放弃这次规划
            logger.warning(f"LLM planner 参数不合法：{e}")
            return None, True, tok
        return {
            "action": name,
            "args": args,
            "reason": str(obj.get("reason") or "LLM planner 选择")[:200],
        }, True, tok

    # -- act -------------------------------------------------------------

    def act(self, state: AgentState) -> dict[str, Any]:
        """执行 plan 选中的**一个**工具。失败也继续，由 observe 决定下一步。"""
        t0 = time.perf_counter()
        plan = state.get("plan") or {}
        action = plan.get("action")
        if not action:
            return {
                "last_result": {},
                "steps": [
                    _step("act", (time.perf_counter() - t0) * 1000, skipped=True)
                ],
            }

        round_no = int(plan.get("round") or state.get("tool_rounds", 0) + 1)
        self.tools.guard.begin_round(round_no)
        result = self.tools.call(
            action, round_no=round_no, **dict(plan.get("args") or {})
        )

        outputs = dict(state.get("tool_outputs") or {})
        key = OUTPUT_KEYS.get(action, action)
        if result.ok:
            outputs[key] = result.data
        else:
            outputs.setdefault(
                key,
                {
                    "ok": False,
                    "error": result.error,
                    "error_type": result.error_type,
                },
            )

        return {
            "tool_outputs": outputs,
            "last_result": result.to_dict(),
            "steps": [
                _step(
                    "act",
                    (time.perf_counter() - t0) * 1000,
                    round=round_no,
                    tool=action,
                    ok=result.ok,
                    error_type=result.error_type,
                    latency_ms=round(result.took_ms, 1),
                )
            ],
        }

    # -- observe ---------------------------------------------------------

    def observe(self, state: AgentState) -> dict[str, Any]:
        """压缩工具结果，判断是否还有下一步，并给出终态。"""
        t0 = time.perf_counter()
        result = state.get("last_result") or {}
        previous = state.get("observations") or []
        contexts = state.get("contexts") or []

        if not result:
            # plan 已给出 budget/max_rounds 时不要覆盖终态。
            terminal = state.get("stop_reason") or (
                "no_evidence" if not contexts else "answered"
            )
            return {
                "stop_reason": terminal,
                "tool_rounds": state.get("tool_rounds", 0),
                "steps": [
                    _step(
                        "observe", (time.perf_counter() - t0) * 1000,
                        skipped=True, stop=terminal,
                    )
                ],
            }

        ok, error_type, error = _result_status(result)
        obs = {
            "tool": result.get("tool", ""),
            "ok": ok,
            "error_type": error_type,
            "error": error[:200],
            "summary": (result.get("summary") or "")[:300],
            "keys": list((result.get("data") or {}).keys())[:12]
            if isinstance(result.get("data"), dict)
            else [],
            "snippet": _snippet(result.get("data")),
            "round": result.get("round", 0),
            "took_ms": result.get("took_ms", 0),
        }
        observations = [*previous, obs]
        rounds = state.get("tool_rounds", 0) + 1

        update: dict[str, Any] = {
            "observations": [obs],
            "tool_rounds": rounds,
        }
        if not ok:
            update["tool_failures"] = [
                {
                    "tool": obs["tool"],
                    "error_type": error_type,
                    "error": error,
                    "round": obs["round"],
                }
            ]

        # 用规则 planner 预判下一步：还有就继续，没有就落到终态。
        sim: AgentState = {
            **state,
            "tool_rounds": rounds,
            "observations": observations,
        }
        next_plan = self._rule_plan(sim)
        started = state.get("started_at", time.monotonic())
        budget_s = state.get("budget_s", DEFAULT_BUDGET_S)

        terminal = ""
        if time.monotonic() - started > budget_s:
            terminal = "budget"
        elif next_plan is not None:
            if rounds >= MAX_TOOL_ROUNDS:
                terminal = "max_rounds"
        elif contexts or any(o.get("ok") for o in observations):
            terminal = "answered"
        elif any(not o.get("ok") for o in observations):
            terminal = "tool_error"
        else:
            terminal = "no_evidence"

        if terminal:
            update["stop_reason"] = terminal
        update["steps"] = [
            _step(
                "observe",
                (time.perf_counter() - t0) * 1000,
                tool=obs["tool"],
                ok=ok,
                error_type=error_type,
                rounds=rounds,
                next=(next_plan or {}).get("action"),
                stop=terminal or None,
            )
        ]
        return update

    # -- answer ----------------------------------------------------------

    def answer(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = state.get("question", "")
        options = state.get("options") or {}
        blocks = self._blocks(state)
        evidence = state.get("tool_outputs") or {}
        mode = state.get("mode", "qa")

        if not blocks and not evidence:
            return {
                "answer": "",
                "llm_error": "检索不到任何相关资料，且没有可用的工具结论",
                "verify_feedback": "",
                "stop_reason": "no_evidence",
                "steps": [_step("answer", (time.perf_counter() - t0) * 1000, skipped=True)],
            }

        if state.get("route") == "plan":
            # 中枢路由：规划/画像/打卡/判分的结果由确定性模板成文，
            # 数字全部来自工具输出，LLM 不参与 —— 无 LLM key 也能用。
            text = self._plan_answer(evidence)
            if text is not None:
                return {
                    "answer": text,
                    "llm_error": "",
                    "verify_feedback": "",
                    "steps": [
                        _step(
                            "answer",
                            (time.perf_counter() - t0) * 1000,
                            deterministic=True,
                            evidence_keys=sorted(evidence.keys()),
                        )
                    ],
                }

        if self.llm is None or not self.llm.available:
            return {
                "answer": "",
                "llm_error": "未配置 LLM API key，生成层跳过（检索与工具结果仍可用）",
                "verify_feedback": "",
                "stop_reason": "tool_error",
                "steps": [_step("answer", (time.perf_counter() - t0) * 1000, skipped=True)],
            }

        history = state.get("history_text", "")
        if mode == "quiz":
            # 部分真题的选项画在图中（options 为空），此时仍走 quiz 口径：
            # 依据图形解析判断 A/B/C/D，并从回答里解析字母。
            messages = agent_quiz_messages(q, options, blocks, evidence, history=history)
            max_tokens = 3000
        elif state.get("route") == "exam":
            stats = (evidence.get("exam_stats") or {})
            qs = (evidence.get("kp_questions") or {}).get("questions") or []
            related = (evidence.get("related") or {}).get("cooccurring_tags") or []
            messages = agent_exam_messages(
                stats.get("tag") or q,
                stats,
                qs,
                related,
                evidence.get("learning_path"),
                history=history,
            )
            max_tokens = 2000
        else:
            messages = agent_qa_messages(q, blocks, evidence, history=history)
            max_tokens = 2000

        # verify 失败后的一次重写：把具体问题带回给模型，而不是原样再问一遍。
        feedback = state.get("verify_feedback", "")
        if feedback:
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        f"上一版回答未通过自检：{feedback}。\n"
                        "请重写完整回答，只引用资料中真实存在的真题编号，"
                        "不要出现任何考试预测或押题表述。"
                        + ("第一行仍必须写成 `答案：X`。" if mode == "quiz" else "")
                    ),
                },
            ]

        resp = self.llm.chat(messages, max_tokens=max_tokens)
        # 长上下文 + 思考型模型可能返回空正文；缩短上下文再试一次。
        # 这不是「答错」，评测里必须单独统计 empty。
        if resp.ok and not resp.text.strip() and mode == "quiz":
            resp = self.llm.chat(
                agent_quiz_messages(q, options, blocks[:4], evidence, history=history),
                max_tokens=3200,
            )
        choice = parse_choice(resp.text) if mode == "quiz" else ""
        empty = not (resp.text or "").strip()
        prior = state.get("tokens") or {}
        update: dict[str, Any] = {
            "answer": resp.text,
            "choice": choice,
            "llm_error": (
                "模型返回空正文"
                if resp.ok and empty
                else ("" if resp.ok else resp.error)
            ),
            "empty": empty,
            "tokens": {
                "prompt": prior.get("prompt", 0) + resp.prompt_tokens,
                "completion": prior.get("completion", 0) + resp.completion_tokens,
            },
            "verify_feedback": "",
            "steps": [
                _step(
                    "answer",
                    (time.perf_counter() - t0) * 1000,
                    ok=resp.ok,
                    model=resp.model,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                    rewrite=bool(feedback),
                )
            ],
        }
        # 有资料/工具结论但生成失败时，stop_reason 不能继续声称 answered。
        if empty:
            update["stop_reason"] = "no_evidence"
        return update

    # -- verify ----------------------------------------------------------

    def verify(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        answer = state.get("answer", "")
        blocks = self._blocks(state)
        rep = check_citations(
            answer, blocks, valid_exam_refs=self.tools.valid_exam_refs
        ).to_dict()
        predictions = detect_predictions(answer)
        refs = extract_exam_refs(answer)
        invalid_refs = (
            [r for r in refs if r not in self.tools.valid_exam_refs]
            if self.tools.valid_exam_refs
            else []
        )

        result = {
            **rep,
            "predictions": predictions,
            "exam_refs": refs,
            "invalid_exam_refs": invalid_refs,
            "ok": not predictions and not invalid_refs,
        }

        choice_ok = True
        if state.get("mode") == "quiz" and state.get("options"):
            choice_ok = state.get("choice", "") in state["options"]
            result["choice_valid"] = choice_ok

        answer_out = answer
        if predictions:
            answer_out += (
                "\n\n⚠️ 本回答包含对考试的预测性表述，已被系统拦截。"
                "本系统只提供基于历年真题的**实际考查情况**，不做任何考试预测。"
            )

        retries = state.get("verify_retries", 0)
        should_retry = (
            not result["ok"]
            and retries < 1
            and bool(answer.strip())
            and not state.get("llm_error")
        )

        update: dict[str, Any] = {
            "answer": answer_out,
            "verification": result,
            "steps": [
                _step(
                    "verify",
                    (time.perf_counter() - t0) * 1000,
                    ok=result["ok"],
                    invalid_exam_refs=invalid_refs,
                    predictions=predictions,
                    choice_valid=choice_ok,
                    retry=should_retry,
                )
            ],
        }
        if should_retry:
            problems: list[str] = []
            if invalid_refs:
                problems.append(f"引用了不存在的真题编号：{'、'.join(invalid_refs[:5])}")
            if predictions:
                problems.append(f"出现押题表述：{'、'.join(predictions[:5])}")
            update["verify_retries"] = retries + 1
            update["verify_feedback"] = "；".join(problems) or "自检未通过"
            # 只重写一次；第二次仍失败就带着告警结束。
            update["stop_reason"] = "verify_retry"
        elif retries:
            # 重写后的最终结论必须覆盖上一轮留下的临时标记：
            # 通过就回到 answered，仍失败保留 verify_retry（重试了但没通过）。
            # 空正文/生成失败的终态由 answer 节点写 no_evidence，这里不覆盖。
            if not result["ok"]:
                update["stop_reason"] = "verify_retry"
            elif answer.strip():
                update["stop_reason"] = "answered"
        return update

    # -- 中枢路由成文 ------------------------------------------------------

    @staticmethod
    def _plan_answer(evidence: dict[str, Any]) -> str | None:
        """plan 路由的确定性成文：数字全部来自工具输出，LLM 不参与。

        返回 None 表示没有任何中枢工具产出，回退到常规 LLM 路径。
        """
        parts: list[str] = []

        plan = evidence.get("plan_result") or {}
        if plan.get("weeks"):
            lines = [
                f"## 📋 {plan['weeks']} 周备考规划"
                f"（{plan.get('start_date')} → {plan.get('exam_date')}）",
                "",
                f"- 节奏：每天 {plan.get('daily_hours')}h，"
                f"总可用 {plan.get('total_hours')}h；目标 {plan.get('target_score')} 分",
            ]
            weak = plan.get("weak_subjects") or []
            if weak:
                src = "（画像推断：答错最多）" if plan.get("weak_from_profile") else ""
                lines.append(f"- 薄弱科目{src}：{'、'.join(weak)}")
            boosted = plan.get("boosted_by_profile") or []
            if boosted:
                lines.append(
                    f"- 画像加权考点（优先级已上调）：{'、'.join(boosted[:6])}"
                )
            stages = plan.get("stages") or []
            if stages:
                lines += ["", "**阶段安排**："]
                for s in stages:
                    lines.append(
                        f"- **{s.get('name')}**（{s.get('weeks')} 周，"
                        f"{s.get('start')} ~ {s.get('end')}）："
                        f"{'、'.join(s.get('focus') or [])}｜检查点：{s.get('checkpoint')}"
                    )
            kps = plan.get("priority_kps") or []
            if kps:
                per: dict[str, list[str]] = {}
                for k in kps:
                    per.setdefault(str(k.get("subject") or ""), []).append(
                        f"{k.get('tag')}（考 {k.get('frequency')} 次）"
                    )
                lines += ["", "**每科优先考点**（按 ROI 排序，频次是真题客观计数）："]
                for subj, tags in per.items():
                    lines.append(f"- {subj}：{'、'.join(tags[:3])}")
            overview = str(plan.get("guide_overview") or "").strip()
            if overview:
                lines += ["", overview]
            for note in (plan.get("risk_notes") or [])[:2]:
                lines.append(f"> {note}")
            lines += [
                "",
                "📌 完整周任务表、打卡与对抗式挑战在左侧「📅 我的规划执行」看板（点击打开）；"
                "这份规划的薄弱科目与重点考点已可由画像持续引用。",
            ]
            parts.append("\n".join(lines))

        judge = evidence.get("judge") or {}
        if judge.get("known"):
            mark = (
                "✅ 答对，已从错题本移出"
                if judge.get("correct")
                else f"❌ 答错（正确答案 {judge.get('truth')}），已记入错题本"
            )
            parts.append(f"**{judge.get('qid')}**：{mark}，画像已更新。")

        checkin = evidence.get("checkin") or {}
        if checkin.get("recorded"):
            mark = "✅ 完成" if checkin.get("done") else "↩️ 取消完成"
            parts.append(
                f"打卡已记录：第 {checkin.get('week')} 周第 "
                f"{(checkin.get('idx') or 0) + 1} 项 {mark}，执行统计已同步画像。"
            )

        profile = evidence.get("profile") or {}
        if not parts and profile:
            # 纯画像查询（「我的画像」「我的进度」）：没有规划/写操作时展示摘要
            quiz = profile.get("quiz") or {}
            exe = profile.get("execution") or {}
            lines = [
                "## 👤 你的备考画像",
                "",
                f"- 累计提问 {profile.get('total_questions', 0)} 次",
            ]
            top = profile.get("top_tags") or []
            if top:
                lines.append(
                    "- 高频提问考点："
                    + "、".join(f"{t['tag']}（{t['count']}次）" for t in top[:5])
                )
            if quiz.get("total"):
                lines.append(
                    f"- 真题作答 {quiz['total']} 题，答对 {quiz.get('correct', 0)} 题"
                )
                wrong = quiz.get("wrong_tags") or []
                if wrong:
                    lines.append(
                        "- 答错考点（已加权进规划）："
                        + "、".join(str(t.get("tag")) for t in wrong[:5])
                    )
            if exe.get("tasks_total"):
                lines.append(
                    f"- 打卡完成 {exe.get('done', 0)}/{exe.get('tasks_total')} 项任务"
                )
            failed = exe.get("failed_checkpoints") or []
            if failed:
                lines.append(f"- 未达成检查点：{'、'.join(failed)}（考点已加权）")
            boost = profile.get("boost_tags") or []
            if boost:
                lines.append(
                    "- 当前规划加权考点："
                    + "、".join(str(t.get("tag")) for t in boost[:8])
                )
            lines += ["", "说「帮我规划复习」即可基于这份画像生成个性化计划。"]
            parts.append("\n".join(lines))

        if not parts:
            return None
        return "\n\n".join(parts)

    # -- 内部 -------------------------------------------------------------

    def _topic(self, state: AgentState) -> str | None:
        return state.get("topic") or extract_topic(state.get("question", ""), self.tools)

    def _propose_expression(
        self, question: str, options: dict[str, str]
    ) -> dict[str, str] | None:
        """让 LLM 只产出一个算式，由本地计算器求值。

        小模型心算不可靠，但「把题目翻译成算式」是它擅长的。
        输出格式用 JSON；供应商不支持 json_mode 时回退正则。
        """
        if self.llm is None or not self.llm.available:
            return None
        opts = "\n".join(f"{k}. {v}" for k, v in options.items())
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 408 计算题助手。把题目里需要的计算写成一个 Python 算术表达式，"
                    "只使用数字、+ - * / // % ** 括号，以及 log2/sqrt/ceil/floor/abs/min/max。"
                    '只输出 JSON：{"expr": "...", "why": "一句话说明这个算式求的是什么"}'
                ),
            },
            {"role": "user", "content": f"题目：{question}\n选项：\n{opts}"},
        ]
        try:
            resp = self.llm.chat(messages, max_tokens=800)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"生成算式失败：{e}")
            return None
        if not resp.ok:
            return None
        text = resp.text
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj.get("expr"), str):
                    return {"expr": obj["expr"], "why": str(obj.get("why") or "")}
            except json.JSONDecodeError:
                pass
        m = re.search(r"expr\s*[:：]\s*[\"']?(.+?)[\"']?\s*$", text, re.M)
        if m:
            return {"expr": m.group(1).strip(), "why": ""}
        return None

    @staticmethod
    def _blocks(state: AgentState, *, max_chars: int = 2400) -> list[ContextBlock]:
        """把 state 里的检索结果转成带编号的资料块（与 answer 的编号一致）。

        必须限制总字符数：实测当前 endpoint 的 flash 模型上下文约 4k token，
        检索 10 条 × 500 字再加图解析/计算证据会让 prompt 超过 4000 token，
        模型返回**空内容**（不是报错），表现为「调用成功但没有回答」。
        压缩到 3200 字后，三题型均正常。
        """
        out: list[ContextBlock] = []
        used = 0
        for i, c in enumerate(state.get("contexts") or [], 1):
            text = c.get("text", "")
            if used + len(text) > max_chars and out:
                break
            out.append(
                ContextBlock(
                    index=i,
                    chunk_id=c.get("chunk_id", ""),
                    text=text,
                    citation=c.get("citation", ""),
                    subject=c.get("subject", ""),
                    source=c.get("source", ""),
                )
            )
            used += len(text)
        return out