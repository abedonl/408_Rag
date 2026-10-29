"""Agent 各节点的实现。

刻意不用 prebuilt ReAct：本环境的 LLM 长输出约 12 token/s、单次完整问答
50s 以上，ReAct 的自由多轮会把延迟放到不可接受。这里用**确定性路由 +
显式工具调用**，每道题最多 2 次检索、1 次工具批、1 次生成，
既省成本，又让每一步都能写进 trace 做分层评测。

判断「要不要多轮」的原则：
    概念题答案通常在某一节里，多轮只会引入噪声 -> 检索 1 轮就答
    计算/图表/考点题单轮检索按定义答不了 -> 调工具，但不无限 ReAct
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from loguru import logger

from cs408rag.agent.exam_analysis import RE_CALC_ASK
from cs408rag.agent.state import AgentState
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
    "先学", "前置", "学习路径", "学习顺序",
)
FIGURE_HINTS = ("下图", "如图", "图中", "图所示", "所示图", "配图", "图示")


def _step(node: str, took_ms: float, **detail: Any) -> dict[str, Any]:
    return {"node": node, "took_ms": round(took_ms, 1), "detail": detail}


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
        if any(h in q for h in EXAM_HINTS):
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

        return {
            "route": route,
            "route_reason": reason,
            "attempts": 0,
            "steps": [
                _step(
                    "route",
                    (time.perf_counter() - t0) * 1000,
                    route=route,
                    reason=reason,
                )
            ],
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

    # -- act -------------------------------------------------------------

    def act(self, state: AgentState) -> dict[str, Any]:
        t0 = time.perf_counter()
        route = state.get("route", "concept")
        q = state.get("question", "")
        qid = state.get("qid")
        out: dict[str, Any] = {}
        used: list[str] = []

        if route == "figure" and qid:
            r = self.tools.call("read_figure", question_id=qid, question=q)
            if r.ok:
                out["figure"] = r.data
                used.append("read_figure")

        elif route == "calc":
            expr = self._propose_expression(q, state.get("options") or {})
            if expr:
                r = self.tools.call("calculate", expr=expr["expr"])
                if r.ok:
                    out["calculate"] = {**r.data, "why": expr.get("why", "")}
                    used.append("calculate")
                else:
                    out["calculate"] = r.data
            else:
                out["calculate"] = {
                    "ok": False,
                    "error": "未能生成可计算的表达式（LLM 不可用或未给出算式）",
                }

        elif route == "exam":
            topic = self._topic(state)
            if topic:
                stats = self.tools.call("get_exam_stats", kp_name=topic)
                if stats.ok:
                    out["exam_stats"] = stats.data
                qs = self.tools.call("get_kp_questions", kp_name=topic, limit=5)
                if qs.ok:
                    out["kp_questions"] = qs.data
                rel = self.tools.call("get_related_kps", kp_name=topic)
                if rel.ok:
                    out["related"] = rel.data
                used.extend(["get_exam_stats", "get_kp_questions", "get_related_kps"])
                if any(k in q for k in ("先学", "前置", "学习路径", "学习顺序", "基础")):
                    lp = self.tools.call("get_learning_path", kp_name=topic)
                    if lp.ok:
                        out["learning_path"] = lp.data
                        used.append("get_learning_path")
            else:
                out["exam_stats"] = {"tag": q[:30], "total": 0}

        # 概念题：用图谱定位章节，帮助引用
        if route in ("concept", "cross") and extract_keywords(q, max_terms=3):
            kp = extract_keywords(q, max_terms=3)[0]
            r = self.tools.call("search_knowledge_graph", kp_name=kp)
            if r.ok and (r.data or {}).get("found"):
                out["knowledge_graph"] = r.data
                used.append("search_knowledge_graph")

        return {
            "tool_outputs": out,
            "steps": [_step("act", (time.perf_counter() - t0) * 1000, tools=used, route=route)],
        }

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
                "steps": [_step("answer", (time.perf_counter() - t0) * 1000, skipped=True)],
            }

        if self.llm is None or not self.llm.available:
            return {
                "answer": "",
                "llm_error": "未配置 LLM API key，生成层跳过（检索与工具结果仍可用）",
                "steps": [_step("answer", (time.perf_counter() - t0) * 1000, skipped=True)],
            }

        if mode == "quiz":
            # 部分真题的选项画在图中（options 为空），此时仍走 quiz 口径：
            # 依据图形解析判断 A/B/C/D，并从回答里解析字母。
            messages = agent_quiz_messages(q, options, blocks, evidence)
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
            )
            max_tokens = 2000
        else:
            messages = agent_qa_messages(q, blocks, evidence)
            max_tokens = 2000

        resp = self.llm.chat(messages, max_tokens=max_tokens)
        # 长上下文 + 思考型模型可能返回空正文；缩短上下文再试一次。
        # 这不是「答错」，评测里必须单独统计 empty。
        if resp.ok and not resp.text.strip() and mode == "quiz":
            resp = self.llm.chat(
                agent_quiz_messages(q, options, blocks[:4], evidence),
                max_tokens=3200,
            )
        choice = parse_choice(resp.text) if mode == "quiz" else ""
        empty = not (resp.text or "").strip()
        return {
            "answer": resp.text,
            "choice": choice,
            "llm_error": (
                "模型返回空正文"
                if resp.ok and empty
                else ("" if resp.ok else resp.error)
            ),
            "empty": empty,
            "tokens": {
                "prompt": resp.prompt_tokens,
                "completion": resp.completion_tokens,
            },
            "steps": [
                _step(
                    "answer",
                    (time.perf_counter() - t0) * 1000,
                    ok=resp.ok,
                    model=resp.model,
                    prompt_tokens=resp.prompt_tokens,
                    completion_tokens=resp.completion_tokens,
                )
            ],
        }

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

        return {
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
                )
            ],
        }

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