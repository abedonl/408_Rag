"""有界 plan-act-observe 与 verify 重试的单测。

用假的检索/图谱/LLM，不依赖 Qdrant / Neo4j / 真实 API。
重点验证「循环有上限、每轮只调一个工具、终态明确」。
"""

from __future__ import annotations

import time

from cs408rag.agent import nodes as nodes_mod
from cs408rag.agent.exam_analysis import ExamAnalyzer
from cs408rag.agent.graph import AgentRunner
from cs408rag.agent.nodes import AgentNodes
from cs408rag.agent.state import initial_state
from cs408rag.agent.tools import AgentTools
from cs408rag.generation.llm import LLMResponse
from cs408rag.retrieval.pipeline import RetrievalResult
from cs408rag.retrieval.vector_store import SearchHit


class FakePipeline:
    def __init__(self, n_hits: int = 3):
        self.n_hits = n_hits
        self.queries: list[str] = []

    def retrieve(self, query, *, subject=None, config=None):
        self.queries.append(query)
        hits = [
            SearchHit(
                chunk_id=f"chunk-{i}",
                score=1.0 - i * 0.1,
                payload={
                    "chunk_id": f"chunk-{i}",
                    "text": f"死锁是并发程序中的一种状态。片段{i}",
                    "subject": "操作系统",
                    "source": "wangdao",
                    "chapter_path": ["操作系统", "并发", "死锁"],
                },
            )
            for i in range(self.n_hits)
        ]
        return RetrievalResult(
            query=query,
            ranked_ids=[h.chunk_id for h in hits],
            scores=[h.score for h in hits],
            hits_by_id={h.chunk_id: h for h in hits},
            routes={},
            took_ms=1.0,
        )


class FakeGraph:
    def kp_profile(self, name):
        return {
            "title": name,
            "subject": "操作系统",
            "path": ["操作系统", "并发", name],
            "siblings": ["互斥", "同步"],
            "ancestors": ["操作系统", "并发"],
            "tags": [name],
            "cooccurring": [],
        }

    def learning_path(self, name):
        return {"found": True, "chain": ["进程", "线程", "互斥", name], "depth": 3}


class FakeLLM:
    available = True

    def __init__(self, answers: list[str] | None = None):
        self.answers = answers or ["死锁是并发程序中的一种状态。"]
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_mode=False):
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.answers) - 1)
        return LLMResponse(
            text=self.answers[idx],
            model="fake",
            prompt_tokens=10,
            completion_tokens=5,
        )


def _runner(llm=None, *, n_hits: int = 3, valid_refs=None) -> AgentRunner:
    tools = AgentTools(
        pipeline=FakePipeline(n_hits),
        analyzer=ExamAnalyzer(),
        graph=FakeGraph(),
        valid_exam_refs=valid_refs,
    )
    return AgentRunner(tools=tools, llm=llm or FakeLLM())


def _planner_actions(res) -> list[str]:
    return [p["action"] for p in res.plan_history]


class TestBoundedLoop:
    def test_exam_two_rounds_stats_then_learning_path(self):
        llm = FakeLLM(["死锁历年考查 6 次；先学进程与线程。"])
        res = _runner(llm).run("死锁考几次？先学什么？", mode="qa")

        assert res.route == "exam"
        assert _planner_actions(res) == ["get_exam_stats", "get_learning_path"]
        assert res.rounds == 2
        assert res.stop_reason == "answered"
        assert res.tool_outputs["exam_stats"]["total"] >= 1
        assert res.tool_outputs["learning_path"]["found"] is True
        # 每个 planner 轮次只执行一个工具（round>0 才是 planner 的工具）
        planned = [c for c in res.tool_calls if c["round"] > 0]
        assert len(planned) == 2

    def test_exam_related_then_questions_with_limit(self):
        res = _runner().run("死锁和哪些考点一起考？给 2 道相关真题", mode="qa")

        assert _planner_actions(res) == ["get_related_kps", "get_kp_questions"]
        assert res.rounds == 2
        assert res.tool_outputs["kp_questions"]["n"] <= 2

    def test_loop_stops_at_three_rounds(self):
        q = "死锁考几次？和哪些考点一起考？先学什么？给 3 道真题"
        res = _runner().run(q, mode="qa")

        assert res.rounds == 3
        assert len(res.plan_history) == 3
        assert res.stop_reason == "max_rounds"

    def test_concept_uses_one_graph_round(self):
        res = _runner().run("什么是死锁", mode="qa")

        assert res.route == "concept"
        assert _planner_actions(res) == ["search_knowledge_graph"]
        assert res.rounds == 1
        assert res.tool_outputs["knowledge_graph"]["title"] == "死锁"

    def test_calc_expression_failure_is_observed_not_fatal(self):
        llm = FakeLLM(['{"expr": "1/0", "why": "故意除零"}', "无法计算。"])
        res = _runner(llm).run("Cache 命中率至少是多少", mode="qa")

        assert res.route == "calc"
        assert _planner_actions(res) == ["calculate"]
        assert res.rounds == 1
        assert res.tool_failures[0]["tool"] == "calculate"
        assert res.tool_failures[0]["error_type"] == "validation"
        # 工具失败不阻断回答（检索仍可用）
        assert res.stop_reason == "answered"

    def test_no_evidence_when_nothing_to_do(self):
        tools = AgentTools(pipeline=FakePipeline(0), analyzer=ExamAnalyzer(), graph=FakeGraph())
        node = AgentNodes(tools, llm=None)
        state = initial_state("嗯")
        state["route"] = "cross"
        state["contexts"] = []

        # 没有关键词 -> 规则 planner 不选工具
        planned = node.plan(state)
        assert planned["plan"] == {}
        state.update(planned)
        state.update(node.act(state))
        out = node.observe(state)

        assert out["stop_reason"] == "no_evidence"
        assert out["tool_rounds"] == 0

    def test_budget_terminal(self):
        node = AgentNodes(AgentTools(pipeline=FakePipeline(1)), llm=None)
        state = initial_state("死锁考几次？", budget_s=1.0)
        state["started_at"] = time.monotonic() - 5

        out = node.plan(state)
        assert out["stop_reason"] == "budget"
        assert out["plan"] == {}


class TestObservation:
    def test_observation_is_truncated(self):
        node = AgentNodes(AgentTools(), llm=None)
        state = initial_state("2009-04")
        state["route"] = "figure"
        state["qid"] = "2009-04"
        state["observations"] = [{"tool": "read_figure", "ok": True}]
        state["last_result"] = {
            "tool": "read_figure",
            "ok": True,
            "data": {"description": "x" * 5000},
            "summary": "图解析",
            "round": 1,
            "took_ms": 1.0,
        }

        out = node.observe(state)
        obs = out["observations"][0]
        assert len(obs["snippet"]) <= 600
        assert obs["ok"] is True
        assert out["stop_reason"] == "answered"

    def test_calculate_ok_false_counts_as_failure(self):
        node = AgentNodes(AgentTools(), llm=None)
        state = initial_state("1/0")
        state["route"] = "calc"
        state["observations"] = [{"tool": "calculate", "ok": False}]
        state["last_result"] = {
            "tool": "calculate",
            "ok": True,
            "data": {"ok": False, "error": "除零", "error_type": "validation"},
            "summary": "计算失败",
            "round": 1,
        }

        out = node.observe(state)
        assert out["observations"][0]["ok"] is False
        assert out["tool_failures"][0]["error_type"] == "validation"


class TestVerifyRetry:
    def test_invalid_exam_ref_rewrites_once(self):
        llm = FakeLLM(
            [
                "死锁是并发程序中的状态，见 2099#1。",
                "死锁是并发程序中的状态，见 2009#01。",
            ]
        )
        res = _runner(llm, valid_refs={"2009#01"}).run("什么是死锁", mode="qa")

        assert res.verify_retries == 1
        assert res.stop_reason == "answered"
        assert res.verification["ok"] is True
        assert "2099" not in res.answer
        assert len(llm.calls) == 2
        assert any("未通过自检" in m["content"] for m in llm.calls[1])

    def test_only_one_retry_even_if_it_keeps_failing(self):
        llm = FakeLLM(["见 2099#1。", "还是见 2099#2。", "第三次 2099#3。"])
        res = _runner(llm, valid_refs={"2009#01"}).run("什么是死锁", mode="qa")

        assert res.verify_retries == 1
        assert res.stop_reason == "verify_retry"
        assert len(llm.calls) == 2
        assert res.verification["ok"] is False

    def test_llm_failing_on_rewrite_does_not_loop(self):
        """重写时 LLM 掉线：verify_feedback 必须被消费，否则 answer↔verify 死循环。"""

        class DropsAfterFirstCall(FakeLLM):
            def chat(self, messages, **kwargs):
                resp = super().chat(messages, **kwargs)
                self.available = False
                return resp

        llm = DropsAfterFirstCall(["见 2099#1。"])
        res = _runner(llm, valid_refs={"2009#01"}).run("什么是死锁", mode="qa")

        assert res.verify_retries == 1
        assert res.stop_reason == "tool_error"
        assert len(llm.calls) == 1


class TestLLMPlanner:
    def test_rejects_action_outside_whitelist(self):
        llm = FakeLLM(['{"action": "rm_rf", "args": {}, "reason": "bad"}'])
        node = AgentNodes(AgentTools(pipeline=FakePipeline(0)), llm=llm)
        state = initial_state("一个没有关键词的问题")
        state["route"] = "cross"
        state["contexts"] = [{"citation": "操作系统 > 死锁"}]

        plan, called, _tok = node._llm_plan(state)
        assert called is True
        assert plan is None

    def test_accepts_valid_action_after_arg_validation(self):
        llm = FakeLLM(
            ['{"action": "get_exam_stats", "args": {"kp_name": "死锁"}, "reason": "统计"}']
        )
        node = AgentNodes(AgentTools(pipeline=FakePipeline(0)), llm=llm)
        state = initial_state("死锁怎么样")
        state["route"] = "cross"
        state["contexts"] = [{"citation": "操作系统 > 死锁"}]

        plan, called, tok = node._llm_plan(state)
        assert called is True
        assert isinstance(tok, dict)  # token 用量随规划结果一起返回
        assert plan is not None and plan["action"] == "get_exam_stats"

    def test_planner_budget_is_one_call(self):
        llm = FakeLLM(['{"action": "get_exam_stats", "args": {"kp_name": "死锁"}}'])
        node = AgentNodes(AgentTools(pipeline=FakePipeline(0)), llm=llm)
        state = initial_state("死锁怎么样")
        state["route"] = "cross"
        state["contexts"] = [{"citation": "操作系统 > 死锁"}]
        state["planner_calls"] = nodes_mod.MAX_PLANNER_LLM_CALLS

        plan, called, _tok = node._llm_plan(state)
        assert plan is None and called is False
        assert llm.calls == []