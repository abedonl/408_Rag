"""澄清 HITL 与成本追踪的单测（Week 3）。不依赖 Qdrant / Neo4j / 真实 LLM。"""

from __future__ import annotations

from cs408rag.agent.cost import compute_cost_cny
from cs408rag.agent.graph import AgentRunner


class _FakePipeline:
    def retrieve(self, query, *, subject=None, config=None):
        from cs408rag.retrieval.pipeline import RetrievalResult
        from cs408rag.retrieval.vector_store import SearchHit

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
            for i in range(3)
        ]
        return RetrievalResult(
            query=query,
            ranked_ids=[h.chunk_id for h in hits],
            scores=[h.score for h in hits],
            hits_by_id={h.chunk_id: h for h in hits},
            routes={},
            took_ms=1.0,
        )


class _FakeGraph:
    def kp_profile(self, name):
        return {"title": name, "subject": "操作系统", "path": [], "siblings": [], "ancestors": [], "tags": [name], "cooccurring": []}

    def learning_path(self, name):
        return {"found": True, "chain": ["进程", name], "depth": 1}


class _FakeLLM:
    available = True
    model = "deepseek/deepseek-chat"

    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_mode=False):
        from cs408rag.generation.llm import LLMResponse

        self.calls.append(messages)
        return LLMResponse(text="死锁历年共考查 6 次。", model=self.model, prompt_tokens=1000, completion_tokens=500)


def _runner() -> tuple[AgentRunner, _FakeLLM]:
    from cs408rag.agent.exam_analysis import ExamAnalyzer
    from cs408rag.agent.tools import AgentTools

    llm = _FakeLLM()
    tools = AgentTools(pipeline=_FakePipeline(), analyzer=ExamAnalyzer(), graph=_FakeGraph())
    return AgentRunner(tools=tools, llm=llm), llm


class TestClarify:
    def test_question_reference_without_qid_clarifies(self):
        runner, llm = _runner()
        r = runner.run("这道题选什么？")
        assert r.needs_clarification is True
        assert r.stop_reason == "clarification"
        assert len(r.clarification_options) >= 2
        assert r.answer.startswith("这个问题我需要先确认一下")
        # 澄清短路：不检索、不调 LLM、不产生成本
        assert r.contexts == [] and llm.calls == [] and r.cost_cny == 0.0

    def test_topic_ambiguity_clarifies_with_candidates(self):
        runner, _ = _runner()
        r = runner.run("这个考点历年考了几次？")
        assert r.needs_clarification is True
        # 5 个高频考点候选 + 1 个换说法
        assert len(r.clarification_options) == 6
        assert any("我想问的是" in o for o in r.clarification_options)

    def test_followup_resolved_qid_skips_clarify(self):
        runner, _ = _runner()
        # 第一轮建立 last_qid：带题号的问题
        runner.run("2009-04 这道题考查什么？", session_id="t-clarify")
        # 第二轮「这题」被消解出 qid，不再触发澄清
        r = runner.run("这题选什么？", session_id="t-clarify")
        assert r.needs_clarification is False

    def test_normal_exam_question_not_clarified(self):
        runner, _ = _runner()
        r = runner.run("死锁这个考点历年考过几次？")
        assert r.needs_clarification is False
        assert r.route == "exam"


class TestCost:
    def test_pricing_prefix_match(self):
        c = compute_cost_cny("deepseek/deepseek-chat", 1000, 1000)
        assert c == 0.002 + 0.008

    def test_unknown_model_uses_default(self):
        c = compute_cost_cny("some-new-model", 1000, 1000)
        assert c == 0.002 + 0.008

    def test_zero_tokens(self):
        assert compute_cost_cny("deepseek-chat", 0, 0) == 0.0

    def test_runner_accumulates_cost(self):
        runner, _ = _runner()
        r = runner.run("死锁这个考点历年考过几次？")
        assert r.tokens.get("prompt", 0) > 0
        assert r.cost_cny > 0
