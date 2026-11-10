"""会话记忆与指代消解的单测（Week 2）。

覆盖：SessionStore 的 LRU/TTL/8 轮裁剪、followup 规则消解、
以及带 session 的两轮问答（「那 2021 年呢」沿用上轮考点）。
不依赖 Qdrant / Neo4j / 真实 LLM。
"""

from __future__ import annotations

import time

from cs408rag.agent.followup import resolve_followup
from cs408rag.agent.graph import AgentRunner
from cs408rag.agent.memory import (
    MAX_TURNS,
    AgentSession,
    InMemorySessionStore,
    Turn,
)


def _turn(i: int) -> Turn:
    return Turn(question=f"问题{i}", answer=f"回答{i}", route="concept")


# ---------------------------------------------------------------------------
# SessionStore
# ---------------------------------------------------------------------------


class TestSessionStore:
    def test_get_or_create_and_get(self):
        store = InMemorySessionStore()
        s = store.get_or_create("s1")
        assert s.session_id == "s1"
        assert store.get("s1") is s
        # 同 id 再 get_or_create 不新建
        assert store.get_or_create("s1") is s
        # 缺省 id 自动生成
        s2 = store.get_or_create()
        assert s2.session_id and s2.session_id != "s1"

    def test_same_session_id_different_clients_isolated(self):
        """同一 session_id 被另一个身份使用：不读也不写对方的历史。"""
        store = InMemorySessionStore()
        a = store.get_or_create("conv-1", client_id="alice")
        store.append_turn(a, Turn(question="alice 的问题", answer="alice 的回答"))

        b = store.get_or_create("conv-1", client_id="bob")
        assert b is not a and b.session_id != "conv-1"
        assert b.turns == [] and b.client_id == "bob"

        # alice 的会话没有被 bob 污染，仍可按原 id 取回
        again = store.get("conv-1")
        assert again.client_id == "alice"
        assert len(again.turns) == 1

    def test_first_client_id_binds_session(self):
        """先由无身份入口建会话，后续带 cid 续聊时补绑定，而不是另开一个。"""
        store = InMemorySessionStore()
        s = store.get_or_create("s1")
        assert s.client_id == ""
        assert store.get_or_create("s1", client_id="alice") is s
        assert s.client_id == "alice"

    def test_ttl_expiry(self):
        store = InMemorySessionStore(ttl_s=0.01)
        store.get_or_create("s1")
        time.sleep(0.02)
        assert store.get("s1") is None

    def test_lru_eviction(self):
        store = InMemorySessionStore(max_sessions=2)
        store.get_or_create("a")
        store.get_or_create("b")
        store.get("a")  # a 变最近使用
        store.get_or_create("c")  # 挤出 b
        assert store.get("a") is not None
        assert store.get("b") is None
        assert store.get("c") is not None

    def test_delete(self):
        store = InMemorySessionStore()
        store.get_or_create("s1")
        assert store.delete("s1") is True
        assert store.delete("s1") is False

    def test_trim_to_summary(self):
        store = InMemorySessionStore()
        s = store.get_or_create("s1")
        for i in range(MAX_TURNS + 3):
            store.append_turn(s, _turn(i))
        assert len(s.turns) == MAX_TURNS
        assert s.turn_count == MAX_TURNS + 3
        # 最早的 3 轮已压进 summary
        assert "问题0" in s.summary and "问题2" in s.summary
        assert "问题0" not in [t.question for t in s.turns]
        # 注入 prompt 的历史：summary + 最近 4 轮
        hist = s.render_history()
        assert "更早的对话摘要" in hist
        assert f"问题{MAX_TURNS + 2}" in hist  # 最后一轮完整保留

    def test_last_qid_topic_updated(self):
        store = InMemorySessionStore()
        s = store.get_or_create("s1")
        store.append_turn(s, Turn(question="2009-04 选什么", answer="B", qid="2009-04", topic="二叉排序树"))
        assert s.last_qid == "2009-04"
        assert s.last_topic == "二叉排序树"


# ---------------------------------------------------------------------------
# followup 指代消解
# ---------------------------------------------------------------------------


def _session(last_qid=None, last_topic=None, with_turn=True) -> AgentSession:
    s = AgentSession(session_id="x")
    s.last_qid = last_qid
    s.last_topic = last_topic
    if with_turn:
        s.turns.append(Turn(question="死锁考过几次", answer="6 次", topic=last_topic, qid=last_qid))
    return s


class TestFollowup:
    def test_year_followup_uses_last_topic(self):
        r = resolve_followup("那 2021 年呢？", _session(last_topic="死锁"))
        assert r is not None
        assert "死锁" in r.question and "2021" in r.question

    def test_year_followup_without_topic_returns_none(self):
        assert resolve_followup("那 2021 年呢？", _session(last_topic=None)) is None

    def test_qid_pronoun_uses_last_qid(self):
        r = resolve_followup("这题选什么？", _session(last_qid="2009-04"))
        assert r is not None and r.qid == "2009-04"
        assert r.question == "这题选什么？"  # 问题本身不改写

    def test_topic_pronoun_replaced(self):
        r = resolve_followup("这个考点和哪些知识点关联？", _session(last_topic="死锁"))
        assert r is not None and "死锁" in r.question

    def test_learning_followup_prepends_topic(self):
        r = resolve_followup("那先学什么？", _session(last_topic="死锁"))
        assert r is not None and r.question.startswith("死锁")

    def test_no_session_or_empty_history(self):
        assert resolve_followup("那 2021 年呢？", None) is None
        assert resolve_followup("那 2021 年呢？", _session(last_topic="死锁", with_turn=False)) is None

    def test_unrelated_question_not_touched(self):
        r = resolve_followup("什么是虚拟内存？", _session(last_qid="2009-04", last_topic="死锁"))
        assert r is None


# ---------------------------------------------------------------------------
# Runner 集成：两轮会话
# ---------------------------------------------------------------------------


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

    def __init__(self):
        self.calls: list[list[dict[str, str]]] = []

    def chat(self, messages, *, temperature=0.0, max_tokens=1024, json_mode=False):
        from cs408rag.generation.llm import LLMResponse

        self.calls.append(messages)
        return LLMResponse(text="死锁历年共考查 6 次。", model="fake", prompt_tokens=10, completion_tokens=5)


class TestRunnerSession:
    def _runner(self) -> tuple[AgentRunner, _FakeLLM]:
        from cs408rag.agent.exam_analysis import ExamAnalyzer
        from cs408rag.agent.tools import AgentTools

        llm = _FakeLLM()
        tools = AgentTools(pipeline=_FakePipeline(), analyzer=ExamAnalyzer(), graph=_FakeGraph())
        return AgentRunner(tools=tools, llm=llm), llm

    def test_two_turns_share_session(self):
        runner, llm = self._runner()
        r1 = runner.run("死锁这个考点历年考过几次？", session_id="t-two-turns")
        assert r1.turn_no == 1
        assert r1.session_id == "t-two-turns"

        r2 = runner.run("那 2021 年呢？", session_id="t-two-turns")
        assert r2.turn_no == 2
        # 指代消解：沿用上轮考点「死锁」
        assert "死锁" in r2.resolved_query and "2021" in r2.resolved_query
        # 第二轮的 prompt 里注入了第一轮历史
        last_user_msg = llm.calls[-1][-1]["content"]
        assert "对话历史" in last_user_msg and "死锁这个考点历年考过几次" in last_user_msg

    def test_new_session_isolated(self):
        runner, _ = self._runner()
        runner.run("死锁考过几次？", session_id="t-iso-a")
        r = runner.run("那 2021 年呢？", session_id="t-iso-b")
        # 新会话没有上轮考点，不做消解
        assert r.resolved_query == ""

    def test_rehydrate_from_history_restores_followups(self):
        """切回旧会话后，用落库历史回灌内存会话，指代消解仍可用。"""
        runner, _ = self._runner()
        history = [
            {"q": "死锁这个考点历年考过几次？", "d": {"answer": "6 次", "route": "exam"}},
            {"q": "那 2021 年呢？", "d": {"answer": "2 次", "route": "exam"}},
        ]
        assert runner.rehydrate_session("t-rehydrate", history) == 2
        session = runner.sessions.get("t-rehydrate")
        assert session is not None
        assert len(session.turns) == 2

        # 回灌后再追问，指代能沿用回灌的最后一轮
        r = runner.run("那 2022 年呢？", session_id="t-rehydrate")
        assert "死锁" in r.resolved_query and "2022" in r.resolved_query

        # 重复回灌不叠加：已有轮次时直接返回当前轮数
        before = len(session.turns)
        assert runner.rehydrate_session("t-rehydrate", history) == before
        assert len(session.turns) == before

    def test_no_session_id_means_stateless(self):
        runner, _ = self._runner()
        r = runner.run("死锁考过几次？")
        assert r.session_id == "" and r.turn_no == 0

    def test_rehydrate_merges_turns_appended_elsewhere(self):
        """同一会话被别的入口续写后，回灌要增量补齐，且重复调用不叠加。"""
        runner, _ = self._runner()
        history = [{"q": "死锁这个考点历年考过几次？", "d": {"answer": "6 次", "route": "exam"}}]
        assert runner.rehydrate_session("t-merge", history) == 1

        history.append({"q": "那 2022 年呢？", "d": {"answer": "1 次", "route": "exam"}})
        assert runner.rehydrate_session("t-merge", history) == 2
        session = runner.sessions.get("t-merge")
        assert [t.question for t in session.turns][-1] == "那 2022 年呢？"

        # 历史没变：只比较长度，不重复灌
        assert runner.rehydrate_session("t-merge", history) == 2
        assert len(session.turns) == 2

    def test_forged_session_id_cannot_read_other_client(self):
        """伪造别人的 session_id：拿不到历史，也不会写进对方会话。"""
        runner, _ = self._runner()
        runner.run(
            "死锁这个考点历年考过几次？", session_id="t-share", client_id="alice"
        )
        alice = runner.sessions.get("t-share")
        assert alice.client_id == "alice" and len(alice.turns) == 1

        r = runner.run("那 2021 年呢？", session_id="t-share", client_id="bob")
        assert r.session_id != "t-share"
        assert r.resolved_query == ""  # bob 没有 alice 的上轮语境

        assert len(alice.turns) == 1  # bob 这轮没有写进 alice 的会话
        assert alice.client_id == "alice"
