"""工具契约与护栏单测。

覆盖的是「错了会直接让 Agent 失控」的路径：

1. 契约与注册表必须一一对应，少一个工具就等于 planner 可以调不存在的工具
2. 参数校验失败要返回 validation，而不是被工具异常吞掉
3. 预算超限返回 budget，不静默重试
4. 超时返回 timeout，不阻塞主流程
5. 只有幂等工具允许自动重试
6. 计算器恶意表达式在进入求值器之前就被挡掉

不依赖 LLM / Qdrant / Neo4j，纯本地可跑。
"""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from cs408rag.agent.contracts import (
    TOOL_SPECS,
    GraphArgs,
    KpQuestionsArgs,
    SearchKBArgs,
    ToolSpec,
    ToolTrace,
    args_digest,
)
from cs408rag.agent.guardrails import ToolGuard
from cs408rag.agent.tools import TOOL_REGISTRY, AgentTools


class _FakeTools:
    """最小工具宿主：只提供 guard 需要的 trace/failures/_execute。"""

    def __init__(self, fn):
        self.fn = fn
        self.trace = ToolTrace()
        self.failures: dict[str, int] = {}

    def _execute(self, name, args):
        return self.fn(name, args)


class TestContracts:
    def test_registry_and_contracts_match(self):
        assert set(TOOL_REGISTRY) == set(TOOL_SPECS)
        assert len(TOOL_SPECS) == 13

    def test_all_specs_have_args_models(self):
        for spec in TOOL_SPECS.values():
            assert spec.args_model.__name__.endswith("Args")

    def test_search_args_bounds(self):
        assert SearchKBArgs(query="死锁", top_k=8).top_k == 8
        with pytest.raises(ValidationError):
            SearchKBArgs(query="", top_k=8)
        with pytest.raises(ValidationError):
            SearchKBArgs(query="死锁", top_k=999)

    def test_kp_questions_limit_bounds(self):
        assert KpQuestionsArgs(kp_name="死锁", limit=2).limit == 2
        with pytest.raises(ValidationError):
            KpQuestionsArgs(kp_name="死锁", limit=0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GraphArgs(kp_name="死锁", unexpected=True)

    def test_args_digest_is_short_and_stable(self):
        a = args_digest({"expr": "1+1"})
        b = args_digest({"expr": "1+1"})
        assert a == b and len(a) == 12
        assert "1+1" not in a


class TestGuardValidation:
    def test_unknown_tool_rejected(self):
        t = AgentTools()
        r = t.call("rm_rf", path="/")
        assert not r.ok and r.error_type == "unknown_tool"
        assert t.trace.n_calls == 1

    def test_empty_kp_name_is_validation(self):
        t = AgentTools()
        r = t.call("get_exam_stats", kp_name="")
        assert not r.ok and r.error_type == "validation"
        assert not r.validated

    def test_limit_out_of_range_is_validation(self):
        t = AgentTools()
        r = t.call("get_kp_questions", kp_name="死锁", limit=999)
        assert not r.ok and r.error_type == "validation"

    def test_extra_kwarg_is_validation(self):
        t = AgentTools()
        r = t.call("get_exam_stats", kp_name="死锁", bogus=1)
        assert not r.ok and r.error_type == "validation"

    def test_bad_qid_is_validation(self):
        t = AgentTools()
        r = t.call("read_figure", question_id="2009/04")
        assert not r.ok and r.error_type == "validation"

    def test_malicious_expression_blocked_before_evaluator(self):
        t = AgentTools()
        r = t.call("calculate", expr="__import__('os').system('echo hacked')")
        assert not r.ok and r.error_type == "validation"
        assert "不允许" in r.error or "禁用" in r.error
        # 没有真正执行工具
        assert t.trace.calls[-1].took_ms == 0


class TestGuardBudget:
    def test_round_budget(self):
        t = AgentTools()
        t.reset()
        for _ in range(6):
            assert t.call("calculate", expr="1+1").ok
        r = t.call("calculate", expr="1+1")
        assert not r.ok and r.error_type == "budget"

        # 新一轮预算重置
        t.guard.begin_round(2)
        assert t.call("calculate", expr="1+1", round_no=2).ok

    def test_minute_budget(self):
        tools = _FakeTools(lambda name, args: ({"ok": True}, "ok"))
        guard = ToolGuard(
            tools,  # type: ignore[arg-type]
            max_calls_per_round=99,
            max_calls_per_minute=2,
        )
        assert guard.call("search_knowledge_graph", kp_name="死锁").ok
        assert guard.call("search_knowledge_graph", kp_name="死锁").ok
        r = guard.call("search_knowledge_graph", kp_name="死锁")
        assert not r.ok and r.error_type == "budget"

    def test_internal_calls_bypass_budget(self):
        tools = _FakeTools(lambda name, args: ({"ok": True}, "ok"))
        guard = ToolGuard(tools, max_calls_per_round=1)  # type: ignore[arg-type]
        assert guard.call("search_knowledge_graph", kp_name="死锁").ok
        assert guard.call(
            "search_knowledge_graph", kp_name="死锁", count_budget=False
        ).ok
        r = guard.call("search_knowledge_graph", kp_name="死锁")
        assert not r.ok and r.error_type == "budget"


class TestGuardExecution:
    def test_timeout_returns_error_type(self):
        def slow(name, args):
            time.sleep(0.3)
            return {"ok": True}, "done"

        tools = _FakeTools(slow)
        guard = ToolGuard(
            tools,  # type: ignore[arg-type]
            specs={
                "slow_tool": ToolSpec(
                    "slow_tool", GraphArgs, "slow", timeout_s=0.05, retries=0
                )
            },
        )
        t0 = time.perf_counter()
        r = guard.call("slow_tool", kp_name="死锁")
        assert not r.ok and r.error_type == "timeout"
        assert time.perf_counter() - t0 < 0.25

    def test_idempotent_tool_retries_once(self):
        calls = {"n": 0}

        def flaky(name, args):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("backend hiccup")
            return {"ok": True}, "second try"

        tools = _FakeTools(flaky)
        guard = ToolGuard(
            tools,  # type: ignore[arg-type]
            specs={
                "flaky": ToolSpec("flaky", GraphArgs, "flaky", retries=1, idempotent=True)
            },
        )
        r = guard.call("flaky", kp_name="死锁")
        assert r.ok and r.attempt == 2 and calls["n"] == 2

    def test_non_idempotent_tool_never_retries(self):
        calls = {"n": 0}

        def flaky(name, args):
            calls["n"] += 1
            raise RuntimeError("boom")

        tools = _FakeTools(flaky)
        guard = ToolGuard(
            tools,  # type: ignore[arg-type]
            specs={
                "writer": ToolSpec(
                    "writer", GraphArgs, "writer", retries=3, idempotent=False
                )
            },
        )
        r = guard.call("writer", kp_name="死锁")
        assert not r.ok and r.error_type == "internal" and calls["n"] == 1

    def test_trace_audit_fields(self):
        t = AgentTools()
        t.reset()
        t.call("calculate", expr="1+1", round_no=2)
        call = t.trace.calls[-1]
        assert call.ok and call.round == 2
        assert len(call.args_digest) == 12
        assert call.validated is True
        assert call.to_dict()["error_type"] == ""

    def test_failures_are_counted(self):
        t = AgentTools()
        t.call("get_exam_stats", kp_name="")
        assert t.failures.get("get_exam_stats") == 1
        assert len(t.trace.failure_details()) == 1


# ---------------------------------------------------------------------------
# 中枢工具：generate_plan / read_profile / record_checkin / judge_quiz
# store 单例全部替换到 tmp_path，不碰真实数据目录。
# ---------------------------------------------------------------------------


class TestHubContracts:
    def test_hub_tools_registered(self):
        for name in ("generate_plan", "read_profile", "record_checkin", "judge_quiz"):
            assert name in TOOL_REGISTRY and name in TOOL_SPECS

    def test_write_tools_are_not_idempotent(self):
        # 写操作绝不自动重试，重试会产生重复事件
        assert TOOL_SPECS["record_checkin"].idempotent is False
        assert TOOL_SPECS["judge_quiz"].idempotent is False
        assert TOOL_SPECS["generate_plan"].idempotent is True
        assert TOOL_SPECS["read_profile"].idempotent is True

    def test_client_id_not_in_args_models(self):
        # LLM 不能通过参数冒充别人：client_id 只能由调用方绑定
        for name in ("generate_plan", "read_profile", "record_checkin", "judge_quiz"):
            assert "client_id" not in TOOL_SPECS[name].args_model.model_fields

    def test_judge_quiz_choice_validation(self):
        t = AgentTools()
        r = t.call("judge_quiz", qid="2009-04", choice="E")
        assert not r.ok and r.error_type == "validation"

    def test_record_checkin_week_bounds(self):
        t = AgentTools()
        r = t.call("record_checkin", week=0)
        assert not r.ok and r.error_type == "validation"

    def test_generate_plan_hours_bounds(self):
        t = AgentTools()
        r = t.call("generate_plan", daily_hours=99)
        assert not r.ok and r.error_type == "validation"


class TestHubTools:
    """行为测试：tmp store + 合成 analyzer，离线可跑。"""

    @staticmethod
    def _patch_stores(monkeypatch, tmp_path):
        from cs408rag.profile.store import ProfileStore

        pstore = ProfileStore(tmp_path / "p.db")
        monkeypatch.setattr(
            "cs408rag.profile.store.get_profile_store", lambda: pstore
        )
        return pstore

    @staticmethod
    def _analyzer():
        from cs408rag.agent.exam_analysis import ExamAnalyzer
        from cs408rag.ingestion.quiz_loader import Question

        def q(year, num, subject, tags):
            return Question(
                year=year, num=num, subject=subject, stem=f"题{num}",
                options={"A": "x"}, answer="A", tags=tags,
            )

        return ExamAnalyzer([
            q(2020, 1, "操作系统", ["死锁概念"]),
            q(2021, 2, "操作系统", ["死锁概念", "页面置换"]),
            q(2022, 3, "操作系统", ["页面置换"]),
            q(2020, 4, "组成原理", ["组相联映射"]),
            q(2021, 5, "数据结构", ["排序算法"]),
        ])

    def test_read_profile_empty(self, monkeypatch, tmp_path):
        self._patch_stores(monkeypatch, tmp_path)
        t = AgentTools(analyzer=self._analyzer())
        t.set_client("u1")
        r = t.call("read_profile")
        assert r.ok
        assert r.data["client_id"] == "u1"
        assert r.data["total_questions"] == 0
        assert r.data["quiz"]["total"] == 0

    def test_record_checkin_then_read_profile(self, monkeypatch, tmp_path):
        pstore = self._patch_stores(monkeypatch, tmp_path)
        t = AgentTools(analyzer=self._analyzer())
        t.set_client("u1")
        r = t.call("record_checkin", week=3, idx=1, done=True, subject="操作系统")
        assert r.ok and r.data["recorded"] is True and r.data["client_id"] == "u1"
        events = pstore.events("u1", kind="checkin")
        assert len(events) == 1 and events[0]["payload"]["week"] == 3
        # Agent 打卡不带 total：不参与完成率分母
        assert "total" not in events[0]["payload"]
        # 他人命名空间看不到
        assert pstore.events("u2") == []

    def test_judge_quiz_writes_notebook_and_profile(self, monkeypatch, tmp_path):
        pstore = self._patch_stores(monkeypatch, tmp_path)
        from cs408rag.notebook.store import NotebookStore

        nstore = NotebookStore(tmp_path / "n.db")
        monkeypatch.setattr("cs408rag.notebook.judge.get_store", lambda: nstore)
        t = AgentTools(analyzer=self._analyzer())
        t.set_client("u1")
        r = t.call("judge_quiz", qid="2009-04", choice="A")
        assert r.ok and r.data["known"] is True
        quiz_events = pstore.events("u1", kind="quiz")
        assert len(quiz_events) == 1
        assert quiz_events[0]["payload"]["qid"] == "2009-04"
        # 判错进错题本、判对不在——以 ground truth 为准，不断言对错本身
        truth = r.data["truth"]
        if truth == "A":
            assert r.data["correct"] is True
        else:
            assert r.data["correct"] is False and nstore.get("u1", "2009-04")

    def test_judge_quiz_unknown_qid_no_write(self, monkeypatch, tmp_path):
        pstore = self._patch_stores(monkeypatch, tmp_path)
        t = AgentTools(analyzer=self._analyzer())
        r = t.call("judge_quiz", qid="2099-99", choice="A")
        assert r.ok and r.data["known"] is False
        assert pstore.events("default") == []

    def test_generate_plan_infers_weak_from_profile(self, monkeypatch, tmp_path):
        pstore = self._patch_stores(monkeypatch, tmp_path)
        # 画像：操作系统答错两次 -> 薄弱科目应被推断为操作系统
        for _ in range(2):
            pstore.record("u1", "quiz", {
                "qid": "2021-02", "subject": "操作系统",
                "tags": ["死锁概念"], "correct": False, "choice": "B",
            })
        t = AgentTools(analyzer=self._analyzer())
        t.set_client("u1")
        r = t.call("generate_plan")
        assert r.ok, r.error
        assert r.data["weak_subjects"] == ["操作系统"]
        assert r.data["weak_from_profile"] is True
        assert r.data["weeks"] > 0 and len(r.data["stages"]) >= 1
        assert "死锁概念" in r.data["boosted_by_profile"]

    def test_generate_plan_explicit_args_win(self, monkeypatch, tmp_path):
        self._patch_stores(monkeypatch, tmp_path)
        t = AgentTools(analyzer=self._analyzer())
        r = t.call(
            "generate_plan",
            weak_subjects=["组成原理", "野科目"],  # 非法科目被过滤
            daily_hours=6,
            target_score=120,
            progress={"组成原理": "一轮完", "数据结构": "瞎写"},  # 非法进度被过滤
        )
        assert r.ok, r.error
        assert r.data["weak_subjects"] == ["组成原理"]
        assert r.data["weak_from_profile"] is False
        assert r.data["daily_hours"] == 6 and r.data["target_score"] == 120


class TestPlanRoute:
    """plan 意图的路由与规则 planner。不需要 LLM/检索。"""

    @staticmethod
    def _nodes():
        from cs408rag.agent.nodes import AgentNodes

        return AgentNodes(AgentTools())

    def test_plan_intent_routes_to_plan(self):
        n = self._nodes()
        for q in ("帮我规划复习", "我的画像", "看看错题本", "帮我打卡"):
            assert n.route({"question": q})["route"] == "plan", q

    def test_exam_intent_not_stolen(self):
        n = self._nodes()
        r = n.route({"question": "死锁考过几次"})
        assert r["route"] == "exam"

    def test_plan_route_skips_retrieval(self):
        from cs408rag.agent.graph import _after_route

        assert _after_route({"route": "plan", "needs_clarification": False}) == "plan"
        assert _after_route({"route": "exam", "needs_clarification": False}) == "retrieve"

    def test_rule_plan_order_and_arg_extraction(self):
        n = self._nodes()
        st = {"route": "plan", "question": "帮我规划复习，每天 6 小时，目标 120 分",
              "observations": []}
        a1 = n._rule_plan(st)
        assert a1["action"] == "read_profile" and a1["args"] == {}
        st["observations"] = [{"tool": "read_profile", "ok": True}]
        a2 = n._rule_plan(st)
        assert a2["action"] == "generate_plan"
        assert a2["args"]["daily_hours"] == 6 and a2["args"]["target_score"] == 120
        st["observations"].append({"tool": "generate_plan", "ok": True})
        assert n._rule_plan(st) is None

    def test_plan_answer_deterministic_render(self):
        from cs408rag.agent.nodes import AgentNodes

        evidence = {
            "plan_result": {
                "weeks": 12, "start_date": "2026-11-06", "exam_date": "2027-01-30",
                "total_hours": 288, "daily_hours": 4, "target_score": 110,
                "weak_subjects": ["操作系统"], "weak_from_profile": True,
                "boosted_by_profile": ["死锁概念"],
                "stages": [{"name": "基础", "weeks": 4, "start": "2026-11-06",
                            "end": "2026-12-04", "focus": ["操作系统"],
                            "checkpoint": "能默写死锁四条件"}],
                "priority_kps": [{"subject": "操作系统", "tag": "死锁概念",
                                  "frequency": 6, "roi_label": "高"}],
                "guide_overview": "概览文本",
                "risk_notes": ["ROI 是启发式排序，不是押题"],
            }
        }
        text = AgentNodes._plan_answer(evidence)
        assert text is not None
        assert "12 周备考规划" in text and "死锁概念" in text
        assert "我的规划执行" in text  # 引导去页内规划看板
        assert AgentNodes._plan_answer({}) is None
        # 纯画像查询
        text2 = AgentNodes._plan_answer({"profile": {
            "total_questions": 3, "quiz": {"total": 2, "correct": 1},
            "execution": {"done": 1, "tasks_total": 4},
            "top_tags": [], "boost_tags": [],
        }})
        assert text2 and "备考画像" in text2