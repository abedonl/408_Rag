"""任务型 harness 的离线测试：stub 工具与考点库，不依赖真实数据/LLM。"""

from __future__ import annotations

from typing import Any

from cs408rag.agent.contracts import ToolResult
from cs408rag.agent.harness import TaskHarness, resolve_tag


class _StubAnalyzer:
    tag_freq = {"银行家算法": 8, "死锁": 12, "Cache": 20}


class _StubTools:
    """模拟 AgentTools.call：按工具名返回固定数据；fail_tools 里的名字返回失败。"""

    def __init__(self, fail_tools: set[str] | None = None) -> None:
        self.fail_tools = fail_tools or set()
        self.calls: list[str] = []

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        self.calls.append(name)
        if name in self.fail_tools:
            return ToolResult(tool=name, ok=False, error="boom", error_type="internal")
        data: Any = {}
        if name == "get_exam_stats":
            data = {"total": 8, "matched_tags": ["银行家算法"],
                    "by_year": {"2021": 2, "2023": 1}, "by_type": {"calc": 3}}
        elif name == "get_kp_questions":
            data = {"questions": [
                {"qid": "2021-01", "subject": "操作系统", "type": "calc",
                 "year": 2021, "num": 1, "question": "题干",
                 "options": {"A": "x", "B": "y"}, "answer": "A",
                 "explanation": "解析"},
            ]}
        elif name == "get_related_kps":
            data = {"cooccurring_tags": [{"tag": "死锁", "cooccurrence": 3}]}
        elif name == "get_learning_path":
            data = {"found": True, "chain": [{"title": "进程"}, {"title": "死锁"}]}
        elif name == "search_knowledge_base":
            data = {"hits": [{"citation": "教材·OS·3.4", "text": "银行家算法是…"}]}
        elif name == "verify_answer":
            data = {"ok": True, "invalid_exam_refs": [], "predictions": []}
        return ToolResult(tool=name, ok=True, data=data, summary=f"{name} ok")


def test_resolve_tag_longest_match() -> None:
    assert resolve_tag("帮我攻克银行家算法", _StubAnalyzer()) == "银行家算法"
    assert resolve_tag("死锁怎么办", _StubAnalyzer()) == "死锁"
    assert resolve_tag("随便聊聊", _StubAnalyzer()) == ""


def test_run_full_plan_template_briefing() -> None:
    tools = _StubTools()
    h = TaskHarness(tools, analyzer=_StubAnalyzer())  # type: ignore[arg-type]
    events: list[dict] = []
    rep = h.run("攻克 银行家算法", on_event=events.append)
    assert rep.ok and rep.stopped == "done"
    assert rep.tag == "银行家算法"
    # 计划事件先行，步骤结束事件带 summary
    assert events[0]["type"] == "plan"
    assert events[-1]["type"] == "done" and events[-1]["stopped"] == "done"
    # 讲义由模板合成，包含考情与前置链
    assert "历年考查 8 次" in rep.briefing
    assert "进程 → 死锁" in rep.briefing
    assert "模板合成" in next(s["summary"] for s in rep.steps if s["key"] == "synthesize")
    # 自测题带答案且进了 quiz
    assert len(rep.quiz) == 1 and rep.quiz[0]["qid"] == "2021-01"
    assert rep.verification.get("ok")


def test_required_failure_stops_task() -> None:
    tools = _StubTools(fail_tools={"get_exam_stats"})
    h = TaskHarness(tools, analyzer=_StubAnalyzer())  # type: ignore[arg-type]
    rep = h.run("攻克 银行家算法")
    assert rep.stopped == "required_failed:stats"
    assert not rep.ok
    # 中断后不再调后续工具
    assert "get_kp_questions" not in tools.calls


def test_optional_failure_degrades_continues() -> None:
    tools = _StubTools(fail_tools={"search_knowledge_base", "get_learning_path"})
    h = TaskHarness(tools, analyzer=_StubAnalyzer())  # type: ignore[arg-type]
    rep = h.run("攻克 银行家算法")
    assert rep.stopped == "done" and rep.ok
    kb_step = next(s for s in rep.steps if s["key"] == "kb")
    assert not kb_step["ok"]
    # 讲义仍在，只是没有教材段落
    assert "历年考查 8 次" in rep.briefing


def test_budget_stops_and_marks() -> None:
    tools = _StubTools()
    h = TaskHarness(tools, analyzer=_StubAnalyzer(), max_steps=2)  # type: ignore[arg-type]
    rep = h.run("攻克 银行家算法")
    assert rep.stopped == "budget"
    assert len(rep.steps) == 2
    assert not rep.ok  # 讲义没生成，任务不算完成


def test_tag_not_found() -> None:
    h = TaskHarness(_StubTools(), analyzer=_StubAnalyzer())  # type: ignore[arg-type]
    rep = h.run("帮我减肥")
    assert rep.stopped == "tag_not_found" and not rep.ok
