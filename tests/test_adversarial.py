"""对抗 Plan 的规则与 schema 单测。"""

from __future__ import annotations

import json

import pytest

from cs408rag.adversarial.dataset import build_goldens, build_injected
from cs408rag.adversarial.plan import ERROR_TYPES, build_golden_plan, inject
from cs408rag.adversarial.rules import review_plan
from cs408rag.adversarial.schema import plan_diff, plan_digest, validate_plan


@pytest.fixture(scope="module")
def golden():
    return build_golden_plan(weak_subject="操作系统")


class TestGoldenPlan:
    def test_no_challenges(self, golden):
        """黄金 Plan 必须 0 挑战，否则后续误报率没有意义。"""
        assert review_plan(golden) == []

    def test_all_subjects_clean(self):
        for subject in ("数据结构", "组成原理", "操作系统", "计算机网络"):
            plan = build_golden_plan(weak_subject=subject)
            assert review_plan(plan) == [], subject


class TestInjectionDetection:
    @pytest.mark.parametrize("error_type", ERROR_TYPES)
    def test_each_error_detected(self, golden, error_type):
        plan = inject(golden, error_type)
        rules = {c.rule for c in review_plan(plan)}
        assert error_type in rules

    def test_injected_dataset_covers_all_types(self):
        rows = build_injected(build_goldens(), n=20)
        labels = {label for row in rows for label in row["labels"]}
        assert labels == set(ERROR_TYPES)

    def test_injected_never_overwrites_each_other(self):
        """注入组合不应互相覆盖，否则检出率会被低估。"""
        for row in build_injected(build_goldens(), n=20):
            rules = {c.rule for c in review_plan(row["plan"])}
            assert set(row["labels"]) <= rules


class _FakePlanner:
    def __init__(self, plan, used: bool = True):
        self.plan = plan
        self.used = used

    def generate(self, goal, *, prev_plan=None, challenges=None):
        return self.plan, self.used


class _FakeCritic:
    def review(self, plan, rules):
        return [], True


class _FakeExpert:
    def arbitrate(self, conflicts):
        return []


class _HighCritic:
    """始终提出一条 high 主观挑战，用于验证退出判定不会忽略 LLM 挑战。"""

    def review(self, plan, rules):
        return (
            [
                {
                    "rule": "llm_critic",
                    "severity": "high",
                    "message": "目标分与周学时不相称",
                    "evidence_ref": "goal.target_score",
                    "locator": "goal.target_score",
                }
            ],
            True,
        )


class _OnceCritic:
    """只在第一轮提一条 high，模拟「修订后问题被解决」。"""

    def __init__(self):
        self.calls = 0

    def review(self, plan, rules):
        self.calls += 1
        if self.calls == 1:
            return (
                [
                    {
                        "rule": "llm_critic",
                        "severity": "high",
                        "message": "目标可达性存疑",
                        "evidence_ref": "goal.target_score",
                        "locator": "goal.target_score",
                    }
                ],
                True,
            )
        return [], True


class TestEvidenceRef:
    def test_accepts_real_paths(self, golden):
        from cs408rag.adversarial.roles import _ref_exists

        assert _ref_exists(golden, "goal.target_score")
        assert _ref_exists(golden, "priority_kps[0].weight")
        assert _ref_exists(golden, "weekly_plan[0].tasks[1].hours")

    def test_rejects_made_up_paths(self, golden):
        from cs408rag.adversarial.roles import _ref_exists

        assert not _ref_exists(golden, "goal.score")
        assert not _ref_exists(golden, "priority_kps[99].weight")
        assert not _ref_exists(golden, "weekly_plan[0].tasks[99].hours")
        assert not _ref_exists(golden, "free text")


class TestStateMachine:
    def _runner(self, plan, *, used=True, use_llm=True, max_rounds=2, critic=None):
        from cs408rag.adversarial.graph import PlanNodes, PlanRunner
        from cs408rag.agent.exam_analysis import get_analyzer

        nodes = PlanNodes(
            planner=_FakePlanner(plan, used),
            critic=critic or _FakeCritic(),
            expert=_FakeExpert(),
            analyzer=get_analyzer(),
        )
        return PlanRunner(nodes=nodes, use_llm=use_llm, max_rounds=max_rounds)

    def test_converges_when_no_high(self, golden):
        r = self._runner(golden).run({"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10})
        assert r.stop_reason == "converged"
        assert r.version == 1

    def test_stalls_when_revision_does_not_change(self, golden):
        bad = inject(golden, "time_overrun")
        r = self._runner(bad).run({"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10})
        assert r.stop_reason in ("stalled", "max_rounds")
        assert r.open_high

    def test_llm_guide_failure_does_not_degrade(self, golden):
        """结构由确定性生成器保证；LLM 文案失败不应把 Plan 判成 degraded。"""
        r = self._runner(golden, used=False).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert r.stop_reason == "converged"
        assert r.plan_source == "deterministic"
        assert r.planner_fallback is False

    def test_no_llm_single_pass(self, golden):
        r = self._runner(golden, use_llm=False).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert r.stop_reason == "converged"
        assert r.llm_calls == 0

    def test_llm_high_is_not_ignored(self, golden):
        """LLM 挑战没有 status 字段时，不能被当成已关闭。"""
        r = self._runner(golden, critic=_HighCritic(), max_rounds=0).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert r.stop_reason == "max_rounds"
        assert len(r.open_high) == 1
        assert r.open_high[0]["rule"] == "llm_critic"

    def test_stalls_when_revision_changes_nothing(self, golden):
        r = self._runner(golden, critic=_HighCritic(), max_rounds=2).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert r.stop_reason == "stalled"
        assert r.version == 2
        assert len(r.plan_history) == 2

    def test_resolved_challenge_stays_in_history(self, golden):
        """收敛后挑战表不能变空：已解决的挑战必须留在轨迹里。"""
        r = self._runner(golden, critic=_OnceCritic(), max_rounds=2).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert r.stop_reason == "converged"
        assert r.version == 2
        assert r.open_high == []
        assert len(r.challenge_history) == 1
        assert r.challenge_history[0]["status"] == "resolved"

    def test_plan_history_matches_version(self, golden):
        r = self._runner(golden).run(
            {"weak_subjects": ["操作系统"], "weeks": 12, "weekly_hours": 10}
        )
        assert len(r.plan_history) == r.version

    def test_validate_ok(self, golden):
        assert validate_plan(golden) == []

    @pytest.mark.parametrize(
        "mutate",
        [
            lambda p: p.pop("goal"),
            lambda p: p["goal"].update({"weeks": -1}),
            lambda p: p.update({"priority_kps": "bad"}),
            lambda p: p["weekly_plan"][0]["tasks"][0].update({"hours": -2}),
        ],
    )
    def test_validate_rejects(self, golden, mutate):
        import copy

        p = copy.deepcopy(golden)
        mutate(p)
        assert validate_plan(p)

    def test_digest_truncates(self, golden):
        d = plan_digest(golden, max_chars=200)
        assert len(d) <= 200

    def test_digest_marks_hidden_weeks(self):
        """摘要只给前 4 周，必须说明后面还有几周，否则 Critic 会误判漏排。"""
        plan = build_golden_plan(weak_subject="操作系统", weeks=12)
        d = plan_digest(plan)
        assert "未展示" in d
        assert "另有 8 周未展示" in d

    def test_diff_marks_change(self, golden):
        changed = inject(golden, "time_overrun")
        diff = plan_diff(golden, changed)
        assert "plan_v_prev" in diff and "plan_v_new" in diff


class TestPlanner:
    def _goal(self, **overrides):
        base = {
            "scope": "full_408",
            "weak_subjects": ["操作系统"],
            "start_date": "2026-10-29",
            "exam_date": "2026-12-19",
            "daily_hours": 4,
            "study_days_per_week": 6,
            "target_score": 115,
            "progress": {
                "数据结构": "一轮完",
                "组成原理": "一轮中",
                "操作系统": "一轮中",
                "计算机网络": "未开始",
            },
            "concerns": [],
        }
        base.update(overrides)
        return base

    def test_next_exam_date(self):
        from datetime import date

        from cs408rag.adversarial.planner import next_exam_date

        assert next_exam_date(date(2026, 10, 29)) == date(2026, 12, 19)
        assert next_exam_date(date(2026, 12, 20)) == date(2027, 12, 25)

    def test_time_budget(self):
        from cs408rag.adversarial.planner import compute_time_budget

        b = compute_time_budget(self._goal())
        assert b["remaining_days"] == 51
        assert b["remaining_weeks"] == 8
        assert b["weekly_hours"] == 24
        assert b["total_hours"] == 192

    def test_phases_are_contiguous_and_sum(self):
        from datetime import date, timedelta

        from cs408rag.adversarial.planner import build_detailed_plan

        p = build_detailed_plan(self._goal())
        stages = p["stages"]
        assert sum(s["weeks"] for s in stages) == p["goal"]["remaining_weeks"]
        for a, b in zip(stages, stages[1:], strict=False):
            assert date.fromisoformat(a["end"]) + timedelta(days=1) == date.fromisoformat(
                b["start"]
            )
        assert stages[-1]["end"] == p["goal"]["exam_date"]

    def test_weak_subject_gets_enough_time(self):
        from cs408rag.adversarial.planner import build_detailed_plan

        for weak in (["操作系统"], ["计算机网络"], ["数据结构", "计算机网络"]):
            p = build_detailed_plan(self._goal(weak_subjects=weak))
            total = sum(t["hours"] for w in p["weekly_plan"] for t in w["tasks"])
            shares = {
                s: sum(
                    t["hours"]
                    for w in p["weekly_plan"]
                    for t in w["tasks"]
                    if t["subject"] == s
                )
                / total
                for s in weak
            }
            assert all(v >= 0.29 for v in shares.values()), (weak, shares)

    def test_single_subject_mode(self):
        from cs408rag.adversarial.planner import build_detailed_plan

        p = build_detailed_plan(self._goal(scope="single_subject"))
        total = sum(t["hours"] for w in p["weekly_plan"] for t in w["tasks"])
        os_hours = sum(
            t["hours"]
            for w in p["weekly_plan"]
            for t in w["tasks"]
            if t["subject"] == "操作系统"
        )
        assert os_hours / total >= 0.60

    def test_weekly_tasks_differ_within_phase(self):
        """同一阶段的每周任务必须有差异：真题分年份区间、冲刺分模考套数。

        否则周计划表会整段重复，用户看到「怎么都一样」。
        """
        from cs408rag.adversarial.planner import build_detailed_plan

        p = build_detailed_plan(self._goal())
        drill: dict[str, list[str]] = {}
        mock: dict[str, list[str]] = {}
        for w in p["weekly_plan"]:
            stage = w.get("stage")
            for t in w["tasks"]:
                if stage == "真题":
                    drill.setdefault(t["subject"], []).append(t["content"])
                elif stage == "冲刺":
                    mock.setdefault(t["subject"], []).append(t["content"])
        for subject, contents in drill.items():
            if len(contents) > 1:
                assert len(set(contents)) == len(contents), (subject, contents)
        for subject, contents in mock.items():
            if len(contents) > 1:
                assert len(set(contents)) == len(contents), (subject, contents)

    def test_concern_routing_changes_plan(self):
        from cs408rag.adversarial.planner import build_detailed_plan

        base = build_detailed_plan(self._goal())
        timed = build_detailed_plan(self._goal(concerns=["时间不够"]))
        assert len(timed["priority_kps"]) < len(base["priority_kps"])

        advanced = {
            s: "一轮完"
            for s in ("数据结构", "组成原理", "操作系统", "计算机网络")
        }
        basics = build_detailed_plan(
            self._goal(progress=advanced, concerns=["基础差"])
        )
        assert "基础" in [s["name"] for s in basics["stages"]]

        tag = build_detailed_plan(self._goal(concerns=["死锁总是错"]))
        os_tags = [k["tag"] for k in tag["priority_kps"] if k["subject"] == "操作系统"]
        assert any("死锁" in t for t in os_tags[:3])

        for p in (base, timed, basics, tag):
            assert len(p["concern_actions"]) == len(p["goal"]["concerns"])
            assert validate_plan(p) == []

    def test_roi_selects_top_frequency(self):
        from cs408rag.adversarial.roi import (
            score_subject_tags,
            select_high_value,
            subject_coverage,
        )
        from cs408rag.agent.exam_analysis import get_analyzer

        a = get_analyzer()
        rows = select_high_value(a, "操作系统", remaining_weeks=8)
        assert len(rows) == 10
        top5 = {
            r.tag
            for r in sorted(
                score_subject_tags(a, "操作系统"),
                key=lambda x: (-x.frequency, x.tag),
            )[:5]
        }
        assert top5 <= {r.tag for r in rows}
        assert subject_coverage(a, "操作系统", rows)["ratio"] > 0.3

    def test_plan_content_is_specific(self):
        """任务要写清「学什么/怎么学/验收」，guide 要有一轮/错题等方法。"""
        from cs408rag.adversarial.plan import build_golden_plan

        p = build_golden_plan(weak_subject="操作系统", weeks=12)
        task = p["weekly_plan"][0]["tasks"][0]
        assert task["content"] and task["method"] and task["acceptance"]
        assert "一轮" in task["method"] or "二轮" in task["method"]
        assert p["weekly_plan"][0]["focus"]
        assert task["content"].startswith("本周主攻")
        guide_text = json.dumps(p["guide"], ensure_ascii=False)
        assert "一轮" in guide_text and "错题" in guide_text
        assert "数据结构" in p["guide"]["subject_order"]
        assert "网络" in p["guide"]["subject_order"]
        assert not any(
            term in guide_text for term in ("王道", "沈悦", "网课", "老师", "教材")
        )