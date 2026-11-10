"""human-in-the-loop 两处改动的测试。

1. 执行跟踪聚合（build_profile 的 checkin/checkpoint，latest-wins）
2. PlanRunner.iter_resume：人工采纳的挑战喂回 planner，且只多修一版
"""

from __future__ import annotations

from cs408rag.adversarial.graph import PlanNodes, PlanRunner
from cs408rag.adversarial.planner import build_detailed_plan
from cs408rag.agent.exam_analysis import ExamAnalyzer
from cs408rag.ingestion.quiz_loader import Question

# ---------------------------------------------------------------------------
# iter_resume：人工裁定后续修一版
# ---------------------------------------------------------------------------


class _RecPlanner:
    """记录 generate 收到的 prev_plan/challenges，固定返回同一 plan。"""

    def __init__(self, plan: dict):
        self.plan = plan
        self.calls: list[dict] = []

    def generate(self, goal, *, prev_plan=None, challenges=None):
        self.calls.append({"prev_plan": prev_plan, "challenges": challenges})
        return self.plan, True


class _QuietCritic:
    def review(self, plan, rules):
        return [], True


class _NoopExpert:
    def arbitrate(self, conflicts):
        return []


def _q(year: int, num: int, subject: str, tags: list[str]) -> Question:
    return Question(
        year=year, num=num, subject=subject,
        stem=f"{year}-{num}", options={}, answer="A", tags=tags,
    )


def _analyzer() -> ExamAnalyzer:
    return ExamAnalyzer(
        [
            _q(2023, 1, "数据结构", ["数组", "线性表"]),
            _q(2022, 2, "数据结构", ["链表", "线性表"]),
            _q(2021, 1, "操作系统", ["进程管理", "死锁概念"]),
            _q(2020, 3, "操作系统", ["死锁概念", "死锁处理策略"]),
        ]
    )


class TestIterResume:
    def test_resume_feeds_accepted_and_stops_after_one_round(self, tmp_path):
        analyzer = _analyzer()
        goal = {
            "weak_subjects": ["数据结构"],
            "weeks": 12,
            "weekly_hours": 10,
            "target_score": 100,
        }
        plan = build_detailed_plan(goal, analyzer)
        planner = _RecPlanner(plan)
        nodes = PlanNodes(
            planner=planner, critic=_QuietCritic(), expert=_NoopExpert(), analyzer=analyzer
        )
        runner = PlanRunner(nodes=nodes, use_llm=True, max_rounds=0)

        accepted = [
            {
                "id": "lc|abc",
                "rule": "llm_critic",
                "severity": "high",
                "message": "复习顺序不合理",
                "status": "open",
                "verdict": "human_accept",
            }
        ]
        dropped = {
            "id": "lc|old",
            "rule": "llm_critic",
            "severity": "low",
            "message": "已被忽略",
            "status": "dropped",
            "verdict": "human_ignore",
        }
        prev = {
            "goal": goal,
            "plan": plan,
            "plan_history": [plan],
            "version": 1,
            "round": 1,
            "challenge_history": [dropped],
        }

        last: dict = {}
        for snap in runner.iter_resume(prev, accepted):
            last = snap

        # 采纳的挑战原样到达 planner，prev_plan 是上一版
        call = planner.calls[-1]
        assert call["prev_plan"] == plan
        assert [c["id"] for c in call["challenges"]] == ["lc|abc"]
        # 只多修一版：version +1，round 推过 seeded max_rounds 即停
        assert last["version"] == 2
        assert len(last["plan_history"]) == 2
        assert last["round"] == 2
        assert last["stop_reason"] in ("converged", "max_rounds")
        # 人工忽略的挑战保留在历史里
        ids = [c["id"] for c in last["challenge_history"]]
        assert "lc|old" in ids

    def test_resume_with_empty_accepted_still_converges(self, tmp_path):
        """全部忽略时不该喂任何挑战（UI 层此时不会调用，这里兜底验证健壮性）。"""
        analyzer = _analyzer()
        goal = {"weak_subjects": ["数据结构"], "weeks": 12, "weekly_hours": 10}
        plan = build_detailed_plan(goal, analyzer)
        planner = _RecPlanner(plan)
        nodes = PlanNodes(
            planner=planner, critic=_QuietCritic(), expert=_NoopExpert(), analyzer=analyzer
        )
        runner = PlanRunner(nodes=nodes, use_llm=True, max_rounds=0)
        prev = {"goal": goal, "plan": plan, "version": 1, "round": 1}

        last: dict = {}
        for snap in runner.iter_resume(prev, []):
            last = snap

        assert planner.calls[-1]["challenges"] == []
        assert last["version"] == 2
