"""不依赖本地数据/模型的纯函数单测，供 CI 跑。

完整测试（规则评测、真实题库、Qdrant）需要 data/processed 与本地模型，
不进公开仓库；CI 只跑这里，避免把版权数据和数 GB 模型带上去。
"""

from __future__ import annotations

from datetime import date, timedelta

from cs408rag.adversarial.planner import (
    _phase_shares,
    allocate_subject_hours,
    compute_time_budget,
    next_exam_date,
    normalize_concerns,
)
from cs408rag.adversarial.roles import _ref_exists
from cs408rag.adversarial.schema import plan_diff, validate_plan
from cs408rag.eval.adversarial_eval import summarize_runs


def _plan() -> dict:
    return {
        "goal": {
            "scope": "full_408",
            "weak_subjects": ["操作系统"],
            "remaining_weeks": 1,
            "weekly_hours": 10,
            "daily_hours": 2.5,
            "study_days_per_week": 4,
            "progress": {"操作系统": "一轮中"},
            "concerns": [],
        },
        "stages": [
            {"name": "强化", "start": "2026-01-01", "end": "2026-01-07", "weeks": 1}
        ],
        "priority_kps": [
            {
                "kp": "死锁",
                "tag": "死锁",
                "subject": "操作系统",
                "weight": 0.2,
                "evidence": ["2019#32"],
            }
        ],
        "weekly_plan": [
            {
                "week": 1,
                "stage": "强化",
                "focus": ["操作系统"],
                "tasks": [
                    {
                        "subject": "操作系统",
                        "hours": 10,
                        "content": "死锁",
                        "method": "先选择后大题",
                        "acceptance": "选择题 ≥80%",
                    }
                ],
            }
        ],
        "concern_actions": [],
    }


def test_next_exam_date():
    assert next_exam_date(date(2026, 10, 29)) == date(2026, 12, 19)
    assert next_exam_date(date(2026, 12, 20)) == date(2027, 12, 25)


def test_time_budget():
    goal = {
        "start_date": "2026-10-29",
        "exam_date": "2026-12-19",
        "daily_hours": 4,
        "study_days_per_week": 6,
    }
    b = compute_time_budget(goal)
    assert b["remaining_days"] == 51
    assert b["remaining_weeks"] == 8
    assert b["weekly_hours"] == 24
    assert b["total_hours"] == 192


def test_phase_shares_sum_to_one():
    for weeks in (4, 8, 16, 30):
        assert abs(sum(_phase_shares(weeks).values()) - 1.0) < 1e-9


def test_subject_shares_sum_and_weak_floor():
    goal = {
        "scope": "full_408",
        "weak_subjects": ["操作系统"],
        "progress": {"操作系统": "一轮中"},
    }
    shares = allocate_subject_hours(goal, {"boost_tags": []})
    assert abs(sum(shares.values()) - 1.0) < 0.01
    assert shares["操作系统"] >= 0.30


def test_normalize_concerns():
    assert normalize_concerns("基础差\n真题低；基础差") == ["基础差", "真题低"]


def test_validate_plan_ok_and_rejects_missing_concern_action():
    plan = _plan()
    assert validate_plan(plan) == []

    bad = _plan()
    bad["goal"]["concerns"] = ["时间不够"]
    assert validate_plan(bad)


def test_ref_exists_supports_stages_and_concerns():
    plan = _plan()
    plan["concern_actions"] = [{"concern": "x", "action": "y"}]
    assert _ref_exists(plan, "stages[0].weeks")
    assert _ref_exists(plan, "concern_actions[0].action")
    assert not _ref_exists(plan, "stages[9].weeks")


def test_plan_diff_marks_change():
    old = _plan()
    new = _plan()
    new["weekly_plan"][0]["tasks"][0]["hours"] = 12
    assert "plan_v_prev" in plan_diff(old, new)


def test_summarize_runs_counterfactual():
    rows = [
        {
            "arm": "single_round",
            "subject": "操作系统",
            "stop_reason": "max_rounds",
            "version": 1,
            "residual_high": 2,
            "llm_calls": 2,
            "tokens": {"prompt": 100, "completion": 50},
            "took_ms": 1000,
            "n_llm_challenges": 2,
            "guide_source": "template",
        },
        {
            "arm": "two_round",
            "subject": "操作系统",
            "stop_reason": "converged",
            "version": 3,
            "residual_high": 0,
            "llm_calls": 4,
            "tokens": {"prompt": 200, "completion": 80},
            "took_ms": 3000,
            "n_llm_challenges": 0,
            "guide_source": "llm",
        },
    ]
    out = summarize_runs(rows)
    assert out["n_goals"] == 1
    assert out["arms"]["two_round"]["avg_residual_high"] == 0
    assert out["paired"][0]["two_versions"] == 3


def test_exam_date_window_is_positive():
    b = compute_time_budget(
        {
            "start_date": "2026-01-01",
            "exam_date": (date(2026, 1, 1) + timedelta(weeks=30)).isoformat(),
            "daily_hours": 3,
            "study_days_per_week": 5,
        }
    )
    assert b["remaining_weeks"] == 30