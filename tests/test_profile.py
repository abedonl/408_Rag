"""用户画像：事件存储、规则提取、聚合视图、planner 加权。"""

from __future__ import annotations

import pytest

from cs408rag.agent.exam_analysis import ExamAnalyzer
from cs408rag.ingestion.quiz_loader import Question
from cs408rag.profile.stats import (
    MAX_BOOST_TAGS,
    PLAN_TTL_DAYS,
    build_profile,
    extract_tags,
)
from cs408rag.profile.store import ProfileStore


def _q(qid_year: int, num: int, subject: str, tags: list[str]) -> Question:
    return Question(
        year=qid_year, num=num, subject=subject, stem=f"题{num}",
        options={"A": "x"}, answer="A", tags=tags,
    )


def _analyzer() -> ExamAnalyzer:
    qs = [
        _q(2020, 1, "操作系统", ["死锁概念"]),
        _q(2021, 2, "操作系统", ["死锁概念", "页面置换"]),
        _q(2022, 3, "操作系统", ["页面置换"]),
        _q(2020, 4, "组成原理", ["组相联映射"]),
        _q(2021, 5, "数据结构", ["排序算法"]),
    ]
    return ExamAnalyzer(qs)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------


class TestProfileStore:
    def test_roundtrip_and_kind_filter(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        store.record("c1", "question", {"text": "死锁是什么", "tags": []})
        store.record("c1", "plan", {"weak_subjects": ["操作系统"]})
        store.record("c2", "question", {"text": "别人的", "tags": []})

        events = store.events("c1")
        assert [e["kind"] for e in events] == ["question", "plan"]
        assert events[0]["payload"]["text"] == "死锁是什么"
        assert store.events("c1", kind="plan")[0]["payload"]["weak_subjects"] == [
            "操作系统"
        ]
        assert len(store.events("c2")) == 1

    def test_clear(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        store.record("c1", "question", {"text": "a"})
        assert store.clear("c1") == 1
        assert store.events("c1") == []


# ---------------------------------------------------------------------------
# extract_tags
# ---------------------------------------------------------------------------


class TestExtractTags:
    def test_full_tag_hit(self):
        # 有整标签命中时，2 字泛化片段（「映射」「地址」「划分」）被抑制
        hits = extract_tags("采用组相联映射时主存地址如何划分？", _analyzer())
        assert hits == [{"tag": "组相联映射", "subject": "组成原理"}]

    def test_fragment_hit(self):
        # 无更强命中时 2 字片段兜底：「死锁」命中「死锁概念」
        tags = [h["tag"] for h in extract_tags("死锁怎么预防？", _analyzer())]
        assert "死锁概念" in tags

    def test_short_fragment_rejected(self):
        # 与所有标签都没有 ≥2 字公共片段时不命中
        assert extract_tags("如何合理安排作息？", _analyzer()) == []

    def test_freq_order(self):
        # 两个整标签命中，按考查频次降序（页面置换 2 次 > 排序算法 1 次）
        hits = extract_tags("页面置换和排序算法都会考吗？", _analyzer())
        assert [h["tag"] for h in hits] == ["页面置换", "排序算法"]

    def test_full_hit_suppresses_weak_fragment(self):
        # 既定取舍：有整标签命中时，「死锁」这类 2 字弱片段让位，
        # 宁可漏掉混合提问里的次要考点，也不让泛化词（映射/地址/划分）进画像
        hits = extract_tags("死锁和页面置换都会考吗？", _analyzer())
        assert [h["tag"] for h in hits] == ["页面置换"]


# ---------------------------------------------------------------------------
# build_profile
# ---------------------------------------------------------------------------


def _q_event(text: str, tags: list[dict], subject: str = "") -> dict:
    return {
        "kind": "question",
        "payload": {"text": text, "subject": subject, "tags": tags},
        "ts": "2026-01-01T00:00:00+00:00",
    }


def _plan_event(payload: dict, ts: str | None = None) -> dict:
    """plan 事件默认用当前时间：写死时间戳会被 30 天时效过滤掉。"""
    from datetime import UTC, datetime

    return {
        "kind": "plan",
        "payload": payload,
        "ts": ts or datetime.now(UTC).isoformat(timespec="seconds"),
    }


class TestBuildProfile:
    def test_empty(self):
        p = build_profile([])
        assert p["total_questions"] == 0
        assert p["plan"] is None
        assert p["boost_tags"] == []

    def test_aggregation(self):
        events = [
            _q_event("死锁是什么", [{"tag": "死锁概念", "subject": "操作系统"}]),
            _q_event(
                "死锁怎么预防",
                [{"tag": "死锁概念", "subject": "操作系统"}],
                subject="操作系统",
            ),
            _q_event("组相联映射", [{"tag": "组相联映射", "subject": "组成原理"}]),
            _plan_event(
                {
                    "weak_subjects": ["操作系统"],
                    "priority_tags": [{"tag": "页面置换", "subject": "操作系统"}],
                }
            ),
        ]
        p = build_profile(events)
        assert p["total_questions"] == 3
        assert p["top_tags"][0] == {
            "tag": "死锁概念",
            "subject": "操作系统",
            "count": 2,
        }
        # 每题计一次：无显式科目时取首个考点的主科目
        assert p["subject_counts"] == {"操作系统": 2, "组成原理": 1}
        assert p["plan"]["weak_subjects"] == ["操作系统"]
        # boost：高频提问考点在前，规划重点补充
        boost_tags = [t["tag"] for t in p["boost_tags"]]
        assert boost_tags[0] == "死锁概念"
        assert "页面置换" in boost_tags

    def test_boost_cap(self):
        events = [
            _q_event(
                f"q{i}",
                [{"tag": f"tag{i}", "subject": "操作系统"}],
                subject="操作系统",
            )
            for i in range(20)
        ]
        events.append(
            _plan_event(
                {"priority_tags": [{"tag": "plan_tag", "subject": "操作系统"}]}
            )
        )
        p = build_profile(events)
        assert len(p["boost_tags"]) <= MAX_BOOST_TAGS
        assert "plan_tag" in [t["tag"] for t in p["boost_tags"]]  # 规划重点留有槽位

    def test_expired_plan_ignored(self):
        """超过 30 天的规划不再影响画像：避免旧目标永久加权。"""
        from datetime import UTC, datetime, timedelta

        old = (datetime.now(UTC) - timedelta(days=PLAN_TTL_DAYS + 1)).isoformat()
        events = [
            _q_event("死锁是什么", [{"tag": "死锁概念", "subject": "操作系统"}]),
            _plan_event(
                {
                    "weak_subjects": ["操作系统"],
                    "priority_tags": [{"tag": "过期考点", "subject": "操作系统"}],
                },
                ts=old,
            ),
        ]
        p = build_profile(events)
        assert p["plan"] is None
        assert "过期考点" not in [t["tag"] for t in p["boost_tags"]]
        assert "死锁概念" in [t["tag"] for t in p["boost_tags"]]  # 提问信号不受影响

    def test_recent_plan_kept(self):
        """29 天内的规划仍有效（边界内）。"""
        from datetime import UTC, datetime, timedelta

        recent = (datetime.now(UTC) - timedelta(days=PLAN_TTL_DAYS - 1)).isoformat()
        events = [
            _plan_event({"weak_subjects": ["操作系统"]}, ts=recent),
        ]
        assert build_profile(events)["plan"]["weak_subjects"] == ["操作系统"]

    def test_plan_with_bad_ts_ignored(self):
        """时间戳缺失/非法按过期处理，不让格式问题把数据当新数据用。"""
        events = [_plan_event({"weak_subjects": ["操作系统"]}, ts="not-a-date")]
        assert build_profile(events)["plan"] is None

    def test_quiz_aggregation(self):
        events = [
            _q_event("什么是死锁", [{"tag": "死锁概念", "subject": "操作系统"}]),
            {
                "kind": "quiz",
                "payload": {
                    "qid": "2021-02",
                    "subject": "操作系统",
                    "tags": ["页面置换"],
                    "correct": False,
                },
                "ts": "t1",
            },
            {
                "kind": "quiz",
                "payload": {
                    "qid": "2020-04",
                    "subject": "组成原理",
                    "tags": ["组相联映射"],
                    "correct": True,
                },
                "ts": "t2",
            },
        ]
        p = build_profile(events)
        assert p["quiz"]["total"] == 2
        assert p["quiz"]["correct"] == 1
        assert p["quiz"]["wrong_tags"] == [
            {"tag": "页面置换", "subject": "操作系统", "count": 1}
        ]
        assert p["quiz"]["wrong_subjects"] == {"操作系统": 1}
        # 答错考点是真正的薄弱信号，排在提问高频之前
        assert p["boost_tags"][0]["tag"] == "页面置换"

    def test_quiz_empty(self):
        p = build_profile([])
        assert p["quiz"] == {
            "total": 0,
            "correct": 0,
            "wrong_tags": [],
            "wrong_subjects": {},
        }


# ---------------------------------------------------------------------------
# planner 加权
# ---------------------------------------------------------------------------


class TestPlannerProfileBoost:
    def _goal(self, **kw):
        goal = {
            "scope": "full_408",
            "weak_subjects": ["操作系统"],
            "daily_hours": 4,
            "study_days_per_week": 6,
            "target_score": 110,
            "progress": {
                "数据结构": "一轮中",
                "组成原理": "一轮中",
                "操作系统": "一轮中",
                "计算机网络": "一轮中",
            },
            "concerns": [],
        }
        goal.update(kw)
        return goal

    def test_profile_tags_enter_priority_kps(self):
        from cs408rag.adversarial.planner import build_detailed_plan

        # 冷僻考点：频次不足以凭 ROI 进 priority_kps，只能靠画像 boost
        plan = build_detailed_plan(self._goal(profile_tags=["死锁产生的必要条件"]))
        tags = [k["tag"] for k in plan["priority_kps"]]
        assert "死锁产生的必要条件" in tags

    def test_invalid_tags_dropped(self):
        from cs408rag.adversarial.planner import normalize_profile_tags
        from cs408rag.agent.exam_analysis import get_analyzer

        out = normalize_profile_tags(
            ["不存在的考点", {"tag": "死锁概念"}, {"tag": "死锁概念"}], get_analyzer()
        )
        assert out == [{"subject": "操作系统", "tag": "死锁概念"}]


# ---------------------------------------------------------------------------
# API 入口的画像钩子（api.main._record_profile_*）：与 Streamlit 同源落库
# ---------------------------------------------------------------------------


class TestApiProfileHooks:
    """API 侧落库是尽力而为的 side-effect：helper 内部自行捕获异常，

    测试只需验证「给了 client_id 就写、没给就不写、题库外不判分不写」。
    """

    @pytest.fixture()
    def hooks(self, tmp_path, monkeypatch):
        import cs408rag.agent.exam_analysis as ea
        import cs408rag.profile.store as ps

        store = ProfileStore(tmp_path / "profile.db")
        monkeypatch.setattr(ea, "get_analyzer", lambda: _analyzer())
        monkeypatch.setattr(ps, "get_profile_store", lambda: store)
        from cs408rag.api.main import _record_profile_question, _record_profile_quiz

        return store, _record_profile_question, _record_profile_quiz

    def test_question_recorded_with_tags(self, hooks):
        store, rec_q, _ = hooks
        rec_q("c1", "死锁怎么预防？", subject="操作系统", qid="2020-01")
        (e,) = store.events("c1")
        assert e["kind"] == "question"
        assert e["payload"]["subject"] == "操作系统"
        assert e["payload"]["qid"] == "2020-01"
        assert [t["tag"] for t in e["payload"]["tags"]] == ["死锁概念"]

    def test_question_skipped_without_client_id(self, hooks):
        store, rec_q, _ = hooks
        rec_q(None, "死锁怎么预防？")
        assert store.events("default") == []

    def test_quiz_wrong_feeds_profile(self, hooks):
        store, _, rec_quiz = hooks
        rec_quiz(
            "c1",
            {"known": True, "qid": "2020-01", "subject": "操作系统",
             "tags": ["死锁概念"], "correct": False},
            "A",
        )
        prof = build_profile(store.events("c1"))
        assert prof["quiz"]["total"] == 1 and prof["quiz"]["correct"] == 0
        assert prof["quiz"]["wrong_tags"][0]["tag"] == "死锁概念"
        assert prof["boost_tags"][0]["tag"] == "死锁概念"  # 答错考点优先级最高

    def test_quiz_correct_recorded(self, hooks):
        store, _, rec_quiz = hooks
        rec_quiz(
            "c1",
            {"known": True, "qid": "2020-01", "subject": "操作系统",
             "tags": ["死锁概念"], "correct": True},
            "B",
        )
        prof = build_profile(store.events("c1"))
        assert prof["quiz"]["total"] == 1 and prof["quiz"]["correct"] == 1
        assert prof["quiz"]["wrong_tags"] == []

    def test_quiz_unknown_not_recorded(self, hooks):
        store, _, rec_quiz = hooks
        rec_quiz("c1", {"known": False, "qid": "2099-99"}, "A")
        assert store.events("c1") == []  # 没有判分依据，不落库


class TestPlanPersistence:
    """规划必须落库，否则刷新后无法判断「是否已制定过规划」这个流程前提。"""

    def test_save_load_and_overwrite(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        assert store.load_plan("c1") is None

        store.save_plan("c1", {"goal": {"remaining_weeks": 6}, "plan": {}})
        got = store.load_plan("c1")
        assert got["goal"]["remaining_weeks"] == 6

        store.save_plan("c1", {"goal": {"remaining_weeks": 12}, "plan": {}})
        assert store.load_plan("c1")["goal"]["remaining_weeks"] == 12

    def test_plan_isolated_by_client(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        store.save_plan("c1", {"goal": {"remaining_weeks": 6}})
        assert store.load_plan("c2") is None

    def test_delete_plan(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        store.save_plan("c1", {"goal": {"remaining_weeks": 6}})
        store.delete_plan("c1")
        assert store.load_plan("c1") is None


class TestConversationOwnership:
    """知道 conv_id 不等于有权读写：所有会话操作都要校验 client_id。"""

    def test_other_client_cannot_read_or_write(self, tmp_path):
        store = ProfileStore(tmp_path / "p.db")
        store.create_conversation("c1", "conv1")
        assert store.append_message("conv1", "turn", {"q": 1}, client_id="c1")

        assert store.load_messages("conv1", client_id="c1") != []
        assert store.load_messages("conv1", client_id="c2") == []

        assert store.append_message("conv1", "turn", {"q": 2}, client_id="c2") is False
        assert store.set_conversation_title("conv1", "x", client_id="c2") is False
        assert store.delete_conversation("conv1", client_id="c2") is False

        # 越权操作全部无效后，原属主的数据仍然完整
        assert len(store.load_messages("conv1", client_id="c1")) == 1
