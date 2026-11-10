"""错题本：SQLite 存储与判分逻辑的单测。

覆盖：NotebookStore 的记录/累加/移出/按 client_id 隔离/科目过滤与排序，
以及 judge 的三条路径（答错记录、答对自动移出、题库外不判分）。
不依赖 Qdrant / LLM；题库用 monkeypatch 替换 _truth_index。
"""

from __future__ import annotations

import pytest

from cs408rag.ingestion.quiz_loader import Question
from cs408rag.notebook import judge as judge_mod
from cs408rag.notebook.judge import judge
from cs408rag.notebook.store import NotebookStore


def _store(tmp_path) -> NotebookStore:
    return NotebookStore(tmp_path / "notebook.db")


def _record(store: NotebookStore, cid: str, qid: str, **kw):
    defaults = {
        "your_choice": "A",
        "correct_choice": "B",
        "subject": "操作系统",
        "tags": ["死锁"],
    }
    defaults.update(kw)
    return store.record_wrong(cid, qid, **defaults)


# ---------------------------------------------------------------------------
# NotebookStore
# ---------------------------------------------------------------------------


class TestNotebookStore:
    def test_record_and_get(self, tmp_path):
        s = _store(tmp_path)
        item = _record(s, "c1", "2009-04")
        assert item["qid"] == "2009-04"
        assert item["wrong_count"] == 1
        assert item["tags"] == ["死锁"]
        assert item["first_wrong_at"] and item["last_wrong_at"]
        assert s.get("c1", "2009-04")["your_choice"] == "A"

    def test_repeat_wrong_accumulates(self, tmp_path):
        s = _store(tmp_path)
        _record(s, "c1", "2009-04")
        item = _record(s, "c1", "2009-04", your_choice="C")
        assert item["wrong_count"] == 2
        assert item["your_choice"] == "C"  # 保留最近一次作答
        assert len(s.list_items("c1")) == 1  # 同一题不重复占位

    def test_client_isolation(self, tmp_path):
        s = _store(tmp_path)
        _record(s, "c1", "2009-04")
        assert s.list_items("c2") == []
        assert s.get("c2", "2009-04") is None
        assert s.remove("c2", "2009-04") is False  # 删不到别人的记录
        assert s.get("c1", "2009-04") is not None

    def test_remove(self, tmp_path):
        s = _store(tmp_path)
        _record(s, "c1", "2009-04")
        assert s.remove("c1", "2009-04") is True
        assert s.get("c1", "2009-04") is None
        assert s.remove("c1", "2009-04") is False  # 已不存在

    def test_list_order_and_subject_filter(self, tmp_path):
        s = _store(tmp_path)
        _record(s, "c1", "2009-04", subject="操作系统")
        _record(s, "c1", "2010-05", subject="数据结构", tags=["树"])
        # 时间戳是秒级，快速连写会相同导致排序不定；直接改库保证确定性
        s._conn.execute(
            "UPDATE wrong_items SET last_wrong_at='2026-01-01T00:00:00+00:00' "
            "WHERE qid='2010-05'"
        )
        s._conn.execute(
            "UPDATE wrong_items SET last_wrong_at='2026-01-02T00:00:00+00:00' "
            "WHERE qid='2009-04'"
        )
        s._conn.commit()
        items = s.list_items("c1")
        assert [i["qid"] for i in items] == ["2009-04", "2010-05"]  # 最近答错在前
        ds = s.list_items("c1", subject="数据结构")
        assert [i["qid"] for i in ds] == ["2010-05"]


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------


def _fake_question(answer: str = "B") -> Question:
    return Question(
        year=2009,
        num=4,
        subject="操作系统",
        stem="题干",
        options={"A": "…", "B": "…", "C": "…", "D": "…"},
        answer=answer,
        tags=["死锁"],
    )


@pytest.fixture()
def fake_truth(monkeypatch):
    index = {"2009-04": _fake_question()}
    monkeypatch.setattr(judge_mod, "_truth_index", lambda: index)
    return index


class TestJudge:
    def test_wrong_records(self, tmp_path, fake_truth):
        s = _store(tmp_path)
        r = judge("c1", "2009-04", "A", store=s)
        assert r["known"] and not r["correct"] and r["recorded"]
        assert r["truth"] == "B"
        assert r["subject"] == "操作系统" and r["tags"] == ["死锁"]  # 供画像 quiz 事件
        assert r["item"]["wrong_count"] == 1
        assert s.get("c1", "2009-04") is not None

    def test_correct_removes(self, tmp_path, fake_truth):
        s = _store(tmp_path)
        judge("c1", "2009-04", "A", store=s)  # 先错一次进本
        r = judge("c1", "2009-04", "B", store=s)
        assert r["known"] and r["correct"] and r["removed"] is True
        assert s.get("c1", "2009-04") is None  # 答对自动移出

    def test_correct_not_in_book(self, tmp_path, fake_truth):
        s = _store(tmp_path)
        r = judge("c1", "2009-04", "B", store=s)
        assert r["correct"] and r["removed"] is False
        assert r["subject"] == "操作系统" and r["tags"] == ["死锁"]  # 答对也带，画像要记

    def test_unknown_qid_not_recorded(self, tmp_path, fake_truth):
        s = _store(tmp_path)
        r = judge("c1", "2099-99", "A", store=s)
        assert r == {"known": False, "qid": "2099-99"}
        assert s.list_items("c1") == []  # 没有标准答案依据的记录不落库
