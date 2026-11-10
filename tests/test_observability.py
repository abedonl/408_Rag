"""后台留档测试：页面隐藏的检索数据仍要可查。"""

from __future__ import annotations

import json
from types import SimpleNamespace

import cs408rag.observability as ob


def _result(routes: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        routes={
            f"r{i}": SimpleNamespace(took_ms=1.5, top_ids=["c1", "c2", "c3"]) for i in range(routes)
        },
        fused=[SimpleNamespace(doc_id="d1", score=0.4231, ranks={"r0": 1, "r1": 2})],
        rerank_applied=True,
    )


def test_record_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "TRACE_PATH", tmp_path / "t.jsonl")
    ob.record_retrieval(_result(), query="KMP next 数组", subject="数据结构", source="streamlit_qa")
    items = ob.read_traces(10)
    assert len(items) == 1
    rec = items[0]
    assert rec["query"] == "KMP next 数组"
    assert rec["subject"] == "数据结构"
    assert rec["source"] == "streamlit_qa"
    assert rec["routes"]["r0"]["top_ids"] == ["c1", "c2", "c3"]
    assert rec["fused"][0]["score"] == 0.4231
    assert rec["rerank"] is True


def test_read_newest_first_and_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "TRACE_PATH", tmp_path / "t.jsonl")
    for q in ("第一问", "第二问", "第三问"):
        ob.record_retrieval(_result(), query=q, source="api")
    items = ob.read_traces(2)
    assert [r["query"] for r in items] == ["第三问", "第二问"]


def test_payload_whitelist(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "TRACE_PATH", tmp_path / "t.jsonl")
    ob.record_payload(
        {"request_id": "abc", "query": "q", "answer": "不该留档的长文本", "trace": {"x": 1}},
        source="api",
        keys=("request_id", "query", "trace"),
    )
    rec = ob.read_traces(1)[0]
    assert rec["request_id"] == "abc"
    assert "answer" not in rec


def test_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "TRACE_PATH", tmp_path / "none.jsonl")
    assert ob.read_traces(5) == []


def test_bad_line_skipped(tmp_path, monkeypatch):
    p = tmp_path / "t.md"
    p.write_text('```json\n{"query":"ok"}\n```\n块外的杂字和坏 json ```json\n{坏}\n```\n', encoding="utf-8")
    monkeypatch.setattr(ob, "TRACE_PATH", p)
    items = ob.read_traces(5)
    assert len(items) == 1
    assert items[0]["query"] == "ok"


def test_record_writes_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr(ob, "TRACE_PATH", tmp_path / "t.md")
    ob.record_retrieval(_result(), query="死锁条件", source="streamlit_qa")
    text = (tmp_path / "t.md").read_text(encoding="utf-8")
    assert text.startswith("## ")
    assert "死锁条件" in text
    assert "```json" in text


def test_export_ablation_md(tmp_path):
    (tmp_path / "ablation_dev.json").write_text(
        json.dumps(
            {
                "split": "dev",
                "n_queries": 10,
                "stages": [{"stage": "S0", "Recall@5": 0.3, "P50_ms": 50}],
                "fusions": [{"stage": "RRF", "Recall@5": 0.4}],
                "graph_rescue": {"hard_queries": 5, "rescued_by_graph": 1, "rescue_rate": 0.2},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = ob.export_ablation_md(reports_dir=tmp_path)
    text = out.read_text(encoding="utf-8")
    assert "## dev（n=10）" in text
    assert "| S0 |" in text
    assert "救回 1 条" in text
