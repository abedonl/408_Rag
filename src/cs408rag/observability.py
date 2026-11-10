"""后台留档：页面不再展示的检索 / 评测数据写进一个 Markdown 文件。

明确取舍：
    页面是给学生用的，三路召回明细、RRF 排名变化、消融结果属于开发者视角，
    不再上屏；但它们不该丢，追加到 data/logs/retrieval_traces.md，
    通过 /v1/traces 查询。每条记录 = 一行标题 + 一个 ```json 代码块：
    人直接看标题，机器解析 json 块。只记 id / 排名 / 耗时等结构信息，
    不复制原文，避免把语料又存一份。
"""

from __future__ import annotations

import json
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loguru import logger

from cs408rag.config import DATA_DIR

TRACE_PATH = DATA_DIR / "logs" / "retrieval_traces.md"
_lock = threading.Lock()

_JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.DOTALL)


def _append(record: dict[str, Any]) -> None:
    try:
        TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
        query = str(record.get("query") or "")[:60].replace("\n", " ")
        head = f"## {record['ts']} · {record.get('source') or '-'} · {query}"
        body = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, TRACE_PATH.open("a", encoding="utf-8") as f:
            f.write(f"{head}\n\n```json\n{body}\n```\n\n")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"检索留档写入失败：{e}")


def record_retrieval(
    result: Any, *, query: str = "", subject: str | None = None, source: str = ""
) -> None:
    """把一次检索的路线 / 排名 / 耗时落盘（失败不影响主流程）。"""
    routes = {
        name: {
            "took_ms": round(float(getattr(tr, "took_ms", 0) or 0), 1),
            "top_ids": list(getattr(tr, "top_ids", []) or [])[:10],
        }
        for name, tr in (getattr(result, "routes", None) or {}).items()
    }
    fused = [
        {
            "doc_id": getattr(f, "doc_id", ""),
            "score": round(float(getattr(f, "score", 0) or 0), 5),
            "ranks": getattr(f, "ranks", {}),
        }
        for f in (getattr(result, "fused", None) or [])[:10]
    ]
    _append(
        {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "source": source,
            "query": query[:120],
            "subject": subject or "",
            "routes": routes,
            "fused": fused,
            "rerank": bool(getattr(result, "rerank_applied", False)),
        }
    )


def record_payload(payload: dict[str, Any], *, source: str, keys: tuple[str, ...]) -> None:
    """接口层留档：只取白名单字段，避免把整段回答也存进去。"""
    record = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "source": source}
    record.update({k: payload.get(k) for k in keys if k in payload})
    _append(record)


def read_traces(limit: int = 50) -> list[dict[str, Any]]:
    """最近的 N 条，按时间倒序。解析 ```json 代码块，块外的文字忽略。"""
    if not TRACE_PATH.exists():
        return []
    text = TRACE_PATH.read_text(encoding="utf-8")
    out: list[dict[str, Any]] = []
    for block in _JSON_BLOCK.findall(text)[-max(1, limit) :]:
        try:
            out.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    return list(reversed(out))


def trace_path() -> Path:
    return TRACE_PATH


def export_ablation_md(
    reports_dir: Path | None = None, out_path: Path | None = None
) -> Path:
    """把 reports/ 下的消融 JSON 渲染成一个可读的 Markdown 表（页面表格的后台存档）。

    重跑消融脚本后再调一次即可刷新。
    """
    root = reports_dir or (Path(__file__).resolve().parents[2] / "reports")
    out = out_path or (root / "experiment_data.md")

    def _table(rows: list[dict[str, Any]], cols: list[str]) -> str:
        lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in rows:
            lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)

    parts: list[str] = ["# 检索实验数据（页面已下线，存档）", ""]
    for name in ("ablation_dev.json", "ablation_holdout.json"):
        p = root / name
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        parts.append(f"## {d.get('split', name)}（n={d.get('n_queries', '?')}）")
        metric_cols = ["stage", "Recall@5", "Recall@10", "MRR@10", "nDCG@10", "P50_ms", "P95_ms"]
        if d.get("stages"):
            parts += ["", "### 各阶段", _table(d["stages"], metric_cols)]
        if d.get("fusions"):
            parts += ["", "### 融合对照", _table(d["fusions"], metric_cols)]
        gr = d.get("graph_rescue") or {}
        if gr:
            parts += [
                "",
                "### 图谱救援",
                f"hard queries {gr.get('hard_queries', '?')} 中由图谱救回 "
                f"{gr.get('rescued_by_graph', '?')} 条（{gr.get('rescue_rate', 0):.1%}）",
            ]
        parts.append("")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out
