"""held-out 生成侧端到端评测（2023–2026）。

这是最终报告的主数字：选择题准确率，held-out 年份从未参与调参。

防泄漏由两处保证，评测时同时校验：
    1. held-out 解析从未入库（chunk_explanations 直接跳过这些年份）
    2. 检索层 `exclude_years` 二次拦截

用法：
    python -m cs408rag.eval.holdout_eval --per-subject 15
    python -m cs408rag.eval.holdout_eval --per-subject 40   # 全量
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from cs408rag.config import PROCESSED_DIR, REPORTS_DIR, settings
from cs408rag.eval.run_eval import load_eval_items

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

RUN_PATH = REPORTS_DIR / "holdout_generation_runs" / "items.jsonl"


def _load_done(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["qid"]] = r
    return out


def check_no_leak() -> dict[str, Any]:
    """校验 held-out 年份的解析没有进索引。"""
    holdout = sorted(settings.holdout_year_set)
    chunks = json.loads((PROCESSED_DIR / "chunks_all.json").read_text(encoding="utf-8"))
    leaked = [
        c["chunk_id"]
        for c in chunks
        if c.get("source") == "explanation" and c.get("year") in holdout
    ]
    return {
        "holdout_years": holdout,
        "explanation_chunks_total": sum(
            1 for c in chunks if c.get("source") == "explanation"
        ),
        "leaked_explanation_chunks": len(leaked),
        "passed": not leaked,
    }


def sample_by_subject(per_subject: int, seed: int = 42) -> list[dict]:
    items = [
        i
        for i in load_eval_items("holdout")
        if len(i.get("options") or {}) >= 2
    ]
    by_subj: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_subj[it["subject"]].append(it)
    rng = random.Random(seed)
    picked: list[dict] = []
    for rows in by_subj.values():
        rows = sorted(rows, key=lambda r: (r["year"], r["qid"]))
        rng.shuffle(rows)
        picked.extend(rows[:per_subject])
    return picked


def run(per_subject: int = 15) -> dict[str, Any]:
    leak = check_no_leak()
    print(json.dumps(leak, ensure_ascii=False), flush=True)
    if not leak["passed"]:
        raise SystemExit("防泄漏校验失败：held-out 解析出现在索引中，评测中止")

    sample = sample_by_subject(per_subject)
    done = _load_done(RUN_PATH)
    todo = [i for i in sample if i["qid"] not in done]
    print(f"样本 {len(sample)} 题，已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    from cs408rag.generation.service import QAService

    svc = QAService(config=None)  # 使用生产默认配置
    RUN_PATH.parent.mkdir(parents=True, exist_ok=True)
    for k, it in enumerate(todo, 1):
        t0 = time.perf_counter()
        res = svc.answer_quiz(it["question"], it["options"], config=svc.config)
        row = {
            "qid": it["qid"],
            "year": it["year"],
            "subject": it["subject"],
            "choice": res.choice,
            "gold": it["answer"],
            "correct": res.choice == it["answer"],
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "prompt_tokens": res.llm_call.get("prompt_tokens", 0),
            "completion_tokens": res.llm_call.get("completion_tokens", 0),
            "empty": not (res.answer or "").strip(),
            "error": res.error,
        }
        with RUN_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[it["qid"]] = row
        print(
            f"  {k}/{len(todo)} {it['qid']} 预测={row['choice']!r} "
            f"正确={row['correct']} {row['took_ms']:.0f}ms",
            flush=True,
        )

    rows = list(done.values())
    n = max(len(rows), 1)
    usable = [r for r in rows if not r["empty"]]
    by_subject: dict[str, list[int]] = defaultdict(list)
    by_year: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        by_subject[r["subject"]].append(int(r["correct"]))
        by_year[r["year"]].append(int(r["correct"]))

    summary = {
        "split": "holdout",
        "n": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / n, 4),
        "n_usable": len(usable),
        "accuracy_usable": round(
            sum(r["correct"] for r in usable) / max(len(usable), 1), 4
        ),
        "empty_answers": sum(r["empty"] for r in rows),
        "by_subject": {
            s: {
                "n": len(v),
                "accuracy": round(sum(v) / len(v), 4),
            }
            for s, v in sorted(by_subject.items())
        },
        "by_year": {
            str(y): {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for y, v in sorted(by_year.items())
        },
        "p50_ms": round(statistics.median(r["took_ms"] for r in rows), 0),
        "avg_tokens": round(
            statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in rows), 0
        ),
        "leak_check": leak,
    }
    out = REPORTS_DIR / "holdout_generation.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="held-out 生成侧评测")
    ap.add_argument("--per-subject", type=int, default=15)
    args = ap.parse_args()
    run(per_subject=args.per_subject)