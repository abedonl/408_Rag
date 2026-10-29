"""层级 1：Oracle Document（纯生成测试）。

把真题 GT 小节的完整文本直接塞给 LLM，不经过检索，测选择题准确率。
用来分离「生成能力」与「检索能力」：

    如果 Oracle 准确率也很低 -> 问题在 prompt / 模型，不在检索
    如果 Oracle 准确率高     -> 检索是瓶颈，继续优化召回

抽样 50 题（固定种子），断点续跑。
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time
from typing import Any

from cs408rag.config import PROCESSED_DIR, REPORTS_DIR
from cs408rag.eval.run_eval import load_eval_items
from cs408rag.generation.llm import get_llm
from cs408rag.generation.prompt import ContextBlock, parse_choice, quiz_messages

sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

RUN_PATH = REPORTS_DIR / "oracle_document_runs" / "items.jsonl"


def _chunks() -> dict[str, dict]:
    return {
        c["chunk_id"]: c
        for c in json.loads((PROCESSED_DIR / "chunks_all.json").read_text(encoding="utf-8"))
    }


def _load_done(path) -> dict[str, dict]:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[r["qid"]] = r
    return out


def run(limit: int = 50, *, qtype: str = "", tag: str = "") -> dict[str, Any]:
    run_path = RUN_PATH if not tag else RUN_PATH.parent / f"{tag}.jsonl"
    out_path = REPORTS_DIR / f"oracle_{tag or 'document'}.json"
    items = [i for i in load_eval_items("dev") if len(i.get("options") or {}) >= 2]
    if qtype:
        from cs408rag.agent.exam_analysis import ExamAnalyzer

        analyzer = ExamAnalyzer()
        items = [i for i in items if analyzer.type_of.get(i["qid"]) == qtype]
        print(f"题型过滤 {qtype}：{len(items)} 题", flush=True)
    sample = random.Random(42).sample(items, min(limit, len(items)))
    chunks = _chunks()
    done = _load_done(run_path)
    todo = [i for i in sample if i["qid"] not in done]
    print(f"样本 {len(sample)} 题，已完成 {len(done)}，待跑 {len(todo)}", flush=True)

    run_path.parent.mkdir(parents=True, exist_ok=True)
    llm = get_llm()
    for k, it in enumerate(todo, 1):
        texts = [chunks[c]["text"] for c in it["gt_chunks"] if c in chunks]
        blocks = [
            ContextBlock(
                index=i,
                chunk_id=cid,
                text=chunks[cid]["text"],
                citation=" > ".join(chunks[cid].get("chapter_path") or []) or "GT 小节",
                subject=chunks[cid].get("subject", ""),
                source="textbook",
            )
            for i, cid in enumerate(it["gt_chunks"], 1)
            if cid in chunks
        ]
        if not blocks:
            continue

        t0 = time.perf_counter()
        resp = llm.chat(
            quiz_messages(it["question"], it["options"], blocks), max_tokens=1600
        )
        took = (time.perf_counter() - t0) * 1000
        choice = parse_choice(resp.text)
        row = {
            "qid": it["qid"],
            "year": it["year"],
            "subject": it["subject"],
            "choice": choice,
            "gold": it["answer"],
            "correct": choice == it["answer"],
            "gt_chars": sum(len(t) for t in texts),
            "n_gt_chunks": len(blocks),
            "took_ms": round(took, 1),
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "empty": not resp.text.strip(),
            "error": resp.error,
            # 保留推理过程，供错题归因（算式理解错 vs 数值算错）
            "answer": (resp.text or "")[:1200],
        }
        with run_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        done[it["qid"]] = row
        print(
            f"  {k}/{len(todo)} {it['qid']} 预测={choice!r} 正确={row['correct']} "
            f"{took:.0f}ms",
            flush=True,
        )

    rows = [r for r in done.values() if r.get("gold")]
    n = max(len(rows), 1)
    acc = sum(r["correct"] for r in rows) / n
    by_subject: dict[str, list[int]] = {}
    for r in rows:
        by_subject.setdefault(r["subject"], []).append(int(r["correct"]))
    summary = {
        "n": len(rows),
        "accuracy": round(acc, 4),
        "empty_answers": sum(r["empty"] for r in rows),
        "by_subject": {
            s: {"n": len(v), "accuracy": round(sum(v) / len(v), 4)}
            for s, v in by_subject.items()
        },
        "avg_gt_chars": round(statistics.mean(r["gt_chars"] for r in rows), 0),
        "p50_ms": round(statistics.median(r["took_ms"] for r in rows), 0),
    }
    out_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Oracle Document 生成能力测试")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--type", default="", help="只测某题型：calc / concept / figure")
    ap.add_argument("--tag", default="", help="输出文件后缀，默认与 --type 相同")
    args = ap.parse_args()
    run(limit=args.limit, qtype=args.type, tag=args.tag or args.type)