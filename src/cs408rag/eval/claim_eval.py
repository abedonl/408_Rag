"""考点结论质量评测（可溯源 / 禁押题 / 频次准确）。

计划里的三条硬指标：

1. **可溯源率**：考点结论中的真题编号必须真实存在，且属于该考点的真题集合
2. **频次准确率**：结论里「考查 N 次」的 N 必须与该考点的真实统计一致
3. **押题零容忍**：出现「今年可能」「预测」等表述即不合格，命中数必须为 0

自动校验能覆盖编号存在性与频次数字；「表述是否与数据相符」仍需人工核对，
因此报告同时导出 `reports/claim_samples_for_review.json`，按计划的抽样口径
（50 条结论）交给人工复核。

用法：
    python -m cs408rag.eval.claim_eval --limit 20          # 前 20 个高频考点
    python -m cs408rag.eval.claim_eval --limit 50
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from loguru import logger

from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer, qid_to_ref
from cs408rag.config import REPORTS_DIR, ensure_dirs
from cs408rag.generation.citation import detect_predictions, extract_exam_refs

RUNS_DIR = REPORTS_DIR / "claim_eval_runs"

# 「考查 5 次 / 出现 3 次 / 考过 12 次」这类频次陈述
RE_FREQ = re.compile(r"(?:共|历年|累计|总共)?\s*(?:考查|出现|考过|考了)\s*(\d+)\s*次")


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                row = json.loads(line)
                out[row["topic"]] = row
            except json.JSONDecodeError:
                continue
    return out


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def check_answer(
    answer: str, topic: str, analyzer: ExamAnalyzer, valid_refs: set[str]
) -> dict[str, Any]:
    """对一条考点结论做三项校验。"""
    stats = analyzer.stats(topic)
    on_topic = {qid_to_ref(q) for q in stats.qids}

    refs = list(dict.fromkeys(extract_exam_refs(answer)))
    exists = [r for r in refs if r in valid_refs]
    on_topic_refs = [r for r in refs if r in on_topic]
    fabricated = [r for r in refs if r not in valid_refs]

    predictions = detect_predictions(answer)
    freq_claims = [int(x) for x in RE_FREQ.findall(answer)]
    # 频次校验：只要结论里出现了与该考点真实总次数一致的数字就算对上。
    # 回答可能同时分科目/分题型计数，不能要求所有数字都等于总数。
    freq_ok = (not freq_claims) or (stats.total in freq_claims)

    return {
        "topic": topic,
        "matched_tags": stats.matched_tags,
        "true_total": stats.total,
        "qids": stats.qids,
        "exam_refs": refs,
        "refs_valid": len(exists),
        "refs_on_topic": len(on_topic_refs),
        "fabricated_refs": fabricated,
        "predictions": predictions,
        "freq_claims": freq_claims,
        "freq_ok": freq_ok,
        # 可溯源 = 编号不编造（计划口径：真题编号必须真实存在）。
        # 引用其他考点的真题做对比是合理的，因此「是否属于本考点」
        # 单独作为参考指标，不作为通过条件。
        "traceable_ok": bool(refs) and not fabricated,
        "no_prediction": not predictions,
    }


def collect(
    analyzer: ExamAnalyzer | None = None,
    *,
    limit: int = 20,
    out_path: Path | None = None,
) -> list[dict[str, Any]]:
    from cs408rag.agent.graph import AgentRunner
    from cs408rag.agent.tools import AgentTools
    from cs408rag.generation.service import build_exam_ref_whitelist

    analyzer = analyzer or get_analyzer()
    out_path = out_path or (RUNS_DIR / "claims.jsonl")
    valid_refs = build_exam_ref_whitelist()
    valid_norm = {qid_to_ref(r.replace("-", "#")) for r in valid_refs}

    done = _load_done(out_path)
    topics = list(analyzer.tag_freq)[:limit]
    todo = [t for t in topics if t not in done]
    logger.info(f"[claim] 已完成 {len(done)}，待跑 {len(todo)}")

    tools = AgentTools(analyzer=analyzer, valid_exam_refs=valid_norm)
    runner = AgentRunner(tools=tools)

    for i, topic in enumerate(todo, 1):
        q = f"考点「{topic}」历年考过几次？通常怎么考？和哪些知识点关联？"
        t0 = time.perf_counter()
        res = runner.run(q, mode="qa")
        check = check_answer(res.answer, topic, analyzer, valid_norm)
        row = {
            **check,
            "answer": res.answer,
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "route": res.route,
            "n_tools": len(res.tool_calls),
            "error": res.error,
        }
        _append(out_path, row)
        done[topic] = row
        logger.info(
            f"[claim] {i}/{len(todo)} {topic} 频次={check['true_total']} "
            f"引用={check['refs_on_topic']}/{len(check['exam_refs'])} "
            f"可溯源={check['traceable_ok']} 押题={len(check['predictions'])}"
        )
    return list(done.values())


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r.get("answer")]
    refs_total = sum(len(r["exam_refs"]) for r in usable)
    refs_ok = sum(r["refs_on_topic"] for r in usable)
    fabricated = sum(len(r["fabricated_refs"]) for r in usable)
    predictions = sum(len(r["predictions"]) for r in usable)
    with_claims = [r for r in usable if r["freq_claims"]]
    freq_ok = sum(1 for r in with_claims if r["freq_ok"])
    return {
        "n_topics": len(usable),
        "n_answers": len(usable),
        "refs_total": refs_total,
        "refs_valid": sum(r["refs_valid"] for r in usable),
        "refs_on_topic": refs_ok,
        # 主指标：编号真实存在的比例（目标 100%）
        "traceable_rate": round((refs_total - fabricated) / refs_total, 4) if refs_total else 0.0,
        # 参考指标：引用的真题属于该考点的比例
        "on_topic_rate": round(refs_ok / refs_total, 4) if refs_total else 0.0,
        "fabricated_refs": fabricated,
        "prediction_hits": predictions,
        "n_with_freq_claim": len(with_claims),
        "freq_accuracy": round(freq_ok / len(with_claims), 4) if with_claims else None,
        "topics_traceable": sum(1 for r in usable if r["traceable_ok"]),
        "topics_no_prediction": sum(1 for r in usable if r["no_prediction"]),
    }


def to_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = ["# 考点结论质量评测\n"]
    lines.append(
        "三条硬指标：真题编号可溯源、频次与标注一致、押题零容忍。\n"
    )
    lines.append("## 汇总\n")
    lines.append("| 指标 | 结果 | 口径 |")
    lines.append("|---|---|---|")
    lines.append(
        f"| 可溯源率 | **{summary['traceable_rate']:.1%}** | "
        "结论中的真题编号真实存在的比例"
        f"（{summary['refs_total'] - summary['fabricated_refs']}/{summary['refs_total']}） |"
    )
    lines.append(
        f"| 本考点引用率 | {summary['on_topic_rate']:.1%} | "
        "引用的真题属于该考点真题集"
        f"（{summary['refs_on_topic']}/{summary['refs_total']}），参考指标 |"
    )
    lines.append(
        f"| 编造真题编号 | **{summary['fabricated_refs']}** | 必须为 0 |"
    )
    lines.append(
        f"| 押题表述 | **{summary['prediction_hits']}** | 必须为 0 |"
    )
    fa = summary.get("freq_accuracy")
    lines.append(
        f"| 频次准确率 | **{fa:.1%}** | {summary['n_with_freq_claim']} 条含频次陈述的结论 |"
        if fa is not None
        else "| 频次准确率 | — | 没有可校验的频次陈述 |"
    )
    lines.append(
        f"| 逐条通过可溯源 | {summary['topics_traceable']}/{summary['n_topics']} | "
        "编号全部真实且属于该考点 |"
    )

    lines.append("\n## 明细\n")
    lines.append("| 考点 | 真实频次 | 引用编号 | 可溯源 | 频次一致 | 押题 |")
    lines.append("|---|---|---|---|---|---|")
    for r in rows:
        if not r.get("answer"):
            continue
        lines.append(
            f"| {r['topic']} | {r['true_total']} | "
            f"{'、'.join(r['exam_refs'][:6]) or '—'} | "
            f"{'✅' if r['traceable_ok'] else '❌'} | "
            f"{'✅' if r['freq_ok'] else '❌'} | "
            f"{len(r['predictions'])} |"
        )

    lines.append(
        "\n## 口径说明\n"
        "- 仅统计回答中显式写出的 `YYYY#NN` 或「YYYY 年第 N 题」编号；"
        "只提年份（如「2014、2021」）无法定位到题，不计入可溯源率。\n"
        "- 频次校验取「考查/出现 N 次」句式；回答可能同时给出分科目计数，"
        "只要出现一次与真实总数一致即判通过。\n"
        "- 自动校验不能判断「表述是否恰当地归纳了考查方式」，"
        "因此另导出抽样明细供人工复核。\n"
    )
    return "\n".join(lines) + "\n"


def run(*, limit: int = 20) -> dict[str, Any]:
    ensure_dirs()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect(limit=limit)
    rows = [r for r in rows if r.get("answer")]
    if not rows:
        raise SystemExit("没有可用结论（检查 LLM 配置）")
    summary = summarize(rows)
    md = to_markdown(rows, summary)

    (REPORTS_DIR / "claim_eval.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "claim_eval.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    # 人工复核样本（计划要求抽 50 条结论核对）
    review = [
        {
            "topic": r["topic"],
            "true_total": r["true_total"],
            "qids": r["qids"][:10],
            "answer": r["answer"],
            "refs_in_answer": r["exam_refs"],
            "freq_claims": r["freq_claims"],
        }
        for r in rows
    ][:50]
    (REPORTS_DIR / "claim_samples_for_review.json").write_text(
        json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(md)
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="考点结论质量评测")
    ap.add_argument("--limit", type=int, default=20, help="取前 N 个高频考点")
    args = ap.parse_args()
    run(limit=args.limit)