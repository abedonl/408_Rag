"""检索消融实验执行器。

产出两张表：

1. 检索链路递增消融
   S0 仅 dense → S1 +sparse+graph(RRF) → S2 +rerank → S3 +rewrite → S4 +filter
   相邻两行的差值就是该组件的净贡献。

2. 融合策略对比
   RRF vs 加权分数融合 vs 单路 dense。

另单独报告「图谱长尾救援」：dense 与 sparse 的 top-k 内都没命中相关项时，
图谱路救回了多少题 —— 这是图谱价值最硬的证据，
因为 RRF 的设计天然让「多路共识」的 chunk 胜出，
单看图谱在 top-10 里的占比会低估它。

用法：
    python -m cs408rag.eval.run_eval                 # 开发集，全部阶段
    python -m cs408rag.eval.run_eval --split holdout # held-out（最终报告用）
    python -m cs408rag.eval.run_eval --stages S0_dense,S1_hybrid
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from cs408rag.config import EVAL_DIR, REPORTS_DIR, ensure_dirs
from cs408rag.eval.metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate,
    evaluate_query,
)
from cs408rag.retrieval.pipeline import STAGES, RetrievalConfig, RetrievalPipeline

KS = (1, 3, 5, 10, 20)


@dataclass
class StageResult:
    stage: str
    label: str
    metrics: AggregateMetrics
    per_query: list[QueryMetrics]
    elapsed_s: float
    route_contribution: dict[str, dict[str, float]]

    def row(self) -> dict[str, Any]:
        r = self.metrics.to_row(ks=(5, 10))
        return {"stage": self.stage, "config": self.label, **r}


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------


def load_eval_items(split: str = "dev") -> list[dict[str, Any]]:
    """载入评测集。

    split:
        dev      开发集（2009-2022），用于调参与消融
        holdout  held-out（2023-2026），只用于最终报告
        all      全部
    """
    p = EVAL_DIR / "eval_set.json"
    if not p.exists():
        raise FileNotFoundError(f"找不到评测集 {p}。先跑 python -m cs408rag.eval.build_eval_set")
    rows = json.loads(p.read_text(encoding="utf-8"))

    if split == "dev":
        rows = [r for r in rows if not r["is_holdout"]]
    elif split == "holdout":
        rows = [r for r in rows if r["is_holdout"]]

    # 只保留有 ground truth 的题
    usable = [r for r in rows if r.get("gt_chunks")]
    logger.info(
        f"评测集 [{split}]：{len(usable)}/{len(rows)} 题有 ground truth"
        f"（{len(rows) - len(usable)} 题因考点标签过于笼统无法定位章节，不计入）"
    )
    return usable


# ---------------------------------------------------------------------------
# 单阶段
# ---------------------------------------------------------------------------


def run_stage(
    pipe: RetrievalPipeline,
    items: Sequence[dict[str, Any]],
    stage: str,
    config: RetrievalConfig,
    *,
    ks: Sequence[int] = KS,
    progress_every: int = 100,
) -> StageResult:
    t0 = time.perf_counter()
    per_query: list[QueryMetrics] = []
    latencies: list[float] = []
    subjects: dict[str, str] = {}
    route_totals: dict[str, float] = {}

    for i, it in enumerate(items, 1):
        qid = it["qid"]
        # 注意：这里**不传 subject**。
        # S4 的科目过滤由 config.use_subject_filter 控制，
        # 若在这里无条件传 subject 就等于提前做了过滤，S4 的消融就没有意义了。
        r = pipe.retrieve(it["question"], config=config)
        per_query.append(
            evaluate_query(qid, r.ranked_ids, set(it["gt_chunks"]), ks=ks)
        )
        latencies.append(r.took_ms)
        subjects[qid] = it["subject"]

        for name, tr in r.routes.items():
            route_totals[name] = route_totals.get(name, 0.0) + tr.took_ms

        if i % progress_every == 0:
            logger.info(f"  [{stage}] {i}/{len(items)}")

    agg = aggregate(per_query, latencies_ms=latencies, subjects=subjects, ks=ks)
    n = max(len(items), 1)
    rc = {k: {"avg_ms": round(v / n, 1)} for k, v in route_totals.items()}

    return StageResult(
        stage=stage,
        label=config.label(),
        metrics=agg,
        per_query=per_query,
        elapsed_s=time.perf_counter() - t0,
        route_contribution=rc,
    )


# ---------------------------------------------------------------------------
# 图谱长尾救援
# ---------------------------------------------------------------------------


def graph_long_tail_rescue(
    pipe: RetrievalPipeline,
    items: Sequence[dict[str, Any]],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    """dense/sparse 都失败时，图谱救回多少。

    这是图谱价值最有力的证据：RRF 让多路共识的 chunk 胜出，
    所以图谱在最终 top-10 的占比往往不高，
    但它可能在「其他路都找不到」的长尾题上不可替代。
    """
    cfg = RetrievalConfig(use_sparse=True, use_graph=True, top_k=top_k, exclude_explanations=True)

    hard = 0
    rescued = 0
    graph_only_hit = 0
    details: list[dict[str, Any]] = []

    for it in items:
        gt = set(it["gt_chunks"])
        r = pipe.retrieve(it["question"], config=cfg)
        dense = set(r.routes["dense"].top_ids) if "dense" in r.routes else set()
        sparse = set(r.routes["sparse"].top_ids) if "sparse" in r.routes else set()
        graph = set(r.routes["graph"].top_ids) if "graph" in r.routes else set()

        dense_ok = bool(dense & gt)
        sparse_ok = bool(sparse & gt)
        graph_ok = bool(graph & gt)

        if not dense_ok and not sparse_ok:
            hard += 1
            if graph_ok:
                rescued += 1
                details.append({"qid": it["qid"], "subject": it["subject"]})

        # 三路里只有图谱命中的比例（更宽的统计）
        if graph_ok and not dense_ok and not sparse_ok:
            graph_only_hit += 1

    return {
        "top_k": top_k,
        "hard_queries": hard,
        "rescued_by_graph": rescued,
        "rescue_rate": round(rescued / hard, 4) if hard else 0.0,
        "graph_only_hits": graph_only_hit,
        "rescued_examples": details[:15],
        "note": (
            "hard_queries = dense 与 sparse 的 top-k 内均未命中任何相关项的查询；"
            "rescue_rate = 其中图谱路命中的比例"
        ),
    }


# ---------------------------------------------------------------------------
# 融合策略对比
# ---------------------------------------------------------------------------


def run_fusion_comparison(
    pipe: RetrievalPipeline,
    items: Sequence[dict[str, Any]],
    *,
    ks: Sequence[int] = (5, 10),
) -> list[StageResult]:
    """固定其余配置，只换融合策略。

    ★ ks 必须包含 5：报告表会读 Recall@5 / Precision@5 / Accuracy@5，
      只算 k=10 时这些列会全部显示 0（旧报告里踩过）。
    """
    out: list[StageResult] = []
    variants = {
        "单路 dense（对照）": RetrievalConfig(exclude_explanations=True),
        "RRF (k=60)": RetrievalConfig(use_sparse=True, use_graph=True, fusion="rrf", rrf_k=60, exclude_explanations=True),
        "RRF (k=20)": RetrievalConfig(use_sparse=True, use_graph=True, fusion="rrf", rrf_k=20, exclude_explanations=True),
        "加权融合 (minmax)": RetrievalConfig(
            use_sparse=True, use_graph=True, fusion="weighted", normalize="minmax", exclude_explanations=True
        ),
        "加权融合 (zscore)": RetrievalConfig(
            use_sparse=True, use_graph=True, fusion="weighted", normalize="zscore", exclude_explanations=True
        ),
        "zscore + 节去重(1)": RetrievalConfig(
            use_sparse=True,
            use_graph=True,
            fusion="weighted",
            normalize="zscore",
            section_dedup=True,
            section_per=1,
            exclude_explanations=True,
        ),
        "zscore + 节去重(2)": RetrievalConfig(
            use_sparse=True,
            use_graph=True,
            fusion="weighted",
            normalize="zscore",
            section_dedup=True,
            section_per=2,
            exclude_explanations=True,
        ),
    }
    for name, cfg in variants.items():
        logger.info(f"融合对比：{name}")
        res = run_stage(pipe, items, name, cfg, ks=ks)
        out.append(res)
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def to_markdown(
    ablations: Sequence[StageResult],
    fusions: Sequence[StageResult] | None = None,
    rescue: dict[str, Any] | None = None,
    *,
    split: str = "dev",
    n_queries: int = 0,
) -> str:
    lines: list[str] = []
    lines.append(f"# 检索消融实验（{split}，{n_queries} 题）\n")

    lines.append("## 1. 检索链路递增消融\n")
    lines.append(
        "| 阶段 | 配置 | Recall@5 | Recall@10 | Precision@10 | Accuracy@10 "
        "| MRR@10 | nDCG@10 | P50 延迟 | P95 延迟 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in ablations:
        m = r.metrics
        lines.append(
            f"| {r.stage} | `{r.label}` | {m.recall_at.get(5, 0):.4f} | "
            f"{m.recall_at.get(10, 0):.4f} | "
            f"{m.precision_at.get(10, 0):.4f} | {m.hit_at.get(10, 0):.4f} | "
            f"{m.mrr:.4f} | {m.ndcg_at.get(10, 0):.4f} | "
            f"{m.p50_latency_ms:.0f}ms | {m.p95_latency_ms:.0f}ms |"
        )
    lines.append(
        "\n> **Accuracy@10** = 题级命中率（前 10 条中只要命中该题 ground truth "
        "小节的任意 chunk 就算 1）。"
        "\n> **Recall@10** = 命中的相关 chunk 数 / 该题全部相关 chunk 数。"
        "GT 小节有多个 chunk 时，Accuracy 会高于 Recall。"
        "\n> **Precision@10** 天然偏低：GT 只是一个小节，10 条里通常只有 1–2 条来自该节。\n"
    )

    if len(ablations) >= 2:
        lines.append("\n**相邻阶段增量（该组件的净贡献）**\n")
        lines.append("| 组件 | ΔRecall@10 | ΔMRR@10 | ΔnDCG@10 | ΔP50 |")
        lines.append("|---|---|---|---|---|")
        for prev, cur in zip(ablations, ablations[1:], strict=False):
            lines.append(
                f"| {cur.stage} 相对 {prev.stage} | "
                f"{cur.metrics.recall_at.get(10, 0) - prev.metrics.recall_at.get(10, 0):+.4f} | "
                f"{cur.metrics.mrr - prev.metrics.mrr:+.4f} | "
                f"{cur.metrics.ndcg_at.get(10, 0) - prev.metrics.ndcg_at.get(10, 0):+.4f} | "
                f"{cur.metrics.p50_latency_ms - prev.metrics.p50_latency_ms:+.0f}ms |"
            )

    if fusions:
        lines.append("\n## 2. 融合策略对比\n")
        lines.append("| 融合策略 | Recall@10 | Precision@10 | Accuracy@10 | MRR@10 | nDCG@10 | P50 |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in fusions:
            m = r.metrics
            lines.append(
                f"| {r.stage} | {m.recall_at.get(10, 0):.4f} | "
                f"{m.precision_at.get(10, 0):.4f} | {m.hit_at.get(10, 0):.4f} | "
                f"{m.mrr:.4f} | {m.ndcg_at.get(10, 0):.4f} | {m.p50_latency_ms:.0f}ms |"
            )

    if rescue:
        lines.append("\n## 3. 图谱路的长尾救援\n")
        lines.append(f"- dense 与 sparse 在 top-{rescue['top_k']} 内均失败的查询：**{rescue['hard_queries']}** 题")
        lines.append(f"- 其中被图谱路救回：**{rescue['rescued_by_graph']}** 题")
        lines.append(f"- 救援率：**{rescue['rescue_rate']:.1%}**")

    lines.append("\n## 4. 分科目表现\n")
    if ablations:
        last = ablations[-1]
        if last.metrics.per_subject:
            lines.append("| 科目 | 题数 | Recall@10 | Precision@10 | Accuracy@10 | MRR@10 | nDCG@10 |")
            lines.append("|---|---|---|---|---|---|---|")
            for s, v in sorted(last.metrics.per_subject.items()):
                lines.append(
                    f"| {s} | {v['n']} | {v['Recall@10']:.4f} | "
                    f"{v.get('Precision@10', 0):.4f} | {v.get('Accuracy@10', 0):.4f} | "
                    f"{v['MRR@10']:.4f} | {v['nDCG@10']:.4f} |"
                )

    return "\n".join(lines) + "\n"


def run(
    *,
    split: str = "dev",
    stages: Sequence[str] | None = None,
    with_fusion: bool = True,
    with_rescue: bool = True,
    limit: int | None = None,
) -> dict[str, Any]:
    ensure_dirs()
    items = load_eval_items(split)
    if limit:
        items = items[:limit]

    pipe = RetrievalPipeline()

    # 预热：首次调用含 jieba 词典加载与模型加载，不能算进延迟
    logger.info("预热中…")
    pipe.retrieve("预热查询 Cache 组相联映射", config=STAGES["S1_hybrid"])

    names = list(stages) if stages else list(STAGES)
    ablations: list[StageResult] = []
    for name in names:
        if name not in STAGES:
            raise KeyError(f"未知阶段 {name!r}，可选：{list(STAGES)}")
        logger.info(f"=== 阶段 {name}（{STAGES[name].label()}）===")
        ablations.append(run_stage(pipe, items, name, STAGES[name]))

    fusions = run_fusion_comparison(pipe, items) if with_fusion else None
    rescue = graph_long_tail_rescue(pipe, items) if with_rescue else None

    md = to_markdown(ablations, fusions, rescue, split=split, n_queries=len(items))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mp = REPORTS_DIR / f"ablation_{split}.md"
    mp.write_text(md, encoding="utf-8")

    jp = REPORTS_DIR / f"ablation_{split}.json"
    jp.write_text(
        json.dumps(
            {
                "split": split,
                "n_queries": len(items),
                "stages": [r.row() for r in ablations],
                "fusions": [r.row() for r in fusions] if fusions else [],
                "graph_rescue": rescue,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(f"实验报告 -> {mp}")
    logger.info(f"原始数据 -> {jp}")
    print("\n" + md)
    return {"markdown": md, "path": str(mp)}


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    ap = argparse.ArgumentParser(description="检索消融实验")
    ap.add_argument("--split", default="dev", choices=["dev", "holdout", "all"])
    ap.add_argument("--stages", default=None, help="逗号分隔，如 S0_dense,S1_hybrid")
    ap.add_argument("--no-fusion", action="store_true")
    ap.add_argument("--no-rescue", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 题，快速验证")
    args = ap.parse_args()

    run(
        split=args.split,
        stages=args.stages.split(",") if args.stages else None,
        with_fusion=not args.no_fusion,
        with_rescue=not args.no_rescue,
        limit=args.limit,
    )