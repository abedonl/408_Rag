"""检索指标。

纯 numpy 实现，不依赖 ir_measures / pytrec_eval —— 便于手算对照与单测。

评测口径（重要）：
    ground truth 的构造链是
        题目 tags → tag_kp_mapping → 知识点节点 → kp_chunk 关联 → 教材 chunk
    所以这些指标衡量的是「文本块级召回」，不是「知识点级召回」。
    映射覆盖率（见 tag_mapper 报告）是这一链的上游，必须一并报告。
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    qid: str
    recall_at: dict[int, float] = field(default_factory=dict)
    precision_at: dict[int, float] = field(default_factory=dict)
    hit_at: dict[int, float] = field(default_factory=dict)
    ndcg_at: dict[int, float] = field(default_factory=dict)
    rr: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0
    # 便于错误分析：哪些相关项没被召回
    missed: list[str] = field(default_factory=list)


def hit_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """题级准确率：前 k 个里只要命中任意一个相关项就算 1，否则 0。

    与 Recall@k 的区别：
        Recall@k = 命中的相关 chunk 数 / 该题全部相关 chunk 数
        Hit@k    = 是否至少命中一个（0/1）
    当 ground truth 小节有多个 chunk 时，两者不同。选择题场景关心的是
    「有没有拿到依据」，所以 Hit@k（报告里写作 Accuracy@k）更贴近产品口径。
    """
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """召回率 = 前 k 个里命中的相关项 / 全部相关项数。"""
    if not relevant:
        return 0.0
    hit = len(set(retrieved[:k]) & relevant)
    return hit / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """Precision@k = 前 k 个里命中的相关项 / k。

    分母固定用 k，而不是实际返回条数：否则检索只返回 2 条时，
    命中 1 条会显示 Precision=0.5，与返回 10 条命中 5 条的 0.5 不可比。
    """
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    """首个相关项的排名倒数。无命中返回 0。"""
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """二值相关性下的 nDCG@k。

    相关项视为 gain=1，IDCG 取理想排序（相关项全部排在前面）。
    """
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    qid: str,
    retrieved: Sequence[str],
    relevant: Iterable[str],
    ks: Sequence[int] = (1, 3, 5, 10, 20),
) -> QueryMetrics:
    """单条 query 的指标。"""
    rel = set(relevant)
    m = QueryMetrics(
        qid=qid,
        num_relevant=len(rel),
        num_retrieved=len(retrieved),
    )
    for k in ks:
        m.recall_at[k] = recall_at_k(retrieved, rel, k)
        m.precision_at[k] = precision_at_k(retrieved, rel, k)
        m.hit_at[k] = hit_at_k(retrieved, rel, k)
        m.ndcg_at[k] = ndcg_at_k(retrieved, rel, k)
    m.rr = reciprocal_rank(retrieved, rel)
    m.missed = sorted(rel - set(retrieved))
    return m


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    n_queries: int
    n_queries_with_relevant: int
    recall_at: dict[int, float]
    precision_at: dict[int, float]
    hit_at: dict[int, float]
    ndcg_at: dict[int, float]
    mrr: float
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    per_subject: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_row(self, ks: Sequence[int] = (5, 10)) -> dict[str, float]:
        """转成实验表的一行。"""
        row: dict[str, float] = {"n_queries": self.n_queries}
        for k in ks:
            row[f"Recall@{k}"] = round(self.recall_at.get(k, 0.0), 4)
            row[f"Precision@{k}"] = round(self.precision_at.get(k, 0.0), 4)
            row[f"Accuracy@{k}"] = round(self.hit_at.get(k, 0.0), 4)
        row["MRR@10"] = round(self.mrr, 4)
        for k in ks:
            row[f"nDCG@{k}"] = round(self.ndcg_at.get(k, 0.0), 4)
        row["P50_ms"] = round(self.p50_latency_ms, 1)
        row["P95_ms"] = round(self.p95_latency_ms, 1)
        return row


def aggregate(
    per_query: list[QueryMetrics],
    latencies_ms: list[float] | None = None,
    subjects: dict[str, str] | None = None,
    ks: Sequence[int] = (5, 10),
) -> AggregateMetrics:
    """汇总多 query 的指标。

    ★ 只对「有相关项」的 query 求平均。
      没有 ground truth 的 query（tag 未映射成功）若计入分母，
      会让成绩看起来更好或更差——都不对。它们的数量单独报告。
    """
    usable = [m for m in per_query if m.num_relevant > 0]
    n_usable = len(usable)

    # 覆盖率是评测链上游指标，单独暴露
    coverage = n_usable / len(per_query) if per_query else 0.0

    def mean_of(fn) -> float:
        if not n_usable:
            return 0.0
        return sum(fn(m) for m in usable) / n_usable

    agg = AggregateMetrics(
        n_queries=len(per_query),
        n_queries_with_relevant=n_usable,
        recall_at={k: mean_of(lambda m, k=k: m.recall_at.get(k, 0.0)) for k in ks},
        precision_at={k: mean_of(lambda m, k=k: m.precision_at.get(k, 0.0)) for k in ks},
        hit_at={k: mean_of(lambda m, k=k: m.hit_at.get(k, 0.0)) for k in ks},
        ndcg_at={k: mean_of(lambda m, k=k: m.ndcg_at.get(k, 0.0)) for k in ks},
        mrr=mean_of(lambda m: m.rr),
    )

    if latencies_ms:
        import numpy as np

        a = np.asarray(latencies_ms, dtype=float)
        agg.avg_latency_ms = float(a.mean())
        agg.p50_latency_ms = float(np.percentile(a, 50))
        agg.p95_latency_ms = float(np.percentile(a, 95))

    if subjects:
        by_subj: dict[str, list[QueryMetrics]] = {}
        for m in usable:
            s = subjects.get(m.qid)
            if s:
                by_subj.setdefault(s, []).append(m)
        for s, ms in by_subj.items():
            n = len(ms)
            agg.per_subject[s] = {
                "n": n,
                "Recall@10": round(sum(m.recall_at.get(10, 0.0) for m in ms) / n, 4),
                "Precision@10": round(
                    sum(m.precision_at.get(10, 0.0) for m in ms) / n, 4
                ),
                "Accuracy@10": round(sum(m.hit_at.get(10, 0.0) for m in ms) / n, 4),
                "MRR@10": round(sum(m.rr for m in ms) / n, 4),
                "nDCG@10": round(sum(m.ndcg_at.get(10, 0.0) for m in ms) / n, 4),
            }

    agg.coverage = coverage  # type: ignore[attr-defined]
    return agg


def long_tail_rescue(
    per_route_topk: dict[str, set[str]],
    relevant_per_query: dict[str, set[str]],
    graph_route: str = "graph",
) -> dict[str, float]:
    """图谱路的长尾增量——图谱价值最硬的证据。

    只看「dense 与 sparse 在 top-k 内都没命中任何相关项」的 query，
    统计其中图谱路命中的比例。
    """
    others = [k for k in per_route_topk if k != graph_route]
    if not others:
        return {}

    hard_queries = [
        qid
        for qid, rel in relevant_per_query.items()
        if rel and all(not (per_route_topk[o] & rel) for o in others)
    ]
    if not hard_queries:
        return {"hard_queries": 0, "rescued": 0, "rescue_rate": 0.0}

    rescued = sum(
        1 for qid in hard_queries if per_route_topk.get(graph_route, set()) & relevant_per_query[qid]
    )
    return {
        "hard_queries": len(hard_queries),
        "rescued": rescued,
        "rescue_rate": round(rescued / len(hard_queries), 4),
        "note": "hard_queries = dense/sparse 的 top-k 内均未命中任何相关项的查询",
    }


# ---------------------------------------------------------------------------
# 手算对照用例（单测用）
# ---------------------------------------------------------------------------

_SELFTEST = [
    # retrieved,                  relevant,      k, 期望
    (["a", "b", "c"], {"a"}, 1, 1.0),
    (["b", "a", "c"], {"a"}, 1, 0.0),
    (["b", "a", "c"], {"a"}, 2, 1.0),
    (["a", "b", "c", "d"], {"a", "c"}, 2, 0.5),
    (["a", "b", "c", "d"], {"a", "c"}, 4, 1.0),
]


def _run_selftest() -> None:
    for retrieved, rel, k, expect in _SELFTEST:
        got = recall_at_k(retrieved, rel, k)
        assert abs(got - expect) < 1e-9, f"recall@k 错误：{retrieved} {rel} k={k} -> {got} != {expect}"

    # nDCG 手算：相关项 {a}，retrieved [a,b,c]
    # DCG = 1/log2(2) = 1 ; IDCG = 1 ; nDCG = 1
    assert abs(ndcg_at_k(["a", "b", "c"], {"a"}, 3) - 1.0) < 1e-9

    # 相关项 {a,b}，retrieved [b,x,a]
    # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 0.6309 = 1.6309
    expect = 1.5 / (1.0 + 1.0 / math.log2(3))
    assert abs(ndcg_at_k(["b", "x", "a"], {"a", "b"}, 3) - expect) < 1e-9

    # MRR：首个相关项在第 3 位 -> 1/3
    assert abs(reciprocal_rank(["x", "y", "a"], {"a"}) - 1 / 3) < 1e-9
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0

    print("metrics 自检通过")


if __name__ == "__main__":
    _run_selftest()