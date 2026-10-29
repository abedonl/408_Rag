"""多路召回结果的融合。

实现两种策略，用于**对照实验**：

    RRF（Reciprocal Rank Fusion）
        score(d) = Σ_r  w_r / (k + rank_r(d))
        只用排名信息，不看原始分数，因此对异构分数量纲天然鲁棒。

    加权融合（Weighted Score Fusion）
        先把各路分数归一化到同一量纲，再按权重线性求和。
        多了一步归一化，且对分数分布漂移敏感。

对照的价值不在「哪个更强」，而在能讲清楚**为什么**：
稠密相似度、稀疏点积、图谱命中强度三者的分数分布完全不同尺度，
加权融合必须额外做归一化，而归一化本身会引入对分布假设的依赖。

参考：Cormack et al. 2009, Reciprocal Rank Fusion outperforms Condorcet and
      individual rank learning methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

NormalizeMethod = Literal["minmax", "zscore", "rank"]


@dataclass
class FusedItem:
    doc_id: str
    score: float
    # 每一路的原始排名（1-based），便于 trace 展示「RRF 前后排名变化」
    ranks: dict[str, int]
    raw_scores: dict[str, float]


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    *,
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[FusedItem]:
    """倒数排名融合。

    Args:
        rankings: {路名: [doc_id 按相关性降序]}
        k: 平滑常数。越大则高排名与低排名的差距越小（默认 60，论文常用值）
        weights: 各路权重，缺省为 1.0

    Returns:
        按融合分数降序的 FusedItem 列表
    """
    if k <= 0:
        raise ValueError("k 必须为正数")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for name, docs in rankings.items():
        w = (weights or {}).get(name, 1.0)
        if w == 0:
            continue
        for rank, doc_id in enumerate(docs, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
            ranks.setdefault(doc_id, {})[name] = rank

    return [
        FusedItem(doc_id=d, score=s, ranks=ranks[d], raw_scores={})
        for d, s in sorted(scores.items(), key=lambda kv: -kv[1])
    ]


# ---------------------------------------------------------------------------
# 加权分数融合
# ---------------------------------------------------------------------------


def _normalize(values: np.ndarray, method: NormalizeMethod) -> np.ndarray:
    if len(values) == 0:
        return values
    if method == "minmax":
        lo, hi = float(values.min()), float(values.max())
        if hi - lo < 1e-12:
            return np.zeros_like(values)
        return (values - lo) / (hi - lo)
    if method == "zscore":
        mu, sd = float(values.mean()), float(values.std())
        if sd < 1e-12:
            return np.zeros_like(values)
        z = (values - mu) / sd
        # 压到 [0,1]，避免负分与量纲问题
        return 1.0 / (1.0 + np.exp(-z))
    if method == "rank":
        order = np.argsort(np.argsort(-values))
        n = len(values)
        return 1.0 - order / max(n - 1, 1)
    raise ValueError(f"未知归一化方法：{method}")


def weighted_score_fusion(
    scored: dict[str, list[tuple[str, float]]],
    *,
    weights: dict[str, float] | None = None,
    normalize: NormalizeMethod = "minmax",
) -> list[FusedItem]:
    """归一化后按权重求和。

    Args:
        scored: {路名: [(doc_id, score) 按分数降序]}
        weights: 各路权重，缺省为 1.0
        normalize: 归一化方法。默认 minmax；zscore 对离群点更稳但依赖分布假设
    """
    fused: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}
    raws: dict[str, dict[str, float]] = {}

    for name, items in scored.items():
        w = (weights or {}).get(name, 1.0)
        if w == 0 or not items:
            continue

        docs = [d for d, _ in items]
        vals = np.asarray([s for _, s in items], dtype=np.float64)
        norm = _normalize(vals, normalize)

        for i, doc_id in enumerate(docs):
            fused[doc_id] = fused.get(doc_id, 0.0) + w * float(norm[i])
            ranks.setdefault(doc_id, {})[name] = i + 1
            raws.setdefault(doc_id, {})[name] = float(vals[i])

    return [
        FusedItem(doc_id=d, score=s, ranks=ranks[d], raw_scores=raws[d])
        for d, s in sorted(fused.items(), key=lambda kv: -kv[1])
    ]


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------

FusionStrategy = Literal["rrf", "weighted"]


def fuse(
    rankings: dict[str, list[str]],
    *,
    strategy: FusionStrategy = "rrf",
    k: int = 60,
    weights: dict[str, float] | None = None,
    scored: dict[str, list[tuple[str, float]]] | None = None,
    normalize: NormalizeMethod = "minmax",
) -> list[FusedItem]:
    """按指定策略融合。

    RRF 只需要 rankings；加权融合需要 scored（含原始分数），
    若未提供则用 rankings 的序号当分数（等价于 rank 归一化）。
    """
    if strategy == "rrf":
        return reciprocal_rank_fusion(rankings, k=k, weights=weights)

    if strategy == "weighted":
        if scored is None:
            # 退化为按排名打分，仍可跑通，但会丢失分数区分的优势
            scored = {
                name: [(d, float(len(docs) - i)) for i, d in enumerate(docs)]
                for name, docs in rankings.items()
            }
        return weighted_score_fusion(scored, weights=weights, normalize=normalize)

    raise ValueError(f"未知融合策略：{strategy}")


# ---------------------------------------------------------------------------
# 诊断
# ---------------------------------------------------------------------------


def contribution_stats(
    per_route: dict[str, list[str]],
    fused: list[FusedItem],
    top_n: int = 10,
) -> dict[str, dict[str, float]]:
    """统计各路对最终 top-N 的贡献。

    用途有二：
      1. 回答「图谱路到底有没有用」——若它在 top-10 里贡献占比极低，就是装饰品
      2. 单独量化长尾增量：dense 与 sparse 都没召回到、只有图谱路召回的比例
    """
    top = fused[:top_n]
    stats: dict[str, dict[str, float]] = {}

    for name, docs in per_route.items():
        docs_set = set(docs)
        hit = sum(1 for f in top if name in f.ranks)
        stats[name] = {
            "in_top_n": hit,
            "share": round(hit / len(top), 4) if top else 0.0,
            "recall_pool": len(docs_set),
        }

    # 长尾增量：其他路都失败，只有这一路在 top-N 里命中
    if len(per_route) > 1:
        for name in per_route:
            others = set().union(*(set(v) for k, v in per_route.items() if k != name))
            rescued = sum(1 for f in top if name in f.ranks and f.doc_id not in others)
            stats[name]["unique_rescue"] = rescued
            stats[name]["unique_rescue_share"] = round(rescued / len(top), 4) if top else 0.0

    return stats