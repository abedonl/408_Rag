"""LLM 成本换算：token → 人民币（Week 3 可观测性）。

定价按模型名子串匹配（「deepseek/deepseek-chat」命中「deepseek-chat」），
未命中走 default。数值是估算口径，用于 trace 观测与成本讲解，不作计费依据。
价格来源为各厂商公开目录价，调整时只改这张表。
"""

from __future__ import annotations

# 每千 token 的人民币价格：(输入价, 输出价)
PRICING_CNY_PER_1K: dict[str, tuple[float, float]] = {
    "deepseek-reasoner": (0.004, 0.016),
    "deepseek-chat": (0.002, 0.008),
    "qwen-plus": (0.0008, 0.002),
    "qwen-turbo": (0.0003, 0.0006),
    "default": (0.002, 0.008),
}


def compute_cost_cny(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """单次/单轮问答的人民币成本估算。"""
    key = (model or "").lower()
    pin, pout = PRICING_CNY_PER_1K["default"]
    for prefix, price in PRICING_CNY_PER_1K.items():
        if prefix != "default" and prefix in key:
            pin, pout = price
            break
    cost = prompt_tokens / 1000 * pin + completion_tokens / 1000 * pout
    return round(cost, 6)
