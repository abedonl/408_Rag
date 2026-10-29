"""模型基准测试。

用同一个 408 问题实测各模型的**首字延迟（TTFT）**、总耗时与生成速度，
挑出最快的换上去。

为什么单独测 TTFT：
    本项目的 endpoint 实测存在「TTFT 随 max_tokens 跳变」的现象
    （max_tokens=50 时 2.9s，150 以上直接 42s+）。总耗时不能反映用户实际体感，
    用户感受到的是「多久开始出字」。所以 TTFT 是首要指标。

用法：
    python scripts/bench_models.py                       # 测默认候选集
    python scripts/bench_models.py --models a,b,c
    python scripts/bench_models.py --list                # 列出 endpoint 上全部模型
    python scripts/bench_models.py --max-tokens 500 --repeat 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field

import httpx
from loguru import logger

from cs408rag.config import settings

# 默认候选：商汤 sensenova endpoint 实测可用的模型（用 --list 可重新获取）。
# 换 endpoint 后这个列表会失效，先跑 --list 再更新。
DEFAULT_MODELS = [
    "sensenova-6.8-flash-lite",
    "sensenova-u1.5-lite",
    "deepseek-flash",
    "deepseek-v4-flash",
    "deepseek-v4.1-flash",
    "glm-5.2",
    "kimi-k3",
]

# 固定测试问题：包含术语 + 要求给例子，接近真实使用
PROMPT = "采用组相联映射时主存地址如何划分？请给出划分依据并举例说明。"
SYSTEM = "你是计算机考研 408 的助教，回答简洁准确，用中文。"


@dataclass
class BenchResult:
    model: str
    ok: bool = False
    ttft_s: float = 0.0
    total_s: float = 0.0
    chars: int = 0
    chunks: int = 0
    completion_tokens: int = 0
    reasoning_chunks: int = 0
    error: str = ""

    @property
    def chars_per_s(self) -> float:
        return self.chars / self.total_s if self.total_s > 0 else 0.0


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }


def _url() -> str:
    return settings.openai_base_url.rstrip("/") + "/chat/completions"


def list_models() -> list[str]:
    r = httpx.get(
        settings.openai_base_url.rstrip("/") + "/models", headers=_headers(), timeout=30
    )
    r.raise_for_status()
    return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))


def bench_one(model: str, *, max_tokens: int = 400, timeout: float = 120.0) -> BenchResult:
    """测一个模型。先发一次极短请求预热，避免冷启动污染结果。"""
    res = BenchResult(model=model)
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }

    try:
        # 预热（不计量）
        with httpx.stream(
            "POST",
            _url(),
            headers=_headers(),
            json={**body, "max_tokens": 1, "stream": False},
            timeout=timeout,
        ) as r:
            r.read()
            if r.status_code >= 400:
                res.error = f"预热 HTTP {r.status_code}: {r.text[:120]}"
                return res

        t0 = time.perf_counter()
        first = None
        with httpx.stream(
            "POST", _url(), headers=_headers(), json=body, timeout=timeout
        ) as r:
            if r.status_code >= 400:
                res.error = f"HTTP {r.status_code}: {r.text[:160]}"
                return res
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = j.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                if delta.get("reasoning_content"):
                    res.reasoning_chunks += 1
                if delta.get("content"):
                    if first is None:
                        first = time.perf_counter() - t0
                    res.chunks += 1
                    res.chars += len(delta["content"])
                usage = j.get("usage") or {}
                if usage.get("completion_tokens"):
                    res.completion_tokens = usage["completion_tokens"]

        res.total_s = time.perf_counter() - t0
        res.ttft_s = first if first is not None else res.total_s
        res.ok = res.chars > 0
        if not res.ok:
            res.error = "未收到任何 content 内容"
    except Exception as e:  # noqa: BLE001
        res.error = str(e)[:200]
    return res


def _print_table(results: list[BenchResult]) -> None:
    print()
    print(f"{'模型':<32} {'TTFT':>8} {'总耗时':>8} {'字数':>6} {'字/秒':>8} {'思考块':>7}  状态")
    print("-" * 96)
    ok = [r for r in results if r.ok]
    for r in sorted(results, key=lambda x: (not x.ok, x.ttft_s)):
        mark = "✓" if r.ok else "✗"
        if r.ok:
            print(
                f"{mark} {r.model:<30} {r.ttft_s:>7.1f}s {r.total_s:>7.1f}s "
                f"{r.chars:>6} {r.chars_per_s:>7.1f} {r.reasoning_chunks:>7}"
            )
        else:
            print(f"{mark} {r.model:<30} {'-':>8} {'-':>8} {'-':>6} {'-':>8} {'-':>7}  {r.error[:40]}")
    print("-" * 96)

    if ok:
        best_ttft = min(ok, key=lambda r: r.ttft_s)
        best_speed = max(ok, key=lambda r: r.chars_per_s)
        print(f"\n首字最快：{best_ttft.model}  （TTFT {best_ttft.ttft_s:.1f}s）")
        print(f"生成最快：{best_speed.model}  （{best_speed.chars_per_s:.1f} 字/秒）")
        # 综合：TTFT 占七成权重（用户最敏感），速度占三成
        def score(r: BenchResult) -> float:
            return r.ttft_s * 0.7 + (100.0 / max(r.chars_per_s, 0.1)) * 0.3

        best = min(ok, key=score)
        print(f"综合最佳：{best.model}")
        print(f"\n换上去：把 .env 里的 LLM_MODEL 和 LLM_MODEL_CHEAP 都改成 {best.model}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    ap = argparse.ArgumentParser(description="模型基准测试")
    ap.add_argument("--models", default="", help="逗号分隔；留空用默认候选集")
    ap.add_argument("--list", action="store_true", help="列出 endpoint 上全部模型")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    if args.list:
        for m in list_models():
            print(m)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()] or DEFAULT_MODELS

    print(f"endpoint: {settings.openai_base_url}")
    print(f"测试问题: {PROMPT}")
    print(f"max_tokens: {args.max_tokens} | 候选 {len(models)} 个")
    print("（每个模型先发一次极短请求预热，再计时）\n")

    results: list[BenchResult] = []
    for i, m in enumerate(models, 1):
        print(f"[{i}/{len(models)}] {m} …", end=" ", flush=True)
        r = bench_one(m, max_tokens=args.max_tokens, timeout=args.timeout)
        results.append(r)
        print(f"TTFT {r.ttft_s:.1f}s / 总 {r.total_s:.1f}s" if r.ok else f"失败: {r.error[:50]}")

    _print_table(results)


if __name__ == "__main__":
    main()