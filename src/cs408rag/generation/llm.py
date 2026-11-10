"""LLM 客户端（OpenAI 兼容接口）。

用于生成层与二期 Agent。设计要点：

    - 中转站不稳定，必须带重试与超时，且失败要能降级而不是整条链路崩
    - 没有配置 API key 时**不能抛异常**：检索链路、评测、图谱都能独立跑通，
      生成层是可选增强。这样 CI 与他人复现不会因为缺 key 而卡住
    - 记录 token 消耗，供成本核算
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from loguru import logger

from cs408rag.config import settings


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    attempts: int = 1
    ok: bool = True
    error: str = ""


@dataclass
class LLMStats:
    calls: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "avg_ms": round(self.total_ms / self.calls, 1) if self.calls else 0.0,
        }


class LLMClient:
    """OpenAI 兼容客户端。"""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backup_model: str | None = None,
    ) -> None:
        self.model = model or settings.llm_model
        # 主模型空正文时降级用的模型（默认读取 LLM_BACKUP_MODEL）
        self.backup_model = (
            backup_model if backup_model is not None else settings.llm_backup_model
        )
        self.api_key = api_key if api_key is not None else settings.openai_api_key
        self.base_url = base_url or settings.openai_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.stats = LLMStats()
        self._client: Any = None
        # 供应商不支持思考开关参数时会置 True，之后不再传
        self._thinking_unsupported = False

    def _extra_body(self, model: str | None = None) -> dict[str, Any]:
        """思考模式开关。

        ★ 实测商汤 sensenova 只认**对象**写法：
            thinking: {"type": "disabled"}   -> 接受，长 prompt 下 reasoning 明显下降
            thinking: "disabled"（字符串）   -> 400 invalid arguments
            output_config: {"effort": "none"} -> 接受，但降思考效果不稳定，
                                                 与 thinking 同传反而更高
        因此只传 thinking 对象。遇到不认该参数的供应商会 400，
        客户端置 _thinking_unsupported 后去掉参数重试。
        """
        if self._thinking_unsupported:
            return {}
        # 不同模型认的参数不一样（实测）：
        #   sensenova-6.8-flash-lite: thinking={"type": "disabled"}
        #   deepseek-v4-flash:        reasoning_effort="none"（thinking 对象会 400）
        model_name = (model or self.model).lower()
        if "deepseek" in model_name or "glm" in model_name:
            return {
                "extra_body": {
                    "reasoning_effort": "high" if settings.llm_enable_thinking else "none"
                }
            }
        if settings.llm_enable_thinking:
            return {
                "extra_body": {
                    "thinking": {"type": "enabled"},
                    "output_config": {"effort": "high"},
                }
            }
        return {"extra_body": {"thinking": {"type": "disabled"}}}

    @property
    def available(self) -> bool:
        """是否配置了可用的 key。未配置时所有调用返回 ok=False，不抛异常。"""
        return bool(
            self.api_key and self.api_key.strip() and not self.api_key.startswith("sk-your")
        )

    def _ensure(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        if not self.available:
            return LLMResponse(
                text="",
                model=self.model,
                ok=False,
                error="未配置 OPENAI_API_KEY，生成层已跳过（检索与评测不受影响）",
            )

        last_err = ""
        effective_max = max_tokens
        budget_bumped = False
        for attempt in range(1, self.max_retries + 1):
            t0 = time.perf_counter()
            try:
                kw: dict[str, Any] = {}
                if json_mode:
                    kw["response_format"] = {"type": "json_object"}
                kw.update(self._extra_body())
                if timeout:
                    # 单次覆盖：规划类长 prompt 在默认超时内跑不完
                    kw["timeout"] = timeout
                resp = self._ensure().chat.completions.create(
                    model=self.model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=effective_max,
                    **kw,
                )
                took = (time.perf_counter() - t0) * 1000
                usage = getattr(resp, "usage", None)
                text = (resp.choices[0].message.content or "").strip()
                out = LLMResponse(
                    text=text,
                    model=self.model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    latency_ms=took,
                    attempts=attempt,
                )
                # ★ 强制思考的模型（实测 sensenova-6.8-flash-lite）会把
                # max_tokens 全部花在 reasoning 上，content 返回空。
                # 表现是 ok=True 但没有回答，比报错更难发现。
                # 这里检测「预算耗尽且正文为空」，翻倍预算重试一次。
                if (
                    not text
                    and out.completion_tokens >= effective_max * 0.9
                    and not budget_bumped
                ):
                    budget_bumped = True
                    effective_max = min(effective_max * 2, 6000)
                    logger.warning(
                        f"模型返回空正文（completion={out.completion_tokens}，"
                        f"可能是思考 token 吃光预算），max_tokens 提到 {effective_max} 重试"
                    )
                    if attempt < self.max_retries:
                        self.stats.calls += 1
                        self.stats.prompt_tokens += out.prompt_tokens
                        self.stats.completion_tokens += out.completion_tokens
                        self.stats.total_ms += took
                        continue
                if not text and self.backup_model and self.backup_model != self.model:
                    backup = self._try_backup(
                        messages,
                        temperature=temperature,
                        max_tokens=max(1600, max_tokens),
                    )
                    if backup is not None:
                        self.stats.calls += 1
                        self.stats.prompt_tokens += out.prompt_tokens + backup.prompt_tokens
                        self.stats.completion_tokens += (
                            out.completion_tokens + backup.completion_tokens
                        )
                        self.stats.total_ms += took + backup.latency_ms
                        return backup
                self.stats.calls += 1
                self.stats.prompt_tokens += out.prompt_tokens
                self.stats.completion_tokens += out.completion_tokens
                self.stats.total_ms += took
                return out
            except Exception as e:  # noqa: BLE001 - 中转站错误种类多，统一重试
                last_err = str(e)
                # 供应商不认思考开关 -> 去掉参数再试
                if (
                    "thinking" in last_err or "reasoning_effort" in last_err or "400" in last_err
                ) and not self._thinking_unsupported:
                    if not self._thinking_unsupported:
                        logger.warning("供应商不支持思考开关参数，改为不带该参数重试")
                        self._thinking_unsupported = True
                        continue
                logger.warning(
                    f"LLM 调用失败（第 {attempt}/{self.max_retries} 次）：{last_err[:160]}"
                )
                if attempt < self.max_retries:
                    # 429 / rpm exhausted 是配额限制，不是网络抖动。
                    # 短退避（1.5^n 秒）救不回来：实测连续 10 次调用只成功 2 次。
                    # 退避 20/40/60 秒，让下一分钟窗口恢复。
                    rate_limited = (
                        "429" in last_err
                        or "rpm" in last_err.lower()
                        or "quota" in last_err.lower()
                    )
                    time.sleep(20 * attempt if rate_limited else 1.5**attempt)

        self.stats.calls += 1
        self.stats.failures += 1
        return LLMResponse(text="", model=self.model, ok=False, error=last_err)

    def _try_backup(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse | None:
        """主模型空正文时，换备用模型再试一次。

        只用于「主模型已失败」的情况，不参与正常调用路径。
        """
        t0 = time.perf_counter()
        try:
            resp = self._ensure().chat.completions.create(
                model=self.backup_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **self._extra_body(self.backup_model),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"备用模型 {self.backup_model} 调用失败：{str(e)[:120]}")
            return None
        took = (time.perf_counter() - t0) * 1000
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            logger.warning(f"备用模型 {self.backup_model} 也返回空正文")
            return None
        usage = getattr(resp, "usage", None)
        logger.warning(f"主模型空正文，已由备用模型 {self.backup_model} 补答")
        return LLMResponse(
            text=text,
            model=self.backup_model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            latency_ms=took,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> Iterator[str]:
        """流式生成，逐段 yield 文本。供 SSE 接口使用。"""
        if not self.available:
            yield "[未配置 LLM API key，无法生成回答。检索结果仍然可用。]"
            return

        try:
            kw: dict[str, Any] = {"stream": True, **self._extra_body()}
            stream = self._ensure().chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **kw,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # 供应商不认思考开关 -> 去掉再试一次
            if (
                "thinking" in msg or "reasoning_effort" in msg or "400" in msg
            ) and not self._thinking_unsupported:
                logger.warning("供应商不支持思考开关，流式改为不带该参数重试")
                self._thinking_unsupported = True
                try:
                    stream = self._ensure().chat.completions.create(
                        model=self.model,
                        messages=messages,  # type: ignore[arg-type]
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices else None
                        if delta:
                            yield delta
                    return
                except Exception as e2:  # noqa: BLE001
                    msg = str(e2)
            logger.warning(f"流式生成失败：{msg}")
            yield f"\n[生成中断：{msg}]"
        finally:
            self.stats.calls += 1


_client: LLMClient | None = None


def get_llm(model: str | None = None) -> LLMClient:
    global _client
    if _client is None or (model and model != _client.model):
        _client = LLMClient(model=model)
    return _client


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    c = LLMClient()
    print("available:", c.available)
    r = c.chat([{"role": "user", "content": "用一句话解释什么是 Cache 的组相联映射。"}])
    print("ok:", r.ok, "| error:", r.error[:120])
    print("text:", r.text[:200])
    print("stats:", c.stats.to_dict())