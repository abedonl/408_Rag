"""工具护栏：白名单 → 参数校验 → 工具特定检查 → 执行 → 超时/错误归一化。

顺序是有意的：

1. **白名单**：不在 `TOOL_CONTRACTS` 里的名字直接拒绝，planner 与 LLM
   fallback 都不能绕过。
2. **预算**：单轮最多 6 次、每分钟最多 60 次（Week 1 是进程级窗口，
   Week 2 接入 session 后按会话分桶）；超限返回
   `error_type="budget"`，不静默重试。
3. **Pydantic 参数**：失败返回 `error_type="validation"`，planner 可以据此
   修正参数重试，而不是把坏参数吞成「工具没用」。
4. **工具特定护栏**：计算器表达式在进入求值器之前先做一次前置检查，
   即使 planner 传了恶意表达式也到不了 simpleeval。
5. **执行 + 超时**：工具跑在独立线程池里，超时返回 `error_type="timeout"`，
   主流程不再等待。Python 线程无法强杀 —— 超时线程可能继续跑完，
   这是已知限制，不假装能终止它。
6. **审计**：每次调用写一条 `ToolResult`，只记录 `args_digest`（参数短哈希），
   不落原始参数。

自动重试只对 `idempotent=True` 且 `retries>0` 的工具生效；本项目的工具
都是只读的，但慢的外部查询（检索/图/图解析）retries=0，避免延迟翻倍。
"""

from __future__ import annotations

import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from typing import TYPE_CHECKING, Any

from loguru import logger
from pydantic import ValidationError

from cs408rag.agent.calculator import validate_expression
from cs408rag.agent.contracts import (
    TOOL_SPECS,
    ToolResult,
    ToolSpec,
    args_digest,
    classify_error,
)

if TYPE_CHECKING:
    from cs408rag.agent.tools import AgentTools


class ToolGuard:
    """包装 `AgentTools._execute`，所有工具调用都必须经过这里。"""

    def __init__(
        self,
        tools: AgentTools,
        *,
        specs: dict[str, ToolSpec] | None = None,
        max_calls_per_round: int = 6,
        max_calls_per_minute: int = 60,
    ) -> None:
        self.tools = tools
        self.specs = specs or TOOL_SPECS
        self.max_calls_per_round = max_calls_per_round
        self.max_calls_per_minute = max_calls_per_minute
        self._round_no = 0
        self._round_calls = 0
        self._minute_calls: deque[float] = deque()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cs408tool")

    # -- 生命周期 ---------------------------------------------------------

    def reset(self) -> None:
        """完全重置：新会话/新实例用。"""
        self._round_no = 0
        self._round_calls = 0
        self._minute_calls.clear()

    def reset_round(self) -> None:
        """新一轮运行：清轮次计数，但保留每分钟窗口。

        每分钟预算是跨请求的速率限制；如果每次 run 都清空，
        连续请求就能绕过它。Week 2 接入 session 后再按 session 分桶。
        """
        self._round_no = 0
        self._round_calls = 0

    def begin_round(self, round_no: int) -> None:
        self._round_no = round_no
        self._round_calls = 0

    # -- 入口 -------------------------------------------------------------

    def call(
        self,
        name: str,
        *,
        round_no: int | None = None,
        count_budget: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        spec = self.specs.get(name)
        if spec is None:
            return self._record(
                ToolResult(
                    tool=name,
                    ok=False,
                    error=f"未知工具 {name!r}（不在白名单）",
                    error_type="unknown_tool",
                    validated=False,
                    round=self._round_no if round_no is None else round_no,
                )
            )

        rnd = self._round_no if round_no is None else round_no
        if count_budget:
            budget_error = self._consume_budget()
            if budget_error:
                return self._record(
                    ToolResult(
                        tool=name,
                        ok=False,
                        error=budget_error,
                        error_type="budget",
                        validated=False,
                        round=rnd,
                    )
                )

        digest = args_digest(kwargs)
        try:
            validated = spec.args_model.model_validate(kwargs)
        except ValidationError as e:
            return self._record(
                ToolResult(
                    tool=name,
                    ok=False,
                    error=_short_validation_error(e),
                    error_type="validation",
                    validated=False,
                    round=rnd,
                    args_digest=digest,
                )
            )

        args = validated.model_dump()
        if name == "calculate":
            problem = validate_expression(str(args.get("expr") or ""))
            if problem:
                return self._record(
                    ToolResult(
                        tool=name,
                        ok=False,
                        error=problem,
                        error_type="validation",
                        validated=False,
                        round=rnd,
                        args_digest=digest,
                    )
                )

        return self._execute(spec, args, rnd, digest)

    # -- 内部 -------------------------------------------------------------

    def _consume_budget(self) -> str:
        now = time.monotonic()
        while self._minute_calls and now - self._minute_calls[0] >= 60.0:
            self._minute_calls.popleft()

        if self._round_calls >= self.max_calls_per_round:
            return (
                f"单轮工具调用超过上限 {self.max_calls_per_round} 次，"
                "本轮后续调用被拒绝"
            )
        if len(self._minute_calls) >= self.max_calls_per_minute:
            return (
                f"单会话每分钟工具调用超过上限 {self.max_calls_per_minute} 次，"
                "请稍后再试"
            )
        self._round_calls += 1
        self._minute_calls.append(now)
        return ""

    def _execute(
        self,
        spec: ToolSpec,
        args: dict[str, Any],
        rnd: int,
        digest: str,
    ) -> ToolResult:
        attempts = 1 + (spec.retries if spec.idempotent else 0)
        t0 = time.perf_counter()
        last_error = ""
        last_type = "internal"

        for attempt in range(1, attempts + 1):
            try:
                future = self._executor.submit(self.tools._execute, spec.name, args)
                data, summary = future.result(timeout=spec.timeout_s)
            except FutureTimeout:
                future.cancel()
                last_error = f"工具超时（>{spec.timeout_s:g}s）"
                last_type = "timeout"
                logger.warning(f"工具 {spec.name} 超时，已停止等待")
            except Exception as e:  # noqa: BLE001 - 工具失败降级，不阻断 Agent
                last_error = str(e)[:300] or type(e).__name__
                last_type = classify_error(e)
                logger.warning(f"工具 {spec.name} 失败（{last_type}）：{e}")
            else:
                return self._record(
                    ToolResult(
                        tool=spec.name,
                        ok=True,
                        data=data,
                        summary=summary,
                        took_ms=(time.perf_counter() - t0) * 1000,
                        validated=True,
                        attempt=attempt,
                        round=rnd,
                        args_digest=digest,
                    )
                )

            # 只对幂等工具、且是可重试的错误类型重试
            if last_type not in ("timeout", "internal") or attempt >= attempts:
                break

        return self._record(
            ToolResult(
                tool=spec.name,
                ok=False,
                error=last_error,
                error_type=last_type,
                took_ms=(time.perf_counter() - t0) * 1000,
                validated=True,
                attempt=attempts,
                round=rnd,
                args_digest=digest,
            )
        )

    def _record(self, result: ToolResult) -> ToolResult:
        self.tools.trace.add(result)
        if not result.ok:
            self.tools.failures[result.tool] = self.tools.failures.get(result.tool, 0) + 1
        return result


def _short_validation_error(e: ValidationError) -> str:
    parts: list[str] = []
    for err in e.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "参数校验失败：" + "；".join(parts)