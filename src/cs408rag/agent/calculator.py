"""计算器工具（基于 simpleeval）。

408 计算题（Cache 命中率、页面置换、TCP 拥塞窗口、码距）LLM 心算必错，
且错得很隐蔽 —— 数字看起来合理但差一位或差一个量级。
把算式交给确定性求值器，是 Agent 在计算题上真正能带来增益的地方。

安全边界：

- 求值交给 `simpleeval`，不再自维护 AST 白名单，也不保留 `eval()` 回退。
  包不可用时 **fail closed**（返回 ok=False），不偷偷降级。
- 前置检查（长度/字符集/禁用词/属性访问）只是纵深防御，真正拒绝未知名字、
  函数与语法的是 simpleeval。
- 幂运算上限压到 10000：默认的 4_000_000 允许 `99999**99999` 这类大整数
  运算长时间占满 CPU，`9**9**9` 虽然会被拒，但仍不够保守。
"""

from __future__ import annotations

import math
import re
from typing import Any

try:  # fail closed：没装包就不求值
    import simpleeval

    # simpleeval 1.0.x 没有构造函数参数，safe_power 在调用时读模块全局。
    simpleeval.MAX_POWER = 10_000
    _IMPORT_ERROR = ""
except Exception as e:  # noqa: BLE001 - 任何导入失败都按不可用处理
    simpleeval = None  # type: ignore[assignment]
    _IMPORT_ERROR = str(e)

_MAX_EXPR_LEN = 500

_FUNCS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "int": int,
    "float": float,
    "ceil": math.ceil,
    "floor": math.floor,
    "sqrt": math.sqrt,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}

_NAMES: dict[str, Any] = {"pi": math.pi, "e": math.e}

# 允许的字符集：数字、字母、四则、幂、括号、逗号、空白。
# 正白名单比黑名单更难漏 —— 字符串、容器、属性、比较、赋值都不在集合里。
_ALLOWED_CHARS = re.compile(r"^[0-9A-Za-z_+\-*/%().,\s]+$")
_ATTR_ACCESS = re.compile(r"\.[A-Za-z_]")
_FORBIDDEN_WORDS = re.compile(
    r"\b(import|lambda|eval|exec|open|getattr|setattr|globals|locals|compile|"
    r"__builtins__|system|popen)\b"
)

# 408 常用单位换算。只做前缀缩放，不擅自把字节换成比特：
# 4KB -> 4096（Byte），4Kb -> 4096（bit）。是否再 ×8 由使用方决定。
_UNITS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "k": 1000,
}

RE_UNIT_EXPR = re.compile(
    r"(\d+(?:\.\d+)?)\s*(K|M|G|T|k)\s*(Byte|bytes|bit|B|b)\b"
)


def normalize_expression(expr: str) -> str:
    """统一全角符号与常见写法，避免 LLM 输出直接触发语法错误。"""
    return (
        (expr or "")
        .strip()
        .replace("×", "*")
        .replace("÷", "/")
        .replace("−", "-")
        .replace("（", "(")
        .replace("）", ")")
        .replace("，", ",")
        .replace("^", "**")
    )


def validate_expression(expr: str) -> str | None:
    """前置护栏。返回错误文案；None 表示可以交给 simpleeval。

    这个函数也被 `ToolGuard` 用来在预算/执行之前挡住恶意表达式，
    所以它不能依赖 simpleeval 的求值行为。
    """
    if not expr:
        return "表达式为空"
    if len(expr) > _MAX_EXPR_LEN:
        return "表达式过长"
    if not _ALLOWED_CHARS.match(expr):
        return "表达式含不允许的字符"
    if _ATTR_ACCESS.search(expr):
        return "不允许属性访问"
    if _FORBIDDEN_WORDS.search(expr):
        return "表达式含禁用词"
    return None


def evaluate(expr: str) -> dict[str, Any]:
    """求值一个算式。返回结果与规范化后的表达式。

    支持：+ - * / // % ** 括号、0b/0x/0o 字面量、log2/sqrt/ceil 等函数。
    不支持：变量赋值、字符串、属性访问、非白名单函数。
    """
    normalized = normalize_expression(expr)
    expanded = convert_units(normalized)["expr"]
    problem = validate_expression(expanded)
    if problem:
        return {"ok": False, "expr": expanded, "error": problem}

    if simpleeval is None:
        return {
            "ok": False,
            "expr": expanded,
            "error": f"计算器依赖 simpleeval 不可用：{_IMPORT_ERROR}",
        }

    try:
        value = simpleeval.SimpleEval(functions=_FUNCS, names=_NAMES).eval(expanded)
    except ZeroDivisionError:
        return {"ok": False, "expr": expanded, "error": "除零"}
    except OverflowError:
        return {"ok": False, "expr": expanded, "error": "数值溢出"}
    except Exception as e:  # noqa: BLE001 - 求值失败统一归一化，不抛给 Agent
        name = type(e).__name__
        if "NumberTooHigh" in name or "Power" in str(e):
            return {"ok": False, "expr": expanded, "error": "幂运算超出安全范围"}
        return {"ok": False, "expr": expanded, "error": f"无法计算：{str(e)[:200]}"}

    if isinstance(value, complex):
        return {"ok": False, "expr": expanded, "error": "结果不是实数"}
    if not isinstance(value, (int, float)):
        return {"ok": False, "expr": expanded, "error": "结果不是数值"}

    return {
        "ok": True,
        "expr": expanded,
        "value": value,
        "is_int": isinstance(value, int) or (isinstance(value, float) and value.is_integer()),
        "display": _display(value),
    }


def _display(v: Any) -> str:
    if isinstance(v, float):
        if v.is_integer():
            return str(int(v))
        return f"{v:.6g}"
    return str(v)


# ---------------------------------------------------------------------------
# 单位换算
# ---------------------------------------------------------------------------


def convert_units(expr: str) -> dict[str, Any]:
    """把 4KB / 10Mb 这类写法按 1024 进制展开成纯数字。

    只处理 408 高频的存储/速率前缀，不追求通用。字节与比特保持各自的
    计数单位，不做隐式 ×8 —— 隐式换算更容易制造「差 8 倍」的错误。
    """

    def repl(m: re.Match[str]) -> str:
        num = float(m.group(1))
        prefix = _UNITS[m.group(2)]
        value = num * prefix
        return str(int(value)) if value.is_integer() else str(value)

    converted = RE_UNIT_EXPR.sub(repl, expr)
    return {"expr": converted, "changed": converted != expr}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    cases = [
        "2**10",
        "1 - 0.05**3",
        "log2(4096)",
        "(1 + 2) * 3 / 4",
        "0b1010 + 0x1F",
        "4KB + 10Mb",
        "__import__('os').system('echo hacked')",
        "9**9**9",
    ]
    for c in cases:
        print(c, "=>", evaluate(c))