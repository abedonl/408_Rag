"""计算器工具。

408 计算题（Cache 命中率、页面置换、TCP 拥塞窗口、码距）LLM 心算必错，
且错得很隐蔽 —— 数字看起来合理但差一位或差一个量级。
把算式交给确定性求值器，是 Agent 在计算题上真正能带来增益的地方。

安全：只允许白名单 AST 节点，不放行属性访问、下标、赋值、lambda。
不使用裸 eval 的字符串限制（黑名单永远漏），而是编译后逐节点校验。
"""

from __future__ import annotations

import ast
import math
import re
from typing import Any

_MAX_EXPR_LEN = 500
_MAX_NODES = 200

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
    ast.USub,
)

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
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
}

# 408 常用单位换算。LLM 经常把 K=1024 与 K=1000 搞混，
# 4096 Byte 说成 4*10^3，或把 Mbps 的 M 当成 2^20。
_UNITS = {
    "K": 1024,
    "M": 1024**2,
    "G": 1024**3,
    "T": 1024**4,
    "k": 1000,
    "m": 1 / 1000,
}


def _validate(tree: ast.AST) -> None:
    nodes = list(ast.walk(tree))
    if len(nodes) > _MAX_NODES:
        raise ValueError("表达式过于复杂")
    for n in nodes:
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"不允许的语法：{type(n).__name__}")
        if isinstance(n, ast.Call):
            if not isinstance(n.func, ast.Name) or n.func.id not in _FUNCS:
                raise ValueError("只允许调用白名单函数")
        if isinstance(n, ast.Name) and n.id not in _FUNCS:
            raise ValueError(f"未知标识符：{n.id}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError("只允许数值常量")


def evaluate(expr: str) -> dict[str, Any]:
    """求值一个算式。返回结果与规范化后的表达式。

    支持：+ - * / // % ** 括号、0b/0x/0o 字面量、log2/sqrt/ceil 等函数。
    不支持：变量赋值、字符串、属性访问。
    """
    expr = (expr or "").strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    if not expr:
        return {"ok": False, "error": "表达式为空"}
    if len(expr) > _MAX_EXPR_LEN:
        return {"ok": False, "error": "表达式过长"}

    try:
        tree = ast.parse(expr, mode="eval")
        _validate(tree)
        value = eval(  # noqa: S307 - AST 已白名单校验，globals 为空
            compile(tree, "<calc>", "eval"),
            {"__builtins__": {}},
            _FUNCS,
        )
    except ZeroDivisionError:
        return {"ok": False, "expr": expr, "error": "除零"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "expr": expr, "error": f"无法计算：{e}"}

    return {
        "ok": True,
        "expr": expr,
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

RE_UNIT_EXPR = re.compile(
    r"(\d+(?:\.\d+)?)\s*(K|M|G|T|k)\s*(B|Byte|bytes|bit|b)\b", re.I
)


def convert_units(expr: str) -> dict[str, Any]:
    """把 4KB / 10Mbps 这类写法展开成纯数字表达式。

    只处理 408 高频的存储/速率单位，不追求通用。
    """
    def repl(m: re.Match[str]) -> str:
        num = float(m.group(1))
        prefix = _UNITS[m.group(2)]
        unit = m.group(3).lower()
        base = 8 if unit in ("bit", "b") else 1
        return f"({num * prefix * base})"

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
        "__import__('os').system('echo hacked')",
    ]
    for c in cases:
        print(c, "=>", evaluate(c))
    print(convert_units("4KB + 10Mbps"))