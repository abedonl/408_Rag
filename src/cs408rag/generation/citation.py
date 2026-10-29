"""引用校验与押题扫描。

两件事：

1. **引用溯源** —— 抽取回答里的 [编号] 与真题编号，校验其真实存在，
   计算 Citation Precision / Recall。生成的回答若引用了上下文里没有的编号，
   说明它在编，这个信号必须被量化。

2. **押题扫描**（红线）—— 扫描预测性表述（「今年可能」「预测」「大概率会考」）。
   产品定位是「基于历年真题的归纳」，一旦输出预测就是无依据的赌注，
   用户追问「依据是什么」无法自圆其说。命中数必须为 0。

这个模块不依赖 LLM，纯规则，可单测。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

# [1] 【1】 ［1］ 等各种括号形式
RE_CITATION = re.compile(r"[\[【［]\s*(\d{1,2})\s*[\]】］]")
# 真题编号，形如 2011#21 / 2011-21 / 2011 年第 21 题
RE_EXAM_REF = re.compile(r"(20\d{2})\s*(?:#|-|年(?:第)?)\s*(\d{1,2})")
# 教材章节引用，形如 5.2.1 / 第 5 章
RE_SECTION_REF = re.compile(r"第\s*[0-9一二三四五六七八九十]+\s*[章节]|\b\d+\.\d+(?:\.\d+)?\b")

# ★ 押题红线：出现任一表述即判定违规
PREDICTION_PATTERNS = (
    "今年可能",
    "今年会考",
    "明年可能",
    "预计会考",
    "预测",
    "押题",
    "大概率会考",
    "很可能会考",
    "可能会出",
    "将会考查",
    "未来会考",
    "必考",
)


@dataclass
class CitationReport:
    """一次生成的引用质量报告。"""

    cited_indices: list[int] = field(default_factory=list)
    valid_indices: list[int] = field(default_factory=list)
    invalid_indices: list[int] = field(default_factory=list)
    exam_refs: list[str] = field(default_factory=list)
    invalid_exam_refs: list[str] = field(default_factory=list)
    prediction_hits: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        """引用的编号里，真实存在的比例。"""
        if not self.cited_indices:
            return 0.0
        return len(self.valid_indices) / len(self.cited_indices)

    @property
    def recall(self) -> float:
        """检索上下文里被引用到的比例。低不代表错，但说明上下文利用率低。"""
        if not self.available_indices:
            return 0.0
        return len(set(self.valid_indices)) / len(self.available_indices)

    @property
    def has_prediction(self) -> bool:
        return bool(self.prediction_hits)

    available_indices: set[int] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "cited": self.cited_indices,
            "valid": self.valid_indices,
            "invalid": self.invalid_indices,
            "citation_precision": round(self.precision, 4),
            "citation_recall": round(self.recall, 4),
            "exam_refs": self.exam_refs,
            "invalid_exam_refs": self.invalid_exam_refs,
            "prediction_hits": self.prediction_hits,
            "has_prediction": self.has_prediction,
        }


def extract_citations(text: str) -> list[int]:
    """抽取 [编号] 引用，保持出现顺序、去重。"""
    seen: list[int] = []
    for m in RE_CITATION.finditer(text or ""):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def extract_exam_refs(text: str) -> list[str]:
    """抽取真题编号，规范化为 `YYYY#NN`。"""
    out: list[str] = []
    for m in RE_EXAM_REF.finditer(text or ""):
        ref = f"{m.group(1)}#{int(m.group(2)):02d}"
        if ref not in out:
            out.append(ref)
    return out


def detect_predictions(text: str) -> list[str]:
    """扫描预测性表述。返回命中的词，空列表表示通过。

    ★ 这是红线检查。命中即判定该回答不合格，不是「扣分」而是「不能输出」。

    但必须排除**否定语境**：模型经常会写「不作预测」「不构成对未来考情的
    任何预测」来表态合规，子串匹配会把这种句子误判为押题（实测 20 条考点
    结论里误伤 2 条，还会给正常回答追加拦截警告）。
    """
    t = text or ""
    hits: list[str] = []
    for p in PREDICTION_PATTERNS:
        for m in re.finditer(re.escape(p), t):
            prev = t[max(0, m.start() - 12) : m.start()]
            if any(neg in prev for neg in ("不", "无", "非", "未", "勿")):
                continue
            hits.append(p)
            break
    return hits


def check_citations(
    answer: str,
    blocks: Sequence,
    *,
    valid_exam_refs: Iterable[str] | None = None,
) -> CitationReport:
    """校验一次生成。

    Args:
        answer: 模型输出
        blocks: 检索上下文（有 .index 属性，即 ContextBlock）
        valid_exam_refs: 真题库里真实存在的编号集合，如 {"2011#21", ...}。
                         给了就校验真题引用，没给则跳过。
    """
    available = {b.index for b in blocks}
    cited = extract_citations(answer)
    valid = [i for i in cited if i in available]
    invalid = [i for i in cited if i not in available]

    refs = extract_exam_refs(answer)
    valid_set = set(valid_exam_refs) if valid_exam_refs is not None else None
    if valid_set is None:
        invalid_refs: list[str] = []
    else:
        invalid_refs = [r for r in refs if r not in valid_set]

    return CitationReport(
        cited_indices=cited,
        valid_indices=valid,
        invalid_indices=invalid,
        exam_refs=refs,
        invalid_exam_refs=invalid_refs,
        prediction_hits=detect_predictions(answer),
        available_indices=available,
    )


def aggregate_citations(reports: Sequence[CitationReport]) -> dict:
    """汇总一批生成的引用质量。"""
    n = len(reports) or 1
    with_cite = [r for r in reports if r.cited_indices]
    return {
        "n": len(reports),
        "citation_precision": round(sum(r.precision for r in reports) / n, 4),
        "citation_recall": round(sum(r.recall for r in reports) / n, 4),
        "answers_with_citation": len(with_cite),
        "answers_with_invalid_citation": sum(1 for r in reports if r.invalid_indices),
        "answers_with_fabricated_exam_ref": sum(1 for r in reports if r.invalid_exam_refs),
        # ★ 必须是 0
        "answers_with_prediction": sum(1 for r in reports if r.has_prediction),
    }


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------

_SELFTEST_CASES = [
    ("依据 [1] 和 [2]，组相联映射把地址划分为标记、组号、块内地址三部分。", [1, 2]),
    ("见 [1][1][3]。", [1, 3]),
    ("没有引用的回答。", []),
    ("【1】和［2］也要能识别。", [1, 2]),
]


def _run_selftest() -> None:
    for text, expect in _SELFTEST_CASES:
        got = extract_citations(text)
        assert got == expect, f"引用抽取错误：{text!r} -> {got} != {expect}"

    assert extract_exam_refs("如 2011#21、2015-33、2019年第 7 题") == [
        "2011#21",
        "2015#33",
        "2019#07",
    ], extract_exam_refs("如 2011#21、2015-33、2019年第 7 题")

    assert detect_predictions("该考点今年可能考。") == ["今年可能"]
    assert detect_predictions("该考点历年考查 5 次。") == []

    class _B:
        def __init__(self, i):
            self.index = i

    rep = check_citations(
        "根据 [1][2] 可知，如 2011#21 所示。今年可能会考。",
        [_B(1), _B(2), _B(3)],
        valid_exam_refs={"2011#21"},
    )
    assert rep.cited_indices == [1, 2]
    assert rep.invalid_indices == []
    assert abs(rep.precision - 1.0) < 1e-9
    assert rep.has_prediction

    rep2 = check_citations("见 [9]。", [_B(1)], valid_exam_refs=set())
    assert rep2.invalid_indices == [9]
    assert rep2.precision == 0.0

    print("citation 自检通过")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    _run_selftest()