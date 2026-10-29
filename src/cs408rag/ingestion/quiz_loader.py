"""真题库加载。

数据源：Become-ILLUSORY/408-quiz 的 questions.json（720 道历年真题，2009-2026）。

字段实测 100% 覆盖：
    year / num / subject / stem_text / stem_html / options / answer
    tags              226 个知识点标签 —— 「易考点」的客观依据
    explanation_text  官方解析 —— 带答案，故必须做年份隔离，见 config.HOLDOUT_YEARS
    svg_count         含图题数量（109/720 = 15%）
    options_in_svg    选项在图中（纯图题，3 道）
    url               原始出处
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from cs408rag.config import EVAL_DIR, PROCESSED_DIR, settings

# 科目规范化：原始数据里的写法不完全统一
SUBJECT_ALIASES = {
    "数据结构": "数据结构",
    "计算机组成原理": "组成原理",
    "组成原理": "组成原理",
    "操作系统": "操作系统",
    "计算机网络": "计算机网络",
}

CANONICAL_SUBJECTS = ["数据结构", "组成原理", "操作系统", "计算机网络"]


@dataclass
class Question:
    """一道真题。"""

    year: int
    num: int
    subject: str
    stem: str
    options: dict[str, str]  # {"A": "...", "B": "..."}
    answer: str
    tags: list[str] = field(default_factory=list)
    explanation: str = ""
    svg_count: int = 0
    options_in_svg: bool = False
    url: str = ""

    @property
    def qid(self) -> str:
        """稳定 ID，形如 2011-21。考点分析的溯源引用依赖它。"""
        return f"{self.year}-{self.num:02d}"

    @property
    def has_figure(self) -> bool:
        return self.svg_count > 0

    @property
    def is_pure_figure(self) -> bool:
        """连选项都在图里——纯文本 RAG 对此类题完全无解。"""
        return self.options_in_svg

    def to_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "year": self.year,
            "num": self.num,
            "subject": self.subject,
            "stem": self.stem,
            "options": self.options,
            "answer": self.answer,
            "tags": self.tags,
            "explanation": self.explanation,
            "svg_count": self.svg_count,
            "options_in_svg": self.options_in_svg,
            "url": self.url,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Question:
        return cls(
            year=int(d["year"]),
            num=int(d["num"]),
            subject=d["subject"],
            stem=d.get("stem") or "",
            options=d.get("options") or {},
            answer=d.get("answer") or "",
            tags=d.get("tags") or [],
            explanation=d.get("explanation") or "",
            svg_count=int(d.get("svg_count") or 0),
            options_in_svg=bool(d.get("options_in_svg")),
            url=d.get("url") or "",
        )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _parse_options(raw: Any) -> dict[str, str]:
    """options 是 [{"label":"A","html":"..","text":".."}, ...]，只取 text。"""
    out: dict[str, str] = {}
    if isinstance(raw, list):
        for o in raw:
            if isinstance(o, dict) and o.get("label"):
                out[str(o["label"]).strip()] = (o.get("text") or o.get("html") or "").strip()
    elif isinstance(raw, dict):
        out = {str(k).strip(): str(v).strip() for k, v in raw.items()}
    return out


def _parse_tags(raw: Any) -> list[str]:
    """tags 可能是 'a,b' / 'a;b' / 'a、b' 或已是 list。"""
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()]
    if not isinstance(raw, str):
        return []
    parts = re.split(r"[,;、，；]+", raw)
    return [p.strip() for p in parts if p.strip()]


def _canon_subject(raw: str) -> str:
    s = (raw or "").strip()
    if s in SUBJECT_ALIASES:
        return SUBJECT_ALIASES[s]
    for alias, canon in SUBJECT_ALIASES.items():
        if alias in s:
            return canon
    return s or "未知"


def load_questions(path: Path | None = None) -> list[Question]:
    """读取 questions.json。"""
    if path is None:
        path = _default_quiz_path()
    if not path.exists():
        raise FileNotFoundError(
            f"找不到真题库 {path}。先跑：python -m cs408rag.ingestion.download --source quiz"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        # 兼容 {"questions": [...]} 之类的包装
        raw = raw.get("questions") or raw.get("data") or next(iter(raw.values()))

    questions: list[Question] = []
    for item in raw:
        q = Question(
            year=int(item["year"]),
            num=int(item["num"]),
            subject=_canon_subject(item.get("subject", "")),
            stem=(item.get("stem_text") or "").strip(),
            options=_parse_options(item.get("options")),
            answer=(item.get("answer") or "").strip(),
            tags=_parse_tags(item.get("tags")),
            explanation=(item.get("explanation_text") or "").strip(),
            svg_count=int(item.get("svg_count") or 0),
            options_in_svg=bool(item.get("options_in_svg")),
            url=item.get("url") or "",
        )
        questions.append(q)

    questions.sort(key=lambda q: (q.year, q.num))
    logger.info(f"载入 {len(questions)} 道真题，年份 {questions[0].year}-{questions[-1].year}")
    return questions


def _default_quiz_path() -> Path:
    from cs408rag.config import RAW_DIR

    return RAW_DIR / "quiz" / "questions.json"


# ---------------------------------------------------------------------------
# 防泄漏隔离（红线）
# ---------------------------------------------------------------------------


@dataclass
class QuestionSplit:
    """按年份切分的结果。

    holdout 集的解析**禁止入索引**，否则 RAG 检索到解析即可直接抄答案，
    准确率虚高且属作弊。见 README「评测口径」。
    """

    dev: list[Question]
    holdout: list[Question]

    def summary(self) -> str:
        return (
            f"开发集 {len(self.dev)} 题（{self._span(self.dev)}）| "
            f"held-out {len(self.holdout)} 题（{self._span(self.holdout)}）"
        )

    @staticmethod
    def _span(qs: list[Question]) -> str:
        if not qs:
            return "空"
        ys = sorted({q.year for q in qs})
        return f"{ys[0]}-{ys[-1]}"


def split_by_year(
    questions: list[Question], holdout_years: set[int] | None = None
) -> QuestionSplit:
    """按年份切分开发集与 held-out 测试集。"""
    holdout_years = holdout_years if holdout_years is not None else settings.holdout_year_set
    dev = [q for q in questions if q.year not in holdout_years]
    holdout = [q for q in questions if q.year in holdout_years]
    logger.info(QuestionSplit(dev, holdout).summary())
    return QuestionSplit(dev=dev, holdout=holdout)


# ---------------------------------------------------------------------------
# 统计（考点分析的原料）
# ---------------------------------------------------------------------------


def tag_frequency(questions: list[Question]) -> dict[str, int]:
    """考点频次——「易考点」的客观依据，不是 LLM 拍脑袋。"""
    freq: dict[str, int] = {}
    for q in questions:
        for t in q.tags:
            freq[t] = freq.get(t, 0) + 1
    return dict(sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])))


def tag_cooccurrence(questions: list[Question]) -> dict[tuple[str, str], int]:
    """考点共现——同一道题涉及多个 tag 时两两连边，用于建 CO_OCCURS 图边。"""
    pairs: dict[tuple[str, str], int] = {}
    for q in questions:
        ts = sorted(set(q.tags))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                key = (ts[i], ts[j])
                pairs[key] = pairs.get(key, 0) + 1
    return dict(sorted(pairs.items(), key=lambda kv: -kv[1]))


def dataset_profile(questions: list[Question]) -> dict[str, Any]:
    """数据集画像，写进 README 用。"""
    from collections import Counter

    return {
        "total": len(questions),
        "years": dict(sorted(Counter(q.year for q in questions).items())),
        "subjects": dict(Counter(q.subject for q in questions).most_common()),
        "with_figure": sum(1 for q in questions if q.has_figure),
        "pure_figure": sum(1 for q in questions if q.is_pure_figure),
        "unique_tags": len(tag_frequency(questions)),
        "with_explanation": sum(1 for q in questions if q.explanation),
        "with_tags": sum(1 for q in questions if q.tags),
    }


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_split(split: QuestionSplit | None = None, questions: list[Question] | None = None) -> Path:
    """把切分结果落盘到 data/eval/，供评测脚本复用。"""
    if split is None:
        split = split_by_year(questions if questions is not None else load_questions())

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, qs in (("dev", split.dev), ("holdout", split.holdout)):
        p = EVAL_DIR / f"{name}_questions.json"
        p.write_text(
            json.dumps([q.to_dict() for q in qs], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"写出 {p}（{len(qs)} 题）")

    meta = {
        "holdout_years": sorted(settings.holdout_year_set),
        "dev": dataset_profile(split.dev),
        "holdout": dataset_profile(split.holdout),
    }
    mp = EVAL_DIR / "split_meta.json"
    mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"写出 {mp}")
    return mp


if __name__ == "__main__":
    from cs408rag.config import ensure_dirs

    ensure_dirs()
    qs = load_questions()
    prof = dataset_profile(qs)
    print(json.dumps(prof, ensure_ascii=False, indent=2))
    print("\nTop 20 考点频次：")
    for t, n in list(tag_frequency(qs).items())[:20]:
        print(f"  {n:>3}  {t}")
    export_split(questions=qs)

    # 顺带产出共现边，二期建图用
    co = tag_cooccurrence(qs)
    cp = PROCESSED_DIR / "tag_cooccurrence.json"
    cp.parent.mkdir(parents=True, exist_ok=True)
    cp.write_text(
        json.dumps([{"a": a, "b": b, "count": c} for (a, b), c in co.items()], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"写出共现边 {len(co)} 条 -> {cp}")