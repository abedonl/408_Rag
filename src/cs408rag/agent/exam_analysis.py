"""考点分析：基于 720 道真题的**可溯源**统计。

产品定位是「基于历年真题的归纳」，不是押题。因此这里的每个数字都来自
`questions.json` 的真实标注，不经过 LLM，也不做任何推测：

    考查频次        该 tag 在 720 道真题里出现的实际次数
    年份 / 科目分布  逐年的真实计数
    题型分布        按题干特征分到 概念 / 计算 / 图表 / 跨科目
    真题原题        可点击回看的 qid（如 2011#21）

**防泄漏**：Agent 评测 held-out 题时，本模块可用 `year_allowed` 把 2023–2026
的真题挡在外面。否则 Agent 可能直接检索到同题答案解析，评测失效。
生产环境默认全量可用（这些是公开真题，考点分析本就该覆盖全部年份）。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cs408rag.ingestion.quiz_loader import Question, load_questions

# ---------------------------------------------------------------------------
# 题型判定
# ---------------------------------------------------------------------------

# 计算题的「问量」触发词。必须与数字同时出现，否则「两者之间的传输时间」
# 这类纯概念题会被误判。另外 `计算(?!机)` 是必须的：直接用 `"计算" in stem`
# 会让「计算机与打印机之间…」被判成计算题（实测 2009-01 就是这么错的）。
RE_CALC_ASK = re.compile(
    r"多少|至少|最多|最少|最小|最大|几次|几个|几位|几片|几块|是多少|为多少|"
    r"计算(?!机)|求(?:[a-zA-Z一-鿿])"
)
RE_DIGIT = re.compile(r"\d")
CALC_QUANTITY = (
    "带宽", "命中率", "传输速率", "容量", "周期", "吞吐", "窗口", "地址位数",
    "字长", "片数", "码长", "校验", "开销", "利用率", "效率", "比例", "百分比",
    "主频", "CPI", "MIPS", "速率", "延迟", "时间", "位数", "个数", "长度",
)
RE_UNIT = re.compile(
    r"\d+(?:\.\d+)?\s*(?:ns|us|μs|ms|s|bit|b|B|KB|MB|GB|TB|kHz|MHz|GHz|"
    r"bps|Kbps|Mbps|Gbps|个|位|字节|片|块|次|条|种|台|%)",
    re.I,
)


def classify_type(q: Question, cross_subject: bool = False) -> str:
    """把题型分到四类。规则可解释，便于人工抽检核对。

    优先级：图表 > 跨科目 > 计算 > 概念。
    图表题优先，因为它是「单轮检索按定义答不了」的最强证据。
    """
    if q.has_figure:
        return "figure"
    if cross_subject:
        return "cross_subject"

    stem = q.stem or ""
    if not RE_CALC_ASK.search(stem):
        return "concept"
    has_number = bool(RE_DIGIT.search(stem)) or bool(RE_UNIT.search(stem))
    has_quantity = any(w in stem for w in CALC_QUANTITY)
    # 数字 + 问量最稳；「求平均周转时间」这类数字在选项里的题，
    # 用「问量 + 量名词」兜底。
    if has_number or has_quantity:
        return "calc"
    return "concept"


# ---------------------------------------------------------------------------
# 考点索引
# ---------------------------------------------------------------------------


@dataclass
class TagStats:
    tag: str
    matched_tags: list[str] = field(default_factory=list)
    total: int = 0
    by_year: dict[int, int] = field(default_factory=dict)
    by_subject: dict[str, int] = field(default_factory=dict)
    by_type: dict[str, int] = field(default_factory=dict)
    by_tag: dict[str, int] = field(default_factory=dict)
    qids: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tag": self.tag,
            "matched_tags": self.matched_tags,
            "total": self.total,
            "by_year": dict(sorted(self.by_year.items())),
            "by_subject": self.by_subject,
            "by_type": self.by_type,
            "by_tag": self.by_tag,
            "qids": self.qids,
            "examples": self.examples,
            # 样本量提醒：单知识点每年 0-1 次，时间序列结论噪声大。
            # 这不是免责声明，是评测口径的一部分（见计划「样本局限须写明」）。
            "note": (
                "720 题覆盖 18 年，单个考点每年多为 0-1 次，"
                "逐年趋势只能作为弱信号，不能当作强结论。"
            ),
        }


class ExamAnalyzer:
    """真题考点分析器。

    Args:
        questions: 题目列表，默认加载全部 720 道
        year_allowed: 允许使用的年份集合。评测 held-out 时传入排除年份后的集合。
    """

    def __init__(
        self,
        questions: Sequence[Question] | None = None,
        *,
        year_allowed: Iterable[int] | None = None,
    ) -> None:
        qs = list(questions) if questions is not None else load_questions()
        self.year_allowed = set(year_allowed) if year_allowed is not None else None
        if self.year_allowed is not None:
            qs = [q for q in qs if q.year in self.year_allowed]
        self.questions = qs
        self.by_qid: dict[str, Question] = {q.qid: q for q in qs}

        # tag -> 题目；tag -> 科目集合（用于判跨科目）
        self.by_tag: dict[str, list[Question]] = defaultdict(list)
        for q in qs:
            for t in q.tags:
                self.by_tag[t].append(q)

        self.tag_freq: dict[str, int] = dict(
            sorted(
                ((t, len(v)) for t, v in self.by_tag.items()),
                key=lambda kv: (-kv[1], kv[0]),
            )
        )

        # ★ 跨科目判定不能只看「tag 的映射知识点分布在几个科目」。
        # 实测：「排序算法」映射到「算法」「排序算法的分析与应用」等节点，
        # 而「算法」在操作系统/组成原理里也有同名节点，于是 23 道纯数据结构题
        # 被误判成 22 道跨科目。改为看**该 tag 的真题实际分布在几个科目**，
        # 以及一道题内多个 tag 的主科目是否不同 —— 不依赖图谱映射质量。
        tag_main_subject = {
            t: Counter(q.subject for q in qs_).most_common(1)[0][0]
            for t, qs_ in self.by_tag.items()
        }
        self.cross_qids = {
            q.qid
            for q in qs
            if len({tag_main_subject[t] for t in q.tags if t in tag_main_subject}) > 1
        }
        self.type_of: dict[str, str] = {
            q.qid: classify_type(q, q.qid in self.cross_qids) for q in qs
        }
        logger.info(
            f"考点分析器就绪：{len(qs)} 题，{len(self.by_tag)} 个考点，"
            f"{len(self.cross_qids)} 道跨科目题"
        )

    # -- 查询 -------------------------------------------------------------

    def question(self, qid: str) -> Question | None:
        """按 qid 取题。兼容 `2011#21` / `2011-21` / `2011-21` 三种写法。"""
        qid = normalize_qid(qid)
        return self.by_qid.get(qid)

    def resolve_tags(self, text: str) -> list[str]:
        """把用户说法对齐到真实 tag。

        规则（顺序即优先级）：
            1. 大小写不敏感的精确匹配 —— 「排序算法」只取这一个 tag
            2. 用户词是 tag 的子串 —— 「死锁」汇总「死锁概念 / 死锁预防 /
               死锁的处理方法 / 死锁产生的必要条件」；「Cache」汇总三个 cache tag
            3. tag 是用户词的子串 —— 用户说得更长时兜底

        为什么不只返回一个 tag：实测「死锁」若只取最接近的单个 tag，
        答案会变成「死锁概念 考查 1 次」—— 统计严重低估且误导。
        主题词本来就该汇总其下所有标签，但必须把命中的 tag 列表一并返回，
        保证结论可溯源。
        """
        t = (text or "").strip().lower()
        if not t:
            return []
        exact = [tag for tag in self.by_tag if tag.lower() == t]
        if exact:
            return exact[:1]

        hits = [tag for tag in self.by_tag if t in tag.lower()]
        if not hits:
            hits = [tag for tag in self.by_tag if tag.lower() in t]
        return sorted(hits, key=lambda x: (-self.tag_freq.get(x, 0), x))[:8]

    def _collect(self, tag: str) -> tuple[list[str], list[Question]]:
        """返回 (命中的 tag 列表, 去重后的题目，按年份序号排序)。"""
        tags = self.resolve_tags(tag)
        seen: dict[str, Question] = {}
        for t in tags:
            for q in self.by_tag[t]:
                seen.setdefault(q.qid, q)
        return tags, sorted(seen.values(), key=lambda q: (q.year, q.num))

    def stats(self, tag: str, *, example_limit: int = 8) -> TagStats:
        """考点的完整统计。tag 不存在时返回 total=0 的空结果，不抛异常。"""
        tags, qs = self._collect(tag)
        out = TagStats(tag=tag.strip(), matched_tags=tags)
        if not qs:
            return out

        out.total = len(qs)
        out.by_year = dict(Counter(q.year for q in qs))
        out.by_subject = dict(Counter(q.subject for q in qs))
        out.by_type = dict(Counter(self.type_of.get(q.qid, "concept") for q in qs))
        out.by_tag = dict(Counter(t for q in qs for t in q.tags if t in tags))
        out.qids = [q.qid for q in qs]
        out.examples = [
            {
                "qid": q.qid,
                "year": q.year,
                "subject": q.subject,
                "type": self.type_of.get(q.qid, "concept"),
                "question": _clip(q.stem, 160),
                "options": q.options,
                "answer": q.answer,
                "tags": q.tags,
            }
            for q in qs[:example_limit]
        ]
        return out

    def questions_for_tag(self, tag: str, *, limit: int = 10) -> list[dict[str, Any]]:
        """该考点的历年真题原题（含答案与解析）。"""
        _, qs = self._collect(tag)
        return [
            {
                "qid": q.qid,
                "year": q.year,
                "num": q.num,
                "subject": q.subject,
                "type": self.type_of.get(q.qid, "concept"),
                "question": q.stem,
                "options": q.options,
                "answer": q.answer,
                "tags": q.tags,
                "explanation": _clip(q.explanation, 500),
                "url": q.url,
            }
            for q in qs[:limit]
        ]

    def related_tags(self, tag: str, *, top_n: int = 8) -> list[dict[str, Any]]:
        """与指定考点同题共现的其他考点，附带共现次数与示例 qid。

        共现来自「同一道真题的多个 tag 两两连边」，是真实数据，
        不是 LLM 猜的「这两个知识点有关系」。
        """
        tags, qs = self._collect(tag)
        if not qs:
            return []
        mine = set(tags)
        pairs: Counter[str] = Counter()
        evidence: dict[str, list[str]] = defaultdict(list)
        for q in qs:
            for other in set(q.tags):
                if other in mine:
                    continue
                pairs[other] += 1
                if len(evidence[other]) < 3:
                    evidence[other].append(q.qid)
        return [
            {
                "tag": t,
                "cooccurrence": n,
                "evidence": evidence[t],
                "frequency": self.tag_freq.get(t, 0),
            }
            for t, n in pairs.most_common(top_n)
        ]

    def all_tags(self, *, subject: str | None = None, limit: int = 226) -> list[dict[str, Any]]:
        """全部考点及其频次，可按科目过滤。"""
        out: list[dict[str, Any]] = []
        for t, n in self.tag_freq.items():
            qs = self.by_tag[t]
            if subject and not any(q.subject == subject for q in qs):
                continue
            out.append(
                {
                    "tag": t,
                    "frequency": n,
                    "subjects": sorted({q.subject for q in qs}),
                }
            )
            if len(out) >= limit:
                break
        return out

    def tag_distribution(self) -> dict[str, Any]:
        """全局概览，给 API/UI 用。"""
        types = Counter(self.type_of.values())
        return {
            "n_questions": len(self.questions),
            "n_tags": len(self.by_tag),
            "by_type": dict(types),
            "by_subject": dict(Counter(q.subject for q in self.questions)),
            "by_year": dict(sorted(Counter(q.year for q in self.questions).items())),
            "top_tags": [
                {"tag": t, "frequency": n} for t, n in list(self.tag_freq.items())[:10]
            ],
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def normalize_qid(qid: str) -> str:
    """统一成 `2011-21`。"""
    m = re.match(r"(20\d{2})\s*(?:#|-|年(?:第)?)\s*(\d{1,2})", str(qid))
    if not m:
        return str(qid)
    return f"{m.group(1)}-{int(m.group(2)):02d}"


def qid_to_ref(qid: str) -> str:
    """评测/展示用的真题编号，`2011-21` -> `2011#21`。"""
    m = re.match(r"(20\d{2})-(\d{2})", qid)
    if not m:
        return qid
    return f"{m.group(1)}#{m.group(2)}"


def _clip(text: str, n: int) -> str:
    t = (text or "").strip()
    return t if len(t) <= n else t[: n - 1] + "…"


_analyzer: ExamAnalyzer | None = None


def get_analyzer() -> ExamAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ExamAnalyzer()
    return _analyzer


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    a = ExamAnalyzer()
    print(json.dumps(a.tag_distribution(), ensure_ascii=False, indent=2))
    for tag in ("排序算法", "中断IO", "补码", "死锁"):
        s = a.stats(tag)
        print(f"\n【{s.tag}】{s.total} 次  年份={s.by_year}  题型={s.by_type}")
        print("  关联：", [(r["tag"], r["cooccurrence"]) for r in a.related_tags(tag, top_n=5)])