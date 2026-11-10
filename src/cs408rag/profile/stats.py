"""画像统计：从提问/规划事件聚合用户画像，纯规则，不调 LLM。

提取口径：整标签命中 > 最长公共片段 ≥3 字；2 字片段只在没有任何更强命中时
兜底（「死锁怎么预防」→「死锁概念」）。2 字片段常态参与会命中「映射」「地址」
「划分」这类泛化词，把无关考点灌进画像和 planner 加权（实测踩过这个坑）。
"""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta
from typing import Any

from cs408rag.agent.exam_analysis import ExamAnalyzer

# plan 事件时效：超过 30 天的规划不再影响画像（plan 块与 boost 的规划重点）。
# 备考目标会随时间变化，旧规划永久加权会把过期考点灌进 planner。
PLAN_TTL_DAYS = 30


def _parse_ts(ts: Any) -> datetime | None:
    """解析事件时间戳；缺 ts 或格式非法返回 None（调用方按过期处理）。"""
    try:
        dt = datetime.fromisoformat(str(ts))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt

MAX_TAGS_PER_QUESTION = 6
MAX_BOOST_TAGS = 8  # 喂给 planner 的上限：boost 考点是必选项，太多会挤掉高频考点


def _longest_fragment(tag: str, text: str) -> int:
    """tag 与 text 的最长公共子串长度（子串须出自 tag）。"""
    best = 0
    for i in range(len(tag)):
        for j in range(i + 2, len(tag) + 1):
            if j - i <= best:
                continue
            if tag[i:j] in text:
                best = j - i
    return best


def _main_subject(analyzer: ExamAnalyzer, tag: str) -> str:
    qs = analyzer.by_tag.get(tag) or []
    if not qs:
        return ""
    return Counter(q.subject for q in qs).most_common(1)[0][0]


def extract_tags(
    text: str, analyzer: ExamAnalyzer, *, limit: int = MAX_TAGS_PER_QUESTION
) -> list[dict[str, str]]:
    """从提问文本提取考点标签，返回 [{"tag", "subject"}]，按考查频次降序。"""
    text = (text or "").strip()
    if not text:
        return []
    low = text.lower()
    full: list[str] = []
    frag3: list[str] = []
    frag2: list[str] = []
    for tag in analyzer.by_tag:
        if len(tag) < 2:
            continue
        if tag.lower() in low:
            full.append(tag)
            continue
        n = _longest_fragment(tag, text)
        if n >= 3:
            frag3.append(tag)
        elif n == 2:
            frag2.append(tag)
    def _by_freq(ts: list[str]) -> list[str]:
        return sorted(ts, key=lambda t: -analyzer.tag_freq.get(t, 0))

    if full or frag3:
        ordered = _by_freq(full) + _by_freq(frag3)
    else:
        # 纯 2 字片段兜底说明用户用的是俗称（「死锁」），泛化命中（「地址」「划分」）
        # 也在这层——只保留频次最高的 2 个，宁可少提也不灌噪声进画像
        ordered = _by_freq(frag2)[:2]
    return [
        {"tag": t, "subject": _main_subject(analyzer, t)} for t in ordered[:limit]
    ]


def build_profile(events: list[dict[str, Any]]) -> dict[str, Any]:
    """把事件流聚合成画像视图。

    返回：
        total_questions / last_active / recent_questions
        top_tags        高频提问考点 [{"tag","subject","count"}]
        subject_counts  提问的科目分布（每题计一次：显式科目优先，否则取首个命中考点的主科目）
        plan            最近一次规划事件（weak_subjects / priority_tags / concerns），可空；
                        超过 30 天的规划视为过期，不再参与（PLAN_TTL_DAYS）
        quiz            作答统计（total / correct / wrong_tags / wrong_subjects）
        boost_tags      喂给 planner 的加权考点：答错考点 > 高频提问 > 规划重点
    """
    q_events = [e for e in events if e["kind"] == "question"]
    plans = [
        e
        for e in events
        if e["kind"] == "plan"
        and (dt := _parse_ts(e.get("ts"))) is not None
        and datetime.now(UTC) - dt <= timedelta(days=PLAN_TTL_DAYS)
    ]

    tag_counts: Counter[str] = Counter()
    tag_subject: dict[str, str] = {}
    subject_counts: Counter[str] = Counter()
    for e in q_events:
        p = e["payload"]
        tags = p.get("tags") or []
        for t in tags:
            tag_counts[t["tag"]] += 1
            tag_subject.setdefault(t["tag"], t.get("subject") or "")
        if p.get("subject"):
            subject_counts[p["subject"]] += 1
        elif tags and tags[0].get("subject"):
            subject_counts[tags[0]["subject"]] += 1

    latest_plan = plans[-1]["payload"] if plans else None

    # quiz 作答事件：答错考点是真正的薄弱信号（提问高频只代表关注，不代表不会）
    quiz_events = [e for e in events if e["kind"] == "quiz"]
    wrong_counts: Counter[str] = Counter()
    wrong_subj: dict[str, str] = {}
    wrong_subject_counts: Counter[str] = Counter()
    quiz_correct = 0
    for e in quiz_events:
        p = e["payload"]
        if p.get("correct"):
            quiz_correct += 1
            continue
        if p.get("subject"):
            wrong_subject_counts[p["subject"]] += 1
        for t in p.get("tags") or []:
            wrong_counts[t] += 1
            wrong_subj.setdefault(t, p.get("subject") or "")

    boost: list[dict[str, str]] = []
    # 顺序：答错考点 > 高频提问 > 规划重点，共 8 槽位
    for tag, _ in wrong_counts.most_common(3):
        boost.append({"tag": tag, "subject": wrong_subj.get(tag, "")})
    for tag, _ in tag_counts.most_common(MAX_BOOST_TAGS):
        if len(boost) >= MAX_BOOST_TAGS - 2:
            break
        item = {"tag": tag, "subject": tag_subject.get(tag, "")}
        if item not in boost:
            boost.append(item)
    for t in (latest_plan or {}).get("priority_tags") or []:
        if len(boost) >= MAX_BOOST_TAGS:
            break
        item = {"tag": t.get("tag", ""), "subject": t.get("subject", "")}
        if item["tag"] and item not in boost:
            boost.append(item)

    return {
        "total_questions": len(q_events),
        "last_active": q_events[-1]["ts"] if q_events else None,
        "recent_questions": [
            (e["payload"].get("text") or "")[:60] for e in q_events[-5:]
        ][::-1],
        "top_tags": [
            {"tag": t, "subject": tag_subject.get(t, ""), "count": n}
            for t, n in tag_counts.most_common(10)
        ],
        "subject_counts": dict(subject_counts.most_common()),
        "plan": latest_plan,
        "quiz": {
            "total": len(quiz_events),
            "correct": quiz_correct,
            "wrong_tags": [
                {"tag": t, "subject": wrong_subj.get(t, ""), "count": n}
                for t, n in wrong_counts.most_common(10)
            ],
            "wrong_subjects": dict(wrong_subject_counts.most_common()),
        },
        "boost_tags": boost,
    }
