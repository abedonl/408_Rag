"""指代/省略消解：纯规则，不调 LLM。

原则（Agent v2 plan）：解析不到就返回 None —— 不硬猜，
无法确定的追问交给澄清流程（Week 3），而不是猜错继续答。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from cs408rag.agent.memory import AgentSession

# 「这题」类指代：需要上一轮有 qid 才能消解
_QID_PRONOUNS = ("这题", "这道题", "该题", "此题", "上述题", "刚那题", "刚才那题")

# 「这个考点」类指代：需要上一轮有 topic
_TOPIC_PRONOUNS = ("这个考点", "该考点", "这个知识点", "该知识点", "这一章")

# 「那 2021 年呢」式年份追问：沿用上轮考点，改写成完整的考点频次年鉴问题
_YEAR_FOLLOWUP_RE = re.compile(r"^那\s*(20\d{2})\s*年呢[？?]?$|^(20\d{2})\s*年呢[？?]?$")

# 学习路径追问：「先学什么」「怎么学」沿用上轮考点
_LEARNING_HINTS = ("先学", "怎么学", "学习路径", "学习顺序", "前置")


@dataclass
class FollowupResolution:
    """消解结果。question 是改写后的有效问题；note 记进 steps 便于 trace 解释。"""

    question: str
    qid: str | None = None
    note: str = ""


def resolve_followup(
    question: str, session: AgentSession | None
) -> FollowupResolution | None:
    """尝试把追问消解成完整问题。返回 None 表示无需消解或消解不了。"""
    if session is None or not session.turns:
        return None
    q = question.strip()

    # 1. 年份追问：「那 2021 年呢？」→ 沿用上轮考点
    m = _YEAR_FOLLOWUP_RE.match(q)
    if m:
        if not session.last_topic:
            return None  # 没有可沿用的考点，不硬猜
        year = m.group(1) or m.group(2)
        return FollowupResolution(
            question=f"{session.last_topic}在{year}年考过几次？分别是哪些真题？",
            note=f"指代消解：沿用上轮考点「{session.last_topic}」+ 年份 {year}",
        )

    # 2. 学习路径追问：「那先学什么？」→ 沿用上轮考点
    if any(h in q for h in _LEARNING_HINTS) and session.last_topic:
        if session.last_topic not in q:
            return FollowupResolution(
                question=f"{session.last_topic}{q}",
                note=f"指代消解：沿用上轮考点「{session.last_topic}」",
            )
        return None

    # 3. 考点指代：「这个考点」→ 替换为上轮 topic
    if session.last_topic and any(p in q for p in _TOPIC_PRONOUNS):
        resolved = q
        for p in _TOPIC_PRONOUNS:
            resolved = resolved.replace(p, session.last_topic)
        if resolved != q:
            return FollowupResolution(
                question=resolved,
                note=f"指代消解：考点指代 →「{session.last_topic}」",
            )

    # 4. 题目指代：「这题选什么」→ 沿用上轮 qid（问题本身不改写）
    if session.last_qid and (
        any(p in q for p in _QID_PRONOUNS) or ("它" in q and len(q) <= 12)
    ):
        return FollowupResolution(
            question=q,
            qid=session.last_qid,
            note=f"指代消解：「这题」→ 上轮真题 {session.last_qid}",
        )

    return None
