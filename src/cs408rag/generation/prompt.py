"""Prompt 构造。

三条硬约束写进模板，不是软性建议：

1. **只许引用给定编号** —— 检索上下文里的 chunk 编号是 [1][2][3]，
   回答只能引用这些编号，不允许凭空引「王道第 X 章」。

2. **禁止押题** —— 这是产品可信度底线。
   允许：「该考点在真题中出现 5 次，如 2011#21、2015#33」
   禁止：「今年可能考」「预测 2027 年会出」
   前者是可验证的归纳，后者是没有数据支撑的赌注。
   一旦模型输出预测性内容，用户追问依据就无法自圆其说。
   提示词里明确禁止，citation.py 里再做一遍规则扫描兜底。

3. **无依据时明确拒答** —— 宁可说「检索到的材料不足以回答」，
   也不要编。408 有标准答案，编造的解析比不回答更糟。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from cs408rag.retrieval.vector_store import SearchHit

SYSTEM_PROMPT = """你是计算机考研 408 的助教，覆盖数据结构、计算机组成原理、操作系统、计算机网络四科。

回答要求：
1. 严格依据【参考资料】作答。资料里没有的内容，明确说「提供的材料不足以回答」，不要凭记忆补充。
2. 引用资料时使用给定的编号，如 [1]、[2]。不要引用不存在的编号。
3. 指明答案属于哪个知识点、在教材的哪一节（资料里已给出章节路径）。
4. 涉及计算时，分步写出计算过程，不要直接给结果。
5. 回答简洁准确，用中文。

严禁事项：
- 不得预测考试内容。可以说「该考点历年考查 N 次」，不得说「今年可能考」「预计会出题」。
- 不得编造法条式的条文编号或教材页码。"""


@dataclass
class ContextBlock:
    """给 LLM 看的一条参考资料。"""

    index: int
    chunk_id: str
    text: str
    citation: str
    subject: str
    source: str
    # 该资料对应的教材插图（已解析成可访问 URL）。
    # 注意：图片内容是**给前端展示**的，不进 prompt —— LLM 看不到图，
    # 只能看到文字，所以上下文里只用图注做提示，避免它臆测图像内容。
    images: list[dict] = field(default_factory=list)

    def render(self) -> str:
        head = f"[{self.index}] {self.citation}"
        if self.source == "explanation":
            head = f"[{self.index}] 真题解析"
        body = self.text
        if self.images:
            caps = [im.get("caption") for im in self.images if im.get("caption")]
            if caps:
                body += "\n（本段配有插图：" + "；".join(caps[:3]) + "）"
            else:
                body += f"\n（本段配有 {len(self.images)} 张插图）"
        return f"{head}\n{body}"


def _resolve(images: Any) -> list[dict]:
    """把 chunk payload 里的图片补全成可访问 URL。出错时不阻断上下文构造。"""
    if not images:
        return []
    try:
        from cs408rag.ingestion.image_index import resolve_images

        return resolve_images(images)
    except Exception:
        return []


def build_context(hits: Sequence[SearchHit], *, max_chars: int = 6000) -> list[ContextBlock]:
    """把检索结果转成带编号的资料块。

    超出预算时按顺序截断 —— 检索结果已按相关性排序，
    后面的本来就该先丢。
    """
    blocks: list[ContextBlock] = []
    used = 0
    for i, h in enumerate(hits, 1):
        text = h.text.strip()
        if used + len(text) > max_chars and blocks:
            break
        blocks.append(
            ContextBlock(
                index=i,
                chunk_id=h.chunk_id,
                text=text,
                citation=h.citation,
                subject=h.subject,
                source=h.source,
                images=_resolve(h.payload.get("images")),
            )
        )
        used += len(text)
    return blocks


def render_context(blocks: Sequence[ContextBlock]) -> str:
    return "\n\n".join(b.render() for b in blocks)


# ---------------------------------------------------------------------------
# 各类任务
# ---------------------------------------------------------------------------


def qa_messages(
    question: str,
    blocks: Sequence[ContextBlock],
) -> list[dict[str, str]]:
    """开放式问答。"""
    user = (
        f"【参考资料】\n{render_context(blocks)}\n\n"
        f"【问题】\n{question}\n\n"
        "请依据参考资料作答，并用 [编号] 标注依据来源。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def quiz_messages(
    question: str,
    options: dict[str, str],
    blocks: Sequence[ContextBlock],
) -> list[dict[str, str]]:
    """选择题专用：这是与评测口径对齐的主要路径。

    要求模型先给选项，再给依据；评测只取选项。
    """
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    user = (
        f"【参考资料】\n{render_context(blocks)}\n\n"
        f"【题目】\n{question}\n\n"
        f"【选项】\n{opts}\n\n"
        "请作答。**第一行**必须严格写成 `答案：X`，X 为选项字母；"
        "之后简要说明依据，并用 [编号] 标注来源。"
        "先把答案写出来，再解释；即使后面被打断，也能解析到选项。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def exam_analysis_messages(
    topic: str,
    stats: dict,
    example_questions: Sequence[dict],
) -> list[dict[str, str]]:
    """考点分析（二期的备考能力）。

    输入必须是**已统计好的真实数据**，模型只做归纳表述，不做任何推测。
    这是「可溯源、禁押题」原则在 prompt 层的落实。
    """
    lines = [
        f"【考点】{topic}",
        f"【历年考查统计】共出现 {stats.get('total', 0)} 次",
    ]
    by_year = stats.get("by_year") or {}
    if by_year:
        lines.append("年份分布：" + "、".join(f"{y}年 {n} 次" for y, n in sorted(by_year.items())))
    by_subject = stats.get("by_subject") or {}
    if by_subject:
        lines.append("科目分布：" + "、".join(f"{s} {n} 次" for s, n in by_subject.items()))

    if example_questions:
        lines.append("\n【历年真题原题】")
        for q in example_questions[:8]:
            lines.append(f"- [{q['qid']}] {q['question'][:120]}（答案 {q.get('answer', '?')}）")

    user = (
        "\n".join(lines)
        + "\n\n请基于以上**真实统计数据与原题**，归纳该考点的考查方式。"
        "\n要求："
        "\n1. 每条结论都要能追溯到具体的真题编号，写成「如 2011#21、2015#33」的形式。"
        "\n2. 只能归纳**已经考过的**形式，不得预测未来会考什么。"
        "\n3. 若统计数据样本量不足以支撑某种说法，明说「样本有限，仅供参考」。"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def rewrite_messages(query: str, n: int = 3) -> list[dict[str, str]]:
    """查询改写（S3 阶段的 LLM 版本，可选）。

    目标是生成**检索友好**的改写，不是同义句改写：
    408 题干往往带具体场景（「某程序访存 1000 次…」），
    改写要剥离场景、突出考点术语，才更容易命中教材的概念小节。
    """
    user = (
        f"考生的问题：{query}\n\n"
        f"请生成 {n} 个用于**检索教材**的查询，要求：\n"
        "1. 剥离具体数值与场景描述，突出考点术语（如「组相联映射 地址划分」）\n"
        "2. 若题目涉及计算，额外给出对应的知识点名称\n"
        "3. 每行一个，不要编号，不要解释\n"
        "4. 保留原始问题作为第一行"
    )
    return [
        {"role": "system", "content": "你是检索查询改写助手。"},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Agent 路径
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + """
6. 【工具结论】是程序计算/解析出的确定性结果，包括计算器、图形解析、真题统计。
   资料与工具结论冲突时，以工具结论为准；不要重新心算去覆盖它。
7. 图形解析可能明确说明「节点值不可读」。此时只能基于结构（树高、连接关系）推理，
   不得假设自己看到了节点上的具体数值。若题目必须知道节点值才能确定唯一答案，
   直接说明这一点并基于资料给出最合理的判断，**不要反复推演未知信息**。"""


def render_evidence(evidence: dict) -> str:
    """把 Agent 工具输出渲染成给模型看的一段文字。"""
    parts: list[str] = []

    if fig := evidence.get("figure"):
        if fig.get("description"):
            parts.append(f"【图形解析（{fig.get('tier')}）】\n{fig['description']}")
        if fig.get("available") is False and fig.get("note"):
            parts.append(f"【图形解析不可用】{fig['note']}")

    if calc := evidence.get("calculate"):
        if calc.get("ok"):
            parts.append(
                f"【计算结果（程序求值，可信）】\n{calc['expr']} = {calc['display']}"
            )
        elif calc.get("error"):
            parts.append(f"【计算失败】{calc['error']}")

    if exam := evidence.get("exam_stats"):
        if exam.get("total"):
            lines = [
                "【考点统计（来自 720 道真题的真实标注）】",
                f"考点：{exam['tag']}（匹配标签：{'、'.join(exam.get('matched_tags') or [])}）",
                f"历年共考查 {exam['total']} 次",
            ]
            by_year = exam.get("by_year") or {}
            if by_year:
                lines.append("年份分布：" + "、".join(f"{y}年{n}次" for y, n in by_year.items()))
            by_type = exam.get("by_type") or {}
            if by_type:
                lines.append("题型分布：" + "、".join(f"{k}{v}次" for k, v in by_type.items()))
            if exam.get("qids"):
                lines.append("真题编号：" + "、".join(exam["qids"][:30]))
            parts.append("\n".join(lines))
        else:
            parts.append(f"【考点统计】没有匹配到「{exam.get('tag')}」的真题记录。")

    if qs := evidence.get("kp_questions"):
        rows = qs.get("questions") or []
        if rows:
            lines = ["【该考点的历年真题原题】"]
            for q in rows[:5]:
                line = f"- {q['qid']}（{q.get('subject', '')}）{q.get('question', '')[:100]}"
                if q.get("answer"):
                    line += f"  答案 {q['answer']}"
                lines.append(line)
            parts.append("\n".join(lines))

    if rel := evidence.get("related"):
        co = rel.get("cooccurring_tags") or []
        if co:
            lines = ["【关联考点（同题共现，证据为真题编号）】"]
            for r in co[:8]:
                lines.append(
                    f"- {r['tag']}：共现 {r['cooccurrence']} 次，如 "
                    + "、".join(r.get("evidence") or [])
                )
            parts.append("\n".join(lines))
        g = rel.get("graph") or {}
        if g.get("title"):
            if g.get("siblings"):
                parts.append("【图谱同节知识点】" + "、".join(g["siblings"][:10]))

    if lp := evidence.get("learning_path"):
        chain = lp.get("chain") or []
        if chain:
            parts.append("【前置学习链（由浅入深）】\n" + " → ".join(chain))

    return "\n\n".join(parts)


def agent_quiz_messages(
    question: str,
    options: dict[str, str],
    blocks: Sequence[ContextBlock],
    evidence: dict | None = None,
    history: str = "",
) -> list[dict[str, str]]:
    """Agent 路径的选择题。最后一行仍必须是 `答案：X`，保证与一期评测口径一致。"""
    opts = "\n".join(f"{k}. {v}" for k, v in options.items())
    if not opts:
        opts = "（选项未以文字给出，见题干或配图；请根据图形解析判断 A/B/C/D）"
    ev = render_evidence(evidence or {})
    hist = f"【对话历史】\n{history}\n\n" if history else ""
    user = (
        hist
        + f"【参考资料】\n{render_context(blocks)}\n\n"
        + (f"{ev}\n\n" if ev else "")
        + f"【题目】\n{question}\n\n"
        f"【选项】\n{opts}\n\n"
        "请作答。**第一行**必须严格写成 `答案：X`，X 为选项字母；"
        "之后简要说明依据，并用 [编号] 标注来源。"
        "先把答案写出来，再解释；即使后面被打断，也能解析到选项。"
        "若使用图形解析，说明你依据的是结构信息还是图中的数值。"
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def agent_qa_messages(
    question: str,
    blocks: Sequence[ContextBlock],
    evidence: dict | None = None,
    history: str = "",
) -> list[dict[str, str]]:
    ev = render_evidence(evidence or {})
    hist = f"【对话历史】\n{history}\n\n" if history else ""
    user = (
        hist
        + f"【参考资料】\n{render_context(blocks)}\n\n"
        + (f"{ev}\n\n" if ev else "")
        + f"【问题】\n{question}\n\n"
        "请依据参考资料与工具结论作答，并用 [编号] 标注依据来源。"
    )
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def agent_exam_messages(
    topic: str,
    stats: dict,
    example_questions: Sequence[dict],
    related: Sequence[dict] | None = None,
    learning_path: dict | None = None,
    history: str = "",
) -> list[dict[str, str]]:
    """备考咨询：在 exam_analysis_messages 基础上补关联考点与前置链。"""
    msgs = exam_analysis_messages(topic, stats, example_questions)
    if history:
        msgs[1]["content"] = f"【对话历史】\n{history}\n\n" + msgs[1]["content"]
    if related:
        rel = "、".join(f"{r['tag']}（共现{r['cooccurrence']}次）" for r in related[:8])
        msgs[1]["content"] += f"\n\n【关联考点（真题共现）】{rel}"
    chain = (learning_path or {}).get("chain") or []
    if chain:
        target = (learning_path or {}).get("target") or topic
        prefix = (
            f"【用户需求：学习路径】目标知识点：{target}\n"
            "前置学习链（由浅入深）：" + " → ".join(chain) + "\n"
            "请**先按这个顺序说明先学什么、再学什么**，再简要补充历年考查情况。\n\n"
        )
        msgs[1]["content"] = prefix + msgs[1]["content"]
    return msgs


def parse_choice(text: str) -> str:
    """从模型输出里抽选择题答案。

    不同模型格式差异很大，实测见过：
        `答案：B` / `【答案】B` / `答案 B` / `**答案：B**` / 单独一行 `B`
    取不到就返回空串 —— 评测时算错，而不是瞎猜一个。
    """
    import re

    patterns = (
        r"答案\s*[:：]\s*\**\s*([ABCD])\b",
        r"【答案】\s*\**\s*([ABCD])\b",
        r"答案\s*(?:是|为)\s*\**\s*([ABCD])\b",
        r"^\s*\**\s*([ABCD])\s*\**\s*$",
        r"\b([ABCD])\b",
    )
    for p in patterns:
        m = re.search(p, text, re.M)
        if m:
            return m.group(1)
    return ""