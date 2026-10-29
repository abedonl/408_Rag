"""层级感知分块。

教材不按定长硬切：优先在章节边界、段落、句子处断开，尽量不切断语义单元。
每个 chunk 都带着完整的 `chapter_path`，这样生成层才能回答
「这个知识点在教材哪一章」，而不是给一堆无出处的文本。

同时负责把真题官方解析转成 chunk —— 这里是**防泄漏红线的落地点**：
held-out 年份的解析绝不能入库，否则 RAG 检索到解析就能直接抄答案。
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cs408rag.config import PROCESSED_DIR, settings
from cs408rag.ingestion.pdf_loader import TextbookSection
from cs408rag.ingestion.quiz_loader import Question

# ---------------------------------------------------------------------------
# 分块参数
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 500  # 目标字符数
DEFAULT_OVERLAP = 80
MIN_CHUNK_SIZE = 60  # 小于此长度的尾块并入前一块

# 断句优先级：段落 > 句号类 > 分号逗号
SPLIT_PATTERNS = [
    re.compile(r"\n\s*\n"),
    re.compile(r"(?<=[。！？；])"),
    re.compile(r"(?<=[，、])"),
]


@dataclass
class Chunk:
    """一个可检索单元。"""

    chunk_id: str
    text: str
    subject: str
    source: str  # textbook | explanation
    chapter_path: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    section_id: str = ""
    # 防泄漏：解析类 chunk 记录其年份，检索层据此过滤 held-out
    year: int | None = None
    # held-out 年份的解析永不入库，这里再兜一层
    is_holdout: bool = False
    # 关联的知识点节点 id（由 tag_mapper / kp_linker 填充）
    kp_ids: list[str] = field(default_factory=list)
    # 章节类别：concept（概念讲解）| exercise（试题精选/答案与解析/习题）
    #
    # 为什么要在 chunk 上区分：王道每章末尾的「本节试题精选」「答案与解析」
    # **直接收录历年真题原文**。它们留在索引里对答标答类问题有用，
    # 但在**检索评测**中必须排除 —— 否则检索器靠题面字面匹配就命中，
    # 占满 top-k 而掩盖真正的概念检索能力。实测这是压低指标的隐形杀手。
    chunk_class: str = "concept"
    # 该 chunk 对应的教材插图：[{image_id, src, caption}]
    # src 是相对 Markdown 文件的路径，实际可用路径由 image_index.json 提供
    images: list[dict] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def chapter(self) -> str:
        return self.chapter_path[0] if self.chapter_path else ""

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_payload(self) -> dict[str, Any]:
        """转成 Qdrant payload。

        ★ meta 必须写进来：解析 chunk 的 qid/answer/tags 都存在 meta 里，
        早先漏了这一行，检索结果里拿不到结构化答案，只能从正文里猜。
        """
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "subject": self.subject,
            "source": self.source,
            "chapter_path": self.chapter_path,
            "chapter": self.chapter,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "section_id": self.section_id,
            "year": self.year,
            "kp_ids": self.kp_ids,
            "chunk_class": self.chunk_class,
            "images": self.images,
            "meta": self.meta,
        }


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]
    return h


# ---------------------------------------------------------------------------
# 文本切分
# ---------------------------------------------------------------------------


def _split_long(text: str, size: int) -> list[str]:
    """把超过 size 的文本按断句优先级切成若干片。"""
    for pat in SPLIT_PATTERNS:
        pieces = [p for p in pat.split(text) if p and p.strip()]
        if len(pieces) > 1:
            out: list[str] = []
            buf = ""
            for p in pieces:
                if len(buf) + len(p) <= size:
                    buf += p
                else:
                    if buf:
                        out.append(buf)
                    # 单句仍超长，递归降级到下一级切分
                    if len(p) > size:
                        out.extend(_split_long(p, size))
                        buf = ""
                    else:
                        buf = p
            if buf:
                out.append(buf)
            return out
    # 所有断句符都失效（如无标点的长串），硬切
    return [text[i : i + size] for i in range(0, len(text), size)]


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """把一段文本切成带重叠的 chunk。"""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    parts = _split_long(text, chunk_size)

    # 过短的尾块并入前一块，避免产生无意义的碎片
    merged: list[str] = []
    for p in parts:
        if merged and len(p) < MIN_CHUNK_SIZE:
            merged[-1] = merged[-1] + p
        else:
            merged.append(p)

    if overlap <= 0 or len(merged) == 1:
        return merged

    # 加重叠：每块头部拼接前一块的尾部
    out = [merged[0]]
    for prev, cur in zip(merged, merged[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        out.append(tail + cur)
    return out


# ---------------------------------------------------------------------------
# 教材 -> chunks
# ---------------------------------------------------------------------------


def chunk_sections(
    sections: Iterable[TextbookSection],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """把教材小节切成 chunks，保留章节路径。"""
    chunks: list[Chunk] = []
    for sec in sections:
        pieces = split_text(sec.text, chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=_stable_id("tb", sec.section_id, str(i)),
                    text=piece,
                    subject=sec.subject,
                    source="textbook",
                    chapter_path=list(sec.path),
                    page_start=sec.page_start,
                    page_end=sec.page_end,
                    section_id=sec.section_id,
                )
            )
    logger.info(f"教材分块：{len(chunks)} 个 chunk")
    return chunks


# ---------------------------------------------------------------------------
# 真题解析 -> chunks（含防泄漏红线）
# ---------------------------------------------------------------------------


def chunk_explanations(
    questions: Iterable[Question],
    holdout_years: set[int] | None = None,
    figures_by_qid: dict[str, list[dict]] | None = None,
) -> tuple[list[Chunk], list[Question]]:
    """把真题官方解析转成 chunks。

    返回 (可入库的 chunks, 被挡下的题目)。

    ★ 红线：held-out 年份的解析一律不入库。这些年份的真题只用于最终评测，
      解析若进索引，RAG 检索到即可直接抄答案，准确率虚高且评测失去意义。
    """
    holdout_years = holdout_years if holdout_years is not None else settings.holdout_year_set

    chunks: list[Chunk] = []
    blocked: list[Question] = []

    for q in questions:
        if q.year in holdout_years:
            blocked.append(q)
            continue
        if not q.explanation.strip():
            continue

        # 解析正文里带上题干与选项：
        #   - 题干让检索能同时命中「问法」与「答法」
        #   - 选项对「为什么选 B / 选项 C 错在哪」这类问答是必要信息，
        #     之前只存题干，模型只能看到答案字母，无法解释其他选项。
        opts = " ".join(f"{k}. {v}" for k, v in (q.options or {}).items())
        text = f"{q.stem}\n" + (f"选项：{opts}\n" if opts else "") + f"解析：{q.explanation}"
        chunks.append(
            Chunk(
                chunk_id=_stable_id("ex", q.qid),
                text=text,
                subject=q.subject,
                source="explanation",
                chapter_path=[],
                section_id=f"exam-{q.qid}",
                year=q.year,
                is_holdout=False,
                # 解析里的 SVG 图（由 exam_figures 提取并落盘）
                images=list((figures_by_qid or {}).get(q.qid, [])),
                meta={"qid": q.qid, "answer": q.answer, "tags": q.tags},
            )
        )

    logger.info(
        f"解析分块：入库 {len(chunks)} 条；"
        f"★ 因 held-out 隔离挡下 {len(blocked)} 题（年份 {sorted({q.year for q in blocked})}）"
    )
    return chunks, blocked


def assert_no_holdout_leak(chunks: Iterable[Chunk], holdout_years: set[int] | None = None) -> None:
    """入库前的最后一道闸。

    宁可在这里炸掉，也不要让 held-out 解析悄悄进了索引 ——
    那种错误不会报错，只会让最终准确率虚高，且极难被发现。
    """
    holdout_years = holdout_years if holdout_years is not None else settings.holdout_year_set
    leaks = [c for c in chunks if c.source == "explanation" and c.year in holdout_years]
    if leaks:
        years = sorted({c.year for c in leaks})
        raise RuntimeError(
            f"防泄漏校验失败：有 {len(leaks)} 条 held-out 年份({years})的解析试图入库。"
            "这会直接污染评测结果，已中止。"
        )


def assert_unique_ids(chunks: Sequence[Chunk]) -> None:
    """校验 chunk_id 全局唯一。

    为什么需要这道检查：
        Qdrant 按 point id 覆盖写入 —— id 撞了不会报错，只是**静默丢数据**。
        实测踩过一次：section_id 只含文件内序号不含文件名，
        不同章节里重复出现的「答案与解析」等标题互相撞车，
        243 处冲突导致 311 个 chunk（5.3%）被覆盖。

        这类 bug 在检索效果上表现为「某些内容永远搜不到」，
        排查起来极其痛苦。让它在这里炸，比事后查索引缺了什么划算得多。
    """
    from collections import Counter

    counts = Counter(c.chunk_id for c in chunks)
    dups = {cid: n for cid, n in counts.items() if n > 1}
    if dups:
        sample = list(dups)[:5]
        raise RuntimeError(
            f"chunk_id 冲突：{len(dups)} 个重复 id，涉及 {sum(dups.values())} 个 chunk。"
            f"写入 Qdrant 时会静默覆盖丢数据。样例：{sample}"
        )


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_chunks(chunks: list[Chunk], name: str) -> str:
    import json

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p = PROCESSED_DIR / f"{name}.json"
    p.write_text(
        json.dumps([c.to_payload() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"写出 {p}（{len(chunks)} 个 chunk）")
    return str(p)


def chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    from collections import Counter

    lens = [c.char_count for c in chunks] or [0]
    return {
        "count": len(chunks),
        "by_source": dict(Counter(c.source for c in chunks)),
        "by_subject": dict(Counter(c.subject for c in chunks)),
        "char_min": min(lens),
        "char_max": max(lens),
        "char_avg": round(sum(lens) / len(lens), 1),
    }


# ---------------------------------------------------------------------------
# 王道教材 -> chunks（★ 主语料）
# ---------------------------------------------------------------------------

# 试题/答案类小节的特征词。
#
# 王道每章末尾有「本节试题精选」与「答案与解析」两节，**直接收录历年真题原文**。
# 它们留在索引里对答标答类问题有用，但在**检索评测**中必须能识别并排除：
# 否则检索器靠题面字面匹配就命中这些小节、占满 top-k，
# 掩盖真正的概念检索能力 —— 实测这是压低评测指标的隐形杀手。
EXERCISE_SECTION_PATTERNS = ("答案与解析", "试题精选", "习题", "真题", "模拟题")


def chunk_md_sections(
    sections: Iterable[Any],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    min_size: int = 120,
) -> list[Chunk]:
    """把王道教材小节切成 chunk，保留完整标题层级。

    与 chunk_sections 的区别：保留完整的 path（书 > 章 > 节 > 小节），
    这样生成层能精确回答「这个知识点在王道第几章第几节」，
    引用溯源时也能给到可点击的章节位置。

    小节标题会拼进 chunk 正文开头 —— 标题本身携带强检索信号
    （如「5.1.1 树的定义」），丢掉它会明显伤害召回。
    """
    chunks: list[Chunk] = []
    for sec in sections:
        title = getattr(sec, "title", "") or ""
        body = sec.text
        path = list(getattr(sec, "path", []) or [])
        header = f"【{' > '.join(path)}】\n" if path else ""

        # 标注章节类别：试题/答案类小节含真题原文，检索评测要能识别并排除
        label = " ".join([*path, title])
        cls = "exercise" if any(p in label for p in EXERCISE_SECTION_PATTERNS) else "concept"

        pieces = split_text(body, chunk_size, overlap)

        # --- 把本节插图分配到具体的 chunk ---
        #
        # 不能无差别挂给所有 chunk：一节常切成 3-5 个 chunk，
        # 都挂上同一批图会让检索结果里塞满无关插图。
        # 用图片的 anchor（图前 60 字正文）或图注去各 chunk 里找，
        # 命中哪个就挂哪个；都找不到就挂到第一个 chunk。
        sec_images = list(getattr(sec, "images", []) or [])
        piece_images: list[list[dict]] = [[] for _ in pieces]
        # pieces 可能为空（整节只剩图片、没有正文），此时没有 chunk 可挂
        if pieces and sec_images:
            for img in sec_images:
                target = 0
                anchor = (img.get("anchor") or "")[-30:]
                caption = img.get("caption") or ""
                for i, piece in enumerate(pieces):
                    hit = (anchor and anchor in piece) or (caption and caption in piece)
                    if hit:
                        target = i
                        break
                piece_images[target].append(
                    {"image_id": img.get("image_id", ""), "src": img.get("src", ""),
                     "caption": caption}
                )

        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    chunk_id=_stable_id("wd", sec.section_id, str(i)),
                    text=f"{header}{title}\n{piece}" if i == 0 else f"{header}{title}（续）\n{piece}",
                    subject=sec.subject,
                    source="wangdao",
                    chapter_path=path,
                    section_id=sec.section_id,
                    chunk_class=cls,
                    images=piece_images[i],
                    meta={
                        "section_title": title,
                        "level": getattr(sec, "level", 2),
                        "source_file": getattr(sec, "source_file", ""),
                        "part": i,
                    },
                )
            )
    n_ex = sum(1 for c in chunks if c.chunk_class == "exercise")
    n_img = sum(len(c.images) for c in chunks)
    n_chunks_with_img = sum(1 for c in chunks if c.images)
    logger.info(
        f"王道教材分块：{len(chunks)} 个 chunk（concept {len(chunks) - n_ex} / exercise {n_ex}）；"
        f"插图 {n_img} 张分布在 {n_chunks_with_img} 个 chunk 上"
    )
    return chunks


# ---------------------------------------------------------------------------
# 思维导图 -> chunks（本项目的主语料来源）
# ---------------------------------------------------------------------------
#
# 背景：naplatte 仓库里的「教材 PDF」实为 XMind 的扁平导出
# （实测 PDF 文本行 84.8% 逐字符命中 XMind 节点标题），没有成段讲解文字。
# 所以语料直接从思维导图树构建。
#
# 关键设计：**不重叠的 DFS 分区**，而不是按多粒度嵌套切分。
#
#   若按深度 2/3/4/5 各切一遍，同一个知识点会出现在多个嵌套 chunk 里，
#   而评测的 ground truth 是「题目 tag -> 知识点 -> 含有该知识点的 chunk」，
#   一个知识点对应多个 chunk 会让 Recall@k 虚高（随便召回一个就算命中）。
#
#   DFS 分区保证**每个知识点节点恰好属于一个 chunk**，
#   于是 ground truth 是干净的一对一/一对少映射，指标才有意义。


MINDMAP_TARGET_SIZE = 420
MINDMAP_MIN_SIZE = 60
MINDMAP_MAX_DEPTH = 9


def _render_outline(entries: list[tuple[int, str]], base_depth: int) -> str:
    """把 [(depth, title)] 渲染成缩进大纲文本。"""
    lines: list[str] = []
    for depth, title in entries:
        indent = "  " * max(depth - base_depth, 0)
        lines.append(f"{indent}{title}")
    return "\n".join(lines)


def chunk_mindmap(
    mindmaps: Iterable[Any],
    *,
    target_size: int = MINDMAP_TARGET_SIZE,
    min_size: int = MINDMAP_MIN_SIZE,
) -> list[Chunk]:
    """把思维导图树切成不重叠的 chunk。

    Args:
        mindmaps: MindMap 列表（来自 xmind_loader.load_all_mindmaps）
        target_size: 单个 chunk 的目标字符数
        min_size: 小于此长度的尾块并入前一块
    """
    chunks: list[Chunk] = []

    for mm in mindmaps:
        nodes = [n for n in mm.nodes if n.depth <= MINDMAP_MAX_DEPTH]
        by_parent: dict[str | None, list[Any]] = {}
        for n in nodes:
            by_parent.setdefault(n.parent_id, []).append(n)
        for kids in by_parent.values():
            kids.sort(key=lambda k: k.title)

        # --- DFS 分区 ---
        buf: list[Any] = []
        buf_len = 0

        def flush(*, force: bool = False) -> None:
            nonlocal buf, buf_len
            if not buf:
                return
            if not force and buf_len < min_size and chunks and chunks[-1].subject == mm.subject:
                # 太短的尾块并进前一块（仅当同科目，避免跨科混淆）
                prev = chunks[-1]
                prev.text = prev.text + "\n" + _render_outline(
                    [(n.depth, n.title) for n in buf], buf[0].depth
                )
                prev.kp_ids = sorted(set(prev.kp_ids) | {n.node_id for n in buf})
                buf, buf_len = [], 0
                return

            base = buf[0].depth
            text = _render_outline([(n.depth, n.title) for n in buf], base)
            head = buf[0]
            chunks.append(
                Chunk(
                    chunk_id=_stable_id("kp", head.node_id, str(len(chunks))),
                    text=text,
                    subject=mm.subject,
                    source="mindmap",
                    # 用起始节点的祖先链当章节路径，供生成层回答「在教材哪一章」
                    chapter_path=list(head.path[:-1]),
                    section_id=head.node_id,
                    # ★ 该 chunk 覆盖的全部知识点节点，是 MENTIONS 边的来源
                    kp_ids=sorted({n.node_id for n in buf}),
                    meta={"kp_titles": [n.title for n in buf[:8]], "depth": base},
                )
            )
            buf, buf_len = [], 0

        def walk(node: Any) -> None:
            nonlocal buf_len
            buf.append(node)
            buf_len += len(node.title) + 1
            if buf_len >= target_size:
                flush()
            for child in by_parent.get(node.node_id, []):
                walk(child)

        roots = by_parent.get(None, [])
        for r in roots:
            walk(r)
        flush(force=True)

    logger.info(f"思维导图分块：{len(chunks)} 个 chunk（不重叠分区）")
    return chunks