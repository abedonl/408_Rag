"""检索子块 -> 返回父小节全文（Parent-Child）。

为什么需要：
    检索用小块更聚焦（术语密度高），但把孤立的 500 字块丢给 LLM，
    它看不到上下文。实测 408 教材一节常被切成 2–5 块，
    只给中间一块时模型容易断章取义。

做法：
    用 chunk 的 section_id 找到同节全部 chunk，按原始顺序拼成父节文本。
    解析 chunk（section_id=exam-xxx）只有一个块，不做扩展。

评测口径提醒：
    这只改变「给 LLM 的上下文」，**不会改变检索指标**。
    检索命中子块仍然按子块算 Recall；父节扩展的收益只能在答案质量上体现。
"""

from __future__ import annotations

import json
from functools import lru_cache

from loguru import logger

from cs408rag.config import PROCESSED_DIR
from cs408rag.retrieval.vector_store import SearchHit


@lru_cache(maxsize=1)
def _parent_texts() -> dict[str, str]:
    """父级标题路径 -> 该父节下全部 chunk 的文本。

    ★ 不能按 section_id 聚合：我们的 section 已经切到叶子小节，
    实测每节通常只有 1 个 chunk（父节长度 == 子块长度，扩展无效）。
    真正的父级是 chapter_path 的上一级，例如
        5.4 树、森林 > 5.4.2 树、森林与二叉树的转换 > 2. 森林转换为二叉树
    的父节是「5.4 树、森林 > 5.4.2 树、森林与二叉树的转换」。
    """
    path = PROCESSED_DIR / "chunks_all.json"
    if not path.exists():
        logger.warning(f"找不到 {path}，父节扩展不可用")
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {}
    for c in raw:
        cp = c.get("chapter_path") or []
        if len(cp) < 2:
            continue
        key = " > ".join(cp[:-1])
        groups.setdefault(key, []).append(c.get("text") or "")
    out = {k: "\n\n".join(v) for k, v in groups.items()}
    logger.info(f"父节索引：{len(out)} 个父级小节")
    return out


def expand_to_parent(
    hits: list[SearchHit],
    *,
    max_chars_per_hit: int = 1500,
) -> list[SearchHit]:
    """把教材子块替换成所属父级小节的完整文本。"""
    texts = _parent_texts()
    out: list[SearchHit] = []
    seen_parents: set[str] = set()
    for h in hits:
        if h.source == "explanation":
            out.append(h)
            continue
        cp = h.payload.get("chapter_path") or []
        key = " > ".join(cp[:-1]) if len(cp) >= 2 else ""
        parent = texts.get(key)
        if not parent or len(parent) <= len(h.text):
            out.append(h)
            continue
        # 同一父节只保留一次：检索常命中同一父节下的多个子块，
        # 展开后文本完全相同，重复塞进去只会浪费上下文预算。
        if key in seen_parents:
            continue
        seen_parents.add(key)
        if len(parent) > max_chars_per_hit:
            parent = parent[:max_chars_per_hit].rstrip() + "……"
        out.append(
            SearchHit(
                chunk_id=h.chunk_id,
                score=h.score,
                payload={**h.payload, "text": parent, "parent_expanded": True},
            )
        )
    return out