"""知识点前置依赖边（PREREQUISITE）。

图谱只有层级（CHILD_OF），没有「学 A 之前要先会 B」。学习路径推荐需要这条边。

设计约束（避免 LLM 乱连）：

1. **候选只取同一父节点下的相邻兄弟**（按导图顺序 A 在前、B 在后）。
   这样生成的图天然无环，不需要额外的环检测。
2. LLM 只做**判定**，不给自由发挥：输入两个标题 + 所属章节，输出 true/false。
   它不能编造节点，也不能跨章连线。
3. 判定为真的边写入 Neo4j：`(A)-[:PREREQUISITE]->(B)`，并保存置信度。

为什么不用「同章所有小节两两组合」：候选量爆炸（几百 × 几百），
且跨节的依赖往往是弱关联，LLM 判定噪声更大。
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cs408rag.config import PROCESSED_DIR

OUT_PATH = PROCESSED_DIR / "prerequisites.json"


@dataclass
class PrereqEdge:
    from_id: str
    from_title: str
    to_id: str
    to_title: str
    chapter: str
    confidence: float = 0.8
    reason: str = ""


@dataclass
class PrereqResult:
    candidates: int = 0
    accepted: int = 0
    rejected: int = 0
    edges: list[PrereqEdge] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidates": self.candidates,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "accepted_rate": round(self.accepted / self.candidates, 4)
            if self.candidates
            else 0.0,
            "edges": [
                {
                    "from_id": e.from_id,
                    "from_title": e.from_title,
                    "to_id": e.to_id,
                    "to_title": e.to_title,
                    "chapter": e.chapter,
                    "confidence": e.confidence,
                    "reason": e.reason,
                }
                for e in self.edges
            ],
        }


def candidate_pairs(*, max_depth: int = 3) -> list[tuple[dict, dict]]:
    """同父节点下按顺序相邻的兄弟对 (A, B)。

    不能读 kp_nodes.json：导出的 JSON 为了精简没有 `parent_id` 字段，
    必须走 xmind_loader（它保留 parent_id 与原始遍历顺序）。
    """
    from cs408rag.ingestion.xmind_loader import load_all_mindmaps

    nodes = [n for mm in load_all_mindmaps() for n in mm.nodes]
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for k in nodes:
        if k.parent_id and k.depth <= max_depth and k.title:
            by_parent[k.parent_id].append(
                {"node_id": k.node_id, "title": k.title, "path": k.path}
            )

    pairs: list[tuple[dict, dict]] = []
    for kids in by_parent.values():
        for a, b in zip(kids, kids[1:], strict=False):
            pairs.append((a, b))
    logger.info(f"前置依赖候选：{len(pairs)} 对（同父相邻兄弟，depth≤{max_depth}）")
    return pairs


def _judge_batch(pairs: list[tuple[dict, dict]], llm, *, chunk_size: int = 8) -> dict[str, bool]:
    """让 LLM 判定一批候选。返回 {a_title||b_title: bool}。"""
    out: dict[str, bool] = {}
    for i in range(0, len(pairs), chunk_size):
        batch = pairs[i : i + chunk_size]
        lines = []
        for a, b in batch:
            chapter = (a.get("path") or ["", ""])[1] if len(a.get("path") or []) > 1 else ""
            lines.append(
                f"- A={a['title']} | B={b['title']} | 章节={chapter}"
            )
        prompt = (
            "你在为计算机考研 408 构建知识点前置依赖。\n"
            "判断每一对 (A, B)：**要学习 B，是否必须先掌握 A？**\n"
            "判定标准：只有 A 的概念/原理会被 B 直接使用，才算前置（true）。\n"
            "仅同章相邻、主题相近、或 A 是 B 的并列分类，都算 false。\n"
            "只输出 JSON，形如 {\"A标题||B标题\": true, \"A标题||B标题\": false}。\n\n"
            + "\n".join(lines)
        )
        try:
            resp = llm.chat(
                [{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"前置依赖判定失败，跳过本批：{e}")
            continue
        if not resp.ok:
            logger.warning(f"前置依赖判定失败：{resp.error[:120]}")
            continue
        m = re.search(r"\{.*\}", resp.text, re.S)
        if not m:
            continue
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        for a, b in batch:
            key = f"{a['title']}||{b['title']}"
            if key in data:
                out[key] = bool(data[key])
        logger.info(f"  判定进度 {min(i + chunk_size, len(pairs))}/{len(pairs)}")
    return out


def build_edges(*, max_depth: int = 3, llm=None) -> PrereqResult:
    pairs = candidate_pairs(max_depth=max_depth)
    result = PrereqResult(candidates=len(pairs))
    if not pairs:
        return result
    if llm is None:
        from cs408rag.generation.llm import get_llm

        llm = get_llm()

    verdicts = _judge_batch(pairs, llm)
    for a, b in pairs:
        key = f"{a['title']}||{b['title']}"
        ok = verdicts.get(key)
        if ok is True:
            result.accepted += 1
            result.edges.append(
                PrereqEdge(
                    from_id=a["node_id"],
                    from_title=a["title"],
                    to_id=b["node_id"],
                    to_title=b["title"],
                    chapter=(a.get("path") or ["", ""])[1]
                    if len(a.get("path") or []) > 1
                    else "",
                )
            )
        elif ok is False:
            result.rejected += 1
    return result


def save(result: PrereqResult) -> Path:
    OUT_PATH.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"前置依赖边 -> {OUT_PATH}")
    return OUT_PATH


def load() -> list[dict]:
    if not OUT_PATH.exists():
        return []
    return json.loads(OUT_PATH.read_text(encoding="utf-8")).get("edges", [])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="构建知识点前置依赖边")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--write-graph", action="store_true", help="写入 Neo4j")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    res = build_edges(max_depth=args.max_depth)
    save(res)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2)[:1200])

    if args.write_graph and res.edges:
        from cs408rag.retrieval.graph_store import GraphStore

        with GraphStore() as gs:
            n = gs.load_prerequisites(
                [
                    {
                        "from_id": e.from_id,
                        "to_id": e.to_id,
                        "confidence": e.confidence,
                    }
                    for e in res.edges
                ]
            )
            print(f"写入 Neo4j：{n} 条 PREREQUISITE 边")