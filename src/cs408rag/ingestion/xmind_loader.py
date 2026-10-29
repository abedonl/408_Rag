"""XMind 思维导图解析 —— 知识点图谱的数据源。

.xmind 本质是 zip，内层 content.json 是标准 JSON 树：
    [ { sheet } ]  ->  sheet["rootTopic"]  ->  children.attached[] 递归

实测（naplatte 系四科导图）：
    操作系统    1708 节点  深度 11
    组成原理    1827 节点  深度 11
    计算机网络  1669 节点  深度 11
    数据结构     978 节点  深度 10
    --------------------------------
    合计        6182 节点

横向关系线（sheet["relationships"]）实测仅 9 条，故图谱主力是层级结构，
横向扩展靠「经父节点两跳取同节兄弟」。
"""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cs408rag.config import PROCESSED_DIR, RAW_DIR

# 文件名 -> 规范科目名
SUBJECT_BY_FILENAME = {
    "数据结构": "数据结构",
    "计算机组成原理": "组成原理",
    "组成原理": "组成原理",
    "操作系统": "操作系统",
    "计算机网络": "计算机网络",
}


@dataclass
class KnowledgePoint:
    """思维导图上的一个知识点节点。"""

    node_id: str
    title: str
    subject: str
    parent_id: str | None
    depth: int
    path: list[str] = field(default_factory=list)  # 从根到本节点的标题链
    is_leaf: bool = False

    @property
    def path_str(self) -> str:
        return " > ".join(self.path)

    @property
    def chapter(self) -> str:
        """取路径上第 1 层（通常是「第一章 …」），用于教材章节对齐。"""
        return self.path[1] if len(self.path) > 1 else ""

    @property
    def section(self) -> str:
        return self.path[2] if len(self.path) > 2 else ""


@dataclass
class Relationship:
    """XMind 横向关系线。"""

    sheet: str
    from_title: str
    to_title: str


@dataclass
class MindMap:
    subject: str
    source_file: str
    root_title: str
    nodes: list[KnowledgePoint]
    relationships: list[Relationship]

    @property
    def leaf_count(self) -> int:
        return sum(1 for n in self.nodes if n.is_leaf)


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def _strip_markup(s: str) -> str:
    """导图节点标题里可能带换行/标记，清一下。"""
    return re.sub(r"\s+", " ", (s or "").replace("​", "")).strip()


def _iter_children(topic: dict) -> list[dict]:
    kids: list[dict] = []
    for key, lst in (topic.get("children") or {}).items():
        if isinstance(lst, list):
            kids.extend(x for x in lst if isinstance(x, dict))
    return kids


def parse_xmind(path: Path, subject: str | None = None) -> MindMap:
    """解析一个 .xmind 文件。"""
    if subject is None:
        stem = path.stem
        subject = next((v for k, v in SUBJECT_BY_FILENAME.items() if k in stem), stem)

    with zipfile.ZipFile(path) as outer:
        # 新版 XMind 用 content.json；老版是 content.xml
        names = outer.namelist()
        if "content.json" in names:
            payload = json.loads(outer.read("content.json").decode("utf-8"))
        elif "content.xml" in names:
            raise NotImplementedError(
                f"{path.name} 是旧版 XMind XML 格式，本项目只支持 content.json。"
                "请用新版 XMind 另存后再试。"
            )
        else:
            raise RuntimeError(f"{path.name} 既无 content.json 也无 content.xml，不是合法 XMind 文件")

    sheets = payload if isinstance(payload, list) else [payload]

    nodes: list[KnowledgePoint] = []
    relationships: list[Relationship] = []
    root_title = ""
    id2title: dict[str, str] = {}

    for sheet in sheets:
        root = sheet.get("rootTopic") or {}
        sheet_name = _strip_markup(sheet.get("title") or "")
        if not root_title:
            root_title = _strip_markup(root.get("title") or "")

        def walk(topic: dict, parent_id: str | None, depth: int, path: list[str]) -> None:
            nid = str(topic.get("id") or f"anon-{len(nodes)}")
            title = _strip_markup(topic.get("title") or "")
            kids = _iter_children(topic)
            node = KnowledgePoint(
                node_id=nid,
                title=title,
                subject=subject,
                parent_id=parent_id,
                depth=depth,
                path=[*path, title],
                is_leaf=not kids,
            )
            nodes.append(node)
            id2title[nid] = title
            for k in kids:
                walk(k, nid, depth + 1, node.path)

        walk(root, None, 0, [])

        for rel in sheet.get("relationships") or []:
            a = id2title.get(str(rel.get("end1Id") or ""), "")
            b = id2title.get(str(rel.get("end2Id") or ""), "")
            if a and b:
                relationships.append(Relationship(sheet=sheet_name, from_title=a, to_title=b))

    mm = MindMap(
        subject=subject,
        source_file=path.name,
        root_title=root_title,
        nodes=nodes,
        relationships=relationships,
    )
    logger.info(
        f"[{subject}] {path.name}: {len(nodes)} 节点（叶子 {mm.leaf_count}），"
        f"最大深度 {max((n.depth for n in nodes), default=0)}，关系线 {len(relationships)}"
    )
    return mm


def find_xmind_files(raw_dir: Path | None = None) -> list[Path]:
    root = raw_dir or (RAW_DIR / "textbook")
    if not root.exists():
        raise FileNotFoundError(f"找不到教材目录 {root}。先跑：python -m cs408rag.ingestion.download --source textbook")
    return sorted(root.rglob("*.xmind"))


def load_all_mindmaps(raw_dir: Path | None = None) -> list[MindMap]:
    return [parse_xmind(p) for p in find_xmind_files(raw_dir)]


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_graph(mindmaps: list[MindMap] | None = None) -> dict[str, Path]:
    """导出知识点节点与层级边，供 Neo4j 建图用。"""
    if mindmaps is None:
        mindmaps = load_all_mindmaps()

    all_nodes: list[dict] = []
    edges: list[dict] = []
    rels: list[dict] = []

    for mm in mindmaps:
        for n in mm.nodes:
            all_nodes.append(
                {
                    "node_id": n.node_id,
                    "title": n.title,
                    "subject": n.subject,
                    "depth": n.depth,
                    "path": n.path,
                    "path_str": n.path_str,
                    "chapter": n.chapter,
                    "section": n.section,
                    "is_leaf": n.is_leaf,
                }
            )
            if n.parent_id:
                edges.append({"child": n.node_id, "parent": n.parent_id, "subject": n.subject})

        for r in mm.relationships:
            rels.append({"subject": mm.subject, "from": r.from_title, "to": r.to_title})

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}

    for name, data in (("kp_nodes.json", all_nodes), ("kp_edges.json", edges), ("kp_relationships.json", rels)):
        p = PROCESSED_DIR / name
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        out[name] = p
        logger.info(f"写出 {name}：{len(data)} 条")

    stats = {
        "subjects": {mm.subject: len(mm.nodes) for mm in mindmaps},
        "total_nodes": len(all_nodes),
        "total_edges": len(edges),
        "max_depth": max((n["depth"] for n in all_nodes), default=0),
        "relationships": len(rels),
    }
    sp = PROCESSED_DIR / "kp_stats.json"
    sp.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    out["kp_stats.json"] = sp
    return out


if __name__ == "__main__":
    from cs408rag.config import ensure_dirs

    ensure_dirs()
    maps = load_all_mindmaps()
    print(f"\n共解析 {len(maps)} 个导图，合计 {sum(len(m.nodes) for m in maps)} 个知识点节点\n")
    for m in maps:
        print(f"  {m.subject:<10} {len(m.nodes):>5} 节点  叶子 {m.leaf_count:>5}  {m.source_file}")
    export_graph(maps)