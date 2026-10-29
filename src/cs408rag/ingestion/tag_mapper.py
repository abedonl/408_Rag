"""真题 tag → XMind 知识点 映射。

为什么必须有这一步：
    真题自带 226 个 tag，是**粗粒度**的（如「排序算法」「中断IO」）；
    XMind 图谱有 6182 个**细粒度**节点（如「快速排序的性能分析」）。
    检索评测的 ground truth 走的是
        题目 tags → 知识点节点 → 教材 chunk
    这条链。映射错了，整条评测链的数字都不可信。

因此映射质量必须单独评估并报告，不能默默吞掉未匹配项。

匹配分三级，逐级降级：
    exact    标题完全相同                          conf 1.00
    contains 一方包含另一方（长度差需在阈值内）      conf 0.75
    llm      LLM 从候选里挑（需 API key，可选）      conf 0.60
    none     未匹配 → 显式记录，绝不静默丢弃
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cs408rag.config import PROCESSED_DIR, settings
from cs408rag.ingestion.xmind_loader import KnowledgePoint

# 「contains」匹配的长度门槛。
#
# 这里踩过一个坑，记录一下：最初对中英文统一用 3，结果 B树 / 链表 / 补码 / 总线 / 子网
# 这些 2 字中文术语**全部**匹配失败 —— 中文 2 字就是完整术语，
# 不能用英文的长度直觉去卡。所以按字符类型分别设阈值：
#   含 CJK 的 tag：2 字即有效（B树、链表、补码）
#   纯 ASCII 的 tag：3 字符起步（IO、ARQ 这类 2 字母缩写太泛）
MIN_CONTAINS_LEN_CJK = 2
MIN_CONTAINS_LEN_ASCII = 3
MAX_MATCHES_PER_TAG = 12  # 一个粗 tag 最多映射到多少个细节点

# 别名表：真题 tag 与导图用词不一致时手工对齐。
# 这些是跑完匹配后逐个核对未匹配项得出的，不是猜的。
TAG_ALIASES: dict[str, str] = {
    "OSI模型": "OSI参考模型",
    "ISO/OSI参考模型": "OSI参考模型",
    "IEEE浮点数表示": "IEEE754标准",
    "浮点数表示": "IEEE754标准",
    "位图法": "位示图法",
    "bitmap": "位示图法",
    "CDMA": "码分多路复用CDM",
    "子网聚合": "子网划分与子网掩码",
    "CSMA/CD限制条件": "CSMA/CD协议",
    "指令种类": "指令格式",
    "程序的装入": "程序的装入与链接",
    "单周期处理器": "单指令周期",
    "总线指标": "总线性能指标",
}

# 匹配前清洗：去掉标点与空白，统一全半角
_PUNCT = re.compile(r"[\s()（）\[\]【】、,，。.：:；;“”\"'’‘\-—_/\\]+")
_CJK = re.compile(r"[一-鿿]")

# 不能单独作为映射目标的泛化节点。
# 实测：contains 优先短标题时，「交换方式」会匹配到「方式」、「总线性能指标」
# 会匹配到「性能指标」——这类节点在教材里遍地都是，拿它当 GT 上游会污染整条链。
# 规则：节点标题本身是这些泛化词、或只有 1 个字符时，不作为映射结果。
GENERIC_NODE_TITLES = {
    "概念", "特点", "分类", "定义", "作用", "方法", "管理", "算法", "结构",
    "原理", "组成", "功能", "性质", "条件", "方式", "过程", "情况", "内容",
    "总结", "应用", "实现", "机制", "技术", "指标", "性能", "性能指标",
    "种类", "类型", "概述", "简介", "特征", "意义", "目的",
}


def normalize(s: str) -> str:
    return _PUNCT.sub("", (s or "").strip()).lower()


def _min_len_for(s: str) -> int:
    """中文 2 字即有效术语；纯 ASCII 需要 3 字符以上。"""
    return MIN_CONTAINS_LEN_CJK if _CJK.search(s) else MIN_CONTAINS_LEN_ASCII


@dataclass
class TagMatch:
    tag: str
    kp_ids: list[str] = field(default_factory=list)
    kp_titles: list[str] = field(default_factory=list)
    method: str = "none"  # exact | contains | llm | none
    confidence: float = 0.0
    frequency: int = 0  # 该 tag 在真题中出现的次数，便于优先处理高频考点

    @property
    def matched(self) -> bool:
        return bool(self.kp_ids)


@dataclass
class MappingReport:
    total_tags: int
    matched: int
    by_method: dict[str, int]
    unmatched_tags: list[str]
    coverage: float
    total_edges: int

    def to_dict(self) -> dict:
        return {
            "total_tags": self.total_tags,
            "matched_tags": self.matched,
            "coverage": round(self.coverage, 4),
            "by_method": self.by_method,
            "total_edges": self.total_edges,
            "unmatched_tags": self.unmatched_tags,
            "note": (
                "coverage 是评测链上游指标：未匹配的 tag 其对应真题无法构造检索 ground truth。"
                "该比例必须写进 README，不得隐瞒。"
            ),
        }


# ---------------------------------------------------------------------------
# 索引与匹配
# ---------------------------------------------------------------------------


def build_kp_index(kps: list[KnowledgePoint]) -> dict[str, list[KnowledgePoint]]:
    """按规范化标题建倒排索引。"""
    idx: dict[str, list[KnowledgePoint]] = {}
    for kp in kps:
        key = normalize(kp.title)
        if key:
            idx.setdefault(key, []).append(kp)
    return idx


def match_tag(tag: str, kp_index: dict[str, list[KnowledgePoint]]) -> TagMatch:
    """对单个 tag 做匹配（不含 llm 级）。

    匹配顺序：
        0. 别名表   —— 人工对齐的用词差异
        1. 完全相同
        2. 包含关系 —— 去掉长度比上限，改用「标题越短越核心」排序

    为什么去掉长度比上限：
        导图节点大量是长描述句（如「叶结点是失败结点，B树的高度不包括最后一层叶子」），
        tag「B树」明明出现在里面，但按长度比会被判为过宽而丢弃。
        这个规则对英文长文档有效，对「短术语 vs 长句标题」的中文场景是错的。
        改法：允许命中，但按标题长度升序排（短的更可能是概念本身），并截断到上限。
    """
    # 0) 别名
    canonical = TAG_ALIASES.get(tag, tag)
    key = normalize(canonical)
    if not key:
        return TagMatch(tag=tag, method="none")

    # 1) 完全相同
    if key in kp_index:
        hits = _rank_candidates(kp_index[key])
        return TagMatch(
            tag=tag,
            kp_ids=[k.node_id for k in hits],
            kp_titles=[k.title for k in hits],
            method="exact",
            confidence=1.0,
        )

    # 2) 包含关系
    if len(key) < _min_len_for(key):
        return TagMatch(tag=tag, method="none")

    hits: list[KnowledgePoint] = []
    for kp_key, kps in kp_index.items():
        if not kp_key:
            continue
        if not (key in kp_key or kp_key in key):
            continue
        # 只过滤单字节点（栈/树/图/度）。它们几乎必然是层级树里的分类节点，
        # 作为映射目标会让 GT 定位到几十上百个无关小节。
        # 更细的泛化词（方式/性能指标）留给 GT 构建端过滤 ——
        # 在这里一刀切会把「处理机调度算法 → 调度算法」这类正确映射也杀掉。
        if len(kp_key) <= 1:
            continue
        hits.extend(kps)

    if not hits:
        return TagMatch(tag=tag, method="none")

    hits = _rank_candidates(hits, tag_key=key)
    return TagMatch(
        tag=tag,
        kp_ids=[k.node_id for k in hits],
        kp_titles=[k.title for k in hits],
        method="contains",
        confidence=0.75,
    )


def _rank_candidates(
    kps: list[KnowledgePoint], *, tag_key: str | None = None
) -> list[KnowledgePoint]:
    """按「越像 tag 对应的概念」排序并截断。

    排序键：
        1. 字面覆盖度降序 —— 节点标题覆盖了 tag 越多字符，越可能是同一个概念
           （早先按标题最短排序，导致「栈」「方式」这类泛化词排第一）
        2. 标题长度升序 —— 短标题更可能是概念名
        3. 深度降序      —— 深节点更具体
    """
    seen: set[str] = set()
    uniq: list[KnowledgePoint] = []
    for k in kps:
        if k.node_id not in seen and k.title.strip():
            seen.add(k.node_id)
            uniq.append(k)

    if tag_key:
        uniq.sort(
            key=lambda k: (
                -len(set(tag_key) & set(normalize(k.title))) / max(len(tag_key), 1),
                len(k.title),
                -k.depth,
            )
        )
    else:
        uniq.sort(key=lambda k: (len(k.title), -k.depth))
    return uniq[:MAX_MATCHES_PER_TAG]


def map_tags(
    tags_with_freq: dict[str, int],
    kps: list[KnowledgePoint],
    *,
    use_llm: bool = False,
) -> tuple[dict[str, TagMatch], MappingReport]:
    """批量映射。tags_with_freq 来自 quiz_loader.tag_frequency()。"""
    kp_index = build_kp_index(kps)
    logger.info(f"知识点索引：{len(kp_index)} 个规范化标题（原始节点 {len(kps)}）")

    matches: dict[str, TagMatch] = {}
    for tag, freq in tags_with_freq.items():
        m = match_tag(tag, kp_index)
        m.frequency = freq
        matches[tag] = m

    unmatched = [t for t, m in matches.items() if not m.matched]

    if use_llm and unmatched:
        logger.info(f"对 {len(unmatched)} 个未匹配 tag 尝试 LLM 补配")
        _llm_match(matches, unmatched, kps)

    by_method: dict[str, int] = {}
    for m in matches.values():
        by_method[m.method] = by_method.get(m.method, 0) + 1

    matched = sum(1 for m in matches.values() if m.matched)
    report = MappingReport(
        total_tags=len(matches),
        matched=matched,
        by_method=by_method,
        unmatched_tags=[t for t, m in matches.items() if not m.matched],
        coverage=matched / len(matches) if matches else 0.0,
        total_edges=sum(len(m.kp_ids) for m in matches.values()),
    )
    logger.info(
        f"映射完成：{matched}/{len(matches)} = {report.coverage:.1%} 覆盖，"
        f"共 {report.total_edges} 条边，未匹配 {len(report.unmatched_tags)} 个"
    )
    return matches, report


def _llm_match(
    matches: dict[str, TagMatch],
    unmatched: list[str],
    kps: list[KnowledgePoint],
    batch: int = 20,
) -> None:
    """用 LLM 为未匹配 tag 补配知识点。

    可选步骤：没有配置 API key 就跳过，未匹配项如实记录即可。
    LLM 只负责**从候选中挑选**，不允许自己编造知识点名。
    """
    if not settings.openai_api_key:
        logger.warning("未配置 OPENAI_API_KEY，跳过 LLM 补配。未匹配项将如实记录。")
        return

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai 未安装，跳过 LLM 补配")
        return

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    # 候选用叶子节点的标题（细粒度、可检索）
    leaf_titles = sorted({k.title for k in kps if k.is_leaf and k.title})
    catalog = "\n".join(leaf_titles)
    if len(catalog) > 60000:  # 控制 prompt 体积
        catalog = catalog[:60000]

    for i in range(0, len(unmatched), batch):
        chunk = unmatched[i : i + batch]
        prompt = (
            "你是计算机考研 408 的助教。下面给出若干「考点标签」，"
            "以及一份知识点清单。\n"
            "请为每个标签从清单中挑出最相关的知识点标题（0-3 个）。\n"
            "严格要求：**只能从清单中原样选取**，不得改写、不得自创。\n"
            "若清单中没有相关项，返回空列表。\n\n"
            f"知识点清单：\n{catalog}\n\n"
            f"待匹配标签：\n{json.dumps(chunk, ensure_ascii=False)}\n\n"
            '只输出 JSON，格式：{"标签": ["知识点标题", ...], ...}'
        )
        try:
            resp = client.chat.completions.create(
                model=settings.llm_model_cheap,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"LLM 补配第 {i // batch + 1} 批失败：{e}")
            continue

        title2kp: dict[str, list[KnowledgePoint]] = {}
        for kp in kps:
            title2kp.setdefault(kp.title, []).append(kp)

        for tag, titles in data.items():
            if tag not in matches or not isinstance(titles, list):
                continue
            hits: list[KnowledgePoint] = []
            for t in titles:
                hits.extend(title2kp.get(str(t).strip(), []))
            if hits:
                hits = hits[:MAX_MATCHES_PER_TAG]
                matches[tag] = TagMatch(
                    tag=tag,
                    kp_ids=[k.node_id for k in hits],
                    kp_titles=[k.title for k in hits],
                    method="llm",
                    confidence=0.60,
                    frequency=matches[tag].frequency,
                )
        logger.info(f"LLM 补配进度 {min(i + batch, len(unmatched))}/{len(unmatched)}")


# ---------------------------------------------------------------------------
# 导出
# ---------------------------------------------------------------------------


def export_mapping(matches: dict[str, TagMatch], report: MappingReport) -> dict[str, str]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "tag": m.tag,
            "frequency": m.frequency,
            "method": m.method,
            "confidence": m.confidence,
            "kp_ids": m.kp_ids,
            "kp_titles": m.kp_titles,
        }
        for m in sorted(matches.values(), key=lambda x: -x.frequency)
    ]
    p1 = PROCESSED_DIR / "tag_kp_mapping.json"
    p1.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    p2 = PROCESSED_DIR / "tag_kp_mapping_report.json"
    p2.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"映射 -> {p1}")
    logger.info(f"映射质量报告 -> {p2}")
    return {"mapping": str(p1), "report": str(p2)}


def load_mapping(path: Path | None = None) -> dict[str, list[str]]:
    """读回映射，返回 {tag: [kp_id, ...]}。"""
    p = path or (PROCESSED_DIR / "tag_kp_mapping.json")
    if not p.exists():
        raise FileNotFoundError(f"找不到映射文件 {p}。先跑一次 tag_mapper。")
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {r["tag"]: r["kp_ids"] for r in rows if r.get("kp_ids")}


if __name__ == "__main__":
    import argparse

    from cs408rag.config import ensure_dirs
    from cs408rag.ingestion.quiz_loader import load_questions, tag_frequency
    from cs408rag.ingestion.xmind_loader import load_all_mindmaps

    ap = argparse.ArgumentParser(description="构建 tag -> 知识点 映射")
    ap.add_argument("--llm", action="store_true", help="对未匹配项启用 LLM 补配（需 API key）")
    args = ap.parse_args()

    ensure_dirs()
    questions = load_questions()
    freq = tag_frequency(questions)
    kps = [n for mm in load_all_mindmaps() for n in mm.nodes]

    matches, report = map_tags(freq, kps, use_llm=args.llm)
    export_mapping(matches, report)

    print(f"\n映射覆盖率：{report.matched}/{report.total_tags} = {report.coverage:.1%}")
    print(f"匹配方式分布：{report.by_method}")
    print(f"总边数：{report.total_edges}")
    if report.unmatched_tags:
        top_unmatched = sorted(
            report.unmatched_tags, key=lambda t: -freq.get(t, 0)
        )[:15]
        print("\n未匹配 tag 中频次最高的 15 个（这些考点的真题无法构造 ground truth）：")
        for t in top_unmatched:
            print(f"  {freq.get(t, 0):>3}  {t}")