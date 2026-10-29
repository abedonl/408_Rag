"""王道复习指导书 Markdown 语料加载。

来源：Scienmicroit/2027-CSKAOYAN-eBooks 的 Markdown/ 目录
      （2027 版王道电子书经 MinerU OCR + AI 整理）

实测质量：标题层级完整、成段正文、LaTeX 公式保留、图片带图注。
单章最大 454KB（操作系统 第2章 进程与线程）。

只收**正文**章节。以下三类按仓库 README 自述是 AI 生成的，不是王道原文，必须排除：
    模拟题 / 高难度模拟题 / 知识点总结

解析策略：
    - 按 ## / ### 标题切分，保留完整标题层级作为 chapter_path
    - 丢弃 <div align="center"><img ...></div> 图片块（对文本检索是噪声）
    - 保留 <p align="center"><em>图 5.1 ...</em></p> 图注（携带信息）
    - 归一化全角缩进，保留 LaTeX 原样
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cs408rag.config import RAW_DIR

# 这些是按 README 自述由 AI 生成的内容，不属于王道原文
EXCLUDED_PATTERNS = ("模拟题", "知识点总结", "高难度")

SUBJECT_DIR_MAP = {
    "数据结构": "数据结构",
    "计算机组成原理": "组成原理",
    "操作系统": "操作系统",
    "计算机网络": "计算机网络",
}

# 图片块：<div align="center"> ... <img ...> ... </div>
RE_IMG_DIV = re.compile(r"<div[^>]*>\s*<img[^>]*>\s*</div>", re.I | re.S)
RE_ANY_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>", re.I)
# 图注：<p align="center"><em>图 5.1 xxx</em></p>
RE_CAPTION = re.compile(r"<p[^>]*>\s*<em>(.*?)</em>\s*</p>", re.I | re.S)
RE_HTML_TAG = re.compile(r"</?(?:p|div|span|br|em|strong)[^>]*>", re.I)
RE_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.M)

# 图片标签，取 src 与 alt
RE_IMG_TAG = re.compile(r"<img[^>]*src=\"([^\"]+)\"[^>]*?(?:alt=\"([^\"]*)\")?[^>]*>", re.I)
# Markdown 图片 ![alt](src)
RE_MD_IMG = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


@dataclass
class MdSection:
    """Markdown 里按标题切出的一节。"""

    section_id: str
    subject: str
    chapter: str
    section: str
    title: str
    path: list[str] = field(default_factory=list)
    level: int = 2
    text: str = ""
    source_file: str = ""
    # 本节包含的图片：[{src, caption, anchor, image_id}]
    images: list[dict] = field(default_factory=list)

    @property
    def chapter_path(self) -> list[str]:
        return self.path


def _strip_markup(s: str) -> str:
    """去掉标签与多余空白，用于图注/锚点文本。"""
    return re.sub(r"\s+", " ", (s or "").replace("​", "")).strip()


def _clean(text: str) -> str:
    """清洗 Markdown 正文。"""
    text = RE_IMG_DIV.sub("", text)
    # 图注转成普通文字
    text = RE_CAPTION.sub(lambda m: f"[图] {m.group(1).strip()}", text)
    text = RE_ANY_IMG.sub("", text)
    text = RE_HTML_TAG.sub("", text)
    # 全角空格缩进（王道排版习惯）归一为普通空格
    text = text.replace("　", " ")
    # 压缩多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉 Markdown 链接语法但保留锚文本（站内引用对检索无意义）
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return text.strip()


def _subject_of(dir_name: str) -> str:
    for key, val in SUBJECT_DIR_MAP.items():
        if key in dir_name:
            return val
    return dir_name


def _extract_images(raw_text: str, *, section_id: str, subject: str, base_dir: str) -> list[dict]:
    """从原始 Markdown 片段里抽图片引用。

    返回 [{image_id, src, caption, anchor}]：
        image_id 形如「数据结构-75752c0b...」，带科目前缀，保证跨科目唯一
                 （实测部分科目图片名是 2-1-image-007 这类短名，容易撞）
        src      Markdown 里的相对路径，如 ./pictures/abc.jpg
        caption  紧随其后的图注（如有），如「图 5.1 树的树形表示」
        anchor   图片前约 60 字的正文，用来定位这张图属于哪一段（进而属于哪个 chunk）

    为什么要 anchor：
        一节正文会被切成多个 chunk，图片应挂到「它出现的那一段」所在的 chunk，
        而不是无差别挂给整节的所有 chunk —— 后者会让每个 chunk 都带一堆无关插图。
    """
    out: list[dict] = []
    subj_slug = _slug(subject)[:8]

    def _iid(fname: str) -> str:
        return f"{subj_slug}-{_slug(fname.rsplit('.', 1)[0])[:32]}"

    for m in RE_IMG_TAG.finditer(raw_text):
        src = (m.group(1) or "").strip()
        if not src:
            continue
        # 跳过外链（本项目语料都是本地 ./pictures/）
        if src.startswith(("http://", "https://", "data:")):
            continue

        # 图注：在图片之后 300 字符内找 <p align="center"><em>...</em></p>
        tail = raw_text[m.end() : m.end() + 300]
        cap = RE_CAPTION.search(tail)
        caption = _strip_markup(RE_HTML_TAG.sub("", cap.group(1))) if cap else ""

        # anchor：图片之前的正文（去掉标签）
        head = raw_text[max(0, m.start() - 400) : m.start()]
        head_clean = _strip_markup(RE_IMG_TAG.sub("", RE_HTML_TAG.sub("", head)))
        anchor = head_clean[-60:] if head_clean else ""

        fname = src.rsplit("/", 1)[-1]
        out.append(
            {
                "image_id": _iid(fname),
                "src": src,
                "caption": caption,
                "anchor": anchor,
            }
        )

    # Markdown 语法 ![alt](src)
    for m in RE_MD_IMG.finditer(raw_text):
        alt, src = (m.group(1) or "").strip(), (m.group(2) or "").strip()
        if not src or src.startswith(("http://", "https://", "data:")):
            continue
        fname = src.rsplit("/", 1)[-1]
        out.append(
            {
                "image_id": _iid(fname),
                "src": src,
                "caption": _strip_markup(alt),
                "anchor": "",
            }
        )

    return out


def parse_markdown_file(path: Path, subject: str) -> list[MdSection]:
    """把一个章节 md 文件切成若干小节。

    ★ 在**原始文本**上定位标题与图片，再逐节清洗。
      不能先整体 _clean 再切：清洗会删掉图片标签，图片位置信息就丢了。
    """
    raw = path.read_text(encoding="utf-8")

    matches = list(RE_HEADING.finditer(raw))
    if not matches:
        imgs = _extract_images(raw, section_id=f"{subject}-{path.stem}", subject=subject, base_dir=str(path.parent))
        return [
            MdSection(
                section_id=f"{subject}-{path.stem}",
                subject=subject,
                chapter="",
                section="",
                title=path.stem,
                path=[path.stem],
                level=1,
                text=_clean(raw),
                source_file=path.name,
                images=imgs,
            )
        ]

    sections: list[MdSection] = []
    # 标题级 -> 当前路径。注意别叫 path，会遮蔽入参的 Path 对象
    stack: list[tuple[int, str]] = []

    # ★ section_id 必须包含**文件名**。
    #
    # 最初写成 f"{subject}-{_slug(title)}-{idx}"，idx 只是文件内的标题序号。
    # 结果不同章节文件里都有「答案与解析」「一、单项选择题」这类重复标题，
    # 序号又恰好相同，section_id 就撞了 —— 实测 243 处冲突，
    # 导致 311 个 chunk（5.3%）在写入 Qdrant 时被同 ID 覆盖，静默丢数据。
    # 这种 bug 不报错，只会让索引悄悄少一块，非常难发现。
    file_key = _slug(path.stem)[:24]

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw)
        raw_text = raw[start:end]
        text = _clean(raw_text)

        # 维护标题栈
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        title_path = [t for _, t in stack]

        section_id = f"{subject}-{file_key}-{idx}"
        imgs = _extract_images(raw_text, section_id=section_id, subject=subject, base_dir=str(path.parent))

        # 一级标题（书名）本身没有正文时跳过
        if not text and level <= 1:
            continue

        sections.append(
            MdSection(
                section_id=section_id,
                subject=subject,
                chapter=title_path[1] if len(title_path) > 1 else "",
                section=title,
                title=title,
                path=list(title_path),
                level=level,
                text=text,
                source_file=path.name,
                images=imgs,
            )
        )
    return sections


_SLUG_BAD = re.compile(r"[\s/\\:*?\"<>|]+")


def _slug(s: str) -> str:
    return _SLUG_BAD.sub("_", s)[:48]


def find_markdown_dirs(root: Path | None = None) -> list[Path]:
    root = root or (RAW_DIR / "wangdao")
    md_root = root / "Markdown"
    if not md_root.exists():
        raise FileNotFoundError(
            f"找不到王道 Markdown 语料 {md_root}。"
            "先跑：python -m cs408rag.ingestion.download --source wangdao"
        )
    return sorted(d for d in md_root.iterdir() if d.is_dir())


def load_wangdao(root: Path | None = None) -> list[MdSection]:
    """载入全部王道正文小节。"""
    all_sections: list[MdSection] = []

    for sub in find_markdown_dirs(root):
        subject = _subject_of(sub.name)
        files = sorted(sub.glob("*.md"))
        kept, skipped = [], []
        for f in files:
            if any(p in f.name for p in EXCLUDED_PATTERNS):
                skipped.append(f.name)
                continue
            kept.append(f)

        count = 0
        for f in kept:
            secs = parse_markdown_file(f, subject)
            all_sections.extend(secs)
            count += len(secs)

        logger.info(
            f"[{subject}] 正文 {len(kept)} 个文件 → {count} 节"
            + (f"；排除 AI 生成内容 {len(skipped)} 个" if skipped else "")
        )

    total_chars = sum(len(s.text) for s in all_sections)
    logger.info(f"王道语料合计：{len(all_sections)} 节，{total_chars / 1e6:.2f} 百万字符")
    return all_sections


def corpus_stats(sections: list[MdSection]) -> dict:
    from collections import Counter

    lens = [len(s.text) for s in sections] or [0]
    return {
        "sections": len(sections),
        "total_chars": sum(lens),
        "by_subject": dict(Counter(s.subject for s in sections)),
        "char_min": min(lens),
        "char_max": max(lens),
        "char_avg": round(sum(lens) / len(lens), 1),
        "sections_over_2k": sum(1 for n in lens if n > 2000),
    }


if __name__ == "__main__":
    import json
    from collections import Counter

    from cs408rag.config import ensure_dirs

    ensure_dirs()
    secs = load_wangdao()
    st = corpus_stats(secs)
    print(json.dumps(st, ensure_ascii=False, indent=2))
    print("\n各科前 3 节的标题路径：")
    seen: Counter = Counter()
    for s in secs:
        if seen[s.subject] < 3:
            seen[s.subject] += 1
            print(f"  [{s.subject}] {s.title}   << {' > '.join(s.path[:-1])[:70]}")
            print(f"      正文 {len(s.text)} 字符")