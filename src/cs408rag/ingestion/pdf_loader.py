"""教材 PDF 解析。

408 教材的难点：大量公式（Cache 地址划分、TCP 拥塞窗口）、图表（二叉树、
状态机、流水线时空图）。PyMuPDF 对公式与图的提取能力有限，因此：

    提取失败的页必须被记录并统计，不得静默丢弃。

分级判据：
    ok       —— 有足量文本
    sparse   —— 文本过少，多半是整页图/公式
    empty    —— 完全无文本，多半是扫描页
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

try:  # PyMuPDF >= 1.24 推荐 `import pymupdf`；旧版只有 `fitz`
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore[no-redef]

from loguru import logger

from cs408rag.config import PROCESSED_DIR, RAW_DIR

# 「第1章」「第一章」「1.2.3 」这类标题
RE_CHAPTER = re.compile(r"^\s*第\s*[0-9一二三四五六七八九十]+\s*章[\s:：、.．]*(.{0,40})")
RE_SECTION = re.compile(r"^\s*(\d+(?:\.\d+){1,3})\s*[、.．:：]?\s*(.{0,50})")
RE_TOO_SHORT = re.compile(r"^\s*$")

# 判定阈值（按字符数）
MIN_CHARS_OK = 120
MIN_CHARS_SPARSE = 20


@dataclass
class PageContent:
    page_no: int  # 1-based
    text: str
    char_count: int
    status: str  # ok | sparse | empty
    image_count: int


@dataclass
class TextbookSection:
    """按标题切出来的一节。"""

    section_id: str
    subject: str
    chapter: str
    section: str
    title: str
    path: list[str]
    page_start: int
    page_end: int
    text: str

    @property
    def chapter_path(self) -> list[str]:
        return self.path


@dataclass
class TextbookDoc:
    subject: str
    source_file: str
    total_pages: int
    pages: list[PageContent] = field(default_factory=list)
    sections: list[TextbookSection] = field(default_factory=list)

    @property
    def failed_pages(self) -> list[PageContent]:
        return [p for p in self.pages if p.status != "ok"]

    @property
    def extraction_ok_ratio(self) -> float:
        return (len(self.pages) - len(self.failed_pages)) / len(self.pages) if self.pages else 0.0


SUBJECT_BY_FILENAME = {
    "数据结构": "数据结构",
    "计算机组成原理": "组成原理",
    "组成原理": "组成原理",
    "操作系统": "操作系统",
    "计算机网络": "计算机网络",
}


def _guess_subject(path: Path) -> str:
    for k, v in SUBJECT_BY_FILENAME.items():
        if k in path.stem:
            return v
    return path.stem


def _classify(text: str) -> str:
    n = len(text.strip())
    if n >= MIN_CHARS_OK:
        return "ok"
    if n >= MIN_CHARS_SPARSE:
        return "sparse"
    return "empty"


def _heading_of(text: str) -> tuple[str, str, str] | None:
    """判断一段短文本是否为标题，返回 (level, number, title)。

    level: 'chapter' | 'section'
    """
    first_line = next((ln for ln in text.splitlines() if not RE_TOO_SHORT.match(ln)), "")
    if not first_line or len(first_line) > 60:
        return None

    m = RE_CHAPTER.match(first_line)
    if m:
        return ("chapter", first_line.strip()[:20], m.group(1).strip() or first_line.strip())

    m = RE_SECTION.match(first_line)
    if m:
        return ("section", m.group(1), m.group(2).strip() or first_line.strip())

    return None


def parse_pdf(path: Path, subject: str | None = None, max_pages: int | None = None) -> TextbookDoc:
    """解析教材 PDF，保留章节层级。"""
    subject = subject or _guess_subject(path)
    doc = fitz.open(path)

    pages: list[PageContent] = []
    # (page_no, level, number, title, text)
    blocks: list[tuple[int, str, str, str, str]] = []

    limit = min(len(doc), max_pages) if max_pages else len(doc)
    for i in range(limit):
        page = doc.load_page(i)
        text = page.get_text("text")
        status = _classify(text)
        pages.append(
            PageContent(
                page_no=i + 1,
                text=text,
                char_count=len(text.strip()),
                status=status,
                image_count=len(page.get_images(full=True)),
            )
        )
        if status != "empty":
            h = _heading_of(text)
            if h:
                level, number, title = h
                blocks.append((i + 1, level, number, title, text))

    total = len(doc)
    doc.close()

    sections = _build_sections(blocks, pages, subject, limit, path.name)

    td = TextbookDoc(
        subject=subject,
        source_file=path.name,
        total_pages=total,
        pages=pages,
        sections=sections,
    )
    logger.info(
        f"[{subject}] {path.name}: {limit}/{total} 页，切出 {len(sections)} 节，"
        f"提取成功率 {td.extraction_ok_ratio:.1%}（失败 {len(td.failed_pages)} 页）"
    )
    return td


def _build_sections(
    blocks: list[tuple[int, str, str, str, str]],
    pages: list[PageContent],
    subject: str,
    page_limit: int,
    source_file: str,
) -> list[TextbookSection]:
    """把标题块之间的正文归到对应小节。"""
    if not blocks:
        # 没有识别出标题：退化为按页分块，至少保证语料可用
        logger.warning(f"[{subject}] 未识别出任何标题，退化为按页分块")
        out = []
        for p in pages:
            if p.status == "empty":
                continue
            out.append(
                TextbookSection(
                    section_id=f"{subject}-p{p.page_no}",
                    subject=subject,
                    chapter="",
                    section="",
                    title=f"第 {p.page_no} 页",
                    path=[],
                    page_start=p.page_no,
                    page_end=p.page_no,
                    text=p.text,
                )
            )
        return out

    blocks = sorted(blocks, key=lambda b: b[0])
    out: list[TextbookSection] = []
    chapter = ""
    section = ""
    path: list[str] = []

    for idx, (page_no, level, number, title, text) in enumerate(blocks):
        next_page = blocks[idx + 1][0] if idx + 1 < len(blocks) else page_limit + 1
        if level == "chapter":
            chapter = title
            section = ""
            path = [title]
        else:
            section = f"{number} {title}".strip()
            path = [chapter] if chapter else []
            path.append(section)

        out.append(
            TextbookSection(
                section_id=f"{subject}-{number}".replace(" ", ""),
                subject=subject,
                chapter=chapter,
                section=section,
                title=title,
                path=list(path),
                page_start=page_no,
                page_end=next_page,
                text=text,
            )
        )

    return out


def find_pdf_files(raw_dir: Path | None = None) -> list[Path]:
    root = raw_dir or (RAW_DIR / "textbook")
    if not root.exists():
        raise FileNotFoundError(
            f"找不到教材目录 {root}。先跑：python -m cs408rag.ingestion.download --source textbook"
        )
    return sorted(root.rglob("*.pdf"))


def load_all_textbooks(raw_dir: Path | None = None) -> list[TextbookDoc]:
    return [parse_pdf(p) for p in find_pdf_files(raw_dir)]


def export_extraction_report(docs: list[TextbookDoc] | None = None) -> Path:
    """导出解析质量报告——失败页要如实暴露，不能假装全绿。"""
    if docs is None:
        docs = load_all_textbooks()

    report = {
        "totals": {
            "subjects": len(docs),
            "pages": sum(d.total_pages for d in docs),
            "sections": sum(len(d.sections) for d in docs),
        },
        "per_subject": [
            {
                "subject": d.subject,
                "source_file": d.source_file,
                "total_pages": d.total_pages,
                "sections": len(d.sections),
                "ok_ratio": round(d.extraction_ok_ratio, 4),
                "sparse_pages": [p.page_no for p in d.pages if p.status == "sparse"],
                "empty_pages": [p.page_no for p in d.pages if p.status == "empty"],
                "avg_images_per_page": round(
                    sum(p.image_count for p in d.pages) / len(d.pages), 2
                )
                if d.pages
                else 0,
            }
            for d in docs
        ],
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p = PROCESSED_DIR / "pdf_extraction_report.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"解析质量报告 -> {p}")
    return p


def export_sections(docs: list[TextbookDoc] | None = None) -> Path:
    if docs is None:
        docs = load_all_textbooks()

    rows = [
        {
            "section_id": s.section_id,
            "subject": s.subject,
            "chapter": s.chapter,
            "section": s.section,
            "title": s.title,
            "path": s.path,
            "page_start": s.page_start,
            "page_end": s.page_end,
            "char_count": len(s.text),
            "text": s.text,
        }
        for d in docs
        for s in d.sections
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    p = PROCESSED_DIR / "textbook_sections.json"
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"教材小节 -> {p}（{len(rows)} 节）")
    return p


if __name__ == "__main__":
    from cs408rag.config import ensure_dirs

    ensure_dirs()
    docs = load_all_textbooks()
    print(f"\n共解析 {len(docs)} 本教材\n")
    for d in docs:
        status = "OK " if d.extraction_ok_ratio > 0.9 else "!! "
        print(
            f"  {status}{d.subject:<10} {d.total_pages:>4} 页  "
            f"{len(d.sections):>4} 节  提取成功率 {d.extraction_ok_ratio:>6.1%}  "
            f"失败 {len(d.failed_pages):>3} 页"
        )
    export_extraction_report(docs)
    export_sections(docs)