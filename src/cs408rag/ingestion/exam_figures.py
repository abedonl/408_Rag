"""真题解析里的内联 SVG 图。

47 道题的官方解析里带内联 SVG（过程图、资源分配表、状态转换图）。
这些图对「看着解析理解过程」有价值，但之前只把解析纯文本入库，图丢了。

做法与教材插图一致：
    1. 从 explanation_html 提取最外层 SVG，落盘为 .svg
    2. 生成 image_id / url 元信息，合并进 image_index.json
    3. 解析 chunk 的 payload.images 挂 [{image_id, src, caption}]
    前端通过 /v1/image/{image_id} 取图。

防泄漏：held-out 年份的解析本就不入库，这里的图也一并不生成索引。
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from loguru import logger

from cs408rag.config import PROCESSED_DIR, settings
from cs408rag.figures.svg_parser import extract_outer_svgs
from cs408rag.ingestion.quiz_loader import Question

EXAM_FIG_DIR = PROCESSED_DIR / "exam_figures"
EXAM_FIG_INDEX = PROCESSED_DIR / "exam_figure_index.json"


def extract_exam_figures(
    questions: Iterable[Question],
    *,
    exclude_years: set[int] | None = None,
    out_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """提取每道题解析里的 SVG，落盘并返回 qid -> images 列表。"""
    out_dir = out_dir or EXAM_FIG_DIR
    exclude_years = (
        exclude_years if exclude_years is not None else set(settings.holdout_year_set)
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # explanation_html 不在 Question 模型里，需要回原始 JSON 取
    html_by_qid = _explanation_html_by_qid()

    result: dict[str, list[dict]] = {}
    index: dict[str, dict] = {}
    n_files = 0

    for q in questions:
        if q.year in exclude_years:
            continue
        html = html_by_qid.get(q.qid, "")
        if not html:
            continue
        svgs = extract_outer_svgs(html)
        if not svgs:
            continue

        images: list[dict] = []
        for i, svg in enumerate(svgs, 1):
            image_id = f"exam-{q.qid}-{i}"
            path = out_dir / f"{image_id}.svg"
            path.write_text(svg, encoding="utf-8")
            n_files += 1
            # 与教材图片保持一致：路径相对 `data/` 目录。
            # image_index.image_abs_path 用的基准是 PROCESSED_DIR.parents[0]，
            # 这里若用项目根，会解析成 data/data/processed/... 而导致 404。
            rel = str(path.relative_to(PROCESSED_DIR.parents[0])).replace("\\", "/")
            index[image_id] = {
                "image_id": image_id,
                "path": rel,
                "subject": q.subject,
                "ext": "svg",
                "bytes": path.stat().st_size,
                "url": f"/v1/image/{image_id}",
            }
            images.append(
                {
                    "image_id": image_id,
                    "src": "",
                    "caption": f"{q.qid} 解析配图 {i}",
                }
            )
        result[q.qid] = images

    EXAM_FIG_INDEX.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        f"真题解析图：{len(result)} 题 / {n_files} 张 SVG -> {out_dir}"
    )
    return result


def load_exam_figure_index() -> dict[str, dict]:
    if not EXAM_FIG_INDEX.exists():
        return {}
    return json.loads(EXAM_FIG_INDEX.read_text(encoding="utf-8"))


def _explanation_html_by_qid() -> dict[str, str]:
    raw_path = PROCESSED_DIR.parent / "raw" / "quiz" / "questions.json"
    if not raw_path.exists():
        logger.warning(f"找不到真题原始 JSON {raw_path}，解析图将为空")
        return {}
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    return {
        f"{int(q['year'])}-{int(q['num']):02d}": q.get("explanation_html") or ""
        for q in raw
    }


if __name__ == "__main__":
    from cs408rag.ingestion.quiz_loader import load_questions

    figs = extract_exam_figures(load_questions())
    print(f"题数 {len(figs)}，图片 {sum(len(v) for v in figs.values())}")
    for qid, imgs in list(figs.items())[:3]:
        print(qid, [i["image_id"] for i in imgs])