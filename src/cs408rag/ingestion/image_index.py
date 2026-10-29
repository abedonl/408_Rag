"""教材插图索引。

需求：检索结果里带上原文的插图。但插图在 `data/raw/` 下（不进仓库），
chunk 里只存相对路径，前端拿不到。所以单独落一份索引文件：

    data/processed/image_index.json
        { image_id: {path, subject, url, ext, bytes} }

chunk 的 payload 里存 `images: [{image_id, src, caption}]`，
输出时用 image_id 查索引拿到可访问的 url。

为什么要单独建索引而不是把绝对路径写进 chunk：
    - 绝对路径不可移植（换台机器就失效）
    - 原始目录不能直接对外暴露（版权，且路径含中文）
    - 索引文件可以增量重建，chunk 不用重算
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from cs408rag.config import PROCESSED_DIR, RAW_DIR

IMAGE_INDEX_PATH = PROCESSED_DIR / "image_index.json"

# 支持的图片扩展名
IMG_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg"}


def _subject_of_dir(dir_name: str) -> str:
    mapping = {
        "数据结构": "数据结构",
        "计算机组成原理": "组成原理",
        "操作系统": "操作系统",
        "计算机网络": "计算机网络",
    }
    for k, v in mapping.items():
        if k in dir_name:
            return v
    return dir_name


def build_image_index(root: Path | None = None) -> dict[str, dict]:
    """扫描王道 Markdown 目录下的 pictures/，并合并真题解析图。"""
    root = root or (RAW_DIR / "wangdao" / "Markdown")

    index: dict[str, dict] = {}
    dup = 0

    if root.exists():
        for sub_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            subject = _subject_of_dir(sub_dir.name)
            for img_path in sub_dir.rglob("*"):
                if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
                    continue
                image_id = f"{subject}-{img_path.stem[:32]}"
                if image_id in index:
                    # 文件名是内容哈希或「序号-image-XXX」形式，加了科目前缀后几乎不会撞；
                    # 真撞了保留先出现的并计数
                    dup += 1
                    continue
                index[image_id] = {
                    "image_id": image_id,
                    # 相对项目根的路径：可移植，且不暴露绝对路径
                    "path": str(img_path.relative_to(PROCESSED_DIR.parents[0])).replace("\\", "/"),
                    "subject": subject,
                    "ext": img_path.suffix.lower().lstrip("."),
                    "bytes": img_path.stat().st_size,
                    # 对外 URL，由 API 的 /v1/image/{id} 提供
                    "url": f"/v1/image/{image_id}",
                }
    else:
        logger.warning(f"找不到王道语料 {root}，图片索引只包含真题解析图")

    # 真题解析里的内联 SVG（由 exam_figures 落盘）
    from cs408rag.ingestion.exam_figures import load_exam_figure_index

    for image_id, meta in load_exam_figure_index().items():
        index.setdefault(image_id, meta)

    if dup:
        logger.warning(f"图片 image_id 冲突 {dup} 处（已保留先出现的）")
    if not index:
        raise FileNotFoundError(
            f"找不到任何图片：{root} 不存在，真题解析图索引也为空。"
            "先跑 python -m cs408rag.ingestion.download --source wangdao"
        )
    logger.info(f"图片索引：{len(index)} 张，来自 {root} + 真题解析图")
    return index


def save_image_index(index: dict[str, dict] | None = None) -> Path:
    index = index if index is not None else build_image_index()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_INDEX_PATH.write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"图片索引 -> {IMAGE_INDEX_PATH}")
    return IMAGE_INDEX_PATH


def load_image_index() -> dict[str, dict]:
    """读回索引。不存在时返回空字典（不抛异常 —— 没有图片不该阻断检索）。"""
    if not IMAGE_INDEX_PATH.exists():
        logger.warning(f"图片索引不存在（{IMAGE_INDEX_PATH}），检索结果将不含插图")
        return {}
    return json.loads(IMAGE_INDEX_PATH.read_text(encoding="utf-8"))


_cache: dict[str, dict] | None = None


def resolve_images(images: list[dict] | None) -> list[dict]:
    """把 chunk 里的图片条目补全成可展示的形式。

    输入：[{image_id, src, caption}]
    输出：[{image_id, caption, url, exists}]，url 可空（索引里没有时）
    """
    global _cache
    if _cache is None:
        _cache = load_image_index()
    if not images:
        return []

    out: list[dict] = []
    for im in images:
        iid = im.get("image_id") or ""
        meta = _cache.get(iid)
        out.append(
            {
                "image_id": iid,
                "caption": im.get("caption") or "",
                "url": meta["url"] if meta else "",
                "subject": meta["subject"] if meta else "",
                "exists": meta is not None,
            }
        )
    return out


def image_abs_path(image_id: str) -> Path | None:
    """解析 image_id 到磁盘绝对路径。找不到返回 None。"""
    global _cache
    if _cache is None:
        _cache = load_image_index()
    meta = _cache.get(image_id)
    if not meta:
        return None
    p = Path(meta["path"])
    if not p.is_absolute():
        p = PROCESSED_DIR.parents[0] / p
    return p if p.exists() else None


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    idx = build_image_index()
    save_image_index(idx)
    from collections import Counter

    print(f"图片总数: {len(idx)}")
    print("按科目:", dict(Counter(v["subject"] for v in idx.values())))
    total = sum(v["bytes"] for v in idx.values())
    print(f"总大小: {total / 1e6:.1f} MB")
    for k, v in list(idx.items())[:3]:
        print(f"  {k} -> {v['path']} ({v['bytes'] / 1024:.0f}KB)")