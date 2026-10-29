"""本地模型下载。

本机实测的 HF 相关障碍（都在 config.export_hf_env 里处理）：
    1. huggingface.co 不可达        -> 走 hf-mirror.com
    2. hf-mirror 对 .DS_Store 403   -> 下载时按模式忽略
    3. Xet 后端 (xethub) 401        -> 禁用 Xet，走普通 HTTP

所以不要直接调 BGEM3FlagModel(model_name) 让它自己下 ——
那样会在 .DS_Store 上 403 或走 Xet 报 401。先用本模块把权重拉到本地缓存。
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from loguru import logger

from cs408rag.config import HF_IGNORE_PATTERNS, settings


def _fill_missing_ignored_files(repo_id: str, *, max_rounds: int = 3) -> list[str]:
    """补齐因 ignore_patterns 被跳过、但快照完整性校验仍要检查的文件。

    为什么要做这件事：
        hf-mirror 对模型仓库里的 .DS_Store 返回 403，所以下载时必须忽略它。
        但 huggingface_hub 之后做快照完整性校验时，仍会发现「少了一个文件」，
        于是报 IncompleteSnapshotError；离线模式更是直接拒绝加载。

    实现方式：不靠猜文件名，而是**从 IncompleteSnapshotError 的报错里解析**
    实际缺失的路径再补空占位。这样对任何模型、任何缺失文件都通用。
    被忽略的本来就是 README 配图与 macOS 垃圾文件，对推理毫无影响。
    """
    settings.export_hf_env()
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    filled: list[str] = []
    for _ in range(max_rounds):
        try:
            path = snapshot_download(repo_id, local_files_only=True)
            # 校验通过
            if filled:
                logger.info(f"补齐占位文件：{filled}")
            return filled
        except IncompleteSnapshotError as e:
            missing = _parse_missing(e)
            if not missing:
                raise
            root = _snapshot_root(repo_id)
            if root is None:
                raise
            for rel in missing:
                target = root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"")
                filled.append(rel)
            logger.debug(f"补齐 {missing}，重试校验")
        except Exception:
            # 网络类错误等：交给调用方处理
            raise
    if filled:
        logger.info(f"补齐占位文件：{filled}")
    return filled


def _parse_missing(exc: Exception) -> list[str]:
    """从报错文本里解析缺失文件列表。"""
    m = re.search(r"missing\s*\(([^)]*)\)", str(exc))
    if not m:
        return []
    return [p.strip() for p in m.group(1).split(",") if p.strip()]


def _snapshot_root(repo_id: str) -> Path | None:
    from huggingface_hub.constants import HF_HUB_CACHE

    base = Path(HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not base.exists():
        return None
    snaps = sorted((p for p in base.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    return snaps[-1] if snaps else None


def download_model(repo_id: str, *, revision: str | None = None) -> Path:
    """把模型快照拉到 HF 缓存，返回本地路径。"""
    settings.export_hf_env()
    from huggingface_hub import snapshot_download

    logger.info(f"下载模型 {repo_id}（镜像 {settings.hf_endpoint}）")
    path = snapshot_download(
        repo_id,
        revision=revision,
        ignore_patterns=HF_IGNORE_PATTERNS,
        max_workers=4,
    )
    _fill_missing_ignored_files(repo_id)

    total = sum(
        os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs
    )
    logger.info(f"完成 {repo_id} -> {path}（{total / 1e9:.2f} GB）")
    return Path(path)


def download_all() -> dict[str, Path]:
    """下载本项目需要的全部模型。"""
    return {
        "embedding": download_model(settings.embedding_model),
        "reranker": download_model(settings.reranker_model),
    }


def is_cached(repo_id: str) -> bool:
    """检查模型是否已在本地缓存（不联网）。"""
    settings.export_hf_env()
    from huggingface_hub import try_to_load_from_cache

    for fname in ("config.json", "model.safetensors", "pytorch_model.bin"):
        r = try_to_load_from_cache(repo_id, fname)
        if isinstance(r, str) and Path(r).exists():
            return True
    return False


def ensure_models() -> None:
    """确保两个模型都在本地，缺哪个下哪个。"""
    for name, repo in (
        ("embedding", settings.embedding_model),
        ("reranker", settings.reranker_model),
    ):
        if is_cached(repo):
            logger.info(f"[{name}] {repo} 已在本地缓存")
        else:
            logger.info(f"[{name}] {repo} 不在缓存，开始下载")
            download_model(repo)


if __name__ == "__main__":
    import sys

    from cs408rag.config import ensure_dirs

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ensure_dirs()

    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for k, p in download_all().items():
            print(f"{k:>10} -> {p}")
    elif which == "embedding":
        print(download_model(settings.embedding_model))
    elif which == "reranker":
        print(download_model(settings.reranker_model))
    else:
        raise SystemExit(f"未知参数 {which!r}，可选：all | embedding | reranker")