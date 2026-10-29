"""语料下载。

实测网络约束（勿改成 git clone）：
    git clone github.com          -> 连接被重置
    raw.githubusercontent.com     -> 不可达
    huggingface.co                -> 不可达
    codeload.github.com           -> 200，可下 zip      ✅ 走这里
    hf-mirror.com                 -> 200，可下模型       ✅ 走这里

因此所有语料统一走 codeload 的 zip 归档接口。
"""

from __future__ import annotations

import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from loguru import logger

from cs408rag.config import RAW_DIR, ensure_dirs

CODELOAD = "https://codeload.github.com/{owner}/{repo}/zip/refs/heads/{branch}"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


@dataclass(frozen=True)
class Source:
    """一个语料来源。"""

    key: str
    owner: str
    repo: str
    branch: str
    note: str

    @property
    def url(self) -> str:
        return CODELOAD.format(owner=self.owner, repo=self.repo, branch=self.branch)

    @property
    def extract_dir(self) -> Path:
        return RAW_DIR / self.key


SOURCES: dict[str, Source] = {
    "wangdao": Source(
        key="wangdao",
        owner="Scienmicroit",
        repo="2027-CSKAOYAN-eBooks",
        branch="main",
        note="★ 主语料：2027 版王道复习指导书正文 Markdown（42 个 md，4.55MB，193 万字符）",
    ),
    "quiz": Source(
        key="quiz",
        owner="Become-ILLUSORY",
        repo="408-quiz",
        branch="main",
        note="★ 真题：720 道历年真题（2009-2026），含 tags / 官方解析 / 图表标记",
    ),
    "textbook": Source(
        key="textbook",
        owner="naplatte",
        repo="CS-Postgraduate-Entrance-Examination-408-Database",
        branch="main",
        note="XMind 思维导图（6182 知识点节点）—— 图谱来源；其 PDF 是导图扁平导出，不用作语料",
    ),
    "baseline_repo": Source(
        key="baseline_repo",
        owner="guohuiyuan",
        repo="408-RAG",
        branch="main",
        note="同期开源竞品，仅用于 README 口径对比参考",
    ),
}


def _download_zip(url: str, dest: Path, *, retries: int = 3, timeout: int = 300) -> None:
    """带重试的流式下载。"""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            logger.info(f"下载 {url}（第 {attempt}/{retries} 次）")
            with requests.get(
                url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}
            ) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length") or 0)
                written = 0
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        written += len(chunk)
                logger.info(f"完成 {dest.name}：{written / 1e6:.1f} MB" + (f" / {total / 1e6:.1f} MB" if total else ""))
                return
        except Exception as e:  # noqa: BLE001 - 网络错误种类多，统一重试
            last_err = e
            logger.warning(f"下载失败：{e}")
            if attempt < retries:
                backoff = 2**attempt
                logger.info(f"{backoff}s 后重试")
                time.sleep(backoff)
    raise RuntimeError(f"下载失败，已重试 {retries} 次：{url}") from last_err


def _extract_zip(zip_path: Path, dest: Path) -> None:
    """解压，并剥掉 GitHub 归档自带的顶层目录。"""
    staging = dest.with_suffix(".staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(staging)

    # GitHub zip 的顶层是 <repo>-<branch>/，剥掉它
    entries = list(staging.iterdir())
    inner = entries[0] if len(entries) == 1 and entries[0].is_dir() else staging

    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(inner), str(dest))
    if inner is not staging:
        shutil.rmtree(staging, ignore_errors=True)


def fetch_source(key: str, *, force: bool = False, keep_zip: bool = False) -> Path:
    """下载并解压一个语料源，返回解压后的目录。

    幂等：目录已存在则跳过，除非 force=True。
    """
    ensure_dirs()
    if key not in SOURCES:
        raise KeyError(f"未知语料源 {key!r}，可选：{list(SOURCES)}")
    src = SOURCES[key]

    if src.extract_dir.exists() and not force:
        logger.info(f"[{key}] 已存在，跳过下载 -> {src.extract_dir}")
        return src.extract_dir

    zip_path = RAW_DIR / f"{key}.zip"
    if force or not zip_path.exists():
        _download_zip(src.url, zip_path)
    else:
        logger.info(f"[{key}] 复用已下载的 {zip_path.name}")

    logger.info(f"[{key}] 解压 -> {src.extract_dir}")
    _extract_zip(zip_path, src.extract_dir)

    if not keep_zip:
        zip_path.unlink(missing_ok=True)

    _verify(key, src.extract_dir)
    return src.extract_dir


def _verify(key: str, root: Path) -> None:
    """下载后的结构自检——宁可炸在这里，也不要让脏数据流进索引。"""
    checks = {
        "quiz": ["questions.json"],
        "wangdao": [],  # 目录名含中文，下面单独处理
        "textbook": [],
        "baseline_repo": ["data/test_data/questions_400.json"],
    }

    if key == "wangdao":
        md_root = root / "Markdown"
        if not md_root.exists():
            raise RuntimeError(f"王道语料校验失败：缺少 {md_root}")
        mds = list(md_root.rglob("*.md"))
        subjects = [d for d in md_root.iterdir() if d.is_dir()]
        if len(subjects) < 4:
            raise RuntimeError(f"王道语料校验失败：只找到 {len(subjects)} 个科目目录，应为 4 个")
        # 正文文件必须存在（仅有模拟题说明下载不完整）
        body = [f for f in mds if not any(p in f.name for p in ("模拟题", "知识点总结", "高难度"))]
        if len(body) < 20:
            raise RuntimeError(f"王道语料校验失败：只找到 {len(body)} 个正文文件，应为 26 个左右")
        logger.info(f"[wangdao] 校验通过：{len(subjects)} 个科目，正文 {len(body)} 个文件，共 {len(mds)} 个 md")
        return

    if key == "textbook":
        pdfs = list(root.rglob("*.pdf"))
        xminds = list(root.rglob("*.xmind"))
        if not xminds:
            raise RuntimeError(f"思维导图校验失败：XMind={len(xminds)}，不应为 0")
        logger.info(f"[textbook] 校验通过：{len(xminds)} 个 XMind（PDF {len(pdfs)} 个仅为导图导出，不作语料）")
        return

    for rel in checks.get(key, []):
        if not (root / rel).exists():
            raise RuntimeError(f"语料源 {key} 缺少必需文件：{rel}")
    logger.info(f"[{key}] 校验通过")


def fetch_all(*, force: bool = False) -> dict[str, Path]:
    """下载全部语料源。"""
    return {k: fetch_source(k, force=force) for k in SOURCES}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="下载 408-RAG 语料")
    ap.add_argument("--source", default="all", choices=[*SOURCES, "all"])
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = ap.parse_args()

    if args.source == "all":
        paths = fetch_all(force=args.force)
        for k, p in paths.items():
            logger.info(f"{k:>14} -> {p}")
    else:
        logger.info(f"{args.source} -> {fetch_source(args.source, force=args.force)}")