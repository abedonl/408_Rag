"""BGE-M3 向量化。

单次推理同时产出两路表示：
    dense_vecs      1024 维稠密向量 —— 语义召回
    lexical_weights 稀疏词权重       —— 术语/公式/缩写精确匹配

稀疏路不是可有可无的：408 题里大量出现「TLB」「CF 标志位」「ARQ」「IEEE 754」
这类术语，稠密向量对它们在长文本中的区分度并不好，而词权重能精确命中。

设计要点：
    - 懒加载 + 进程内单例：模型 2GB+，绝不能每次调用都重新加载
    - GPU 推理是同步阻塞的，FastAPI 侧要用 run_in_executor 包起来（见 api/）
    - 提供 unload()：16GB 内存的机器上，BGE-M3 与 reranker 不宜同时常驻
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger

from cs408rag.config import settings


@dataclass
class EmbeddingResult:
    dense: np.ndarray  # (n, 1024) float32
    sparse: list[dict[int, float]]  # 每条的 {token_id: weight}
    count: int


class BGEM3Embedder:
    """BGE-M3 封装。线程安全，懒加载。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int | None = None,
        max_length: int = 8192,
    ) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size or settings.embedding_batch_size
        self.max_length = max_length

        self._model: Any = None
        self._lock = threading.Lock()
        # ★ GPU 推理固定到单线程。
        # 实测：串行 encode 32ms，8 线程并发调用同一模型变成 254–285ms，
        # 因为 PyTorch CUDA context 跨线程有固定开销。API 的线程池会把请求
        # 分散到不同 worker，因此这里统一收敛到一个专用线程，并发请求排队。
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bge")

    # -- 生命周期 ---------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        # 模型必须在推理线程里初始化，否则 CUDA context 会绑定到调用线程，
        # 后续在 executor 线程推理又要重新建上下文。
        self._executor.submit(self._load_impl).result()

    def _load_impl(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            settings.export_hf_env()  # 必须在 import 前设好镜像，否则会去连不可达的 huggingface.co

            # hf-mirror 对 .DS_Store 返回 403，下载时只能忽略它；
            # 但忽略会导致快照完整性校验不过。补空占位文件自愈。
            from cs408rag.models import _fill_missing_ignored_files

            _fill_missing_ignored_files(self.model_name)

            from FlagEmbedding import BGEM3FlagModel

            device = self._resolve_device()
            logger.info(f"加载 BGE-M3：{self.model_name} -> {device}")
            # ★ 必须显式传 device。
            #   只传 use_fp16 而不传 device 的话，FlagEmbedding 会留在 CPU 上，
            #   日志显示 cuda 但实际跑在 CPU（实测 3.5 条/秒，GPU 应快一个数量级）。
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=(device == "cuda"),
                device=device,
            )
            logger.info("BGE-M3 加载完成")

    def unload(self) -> None:
        """释放模型与显存。16GB 内存下与 reranker 交替使用时调它。"""
        with self._lock:
            if self._model is None:
                return
            self._model = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("BGE-M3 已卸载")

    def _resolve_device(self) -> str:
        if self.device != "cuda":
            return self.device
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning(
                    "配置要求 cuda，但当前 torch 是 CPU 版（pip 默认在 Windows 装 CPU wheel）。"
                    "回退到 CPU——会慢很多。修法见 pyproject.toml 的 pytorch-cu124 索引。"
                )
                return "cpu"
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        except ImportError:
            return "cpu"

    # -- 推理 -------------------------------------------------------------

    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        return_sparse: bool = True,
        show_progress: bool = True,
    ) -> EmbeddingResult:
        """批量编码。返回 dense + sparse 两路。"""
        if not texts:
            return EmbeddingResult(
                dense=np.zeros((0, settings.embedding_dim), dtype=np.float32),
                sparse=[],
                count=0,
            )

        return self._executor.submit(
            self._encode_impl, texts, batch_size, return_sparse
        ).result()

    def _encode_impl(
        self, texts: list[str], batch_size: int | None, return_sparse: bool
    ) -> EmbeddingResult:
        self._load_impl()
        assert self._model is not None

        out = self._model.encode(
            texts,
            batch_size=batch_size or self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        sparse: list[dict[int, float]] = []
        if return_sparse and "lexical_weights" in out:
            for w in out["lexical_weights"]:
                # FlagEmbedding 有时给 str key，Qdrant 需要 int
                sparse.append({int(k): float(v) for k, v in w.items() if float(v) > 0})
        else:
            sparse = [{} for _ in texts]

        return EmbeddingResult(dense=dense, sparse=sparse, count=len(texts))

    def encode_query(self, query: str) -> tuple[np.ndarray, dict[int, float]]:
        """单条查询编码。查询不加任何前缀（BGE-M3 无需指令前缀）。"""
        r = self.encode([query], show_progress=False)
        return r.dense[0], (r.sparse[0] if r.sparse else {})


# ---------------------------------------------------------------------------
# 进程内单例
# ---------------------------------------------------------------------------

_embedder: BGEM3Embedder | None = None
_singleton_lock = threading.Lock()


def get_embedder() -> BGEM3Embedder:
    global _embedder
    if _embedder is None:
        with _singleton_lock:
            if _embedder is None:
                _embedder = BGEM3Embedder()
    return _embedder


def unload_embedder() -> None:
    global _embedder
    if _embedder is not None:
        _embedder.unload()


# ---------------------------------------------------------------------------
# 稀疏向量工具
# ---------------------------------------------------------------------------


def sparse_to_qdrant(w: dict[int, float]) -> tuple[list[int], list[float]]:
    """转成 Qdrant 的 SparseVector(indices, values)。"""
    if not w:
        return [], []
    idx = sorted(w)
    return idx, [w[i] for i in idx]


if __name__ == "__main__":
    e = BGEM3Embedder()
    r = e.encode(
        [
            "在 Cache 中，采用组相联映射时地址如何划分？",
            "TLB 是页表的高速缓存，用于加速虚拟地址到物理地址的转换。",
        ]
    )
    print("dense shape:", r.dense.shape)
    print("sparse 非零项数:", [len(s) for s in r.sparse])
    d, s = e.encode_query("CF 标志位什么时候被置位")
    print("query dense shape:", d.shape, "sparse 非零:", len(s))