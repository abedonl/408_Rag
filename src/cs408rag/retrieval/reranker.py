"""交叉编码器重排。

bge-reranker-v2-m3 做 query-document 逐对打分。相比双塔召回，
它能建模 query 与 document 的交互，精排效果显著，但代价是每个候选都要过一次模型，
因此只在融合后的 top-N 上跑。

实现说明：
    直接用 transformers 的 AutoModelForSequenceClassification，
    不走 FlagEmbedding 的 FlagReranker —— 后者的类名与参数跨版本变动过，
    直接调 transformers 更稳定，也更容易看清每一步在做什么。

与 BGE-M3 的内存权衡：
    16GB 机型上二者不宜同时常驻。提供 unload()，检索线程可交替加载。
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from cs408rag.config import settings


@dataclass
class RerankResult:
    index: int  # 在输入候选列表中的下标
    score: float
    text: str


class CrossEncoderReranker:
    """bge-reranker-v2-m3 封装。线程安全，懒加载。"""

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 1024,
    ) -> None:
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size
        self.max_length = max_length

        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            settings.export_hf_env()
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            dev = self._resolve_device()
            logger.info(f"加载 reranker：{self.model_name} -> {dev}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.eval()
            self._model.to(dev)
            if dev == "cuda":
                self._model.half()
            logger.info("reranker 加载完成")

    def unload(self) -> None:
        with self._lock:
            if self._model is None:
                return
            self._model = None
            self._tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("reranker 已卸载")

    def _resolve_device(self) -> str:
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("配置要求 cuda 但 torch 为 CPU 版，reranker 回退 CPU（会很慢）")
            return "cpu"
        return self.device

    # -- 推理 -------------------------------------------------------------

    def score_pairs(self, query: str, docs: Sequence[str]) -> list[float]:
        """给 (query, doc) 逐对打分。返回与 docs 等长的分数列表。"""
        if not docs:
            return []
        self.load()
        assert self._model is not None and self._tokenizer is not None

        dev = next(self._model.parameters()).device
        scores: list[float] = []

        for start in range(0, len(docs), self.batch_size):
            batch = list(docs[start : start + self.batch_size])
            enc = self._tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(dev)

            with torch.no_grad():
                logits = self._model(**enc).logits.view(-1).float()

            scores.extend(logits.cpu().tolist())

        return scores

    def rerank(self, query: str, docs: Sequence[str], *, top_n: int | None = None) -> list[RerankResult]:
        """重排，返回按分数降序的结果。"""
        if not docs:
            return []
        scores = self.score_pairs(query, docs)
        results = [RerankResult(index=i, score=s, text=d) for i, (s, d) in enumerate(zip(scores, docs, strict=False))]
        results.sort(key=lambda r: -r.score)
        return results[:top_n] if top_n else results


# ---------------------------------------------------------------------------
# 单例
# ---------------------------------------------------------------------------

_reranker: CrossEncoderReranker | None = None
_lock = threading.Lock()


def get_reranker() -> CrossEncoderReranker:
    global _reranker
    if _reranker is None:
        with _lock:
            if _reranker is None:
                _reranker = CrossEncoderReranker()
    return _reranker


def unload_reranker() -> None:
    global _reranker
    if _reranker is not None:
        _reranker.unload()


# ---------------------------------------------------------------------------
# 可选降级：无 reranker 时的替代
# ---------------------------------------------------------------------------


def lexical_overlap_score(query: str, doc: str) -> float:
    """字符级 Jaccard 重叠。

    仅作为 reranker 不可用时的降级方案（例如模型没下载好、显存不够）。
    它远弱于交叉编码器，但比「不作任何精排」强，且零成本。
    评测时必须标注用的是哪一路，否则数字不可比。
    """
    import jieba

    q = {w for w in jieba.lcut(query) if len(w) > 1}
    d = {w for w in jieba.lcut(doc) if len(w) > 1}
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


if __name__ == "__main__":
    r = CrossEncoderReranker()
    q = "在 Cache 的组相联映射中，主存地址如何划分？"
    docs = [
        "组相联映射将 Cache 分为若干组，主存地址划分为标记、组号和块内地址三部分。",
        "页表用于虚拟地址到物理地址的转换，通常采用多级页表结构。",
        "TCP 拥塞控制包含慢启动、拥塞避免、快重传和快恢复四个阶段。",
    ]
    for x in r.rerank(q, docs):
        print(f"  {x.score:>8.4f}  idx={x.index}  {x.text[:40]}")