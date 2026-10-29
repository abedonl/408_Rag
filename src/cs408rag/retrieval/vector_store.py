"""Qdrant 向量库封装。

单个 collection，两个 named vector：
    dense   1024 维 COSINE  语义召回
    sparse  SparseVector    术语/公式/缩写精确匹配（点积）

payload 里放 subject / source / chapter_path / year 等元数据，
供检索时的过滤与生成层的引用溯源使用。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger
from qdrant_client import QdrantClient, models

from cs408rag.config import settings
from cs408rag.ingestion.chunker import Chunk
from cs408rag.retrieval.embedder import sparse_to_qdrant

DENSE_VEC = "dense"
SPARSE_VEC = "sparse"


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]

    @classmethod
    def from_point(cls, point: Any, score: float = 0.0) -> SearchHit:
        """从 Qdrant 返回的 point 构造。

        ★ chunk_id 必须取自 payload，不能用 point.id。

        Qdrant 的 point id 是 uint64（我们把字符串 chunk_id 哈希成了整数），
        如果直接拿它当 chunk_id，所有跨模块的 id 比对都会失败 ——
        图谱路回查拿不到正文、评测的 ground truth 一条也对不上，
        表现为「召回率恒为 0」而没有任何报错。这里踩过，特此记下。
        """
        payload = point.payload or {}
        cid = payload.get("chunk_id") or str(point.id)
        return cls(chunk_id=cid, score=score, payload=payload)

    @property
    def text(self) -> str:
        return self.payload.get("text", "")

    @property
    def subject(self) -> str:
        return self.payload.get("subject", "")

    @property
    def chapter_path(self) -> list[str]:
        return self.payload.get("chapter_path") or []

    @property
    def source(self) -> str:
        return self.payload.get("source", "")

    @property
    def citation(self) -> str:
        """给生成层用的出处标签。"""
        if self.source == "explanation":
            y = self.payload.get("year")
            return f"真题解析 {y}年#{self.payload.get('section_id', '')}"
        p = " > ".join(self.chapter_path)
        return f"{self.subject} {p}".strip()


class VectorStore:
    def __init__(
        self,
        url: str | None = None,
        collection: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.url = url or settings.qdrant_url
        self.collection = collection or settings.qdrant_collection
        self._client = QdrantClient(url=self.url, api_key=(api_key or settings.qdrant_api_key) or None)

    # -- 客户端 -----------------------------------------------------------

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception as e:  # noqa: BLE001
            logger.error(f"Qdrant 连接失败（{self.url}）：{e}")
            return False

    # -- collection 管理 --------------------------------------------------

    def recreate(self, *, dim: int | None = None) -> None:
        """重建 collection。会丢掉已有索引，调用方需明确意图。"""
        dim = dim or settings.embedding_dim
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VEC: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VEC: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                ),
            },
        )
        self._ensure_payload_indexes()
        logger.info(f"重建 collection {self.collection}（dense {dim}d + sparse）")

    def ensure(self, *, dim: int | None = None) -> bool:
        """不存在则创建。返回是否新建。"""
        if self._client.collection_exists(self.collection):
            return False
        self.recreate(dim=dim)
        return True

    def _ensure_payload_indexes(self) -> None:
        """给过滤字段建索引——没索引的过滤会很慢。"""
        for field, schema in (
            ("subject", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("section_id", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
            ("chunk_class", models.PayloadSchemaType.KEYWORD),
        ):
            try:
                self._client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as e:  # noqa: BLE001 - 已存在时报错可忽略
                logger.debug(f"payload index {field} 未创建：{e}")

    def info(self) -> dict[str, Any]:
        c = self._client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "points": c.points_count,
            "status": str(c.status),
        }

    def count(self) -> int:
        return self._client.count(self.collection, exact=True).count

    # -- 写入 -------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: Sequence[Chunk],
        dense: np.ndarray,
        sparse: list[dict[int, float]],
        *,
        batch_size: int = 256,
    ) -> int:
        """批量写入。向量与 chunk 一一对应。"""
        if len(chunks) != len(dense) or len(chunks) != len(sparse):
            raise ValueError(
                f"数量不匹配：chunks={len(chunks)} dense={len(dense)} sparse={len(sparse)}"
            )

        total = 0
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            points: list[models.PointStruct] = []

            for i in range(start, end):
                ch = chunks[i]
                sp_idx, sp_val = sparse_to_qdrant(sparse[i])
                vector: dict[str, Any] = {DENSE_VEC: dense[i].tolist()}
                if sp_idx:
                    vector[SPARSE_VEC] = models.SparseVector(indices=sp_idx, values=sp_val)
                points.append(
                    models.PointStruct(
                        id=_point_id(ch.chunk_id),
                        vector=vector,
                        payload=ch.to_payload(),
                    )
                )

            self._client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(points)

        logger.info(f"写入 {total} 个 point 到 {self.collection}")
        return total

    # -- 检索 -------------------------------------------------------------

    def search_dense(
        self,
        vector: np.ndarray,
        *,
        top_k: int = 20,
        query_filter: models.Filter | None = None,
        with_payload: bool = True,
    ) -> list[SearchHit]:
        res = self._client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            using=DENSE_VEC,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
        ).points
        return [SearchHit.from_point(p, float(p.score)) for p in res]

    def search_sparse(
        self,
        weights: dict[int, float],
        *,
        top_k: int = 20,
        query_filter: models.Filter | None = None,
        with_payload: bool = True,
    ) -> list[SearchHit]:
        idx, val = sparse_to_qdrant(weights)
        if not idx:
            return []
        res = self._client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(indices=idx, values=val),
            using=SPARSE_VEC,
            limit=top_k,
            query_filter=query_filter,
            with_payload=with_payload,
        ).points
        return [SearchHit.from_point(p, float(p.score)) for p in res]

    def get_by_ids(self, chunk_ids: Sequence[str], *, with_payload: bool = True) -> list[SearchHit]:
        res = self._client.retrieve(
            collection_name=self.collection,
            ids=[_point_id(c) for c in chunk_ids],
            with_payload=with_payload,
        )
        return [SearchHit.from_point(p, 0.0) for p in res]


def _point_id(chunk_id: str) -> int:
    """Qdrant 只接受 uint64 或 UUID 作 id，这里把 16 位 hex 稳定映射成整数。

    注意：这个映射是**单向**的，反推不回原字符串。
    所以取回结果时 chunk_id 一律从 payload 读（见 SearchHit.from_point），
    绝不能拿 point.id 当 chunk_id —— 那样所有跨模块的 id 比对都会失败。
    """
    if chunk_id.isdigit():
        return int(chunk_id)
    import hashlib

    return int.from_bytes(hashlib.blake2b(chunk_id.encode(), digest_size=8).digest(), "big")


# ---------------------------------------------------------------------------
# 过滤器构造
# ---------------------------------------------------------------------------


def build_filter(
    *,
    subject: str | None = None,
    source: str | None = None,
    exclude_sources: Sequence[str] | None = None,
    exclude_chunk_classes: Sequence[str] | None = None,
    exclude_years: Sequence[int] | None = None,
) -> models.Filter | None:
    """构造 Qdrant 过滤条件。

    exclude_years 是防泄漏的第二道闸：即便有 held-out 解析意外入库，
    检索时也会被挡掉。第一道闸在 chunker.assert_no_holdout_leak。

    exclude_sources 用于**检索评测时排除真题解析 chunk**。
    原因：解析 chunk 的正文里逐字包含题干（构造时是 f"{题干}\\n解析：{解析}"），
    对 dev 集的题，检索器会直接命中「这道题自己的解析」，
    把真正的教材章节挤出 top-10 —— 这是自匹配污染，会同时
    虚高一部分指标、压低另一部分。评测教材检索能力时必须排除它。
    """
    must: list[models.Condition] = []
    must_not: list[models.Condition] = []

    if subject:
        must.append(models.FieldCondition(key="subject", match=models.MatchValue(value=subject)))
    if source:
        must.append(models.FieldCondition(key="source", match=models.MatchValue(value=source)))
    if exclude_sources:
        must_not.append(
            models.FieldCondition(key="source", match=models.MatchAny(any=list(exclude_sources)))
        )
    # 排除试题/答案类 chunk（它们含真题原文，会靠字面匹配占满 top-k）
    if exclude_chunk_classes:
        must_not.append(
            models.FieldCondition(
                key="chunk_class", match=models.MatchAny(any=list(exclude_chunk_classes))
            )
        )
    if exclude_years:
        must_not.append(
            models.FieldCondition(key="year", match=models.MatchAny(any=list(exclude_years)))
        )

    if not must and not must_not:
        return None
    return models.Filter(must=must or None, must_not=must_not or None)


if __name__ == "__main__":
    vs = VectorStore()
    print("ping:", vs.ping())
    try:
        print(vs.info())
    except Exception as e:  # noqa: BLE001
        print("collection 尚未创建：", e)