"""原生 BM25 召回（jieba + Okapi）。

为什么在 BGE-M3 sparse 之外再加一路：
    BGE-M3 的 sparse 是模型学出来的 lexical weights，不是 BM25。
    实测（dev 452 题，题级命中）四路并集比三路高 +2.7pp@10 / +1.3pp@50，
    说明两者捕捉的词法信号不完全重合。

索引只包含**教材概念 chunk**：
    解析 chunk 含答案（评测防泄漏，问答应答也不该靠它做召回），
    exercise 小节逐字收录真题原文（会靠题面匹配占满 top-k）。
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from cs408rag.config import INDEX_DIR, PROCESSED_DIR

# 用 JSON 而不是 pickle：pickle 会记录类的模块路径，
# 索引若在 `python -m cs408rag.retrieval.bm25` 下保存，类模块是 `__main__`，
# 服务进程加载时会报 Can't get attribute 'BM25Index'（实测踩过）。
INDEX_PATH = INDEX_DIR / "bm25.json"
K1, B = 1.5, 0.75


def _tokenize(text: str) -> list[str]:
    import jieba

    return [t.lower() for t in jieba.lcut(text) if len(t) >= 2 or t.isalnum()]


@dataclass
class BM25Index:
    chunk_ids: list[str] = field(default_factory=list)
    tf: list[Counter] = field(default_factory=list)
    dl: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    idf: dict[str, float] = field(default_factory=dict)
    postings: dict[str, list[int]] = field(default_factory=dict)
    meta: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: list[dict]) -> BM25Index:
        idx = cls()
        for c in chunks:
            toks = _tokenize(
                (c.get("text") or "") + " " + " ".join(c.get("chapter_path") or [])
            )
            idx.chunk_ids.append(c["chunk_id"])
            idx.tf.append(Counter(toks))
            idx.dl.append(len(toks))
            idx.meta[c["chunk_id"]] = {
                "subject": c.get("subject", ""),
                "section_id": c.get("section_id", ""),
            }

        df: Counter[str] = Counter()
        for counts in idx.tf:
            df.update(set(counts))
        n = len(idx.chunk_ids)
        idx.avgdl = sum(idx.dl) / max(n, 1)
        idx.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        for i, counts in enumerate(idx.tf):
            for t in counts:
                idx.postings.setdefault(t, []).append(i)
        return idx

    def search(
        self, query: str, *, top_k: int = 20, subject: str | None = None
    ) -> list[tuple[str, float]]:
        q_toks = _tokenize(query)
        scores: dict[int, float] = defaultdict(float)
        for t in q_toks:
            if t not in self.postings:
                continue
            w = self.idf.get(t, 0.0)
            for i in self.postings[t]:
                if subject and self.meta[self.chunk_ids[i]]["subject"] != subject:
                    continue
                f = self.tf[i][t]
                scores[i] += w * f * (K1 + 1) / (
                    f + K1 * (1 - B + B * self.dl[i] / self.avgdl)
                )
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.chunk_ids[i], s) for i, s in ranked]

    def save(self, path: Path | None = None) -> Path:
        path = path or INDEX_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "chunk_ids": self.chunk_ids,
            "tf": [dict(c) for c in self.tf],
            "dl": self.dl,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "postings": self.postings,
            "meta": self.meta,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        logger.info(f"BM25 索引 -> {path}（{len(self.chunk_ids)} 文档）")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> BM25Index:
        path = path or INDEX_PATH
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            chunk_ids=d["chunk_ids"],
            tf=[Counter(x) for x in d["tf"]],
            dl=d["dl"],
            avgdl=d["avgdl"],
            idf=d["idf"],
            postings=d["postings"],
            meta=d["meta"],
        )


def build_from_chunks_file(path: Path | None = None) -> BM25Index:
    path = path or (PROCESSED_DIR / "chunks_all.json")
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        c
        for c in raw
        if c.get("source") != "explanation" and c.get("chunk_class") != "exercise"
    ]
    logger.info(f"BM25 语料：{len(chunks)} 个教材概念 chunk")
    idx = BM25Index.build(chunks)
    idx.save()
    return idx


_index: BM25Index | None = None


def get_bm25() -> BM25Index | None:
    """懒加载。索引不存在时返回 None（不让检索链路因此崩掉）。"""
    global _index
    if _index is None:
        try:
            if INDEX_PATH.exists():
                _index = BM25Index.load()
            else:
                _index = build_from_chunks_file()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"BM25 索引不可用，本路跳过：{e}")
            return None
    return _index


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    idx = build_from_chunks_file()
    for q in ("组相联映射 地址划分", "TCP 拥塞控制 慢开始", "死锁 银行家算法"):
        print(q, "->", idx.search(q, top_k=3))