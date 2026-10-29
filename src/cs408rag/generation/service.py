"""答题服务：把检索、生成、引用校验串起来。

两个入口：
    answer()    开放式问答
    answer_quiz()  选择题 —— 与评测口径对齐的主路径

关键设计：
    - 生成失败（无 API key / 中转站故障）时**不抛异常**，
      而是返回检索结果 + 明确的降级说明。检索本身是可独立交付的能力。
    - 每次回答都跑一遍引用校验与押题扫描，结果随响应一起返回，
      而不是事后抽查。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cs408rag.generation.citation import CitationReport, check_citations
from cs408rag.generation.llm import LLMClient, get_llm
from cs408rag.generation.prompt import (
    ContextBlock,
    build_context,
    parse_choice,
    qa_messages,
    quiz_messages,
)
from cs408rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline, RetrievalResult

# 生产默认配置 = S1 混合检索（稠密+稀疏+图谱，RRF），**不启用交叉编码器重排**。
#
# 为什么不启用重排 —— 这是数据决定的，不是省时间：
# 完整对照实验（开发集 365 题）显示 bge-reranker-v2-m3 会让 Recall@10 下降 7.2pp，
# 即使对**同一组候选**只做纯重排序（候选集不变），MRR 也从 0.294 降到 0.279。
# 换了输入（去掉【章节路径】头）与混合方式（α 分数融合扫描）也一样。
# 详见 README「实验数据 —— 一个与直觉相反的结论」与 reports/ablation_rerank.md。
DEFAULT_QA_CONFIG = RetrievalConfig(
    use_sparse=True,
    # 消融：图谱路对概念/计算/图表题都是负贡献（Acc@10 0.603/0.719/0.697 vs
    # 不加图谱的 0.636/0.766/0.737），还多 30ms。问答默认不跑图谱。
    use_graph=False,
    # 原生 BM25 对概念题与图表题有 +1pp / +4pp，对计算题略负。
    # 单轮问答不知道题型，按多数场景开启；Agent 会在计算题里关闭。
    use_bm25=True,
    use_rerank=False,
    # 加权 zscore 融合在 dev 上全面优于 RRF（R@10 +2.1pp、Hit@10 +2.4pp），
    # 且同一小节的多个 chunk 会挤占 top-k，所以问答默认做节去重。
    fusion="weighted",
    normalize="zscore",
    section_dedup=True,
    section_per=1,
    # 图谱兄弟扩展在「不去重」基线上 +2.4pp Hit，但与节去重的收益重叠：
    # 实测 dedup 基线 Hit 0.633，加扩展后仍是 0.633、MRR 反降 0.029。
    # 因此生产关闭，保留为可选实验开关（use_graph_expand）。
    # 保留真题解析：它是有价值的语料，且带答案，回答「这道题为什么选 B」时必需。
    exclude_explanations=False,
    # ★ 排除试题/答案类小节。
    #
    # 这个开关最初设成 False（想着「答标答类问题用得上」），实测发现它会让
    # **top-1 全是「答案与解析」「本节习题精选」**：
    #   问「平衡二叉树定义」→ 返回 7.3.5 答案与解析
    #   问「SSTF 调度序列」  → 返回 5.3.6 本节习题精选
    # 学生问概念却拿到答案页，产品上是错的。
    #
    # 这些小节逐字收录真题原文，对题面的字面匹配度天然极高，
    # 会把概念讲解章节全部挤到后面。
    #
    # 排除后：
    #   「平衡二叉树定义」→ 7.3.2 平衡二叉树的定义 ✓
    #   「IEEE754 表示」 → 2.3.1 IEEE754 标准的浮点数格式 ✓
    #   「SSTF 调度」    → 5.3.3 磁盘调度算法 > SCAN 算法 ✓
    #
    # 答案类内容仍可通过真题解析 chunk（source=explanation）获得。
    exclude_exercise=True,
    # 返回 10 条：实测 Acc@10 0.624 -> 0.668（多 2 条候选）。
    # 上下文构造会按 6000 字预算截断，实际进 prompt 约 5–6 条父节。
    rerank_top_n=10,
)


@dataclass
class AnswerResult:
    question: str
    answer: str = ""
    choice: str = ""
    ok: bool = False
    error: str = ""
    blocks: list[ContextBlock] = field(default_factory=list)
    citation: CitationReport | None = None
    retrieval: RetrievalResult | None = None
    took_ms: float = 0.0
    llm_stats: dict[str, Any] = field(default_factory=dict)
    # 本次调用的 token 与延迟。llm_stats 是客户端的累计值，
    # 批量评测时用它算成本会把前面所有请求都算进来（实测踩过）。
    llm_call: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "choice": self.choice,
            "ok": self.ok,
            "error": self.error,
            "took_ms": round(self.took_ms, 1),
            "contexts": [
                {
                    "index": b.index,
                    "citation": b.citation,
                    "source": b.source,
                    "subject": b.subject,
                    "chunk_id": b.chunk_id,
                    "text": b.text[:300],
                    # 教材插图，已解析成 /v1/image/{id} 可直接渲染
                    "images": b.images,
                }
                for b in self.blocks
            ],
            "citation": self.citation.to_dict() if self.citation else None,
            "retrieval_trace": self.retrieval.trace_summary() if self.retrieval else None,
            "llm_stats": self.llm_stats,
            "llm_call": self.llm_call,
        }


class QAService:
    def __init__(
        self,
        pipeline: RetrievalPipeline | None = None,
        llm: LLMClient | None = None,
        config: RetrievalConfig | None = None,
        *,
        valid_exam_refs: set[str] | None = None,
        parent_expand: bool = True,
    ) -> None:
        self.pipeline = pipeline or RetrievalPipeline()
        self.llm = llm or get_llm()
        self.config = config or DEFAULT_QA_CONFIG
        # 真题编号白名单，用于校验回答里的真题引用是否真实存在。
        # 传 None 则跳过该项校验。
        self.valid_exam_refs = valid_exam_refs
        # 检索命中子块后，给 LLM 的是所属小节的完整文本（Parent-Child）。
        # 只影响上下文，不改变检索指标；无 chunks_all.json 时自动退化为子块。
        self.parent_expand = parent_expand

    def _context_hits(self, retrieval: RetrievalResult, cfg: RetrievalConfig) -> list:
        hits = retrieval.top_hits(cfg.rerank_top_n)
        if self.parent_expand:
            from cs408rag.retrieval.parent import expand_to_parent

            hits = expand_to_parent(hits, max_chars_per_hit=1200)
        return hits

    # -- 开放式问答 -------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        subject: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> AnswerResult:
        t0 = time.perf_counter()
        cfg = config or self.config

        retrieval = self.pipeline.retrieve(question, subject=subject, config=cfg)
        blocks = build_context(self._context_hits(retrieval, cfg))

        res = AnswerResult(question=question, blocks=blocks, retrieval=retrieval)

        if not blocks:
            res.error = "检索不到任何相关资料"
            res.took_ms = (time.perf_counter() - t0) * 1000
            return res

        llm_resp = self.llm.chat(qa_messages(question, blocks), max_tokens=2400)
        res.ok = llm_resp.ok
        res.error = llm_resp.error
        res.answer = llm_resp.text
        res.citation = check_citations(
            res.answer, blocks, valid_exam_refs=self.valid_exam_refs
        )
        res.took_ms = (time.perf_counter() - t0) * 1000
        res.llm_stats = self.llm.stats.to_dict()
        res.llm_call = {
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
            "latency_ms": round(llm_resp.latency_ms, 1),
            "model": llm_resp.model,
        }

        if res.citation and res.citation.has_prediction:
            # 红线：命中预测性表述，替换为占位说明而不是把违规内容放出去
            logger.warning(f"押题表述被拦截：{res.citation.prediction_hits}")
            res.answer += (
                "\n\n⚠️ 本回答包含对考试的预测性表述，已被系统拦截。"
                "本系统只提供基于历年真题的**实际考查情况**，不做任何考试预测。"
            )

        return res

    # -- 选择题 -----------------------------------------------------------

    def answer_quiz(
        self,
        question: str,
        options: dict[str, str],
        *,
        subject: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> AnswerResult:
        """选择题作答。评测用这条路径。"""
        t0 = time.perf_counter()
        cfg = config or self.config

        retrieval = self.pipeline.retrieve(question, subject=subject, config=cfg)
        blocks = build_context(self._context_hits(retrieval, cfg))

        res = AnswerResult(question=question, blocks=blocks, retrieval=retrieval)

        if not blocks:
            res.error = "检索不到任何相关资料"
            res.took_ms = (time.perf_counter() - t0) * 1000
            return res

        llm_resp = self.llm.chat(
            quiz_messages(question, options, blocks), max_tokens=3000
        )
        # 少数模型会在长上下文里把预算全花在思考上；缩短上下文再试一次，
        # 空正文是评测里最大的假错误来源，不能直接当成「答错」。
        if llm_resp.ok and not llm_resp.text.strip():
            llm_resp = self.llm.chat(
                quiz_messages(question, options, blocks[:4]), max_tokens=3200
            )
        res.ok = llm_resp.ok
        res.error = (
            "模型返回空正文"
            if llm_resp.ok and not llm_resp.text.strip()
            else llm_resp.error
        )
        res.answer = llm_resp.text
        res.choice = parse_choice(llm_resp.text)
        res.citation = check_citations(
            res.answer, blocks, valid_exam_refs=self.valid_exam_refs
        )
        res.took_ms = (time.perf_counter() - t0) * 1000
        res.llm_stats = self.llm.stats.to_dict()
        res.llm_call = {
            "prompt_tokens": llm_resp.prompt_tokens,
            "completion_tokens": llm_resp.completion_tokens,
            "latency_ms": round(llm_resp.latency_ms, 1),
            "model": llm_resp.model,
        }
        return res

    # -- 流式 -------------------------------------------------------------

    def stream_answer(
        self,
        question: str,
        *,
        subject: str | None = None,
        config: RetrievalConfig | None = None,
    ) -> Iterator[dict[str, Any]]:
        """流式问答。先发检索结果，再逐段发正文。

        这样前端能立刻渲染出「找到了哪些章节」，不用等 LLM 出字。
        """
        cfg = config or self.config
        retrieval = self.pipeline.retrieve(question, subject=subject, config=cfg)
        blocks = build_context(self._context_hits(retrieval, cfg))

        yield {
            "type": "contexts",
            "data": [
                {"index": b.index, "citation": b.citation, "source": b.source}
                for b in blocks
            ],
        }
        if not blocks:
            yield {"type": "error", "data": "检索不到任何相关资料"}
            return

        buf: list[str] = []
        for piece in self.llm.stream(qa_messages(question, blocks), max_tokens=2000):
            buf.append(piece)
            yield {"type": "delta", "data": piece}

        full = "".join(buf)
        rep = check_citations(full, blocks, valid_exam_refs=self.valid_exam_refs)
        yield {"type": "citation", "data": rep.to_dict()}
        if rep.has_prediction:
            logger.warning(f"押题表述被拦截：{rep.prediction_hits}")
            yield {
                "type": "warning",
                "data": "本回答包含预测性表述，本系统只提供历年真题的实际考查情况，不做考试预测。",
            }


def build_exam_ref_whitelist() -> set[str]:
    """从真题库构造编号白名单，用于校验回答里的真题引用。

    ★ 必须用 `YYYY#NN` 格式：`citation.extract_exam_refs` 输出的就是这个格式。
    早先用 `q.qid`（`YYYY-NN`）做白名单，两边格式不一致，
    导致**所有真实引用都被判成编造** —— 实测 2009#25 等 5 个真编号全进了
    invalid_exam_refs。这类 bug 不报错、只让校验恒为 False，很难发现。
    """
    from cs408rag.ingestion.quiz_loader import load_questions

    return {f"{q.year}#{q.num:02d}" for q in load_questions()}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    svc = QAService(valid_exam_refs=build_exam_ref_whitelist())
    q = " ".join(sys.argv[1:]) or "采用组相联映射时主存地址如何划分？"
    r = svc.answer(q)
    print(f"ok={r.ok} took={r.took_ms:.0f}ms error={r.error[:100]}")
    print(f"引用块数: {len(r.blocks)}")
    for b in r.blocks[:3]:
        print(f"  [{b.index}] {b.citation[:70]}")
    if r.citation:
        print(f"引用校验: {r.citation.to_dict()}")
    print(f"\n回答:\n{r.answer[:600]}")