"""Agent 的八个工具。

设计约束：

1. **工具不替代检索链路，而是编排它。** `search_knowledge_base` 就是一期
   三路 RRF 检索，一期的消融结论因此仍然成立，Agent 的增量可以单独量化。

2. **每个工具都返回可溯源的数据。** 考点统计带 qid，真题原题带编号与年份，
   共现关系带证据题号。LLM 只能归纳这些真实数据，不能凭空发挥。

3. **失败不抛异常。** 任何一个工具挂掉（Neo4j 没起、没有 LLM key），
   Agent 降级为「其余工具可用」并在 trace 中记录，而不是整条链路崩。
   但失败会被计数 —— 静默的空结果会被误读成「这个工具没用」。

4. **评测隔离。** `eval_mode=True` 时：
   - `get_kp_questions` 排除当前题，且不返回答案与解析（否则等于给答案）
   - 分析器应已用 `year_allowed` 排除 held-out 年份
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from cs408rag.agent.calculator import evaluate
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer
from cs408rag.figures.read_figure import read_figure
from cs408rag.generation.citation import (
    check_citations,
    detect_predictions,
    extract_exam_refs,
)
from cs408rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: str = ""
    took_ms: float = 0.0
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "took_ms": round(self.took_ms, 1),
            "summary": self.summary,
        }


@dataclass
class ToolTrace:
    calls: list[ToolResult] = field(default_factory=list)

    def add(self, r: ToolResult) -> ToolResult:
        self.calls.append(r)
        return r

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    @property
    def failures(self) -> int:
        return sum(1 for c in self.calls if not c.ok)

    def by_name(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.tool] = out.get(c.tool, 0) + 1
        return out

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


# ---------------------------------------------------------------------------
# 默认检索配置：一期实测最优（三路 RRF，不重排）
# ---------------------------------------------------------------------------


AGENT_RETRIEVAL_CONFIG = RetrievalConfig(
    use_sparse=True,
    # 图谱路对通用召回负贡献，Agent 只在需要层级/关联时显式调用图谱工具
    use_graph=False,
    use_bm25=True,
    use_rerank=False,
    # 与问答服务一致：zscore 融合 + 小节去重，让上下文覆盖更多不同小节
    fusion="weighted",
    normalize="zscore",
    section_dedup=True,
    section_per=1,
    # 图谱扩展与节去重收益重叠，生产关闭（见 service.DEFAULT_QA_CONFIG 注释）
    exclude_explanations=False,
    exclude_exercise=True,
    rerank_top_n=8,
)


class AgentTools:
    """工具集。依赖按需注入，便于单测与离线运行。"""

    def __init__(
        self,
        *,
        pipeline: RetrievalPipeline | None = None,
        analyzer: ExamAnalyzer | None = None,
        graph: Any = None,
        eval_mode: bool = False,
        config: RetrievalConfig | None = None,
        valid_exam_refs: set[str] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._analyzer = analyzer
        self._graph = graph
        self.eval_mode = eval_mode
        self.config = config or AGENT_RETRIEVAL_CONFIG
        self.valid_exam_refs = valid_exam_refs
        self.current_qid: str | None = None
        self.trace = ToolTrace()
        self.failures: dict[str, int] = {}

    # -- 懒加载 -----------------------------------------------------------

    @property
    def pipeline(self) -> RetrievalPipeline:
        if self._pipeline is None:
            self._pipeline = RetrievalPipeline()
        return self._pipeline

    @property
    def analyzer(self) -> ExamAnalyzer:
        if self._analyzer is None:
            self._analyzer = get_analyzer()
        return self._analyzer

    @property
    def graph(self) -> Any:
        if self._graph is None:
            from cs408rag.retrieval.graph_store import GraphStore

            self._graph = GraphStore()
        return self._graph

    def set_current(self, qid: str | None) -> None:
        """标记当前正在作答的题，评测模式下用于排除自身。"""
        self.current_qid = qid

    def reset(self, *, current_qid: str | None = None) -> None:
        """开始新的一次 Agent 运行前清空 trace。

        trace 是实例级状态。实测踩过：连续跑三题不做清理，
        第三题的 tool_calls 里会混进前两题的调用，成本统计直接失真。
        """
        self.trace = ToolTrace()
        self.failures = {}
        self.current_qid = current_qid

    # -- 统一调用入口 -----------------------------------------------------

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            return self.trace.add(
                ToolResult(tool=name, ok=False, error=f"未知工具 {name!r}")
            )
        t0 = time.perf_counter()
        try:
            data, summary = fn(self, **kwargs)
            res = ToolResult(
                tool=name,
                ok=True,
                data=data,
                summary=summary,
                took_ms=(time.perf_counter() - t0) * 1000,
            )
        except Exception as e:  # noqa: BLE001 - 工具失败降级，不阻断 Agent
            self.failures[name] = self.failures.get(name, 0) + 1
            logger.warning(f"工具 {name} 失败：{e}")
            res = ToolResult(
                tool=name,
                ok=False,
                error=str(e)[:300],
                took_ms=(time.perf_counter() - t0) * 1000,
            )
        return self.trace.add(res)


# ---------------------------------------------------------------------------
# 工具实现。每个返回 (data, 一句话摘要)
# ---------------------------------------------------------------------------


def _search_knowledge_base(
    self: AgentTools,
    query: str,
    subject: str | None = None,
    top_k: int = 8,
) -> tuple[Any, str]:
    cfg = RetrievalConfig(**{**self.config.__dict__, "rerank_top_n": top_k})
    r = self.pipeline.retrieve(query, subject=subject, config=cfg)
    hits = [
        {
            "chunk_id": h.chunk_id,
            "citation": h.citation,
            "subject": h.subject,
            "source": h.source,
            "score": round(h.score, 5) if h.score is not None else None,
            "text": h.text[:500],
        }
        for h in r.top_hits(top_k)
    ]
    data = {
        "query": query,
        "n_hits": len(hits),
        "hits": hits,
        "routes": {k: len(v.top_ids) for k, v in r.routes.items()},
        "took_ms": round(r.took_ms, 1),
    }
    summary = f"检索返回 {len(hits)} 条"
    if hits:
        summary += f"，首条：{hits[0]['citation'][:50]}"
    return data, summary


def _search_knowledge_graph(self: AgentTools, kp_name: str) -> tuple[Any, str]:
    prof = self.graph.kp_profile(kp_name)
    if not prof:
        return {"kp_name": kp_name, "found": False}, f"图谱中未找到「{kp_name}」"
    data = {
        "kp_name": kp_name,
        "found": True,
        "title": prof.get("title"),
        "subject": prof.get("subject"),
        "path": prof.get("path"),
        "ancestors": (prof.get("ancestors") or [])[:10],
        "siblings": (prof.get("siblings") or [])[:12],
        "tags": prof.get("tags") or [],
        "cooccurring_tags": (prof.get("cooccurring") or [])[:10],
    }
    return data, f"图谱定位到「{data['title']}」，同节 {len(data['siblings'])} 个知识点"


def _read_figure_tool(self: AgentTools, question_id: str, question: str = "") -> tuple[Any, str]:
    r = read_figure(question_id, question=question)
    return r.to_dict(), f"{question_id} 图解析 {r.tier}，可用={r.available}"


def _calculate(self: AgentTools, expr: str) -> tuple[Any, str]:
    r = evaluate(expr)
    if not r["ok"]:
        return r, f"计算失败：{r['error']}"
    return r, f"{r['expr']} = {r['display']}"


def _get_exam_stats(self: AgentTools, kp_name: str) -> tuple[Any, str]:
    s = self.analyzer.stats(kp_name, example_limit=5).to_dict()
    if not s["total"]:
        return s, f"「{kp_name}」没有匹配到任何真题考点"
    if self.eval_mode:
        # 评测模式不能把同 tag 真题的答案交给模型：
        # get_kp_questions 已经剥离了答案，stats.examples 也必须一致，
        # 否则 Agent 可以绕过前者、从统计示例里拿到答案。
        for ex in s.get("examples") or []:
            ex.pop("answer", None)
    # 控制返回体积：qids 已足够溯源，示例只留 5 条
    s["qids"] = s["qids"][:40]
    return s, f"「{kp_name}」历年考查 {s['total']} 次（{len(s['matched_tags'])} 个标签）"


def _get_kp_questions(self: AgentTools, kp_name: str, limit: int = 5) -> tuple[Any, str]:
    rows = self.analyzer.questions_for_tag(kp_name, limit=limit + 2)
    if self.eval_mode:
        # 评测模式：不能把答案交给模型，也不能返回当前题自身
        rows = [r for r in rows if r["qid"] != self.current_qid][:limit]
        for r in rows:
            r.pop("answer", None)
            r.pop("explanation", None)
    else:
        rows = rows[:limit]
    return (
        {"kp_name": kp_name, "n": len(rows), "questions": rows},
        f"返回 {len(rows)} 道真题原题",
    )


def _get_related_kps(self: AgentTools, kp_name: str) -> tuple[Any, str]:
    related = self.analyzer.related_tags(kp_name, top_n=8)
    graph_info: dict[str, Any] = {}
    try:
        prof = self.graph.kp_profile(kp_name)
        if prof:
            graph_info = {
                "title": prof.get("title"),
                "path": prof.get("path"),
                "siblings": (prof.get("siblings") or [])[:10],
                "ancestors": (prof.get("ancestors") or [])[:8],
            }
    except Exception as e:  # noqa: BLE001 - 图谱是增强，不是必需
        graph_info = {"error": str(e)[:150]}
    data = {
        "kp_name": kp_name,
        "cooccurring_tags": related,
        "graph": graph_info,
        "note": "共现关系来自同一道真题的多个考点标签，evidence 为证据题号。",
    }
    return data, f"共现考点 {len(related)} 个" + (
        "" if graph_info.get("title") else "；图谱未定位到该知识点"
    )


def _verify_answer(
    self: AgentTools,
    question: str,
    answer: str,
    contexts: Sequence[Any] | None = None,
) -> tuple[Any, str]:
    blocks = list(contexts or [])
    rep = check_citations(answer, blocks, valid_exam_refs=self.valid_exam_refs)
    refs = extract_exam_refs(answer)
    invalid = [r for r in refs if self.valid_exam_refs and r not in self.valid_exam_refs]
    predictions = detect_predictions(answer)
    data = {
        "citation": rep.to_dict(),
        "exam_refs": refs,
        "invalid_exam_refs": invalid,
        "predictions": predictions,
        "ok": not invalid and not predictions,
    }
    problems = []
    if invalid:
        problems.append(f"{len(invalid)} 个真题编号不存在")
    if predictions:
        problems.append(f"命中押题表述 {predictions[:3]}")
    summary = "校验通过" if not problems else "；".join(problems)
    return data, summary


def _get_learning_path(self: AgentTools, kp_name: str) -> tuple[Any, str]:
    """沿 PREREQUISITE 反向走，返回学习该知识点的前置顺序。"""
    path = self.graph.learning_path(kp_name)
    if not path.get("found"):
        return path, f"图谱中没有找到「{kp_name}」的前置链"
    chain = path.get("chain") or []
    return path, f"前置链 {len(chain)} 个节点，深度 {path.get('depth', 0)}"


TOOL_REGISTRY: dict[str, Callable[..., tuple[Any, str]]] = {
    "search_knowledge_base": _search_knowledge_base,
    "search_knowledge_graph": _search_knowledge_graph,
    "read_figure": _read_figure_tool,
    "calculate": _calculate,
    "get_exam_stats": _get_exam_stats,
    "get_kp_questions": _get_kp_questions,
    "get_related_kps": _get_related_kps,
    "get_learning_path": _get_learning_path,
    "verify_answer": _verify_answer,
}

TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search_knowledge_base": {
        "desc": "在教材与真题解析中做三路混合检索，返回带章节路径的资料片段。",
        "args": {"query": "检索词", "subject": "可选，四科之一", "top_k": "可选，默认 8"},
    },
    "search_knowledge_graph": {
        "desc": "在知识点图谱中定位一个知识点，返回章节路径、祖先与同节兄弟。",
        "args": {"kp_name": "知识点名称"},
    },
    "read_figure": {
        "desc": "读取某道真题的配图。T1 重建拓扑，T2 提取文本，T3 需视觉模型。",
        "args": {"question_id": "真题编号，如 2009-04"},
    },
    "calculate": {
        "desc": "计算一个算式（四则、幂、log2/sqrt/ceil 等），避免心算错误。",
        "args": {"expr": "算式，如 (1-0.05)**3 或 log2(4096)"},
    },
    "get_exam_stats": {
        "desc": "考点在历年真题中的考查频次、年份/科目/题型分布。",
        "args": {"kp_name": "考点名，如 排序算法"},
    },
    "get_kp_questions": {
        "desc": "某考点的历年真题原题（含年份、题干、选项；非评测模式含答案解析）。",
        "args": {"kp_name": "考点名", "limit": "条数，默认 5"},
    },
    "get_related_kps": {
        "desc": "与某考点关联的其他考点：真题共现（带证据题号）+ 图谱同节/祖先。",
        "args": {"kp_name": "考点名"},
    },
    "get_learning_path": {
        "desc": "该知识点的前置学习链（沿 PREREQUISITE 反向），用于学习路径推荐。",
        "args": {"kp_name": "知识点名"},
    },
    "verify_answer": {
        "desc": "自检：引用编号是否存在、真题编号是否真实、是否出现押题表述。",
        "args": {"question": "问题", "answer": "待校验的回答", "contexts": "资料块"},
    },
}


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    t = AgentTools()
    for name, kw in [
        ("get_exam_stats", {"kp_name": "死锁"}),
        ("get_related_kps", {"kp_name": "Cache"}),
        ("calculate", {"expr": "log2(4096)"}),
        ("read_figure", {"question_id": "2009-04"}),
        ("search_knowledge_base", {"query": "组相联映射 地址划分"}),
    ]:
        r = t.call(name, **kw)
        print(f"{name}: ok={r.ok} {r.summary} ({r.took_ms:.0f}ms)")
    print("trace:", t.trace.by_name(), "failures:", t.failures)