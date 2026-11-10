"""Agent 的九个工具。

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

from collections.abc import Callable, Sequence
from typing import Any

from cs408rag.agent.calculator import evaluate
from cs408rag.agent.contracts import ToolResult, ToolTrace
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer
from cs408rag.agent.guardrails import ToolGuard
from cs408rag.figures.read_figure import read_figure
from cs408rag.generation.citation import (
    check_citations,
    detect_predictions,
    extract_exam_refs,
)
from cs408rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

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
        # 写操作（打卡/判分）与画像读取的命名空间。由调用方在 run 前绑定，
        # LLM 无法通过参数指定别人 —— 四个新工具的参数模型里都没有 client_id。
        self.client_id = "default"
        self.trace = ToolTrace()
        self.failures: dict[str, int] = {}
        # 所有调用都经过护栏：白名单、参数契约、预算、超时与审计。
        self.guard = ToolGuard(self)

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

    def set_client(self, client_id: str | None) -> None:
        """绑定当前用户。reset() 不清它 —— 用户身份跨轮次不变。"""
        self.client_id = (client_id or "").strip() or "default"

    def reset(self, *, current_qid: str | None = None) -> None:
        """开始新的一次 Agent 运行前清空 trace。

        trace 是实例级状态。实测踩过：连续跑三题不做清理，
        第三题的 tool_calls 里会混进前两题的调用，成本统计直接失真。
        """
        self.trace = ToolTrace()
        self.failures = {}
        # 只清轮次预算；每分钟窗口跨请求保留，否则连续请求可绕过限流。
        self.guard.reset_round()
        self.current_qid = current_qid

    # -- 统一调用入口 -----------------------------------------------------

    def call(
        self,
        name: str,
        *,
        round_no: int | None = None,
        count_budget: bool = True,
        **kwargs: Any,
    ) -> ToolResult:
        """经护栏调用工具。

        `count_budget=False` 用于 retrieve 节点内部的检索调用：它们同样
        需要参数校验与审计，但不占用 planner 的单轮工具预算，
        否则 3 条查询 × 2 次重试就能把 6 次预算吃光。
        """
        return self.guard.call(
            name, round_no=round_no, count_budget=count_budget, **kwargs
        )

    def _execute(self, name: str, args: dict[str, Any]) -> tuple[Any, str]:
        """真正执行工具。只允许护栏调用，不对外暴露。"""
        fn = TOOL_REGISTRY[name]
        return fn(self, **args)


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


# ---------------------------------------------------------------------------
# 中枢工具：规划 / 画像 / 打卡 / 判分。
# 这四个工具把「规划看板」与「画像闭环」接进 Agent 聊天入口：
# 规划与画像是只读计算；打卡与判分是写操作（契约里 idempotent=False）。
# ---------------------------------------------------------------------------


def _profile_view(self: AgentTools) -> dict[str, Any]:
    from cs408rag.profile.stats import build_profile
    from cs408rag.profile.store import get_profile_store

    return build_profile(get_profile_store().events(self.client_id))


def _generate_plan(
    self: AgentTools,
    weak_subjects: list[str] | None = None,
    progress: dict[str, str] | None = None,
    daily_hours: int = 4,
    study_days_per_week: int = 6,
    target_score: int = 110,
    concerns: list[str] | None = None,
) -> tuple[Any, str]:
    from cs408rag.adversarial.planner import (
        PROGRESS_LEVELS,
        SUBJECTS,
        build_detailed_plan,
    )

    prof = _profile_view(self)
    weak = [s for s in (weak_subjects or []) if s in SUBJECTS]
    weak_from_profile = False
    if not weak:
        # 没指定薄弱科目时，从画像里答错最多的科目推断 —— 规划自动吃画像
        wrong_subj = (prof.get("quiz") or {}).get("wrong_subjects") or {}
        weak = list(wrong_subj)[:2]
        weak_from_profile = bool(weak)
    prog = {s: "未开始" for s in SUBJECTS}
    prog.update(
        {
            s: str(p)
            for s, p in (progress or {}).items()
            if s in SUBJECTS and str(p) in PROGRESS_LEVELS
        }
    )
    goal = {
        "weak_subjects": weak,
        "progress": prog,
        "daily_hours": daily_hours,
        "study_days_per_week": study_days_per_week,
        "target_score": target_score,
        "concerns": [str(c)[:100] for c in (concerns or [])],
        "profile_tags": prof.get("boost_tags") or [],
    }
    plan = build_detailed_plan(goal, analyzer=self.analyzer)
    g = plan["goal"]
    # 回传完整 plan + 摘要：摘要供 LLM 组织回答，完整 plan 供 UI 直接渲染看板
    data = {
        # 完整 plan 一并回传：UI 直接用它渲染看板，不再按 goal 重跑一次 PlanRunner
        "plan": plan,
        "goal": goal,
        "weeks": g["remaining_weeks"],
        "start_date": g["start_date"],
        "exam_date": g["exam_date"],
        "total_hours": g["total_hours"],
        "daily_hours": g["daily_hours"],
        "target_score": target_score,
        "weak_subjects": weak,
        "weak_from_profile": weak_from_profile,
        "boosted_by_profile": [
            t["tag"] for t in (prof.get("boost_tags") or [])[:8] if t.get("tag")
        ],
        "stages": [
            {
                "name": s.get("name"),
                "weeks": s.get("weeks"),
                "start": s.get("start"),
                "end": s.get("end"),
                "focus": s.get("focus"),
                "checkpoint": s.get("checkpoint"),
            }
            for s in plan.get("stages") or []
        ],
        "priority_kps": [
            {
                "subject": k.get("subject"),
                "tag": k.get("tag"),
                "frequency": k.get("frequency"),
                "roi_label": k.get("roi_label"),
            }
            for k in (plan.get("priority_kps") or [])[:12]
        ],
        "guide_overview": str((plan.get("guide") or {}).get("overview") or "")[:800],
        "risk_notes": plan.get("risk_notes") or [],
    }
    summary = (
        f"已生成 {data['weeks']} 周规划"
        f"（{len(data['stages'])} 个阶段，总 {data['total_hours']:g}h）"
        + ("；薄弱科目来自画像" if weak_from_profile else "")
    )
    return data, summary


def _read_profile(self: AgentTools) -> tuple[Any, str]:
    prof = _profile_view(self)
    quiz = prof.get("quiz") or {}
    exe = prof.get("execution") or {}
    data = {
        "client_id": self.client_id,
        "total_questions": prof.get("total_questions", 0),
        "top_tags": (prof.get("top_tags") or [])[:8],
        "quiz": {
            "total": quiz.get("total", 0),
            "correct": quiz.get("correct", 0),
            "wrong_tags": (quiz.get("wrong_tags") or [])[:8],
            "wrong_subjects": quiz.get("wrong_subjects") or {},
        },
        "execution": {
            "checkins": exe.get("checkins", 0),
            "done": exe.get("done", 0),
            "tasks_total": exe.get("tasks_total", 0),
            "failed_checkpoints": exe.get("failed_checkpoints") or [],
        },
        "plan": prof.get("plan"),
        "boost_tags": prof.get("boost_tags") or [],
    }
    summary = (
        f"画像：提问 {data['total_questions']} 次 · "
        f"作答 {quiz.get('correct', 0)}/{quiz.get('total', 0)} 对 · "
        f"打卡 {exe.get('done', 0)}/{exe.get('tasks_total', 0)}"
    )
    return data, summary


def _record_checkin(
    self: AgentTools,
    week: int,
    idx: int = 0,
    done: bool = True,
    subject: str = "",
    kps: list[str] | None = None,
    hours: float | None = None,
    stage: str = "",
) -> tuple[Any, str]:
    from cs408rag.profile.store import get_profile_store

    payload = {
        "week": week,
        "idx": idx,
        "stage": stage,
        "subject": subject,
        "kps": [str(k) for k in (kps or [])][:8],
        "hours": hours,
        "done": bool(done),
        # 不带 total：画像的任务总数只认 UI 打卡事件的 total，
        # Agent 口头打卡不参与分母，避免虚报完成率。
    }
    get_profile_store().record(self.client_id, "checkin", payload)
    mark = "完成" if done else "取消完成"
    return (
        {"recorded": True, "client_id": self.client_id, **payload},
        f"已记录第 {week} 周第 {idx + 1} 项打卡（{mark}），画像执行统计已更新",
    )


def _judge_quiz(self: AgentTools, qid: str, choice: str) -> tuple[Any, str]:
    from cs408rag.notebook.judge import judge
    from cs408rag.profile.store import get_profile_store

    r = judge(self.client_id, qid, choice)
    if not r.get("known"):
        return r, f"题库中没有 {qid} 或该题无答案，未判分"
    get_profile_store().record(
        self.client_id,
        "quiz",
        {
            "qid": r["qid"],
            "subject": r.get("subject") or "",
            "tags": r.get("tags") or [],
            "correct": bool(r.get("correct")),
            "choice": choice,
        },
    )
    mark = "答对，已从错题本移出" if r.get("correct") else f"答错（正确答案 {r.get('truth')}），已进错题本"
    return r, f"{r['qid']} 作答 {choice}：{mark}"


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
    "generate_plan": _generate_plan,
    "read_profile": _read_profile,
    "record_checkin": _record_checkin,
    "judge_quiz": _judge_quiz,
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