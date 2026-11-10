"""FastAPI 服务层。

接口：
    POST /v1/query            开放式问答
    POST /v1/query/stream     SSE 流式问答
    POST /v1/quiz             选择题作答
    POST /v1/retrieve         只要检索结果（不调 LLM），调试与前端渲染用
    GET  /v1/health
    GET  /v1/trace/{id}       单次请求的完整检索路径与耗时分解
    GET  /v1/profile          用户画像（提问/作答/规划事件的聚合视图）
    DELETE /v1/profile        清空画像

工程要点（面试可讲）：
    - **异步 ↔ 同步 GPU 推理的桥接**：BGE-M3 与 reranker 是同步阻塞的，
      直接在 async 函数里调用会堵死事件循环。这里统一用
      `run_in_threadpool` 把它们甩到线程池。
      torch 推理释放 GIL，线程池能真正并行。
    - **请求级 trace**：每次请求分配 request_id，把各阶段耗时、
      三路召回明细、RRF 前后排名变化全存下来，供 /v1/trace 还原。
    - **降级**：LLM 不可用时返回检索结果 + 明确说明，而不是 500。
    - **画像一致化**：请求带 client_id 时，提问/作答事件写入与 Streamlit
      同一个 profile.db，两个入口共享一份画像；落库是尽力而为的
      side-effect，失败只记日志，不拖垮主请求。
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from cs408rag.config import settings
from cs408rag.generation.service import QAService
from cs408rag.retrieval.pipeline import STAGES, RetrievalConfig, RetrievalPipeline

# 内存里的 trace 环形缓冲。生产该用 Redis/ClickHouse，演示够用。
TRACE_CAPACITY = 500
_traces: OrderedDict[str, dict[str, Any]] = OrderedDict()

_state: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# 生命周期
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("启动中：加载检索链路…")
    # 预热放到线程池，避免阻塞启动
    await run_in_threadpool(_warmup)

    # LLM 也要预热，但**不能阻塞启动**：首次调用实测要 45 秒（模型冷启动），
    # 之后就降到 1.5 秒。若让第一个用户承担这 45 秒，体验会很差。
    # 所以丢到后台线程里跑，启动照常返回。
    threading.Thread(target=_warm_llm, daemon=True).start()

    logger.info("就绪")
    yield
    logger.info("关闭中…")
    _state.clear()


def _warm_llm() -> None:
    """后台预热 LLM，避免首个请求吃冷启动延迟。失败不影响服务。"""
    try:
        svc = _get_service()
        if not svc.llm.available:
            return
        t0 = time.perf_counter()
        r = svc.llm.chat([{"role": "user", "content": "hi"}], max_tokens=1)
        if r.ok:
            logger.info(f"LLM 预热完成（{(time.perf_counter() - t0):.1f}s，模型 {svc.llm.model}）")
        else:
            logger.warning(f"LLM 预热失败：{r.error[:160]}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"LLM 预热异常：{e}")


def _warmup() -> None:
    from cs408rag.retrieval.embedder import get_embedder

    get_embedder().load()
    pipe = _get_pipeline()
    # 首调用含 jieba 词典与术语表加载，预热一次让延迟数据干净
    pipe.retrieve("预热", config=RetrievalConfig(exclude_explanations=False))


def _get_pipeline() -> RetrievalPipeline:
    if "pipeline" not in _state:
        _state["pipeline"] = RetrievalPipeline()
    return _state["pipeline"]


def _get_service() -> QAService:
    if "service" not in _state:
        from cs408rag.generation.service import build_exam_ref_whitelist

        _state["service"] = QAService(
            pipeline=_get_pipeline(),
            valid_exam_refs=build_exam_ref_whitelist(),
        )
    return _state["service"]


def _get_runner():
    """Agent 运行器。与问答服务共享同一个检索链路与模型实例。"""
    if "runner" not in _state:
        from cs408rag.agent.graph import AgentRunner
        from cs408rag.agent.tools import AgentTools
        from cs408rag.generation.service import build_exam_ref_whitelist

        tools = AgentTools(
            pipeline=_get_pipeline(),
            valid_exam_refs=build_exam_ref_whitelist(),
        )
        _state["runner"] = AgentRunner(tools=tools)
    return _state["runner"]


app = FastAPI(
    title="408-RAG",
    description="面向考研 408 的真题检索问答与考点分析系统",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 请求/响应模型
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    subject: str | None = Field(None, description="限定科目，如「数据结构」")
    stage: str = Field("qa", description="检索配置：qa 或 S0_dense/S1_hybrid/…")
    top_n: int = Field(8, ge=1, le=30)
    client_id: str | None = Field(
        None, max_length=64, description="匿名身份 ID；给了就把本次提问计入用户画像"
    )


class QuizRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=3000)
    options: dict[str, str] = Field(..., description='{"A": "...", "B": "..."}')
    subject: str | None = None
    stage: str = "qa"


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    subject: str | None = None
    stage: str = "S4_filter"
    top_k: int = Field(10, ge=1, le=50)


class AgentRequest(BaseModel):
    """Agent 问答。mode=quiz 时走选择题口径，须给 options。"""

    query: str = Field(..., min_length=1, max_length=3000)
    options: dict[str, str] | None = Field(None, description='mode=quiz 时必填')
    subject: str | None = None
    qid: str | None = Field(None, description="真题编号，如 2009-04；给了会启用图形解析")
    mode: str = Field("qa", description="qa 或 quiz")
    client_id: str | None = Field(
        None, max_length=64, description="匿名身份 ID；给了就把本次提问计入用户画像"
    )


class ChatRequest(AgentRequest):
    """多轮对话请求。session_id 缺省时服务端新建会话。

    会话只存内存（LRU 200 + TTL 2h），重启丢历史是明确取舍。
    profile 目前仅随会话保存（如 {"weak_subjects": ["操作系统"], "target_score": 110}），
    供 Agent 侧使用，不强行耦合对抗 Plan。
    """

    session_id: str | None = Field(None, max_length=64)
    profile: dict[str, Any] | None = None


class ContextOut(BaseModel):
    index: int
    citation: str
    source: str
    subject: str
    chunk_id: str
    text: str
    images: list[dict] = []


def _hits_to_payload(hits: list, start: int = 1) -> list[dict]:
    """把检索结果转成 API 输出，并把教材插图补全成可访问的 URL。

    插图处理：chunk 的 payload 里只有 {image_id, src, caption}（相对路径），
    这里用 image_index.json 解析出 /v1/image/{id}，前端可直接 <img src>。
    """
    from cs408rag.ingestion.image_index import resolve_images

    out: list[dict] = []
    for i, h in enumerate(hits, start):
        out.append(
            {
                "index": i,
                "chunk_id": h.chunk_id,
                "score": round(h.score, 4),
                "subject": h.subject,
                "source": h.source,
                "chapter_path": h.chapter_path,
                "citation": h.citation,
                "text": h.text,
                "images": resolve_images(h.payload.get("images")),
            }
        )
    return out


def _resolve_config(stage: str, top_n: int | None = None) -> RetrievalConfig:
    if stage == "qa":
        from cs408rag.generation.service import DEFAULT_QA_CONFIG

        cfg = DEFAULT_QA_CONFIG
    elif stage in STAGES:
        cfg = STAGES[stage]
    else:
        raise HTTPException(400, f"未知 stage={stage!r}，可选：qa, {', '.join(STAGES)}")
    if top_n is not None:
        cfg = RetrievalConfig(**{**cfg.__dict__, "rerank_top_n": top_n})
    return cfg


def _record_trace(rid: str, payload: dict[str, Any], *, source: str = "api") -> None:
    _traces[rid] = payload
    while len(_traces) > TRACE_CAPACITY:
        _traces.popitem(last=False)
    # 页面不再展示的三路召回 / RRF 排名，落盘留档供 /v1/traces 查询
    from cs408rag.observability import record_payload

    record_payload(
        payload,
        source=source,
        keys=("request_id", "query", "subject", "took_ms", "trace", "stage", "request"),
    )


# ---------------------------------------------------------------------------
# 用户画像：与 Streamlit 路径同源，写入同一个 profile.db
# ---------------------------------------------------------------------------


def _record_profile_question(
    client_id: str | None, text: str, *, subject: str | None = None, qid: str | None = None
) -> None:
    """提问事件落库（尽力而为）：画像只是增强信号，失败不拖垮主请求。"""
    if not client_id:
        return
    try:
        from cs408rag.agent.exam_analysis import get_analyzer
        from cs408rag.profile.stats import extract_tags
        from cs408rag.profile.store import get_profile_store

        tags = extract_tags(text, get_analyzer())
        get_profile_store().record(
            client_id,
            "question",
            {"text": text[:200], "subject": subject or "", "qid": qid or "", "tags": tags},
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"画像提问事件落库失败：{e}")


def _persist_agent_plan(client_id: str | None, res_dict: dict[str, Any]) -> None:
    """Agent 工具生成的规划落库（仅当显式给了 client_id）。

    规划是独立记录、按身份绑定：不传 client_id 就不写，避免匿名请求
    互相覆盖同一个 default 规划。
    """
    if not client_id:
        return
    plan_out = (res_dict.get("tool_outputs") or {}).get("plan_result")
    if not plan_out:
        return
    try:
        from cs408rag.plan_persistence import persist_plan

        persist_plan(
            client_id, plan_out, plan_out.get("goal"), stop_reason="agent_chat"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"规划落库失败：{e}")


def _record_profile_quiz(client_id: str, result: dict[str, Any], choice: str) -> None:
    """作答事件落库：答错考点是画像加权里优先级最高的薄弱信号。"""
    if not result.get("known"):
        return
    try:
        from cs408rag.profile.store import get_profile_store

        get_profile_store().record(
            client_id,
            "quiz",
            {
                "qid": result.get("qid", ""),
                "subject": result.get("subject") or "",
                "tags": result.get("tags") or [],
                "correct": bool(result.get("correct")),
                "choice": choice,
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"画像作答事件落库失败：{e}")


# ---------------------------------------------------------------------------
# 接口
# ---------------------------------------------------------------------------


@app.get("/v1/health")
async def health() -> dict[str, Any]:
    pipe = _get_pipeline()
    out: dict[str, Any] = {
        "status": "ok",
        "llm_available": _get_service().llm.available,
        "llm_model": settings.llm_model,
        "collection": settings.qdrant_collection,
    }
    try:
        out["indexed_chunks"] = await run_in_threadpool(pipe.store.count)
    except Exception as e:  # noqa: BLE001
        out["status"] = "degraded"
        out["qdrant_error"] = str(e)[:200]
    try:
        out["graph_stats"] = await run_in_threadpool(pipe.graph.stats)
    except Exception as e:  # noqa: BLE001
        out["graph_error"] = str(e)[:200]
    return out


@app.post("/v1/retrieve")
async def retrieve(req: RetrieveRequest) -> dict[str, Any]:
    """只检索，不调 LLM。前端渲染「找到了哪些章节」用这个。"""
    rid = uuid.uuid4().hex[:12]
    cfg = _resolve_config(req.stage, top_n=req.top_k)

    t0 = time.perf_counter()
    r = await run_in_threadpool(
        _get_pipeline().retrieve, req.query, subject=req.subject, config=cfg
    )
    took = (time.perf_counter() - t0) * 1000

    payload = {
        "request_id": rid,
        "query": req.query,
        "took_ms": round(took, 1),
        "results": _hits_to_payload(r.top_hits(req.top_k)),
        "trace": r.trace_summary(),
    }
    _record_trace(rid, payload)
    return payload


@app.get("/v1/image/{image_id}")
async def get_image(image_id: str):
    """提供教材插图。

    图片原文件在 data/raw/ 下（不进仓库），这里按 image_index.json 解析后返回。
    找不到返回 404 —— 前端据此决定是否渲染。
    """
    from fastapi.responses import FileResponse

    from cs408rag.ingestion.image_index import image_abs_path

    p = image_abs_path(image_id)
    if p is None:
        raise HTTPException(404, f"图片 {image_id} 不存在或未建索引")
    return FileResponse(p, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/v1/images/stats")
async def image_stats() -> dict[str, Any]:
    """图片索引概览，方便确认索引是否建好。"""
    from collections import Counter

    from cs408rag.ingestion.image_index import load_image_index

    idx = load_image_index()
    return {
        "total": len(idx),
        "by_subject": dict(Counter(v.get("subject", "") for v in idx.values())),
        "total_mb": round(sum(v.get("bytes", 0) for v in idx.values()) / 1e6, 1),
    }


@app.post("/v1/query")
async def query(req: QueryRequest) -> dict[str, Any]:
    rid = uuid.uuid4().hex[:12]
    svc = _get_service()
    cfg = _resolve_config(req.stage, top_n=req.top_n)

    res = await run_in_threadpool(
        svc.answer, req.query, subject=req.subject, config=cfg
    )
    _record_profile_question(req.client_id, req.query, subject=req.subject)
    payload = {"request_id": rid, **res.to_dict()}
    if res.retrieval:
        _record_trace(rid, payload)
    return payload


@app.post("/v1/quiz")
async def quiz(req: QuizRequest) -> dict[str, Any]:
    rid = uuid.uuid4().hex[:12]
    svc = _get_service()
    cfg = _resolve_config(req.stage)

    res = await run_in_threadpool(
        svc.answer_quiz, req.question, req.options, subject=req.subject, config=cfg
    )
    payload = {"request_id": rid, **res.to_dict()}
    _record_trace(rid, payload)
    return payload


# ---------------------------------------------------------------------------
# Agent 与考点分析
# ---------------------------------------------------------------------------


@app.post("/v1/agent/query")
async def agent_query(req: AgentRequest) -> dict[str, Any]:
    """Agent 编排入口：route → retrieve → assess → plan → act → observe → answer → verify。

    返回完整推理链（steps + plan_history + tool_calls + verification），
    前端据此展示「每轮用了什么工具、为什么、引用是否可溯源」。
    """
    if req.mode not in ("qa", "quiz"):
        raise HTTPException(400, "mode 只能是 qa 或 quiz")
    # 部分真题的选项画在图中，options 会是空对象 —— 不能因此拒绝请求。
    # 具体校验交给 Agent：选项为空时它会依据图形解析判断。

    rid = uuid.uuid4().hex[:12]
    runner = _get_runner()
    res = await run_in_threadpool(
        runner.run,
        req.query,
        options=req.options,
        subject=req.subject,
        qid=req.qid,
        mode=req.mode,
        client_id=req.client_id,
    )
    _record_profile_question(req.client_id, req.query, subject=req.subject, qid=req.qid)
    payload = {"request_id": rid, **res.to_dict()}
    _persist_agent_plan(req.client_id, payload)
    _record_trace(rid, payload)
    return payload


@app.post("/v1/agent/chat")
async def agent_chat(req: ChatRequest) -> dict[str, Any]:
    """多轮对话入口：在 /v1/agent/query 的基础上增加会话记忆与指代消解。

    - 带同一个 session_id 追问（「那 2021 年呢」「这题选什么」）会自动沿用上轮
      考点/真题；追问被改写时响应里的 `resolved_query` 是实际执行的问题。
    - 响应新增 `session_id` / `turn_no` / `resolved_query`。
    """
    if req.mode not in ("qa", "quiz"):
        raise HTTPException(400, "mode 只能是 qa 或 quiz")

    rid = uuid.uuid4().hex[:12]
    runner = _get_runner()
    res = await run_in_threadpool(
        runner.run,
        req.query,
        options=req.options,
        subject=req.subject,
        qid=req.qid,
        mode=req.mode,
        session_id=req.session_id or rid,
        profile=req.profile,
        client_id=req.client_id,
    )
    _record_profile_question(req.client_id, req.query, subject=req.subject, qid=req.qid)
    payload = {"request_id": rid, **res.to_dict()}
    _persist_agent_plan(req.client_id, payload)
    _record_trace(rid, payload)
    return payload


@app.get("/v1/agent/session/{session_id}")
async def agent_session_get(
    session_id: str,
    client_id: str = Query(..., min_length=1, max_length=64),
) -> dict[str, Any]:
    """查看会话历史（turns / last_qid / last_topic / summary）。

    必须带 client_id；会话绑定了其他身份时返回 404，避免越权读取。
    """
    from cs408rag.agent.memory import get_session_store

    s = await run_in_threadpool(get_session_store().get, session_id)
    if s is None or (s.client_id and s.client_id != client_id):
        raise HTTPException(404, "会话不存在或已过期（内存会话 TTL 2h，重启即失效）")
    return s.to_dict()


@app.delete("/v1/agent/session/{session_id}")
async def agent_session_delete(
    session_id: str,
    client_id: str = Query(..., min_length=1, max_length=64),
) -> dict[str, Any]:
    """清空会话（「新会话」等价于丢弃旧 session_id）。

    必须带 client_id；会话绑定了其他身份时返回 404，避免越权删除。
    """
    from cs408rag.agent.memory import get_session_store

    store = get_session_store()
    s = await run_in_threadpool(store.get, session_id)
    if s is None or (s.client_id and s.client_id != client_id):
        raise HTTPException(404, "会话不存在或已过期")
    ok = await run_in_threadpool(store.delete, session_id)
    if not ok:
        raise HTTPException(404, "会话不存在或已过期")
    return {"session_id": session_id, "deleted": True}


# ---------------------------------------------------------------------------
# 错题本（匿名 client_id + SQLite，答对自动移出）
# ---------------------------------------------------------------------------


class JudgeRequest(BaseModel):
    client_id: str = Field(min_length=1, max_length=64)
    qid: str = Field(min_length=1, max_length=32)
    your_choice: str = Field(min_length=1, max_length=8)


@app.post("/v1/notebook/judge")
async def notebook_judge(req: JudgeRequest) -> dict[str, Any]:
    """作答判分：对照题库 ground truth，答错记入错题本、答对自动移出。

    qid 不在题库（临时粘贴的题）时返回 known=False，不落库。
    """
    from cs408rag.notebook.judge import judge

    choice = req.your_choice.strip().upper()
    if choice not in ("A", "B", "C", "D"):
        raise HTTPException(400, "your_choice 只能是 A/B/C/D")
    res = await run_in_threadpool(judge, req.client_id, req.qid, choice)
    _record_profile_quiz(req.client_id, res, choice)
    return res


@app.get("/v1/notebook/items")
async def notebook_items(client_id: str, subject: str | None = None) -> dict[str, Any]:
    """错题列表，按最近答错时间倒序。"""
    from cs408rag.notebook.store import get_store

    items = await run_in_threadpool(get_store().list_items, client_id, subject)
    return {"items": items}


@app.delete("/v1/notebook/items/{qid}")
async def notebook_delete(qid: str, client_id: str) -> dict[str, Any]:
    """手动移出错题（答对自动移出走 /v1/notebook/judge）。"""
    from cs408rag.notebook.store import get_store

    ok = await run_in_threadpool(get_store().remove, client_id, qid)
    return {"qid": qid, "deleted": ok}


# ---------------------------------------------------------------------------
# 用户画像（提问/作答/规划事件的聚合视图，与 Streamlit 侧栏同源）
# ---------------------------------------------------------------------------


@app.get("/v1/profile")
async def profile_get(client_id: str) -> dict[str, Any]:
    """读取画像：高频提问考点、答错考点、最近规划等，供任意前端渲染。"""
    from cs408rag.profile.stats import build_profile
    from cs408rag.profile.store import get_profile_store

    events = await run_in_threadpool(get_profile_store().events, client_id)
    return build_profile(events)


@app.delete("/v1/profile")
async def profile_delete(client_id: str) -> dict[str, Any]:
    """清空画像：事件全删，不可恢复。"""
    from cs408rag.profile.store import get_profile_store

    n = await run_in_threadpool(get_profile_store().clear, client_id)
    return {"client_id": client_id, "cleared": n}


@app.get("/v1/exam/overview")
async def exam_overview() -> dict[str, Any]:
    """考点库概览：题量、标签数、题型/科目/年份分布。"""
    from cs408rag.agent.exam_analysis import get_analyzer

    return await run_in_threadpool(get_analyzer().tag_distribution)


@app.get("/v1/exam/tags")
async def exam_tags(subject: str | None = None, limit: int = 50) -> dict[str, Any]:
    from cs408rag.agent.exam_analysis import get_analyzer

    a = get_analyzer()
    return {"tags": a.all_tags(subject=subject, limit=limit)}


@app.get("/v1/exam/stats")
async def exam_stats(tag: str, example_limit: int = 5) -> dict[str, Any]:
    """某个考点的真实考查统计（频次、年份、科目、题型、真题编号）。"""
    from cs408rag.agent.exam_analysis import get_analyzer

    a = get_analyzer()
    s = a.stats(tag, example_limit=example_limit).to_dict()
    return s


@app.get("/v1/exam/questions")
async def exam_questions(tag: str, limit: int = 10) -> dict[str, Any]:
    from cs408rag.agent.exam_analysis import get_analyzer

    a = get_analyzer()
    return {"tag": tag, "questions": a.questions_for_tag(tag, limit=limit)}


@app.get("/v1/figures/{qid}")
async def figure(qid: str, question: str = "") -> dict[str, Any]:
    """图表题分层解析。T3（位图）当前不可用，会明确返回原因。"""
    from cs408rag.figures.read_figure import read_figure

    r = await run_in_threadpool(read_figure, qid, question=question)
    return r.to_dict()


@app.post("/v1/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    """SSE 流式问答。

    先把检索到的章节推给前端（立即可渲染），再逐段推正文。
    同步的检索与流式生成都在线程池里跑，不阻塞事件循环。
    """
    import json

    cfg = _resolve_config(req.stage, top_n=req.top_n)
    svc = _get_service()
    _record_profile_question(req.client_id, req.query, subject=req.subject)

    async def gen():
        # 检索是同步阻塞的，丢进线程池
        from cs408rag.generation.prompt import build_context

        retrieval = await run_in_threadpool(
            svc.pipeline.retrieve, req.query, subject=req.subject, config=cfg
        )
        blocks = build_context(retrieval.top_hits(cfg.rerank_top_n))
        yield f"data: {json.dumps({'type': 'contexts', 'data': [{'index': b.index, 'citation': b.citation, 'source': b.source} for b in blocks]}, ensure_ascii=False)}\n\n"

        if not blocks:
            yield f"data: {json.dumps({'type': 'error', 'data': '检索不到相关资料'}, ensure_ascii=False)}\n\n"
            return

        # 生成用同步迭代器，逐块转发。每块之间让出事件循环。
        from cs408rag.generation.prompt import qa_messages

        it = svc.llm.stream(qa_messages(req.query, blocks), max_tokens=2000)
        buf: list[str] = []
        while True:
            piece = await run_in_threadpool(lambda: next(it, None))
            if piece is None:
                break
            buf.append(piece)
            yield f"data: {json.dumps({'type': 'delta', 'data': piece}, ensure_ascii=False)}\n\n"

        from cs408rag.generation.citation import check_citations

        rep = check_citations("".join(buf), blocks, valid_exam_refs=svc.valid_exam_refs)
        yield f"data: {json.dumps({'type': 'citation', 'data': rep.to_dict()}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/trace/{request_id}")
async def get_trace(request_id: str) -> dict[str, Any]:
    t = _traces.get(request_id)
    if not t:
        raise HTTPException(404, f"没有 request_id={request_id} 的记录（环形缓冲容量 {TRACE_CAPACITY}）")
    return t


@app.get("/v1/traces")
async def list_traces(limit: int = 50) -> dict[str, Any]:
    """落盘的检索留档（最近的在前）。数据来自页面已隐藏的检索调试 / 实验视图。"""
    from cs408rag.observability import read_traces

    items = read_traces(max(1, min(limit, 500)))
    return {"count": len(items), "items": items}


@app.get("/v1/ablation")
async def ablation() -> dict[str, Any]:
    """检索消融结果（页面「实验数据」表已下线，原始 JSON 仍保留在 reports/）。"""
    from pathlib import Path

    p = Path(__file__).resolve().parents[3] / "reports" / "ablation_dev.json"
    if not p.exists():
        raise HTTPException(404, "还没有消融结果：先运行检索消融脚本生成 reports/ablation_dev.json")
    import json

    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/v1/stages")
async def stages() -> dict[str, str]:
    """列出可用的检索阶段，供消融实验与前端切换。"""
    from cs408rag.generation.service import DEFAULT_QA_CONFIG

    out = {"qa": DEFAULT_QA_CONFIG.label()}
    out.update({k: v.label() for k, v in STAGES.items()})
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.api_host, port=settings.api_port)