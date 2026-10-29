"""Streamlit 演示界面。

六块：
    1. 问答 —— 提问、看流式回答、看引用了哪些章节
    2. Agent 备考助手 —— 推理链、工具调用、图解析、考点卡片
    3. 检索调试 —— 展示三路各自召回了什么、RRF 融合前后排名怎么变
    4. 考点分析 —— 基于历年真题频次的客观统计与关联
    5. 实验数据 —— 检索消融结果
    6. 对抗规划 —— 规则/LLM 实时质疑 Plan，展示修订轨迹

第 2、3、6 块是这个界面真正的价值：面试时能直观说明
「这个数字是怎么来的」「Agent 到底做了什么」「质疑是否可验证」，
而不是只给一个最终答案。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 允许直接 streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="408-RAG", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="加载检索链路…")
def get_pipeline():
    from cs408rag.retrieval.embedder import get_embedder
    from cs408rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

    get_embedder().load()
    pipe = RetrievalPipeline()
    pipe.retrieve("预热", config=RetrievalConfig(exclude_explanations=False))
    return pipe


@st.cache_data(show_spinner=False)
def load_eval_summary() -> dict | None:
    import json

    p = ROOT / "reports" / "ablation_dev.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="加载 Agent 与考点库…")
def get_runner():
    from cs408rag.agent.graph import AgentRunner
    from cs408rag.agent.tools import AgentTools
    from cs408rag.generation.service import build_exam_ref_whitelist

    tools = AgentTools(
        pipeline=get_pipeline(),
        valid_exam_refs=build_exam_ref_whitelist(),
    )
    return AgentRunner(tools=tools)


def render_images(images: list[dict], *, width: int = 420) -> None:
    """渲染教材插图。

    chunk 里存的是 image_id，这里用 image_index.json 解析到磁盘路径再展示。
    找不到的图不报错，静默跳过 —— 缺图不该把整个结果页搞崩。
    """
    from cs408rag.ingestion.image_index import image_abs_path

    shown = 0
    for im in images or []:
        p = image_abs_path(im.get("image_id", ""))
        if p is None:
            continue
        cap = im.get("caption") or ""
        st.image(str(p), caption=cap or None, width=width)
        shown += 1
    if shown == 0 and images:
        st.caption(f"（本节有 {len(images)} 张插图，但索引中未找到对应文件）")


# 对抗规划：角色 -> (图标, 说明)。颜色靠图标区分，避免依赖主题色。
ADV_NODE_LABELS = {
    "plan_gen": ("🧭 规划", "生成 / 修订 Plan"),
    "rule_guard": ("🛡 规则质疑", "6 类可验证规则，不调 LLM"),
    "llm_critic": ("🗣 LLM 质疑", "只挑规则未覆盖的问题"),
    "arbitrate": ("⚖️ 专家裁定", "真题频次 + 前置边"),
    "advance": ("📌 推进", "决定继续或退出"),
}

ADV_STOP_LABELS = {
    "converged": "✅ 收敛",
    "max_rounds": "⏹ 达到最大轮数",
    "stalled": "⏸ 修订停滞",
    "degraded": "⚠️ 降级为规则",
    "budget": "⏱ 预算上限",
    "error": "❌ 执行失败",
}

ADV_TYPE_LABELS = {
    "concept": "概念",
    "calc": "计算",
    "figure": "图表",
    "cross_subject": "跨科目",
}


def _types_text(types: dict | None) -> str:
    return " / ".join(
        f"{ADV_TYPE_LABELS.get(k, k)} {v}" for k, v in (types or {}).items()
    )


def _adv_detail(entry: dict) -> str:
    node = entry.get("node")
    if node == "plan_gen":
        plan_src = entry.get("plan_source") or (
            "llm_enriched" if entry.get("used_llm") else "deterministic"
        )
        guide_src = entry.get("guide_source") or (
            "llm" if entry.get("used_llm") else "template"
        )
        return (
            f"第 {entry.get('version')} 版 · 结构 {plan_src} · 文案 {guide_src}"
            f" · {entry.get('took_ms', 0)}ms"
        )
    if node == "rule_guard":
        return (
            f"{entry.get('n_challenges', 0)} 条挑战"
            f"（high {entry.get('n_high', 0)}） · {entry.get('took_ms', 0)}ms"
        )
    if node == "llm_critic":
        if entry.get("skipped"):
            return "未启用 LLM，已跳过"
        state = "成功" if entry.get("ok") else "失败 / 空回答"
        return f"新增 {entry.get('n_new', 0)} 条 · {state} · {entry.get('took_ms', 0)}ms"
    if node == "arbitrate":
        return (
            f"{entry.get('n_conflicts', 0)} 条争议 → "
            f"{entry.get('n_verdicts', 0)} 条裁定 · {entry.get('took_ms', 0)}ms"
        )
    if node == "advance":
        stop = ADV_STOP_LABELS.get(entry.get("stop_reason") or "", "继续修订")
        return (
            f"第 {entry.get('round')} 轮 · 残留 high {entry.get('open_high', 0)} · "
            f"新增 {entry.get('new_high', 0)} · 解决 {entry.get('resolved', 0)} → {stop}"
        )
    return ""


def render_adv_timeline(entries: list[dict], placeholder) -> None:
    """把状态机的每一步画成按角色区分的 st.status。"""
    with placeholder.container():
        for entry in entries:
            icon, desc = ADV_NODE_LABELS.get(entry.get("node"), ("•", entry.get("node") or ""))
            detail = _adv_detail(entry)
            # 计数写进标题：st.status 折叠时 caption 不可见，时间线会退化成
            # 一排没有信息的「已完成」。
            with st.status(f"{icon} {desc} · {detail}", state="complete", expanded=False):
                st.caption(detail)


# ---------------------------------------------------------------------------

st.title("📚 408-RAG")
st.caption(
    "面向考研 408 的真题检索问答系统 · "
    "720 道真题（2009–2026）· 5296 个教材 chunk · 6182 个知识点节点"
)

tab_qa, tab_agent, tab_debug, tab_exam, tab_eval, tab_adv = st.tabs(
    [
        "💬 问答",
        "🤖 Agent 备考助手",
        "🔍 检索调试",
        "📊 考点分析",
        "📈 实验数据",
        "🛡 对抗规划",
    ]
)

# ---------------------------------------------------------------------------
# 问答
# ---------------------------------------------------------------------------
with tab_qa:
    from cs408rag.config import settings
    from cs408rag.generation.prompt import build_context
    from cs408rag.generation.service import DEFAULT_QA_CONFIG

    col1, col2 = st.columns([3, 1])
    with col1:
        q = st.text_input("提问", value="采用组相联映射时主存地址如何划分？")
    with col2:
        subject = st.selectbox("科目", ["不限", "数据结构", "组成原理", "操作系统", "计算机网络"])

    if st.button("提问", type="primary"):
        pipe = get_pipeline()
        subj = None if subject == "不限" else subject
        with st.spinner("检索中…"):
            r = pipe.retrieve(q, subject=subj, config=DEFAULT_QA_CONFIG)
            blocks = build_context(r.top_hits(8))

        st.subheader("引用来源")
        for b in blocks:
            n_img = len(b.images)
            label = f"[{b.index}] {b.citation}" + (f"  🖼 {n_img}" if n_img else "")
            with st.expander(label, expanded=(n_img > 0)):
                st.write(b.text)
                if n_img:
                    st.markdown(f"**教材插图（{n_img} 张）**")
                    render_images(b.images)

        st.subheader("回答")
        import os

        has_key = bool(os.getenv("OPENAI_API_KEY") or settings.openai_api_key.strip())
        if not has_key or settings.openai_api_key.startswith("sk-your"):
            st.warning(
                "未配置 LLM API key，跳过生成。**检索部分完全可用** —— "
                "上面的引用来源就是检索链路的结果。\n\n"
                "配置方法：`cp .env.example .env` 后填入 `OPENAI_API_KEY`。"
            )
        else:
            from cs408rag.generation.llm import get_llm
            from cs408rag.generation.prompt import qa_messages

            # 首字延迟实测 15–45 秒（取决于模型与 max_tokens），
            # 期间一个 token 都不会到。若什么都不显示，界面看起来像卡死。
            # 所以先给一个明确的等待提示，收到第一个字后再切成流式渲染。
            ph = st.empty()
            ph.info("⏳ 模型生成中…首字延迟可能 15–45 秒，检索结果已在上面")
            buf = ""
            started = False
            for piece in get_llm().stream(qa_messages(q, blocks)):
                if not started:
                    started = True
                    ph.empty()
                buf += piece
                ph.markdown(buf + "▌")
            ph.markdown(buf if buf else "（模型未返回内容）")

# ---------------------------------------------------------------------------
# Agent 备考助手
# ---------------------------------------------------------------------------
with tab_agent:
    import json

    st.subheader("Agent 编排：检索 + 计算器 + 图形解析 + 考点统计")
    st.caption(
        "概念题走检索，计算题调用计算器，图表题做 SVG 拓扑重建，"
        "备考咨询查真实真题频次。下面是完整的推理链与工具调用记录。"
    )

    mode = st.radio(
        "模式", ["开放问答", "选择题"], horizontal=True, key="agent_mode"
    )
    aq = st.text_input(
        "问题", value="死锁这个考点历年考过几次？和哪些知识点关联？", key="agent_q"
    )
    ac1, ac2 = st.columns([2, 1])
    with ac1:
        aqid = st.text_input(
            "真题编号（可选，如 2009-04；给了会启用图形解析）", value="", key="agent_qid"
        )
    with ac2:
        asubj = st.selectbox(
            "科目",
            ["不限", "数据结构", "组成原理", "操作系统", "计算机网络"],
            key="agent_subj",
        )

    opts: dict[str, str] = {}
    if mode == "选择题":
        st.markdown("**选项**")
        oc = st.columns(4)
        for i, letter in enumerate("ABCD"):
            with oc[i]:
                opts[letter] = st.text_input(letter, key=f"opt_{letter}")

    if st.button("运行 Agent", type="primary", key="agent_run"):
        runner = get_runner()
        subj = None if asubj == "不限" else asubj
        with st.spinner("Agent 执行中（含 LLM 生成，通常 10–60 秒）…"):
            res = runner.run(
                aq,
                options=opts if mode == "选择题" else None,
                subject=subj,
                qid=aqid.strip() or None,
                mode="quiz" if mode == "选择题" else "qa",
            )
        d = res.to_dict()

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("路由", d["route"])
        k2.metric("耗时", f"{d['took_ms'] / 1000:.1f}s")
        k3.metric("工具调用", d["n_tool_calls"])
        k4.metric("选择题答案", d["choice"] or "—")

        st.caption(f"路由理由：{d['route_reason']}")

        st.markdown("**推理链**")
        st.dataframe(
            [
                {
                    "步骤": s["node"],
                    "耗时(ms)": s["took_ms"],
                    "详情": json.dumps(s.get("detail", {}), ensure_ascii=False)[:160],
                }
                for s in d["steps"]
            ],
            use_container_width=True,
        )

        st.markdown("**工具调用**")
        for t in d["tool_calls"]:
            icon = "✅" if t["ok"] else "❌"
            with st.expander(
                f"{icon} {t['tool']} · {t['took_ms']:.0f}ms · {t['summary']}"
            ):
                st.json(t.get("data"))

        st.markdown("**回答**")
        st.markdown(d["answer"] or f"（无回答：{d['error'] or '未知原因'}）")

        v = d.get("verification") or {}
        if v:
            if v.get("ok"):
                st.success(
                    f"引用校验通过 · 引用精度 {v.get('citation_precision', 0):.0%} · "
                    "未发现押题表述"
                )
            else:
                st.warning(
                    f"校验告警：无效真题编号 {v.get('invalid_exam_refs') or '无'}；"
                    f"押题表述 {v.get('predictions') or '无'}"
                )

    st.divider()
    st.markdown("**单独查看某题的图形解析**")
    fq = st.text_input("真题编号", value="2009-04", key="fig_qid")
    if st.button("解析图形", key="fig_run"):
        from cs408rag.figures.read_figure import read_figure

        fr = read_figure(fq.strip())
        st.markdown(f"层级：**{fr.tier}** · 可用：**{fr.available}**")
        st.text(fr.description)
        if fr.note:
            st.info(fr.note)

# ---------------------------------------------------------------------------
# 检索调试
# ---------------------------------------------------------------------------
with tab_debug:
    st.subheader("各路召回对照")
    st.caption(
        "同一问题下，dense / sparse / BM25（可选图谱）各自召回了什么，"
        "以及融合前后的排名变化。生产配置默认不含图谱，可勾选对照。"
    )
    dq = st.text_input("调试查询", value="KMP 算法中 next 数组怎么求", key="dq")
    include_graph = st.checkbox("包含图谱路（生产默认关闭）", value=False, key="dbg_graph")

    if st.button("运行", key="run_debug"):
        from dataclasses import replace

        from cs408rag.generation.service import DEFAULT_QA_CONFIG

        pipe = get_pipeline()
        cfg = replace(DEFAULT_QA_CONFIG, use_graph=include_graph, exclude_explanations=False)
        r = pipe.retrieve(dq, config=cfg)

        route_names = ["dense", "sparse"] + (["graph"] if include_graph else ["bm25"])
        cols = st.columns(len(route_names))
        for col, name in zip(cols, route_names, strict=True):
            with col:
                tr = r.routes.get(name)
                st.markdown(f"**{name}**")
                if not tr:
                    st.caption("未启用")
                    continue
                st.caption(f"{tr.took_ms:.0f}ms · 召回 {len(tr.top_ids)}")
                for cid in tr.top_ids[:6]:
                    h = r.hits_by_id.get(cid)
                    if h:
                        snippet = h.text.replace("\n", " ")[:26]
                        st.markdown(
                            f"- `{cid[:8]}` {' > '.join(h.chapter_path)[:42]} · {snippet}"
                        )

        st.caption(
            "同一章节路径可能出现多次——它们是**同一小节被切出的不同 chunk**，"
            "chunk_id 与文本片段不同；三路各自的排名也不同。"
        )

        st.divider()
        st.markdown("**RRF 融合后 top-10**")
        for f in r.fused[:10]:
            h = r.hits_by_id.get(f.doc_id)
            if h:
                snippet = h.text.replace("\n", " ")[:22]
                st.markdown(
                    f"- `{f.score:.5f}` `{f.doc_id[:8]}` 来源 `{f.ranks}` — "
                    f"{' > '.join(h.chapter_path)[:48]} · {snippet}"
                )

        if r.rerank_applied:
            st.success("已应用交叉编码器重排")

# ---------------------------------------------------------------------------
# 考点分析
# ---------------------------------------------------------------------------
with tab_exam:
    from cs408rag.agent.exam_analysis import get_analyzer

    st.subheader("考点频次")
    st.caption(
        "统计自 720 道历年真题的标签，**是客观计数不是模型生成的判断**。"
        "本系统只归纳「历年实际考了什么」，不做任何考试预测。"
    )
    analyzer = get_analyzer()
    dist = analyzer.tag_distribution()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("真题", dist["n_questions"])
    c2.metric("考点标签", dist["n_tags"])
    c3.metric("含图题", dist["by_type"].get("figure", 0))
    c4.metric("计算题", dist["by_type"].get("calc", 0))

    top_n = st.slider("显示前 N 个考点", 10, 50, 20)
    items = analyzer.tag_freq
    st.bar_chart({"考查次数": {k: items[k] for k in list(items)[:top_n]}})

    st.markdown("**查某个考点**")
    kw = st.text_input("考点关键词（支持主题词，如「死锁」会汇总其下全部标签）", value="死锁")
    if kw:
        s = analyzer.stats(kw, example_limit=10)
        if not s.total:
            st.warning(f"没有匹配到「{kw}」的真题考点。")
        else:
            st.write(
                f"**{s.tag}**：历年被考查 **{s.total}** 次 · "
                f"匹配标签 {len(s.matched_tags)} 个（{'、'.join(s.matched_tags)}）"
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                st.markdown("**年份分布**")
                st.bar_chart({"次数": {str(y): n for y, n in sorted(s.by_year.items())}})
            with cc2:
                st.markdown("**题型分布**")
                st.bar_chart({"次数": s.by_type})

            rel = analyzer.related_tags(kw, top_n=8)
            if rel:
                st.markdown("**关联考点（同题共现，证据为真题编号）**")
                st.dataframe(
                    [
                        {
                            "关联考点": r["tag"],
                            "共现次数": r["cooccurrence"],
                            "证据真题": "、".join(r["evidence"]),
                        }
                        for r in rel
                    ],
                    use_container_width=True,
                )

            st.markdown("**历年真题原题**")
            for q in analyzer.questions_for_tag(kw, limit=10):
                with st.expander(
                    f"[{q['qid']}] {q['subject']} · {q['type']} · 答案 {q['answer']}"
                ):
                    st.write(q["question"])
                    for k, v in (q.get("options") or {}).items():
                        st.markdown(f"- **{k}.** {v}")
                    if q.get("explanation"):
                        st.caption(q["explanation"])

# ---------------------------------------------------------------------------
# 实验数据
# ---------------------------------------------------------------------------
with tab_eval:
    st.subheader("检索消融实验")
    data = load_eval_summary()
    if not data:
        st.info(
            "还没有实验数据。运行：`uv run --no-sync python -m cs408rag.eval.run_eval --split dev`"
        )
    else:
        st.caption(f"数据集：{data.get('split')} · {data.get('n_queries')} 题")
        st.markdown(
            "**评测口径**：ground truth 由 `题目 tags → 知识点 → 教材小节 → 该节全部 chunk` "
            "构造，只取最匹配的**一个小节**。题目只在该题科目内匹配，并排除"
            "「答案与解析 / 试题精选 / 复习提示」类小节（它们逐字含题干，会自匹配）。"
        )

        stages = data.get("stages", [])
        if stages:
            st.markdown("**检索链路递增消融**")
            st.dataframe(stages, width="stretch")

            st.markdown(
                """
#### 表格口径

- **Recall@10**：命中的相关 chunk 数 / 该题全部相关 chunk 数。
- **Accuracy@10（题级命中）**：前 10 条里只要命中该题 GT 小节的任意 chunk 就算 1。
  GT 小节常被切成 2–5 块，所以 Accuracy 通常高于 Recall。
- **Precision@10** 天然偏低：GT 只是一个小节，10 条里通常只有 1–2 条来自该节。
- **MRR / nDCG**：前者看第一个相关项排多前，后者看整体排序质量。
"""
            )

            stage_notes = {
                "S0_dense": (
                    "仅稠密向量检索，基线",
                    "作为所有增量的参照；它代表「只做语义召回」的上限。",
                ),
                "S1_hybrid": (
                    "加入 BGE-M3 sparse + 图谱，RRF 融合",
                    "主要增益来自 sparse 的术语精确匹配（+4.3pp）；图谱贡献很小，"
                    "后面按题型拆开时甚至是负的。",
                ),
                "S1b_bm25": (
                    "在 S1 上再加原生 BM25",
                    "全题等权融合时 R@10 略降 0.56pp，但按题型选择后概念 +1.0pp、"
                    "图表 +3.9pp，所以最终配置是「按题型决定是否启用」，不是全开。",
                ),
                "S2_rerank": (
                    "加入 bge-reranker-v2-m3 交叉编码器重排",
                    "负收益：R@10 −7.2pp、延迟 +246ms。固定候选集做纯重排也只 +0.0013 MRR，"
                    "说明是重排器偏好与「概念讲解章节」这个相关性定义不一致。",
                ),
                "S3_rewrite": (
                    "加入无 LLM 的轻量查询改写",
                    "几乎无变化（+0.15pp），因为改写只做关键词分解，没有引入新信息。",
                ),
                "S4_filter": (
                    "加入按科目过滤",
                    "增量为 0：dense+sparse 本身已很少跨科误召回，过滤没有可去掉的噪声。",
                ),
                "PROD": (
                    "最终生产配置：dense+sparse+BM25（按题型选）、去掉图谱、"
                    "zscore 融合、小节去重、返回 10 条",
                    "Accuracy@10 最高（0.668）、P50 最低（50ms），所以选它。"
                    "它不是递增消融的一环，而是按题型消融后的组合。",
                ),
            }
            st.markdown("#### 逐行解读")
            for r in stages:
                name = r.get("stage", "")
                what, concl = stage_notes.get(name, ("", ""))
                st.markdown(f"**{name}** — {what}")
                st.markdown(
                    f"结果：Recall@10 **{r.get('Recall@10', 0):.4f}** · "
                    f"Accuracy@10 **{r.get('Accuracy@10', 0):.4f}** · "
                    f"MRR {r.get('MRR@10', 0):.4f} · P50 {r.get('P50_ms', 0):.0f}ms"
                )
                st.markdown(f"结论：{concl}")

            st.success(
                "**最终为什么选 PROD**：问答场景要的是「有没有拿到正确小节」，"
                "PROD 的 Accuracy@10 最高、延迟最低；如果产品改成「必须给多段原文」，"
                "可以换回 RRF（chunk 级 Recall@10 更高），但题级命中会下降。"
            )

        fusions = data.get("fusions", [])
        if fusions:
            st.markdown("**融合策略对比**")
            st.dataframe(fusions, width="stretch")
            fusion_notes = {
                "单路 dense（对照）": (
                    "只用稠密向量，不融合",
                    "R@10 最低（0.4731），但延迟也最低；用来回答「融合是否值得」。",
                ),
                "RRF (k=60)": (
                    "只用各路排名做 RRF，k=60",
                    "chunk 级 R@10 最高（0.5161）；对分数量纲不敏感，是纯检索场景的最优。",
                ),
                "RRF (k=20)": (
                    "RRF 的 k 值敏感性对照",
                    "与 k=60 几乎相同（0.5160），说明结果对该参数不敏感。",
                ),
                "加权融合 (minmax)": (
                    "先 minmax 归一化分数再加权",
                    "明显更差（0.4749，−3.7pp）：归一化后每路第 1 名都是 1.0，"
                    "强命中和弱命中被拉平。",
                ),
                "加权融合 (zscore)": (
                    "先 zscore 归一化分数再加权",
                    "比 minmax 稳（0.5042），但仍略低于 RRF；MRR 反而更高。",
                ),
                "zscore + 节去重(1)": (
                    "zscore 融合后，每个小节只保留 1 条",
                    "chunk 级 R@10 降到 0.3546，但题级 Accuracy@10 最高（0.6327）——"
                    "top-10 覆盖更多不同小节。生产以它为融合基础。",
                ),
                "zscore + 节去重(2)": (
                    "每个小节保留 2 条",
                    "折中：R@10 回升到 0.4537，但 Accuracy@10 降到 0.5819。",
                ),
            }
            st.markdown("#### 逐行解读")
            for r in fusions:
                name = r.get("stage", "")
                what, concl = fusion_notes.get(name, ("", ""))
                st.markdown(f"**{name}** — {what}")
                st.markdown(
                    f"结果：Recall@10 **{r.get('Recall@10', 0):.4f}** · "
                    f"Accuracy@10 **{r.get('Accuracy@10', 0):.4f}** · "
                    f"MRR {r.get('MRR@10', 0):.4f} · P50 {r.get('P50_ms', 0):.0f}ms"
                )
                st.markdown(f"结论：{concl}")

            st.success(
                "**融合层的最终选择**：chunk 级召回最高的是 RRF；但问答要的是题级命中，"
                "所以生产用 zscore + 节去重，再按题型决定是否加 BM25、并去掉图谱。"
                "即链路表的 `PROD` 行。"
            )

        rescue = data.get("graph_rescue")
        if rescue:
            st.markdown("**图谱路长尾救援**")
            st.json(rescue)
            st.markdown(
                f"""
#### 信息解读

- `hard_queries`：dense 与 sparse 的 top-k **都没有命中**任何相关 chunk 的题（{rescue.get('hard_queries', 0)} 道）。
- `rescued_by_graph`：这些难题里，图谱路命中的数量（{rescue.get('rescued_by_graph', 0)} 道）。
- `rescue_rate = rescued / hard`（{rescue.get('rescue_rate', 0):.1%}）。

**结论：图谱作为通用召回路的增量很小**，且按题型拆开后对三类题都是负贡献，
所以生产配置已移除图谱。它真正的价值在二期考点分析：
知识点画像、同题考点共现、学习路径——这些不依赖检索召回。
"""
            )

# ---------------------------------------------------------------------------
# 对抗规划
# ---------------------------------------------------------------------------
with tab_adv:
    import json
    import math
    import time
    from datetime import date, timedelta

    from cs408rag.adversarial.graph import PlanResult, PlanRunner
    from cs408rag.adversarial.planner import SUBJECTS, next_exam_date, normalize_concerns
    from cs408rag.adversarial.schema import plan_diff

    ADV_SUBJECTS = SUBJECTS
    ADV_PROGRESS = ["未开始", "一轮中", "一轮完", "二轮中", "二轮完"]
    ADV_CONCERN_TAGS = [
        "基础差 / 跨考，看到代码就慌",
        "时间不够，每天只能学 3–4 小时",
        "做题慢，计算题总算错",
        "某个知识点总错（可写具体知识点）",
        "真题正确率低，不知道错在哪",
        "记不住，背了又忘",
        "不知道从哪开始",
    ]

    st.subheader("对抗性 Plan Agent：日期驱动 + 高性价比考点")
    st.caption(
        "结构由真实真题统计计算（阶段 / 科目份额 / 考点 ROI / 周任务），"
        "LLM 只把方法建议写顺。6 类挑战由纯函数规则判定，"
        "是否继续修订由状态机决定，而模型没有「再聊聊」的权限。"
    )

    # 时间相关放在 form 外：改动后立即刷新预算，不必等提交
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        adv_exam = st.date_input(
            "考试日期",
            value=next_exam_date(),
            min_value=date.today() + timedelta(days=7),
            key="adv_exam",
            help="默认下一次考研（12 月 25 日前最后一个周六）",
        )
    with tc2:
        adv_daily = st.number_input(
            "每天 408 小时", 1.0, 12.0, 4.0, 0.5, key="adv_daily"
        )
    with tc3:
        adv_days = st.slider("每周学习天数", 1, 7, 6, key="adv_days")

    _days = max(1, (adv_exam - date.today()).days)
    _weeks = max(1, math.ceil(_days / 7))
    _weekly = float(adv_daily) * int(adv_days)
    st.info(
        f"距 {adv_exam} 还有 **{_days} 天 ≈ {_weeks} 周**；"
        f"每天 {adv_daily:g}h × 每周 {adv_days} 天 = **{_weekly:g}h/周**，"
        f"总可用约 **{_weeks * _weekly:g}h**。"
    )

    with st.form("adv_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            adv_scope_label = st.radio(
                "规划范围", ["全科 408", "单科补弱"], horizontal=True, key="adv_scope"
            )
        with c2:
            adv_weak = st.multiselect(
                "薄弱科目（建议 1–2 个）",
                ADV_SUBJECTS,
                default=["操作系统"],
                max_selections=2,
                key="adv_weak",
            )
        with c3:
            adv_target = st.number_input("目标分", 60, 150, 115, key="adv_target")

        st.markdown("**四科当前进度**")
        pc = st.columns(4)
        adv_progress: dict[str, str] = {}
        for i, subj in enumerate(ADV_SUBJECTS):
            with pc[i]:
                adv_progress[subj] = st.selectbox(
                    subj, ADV_PROGRESS, index=1, key=f"adv_prog_{subj}"
                )

        st.markdown("**你的备考问题**")
        adv_concern_tags = st.multiselect(
            "常见问题（可多选）",
            ADV_CONCERN_TAGS,
            key="adv_concern_tags",
            help="会与下面的自由文本合并；每条问题都会在 Plan 里给出对应动作",
        )
        adv_concern_text = st.text_area(
            "也可以直接写（每行一条，最多 8 条）",
            placeholder="例：基础差，看到代码就慌\n真题正确率只有 50%\n死锁总是错",
            height=110,
            key="adv_concern_text",
        )

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            adv_rounds = st.number_input("最大修订轮数", 0, 2, 2, key="adv_rounds")
        with fc2:
            adv_llm = st.checkbox(
                "启用 LLM 文案（可选）",
                value=False,
                key="adv_llm",
                help="结构 Plan 不依赖 LLM；勾选后只多一次调用润色经验贴，限流时自动退回模板",
            )
        with fc3:
            st.caption("结构 Plan 不依赖 LLM；\nLLM 失败只退回模板文案。")

        submitted = st.form_submit_button("生成并质疑", type="primary")

    if submitted:
        concerns = normalize_concerns(adv_concern_text)
        for tag in adv_concern_tags:
            if tag and tag not in concerns:
                concerns.append(tag)
        scope = "full_408" if adv_scope_label == "全科 408" else "single_subject"
        weak = list(adv_weak)
        if scope == "single_subject" and not weak:
            weak = ["操作系统"]
        goal = {
            "scope": scope,
            "weak_subjects": weak,
            "start_date": date.today().isoformat(),
            "exam_date": adv_exam.isoformat(),
            "daily_hours": float(adv_daily),
            "study_days_per_week": int(adv_days),
            "target_score": int(adv_target),
            "progress": adv_progress,
            "concerns": concerns[:8],
        }
        runner = PlanRunner(use_llm=bool(adv_llm), max_rounds=int(adv_rounds))
        placeholder = st.empty()
        progress = st.progress(0.0, text="规划中…")
        entries: list[dict] = []
        last: dict = {}
        before = runner.stats_snapshot()
        t0 = time.perf_counter()
        try:
            for snapshot in runner.iter_run(goal):
                last = snapshot
                timeline = snapshot.get("timeline") or []
                if len(timeline) != len(entries):
                    entries = list(timeline)
                    render_adv_timeline(entries, placeholder)
                    progress.progress(
                        min(0.95, len(entries) / 8),
                        text=f"已完成 {len(entries)} 个节点…",
                    )
        finally:
            progress.empty()
        after = runner.stats_snapshot()

        result = PlanResult(
            goal=goal,
            plan=last.get("plan") or {},
            plan_history=last.get("plan_history") or [],
            challenges=last.get("challenges") or [],
            challenge_history=last.get("challenge_history") or last.get("challenges") or [],
            timeline=last.get("timeline") or [],
            stop_reason=last.get("stop_reason") or "",
            version=int(last.get("version") or 0),
            rounds=int(last.get("round") or 0),
            llm_calls=int(last.get("llm_calls") or 0),
            llm_errors=int(last.get("llm_errors") or 0),
            planner_fallback=bool(last.get("planner_fallback")),
            plan_source=str(last.get("plan_source") or "deterministic"),
            guide_source=str(last.get("guide_source") or "template"),
            role_models=runner.role_models,
            tokens={
                "prompt": after["prompt"] - before["prompt"],
                "completion": after["completion"] - before["completion"],
            },
            error=str(last.get("error") or ""),
            took_ms=(time.perf_counter() - t0) * 1000,
        )
        st.session_state["adv_result"] = result.to_dict()

    result = st.session_state.get("adv_result")
    if not result:
        st.info(
            "填写条件后点击「生成并质疑」。结构 Plan 会立即生成；"
            "启用 LLM 时只多花一次调用来润色经验贴文案。"
        )
    else:
        plan = result.get("plan") or {}
        g = plan.get("goal") or {}
        stop = result.get("stop_reason") or ""
        stages = plan.get("stages") or []
        kps = plan.get("priority_kps") or []
        actions = plan.get("concern_actions") or []
        weekly = plan.get("weekly_plan") or []
        guide = plan.get("guide") or {}

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("退出原因", ADV_STOP_LABELS.get(stop, stop or "—"))
        m2.metric("剩余周数", g.get("remaining_weeks", "—"))
        m3.metric("总可用小时", f"{g.get('total_hours', 0):g}")
        m4.metric("Plan 版本", result.get("version", 0))
        m5.metric("残留 high", len(result.get("open_high") or []))
        m6.metric("耗时", f"{(result.get('took_ms') or 0) / 1000:.1f}s")

        st.caption(
            f"每天 {g.get('daily_hours', '?')}h × {g.get('study_days_per_week', '?')} 天 "
            f"= {g.get('weekly_hours', '?')}h/周 ｜ 考试 {g.get('exam_date')} ｜ "
            f"范围 {g.get('scope')} ｜ 结构 {result.get('plan_source')} ｜ "
            f"文案 {result.get('guide_source')} ｜ LLM {result.get('llm_calls', 0)} 次 ｜ "
            f"token {(result.get('tokens') or {}).get('prompt', 0)} + "
            f"{(result.get('tokens') or {}).get('completion', 0)}"
        )
        if result.get("error"):
            st.error(result["error"])
        if result.get("planner_fallback"):
            st.warning("结构 Plan 生成失败，已使用兜底；本次结果不可信。")
        if result.get("guide_source") == "template" and result.get("llm_calls"):
            st.caption("LLM 文案未通过校验（空回答 / 含资料名 / 结构不合法），已使用模板文案。")

        st.markdown("**阶段路线**")
        if stages:
            st.dataframe(
                [
                    {
                        "阶段": s.get("name"),
                        "开始": s.get("start"),
                        "结束": s.get("end"),
                        "周数": s.get("weeks"),
                        "重点": "、".join(s.get("focus") or []),
                        "交付物": "；".join(s.get("deliverables") or []),
                        "检查点": s.get("checkpoint"),
                    }
                    for s in stages
                ],
                width="stretch",
                hide_index=True,
            )

        t1, t2, t3, t4, t5 = st.tabs(
            ["高性价比知识点", "你的问题 → 计划", "周计划", "经验贴 / 方法", "修订与挑战"]
        )

        with t1:
            st.caption(
                "ROI 是「频次 + 近 5 年 + 计算/图表占比 + 学习成本」的启发式排序，"
                "不是押题。展开可看教材小节与子考点。"
            )
            if kps:
                subj_filter = st.multiselect(
                    "科目筛选",
                    ADV_SUBJECTS,
                    default=ADV_SUBJECTS,
                    key="adv_kp_subj",
                )
                shown_kps = [
                    k for k in kps if not subj_filter or k.get("subject") in subj_filter
                ]
                st.dataframe(
                    [
                        {
                            "科目": k.get("subject"),
                            "考点": k.get("tag"),
                            "频次": k.get("frequency"),
                            "近5年": k.get("recent_5y"),
                            "题型": _types_text(k.get("types")),
                            "ROI": f"{k.get('roi_label')} ({k.get('roi'):.2f})",
                            "优先学": "、".join(k.get("must_learn") or []),
                            "证据": "、".join(k.get("evidence") or []),
                        }
                        for k in shown_kps
                    ],
                    width="stretch",
                    hide_index=True,
                )
                for k in shown_kps[:15]:
                    with st.expander(
                        f"[{k.get('roi_label')}] {k.get('subject')} · {k.get('tag')} "
                        f"· 共 {k.get('frequency')} 次 / 近5年 {k.get('recent_5y')} 次"
                    ):
                        st.write(k.get("why"))
                        st.caption(
                            "教材小节：" + ("、".join(k.get("kp_titles") or []) or "—")
                        )
                        if k.get("must_learn"):
                            st.markdown("**优先学**：" + "、".join(k["must_learn"]))
                        if k.get("later"):
                            st.markdown("**可后置**：" + "、".join(k["later"]))
                        if k.get("subtopics"):
                            st.dataframe(
                                [
                                    {
                                        "子考点": s.get("tag"),
                                        "频次": s.get("frequency"),
                                        "题型": _types_text(s.get("types")),
                                    }
                                    for s in k["subtopics"]
                                ],
                                width="stretch",
                                hide_index=True,
                            )
            else:
                st.caption("没有选中考点。")

        with t2:
            if actions:
                st.dataframe(
                    [
                        {
                            "你的问题": a.get("concern"),
                            "诊断": a.get("diagnosis"),
                            "计划怎么解决": a.get("action"),
                            "阶段": a.get("stage"),
                            "检查点": a.get("checkpoint"),
                        }
                        for a in actions
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("没有填写备考问题；可以在表单里自由描述，系统会逐条映射成动作。")

        with t3:
            if weekly:
                st.caption("每周总小时 = 表单预算；任务都带「学什么 / 怎么学 / 验收标准」。")
                st.dataframe(
                    [
                        {
                            "周": w.get("week"),
                            "阶段": w.get("stage"),
                            "本周主攻": "、".join(w.get("focus") or []),
                            "科目": t.get("subject"),
                            "小时": t.get("hours"),
                            "知识点": "、".join(t.get("kps") or []),
                            "学什么": t.get("content"),
                            "怎么学": t.get("method"),
                            "验收标准": t.get("acceptance"),
                        }
                        for w in weekly
                        for t in (w.get("tasks") or [])
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.caption("没有周计划。")

        with t4:
            if guide.get("overview"):
                st.markdown(f"**总览**\n\n{guide['overview']}")
            if guide.get("note"):
                st.caption(guide["note"])
            if guide.get("subject_order"):
                st.markdown(f"**起步顺序**\n\n{guide['subject_order']}")
            pa = guide.get("phase_advice") or []
            if pa:
                st.markdown("**阶段建议**")
                for p in pa:
                    st.markdown(f"- **{p.get('stage')}**：{p.get('advice')}")
            ss = guide.get("subject_strategy") or []
            if ss:
                st.markdown("**分科策略**")
                for s in ss:
                    st.markdown(
                        f"- **{s.get('subject')}**：{s.get('strategy')}"
                        + (
                            "｜注意：" + "；".join(s.get("pitfalls") or [])
                            if s.get("pitfalls")
                            else ""
                        )
                    )
            if guide.get("weekly_routine"):
                st.markdown(f"**每周节奏**\n\n{guide['weekly_routine']}")
            if guide.get("pitfalls"):
                st.markdown("**常见坑**")
                for p in guide["pitfalls"]:
                    st.markdown(f"- {p}")
            if guide.get("checkpoints"):
                st.markdown("**检查点**")
                for p in guide["checkpoints"]:
                    st.markdown(f"- {p}")
            st.caption("按你的选择：经验贴只写方法，不推荐教材 / 网课 / 老师。")

        with t5:
            challenges = result.get("challenge_history") or result.get("challenges") or []
            open_n = sum(1 for c in challenges if c.get("status", "open") == "open")
            resolved_n = sum(1 for c in challenges if c.get("status") == "resolved")
            st.caption(
                f"共 {len(challenges)} 条挑战 · 已解决 {resolved_n} · 仍开放 {open_n}"
                "（已解决的挑战保留在轨迹里）"
            )
            rule_labels = {
                "fabricated_ref": "编造真题编号",
                "time_overrun": "周学时超支",
                "missing_high_freq_kp": "漏掉高频考点",
                "prerequisite_order": "前置顺序颠倒",
                "uncited_priority": "高权重无引用",
                "weak_subject_underweight": "薄弱科目占比不足",
                "llm_critic": "LLM 主观挑战",
            }
            if challenges:
                sev = st.multiselect(
                    "按严重度筛选",
                    ["high", "medium", "low"],
                    default=["high", "medium", "low"],
                    key="adv_sev",
                )
                shown = [c for c in challenges if c.get("severity") in sev]
                st.dataframe(
                    [
                        {
                            "类型": rule_labels.get(c.get("rule"), c.get("rule")),
                            "严重度": c.get("severity"),
                            "位置": c.get("locator") or c.get("evidence_ref") or "",
                            "问题": c.get("message"),
                            "状态": c.get("status", "open"),
                            "裁定": c.get("verdict") or "",
                        }
                        for c in shown
                    ],
                    width="stretch",
                    hide_index=True,
                )
                for c in shown[:20]:
                    with st.expander(
                        f"[{c.get('severity')}] {rule_labels.get(c.get('rule'), c.get('rule'))}"
                        f" · {c.get('locator') or c.get('evidence_ref') or ''}"
                    ):
                        st.write(c.get("message"))
                        if c.get("evidence"):
                            st.json(c["evidence"])
                        if c.get("evidence_ref"):
                            st.caption(f"证据路径：`{c['evidence_ref']}`")
                        if c.get("fix"):
                            st.caption(f"建议：{c['fix']}")
            else:
                st.success("没有挑战。")

            history = result.get("plan_history") or []
            st.markdown("**修订 diff**")
            if len(history) >= 2:
                st.code(plan_diff(history[0], history[-1]), language="diff")
            else:
                st.caption("只有 1 版 Plan，没有可展示的修订。")

        st.download_button(
            "下载完整结果 JSON",
            data=json.dumps(result, ensure_ascii=False, indent=2),
            file_name="adversarial_plan.json",
            mime="application/json",
        )
