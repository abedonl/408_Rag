"""Streamlit 演示界面：单页 Agent 中枢。

聊天是唯一主入口（13 个确定性工具：教材检索 / 知识图谱 / 考点统计 / 真题检索 /
关联考点 / 学习路径 / 图形解析 / 计算器 / 答案校验 / 生成规划 / 画像 / 判分）。
看板与图表内嵌在会话流里，与对话联动：
    📅 我的规划执行 —— 侧栏入口，点击弹出：阶段路线 / 周计划 /
       HITL 人工裁定（规划是独立记录，不进会话流）
    📊 考点图表 —— 考情类回答内嵌该考点的客观频次 / 题型分布（不是模型预测）
侧栏为会话列表 + 系统自动生成的学习总结。单轮问答页已下线：其价值是消融对照，
演示时走 API / 评测脚本（reports/ablation_dev.json）。

面试叙事：这个页面能直观说明「这个数字是怎么来的」「Agent 到底做了什么」
「计划是否可执行」，而不是只给一个最终答案。
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import streamlit as st

# 允许直接 streamlit run app/streamlit_app.py
ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for _p in (ROOT / "src", APP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import plan_ui  # noqa: E402

from cs408rag.profile.store import get_profile_store  # noqa: E402

st.set_page_config(page_title="408 备考 Agent", page_icon="🤖", layout="wide")


@st.cache_resource(show_spinner="加载检索链路…")
def get_pipeline():
    from cs408rag.retrieval.embedder import get_embedder
    from cs408rag.retrieval.pipeline import RetrievalConfig, RetrievalPipeline

    get_embedder().load()
    pipe = RetrievalPipeline()
    pipe.retrieve("预热", config=RetrievalConfig(exclude_explanations=False))
    return pipe


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


def _recommend_questions(
    text: str, *, limit: int = 3, exclude_qid: str | None = None
) -> list[dict]:
    """按对话文本命中的考点标签推荐真题，高频考点优先。

    匹配是纯子串：两百多个标签扫一遍对话文本是毫秒级，不值得动用检索链路。
    """
    from cs408rag.agent.exam_analysis import get_analyzer

    analyzer = get_analyzer()
    out: list[dict] = []
    seen: set[str] = {exclude_qid} if exclude_qid else set()
    for tag in analyzer.tag_freq:  # tag_freq 已按考查次数降序
        if not tag or tag not in text:
            continue
        for q in analyzer.questions_for_tag(tag, limit=limit):
            if q["qid"] in seen:
                continue
            seen.add(q["qid"])
            out.append(q)
            if len(out) >= limit:
                return out
    return out


ROUTE_LABELS = {
    "concept": ("概念题", "检索教材后由 LLM 归纳，引用可逐条核对"),
    "calc": ("计算题", "路由到确定性计算器，不靠 LLM 算数"),
    "figure": ("图表题", "路由到图形解析（拓扑重建），不靠 LLM 看图"),
    "exam": ("考点统计", "路由到真题频次统计，只归纳客观计数"),
    "cross": ("跨科目题", "多路检索合并后归纳"),
    "plan": ("备考中枢", "规划/画像/判分工具编排，结果确定性成文，不经 LLM"),
    "irrelevant": ("话题外", "与 408 无关，不检索不调工具"),
}
TOOL_LABELS = {
    "search_knowledge_base": "教材检索",
    "search_knowledge_graph": "知识图谱",
    "read_figure": "图形解析",
    "calculate": "计算器",
    "get_exam_stats": "考点统计",
    "get_kp_questions": "真题检索",
    "get_related_kps": "关联考点",
    "get_learning_path": "学习路径",
    "verify_answer": "答案校验",
    "generate_plan": "生成规划",
    "read_profile": "读取画像",
    "record_checkin": "打卡记录",
    "judge_quiz": "作答判分",
}

# Agent 的每种终止原因都要有对应文案：用户需要分清「没资料」和「系统坏了」
STOP_REASON_VIEW = {
    "answered": ("✅", "已给出答案"),
    "no_evidence": ("🔍", "没检索到可用资料（不是系统故障）"),
    "tool_error": ("⚠️", "工具执行失败，本轮结果不完整"),
    "budget": ("⏱️", "Token 预算用尽，提前结束"),
    "max_rounds": ("🔁", "达到最大轮数，提前结束"),
    "clarification": ("❓", "需要补充信息才能继续"),
    "irrelevant": ("🚫", "与 408 无关，未检索资料"),
    "verify_retry": ("🔁", "重试一次后仍未通过校验"),
}


def _stop_reason_view(reason: str | None, error: str | None = None) -> tuple[str, str]:
    icon, text = STOP_REASON_VIEW.get(
        reason or "", ("•", f"未记录的终止原因（{reason or '空'}）")
    )
    if error:
        text = f"{text}：{error[:80]}"
    return icon, text


# ---------------------------------------------------------------------------

st.title("🤖 408 备考 Agent")
st.caption(
    "单页 Agent 中枢：聊天可调检索 / 图谱 / 规划 / 画像 / 判分等 13 个工具 · "
    "720 道真题（2009–2026）· 5296 个教材 chunk · 6182 个知识点节点 · "
    "规划看板与考点图表内嵌会话，随对话联动"
)

# ---------------------------------------------------------------------------
# 用户画像（侧栏）：提问/规划事件按 ?cid= 累积，SQLite 持久化
# ---------------------------------------------------------------------------
_cid_param = (st.query_params.get("cid") or "").strip()
if _cid_param:
    cid = _cid_param
else:
    # 缺省不再共用 "default"：每个浏览器会话分配独立身份并写回 URL，
    # 刷新后保持同一身份，同时避免所有匿名访问读到彼此的数据。
    cid = st.session_state.get("_anon_cid") or f"anon-{uuid.uuid4().hex[:8]}"
    st.session_state["_anon_cid"] = cid
    st.query_params["cid"] = cid


def _record_question(text: str, *, subject: str | None = None, qid: str | None = None) -> None:
    """提问落库：提取命中的考点标签一并存下，画像统计直接用。"""
    from cs408rag.agent.exam_analysis import get_analyzer
    from cs408rag.profile.stats import extract_tags
    from cs408rag.profile.store import get_profile_store

    tags = extract_tags(text, get_analyzer())
    get_profile_store().record(
        cid,
        "question",
        {"text": text[:200], "subject": subject or "", "qid": qid or "", "tags": tags},
    )


def _quiz_feedback(q: dict, mi: int) -> None:
    """推荐卡片内作答判分：点选项=提交作答。

    接 notebook judge（答错进错题本、答对移出），同时记画像 quiz 事件 ——
    答错考点是 planner 加权里优先级最高的薄弱信号。
    作答状态按 qid 存 session_state，同一题在多张卡片里共享。
    """
    from cs408rag.notebook.judge import judge
    from cs408rag.notebook.store import get_store
    from cs408rag.profile.store import get_profile_store

    opts = q.get("options") or {}
    if not q.get("answer") or not opts:
        return
    answered = st.session_state.get(f"quiz_ans_{q['qid']}")
    if answered:
        mark = "✅ 答对" if answered["correct"] else f"❌ 答错（正确答案 {answered['truth']}）"
        st.caption(f"你的作答 {answered['choice']} · {mark}，已同步错题本与画像")
        return
    item = get_store().get(cid, q["qid"])
    if item:
        st.caption(f"📕 此题在错题本中（累计错 {item['wrong_count']} 次），再挑战一次：")
    cols = st.columns(len(opts))
    for col, k in zip(cols, opts, strict=True):
        if col.button(k, key=f"quiz_btn_{mi}_{q['qid']}_{k}", width="stretch"):
            r = judge(cid, q["qid"], k)
            st.session_state[f"quiz_ans_{q['qid']}"] = {
                "choice": k,
                "correct": bool(r.get("correct")),
                "truth": r.get("truth") or "",
            }
            get_profile_store().record(
                cid,
                "quiz",
                {
                    "qid": q["qid"],
                    "subject": q.get("subject") or "",
                    "tags": q.get("tags") or [],
                    "correct": bool(r.get("correct")),
                    "choice": k,
                },
            )
            st.rerun()


def _delete_conversation(conv_id: str) -> None:
    """删除会话；删的是当前会话时，自动落到剩余最近的会话或新建一条。

    规划是独立记录（plans 表，按身份绑定），删会话不会动规划。
    """
    store = get_profile_store()
    store.delete_conversation(conv_id, client_id=cid)
    if st.session_state.get("agent_session_id") != conv_id:
        st.rerun()
    rest = [c for c in store.list_conversations(cid) if c["n_messages"] > 0]
    if rest:
        st.session_state["agent_session_id"] = rest[0]["id"]
        st.session_state["agent_chat"] = _load_chat(rest[0]["id"])
    else:
        _new_id = uuid.uuid4().hex[:12]
        store.create_conversation(cid, _new_id)
        st.session_state["agent_session_id"] = _new_id
        st.session_state["agent_chat"] = []
    st.rerun()


# 画像在顶部计算（规划看板的预填/加权要用）；渲染挪到文件末尾，
# 这样当次提问/规划的事件能立刻反映在侧栏里。
def _load_profile(events: list[dict] | None = None) -> dict:
    from cs408rag.profile.stats import build_profile
    from cs408rag.profile.store import get_profile_store

    if events is None:
        events = get_profile_store().events(cid)
    return build_profile(events)


def _summary_from_events(events: list[dict], *, period_label: str) -> str:
    """基于一个时间窗口的事件生成学习总结文案。

    优先用 LLM 生成有洞察的文案（system role + 强格式约束）；
    未配置 LLM key 时降级为统计摘要。
    """
    from cs408rag.generation.llm import get_llm
    from cs408rag.profile.stats import build_profile

    if not events:
        return f"{period_label}还没有学习记录。"
    prof = build_profile(events)
    data = {
        "period": period_label,
        "total_questions": prof["total_questions"],
        "subject_counts": prof["subject_counts"],
        "top_tags": prof["top_tags"][:8],
        "quiz": prof["quiz"],
        "recent_questions": prof["recent_questions"],
    }
    llm = get_llm()
    if llm.available:
        sys_prompt = (
            "你是 408 考研备考助手。根据以下学习数据，生成一段简洁的学习情况总结。"
            "严格按以下结构输出，不要加额外标题或解释性文字：\n"
            "**覆盖情况**：（该周期提问次数、涉及科目、核心考点）\n"
            "**薄弱信号**：（答错考点，如无则写暂无）\n"
            "**下一步建议**：（1-2 条具体建议，针对薄弱或未覆盖的科目）"
        )
        resp = llm.chat(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
            ],
            max_tokens=600,
        )
        if resp.ok and resp.text:
            return resp.text
    # 降级：统计摘要
    lines = [f"**覆盖情况**：{period_label}提问 {data['total_questions']} 次"]
    if data["subject_counts"]:
        lines.append("涉及科目：" + "、".join(f"{s}×{n}" for s, n in data["subject_counts"].items()))
    if data["top_tags"]:
        lines.append("核心考点：" + "、".join(t["tag"] for t in data["top_tags"][:6]))
    quiz = data["quiz"]
    if quiz["total"]:
        rate = quiz["correct"] / quiz["total"]
        lines.append(f"**薄弱信号**：作答 {quiz['total']} 道，正确率 {rate:.0%}")
        if quiz["wrong_tags"]:
            lines.append("答错考点：" + "、".join(t["tag"] for t in quiz["wrong_tags"][:5]))
    else:
        lines.append("**薄弱信号**：暂无答题记录")
    lines.append("**下一步建议**：针对薄弱考点多刷真题，兼顾未覆盖科目。")
    return "\n".join(lines)


def _ensure_period_summaries(cid: str, events: list[dict] | None = None) -> None:
    """检查并补齐最近一期周/月总结。

    周总结覆盖上周一 00:00 ~ 上周日 23:59（周一生成上周）；
    月总结覆盖上月 1 号 ~ 上月末（一号生成上月）。
    只补最近一期，避免历史补生成刷屏。Streamlit 无后台进程，
    所以在页面加载时触发——效果等同自动生成。

    events 可选：复用外部已拉取的事件列表，避免重复查询。
    """
    from datetime import date, timedelta

    from cs408rag.profile.store import get_profile_store

    store = get_profile_store()
    today = date.today()
    this_monday = today - timedelta(days=today.weekday())
    last_monday = this_monday - timedelta(days=7)
    last_sunday = this_monday - timedelta(days=1)
    month_first = today.replace(day=1)
    prev_month_last = month_first - timedelta(days=1)
    prev_month_first = prev_month_last.replace(day=1)

    all_events = events if events is not None else store.events(cid)

    def _gen(period_type: str, period_date: date, start: date, end: date, label: str) -> None:
        pds = period_date.isoformat()
        if store.has_summary(cid, period_type, pds):
            return
        evs = [
            e for e in all_events
            if start.isoformat() <= e["ts"][:10] <= end.isoformat()
        ]
        content = _summary_from_events(evs, period_label=label)
        store.save_summary(cid, period_type, pds, content)

    with st.spinner("生成周期学习总结…"):
        _gen(
            "weekly", last_monday, last_monday, last_sunday,
            f"{last_monday.isoformat()} 那周",
        )
        _gen(
            "monthly", prev_month_first, prev_month_first, prev_month_last,
            f"{prev_month_first.strftime('%Y年%m月')}",
        )


# 事件只拉一次，build_profile 与周期总结复用，避免重复查询。
_all_events = get_profile_store().events(cid)
profile = _load_profile(_all_events)

# 周期总结：同一 cid 在本次会话生命周期内只检查/生成一次。
# 新建会话、切换会话、发消息都会触发 rerun，若每次都跑且赶上 LLM 生成，
# 会把整个页面阻塞几秒到十几秒（表现为"点击没反应"）。
_sum_key = f"_summaries_checked_{cid}"
if not st.session_state.get(_sum_key):
    _ensure_period_summaries(cid, _all_events)
    st.session_state[_sum_key] = True

# ---------------------------------------------------------------------------
# Agent 中枢主区（聊天）
# ---------------------------------------------------------------------------

def _load_chat(conv_id: str) -> list[dict]:
    """从 DB 读取会话的聊天记录（只取 turn 消息）。"""
    return [
        m["content"]
        for m in get_profile_store().load_messages(conv_id, client_id=cid)
        if m["role"] == "turn"
    ]


def _activate_conversation(conv_id: str, chat: list[dict]) -> None:
    """把某个会话设为当前会话（切换/新建共用的唯一入口）。"""
    st.session_state["agent_session_id"] = conv_id
    st.session_state["agent_chat"] = chat
    # 上一个会话的重试/待提交提问不能带进新会话
    st.session_state.pop("agent_pending", None)


def _new_conversation(cid: str) -> None:
    """新建会话并切换。

    用 on_click 回调而不是 if st.button 分支：回调在本次脚本渲染前执行，
    新会话 ID / 空聊天记录一定先于主区渲染生效，不会出现
    「记录建了、页面还显示上一个会话」。cid 由 args 显式传入，
    不依赖回调里读脚本全局变量（那可能是上一次运行的旧值）。
    """
    _new_id = uuid.uuid4().hex[:12]
    get_profile_store().create_conversation(cid, _new_id)
    _activate_conversation(_new_id, [])


def _switch_conversation(conv_id: str, cid: str) -> None:
    """会话按钮回调：切换会话并加载历史。

    用 on_click 而不是在 if st.button 分支里改 state —— 回调在脚本重跑前执行，
    切换结果同一次 rerun 就生效，不会出现「点了没反应」。
    """
    chat = [
        m["content"]
        for m in get_profile_store().load_messages(conv_id, client_id=cid)
        if m["role"] == "turn"
    ]
    _activate_conversation(conv_id, chat)
    # 历史回灌放在提交时做（见 runner.rehydrate_session）：幂等且廉价，
    # 点一下会话按钮不需要先把整套模型拉起来。


def _identity_switch(cid: str) -> None:
    """身份入口：任何用户（包括还没建规划的）都要能看到并切换身份。

    必须在渲染引导卡（内部 st.stop）之前调用，否则未规划用户
    连切换身份的输入框都看不到。
    """
    cid_input = st.text_input(
        "身份 ID（同一 ID 跨会话累积）", value=cid, key="cid_input"
    ).strip()
    if cid_input and cid_input != cid:
        st.query_params["cid"] = cid_input
        # 切换身份：清空上一个身份绑定的所有内存状态，
        # 否则上一个身份的会话/规划/待确认 prompt 还会挂在页面上。
        for _k in (
            "agent_session_id",
            "agent_chat",
            plan_ui.RESULT_KEY,
            plan_ui.USE_LLM_KEY,
        ):
            st.session_state.pop(_k, None)
        st.rerun()


# 会话状态：session_id + 本地渲染用的聊天记录（持久化到 profile.db，刷新不丢）
if "agent_session_id" not in st.session_state:
    # 优先复用最近一个「有消息」的会话；没有历史内容时复用已有的空会话；
    # 都没有才新建——避免每次新开标签页都造一条空记录，把侧栏刷成一串会话。
    _convs_all = get_profile_store().list_conversations(cid)
    _nonempty = [c for c in _convs_all if c["n_messages"] > 0]
    if _nonempty:
        st.session_state["agent_session_id"] = _nonempty[0]["id"]
    elif _convs_all:
        st.session_state["agent_session_id"] = _convs_all[0]["id"]
    else:
        _sid = uuid.uuid4().hex[:12]
        get_profile_store().create_conversation(cid, _sid)
        st.session_state["agent_session_id"] = _sid
if "agent_chat" not in st.session_state:
    st.session_state["agent_chat"] = _load_chat(
        st.session_state["agent_session_id"]
    )

# 身份入口先于「无规划 st.stop」渲染：未规划用户也要能切回自己的身份。
# st.stop 只挡聊天区，已渲染的侧栏内容不受影响。
with st.sidebar:
    st.header("💬 会话")
    _identity_switch(cid)

# 当前规划：落库读取 + 首次严格引导（无规划时模块内部 st.stop()）
plan_ui.load_saved(cid)
plan_ui.render_onboarding(cid, profile)

top_l, top_r = st.columns([3, 1])
top_l.caption(f"会话 `{st.session_state['agent_session_id']}`")
top_r.button(
    "新会话",
    key="agent_new_session",
    on_click=_new_conversation,
    args=(cid,),
    use_container_width=True,
)

# 新会话是空页：明确提示「已经切过来了」，避免看起来像没反应
if not st.session_state["agent_chat"]:
    st.info("新会话已开始，直接提问即可。")


# 历史渲染：用户问句 + 回答 + 每轮元信息与推理链
for mi, msg in enumerate(st.session_state["agent_chat"]):
    with st.chat_message("user"):
        st.markdown(msg["q"])
    with st.chat_message("assistant"):
        d = msg["d"]
        if d.get("resolved_query"):
            st.caption(f"已按上轮语境改写为：{d['resolved_query']}")
        if d.get("answer"):
            st.markdown(d["answer"])
        else:
            # 无答案时不能一律显示「未知原因」：按 stop_reason 给出可区分的提示
            _icon, _text = _stop_reason_view(
                d.get("stop_reason"), d.get("error")
            )
            st.markdown(f"{_icon} **{_text}**")
            if d.get("stop_reason") == "tool_error":
                if st.button("🔁 重试这一问", key=f"retry_stop_{mi}"):
                    st.session_state["agent_pending"] = msg["q"]
                    st.rerun()
        # 考情类回答内嵌客观统计图（数据来自 get_exam_stats 工具输出，非模型预测）
        es = (d.get("tool_outputs") or {}).get("exam_stats")
        if isinstance(es, dict) and es.get("total"):
            st.caption(
                f"📊 「{es.get('tag')}」历年考查 {es['total']} 次"
                f"（{len(es.get('matched_tags') or [])} 个标签）· 客观计数"
            )
            ec1, ec2 = st.columns(2)
            with ec1:
                st.caption("年份分布")
                st.bar_chart({"次数": {str(y): n for y, n in (es.get("by_year") or {}).items()}})
            with ec2:
                st.caption("题型分布")
                st.bar_chart({"次数": es.get("by_type") or {}})
        v = d.get("verification") or {}
        cost = f" · ¥{d['cost_cny']:.4f}" if d.get("cost_cny") else ""
        steps = d.get("steps") or []
        rt = d.get("route") or ""
        rlabel, rdesc = ROUTE_LABELS.get(rt, (rt or "—", ""))
        rreason = next(
            (
                str(s.get("detail", {}).get("reason") or "")
                for s in steps
                if s.get("node") == "route"
            ),
            "",
        )
        flow = " → ".join(dict.fromkeys(str(s.get("node")) for s in steps))
        st.markdown(
            f"**🤖 Agent 执行过程** · 路由「{rlabel}」——{rreason or rdesc}"
        )
        st.caption(
            f"链路 {flow or '—'} · {d.get('took_ms', 0) / 1000:.1f}s · "
            f"工具 {d.get('n_tool_calls', 0)} 次/{d.get('rounds', 0)} 轮 · "
            f"停止原因 {_stop_reason_view(d.get('stop_reason'))[1]}{cost}"
        )
        tool_calls = d.get("tool_calls") or []
        if tool_calls:
            for t in tool_calls:
                icon = "✅" if t["ok"] else "❌"
                st.markdown(
                    f"&nbsp;&nbsp;{icon} **{TOOL_LABELS.get(t['tool'], t['tool'])}**"
                    f" `{t['tool']}` · {t['took_ms']:.0f}ms · {t['summary']}"
                )
            st.caption(
                "以上工具全部是确定性实现（检索 / 统计 / 计算器 / 图形解析），"
                "LLM 只负责决定调谁和最终成文。"
            )
        else:
            st.caption("本轮未调用工具。")
        if v:
            if v.get("ok"):
                st.caption("✅ 答案校验通过（引用与真题编号均有效、无押题表述）")
            else:
                problems = []
                if v.get("invalid_exam_refs"):
                    problems.append(f"无效真题编号 {v['invalid_exam_refs']}")
                if v.get("predictions"):
                    problems.append(f"命中押题表述 {v['predictions'][:3]}")
                st.caption("⚠️ 校验未通过：" + "；".join(problems))
        with st.expander("步骤耗时明细"):
            st.dataframe(
                [
                    {
                        "步骤": s["node"],
                        "耗时(ms)": s["took_ms"],
                        "详情": json.dumps(s.get("detail", {}), ensure_ascii=False)[:160],
                    }
                    for s in steps
                ],
                width="stretch",
            )
        reco = msg.get("reco") or []
        if reco:
            st.markdown("**📎 相关真题**（先自测再看答案；作答会同步错题本与画像）")
            for q in reco:
                with st.container(border=True):
                    st.markdown(
                        f"**[{q['qid']}]** {q['subject']} · {q.get('type') or '—'} "
                        f"· 出处：{q['year']} 年 408 第 {q['num']} 题"
                        + (f" · [原题链接]({q['url']})" if q.get("url") else "")
                    )
                    st.write(q["question"])
                    for k, v in (q.get("options") or {}).items():
                        st.markdown(f"- **{k}.** {v}")
                    with st.expander("参考答案与解析"):
                        st.markdown(f"**答案：{q['answer']}**")
                        if q.get("explanation"):
                            st.caption(q["explanation"])
                        _quiz_feedback(q, mi)
        if d.get("needs_clarification"):
            st.markdown("**选一个继续：**")
            for i, opt in enumerate(d.get("clarification_options") or []):
                if st.button(
                    opt, key=f"clarify_{mi}_{i}", width="stretch"
                ):
                    st.session_state["agent_pending"] = opt
                    st.rerun()

# 澄清按钮的点击结果与手动输入走同一条发送路径
pending = st.session_state.pop("agent_pending", None)
prompt = pending or st.chat_input(
    "提问、追问，或「帮我规划复习」「我的画像」"
)
if prompt:
    # 规划意图：弹条件确认框（HITL），确认后在 dialog 内生成独立规划记录
    if plan_ui.wants_plan(prompt):
        plan_ui.open_dialog(cid, profile)
    else:
        runner = get_runner()
        # 每轮都做一次增量回灌（幂等）：切会话、刷新、多标签都以落库历史为准，
        # 不再依赖一次性的标记——标记会在 rerun / 异常后丢掉。
        runner.rehydrate_session(
            st.session_state["agent_session_id"],
            st.session_state["agent_chat"],
            client_id=cid,
        )
        with st.spinner("Agent 执行中（含 LLM 生成，通常 10–60 秒）…"):
            res = runner.run(
                prompt,
                session_id=st.session_state["agent_session_id"],
                client_id=cid,
            )
        msg: dict = {"q": prompt, "d": res.to_dict()}
        plan_out = (msg["d"].get("tool_outputs") or {}).get("plan_result")
        if plan_out:
            # 看板与画像事件同一事务写入（由规划模块统一处理）
            plan_ui.apply_agent_plan(cid, plan_out)
        reco_text = (
            f"{msg['d'].get('resolved_query') or ''} {prompt} {msg['d'].get('answer') or ''}"
        )
        if msg["d"].get("route") == "plan":
            msg["reco"] = []
        else:
            msg["reco"] = _recommend_questions(reco_text)
        _record_question(prompt)
        st.session_state["agent_chat"].append(msg)

        _conv_id = st.session_state["agent_session_id"]
        get_profile_store().append_message(_conv_id, "turn", msg, client_id=cid)
        if len(st.session_state["agent_chat"]) == 1:
            get_profile_store().set_conversation_title(
                _conv_id, prompt, client_id=cid
            )
        st.rerun()

# ---------------------------------------------------------------------------
# 侧栏：会话列表 + 学习总结（系统自动生成，用户无需操作）
# ---------------------------------------------------------------------------
with st.sidebar:
    # 身份入口已提前渲染（见 _identity_switch），这里只放会话与总结。
    # 当前规划 + 重新制定：规划模块统一渲染
    plan_ui.render_sidebar(cid, profile)

    st.button(
        "➕ 新建会话",
        key="sb_new_conv",
        use_container_width=True,
        on_click=_new_conversation,
        args=(cid,),
    )

    _cur = st.session_state.get("agent_session_id")
    _all_convs = get_profile_store().list_conversations(cid)
    # 只展示有消息的会话 + 当前会话，避免空会话把列表刷成一串「（空会话）」
    _convs = [
        c for c in _all_convs if c["n_messages"] > 0 or c["id"] == _cur
    ]
    # 当前会话置顶：默认按更新时间排序，切到旧会话后它会留在列表深处，
    # 看起来像「没切过来」；置顶后当前会话永远在第一屏。
    _convs = sorted(_convs, key=lambda c: c["id"] != _cur)

    # 会话列表默认只显示最近 8 条：条数一多会把下方「学习总结」挤出几屏。
    _CONV_LIMIT = 8
    _expand = bool(st.session_state.get("_conv_expand"))
    _shown = _convs if _expand else _convs[:_CONV_LIMIT]

    if not _convs:
        st.caption("暂无会话，提问后会自动保存")
    for _c in _shown:
        _title = _c["title"] or "（未命名会话）"
        # 单行按钮会静默截断长标题，这里显式截断并用 tooltip 给出完整标题
        _label = _title if len(_title) <= 18 else _title[:18] + "…"
        _is_cur = _c["id"] == _cur
        # 用横向容器而不是 columns：窄窗口下 columns 会把删除按钮折到下一行，
        # 每行变成两行高，会话一多就把下半部分全挤走。
        with st.container(
            horizontal=True,
            wrap=False,
            vertical_alignment="center",
            gap="small",
        ):
            # 当前会话也允许点击（重复加载历史），用 ● 与 primary 样式标出
            st.button(
                f"{'● ' if _is_cur else ''}{_label}",
                key=f"conv_{_c['id']}",
                help=_title,
                width="stretch",
                type="primary" if _is_cur else "secondary",
                on_click=_switch_conversation,
                args=(_c["id"], cid),
            )
            if st.button(
                "🗑",
                key=f"del_{_c['id']}",
                help="删除这个会话（规划是独立记录，不受影响）",
            ):
                _delete_conversation(_c["id"])

    if len(_convs) > _CONV_LIMIT and st.button(
        "收起会话列表" if _expand else f"显示全部 {len(_convs)} 个会话",
        key="sb_conv_toggle",
        use_container_width=True,
    ):
        st.session_state["_conv_expand"] = not _expand
        st.rerun()

    # 空会话在侧栏已隐藏，但 DB 里会随刷新累积，给一个手动清理入口。
    _n_empty = sum(
        1 for c in _all_convs if c["n_messages"] == 0 and c["id"] != _cur
    )
    if _n_empty:
        if st.button(
            f"🧹 清理 {_n_empty} 个空会话",
            key="sb_clean_empty",
            use_container_width=True,
        ):
            get_profile_store().delete_empty_conversations(cid, keep_id=_cur)
            st.rerun()

    st.divider()
    st.header("📊 学习总结")
    st.caption("每周一生成上周总结、每月 1 号生成上月总结，点击日期查看")

    _weekly = get_profile_store().list_summaries(cid, "weekly")
    _monthly = get_profile_store().list_summaries(cid, "monthly")

    if _weekly:
        st.markdown("**周总结**")
        # 只列最近 4 期，避免时间一长侧栏被日期列表占满；更早的收进折叠区。
        # Streamlit 不支持 expander 嵌套，所以折叠区里用标题 + 正文分段。
        for _s in _weekly[:4]:
            with st.expander(f"📅 {_s['period_date']} 那周", expanded=False):
                st.markdown(_s["content"])
        if len(_weekly) > 4:
            with st.expander(f"📂 更早的 {len(_weekly) - 4} 期周总结", expanded=False):
                for _s in _weekly[4:]:
                    st.markdown(f"**{_s['period_date']} 那周**")
                    st.markdown(_s["content"])
                    st.divider()
    else:
        st.caption("暂无周总结（上周学习后自动生成）")

    if _monthly:
        st.markdown("**月总结**")
        for _s in _monthly[:3]:
            ym = _s["period_date"][:7].replace("-", "年") + "月"
            with st.expander(f"📅 {ym}", expanded=False):
                st.markdown(_s["content"])
        if len(_monthly) > 3:
            with st.expander(f"📂 更早的 {len(_monthly) - 3} 期月总结", expanded=False):
                for _s in _monthly[3:]:
                    ym = _s["period_date"][:7].replace("-", "年") + "月"
                    st.markdown(f"**{ym}**")
                    st.markdown(_s["content"])
                    st.divider()
    else:
        st.caption("暂无月总结（上月学习后自动生成）")

