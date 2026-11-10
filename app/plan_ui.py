"""规划（Plan）模块：状态、落库、条件弹窗、执行看板。

规划是全局流程前提，也是**独立于会话的一条记录**：存在 `plans` 表里，
按身份（client_id）绑定，不属于任何会话，也不进入 Agent 的多轮聊天历史。
所以这里不往聊天流写消息，只更新规划状态与看板。

写入只有一条通道：

    PlanRunner(iter_run / iter_resume) → result dict
        → session_state[RESULT_KEY]（看板渲染）
        → persist()：plans 表 + profile_events 同事务

聊天里 Agent 工具生成的规划也走 `apply_agent_plan()`，直接用它返回的完整
plan 组看板，不再按 goal 二次跑 PlanRunner。
"""

from __future__ import annotations

import math
import time
from datetime import date, timedelta

import streamlit as st

from cs408rag.adversarial.graph import PlanResult, PlanRunner
from cs408rag.adversarial.planner import SUBJECTS, next_exam_date, normalize_concerns
from cs408rag.profile.store import get_profile_store

# 规划状态键：外部（streamlit_app）需要知道的就这两个
RESULT_KEY = "adv_result"
USE_LLM_KEY = "adv_use_llm"

# 规划意图关键词：用户输入含这些词时，先弹条件确认框再生成
PLAN_INTENT_KEYWORDS = ("规划", "计划", "复习安排", "备考方案", "怎么复习", "复习计划")

ADV_TYPE_LABELS = {
    "concept": "概念",
    "calc": "计算",
    "figure": "图表",
    "cross_subject": "跨科目",
}

# 规划是独立记录：生成后只弹一次提示，不写进任何会话
FLASH_KEY = "_plan_flash"


def _types_text(types: dict | None) -> str:
    return " / ".join(
        f"{ADV_TYPE_LABELS.get(k, k)} {v}" for k, v in (types or {}).items()
    )


def has_plan() -> bool:
    return bool(st.session_state.get(RESULT_KEY))


def wants_plan(text: str) -> bool:
    return any(k in text for k in PLAN_INTENT_KEYWORDS)


# ---------------------------------------------------------------------------
# 状态与落库
# ---------------------------------------------------------------------------


def load_saved(cid: str) -> bool:
    """首次进入时把落库的规划读回 session_state（刷新不丢）。"""
    if RESULT_KEY not in st.session_state:
        saved = get_profile_store().load_plan(cid)
        if saved:
            st.session_state[RESULT_KEY] = saved
            st.session_state[USE_LLM_KEY] = True
    return has_plan()


def persist(
    cid: str,
    result: dict,
    goal: dict | None = None,
    *,
    stop_reason: str | None = None,
) -> None:
    """规划落库：委托给非 UI 层 persist_plan（Streamlit / API 共用）。"""
    from cs408rag.plan_persistence import persist_plan

    persist_plan(cid, result, goal, stop_reason=stop_reason)


def _snapshot_to_result(runner, goal, last, before, t0) -> dict:
    """把 graph 快照转成结果 dict；首轮生成与人工修订共用。"""
    after = runner.stats_snapshot()
    return PlanResult(
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
    ).to_dict()


def board_from_agent(plan_out: dict) -> dict | None:
    """直接用 Agent 工具那次算出的完整 plan 组看板，不再二次跑 PlanRunner。

    以前用 goal 重跑一遍，看板与 Agent 实际产出的那份不是同一个对象，
    规则/画像一变就会出现「回答里说的和看板上显示的不一致」。
    """
    plan = plan_out.get("plan")
    if not isinstance(plan, dict) or not plan.get("weekly_plan"):
        return None
    goal = plan.get("goal") or plan_out.get("goal") or {}
    return {
        "goal": goal,
        "plan": plan,
        "plan_history": [plan],
        "challenges": [],
        "challenge_history": [],
        "timeline": [
            {"node": "agent_tool:generate_plan", "took_ms": 0, "used_llm": False}
        ],
        "stop_reason": "agent_tool",
        "open_high": 0,
        "llm_challenges": 0,
        "version": 0,
        "rounds": 0,
        "llm_calls": 0,
        "llm_errors": 0,
        "planner_fallback": False,
        "plan_source": plan_out.get("plan_source") or "deterministic",
        "guide_source": (plan.get("meta") or {}).get("guide_source") or "template",
        "role_models": {},
        "tokens": {"prompt": 0, "completion": 0},
        "error": "",
        "took_ms": 0.0,
    }


def apply_agent_plan(cid: str, plan_out: dict) -> bool:
    """聊天里生成的规划：落状态 + 落库（看板与画像同一事务）。"""
    board = board_from_agent(plan_out)
    if not board:
        return False
    st.session_state[RESULT_KEY] = board
    st.session_state[USE_LLM_KEY] = False
    persist(cid, plan_out, plan_out.get("goal"), stop_reason="agent_chat")
    return True


# ---------------------------------------------------------------------------
# 首次引导
# ---------------------------------------------------------------------------


def render_onboarding(cid: str, profile: dict) -> None:
    """没有规划时渲染引导卡并终止脚本（严格引导，不提供跳过）。"""
    if has_plan():
        return
    with st.container(border=True):
        st.markdown("### 👋 先定一个属于你的复习规划")
        st.markdown(
            "这个系统的所有内容都围绕你的规划展开。"
            "请先确认考试日期、每天可用时间和薄弱科目，我再给你排阶段与周计划。"
        )
        st.markdown(
            "1. **确认条件** —— 考试日期 / 每天小时 / 学习天数 / 薄弱科目\n"
            "2. **生成并校验** —— 规则引擎排骨架，LLM 写方法与每周理由\n"
            "3. **开始学习** —— 规划落地后，问答、真题、图谱、总结全部解锁"
        )
        if st.button(
            "开始制定规划",
            type="primary",
            use_container_width=True,
            key="onboard_start_plan",
        ):
            open_dialog(cid, profile)
    st.caption("制定完成后刷新也不会丢：规划会保存在本地数据库里。")
    st.stop()


# ---------------------------------------------------------------------------
# 条件弹窗（HITL）
# ---------------------------------------------------------------------------


def open_dialog(cid: str, profile: dict) -> None:
    """打开「调整规划条件」弹窗（Streamlit dialog 直接调用即弹窗）。"""
    _plan_dialog(cid, profile)


@st.dialog("✏️ 调整规划条件", width="large")
def _plan_dialog(cid: str, profile: dict) -> None:
    """Agent 规划决策弹窗：展示当前条件，用户可编辑后确认重新生成。

    HITL 语义：Agent 先基于画像推断条件生成一版，再把条件交给用户
    审查/编辑，确认后用新条件重新生成。
    """
    adv_subjects = SUBJECTS
    adv_progress_opts = ["未开始", "一轮中", "一轮完", "二轮中", "二轮完"]
    concern_tags = [
        "基础差 / 跨考，看到代码就慌",
        "时间不够，每天只能学 3–4 小时",
        "做题慢，计算题总算错",
        "某个知识点总错（可写具体知识点）",
        "真题正确率低，不知道错在哪",
        "记不住，背了又忘",
        "不知道从哪开始",
    ]

    # 预填：当前规划的 goal > 画像推断 > 默认
    cur = (st.session_state.get(RESULT_KEY) or {}).get("plan", {}).get("goal", {})
    last_plan = profile.get("plan") or {}
    prefill_weak = [s for s in (last_plan.get("weak_subjects") or []) if s in adv_subjects]
    if not prefill_weak:
        prefill_weak = [
            s for s in ((profile.get("quiz") or {}).get("wrong_subjects") or {})
            if s in adv_subjects
        ][:1]
    if not prefill_weak and profile.get("subject_counts"):
        prefill_weak = [next(iter(profile["subject_counts"]))]
    prefill_weak = (prefill_weak or ["操作系统"])[:2]
    profile_tags = profile.get("boost_tags") or []
    if profile_tags:
        st.caption(
            "已按提问画像加权："
            + "、".join(t["tag"] for t in profile_tags[:5])
        )

    # 时间输入放 form 外：改动后立即刷新预算
    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        adv_exam = st.date_input(
            "考试日期",
            value=date.fromisoformat(cur["exam_date"]) if cur.get("exam_date") else next_exam_date(),
            min_value=date.today() + timedelta(days=7),
            key="dlg_exam",
        )
    with tc2:
        adv_daily = st.number_input(
            "每天 408 小时", 1.0, 12.0,
            value=float(cur.get("daily_hours") or 4.0), step=0.5, key="dlg_daily",
        )
    with tc3:
        adv_days = st.slider(
            "每周学习天数", 1, 7,
            value=int(cur.get("study_days_per_week") or 6), key="dlg_days",
        )
    days = max(1, (adv_exam - date.today()).days)
    weeks = max(1, math.ceil(days / 7))
    weekly = float(adv_daily) * int(adv_days)
    st.info(f"距考试 {days} 天 ≈ {weeks} 周 · 每周 {weekly:g}h · 总约 {weeks * weekly:g}h")

    with st.form("plan_dialog_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            scope_label = st.radio(
                "规划范围", ["全科 408", "单科补弱"], horizontal=True,
                index=0 if cur.get("scope", "full_408") == "full_408" else 1,
                key="dlg_scope",
            )
        with c2:
            adv_weak = st.multiselect(
                "薄弱科目（1–2 个）", adv_subjects,
                default=cur.get("weak_subjects") or prefill_weak,
                max_selections=2, key="dlg_weak",
            )
        with c3:
            adv_target = st.number_input(
                "目标分", 60, 150, value=int(cur.get("target_score") or 115), key="dlg_target",
            )

        st.markdown("**四科当前进度**")
        pc = st.columns(4)
        progress: dict[str, str] = {}
        cur_prog = cur.get("progress") or {}
        for i, subj in enumerate(adv_subjects):
            with pc[i]:
                progress[subj] = st.selectbox(
                    subj, adv_progress_opts,
                    index=adv_progress_opts.index(cur_prog.get(subj, "一轮中")),
                    key=f"dlg_prog_{subj}",
                )

        st.markdown("**备考问题**")
        concern_sel = st.multiselect(
            "常见问题（可多选）", concern_tags, key="dlg_concern_tags",
        )
        concern_text = st.text_area(
            "也可以直接写（每行一条）", height=90, key="dlg_concern_text",
        )

        fc1, fc2 = st.columns(2)
        with fc1:
            adv_rounds = st.number_input(
                "最大修订轮数", 0, 2, value=int(cur.get("max_rounds", 2) if "max_rounds" in cur else 2),
                key="dlg_rounds",
            )
        with fc2:
            adv_llm = st.checkbox(
                "启用 LLM 参与（文案 + 每周理由）",
                value=True,
                key="dlg_llm",
            )

        submitted = st.form_submit_button("确认并重新生成", type="primary")

    if submitted:
        concerns = normalize_concerns(concern_text)
        for tag in concern_sel:
            if tag and tag not in concerns:
                concerns.append(tag)
        scope = "full_408" if scope_label == "全科 408" else "single_subject"
        weak = list(adv_weak) or ["操作系统"]
        goal = {
            "scope": scope,
            "weak_subjects": weak,
            "start_date": date.today().isoformat(),
            "exam_date": adv_exam.isoformat(),
            "daily_hours": float(adv_daily),
            "study_days_per_week": int(adv_days),
            "target_score": int(adv_target),
            "progress": progress,
            "concerns": concerns[:8],
            "profile_tags": profile_tags,
        }
        runner = PlanRunner(use_llm=bool(adv_llm), max_rounds=int(adv_rounds))
        st.session_state[USE_LLM_KEY] = bool(adv_llm)
        pbar = st.progress(0.0, text="正在生成计划…")
        last: dict = {}
        before = runner.stats_snapshot()
        t0 = time.perf_counter()
        try:
            for snapshot in runner.iter_run(goal):
                last = snapshot
                n = len(snapshot.get("timeline") or [])
                pbar.progress(min(0.95, n / 8), text="正在生成并校验计划…")
        finally:
            pbar.empty()
        result = _snapshot_to_result(runner, goal, last, before, t0)
        st.session_state[RESULT_KEY] = result
        # 落库 + 画像事件：同一事务写入（刷新后仍是「已规划」状态）
        persist(cid, result, goal)
        # 规划是独立记录：不写进会话、不参与多轮对话，只刷新看板并提示一次
        st.session_state[FLASH_KEY] = "✅ 规划已更新（独立记录，不属于任何会话）"
        st.rerun()


# ---------------------------------------------------------------------------
# 看板（侧栏入口 → 弹窗；不占 Agent 会话流）
# ---------------------------------------------------------------------------


def _render_week(w: dict, *, with_method: bool) -> None:
    tasks = w.get("tasks") or []
    total = sum(float(t.get("hours") or 0) for t in tasks)
    focus = "、".join(w.get("focus") or [])
    st.markdown(
        f"**第 {w.get('week')} 周**　·　{w.get('stage')}"
        f"　·　合计 {total:g}h"
        + (f"　·　主攻 {focus}" if focus else "")
    )
    st.caption(f"{w.get('start')} → {w.get('end')}")
    # LLM 为本周写的推进逻辑；缺失时（未启用/调用失败）就不显示
    if w.get("reason"):
        st.markdown(f"> {w.get('reason')}")
    methods = {t.get("method") for t in tasks}
    accepts = {t.get("acceptance") for t in tasks}
    lines: list[str] = []
    for t in tasks:
        kps = t.get("kps") or []
        content = str(t.get("content") or "")
        # 「学什么」里若已含考点名（如「补漏 A、B」），前缀就不再重复一遍
        kps_prefix = (
            " · " + "、".join(kps)
            if kps and not all(k in content for k in kps)
            else ""
        )
        line = (
            f"- **{t.get('subject')}** · "
            f"{float(t.get('hours') or 0):g}h"
            + kps_prefix
            + f"：{content}"
        )
        # 验收标准各科不同时才逐条显示，相同就收进卡片底部
        if len(accepts) > 1 and t.get("acceptance"):
            line += f"　—　验收：{t.get('acceptance')}"
        lines.append(line)
    if lines:
        st.markdown("\n".join(lines))
    if with_method and len(methods) == 1:
        m = str(next(iter(methods)) or "")
        if m:
            st.caption(f"怎么学：{m}")
    if with_method and len(accepts) == 1:
        a = str(next(iter(accepts)) or "")
        if a:
            st.caption(f"验收标准：{a}")


def open_board(cid: str) -> None:
    """侧栏入口：点击后弹出规划执行看板。

    看板不写进 Agent 会话流（规划是独立记录），入口固定在左侧，
    需要看细节时再打开，避免长计划把聊天记录挤走。
    """
    _board_dialog(cid)


@st.dialog("📅 我的规划执行 · 阶段路线 / 周计划 / 人工裁定", width="large")
def _board_dialog(cid: str) -> None:
    result = st.session_state.get(RESULT_KEY)
    if not result:
        st.info("还没有规划，先在侧栏「✏️ 重新制定规划」生成一份。")
        return
    adv_meta = ((result.get("plan") or {}).get("meta") or {})
    if adv_meta.get("weekly_source") == "llm":
        st.caption(
            "计划骨架由确定性引擎生成（小时/考点可校验）；每周理由由 LLM 生成，"
            "方法与资料清单为模板。"
        )
    elif st.session_state.get(USE_LLM_KEY):
        st.caption("计划骨架由确定性引擎生成；LLM 本轮未生效（调用失败或未返回）。")
    else:
        st.caption("本计划由确定性引擎生成（未启用 LLM）。")

    adv_subjects = SUBJECTS

    # 条件修改入口在侧栏（「✏️ 重新制定规划」）：dialog 里不能再开 dialog
    st.caption("要改考试日期 / 每天小时 / 薄弱科目，用侧栏「✏️ 重新制定规划」。")

    plan = result.get("plan") or {}
    g = plan.get("goal") or {}
    stages = plan.get("stages") or []
    kps = plan.get("priority_kps") or []
    actions = plan.get("concern_actions") or []
    weekly = plan.get("weekly_plan") or []
    guide = plan.get("guide") or {}

    m1, m2, m3 = st.columns(3)
    m1.metric("剩余周数", g.get("remaining_weeks", "—"))
    m2.metric("每周可用小时", f"{g.get('weekly_hours', 0):g}")
    m3.metric("考试日期", g.get("exam_date") or "—")

    st.caption(
        f"按每天 {g.get('daily_hours', '?')}h × "
        f"{g.get('study_days_per_week', '?')} 天排课，"
        f"目标 {g.get('target_score', '?')} 分"
    )
    if result.get("error"):
        st.error(result["error"])
    if result.get("planner_fallback"):
        st.warning("结构 Plan 生成失败，已使用兜底；本次结果不可信。")

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

    st.divider()

    # ① 学什么 -------------------------------------------------
    st.markdown("#### ① 学什么 · 高性价比知识点")
    st.caption(
        "ROI = 频次 + 近 5 年 + 计算/图表占比 + 学习成本，是启发式排序，不是押题。"
    )
    if kps:
        subj_filter = st.multiselect(
            "科目筛选",
            adv_subjects,
            default=adv_subjects,
            key="adv_kp_subj",
        )
        shown_kps = [
            k for k in kps if not subj_filter or k.get("subject") in subj_filter
        ]
        if len(shown_kps) > 15:
            st.caption(f"筛出 {len(shown_kps)} 个考点，下面按 ROI 显示前 15 个。")
        for k in shown_kps[:15]:
            types = _types_text(k.get("types"))
            with st.expander(
                f"[{k.get('roi_label')}] {k.get('subject')} · {k.get('tag')}"
                f" · {k.get('frequency')} 次（近5年 {k.get('recent_5y')}）"
                + (f" · {types}" if types else "")
            ):
                st.write(k.get("why"))
                if k.get("must_learn"):
                    st.markdown("**优先学**：" + "、".join(k["must_learn"]))
                if k.get("later"):
                    st.markdown("**可后置**：" + "、".join(k["later"]))
                st.caption(
                    "教材小节：" + ("、".join(k.get("kp_titles") or []) or "—")
                )
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

    st.divider()

    # ② 怎么针对你 -------------------------------------------------
    st.markdown("#### ② 怎么针对你 · 问题 → 计划调整")
    if actions:
        prev_cp: str | None = None
        for a in actions:
            with st.container(border=True):
                st.markdown(
                    f"**{a.get('concern')}**　·　阶段 `{a.get('stage')}`"
                )
                st.caption(f"诊断：{a.get('diagnosis')}")
                st.markdown(f"**→ 计划怎么解决**：{a.get('action')}")
                if a.get("source"):
                    st.caption(f"相关经验：《{a['source']}》（来源见 ④ 参考经验贴）")
                cp = str(a.get("checkpoint") or "")
                if cp and cp != prev_cp:
                    st.caption(f"检查点：{cp}")
                prev_cp = cp
    else:
        st.caption(
            "没有填写备考问题，这份计划按你的进度和薄弱科目生成。"
            "在表单里自由描述（如「计组看不懂」「做题慢」），会逐条映射成调整动作并附相关经验贴。"
        )

    st.divider()

    # ③ 何时做什么 -------------------------------------------------
    st.markdown("#### ③ 何时做什么 · 周计划")
    if weekly:
        st.caption(
            "每周总小时 = 表单预算；只展开当前周，其余按阶段收进折叠区，"
            "避免备考周期长时整列卡片刷屏。"
        )

        # 当前周：按今天落在哪一周判定；计划还没开始取第一周，已结束取最后一周
        today_iso = date.today().isoformat()
        cur_w = next(
            (
                w
                for w in weekly
                if str(w.get("start") or "") <= today_iso <= str(w.get("end") or "")
            ),
            None,
        )
        if cur_w is None:
            cur_w = (
                weekly[-1]
                if today_iso > str(weekly[-1].get("end") or "")
                else weekly[0]
            )
        cur_no = cur_w.get("week")

        with st.container(border=True):
            _render_week(cur_w, with_method=True)

        others = [w for w in weekly if w.get("week") != cur_no]
        if others:
            with st.expander(
                f"📋 其余 {len(others)} 周计划"
                f"（{others[0].get('start')} → {others[-1].get('end')}）",
                expanded=False,
            ):
                prev_stage: str | None = None
                for w in others:
                    stage = w.get("stage")
                    if stage != prev_stage:
                        st.markdown(f"**{stage}阶段**")
                        methods = {
                            t.get("method") for t in (w.get("tasks") or [])
                        }
                        if len(methods) == 1:
                            m = str(next(iter(methods)) or "")
                            if m:
                                st.caption(f"怎么学：{m}")
                        prev_stage = stage
                    _render_week(w, with_method=False)
                    st.write("")
    else:
        st.caption("没有周计划。")

    st.divider()

    # ④ 怎么学：第一人称叙述，分段写，不堆条目
    st.markdown("#### ④ 怎么学 · 经验贴 / 方法")
    if guide.get("note"):
        st.caption(guide["note"])
    if guide.get("expectation"):
        st.markdown(f"**先说心态。** {guide['expectation']}")
    if guide.get("subject_order"):
        st.markdown(f"**起步顺序。** {guide['subject_order']}")
    if guide.get("crunch"):
        st.markdown(f"**如果时间已经不够。** {guide['crunch']}")
    if guide.get("weekly_routine"):
        st.markdown(f"**一周怎么过。** {guide['weekly_routine']}")
    if guide.get("paper_usage"):
        st.markdown(f"**真题。** {guide['paper_usage']}")
    if guide.get("review_workflow"):
        st.markdown(f"**错题。** {guide['review_workflow']}")
    ss = guide.get("subject_strategy") or []
    if ss:
        st.markdown("**分科上，我是这么做的。**")
        for s in ss:
            st.markdown(f"**{s.get('subject')}**：{s.get('strategy')}")
    if guide.get("pacing"):
        st.markdown(f"**考场上。** {guide['pacing']}")
    st.caption("按你的选择：经验贴只写方法，不推荐教材 / 网课 / 老师。")
    srcs = guide.get("sources") or []
    if srcs:
        with st.expander(f"参考经验贴（{len(srcs)} 篇，已筛掉引流 / 卖资料贴）"):
            for s in srcs:
                likes = f"，{s['likes']} 赞" if s.get("likes") else ""
                line = (
                    f"- **{s.get('title')}** · {s.get('author')}"
                    f"（{s.get('platform')}{likes}）"
                )
                if s.get("url"):
                    line += f" · [原文]({s['url']})"
                if s.get("used_for"):
                    line += f"\n  用于：{s['used_for']}"
                st.markdown(line)

    # ⑤ 需要你决定的事：只有 LLM 主观挑战缺客观裁判，才交给用户
    challenges = result.get("challenge_history") or result.get("challenges") or []
    open_llm = [
        c
        for c in challenges
        if c.get("rule") == "llm_critic" and c.get("status", "open") == "open"
    ]
    if open_llm:
        st.divider()
        st.markdown("#### ⑤ 需要你决定的事")
        with st.container(border=True):
            st.markdown("**🧑 人工裁定** · LLM 主观挑战没有客观裁判，由你决定")
            st.caption("未勾选的视为忽略：忽略的挑战会被关闭，且之后不会再被提出")
            accepted: list[dict] = []
            for c in open_llm[:8]:
                if st.checkbox(
                    f"采纳 [{c.get('severity')}] {c.get('message')}",
                    key=f"hr_{c.get('id')}",
                ):
                    accepted.append(c)
            if st.button("按裁定修订一版", key="adv_human_revise"):
                for c in open_llm:
                    if c in accepted:
                        c["verdict"] = "human_accept"
                    else:
                        c["verdict"] = "human_ignore"
                        c["status"] = "dropped"
                if not accepted:
                    # 全部忽略：裁定已就地写进 challenges，无需再跑一版
                    st.rerun()
                prev = {
                    "goal": result.get("goal") or {},
                    "plan": result.get("plan") or {},
                    "plan_history": result.get("plan_history") or [],
                    "version": int(result.get("version") or 0),
                    "round": int(result.get("rounds") or 0),
                    "challenge_history": list(challenges),
                }
                runner2 = PlanRunner(
                    use_llm=bool(st.session_state.get(USE_LLM_KEY, True)),
                    max_rounds=0,
                )
                before2 = runner2.stats_snapshot()
                t02 = time.perf_counter()
                last2: dict = {}
                with st.spinner("按你的裁定修订中…"):
                    for snap in runner2.iter_resume(prev, accepted):
                        last2 = snap
                new_result = _snapshot_to_result(
                    runner2, prev["goal"], last2, before2, t02
                )
                new_result["timeline"] = (
                    (result.get("timeline") or [])
                    + [
                        {
                            "node": "human_review",
                            "n_accept": len(accepted),
                            "n_ignore": len(open_llm) - len(accepted),
                            "took_ms": 0,
                        }
                    ]
                    + (last2.get("timeline") or [])
                )
                st.session_state[RESULT_KEY] = new_result
                persist(cid, new_result, prev["goal"])
                st.rerun()


# ---------------------------------------------------------------------------
# 侧栏「当前规划」
# ---------------------------------------------------------------------------


def render_sidebar(cid: str, profile: dict) -> None:
    """侧栏顶部：当前规划摘要 + 看板/重新制定入口（提醒用户自己在哪个阶段）。"""
    plan_state = st.session_state.get(RESULT_KEY) or {}
    goal = (plan_state.get("plan") or {}).get("goal") or plan_state.get("goal") or {}
    if not goal:
        return
    # 规划刚生成/更新：一次性反馈（规划不写进会话，只能在侧栏提示）
    flash = st.session_state.pop(FLASH_KEY, None)
    if flash:
        st.success(flash)
    st.subheader("🎯 当前规划")
    st.caption(
        f"{goal.get('remaining_weeks') or '?'} 周 · "
        f"每天 {goal.get('daily_hours') or '?'}h · "
        f"考试 {goal.get('exam_date') or '?'}"
    )
    weak = "、".join(goal.get("weak_subjects") or [])
    if weak:
        st.caption(f"薄弱：{weak}")
    if st.button("📅 我的规划执行", key="sb_open_board", use_container_width=True):
        open_board(cid)
    if st.button("✏️ 重新制定规划", key="sb_replan", use_container_width=True):
        open_dialog(cid, profile)
