"""日期驱动、数据驱动的详细备考 Plan 生成器。

设计原则：**结构由规则算，LLM 只负责把方法建议写顺**。

    compute_time_budget → route_concerns → allocate_phases
    → allocate_subject_hours → build_priority_kps → build_weekly_plan
    → build_guide_template

因此即使 LLM 空回答/超时，Plan 仍然是一份完整、可执行、可被规则检查的结构，
不会退化成「只有每周几小时」的课表。
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from datetime import date, timedelta
from typing import Any

from cs408rag.adversarial.roi import (
    TagROI,
    prereq_order,
    select_high_value,
    subject_coverage,
)
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer

SUBJECTS = ["数据结构", "组成原理", "操作系统", "计算机网络"]
SUBJECT_SCORES = {"数据结构": 45, "组成原理": 45, "操作系统": 35, "计算机网络": 25}
PROGRESS_LEVELS = ["未开始", "一轮中", "一轮完", "二轮中", "二轮完"]
PHASES = ["基础", "强化", "真题", "冲刺"]

# 经验贴里反复出现的起步顺序：DS 先（其他三门基础），组成+OS 连着（交叉多），网络最后。
# 极短周期拆不开时，用「每周主攻 1–2 科 + 其他科维持」而不是四科齐头并进。
SUBJECT_BLOCKS = [["数据结构"], ["组成原理", "操作系统"], ["计算机网络"]]

# 公开经验贴里反复出现的分科方法；只写方法，不绑定任何教材/网课/老师。
SUBJECT_METHODS = {
    "数据结构": {
        "focus": "选择题 + 算法题",
        "method": (
            "概念用选择题巩固；树/图/排序的算法题先写思路和伪代码，再对答案；"
            "复杂度自己推一遍，不要只背结论"
        ),
        "pitfalls": ["算法题只看不写", "只背代码不理解过程", "忽略复杂度分析"],
    },
    "组成原理": {
        "focus": "计算题 + 数据通路",
        "method": (
            "Cache/流水线/IEEE754/指令周期先写公式再代数；"
            "数据通路和控制信号用画图或表格整理"
        ),
        "pitfalls": ["公式死记硬背", "计算题不写步骤", "只看图不动笔"],
    },
    "操作系统": {
        "focus": "概念 + 计算 + 同步大题",
        "method": (
            "进程/内存/文件概念用选择题；调度/页面置换/银行家算法做计算题；"
            "PV 大题先画同步关系再写代码"
        ),
        "pitfalls": ["PV 只背模板", "页面置换不算命中率", "概念混淆"],
    },
    "计算机网络": {
        "focus": "分层主线 + 计算",
        "method": (
            "按分层模型抓主线；TCP 窗口/子网划分/CSMA-CD 先画时序或表格；"
            "协议字段用对比表记，不要逐条死背"
        ),
        "pitfalls": ["死背协议细节", "计算题不画图", "忽略分层关系"],
    },
}


# ---------------------------------------------------------------------------
# 日期与预算
# ---------------------------------------------------------------------------


def _as_date(value: Any, default: date | None = None) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            pass
    return default or date.today()


def _last_saturday_on_or_before(year: int, month: int, day: int) -> date:
    d = date(year, month, day)
    # weekday(): 周一=0 ... 周六=5
    return d - timedelta(days=(d.weekday() - 5) % 7)


def next_exam_date(today: date | None = None) -> date:
    """下一次考研：当年 12 月 25 日前最后一个周六；已过则下一年。"""
    today = today or date.today()
    exam = _last_saturday_on_or_before(today.year, 12, 25)
    if exam < today:
        exam = _last_saturday_on_or_before(today.year + 1, 12, 25)
    return exam


def compute_time_budget(goal: dict[str, Any], today: date | None = None) -> dict[str, Any]:
    start = _as_date(goal.get("start_date"), today or date.today())
    exam = _as_date(goal.get("exam_date"), next_exam_date(start))
    if exam <= start:
        raise ValueError(f"考试日期 {exam} 必须晚于开始日期 {start}")
    daily = max(0.5, float(goal.get("daily_hours") or 4))
    days_per_week = max(1, min(7, int(goal.get("study_days_per_week") or 6)))
    days = (exam - start).days
    weeks = max(1, math.ceil(days / 7))
    weekly_hours = round(daily * days_per_week, 2)
    return {
        "start_date": start.isoformat(),
        "exam_date": exam.isoformat(),
        "remaining_days": days,
        "remaining_weeks": weeks,
        "daily_hours": daily,
        "study_days_per_week": days_per_week,
        "weekly_hours": weekly_hours,
        "total_hours": round(weekly_hours * weeks, 2),
    }


# ---------------------------------------------------------------------------
# 用户备考问题
# ---------------------------------------------------------------------------

_CONCERN_RULES: list[tuple[str, list[str]]] = [
    ("basics", ["基础差", "零基础", "跨考", "没学过", "看不懂", "代码就慌", "底子"]),
    ("time", ["时间不够", "来不及", "每天只能", "没时间", "太忙", "时间少"]),
    ("slow_calc", ["做题慢", "计算题", "总算错", "算错", "粗心", "速度慢", "做不完"]),
    ("memory", ["记不住", "背了又忘", "忘得快", "背不下来", "容易忘"]),
    ("past_paper", ["真题", "正确率", "错在哪", "错题", "刷题"]),
]


def normalize_concerns(raw: Any) -> list[str]:
    if isinstance(raw, str):
        parts = re.split(r"[\n;；]+", raw)
    elif isinstance(raw, list):
        parts = [str(x) for x in raw]
    else:
        parts = []
    out: list[str] = []
    for p in parts:
        p = p.strip().strip("-•").strip()
        if p and p not in out:
            out.append(p[:120])
    return out[:8]


def _longest_tag_hit(tag: str, text: str) -> str:
    """tag 与用户原话的最长公共子串（至少 2 字）。

    `resolve_tags("死锁总是错")` 返回空，因为它要求整段文本是 tag 的子串；
    自由文本里用户会写「死锁总是错」，这里改为找 tag 的片段。
    """
    best = ""
    for i in range(len(tag)):
        for j in range(i + 2, len(tag) + 1):
            sub = tag[i:j]
            if sub in text and len(sub) > len(best):
                best = sub
    return best


def _match_tags(analyzer: ExamAnalyzer, text: str) -> list[tuple[str, str]]:
    """在用户原话里找具体科目 / 知识点，返回 [(subject, tag)]。"""
    hits: list[tuple[str, str]] = []
    # 1) 精确主题词
    for tag in analyzer.resolve_tags(text):
        qs = analyzer.by_tag.get(tag) or []
        if qs:
            hits.append((qs[0].subject, tag))
    # 2) 扫描全部 tag，取在文本中出现的最长片段
    longest: dict[str, tuple[int, str]] = {}
    for tag in analyzer.by_tag:
        if len(tag) < 2:
            continue
        hit = _longest_tag_hit(tag, text)
        if not hit:
            continue
        for q in analyzer.by_tag[tag][:1]:
            subj = q.subject
            if len(hit) > longest.get(subj, (0, ""))[0]:
                longest[subj] = (len(hit), tag)
    for subj, (_, tag) in longest.items():
        if (subj, tag) not in hits:
            hits.append((subj, tag))
    return hits


def route_concerns(
    goal: dict[str, Any], analyzer: ExamAnalyzer
) -> dict[str, Any]:
    """每条用户问题 -> 确定性动作 + 对 Plan 的影响。"""
    concerns = normalize_concerns(goal.get("concerns"))
    actions: list[dict[str, Any]] = []
    effects: dict[str, Any] = {
        "keep_basics": False,
        "top_n_delta": 0,
        "timed_practice": False,
        "review_every_week": False,
        "start_phase": None,
        "boost_tags": [],
        "notes": [],
    }

    for concern in concerns:
        matched = [name for name, kws in _CONCERN_RULES if any(k in concern for k in kws)]
        tag_hits = _match_tags(analyzer, concern)
        primary = matched[0] if matched else ("specific" if tag_hits else "generic")

        if "basics" in matched:
            effects["keep_basics"] = True
            effects["notes"].append("保留压缩基础阶段，先概念后计算")
        if "time" in matched:
            effects["top_n_delta"] -= 2
            effects["notes"].append("每科考点数减少 2 个，只保高频")
        if "slow_calc" in matched:
            effects["timed_practice"] = True
            effects["notes"].append("增加限时训练与计算题步骤检查")
        if "memory" in matched:
            effects["review_every_week"] = True
            effects["notes"].append("每周安排一次间隔回顾")
        if "past_paper" in matched and not effects["keep_basics"]:
            effects["start_phase"] = "真题"
            effects["notes"].append("提前进入真题/错题阶段")
        for subj, tag in tag_hits:
            if {"subject": subj, "tag": tag} not in effects["boost_tags"]:
                effects["boost_tags"].append({"subject": subj, "tag": tag})
        if tag_hits:
            effects["notes"].append(f"优先安排 {'、'.join(t for _, t in tag_hits[:3])}")

        diagnosis, action, stage, checkpoint = _concern_text(
            primary, concern, tag_hits
        )
        actions.append(
            {
                "concern": concern,
                "diagnosis": diagnosis,
                "action": action,
                "stage": stage,
                "checkpoint": checkpoint,
            }
        )

    if not concerns:
        effects["notes"].append("没有填写具体问题，按通用节奏安排")
    return {"actions": actions, "effects": effects}


def _concern_text(
    primary: str, concern: str, tag_hits: list[tuple[str, str]]
) -> tuple[str, str, str, str]:
    focus = "、".join(t for _, t in tag_hits[:2])
    if primary == "basics":
        return (
            "基础阶段被压缩或跳过，导致概念题靠背、计算题靠猜",
            "保留 1–2 周基础：先理解概念与例题，再做对应选择题；大题先标记，不硬啃",
            "基础",
            "基础阶段末：每科高频考点能用自己的话解释，对应选择题 ≥70%",
        )
    if primary == "time":
        return (
            "可投入时间与考点数量不匹配，容易排太满",
            "每科只保高频考点；长尾不新学，通过真题错题补；每周留 1 天机动",
            "强化",
            "连续两周完成率 ≥80%，没有靠熬夜补进度",
        )
    if primary == "slow_calc":
        return (
            "计算题缺少步骤模板与限时训练，错误多来自过程而非概念",
            "计算/图表类考点优先；每题写步骤，限时训练后按错因二刷同类题",
            "强化",
            "近 5 年同类计算题限时正确率 ≥80%",
        )
    if primary == "memory":
        return (
            "新内容只学一遍，没有间隔回顾",
            "每周固定一次回顾：只做错题与高频清单，不新增知识点",
            "强化",
            "上周错题二刷正确率 ≥90%",
        )
    if primary == "past_paper":
        return (
            "真题练习没有限时或没有归因，正确率无法定位问题",
            "真题阶段提前：限时做→逐题标错因（概念/计算/读图/粗心）→同类题二刷",
            "真题",
            "连续两套近 5 年真题选择题正确率 ≥70%",
        )
    if primary == "specific" and focus:
        return (
            f"用户明确提到「{focus}」，需要优先确认掌握度",
            f"把「{focus}」提前到强化前段：先做近 5 年真题定位错因，再按题型补",
            "强化",
            f"「{focus}」近 5 年真题正确率 ≥80%",
        )
    return (
        "问题描述较笼统，无法直接定位到考点",
        "把问题拆成每周可检查的小目标；每周复盘一次，记录未完成原因",
        "强化",
        "每周有明确的完成/未完成判定，而不是只看学习时长",
    )


# ---------------------------------------------------------------------------
# 阶段
# ---------------------------------------------------------------------------


def _phase_shares(weeks: int) -> dict[str, float]:
    if weeks >= 24:
        return {"基础": 0.30, "强化": 0.30, "真题": 0.25, "冲刺": 0.15}
    if weeks >= 16:
        return {"基础": 0.15, "强化": 0.35, "真题": 0.35, "冲刺": 0.15}
    if weeks >= 10:
        return {"强化": 0.35, "真题": 0.45, "冲刺": 0.20}
    if weeks >= 6:
        return {"强化": 0.25, "真题": 0.50, "冲刺": 0.25}
    return {"真题": 0.60, "冲刺": 0.40}


def _allocate_counts(weeks: int, shares: dict[str, float]) -> dict[str, int]:
    raw = {k: weeks * v for k, v in shares.items()}
    counts = {k: int(math.floor(v)) for k, v in raw.items()}
    left = weeks - sum(counts.values())
    for k in sorted(raw, key=lambda x: -(raw[x] - counts[x]))[:left]:
        counts[k] += 1
    # 至少给有份额的阶段 1 周；释放的周数从最大阶段扣
    for k, v in list(counts.items()):
        if v == 0 and shares.get(k, 0) > 0:
            donor = max(counts, key=lambda x: counts[x])
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[k] = 1
    return {k: v for k, v in counts.items() if v > 0}


def _start_phase(goal: dict[str, Any], weeks: int, effects: dict[str, Any]) -> str:
    progress = _progress(goal)
    values = list(progress.values())
    if effects.get("keep_basics") or any(
        p in ("未开始", "一轮中") for p in values
    ):
        start = "基础" if weeks >= 4 else "强化"
    elif values and all(p in ("二轮中", "二轮完") for p in values):
        start = "冲刺" if all(p == "二轮完" for p in values) else "真题"
    else:
        start = "强化"
    forced = effects.get("start_phase")
    if forced == "真题" and weeks <= 12 and start in ("基础", "强化"):
        start = "真题"
    return start


def _progress(goal: dict[str, Any]) -> dict[str, str]:
    raw = goal.get("progress") or {}
    return {
        s: str(raw.get(s) or "未开始")
        for s in SUBJECTS
    }


def _phase_meta(name: str, focus: list[str], daily: float) -> tuple[list[str], str]:
    if name == "基础":
        return (
            [
                "本阶段选定的高频考点：概念 + 例题 + 对应选择题",
                "大题先标记清单，不在第一轮硬啃",
                "每章结束用 10 分钟复述框架",
            ],
            "对应选择题 ≥70%；能不看书画出该科框架",
        )
    if name == "强化":
        return (
            [
                "每科高频考点刷近 5 年真题中的对应题（先选择后大题）",
                "一轮标记的大题过一遍，计算/图表题写出步骤",
                "错题按概念 / 计算 / 读图 / 粗心分类",
            ],
            "近 5 年对应选择题 ≥80%；计算/图表题能独立写出步骤",
        )
    if name == "真题":
        return (
            [
                "先二刷错题，再做真题套卷；近 3 年严格计时",
                "早年真题可只做选择/重点题，不必整套硬啃",
                "每套逐题归因，留最近 2 套做考前模考",
            ],
            "连续两套计时套卷完成；错题归因；同类二刷 ≥90%",
        )
    return (
        [
            "3 套全真模考，按考试时间做",
            "错题三刷 + 高频清单过一遍",
            "不再开新知识点，调整作息到考试时段",
        ],
        "模考分数稳定；高频清单无盲区；不新学",
    )


def allocate_phases(
    budget: dict[str, Any],
    goal: dict[str, Any],
    effects: dict[str, Any],
) -> list[dict[str, Any]]:
    weeks = int(budget["remaining_weeks"])
    start = _start_phase(goal, weeks, effects)
    baseline = _phase_shares(weeks)
    ordered = PHASES[PHASES.index(start):]
    baseline = {k: v for k, v in baseline.items() if k in ordered}

    counts: dict[str, int] = {}
    if start == "基础" and "基础" not in baseline:
        # 10–15 周的表里没有基础；进度需要时先塞 1–2 周
        base_weeks = 1 if weeks <= 12 else 2
        counts["基础"] = base_weeks
        rest = weeks - base_weeks
        rest_shares = {
            k: v
            for k, v in _phase_shares(weeks).items()
            if k in ordered and k != "基础"
        }
        if not rest_shares:  # 防御：只剩基础
            counts["基础"] = weeks
        else:
            total = sum(rest_shares.values())
            counts.update(_allocate_counts(rest, {k: v / total for k, v in rest_shares.items()}))
    else:
        total = sum(baseline.values()) or 1
        counts = _allocate_counts(weeks, {k: v / total for k, v in baseline.items()})

    # 如果表里有基础、但最早阶段是强化，baseline 已删掉基础，无需再处理
    counts = {k: v for k, v in counts.items() if v > 0}
    # 防御：四舍五入后周数不一致时补给最大阶段
    diff = weeks - sum(counts.values())
    if diff:
        k = max(counts, key=lambda x: counts[x])
        counts[k] += diff

    daily = float(budget["daily_hours"])
    start_date = date.fromisoformat(budget["start_date"])
    exam = date.fromisoformat(budget["exam_date"])
    cursor = start_date
    out: list[dict[str, Any]] = []
    for name in PHASES:
        n = counts.get(name, 0)
        if n <= 0:
            continue
        end = min(cursor + timedelta(days=n * 7 - 1), exam)
        progress = _progress(goal)
        if name == "基础":
            focus = [s for s, p in progress.items() if p in ("未开始", "一轮中")] or list(
                goal.get("weak_subjects") or SUBJECTS
            )
        elif name == "冲刺":
            focus = list(goal.get("weak_subjects") or SUBJECTS)
        else:
            focus = SUBJECTS
        deliverables, checkpoint = _phase_meta(name, focus, daily)
        out.append(
            {
                "name": name,
                "start": cursor.isoformat(),
                "end": end.isoformat(),
                "weeks": n,
                "focus": focus,
                "daily_hours": daily,
                "deliverables": deliverables,
                "checkpoint": checkpoint,
            }
        )
        cursor = end + timedelta(days=1)
    if out:
        out[-1]["end"] = exam.isoformat()
        out[-1]["weeks"] = max(
            1, math.ceil((exam - date.fromisoformat(out[-1]["start"])).days / 7)
        )
        # 修正总周数（最后一段按实际天数算）
        diff = weeks - sum(p["weeks"] for p in out)
        if diff:
            out[-1]["weeks"] = max(1, out[-1]["weeks"] + diff)
    return out


# ---------------------------------------------------------------------------
# 科目份额
# ---------------------------------------------------------------------------


def allocate_subject_hours(
    goal: dict[str, Any], effects: dict[str, Any]
) -> dict[str, float]:
    weak = list(goal.get("weak_subjects") or [])
    progress = _progress(goal)
    scope = str(goal.get("scope") or "full_408")
    boost = {}
    for b in effects.get("boost_tags") or []:
        boost[b["subject"]] = boost.get(b["subject"], 0) + 0.04

    if scope == "single_subject":
        weak_list = weak or ["操作系统"]
        # 单科模式也可能多选薄弱科目；每个薄弱科目至少 30%，
        # 否则规则会把它判成 underweight（0.29 就是典型的取整踩线）。
        weak_share = max(0.30, 0.65 / len(weak_list))
        if weak_share * len(weak_list) > 0.90:
            weak_share = 0.90 / len(weak_list)
        others = [s for s in SUBJECTS if s not in weak_list]
        base = sum(SUBJECT_SCORES[s] for s in others) or 1
        rest = max(0.10, 1.0 - weak_share * len(weak_list))
        shares = {s: rest * SUBJECT_SCORES[s] / base for s in others}
        for s in weak_list:
            shares[s] = weak_share
    else:
        shares = {s: SUBJECT_SCORES[s] / 150 for s in SUBJECTS}
        for s in weak:
            shares[s] = shares.get(s, 0) + 0.08
        for s, p in progress.items():
            if p in ("未开始", "一轮中"):
                shares[s] = shares.get(s, 0) + 0.05
            elif p == "二轮完":
                shares[s] = shares.get(s, 0) - 0.05
        for s, bonus in boost.items():
            shares[s] = shares.get(s, 0) + bonus

    shares = {s: max(0.05, min(0.60, v)) for s, v in shares.items() if v > 0}
    total = sum(shares.values()) or 1
    shares = {s: v / total for s, v in shares.items()}

    # 规则 weak_subject_underweight 在 <0.3 时报警。把薄弱科目钉在 30%，
    # 剩下的时间再按原比例分给其他科目；先归一化再抬到 0.3 会被第二次
    # 归一化拉回 0.29，导致 golden Plan 误报。
    weak_in_scope = [s for s in weak if s in shares]
    non_weak = [s for s in shares if s not in weak_in_scope]
    if weak_in_scope:
        for s in weak_in_scope:
            shares[s] = max(shares[s], 0.30)
        remaining = 1.0 - sum(shares[s] for s in weak_in_scope)
        if non_weak and remaining > 0:
            base = sum(shares[x] for x in non_weak) or 1
            for x in non_weak:
                shares[x] = remaining * shares[x] / base
        elif remaining <= 0:
            total = sum(shares.values()) or 1
            shares = {s: v / total for s, v in shares.items()}
    return {s: round(v, 4) for s, v in shares.items()}


# ---------------------------------------------------------------------------
# 考点与周计划
# ---------------------------------------------------------------------------


def build_priority_kps(
    goal: dict[str, Any],
    analyzer: ExamAnalyzer,
    budget: dict[str, Any],
    effects: dict[str, Any],
) -> tuple[dict[str, list[TagROI]], dict[str, dict[str, Any]]]:
    weeks = int(budget["remaining_weeks"])
    scope = str(goal.get("scope") or "full_408")
    weak = list(goal.get("weak_subjects") or [])
    boost_by_subject: dict[str, list[str]] = {}
    for b in effects.get("boost_tags") or []:
        boost_by_subject.setdefault(b["subject"], []).append(b["tag"])

    kps: dict[str, list[TagROI]] = {}
    coverage: dict[str, dict[str, Any]] = {}
    for subject in SUBJECTS:
        n = None
        if scope == "single_subject" and subject not in weak:
            n = 4  # 非薄弱科目只保留少量维持性考点
        else:
            base_n = 10 if weeks <= 8 else (15 if weeks <= 16 else 20)
            n = max(6, base_n + int(effects.get("top_n_delta") or 0))
        rows = select_high_value(
            analyzer,
            subject,
            remaining_weeks=weeks,
            boost_tags=boost_by_subject.get(subject, []),
            top_n=n,
        )
        rows = prereq_order(analyzer, rows)
        kps[subject] = rows
        coverage[subject] = subject_coverage(analyzer, subject, rows)

    # 权重按科目内归一化：规则 uncited_priority 的阈值是 0.15，
    # 如果跨 40 个考点归一化，前几名也只有 0.03，规则会永远不触发。
    for rows in kps.values():
        total_roi = sum(r.roi for r in rows) or 1
        for r in rows:
            r.weight = round(r.roi / total_roi, 3)
    return kps, coverage


def _round_hours(x: float) -> float:
    return max(0.5, round(x * 2) / 2)


def _rebalance_tasks(
    tasks: list[dict[str, Any]], protected: set[str], weekly_hours: float
) -> None:
    """把每周任务凑到 weekly_hours，但不动薄弱科目的最低份额。"""
    diff = round(weekly_hours - sum(t["hours"] for t in tasks), 1)
    steps = int(round(abs(diff) / 0.5))
    for _ in range(steps):
        pool = [t for t in tasks if t["subject"] not in protected] or tasks
        if diff > 0:
            t = min(pool, key=lambda x: x["hours"])
            t["hours"] = round(t["hours"] + 0.5, 2)
            diff -= 0.5
        else:
            t = max(pool, key=lambda x: x["hours"])
            if t["hours"] <= 0.5:
                break
            t["hours"] = round(t["hours"] - 0.5, 2)
            diff += 0.5


def _task_text(
    phase: str,
    subject: str,
    kps: list[TagROI],
    *,
    timed: bool,
    review: bool,
) -> tuple[str, str, str]:
    names = "、".join(r.tag for r in kps) or subject
    must = "、".join(
        [m for r in kps[:2] for m in (r.must_learn or [])][:4]
    ) or names
    later = "、".join([m for r in kps[:2] for m in (r.later or [])][:3])
    hint = SUBJECT_METHODS.get(subject, {})
    if phase == "基础":
        content = f"先理解 {names}：{must}" + (f"；可后置 {later}" if later else "")
        method = (
            "对应一轮：概念 + 例题 → 课后选择题；先看懂标准解题步骤，"
            "不急着独立写大题；每学完一节立刻做对应选择题；"
            "错题回到概念/笔记位置，不攒错题"
            + (f"；{hint['method']}" if hint.get("method") else "")
        )
        accept = f"{names} 能用自己的话解释；对应选择题 ≥70%"
    elif phase == "强化":
        content = f"按题型刷 {names}：{must}" + (f"；可后置 {later}" if later else "")
        method = (
            "对应二轮：重心从看懂转向做题；第二轮内容快过，重点看一轮卡住和"
            "错题对应知识点；大题上手；固定题型专项练；选择题二刷要能说出"
            "每个选项为什么对/错；错题反复出现就回对应知识点"
            + (f"；{hint['method']}" if hint.get("method") else "")
        )
        accept = f"近 5 年 {names} 选择题 ≥80%；计算/图表题能独立写出步骤"
    elif phase == "真题":
        content = f"真题套卷（{subject}）" + (f"；补漏 {names}" if kps else "")
        method = (
            "对应三轮：从 09 年开始；前期分科目/题型适应，近 10 年套卷严格计时（时间不够至少近 3–5 年）；"
            "每套逐题复盘吃透考点，标记选择题反复考点/大题固定角度；"
            "二刷检验能否独立写完整步骤；模拟 3–5 套，重心仍真题"
        )
        accept = "套卷 3 小时计时完成；错题归因完成；同类题二刷 ≥90%"
    else:
        content = f"全真模考 + 高频清单：{names}"
        method = (
            "对应四轮：知识点快翻高频和反复错；模拟 3–5 套；"
            "时间不够宁可少模拟也要真题再刷；考前收窄，只过错题、"
            "核心计算、流程步骤、标准题型框架；最后一周反复过"
        )
        accept = "模考分数稳定；高频清单无盲区"
    if timed:
        method += "；本任务限时"
        accept = accept.replace("≥", "限时 ≥")
    if review:
        method += "；开始前先回顾上周错题"
    return content, method, accept


def build_weekly_plan(
    goal: dict[str, Any],
    budget: dict[str, Any],
    phases: list[dict[str, Any]],
    shares: dict[str, float],
    kps_by_subject: dict[str, list[TagROI]],
    effects: dict[str, Any],
) -> list[dict[str, Any]]:
    weeks = int(budget["remaining_weeks"])
    weekly_hours = float(budget["weekly_hours"])
    start = date.fromisoformat(budget["start_date"])
    exam = date.fromisoformat(budget["exam_date"])
    scope = str(goal.get("scope") or "full_408")
    weak = list(goal.get("weak_subjects") or [])
    main_subject = weak[0] if weak else "操作系统"
    subjects = [s for s in shares if shares[s] > 0]
    if scope == "single_subject":
        # 主科多做一点，其他科只维持
        subjects = [main_subject] + [s for s in subjects if s != main_subject]

    learning_weeks = sum(p["weeks"] for p in phases if p["name"] != "冲刺") or 1
    queues: dict[str, list[TagROI]] = {s: list(kps_by_subject.get(s) or []) for s in subjects}
    per_task = {
        s: max(1, math.ceil(len(queues[s]) / learning_weeks)) for s in subjects
    }

    # 阶段 -> 周序号
    phase_of_week: dict[int, str] = {}
    idx = 1
    for p in phases:
        for _ in range(p["weeks"]):
            if idx <= weeks:
                phase_of_week[idx] = p["name"]
            idx += 1
    if not phase_of_week:
        phase_of_week = {i: "强化" for i in range(1, weeks + 1)}

    # 学习周按经验贴顺序分块：DS → 组成+OS → 网络；
    # 真题/冲刺主攻薄弱科目。极短周期分不到块时，至少标出本周主攻。
    learning_weeks = sorted(
        w for w, p in phase_of_week.items() if p in ("基础", "强化")
    )
    focus_map: dict[int, list[str]] = {}
    if learning_weeks:
        for i, w in enumerate(learning_weeks):
            block = SUBJECT_BLOCKS[min(2, (i * 3) // len(learning_weeks))]
            focus_map[w] = list(block)
    for w, p in phase_of_week.items():
        if p in ("真题", "冲刺"):
            focus_map[w] = weak[:2] or ["四科"]

    out: list[dict[str, Any]] = []
    for w in range(1, weeks + 1):
        phase = phase_of_week.get(w, phases[-1]["name"] if phases else "强化")
        w_start = start + timedelta(days=(w - 1) * 7)
        w_end = min(w_start + timedelta(days=6), exam)
        tasks: list[dict[str, Any]] = []

        week_hours = {s: _round_hours(weekly_hours * shares[s]) for s in subjects}
        # 0.30 × 24 = 7.2，四舍五入成 7.0 会让实际占比掉到 0.292，
        # 触发 weak_subject_underweight。薄弱科目向上取到 0.5h 粒度。
        if scope == "full_408":
            for s in subjects:
                if s in weak:
                    week_hours[s] = max(
                        week_hours[s], math.ceil(0.30 * weekly_hours * 2) / 2
                    )
        # 本周主攻要真的多给时间，否则「主攻」只是文案。
        # 从非主攻、非薄弱的科目里拿 10% 周预算，按可扣余量比例扣。
        focus_now = focus_map.get(w, [])
        focus_set = {s for s in subjects if s in focus_now}
        donors = [s for s in subjects if s not in focus_set and s not in weak]
        if focus_set and donors:
            spare = sum(max(0.0, week_hours[s] - 0.5) for s in donors)
            bonus = min(weekly_hours * 0.10, spare)
            if bonus >= 0.5:
                per = bonus / len(focus_set)
                for s in focus_set:
                    week_hours[s] = round((week_hours[s] + per) * 2) / 2
                for s in donors:
                    take = bonus * (max(0.0, week_hours[s] - 0.5) / spare) if spare else 0
                    week_hours[s] = max(0.5, round((week_hours[s] - take) * 2) / 2)

        for subject in subjects:
            hours = week_hours[subject]
            queue = queues[subject]
            take: list[TagROI] = []
            if phase in ("基础", "强化"):
                take = queue[: per_task[subject]]
                queues[subject] = queue[len(take):]
            elif phase == "真题" and queue:
                # 真题阶段也继续消化队列，避免「列了 10 个考点但只排了 6 个」
                take = queue[: per_task[subject]]
                queues[subject] = queue[len(take):]
            elif phase == "冲刺":
                take = (kps_by_subject.get(subject) or [])[:5]

            if not take:
                # 队列空了：用该科最高频考点做回顾/模考，不再引入新考点
                take = (kps_by_subject.get(subject) or [])[:3]
            content, method, accept = _task_text(
                phase,
                subject,
                take,
                timed=bool(effects.get("timed_practice")),
                review=bool(effects.get("review_every_week")),
            )
            if subject in focus_map.get(w, []):
                content = "本周主攻：" + content
            kp_names: list[str] = []
            for i, r in enumerate(take):
                kp_names.append(r.tag)
                # 只给第一个考点带教材标题，避免 prerequisite_order 把同周多个
                # 有前置关系的标题误判成 medium
                if i == 0 and r.kp_titles:
                    kp_names.append(r.kp_titles[0])
            evidence: list[str] = []
            for r in take:
                for e in r.evidence:
                    if e not in evidence:
                        evidence.append(e)
            tasks.append(
                {
                    "subject": subject,
                    "hours": hours,
                    "kps": list(dict.fromkeys(kp_names)),
                    "content": content,
                    "method": method,
                    "acceptance": accept,
                    "evidence": evidence[:3],
                }
            )

        # 保证每周总小时 = weekly_hours，但优先保证薄弱科目不被削减
        if tasks:
            _rebalance_tasks(tasks, set(weak), weekly_hours)
        out.append(
            {
                "week": w,
                "stage": phase,
                "focus": focus_map.get(w, []),
                "start": w_start.isoformat(),
                "end": w_end.isoformat(),
                "tasks": tasks,
            }
        )
    return out


# ---------------------------------------------------------------------------
# 经验贴式方法建议（不写教材/网课/老师）
# ---------------------------------------------------------------------------


def build_guide_template(
    goal: dict[str, Any],
    budget: dict[str, Any],
    phases: list[dict[str, Any]],
    shares: dict[str, float],
    kps_by_subject: dict[str, list[TagROI]],
    concern_actions: list[dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    progress = _progress(goal)
    weak = list(goal.get("weak_subjects") or [])
    phase_names = " → ".join(p["name"] for p in phases)
    overview = (
        f"从 {budget['start_date']} 到 {budget['exam_date']} 共 "
        f"{budget['remaining_days']} 天（{budget['remaining_weeks']} 周）；"
        f"每天 {budget['daily_hours']:g}h × 每周 {budget['study_days_per_week']} 天 "
        f"= {budget['weekly_hours']:g}h/周，总可用 {budget['total_hours']:g}h。"
        f"当前进度："
        + "、".join(f"{s}{progress.get(s, '未开始')}" for s in SUBJECTS)
        + f"。路线按 {phase_names} 安排（对应经验贴的一轮 → 二轮 → 三轮 → 四轮，"
        "但周数按你的剩余时间压缩，不照搬 6 个月）；薄弱科目："
        + ("、".join(weak) if weak else "无特别指定")
        + "。"
    )

    phase_advice = []
    for p in phases:
        name = p["name"]
        if name == "基础":
            text = (
                "对应一轮：先看懂概念和标准解题步骤，不急着独立写大题；"
                "每学完一节立刻做对应选择题；错题回到概念/笔记位置，不攒错题。"
            )
        elif name == "强化":
            text = (
                "对应二轮：重心从看懂转向做题；第二轮内容快过，重点看一轮卡住和"
                "错题对应知识点；大题上手；固定题型专项练；选择题二刷要能说出"
                "每个选项为什么对/错。"
            )
        elif name == "真题":
            text = (
                "对应三轮：从 09 年开始；前期分科目/题型适应，近 10 年套卷严格计时（时间不够至少近 3–5 年）；"
                "每套逐题复盘吃透考点；二刷检验能否独立写完整步骤；模拟 3–5 套，"
                "重心仍真题。"
            )
        else:
            text = (
                "对应四轮：知识点快翻高频和反复错；模拟 3–5 套；时间不够宁可少模拟"
                "也要真题再刷；考前收窄，只过错题、核心计算、流程步骤、标准题型；"
                "最后一周反复过。"
            )
        phase_advice.append({"stage": name, "advice": text})

    subject_strategy = []
    for s in SUBJECTS:
        rows = kps_by_subject.get(s) or []
        top = "、".join(r.tag for r in rows[:3]) or "暂无"
        cov = coverage.get(s, {}).get("ratio", 0)
        tail = (
            "先补概念与选择题。"
            if progress.get(s) in ("未开始", "一轮中")
            else "直接进入刷题与真题。"
        )
        tc: dict[str, int] = {}
        for r in rows[:8]:
            for key, val in (r.types or {}).items():
                tc[key] = tc.get(key, 0) + val
        total_types = sum(tc.values()) or 1
        calc_fig = (tc.get("calc", 0) + tc.get("figure", 0)) / total_types
        sm = SUBJECT_METHODS.get(s, {})
        strategy = (
            f"主线是{sm.get('focus', '高频考点')}：{sm.get('method', '')}。"
            f"优先 {top}；本轮按 tag 覆盖该科历史考查约 {cov:.0%}。"
            f"你当前进度是{progress.get(s, '未开始')}，{tail}"
        )
        if calc_fig >= 0.3:
            strategy += f" 该科选中考点里计算/图表占 {calc_fig:.0%}，优先练步骤和画图。"
        subject_strategy.append(
            {
                "subject": s,
                "strategy": strategy,
                "pitfalls": list(sm.get("pitfalls") or [])[:3]
                + ["错题只抄答案，不写错因"],
            }
        )

    days = budget["study_days_per_week"]
    daily = budget["daily_hours"]
    if daily >= 4:
        split = "例如 2h 新内容 + 1.5h 刷题 + 0.5h 错题复盘"
    elif daily >= 2:
        split = "例如 1h 新内容 + 0.75h 刷题 + 0.25h 错题复盘"
    else:
        split = "时间少时先保证 30min 刷题 + 15min 错题，再挤新内容"
    rest = "每周至少留半天机动" if days >= 7 else "每周留 1 天机动或复盘"
    weekly_routine = (
        f"每周学 {days} 天、每天 {daily:g}h；{split}；{rest}。"
        "错题优先于新题；复盘只做错题与高频清单，不新增知识点。"
    )
    pitfalls = [
        "只看不练，选择题看着会、做起来错",
        "第一轮只看不练，后面还得返工",
        "错题攒着，想回头再补",
        "大题留到最后，前面只做选择题",
        "真题只做近几年，早年为省时间完全不碰",
        "真题做了不限时，考场节奏没练过",
        "模拟题做太多，挤掉真题二刷",
        "考前还开新知识点",
        "计划排太满，没有机动日，一次拖延整周崩",
        "熬夜补进度，白天效率下降",
    ]
    checkpoints = [f"{p['name']}：{p['checkpoint']}" for p in phases]
    checkpoints += [
        "选择题二刷：能说出每个选项为什么对/错",
        "大题：能独立写完整步骤，不只记答案",
    ]
    if concern_actions:
        checkpoints.append(
            "用户问题：每条都要有可检查的结果，见「你的问题 → 计划怎么解决」"
        )

    rationale = (
        f"按剩余 {budget['remaining_weeks']} 周压缩为 {phase_names}；"
        "科目时间按 408 分值权重分配，薄弱科目与进度落后的科目加权；"
        "考点按频次、近 5 年出现、计算/图表占比排序，短周期只保高频，长尾不新学。"
    )[:300]
    return {
        "overview": overview,
        "phase_advice": phase_advice,
        "subject_order": (
            "起步晚时建议一门一门过：数据结构 → 组成原理 + 操作系统 → 计算机网络；"
            "数据结构是其他三门的基础，组成与 OS 交叉多，网络最后（杂但浅）。"
            "8 周这种极短周期拆不开，采用「每周主攻 1–2 科 + 其他科维持」，"
            "不四科齐头并进。"
        ),
        "subject_strategy": subject_strategy,
        "weekly_routine": weekly_routine,
        "pitfalls": pitfalls,
        "checkpoints": checkpoints,
        "note": (
            "以下是根据公开经验贴常见做法整理的通用方法，不是提分承诺；"
            "请按自己的时间、进度和薄弱科目裁剪。"
        ),
    }, rationale


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def _challenge_dict(c: Any) -> dict[str, Any]:
    if hasattr(c, "to_dict"):
        return c.to_dict()
    return dict(c) if isinstance(c, dict) else {}


def build_detailed_plan(
    goal: dict[str, Any],
    analyzer: ExamAnalyzer | None = None,
    challenges: Iterable[Any] | None = None,
) -> dict[str, Any]:
    analyzer = analyzer or get_analyzer()
    goal = dict(goal or {})
    goal.setdefault("scope", "full_408")
    goal.setdefault("weak_subjects", [])
    goal.setdefault("progress", {s: "未开始" for s in SUBJECTS})
    goal.setdefault("daily_hours", 4)
    goal.setdefault("study_days_per_week", 6)
    goal.setdefault("target_score", 110)
    goal.setdefault("concerns", [])

    routed = route_concerns(goal, analyzer)
    effects = routed["effects"]

    # 规则挑战 -> 生成器的修正（主要针对缺失考点/薄弱科目）
    for c in challenges or []:
        d = _challenge_dict(c)
        rule = d.get("rule")
        ev = d.get("evidence") or {}
        if rule == "missing_high_freq_kp" and ev.get("tag"):
            subj = ev.get("subject") or (
                goal["weak_subjects"][0] if goal["weak_subjects"] else "操作系统"
            )
            item = {"subject": subj, "tag": ev["tag"]}
            if item not in effects["boost_tags"]:
                effects["boost_tags"].append(item)
        elif rule == "weak_subject_underweight" and ev.get("subject"):
            if ev["subject"] not in goal["weak_subjects"]:
                goal["weak_subjects"].append(ev["subject"])

    budget = compute_time_budget(goal)
    phases = allocate_phases(budget, goal, effects)
    shares = allocate_subject_hours(goal, effects)
    kps_by_subject, coverage = build_priority_kps(goal, analyzer, budget, effects)
    weekly = build_weekly_plan(goal, budget, phases, shares, kps_by_subject, effects)
    guide, rationale = build_guide_template(
        goal, budget, phases, shares, kps_by_subject, routed["actions"], coverage
    )

    goal_out = {
        **goal,
        "start_date": budget["start_date"],
        "exam_date": budget["exam_date"],
        "remaining_days": budget["remaining_days"],
        "remaining_weeks": budget["remaining_weeks"],
        "daily_hours": budget["daily_hours"],
        "study_days_per_week": budget["study_days_per_week"],
        "weekly_hours": budget["weekly_hours"],
        "total_hours": budget["total_hours"],
        "subjects": SUBJECTS,
        "progress": _progress(goal),
    }
    priority = [r.to_dict() for s in SUBJECTS for r in kps_by_subject.get(s, [])]
    plan = {
        "goal": goal_out,
        "stages": phases,
        "priority_kps": priority,
        "weekly_plan": weekly,
        "guide": guide,
        "concern_actions": routed["actions"],
        "rationale": rationale,
        "risk_notes": [
            "ROI 是频次/近年/题型/成本的启发式排序，不是押题",
            "本周期每科按 tag 覆盖历史考查约 "
            + "、".join(f"{s}{coverage.get(s, {}).get('ratio', 0):.0%}" for s in SUBJECTS),
        ],
        "open_questions": [],
        "coverage": coverage,
        "meta": {"plan_source": "deterministic", "guide_source": "template"},
    }
    return plan