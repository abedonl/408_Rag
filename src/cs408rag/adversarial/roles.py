"""Planner / Critic / Expert 三个角色。

每个角色只能输出固定 JSON schema，没有自由文本字段——这是防闲聊的第一道闸。
权限差异（而不是 prompt 措辞）才是角色存在的理由：

    Planner  唯一能生成/修改 Plan；结构由 planner.py 用真实数据算出，
              LLM 只把方法建议写顺（不编造阶段、小时、考点、真题号）
    Critic   不看 Planner 的推理过程，只挑规则没覆盖的问题，不能改 Plan
    Expert   只在争议时介入，用真题/前置边做数据裁定，不能改 Plan

LLM 失败（空正文/JSON 解析失败）一律降级，绝不抛异常。
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from cs408rag.adversarial.planner import build_detailed_plan
from cs408rag.adversarial.rules import Challenge
from cs408rag.adversarial.schema import plan_digest
from cs408rag.agent.exam_analysis import ExamAnalyzer, get_analyzer
from cs408rag.config import settings
from cs408rag.generation.llm import LLMClient, get_llm

# 用户明确要求「只写方法，不提资料」；这些词出现就退回模板
FORBIDDEN_GUIDE_TERMS = ("王道", "沈悦", "网课", "老师", "教材", "参考书", "视频课")


def _extract_json(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _role_client(model: str) -> LLMClient:
    """按角色创建客户端；与主模型一致时复用全局单例（共享 token 统计）。

    角色专用客户端必须把主模型设为备用：Critic 用 glm 时如果返回空正文，
    原实现会拿同一个 glm 再试一次（等于没有降级）。跨模型兜底才有意义。
    """
    main = get_llm()
    if not model or model == main.model:
        return main
    return LLMClient(model=model, backup_model=main.model)


def _ref_exists(plan: dict[str, Any], ref: str) -> bool:
    """evidence_ref 必须指向 Plan 中真实存在的字段路径。"""
    patterns = (
        (r"goal\.([A-Za-z_]\w*)", "goal"),
        (r"priority_kps\[(\d+)\]\.([A-Za-z_]\w*)", "kp"),
        (r"weekly_plan\[(\d+)\]\.tasks\[(\d+)\]\.([A-Za-z_]\w*)", "task"),
        (r"stages\[(\d+)\]\.([A-Za-z_]\w*)", "stage"),
        (r"concern_actions\[(\d+)\]\.([A-Za-z_]\w*)", "concern"),
    )
    for pattern, kind in patterns:
        m = re.fullmatch(pattern, ref)
        if not m:
            continue
        if kind == "goal":
            goal = plan.get("goal")
            return isinstance(goal, dict) and m.group(1) in goal
        if kind == "kp":
            kps = plan.get("priority_kps") or []
            i = int(m.group(1))
            return i < len(kps) and isinstance(kps[i], dict) and m.group(2) in kps[i]
        if kind == "stage":
            stages = plan.get("stages") or []
            i = int(m.group(1))
            return i < len(stages) and isinstance(stages[i], dict) and m.group(2) in stages[i]
        if kind == "concern":
            actions = plan.get("concern_actions") or []
            i = int(m.group(1))
            return i < len(actions) and isinstance(actions[i], dict) and m.group(2) in actions[i]
        weeks = plan.get("weekly_plan") or []
        wi, ti = int(m.group(1)), int(m.group(2))
        if wi >= len(weeks) or not isinstance(weeks[wi], dict):
            return False
        tasks = weeks[wi].get("tasks") or []
        return ti < len(tasks) and isinstance(tasks[ti], dict) and m.group(3) in tasks[ti]
    return False


def _guide_context(plan: dict[str, Any]) -> str:
    """给 LLM 的素材：只给算好的事实，避免它自己发明数字。"""
    goal = plan.get("goal") or {}
    lines = [
        "已算好的事实（不得修改）：",
        f"- 考试日期 {goal.get('exam_date')}，剩余 {goal.get('remaining_weeks')} 周，"
        f"每天 {goal.get('daily_hours')}h × {goal.get('study_days_per_week')} 天 = "
        f"{goal.get('weekly_hours')}h/周，总 {goal.get('total_hours')}h",
        f"- 范围 {goal.get('scope')}，薄弱科目 "
        f"{'、'.join(goal.get('weak_subjects') or []) or '无'}",
        "- 阶段：" + "；".join(
            f"{s.get('name')} {s.get('start')}~{s.get('end')}（{s.get('weeks')} 周）"
            for s in plan.get("stages") or []
        ),
        "- 高性价比考点（前 10）：" + "；".join(
            f"{k.get('subject')} {k.get('tag')} 共{k.get('frequency')}次"
            f"/近5年{k.get('recent_5y')}次"
            for k in (plan.get("priority_kps") or [])[:10]
        ),
        "- 用户问题与动作：" + "；".join(
            f"{a.get('concern')}→{a.get('action')}"
            for a in plan.get("concern_actions") or []
        ),
        "",
        "模板建议（请改写，不要改变事实）：",
        json.dumps(plan.get("guide") or {}, ensure_ascii=False),
    ]
    return "\n".join(lines)


class Planner:
    # 经验贴文案改写默认关闭：这套模型（agnes-3.0-flash）思考 token 占地大，
    # 实测改写耗时 90s+ 且仍回退模板，性价比不如直接用模板文案。
    # 每周理由（_enrich_weekly）仍然走 LLM，那才是「Agent 参与」的部分。
    ENRICH_GUIDE = False

    def __init__(
        self, llm: LLMClient | None = None, analyzer: ExamAnalyzer | None = None
    ) -> None:
        self.llm = llm or _role_client(settings.adv_planner_model or settings.llm_model)
        self.analyzer = analyzer or get_analyzer()
        # 文案是可选增强：单独一个短重试客户端，失败要快，不能把 90s 预算耗光
        if isinstance(self.llm, LLMClient):
            self.guide_llm = LLMClient(
                model=self.llm.model,
                backup_model=self.llm.backup_model,
                timeout=45.0,
                max_retries=1,
            )
        else:
            self.guide_llm = self.llm

    def generate(
        self,
        goal: dict[str, Any],
        *,
        prev_plan: dict[str, Any] | None = None,
        challenges: list[Challenge] | None = None,
    ) -> tuple[dict[str, Any], bool]:
        """返回 (plan, used_llm)。

        结构永远由 `build_detailed_plan` 生成；LLM 只重写 guide。
        因此空回答/坏 JSON 不会让 Plan 缺失，只是文案退回模板。
        """
        plan = build_detailed_plan(goal, self.analyzer, challenges)
        plan.setdefault("meta", {})["guide_attempted"] = False

        # 修订时结构会重算，但 guide 是方法建议，不必每版重新调 LLM：
        # 复用上一版文案，既省一次调用，也避免 429 把预算耗光。
        if prev_plan and prev_plan.get("guide"):
            prev_meta = prev_plan.get("meta") or {}
            plan["guide"] = prev_plan["guide"]
            plan["meta"]["guide_source"] = prev_meta.get("guide_source", "template")
            # 每周理由同样复用：结构条数通常不变，重新调 LLM 只会白烧预算。
            prev_reasons = {
                int(w.get("week")): w.get("reason")
                for w in (prev_plan.get("weekly_plan") or [])
                if isinstance(w, dict)
                and isinstance(w.get("week"), (int, float))
                and w.get("reason")
            }
            for week in plan.get("weekly_plan") or []:
                w_no = week.get("week")
                if isinstance(w_no, (int, float)) and prev_reasons.get(int(w_no)):
                    week["reason"] = prev_reasons[int(w_no)]
            if prev_reasons:
                plan["meta"]["weekly_source"] = prev_meta.get("weekly_source", "llm")
            return plan, False

        guide = self._enrich_guide(plan, challenges) if self.ENRICH_GUIDE else None
        plan["meta"]["guide_attempted"] = self.ENRICH_GUIDE
        used_llm = False
        if guide:
            plan["guide"] = guide
            plan["meta"]["guide_source"] = "llm"
            plan["meta"]["plan_source"] = "llm_enriched"
            used_llm = True

        # 每周一句「为什么这么排」：只补 reason 字段，不改小时/考点/阶段，
        # 这样相邻周不会因为骨架兜底而读起来完全一样。
        reasons = self._enrich_weekly(plan)
        if reasons:
            for week in plan.get("weekly_plan") or []:
                w_no = week.get("week")
                r = reasons.get(int(w_no)) if isinstance(w_no, (int, float)) else None
                if r:
                    week["reason"] = r
            plan["meta"]["weekly_source"] = "llm"
            used_llm = True
        return plan, used_llm

    def _enrich_weekly(self, plan: dict[str, Any]) -> dict[int, str] | None:
        """让 LLM 为每一周写一句推进逻辑，返回 {周次: 理由}。

        输入只给已算好的骨架（周次/阶段/小时/考点），输出也只要理由文本，
        数字与考点一律不经过 LLM，规则校验仍对原结构生效。
        """
        weeks = plan.get("weekly_plan") or []
        if not weeks:
            return None
        goal = plan.get("goal") or {}
        out: dict[int, str] = {}
        # 分批生成：一次让模型写全部周理由会被输出预算截断（实测返回残缺 JSON）；
        # 只给前 8 周写理由，后续周在看板里本就折叠，避免每次规划多等 1 分钟。
        for start in range(0, min(len(weeks), 8), 8):
            chunk = weeks[start : start + 8]
            lines: list[str] = []
            for w in chunk:
                focus = "、".join(w.get("focus") or []) or "全部科目"
                parts: list[str] = []
                for t in w.get("tasks") or []:
                    kps = "、".join((t.get("kps") or [])[:3])
                    parts.append(
                        f"{t.get('subject')}{float(t.get('hours') or 0):g}h "
                        f"{kps + '：' if kps else ''}{str(t.get('content') or '')[:40]}"
                    )
                lines.append(
                    f"第{w.get('week')}周[{w.get('stage')}] 主攻{focus}｜"
                    + "；".join(parts)
                )
            user = [
                f"范围 {goal.get('scope')}，薄弱科目 "
                f"{'、'.join(goal.get('weak_subjects') or []) or '无'}，"
                f"用户问题 {'、'.join(goal.get('concerns') or []) or '无'}。",
                "下面是已算好的每周骨架（周次、阶段、小时、考点不可修改）：",
                "\n".join(lines),
                "为每一周写一句 20–45 字的「本周为什么这么排」：",
                "1. 点出这一周的推进逻辑（新学→巩固→二刷→真题→模考的递进）；",
                "2. 相邻周的措辞不得重复；",
                "3. 不要罗列小时数或考点名单，不推荐教材、网课、老师；",
                '4. 严格输出 JSON：{"weeks":[{"week":1,"reason":"..."}]}',
            ]
            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是 408 备考规划师。只解释每周安排的理由，"
                        "不编造数字、真题号或频次，只输出 JSON。"
                    ),
                },
                {"role": "user", "content": "\n".join(user)},
            ]
            resp = self.guide_llm.chat(
                messages, max_tokens=4000, json_mode=True, timeout=90
            )
            if not resp.ok:
                logger.warning(
                    f"Planner weekly 调用失败，跳过本批理由：{resp.error[:120]}"
                )
                continue
            obj = _extract_json(resp.text)
            if not obj:
                logger.warning("Planner weekly 不是合法 JSON，跳过本批理由")
                continue
            text = json.dumps(obj, ensure_ascii=False)
            if any(term in text for term in FORBIDDEN_GUIDE_TERMS):
                logger.warning("Planner weekly 提到了资料/网课/老师，跳过本批理由")
                continue
            if re.search(r"20\d{2}#\d+", text):
                logger.warning("Planner weekly 含未提供的真题号，跳过本批理由")
                continue
            for item in obj.get("weeks") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    w_no = int(item.get("week"))
                except (TypeError, ValueError):
                    continue
                reason = str(item.get("reason") or "").strip()
                if reason:
                    out[w_no] = reason[:80]
        return out or None

    def _enrich_guide(
        self, plan: dict[str, Any], challenges: list[Challenge] | None
    ) -> dict[str, Any] | None:
        user = [
            _guide_context(plan),
            "请把模板建议改写成自然、具体、像经验贴的中文，但：",
            "1. 不得改变阶段、日期、小时、考点、真题编号；",
            "2. 不得推荐任何教材、网课、老师、参考书；",
            "3. 只写方法：怎么理解、怎么刷题、怎么归因、怎么复盘；",
            "4. overview 120–180 字；分段建议 50–90 字；分科策略用第一人称、连贯成段；",
            "5. 不要输出 sources，资料清单由模板保留；",
            "6. 严格输出 JSON："
            '{"overview":"...","phase_advice":[{"stage":"...","advice":"..."}],'
            '"subject_order":"...",'
            '"subject_strategy":[{"subject":"...","strategy":"..."}],'
            '"weekly_routine":"...","paper_usage":"...","review_workflow":"...",'
            '"pacing":"...","expectation":"...","crunch":"...","note":"..."}',
        ]
        if challenges:
            high_msgs = []
            for c in challenges:
                d = c.to_dict() if hasattr(c, "to_dict") else c
                if d.get("severity") == "high":
                    high_msgs.append(d.get("message"))
            if high_msgs:
                user.append(
                    "需要顺带回应的挑战：" + json.dumps(high_msgs, ensure_ascii=False)
                )
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 408 备考规划师。只写学习方法与节奏，不推荐资料/网课/老师，"
                    "不编造数字、真题号或频次。只输出 JSON。"
                ),
            },
            {"role": "user", "content": "\n".join(user)},
        ]
        resp = self.guide_llm.chat(messages, max_tokens=4000, json_mode=True)
        if not resp.ok:
            logger.warning(f"Planner guide 调用失败，使用模板：{resp.error[:120]}")
            return None
        obj = _extract_json(resp.text)
        if not obj or not str(obj.get("overview") or "").strip():
            logger.warning("Planner guide 不是合法 JSON，使用模板")
            return None
        text = json.dumps(obj, ensure_ascii=False)
        if any(term in text for term in FORBIDDEN_GUIDE_TERMS):
            logger.warning("Planner guide 提到了资料/网课/老师，按用户要求退回模板")
            return None
        if re.search(r"20\d{2}#\d+", text):
            logger.warning("Planner guide 含未提供的真题号，退回模板")
            return None
        # 结构兜底：缺字段就补模板里的
        template = plan.get("guide") or {}
        for key in (
            "overview",
            "phase_advice",
            "subject_order",
            "subject_strategy",
            "weekly_routine",
            "paper_usage",
            "review_workflow",
            "pacing",
            "expectation",
            "crunch",
            "note",
            "sources",
        ):
            if key not in obj or not obj[key]:
                obj[key] = template.get(key)
        return obj


class Critic:
    def __init__(self, llm: LLMClient | None = None) -> None:
        # Critic 默认用空回答率更低的备用模型；与 Planner 分开，避免自证
        self.llm = llm or _role_client(
            settings.adv_critic_model or get_llm().backup_model or settings.llm_model
        )

    def review(
        self, plan: dict[str, Any], rule_challenges: list[Challenge]
    ) -> tuple[list[dict[str, Any]], bool]:
        rules = []
        for c in rule_challenges:
            if isinstance(c, dict):
                rules.append(
                    {
                        "rule": c.get("rule"),
                        "severity": c.get("severity"),
                        "message": c.get("message"),
                        "locator": c.get("locator"),
                    }
                )
            else:
                rules.append(
                    {
                        "rule": c.rule,
                        "severity": c.severity,
                        "message": c.message,
                        "locator": c.locator,
                    }
                )
        concerns = (plan.get("goal") or {}).get("concerns") or []
        messages = [
            {
                "role": "system",
                "content": (
                    "你是备考 Plan 的风控官。检查：目标可达性、阶段与剩余时间是否匹配、"
                    "每周任务是否具体（学什么/怎么学/验收标准）、高性价比考点是否覆盖、"
                    "用户自述问题是否被回应、前置顺序是否合理。"
                    "不得引入新事实，不能修改 Plan，只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Plan 摘要：\n{plan_digest(plan)}\n\n"
                    f"已由规则检出的问题：\n{json.dumps(rules, ensure_ascii=False)}\n\n"
                    f"用户自述问题：{json.dumps(concerns, ensure_ascii=False)}\n\n"
                    "请补充 0–3 条主观问题。每条必须给 evidence_ref，取值为 Plan 的真实字段路径，"
                    "如 `goal.daily_hours`、`stages[0].weeks`、`priority_kps[0].weight`、"
                    "`weekly_plan[0].tasks[1].content`、`concern_actions[0].action`。\n"
                    '输出：{"challenges":[{"severity":"high|medium|low","claim":"...",'
                    '"evidence_ref":"...","fix":"..."}]}'
                ),
            },
        ]
        resp = self.llm.chat(messages, max_tokens=4000, json_mode=True, timeout=120)
        if not resp.ok:
            return [], False
        obj = _extract_json(resp.text) or {}
        items = obj.get("challenges") or []
        out: list[dict[str, Any]] = []
        for it in items[:3]:
            if not isinstance(it, dict) or not it.get("claim") or not it.get("evidence_ref"):
                continue
            ref = str(it["evidence_ref"]).strip()[:120]
            # 证据必须指向 Plan 的真实字段路径，否则丢弃（防编造）
            if not _ref_exists(plan, ref):
                continue
            sev = str(it.get("severity") or "medium").lower()
            if sev not in ("high", "medium", "low"):
                sev = "medium"
            out.append(
                {
                    "rule": "llm_critic",
                    "severity": sev,
                    "message": str(it["claim"])[:300],
                    "evidence_ref": ref,
                    "fix": str(it.get("fix") or "")[:200],
                }
            )
        return out, True


class Expert:
    def __init__(
        self, llm: LLMClient | None = None, analyzer: ExamAnalyzer | None = None
    ) -> None:
        # Expert 裁定交叉检查：默认回到主模型，与 Critic 的模型不同
        self.llm = llm or _role_client(settings.adv_expert_model or settings.llm_model)
        self.analyzer = analyzer or get_analyzer()

    def arbitrate(self, conflicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not conflicts:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 408 学科专家。对争议条目做裁定，依据只能是给定的真题统计与前置关系。"
                    "action 只能是 keep / downgrade / drop。只输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"争议条目：\n{json.dumps(conflicts, ensure_ascii=False)}\n\n"
                    '输出：{"verdicts":[{"challenge_id":"...","action":"keep|downgrade|drop",'
                    '"reason":"...","evidence":"..."}]}'
                ),
            },
        ]
        resp = self.llm.chat(messages, max_tokens=4000, json_mode=True, timeout=120)
        if not resp.ok:
            return []
        obj = _extract_json(resp.text) or {}
        valid_ids = {str(c.get("id")) for c in conflicts if c.get("id")}
        out: list[dict[str, Any]] = []
        for v in (obj.get("verdicts") or [])[:3]:
            if not isinstance(v, dict):
                continue
            if str(v.get("challenge_id")) not in valid_ids:
                continue
            if v.get("action") not in ("keep", "downgrade", "drop"):
                continue
            out.append(v)
        return out