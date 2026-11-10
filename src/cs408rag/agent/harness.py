"""任务型 Agent harness：目标 → 显式计划 → 逐步执行 → 观察 → 校验 → 合成。

与 graph.py 的单轮问答链路并列：那里回答一个问题，这里完成一个**多步任务**
（默认任务：「攻克一个考点」）。harness 的每个组件都是显式的、可审查的：

- **计划先行**：`plan()` 返回步骤清单，执行前就能看到要做什么
- **预算**：`max_steps` 步数上限，超限即停，不允许无限循环
- **观察**：每步的 ToolResult 落 `steps`，失败按 required/optional 决定中断或降级跳过
- **校验门**：最终讲义必须过 verify_answer（真题编号真实、无押题表述）
- **事件流**：`on_event` 回调逐步推送，UI 可实时渲染执行过程

LLM 在这里只做最后一步成文（可选）；没有 LLM 时用确定性模板合成，
任务照样完整跑完 —— harness 不依赖模型可用性。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from cs408rag.agent.tools import AgentTools

# 事件类型：plan / step_start / step_end / done
EventFn = Callable[[dict[str, Any]], None]
# LLM 成文接口：输入 (goal, observations) 返回讲义文本；None 走确定性模板
LLMFn = Callable[[str, dict[str, Any]], str]


@dataclass(frozen=True)
class StepSpec:
    key: str
    title: str
    tool: str  # TOOL_REGISTRY 名；"llm" 表示合成步骤
    args: dict[str, Any] = field(default_factory=dict)
    required: bool = True  # 失败是否中断整个任务


@dataclass
class TaskReport:
    goal: str
    tag: str
    ok: bool
    stopped: str  # done / tag_not_found / budget / required_failed:<key>
    steps: list[dict[str, Any]]
    observations: dict[str, Any]
    briefing: str
    quiz: list[dict[str, Any]]
    verification: dict[str, Any]
    took_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "tag": self.tag,
            "ok": self.ok,
            "stopped": self.stopped,
            "steps": self.steps,
            "briefing": self.briefing,
            "quiz": self.quiz,
            "verification": self.verification,
            "took_ms": round(self.took_ms, 1),
        }


def resolve_tag(goal: str, analyzer: Any) -> str:
    """从目标文本里定位考点：取在文本中出现的最长 tag（至少 2 字）。"""
    best = ""
    for tag in getattr(analyzer, "tag_freq", {}) or {}:
        if len(tag) >= 2 and tag in goal and len(tag) > len(best):
            best = tag
    return best


class TaskHarness:
    """多步任务执行器。预算、降级、校验都在这里，不让 LLM 碰。"""

    def __init__(
        self,
        tools: AgentTools,
        *,
        analyzer: Any = None,
        max_steps: int = 8,
        llm_fn: LLMFn | None = None,
    ) -> None:
        self.tools = tools
        if analyzer is None:
            from cs408rag.agent.exam_analysis import get_analyzer

            analyzer = get_analyzer()
        self.analyzer = analyzer
        self.max_steps = max(1, max_steps)
        self.llm_fn = llm_fn

    # -- 计划 -------------------------------------------------------------

    def plan(self, tag: str) -> list[StepSpec]:
        """「攻克考点」任务的标准计划。统计与真题是命脉，挂了任务就中断；
        关联考点 / 前置链 / 教材检索是增强，挂了降级继续。"""
        return [
            StepSpec("stats", "查考频与题型分布", "get_exam_stats", {"kp_name": tag}),
            StepSpec(
                "questions", "取历年真题（作自测题）", "get_kp_questions",
                {"kp_name": tag, "limit": 3},
            ),
            StepSpec(
                "related", "查关联考点（共现证据）", "get_related_kps",
                {"kp_name": tag}, required=False,
            ),
            StepSpec(
                "path", "查前置学习链", "get_learning_path",
                {"kp_name": tag}, required=False,
            ),
            StepSpec(
                "kb", "检索教材对应讲解", "search_knowledge_base",
                {"query": f"{tag} 的概念、解题步骤与易错点", "top_k": 5},
                required=False,
            ),
            StepSpec("synthesize", "合成攻克讲义", "llm"),
            StepSpec("verify", "校验讲义（引用与真题编号）", "verify_answer", {}),
        ]

    # -- 执行 -------------------------------------------------------------

    def run(self, goal: str, on_event: EventFn | None = None) -> TaskReport:
        t0 = time.perf_counter()
        emit = on_event or (lambda e: None)
        steps: list[dict[str, Any]] = []
        obs: dict[str, Any] = {}

        tag = resolve_tag(goal, self.analyzer)
        if not tag:
            emit({"type": "done", "stopped": "tag_not_found"})
            return TaskReport(
                goal=goal, tag="", ok=False, stopped="tag_not_found",
                steps=steps, observations=obs, briefing="",
                quiz=[], verification={},
                took_ms=(time.perf_counter() - t0) * 1000,
            )

        plan = self.plan(tag)
        emit({"type": "plan", "tag": tag,
              "steps": [{"key": s.key, "title": s.title} for s in plan]})

        stopped = "done"
        for spec in plan[: self.max_steps]:
            emit({"type": "step_start", "key": spec.key, "title": spec.title,
                  "tool": spec.tool})
            if spec.tool == "llm":
                briefing = self._synthesize(goal, tag, obs)
                obs["briefing"] = briefing
                rec = {"key": spec.key, "title": spec.title, "tool": "llm",
                       "ok": True, "took_ms": 0.0,
                       "summary": f"讲义 {len(briefing)} 字"
                                + ("" if self.llm_fn else "（模板合成，未用 LLM）")}
            elif spec.key == "verify":
                briefing = obs.get("briefing", "")
                r = self.tools.call(
                    "verify_answer", count_budget=False,
                    question=goal, answer=briefing or "（空）",
                )
                obs["verification"] = r.data or {}
                rec = {"key": spec.key, "title": spec.title, "tool": spec.tool,
                       "ok": r.ok, "took_ms": r.took_ms, "summary": r.summary}
            else:
                r = self.tools.call(spec.tool, count_budget=False, **spec.args)
                obs[spec.key] = r.data
                rec = {"key": spec.key, "title": spec.title, "tool": spec.tool,
                       "ok": r.ok, "took_ms": r.took_ms, "summary": r.summary}
                if not r.ok and spec.required:
                    stopped = f"required_failed:{spec.key}"
            steps.append(rec)
            emit({"type": "step_end", **rec})
            if stopped.startswith("required_failed"):
                break
        else:
            # for 正常结束（未被 break）；若计划因预算被截断也走这里，单独标注
            pass
        if stopped == "done" and len(plan) > self.max_steps:
            stopped = "budget"

        briefing = obs.get("briefing", "")
        verification = obs.get("verification") or {}
        quiz = self._build_quiz(obs.get("questions"))
        ok = stopped == "done" and bool(briefing)
        emit({"type": "done", "stopped": stopped})
        return TaskReport(
            goal=goal, tag=tag, ok=ok, stopped=stopped,
            steps=steps, observations=obs, briefing=briefing,
            quiz=quiz, verification=verification,
            took_ms=(time.perf_counter() - t0) * 1000,
        )

    # -- 合成 -------------------------------------------------------------

    def _synthesize(self, goal: str, tag: str, obs: dict[str, Any]) -> str:
        if self.llm_fn is not None:
            try:
                text = self.llm_fn(goal, obs)
                if text and text.strip():
                    return text.strip()
            except Exception:  # noqa: BLE001 - LLM 挂了降级到模板，任务不中断
                pass
        return self._template_briefing(tag, obs)

    def _template_briefing(self, tag: str, obs: dict[str, Any]) -> str:
        """无 LLM 时的确定性讲义：全部由工具观察数据拼装，可溯源。"""
        parts: list[str] = [f"## 攻克「{tag}」\n"]
        stats = obs.get("stats") or {}
        if stats.get("total"):
            years = sorted((stats.get("by_year") or {}).items())
            recent = sum(n for y, n in years if str(y) >= "2021")
            parts.append(
                f"**考情**：历年考查 {stats['total']} 次"
                f"（匹配标签 {len(stats.get('matched_tags') or [])} 个），"
                f"2021 年以来 {recent} 次；"
                f"题型分布 {stats.get('by_type') or {}}。"
                "频次是真题客观计数，不是预测。\n"
            )
        path = obs.get("path") or {}
        chain = path.get("chain") or []
        if path.get("found") and chain:
            names = " → ".join(str(c.get("title") or c.get("name") or c) for c in chain[:8])
            parts.append(f"**前置链**（先补这些再看它）：{names}\n")
        related = obs.get("related") or {}
        rel = related.get("cooccurring_tags") or []
        if rel:
            names = [
                r.get("tag") if isinstance(r, dict) else str(r) for r in rel[:6]
            ]
            parts.append(f"**常一起考**：{'、'.join(n for n in names if n)}\n")
        kb = obs.get("kb") or {}
        hits = kb.get("hits") or []
        if hits:
            parts.append("**教材讲解**（可回原文核对）：")
            for h in hits[:3]:
                parts.append(f"- [{h.get('citation')}] {str(h.get('text'))[:160]}…")
            parts.append("")
        q = self._build_quiz(obs.get("questions"))
        if q:
            parts.append(
                f"**自测**：下面 {len(q)} 道历年真题，先自测再看答案；"
                "作答会同步错题本与你的画像。\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _build_quiz(questions_data: Any) -> list[dict[str, Any]]:
        rows = (questions_data or {}).get("questions") or []
        out = []
        for r in rows:
            if r.get("answer") and r.get("options"):
                out.append(r)
            if len(out) >= 3:
                break
        return out
