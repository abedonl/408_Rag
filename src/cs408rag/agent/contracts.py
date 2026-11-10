"""工具契约：参数模型、调用规格、结果与审计轨迹。

设计约束：

1. **参数先校验，再执行。** `TOOL_SPECS` 以前只是给模型看的文档字符串，
   坏参数会被工具内部的异常吞成 `ok=False`，planner 无从修正。
   现在每个工具都有 Pydantic 参数模型，失败返回 `error_type="validation"`。

2. **白名单是唯一入口。** `TOOL_CONTRACTS` 的键就是允许调用的工具集合，
   planner 与 LLM fallback 都只能从这里选。

3. **超时是尽力而为。** 工具跑在独立线程池里，超时后主流程不再等待；
   Python 线程无法强杀，超时线程可能继续占用资源 —— 这是已知限制，
   不假装能杀，只保证不阻塞 Agent。

4. **审计不落原始参数。** `ToolResult.args_digest` 只保留参数摘要的短哈希，
   避免把可能是用户原文的参数写进 trace。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ErrorType = Literal[
    "validation",
    "not_found",
    "timeout",
    "internal",
    "budget",
    "unknown_tool",
]


# ---------------------------------------------------------------------------
# 结果与轨迹
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: str = ""
    error_type: str = ""
    took_ms: float = 0.0
    summary: str = ""
    validated: bool = True
    attempt: int = 1
    round: int = 0
    args_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type,
            "took_ms": round(self.took_ms, 1),
            "summary": self.summary,
            "validated": self.validated,
            "attempt": self.attempt,
            "round": self.round,
            "args_digest": self.args_digest,
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

    def failure_details(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": c.tool,
                "error_type": c.error_type or "internal",
                "error": c.error[:200],
                "round": c.round,
            }
            for c in self.calls
            if not c.ok
        ]

    def by_name(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.calls:
            out[c.tool] = out.get(c.tool, 0) + 1
        return out

    def to_list(self) -> list[dict[str, Any]]:
        return [c.to_dict() for c in self.calls]


# ---------------------------------------------------------------------------
# 参数模型
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    """默认拒绝未知字段：多了字段说明调用方与契约不一致，宁可失败。"""

    model_config = ConfigDict(extra="forbid")


class SearchKBArgs(_Strict):
    query: str = Field(min_length=1, max_length=500)
    subject: str | None = Field(default=None, max_length=32)
    top_k: int = Field(default=8, ge=1, le=20)


class GraphArgs(_Strict):
    kp_name: str = Field(min_length=1, max_length=64)


class FigureArgs(_Strict):
    question_id: str = Field(pattern=r"^\d{4}-\d{2}$")
    question: str = Field(default="", max_length=2000)


class CalcArgs(_Strict):
    expr: str = Field(min_length=1, max_length=500)


class ExamStatsArgs(_Strict):
    kp_name: str = Field(min_length=1, max_length=64)


class KpQuestionsArgs(_Strict):
    kp_name: str = Field(min_length=1, max_length=64)
    limit: int = Field(default=5, ge=1, le=20)


class RelatedArgs(_Strict):
    kp_name: str = Field(min_length=1, max_length=64)


class LearningPathArgs(_Strict):
    kp_name: str = Field(min_length=1, max_length=64)


class VerifyArgs(_Strict):
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(min_length=1, max_length=20000)
    contexts: Any = None


class GeneratePlanArgs(_Strict):
    """生成备考规划。全部参数可缺省 —— 缺省时薄弱科目与加权考点从画像推断。"""

    weak_subjects: list[str] = Field(default_factory=list, max_length=4)
    progress: dict[str, str] = Field(default_factory=dict)
    daily_hours: int = Field(default=4, ge=1, le=12)
    study_days_per_week: int = Field(default=6, ge=1, le=7)
    target_score: int = Field(default=110, ge=60, le=150)
    concerns: list[str] = Field(default_factory=list, max_length=10)


class ReadProfileArgs(_Strict):
    """无参数：读的是调用方绑定的 client_id 的画像，LLM 无法指定别人。"""


class RecordCheckinArgs(_Strict):
    week: int = Field(ge=1, le=60)
    idx: int = Field(default=0, ge=0, le=20)
    done: bool = True
    subject: str = Field(default="", max_length=16)
    kps: list[str] = Field(default_factory=list, max_length=8)
    hours: float | None = Field(default=None, ge=0, le=24)
    stage: str = Field(default="", max_length=16)


class JudgeQuizArgs(_Strict):
    qid: str = Field(pattern=r"^\d{4}-\d{2}$")
    choice: str = Field(pattern=r"^[A-D]$")


# ---------------------------------------------------------------------------
# 调用规格
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    name: str
    args_model: type[BaseModel]
    desc: str
    timeout_s: float = 15.0
    retries: int = 0
    idempotent: bool = True


# 检索/统计/图解析/计算/自检/读画像/生成规划是只读的（idempotent=True）；
# record_checkin / judge_quiz 是写操作（idempotent=False），即使配了 retries
# 也绝不自动重试 —— 写操作重试可能产生重复事件。
# 是否重试由 retries 单独控制：慢的外部查询不自动重试，避免延迟翻倍。
TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in [
        ToolSpec(
            "search_knowledge_base",
            SearchKBArgs,
            "在教材与真题解析中做三路混合检索，返回带章节路径的资料片段。",
            timeout_s=30.0,
        ),
        ToolSpec(
            "search_knowledge_graph",
            GraphArgs,
            "在知识点图谱中定位一个知识点，返回章节路径、祖先与同节兄弟。",
        ),
        ToolSpec(
            "read_figure",
            FigureArgs,
            "读取某道真题的配图。T1 重建拓扑，T2 提取文本，T3 需视觉模型。",
        ),
        ToolSpec(
            "calculate",
            CalcArgs,
            "计算一个算式（四则、幂、log2/sqrt/ceil 等），避免心算错误。",
            timeout_s=5.0,
            retries=1,
        ),
        ToolSpec(
            "get_exam_stats",
            ExamStatsArgs,
            "考点在历年真题中的考查频次、年份/科目/题型分布。",
        ),
        ToolSpec(
            "get_kp_questions",
            KpQuestionsArgs,
            "某考点的历年真题原题（含年份、题干、选项；非评测模式含答案解析）。",
        ),
        ToolSpec(
            "get_related_kps",
            RelatedArgs,
            "与某考点关联的其他考点：真题共现（带证据题号）+ 图谱同节/祖先。",
        ),
        ToolSpec(
            "get_learning_path",
            LearningPathArgs,
            "该知识点的前置学习链（沿 PREREQUISITE 反向），用于学习路径推荐。",
        ),
        ToolSpec(
            "verify_answer",
            VerifyArgs,
            "自检：引用编号是否存在、真题编号是否真实、是否出现押题表述。",
            timeout_s=10.0,
        ),
        ToolSpec(
            "generate_plan",
            GeneratePlanArgs,
            "生成个性化备考规划（阶段+优先级考点+时间预算），自动读画像加权。",
            timeout_s=30.0,
        ),
        ToolSpec(
            "read_profile",
            ReadProfileArgs,
            "读当前用户的画像摘要：高频提问、作答对错、打卡执行、加权考点。",
        ),
        ToolSpec(
            "record_checkin",
            RecordCheckinArgs,
            "记录一次周任务打卡（写操作，画像按 latest-wins 聚合）。",
            idempotent=False,
        ),
        ToolSpec(
            "judge_quiz",
            JudgeQuizArgs,
            "对一道真题作答判分（写操作）：答错进错题本、答对移出，同步画像。",
            idempotent=False,
        ),
    ]
}


def args_digest(mapping: dict[str, Any], *, length: int = 12) -> str:
    """参数摘要的短哈希。只用于审计关联，不保留原文。"""
    try:
        payload = json.dumps(mapping, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(mapping)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def classify_error(exc: BaseException) -> ErrorType:
    """把任意异常归一化成 planner 能处理的错误类型。"""
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, (FileNotFoundError, KeyError)):
        return "not_found"
    if isinstance(exc, (ValueError, TypeError, AssertionError)):
        return "validation"
    return "internal"