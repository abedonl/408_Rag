"""对抗 Plan 评测：规则检出率 + 单轮生成 vs 2 轮对抗。

两部分互相独立：

1. **规则评测**（永远跑）：黄金 Plan 误报率、6 类注入错误检出率。
   只跑纯函数规则，不调 LLM，可完全离线复现。
2. **单轮 vs 2 轮对抗**（有 LLM 时跑）：同一批目标，分别用
   「生成 1 版」和「最多 3 版（2 次修订）」执行，比较残留 high 挑战、
   版本数、LLM 调用、token 与延迟。对比如实报告，不预设多轮更好。

用法：
    python -m cs408rag.eval.adversarial_eval --no-llm     # 全离线，只跑规则
    python -m cs408rag.eval.adversarial_eval             # 规则 + 单轮/2 轮对比
    python -m cs408rag.eval.adversarial_eval --limit 2   # 只跑 2 个薄弱科目
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from cs408rag.adversarial.dataset import build_dataset, load
from cs408rag.adversarial.dataset import save as save_dataset
from cs408rag.adversarial.plan import ERROR_TYPES
from cs408rag.adversarial.rules import review_plan
from cs408rag.config import REPORTS_DIR

RUNS_DIR = REPORTS_DIR / "adversarial_eval_runs"
RUNS_PATH = RUNS_DIR / "plans.jsonl"
# 缓存键带版本号：状态机语义变化后旧结果不再复用，但不清空历史文件
RUN_KEY_VERSION = "v8"

# 每个薄弱科目一条目标，覆盖四种备考起点
SUBJECTS = ["数据结构", "组成原理", "操作系统", "计算机网络"]


def _goal(
    *,
    scope: str,
    weak: list[str],
    weeks: int,
    daily: float,
    days: int,
    progress: str,
    concerns: list[str] | None = None,
) -> dict[str, Any]:
    start = date.today()
    return {
        "scope": scope,
        "weak_subjects": weak,
        "start_date": start.isoformat(),
        "exam_date": (start + timedelta(weeks=weeks)).isoformat(),
        "daily_hours": daily,
        "study_days_per_week": days,
        "target_score": 115,
        "progress": {s: progress for s in SUBJECTS},
        "concerns": concerns or [],
    }


# 覆盖 8/16/30 周、全科/单科、不同进度与用户问题
DEFAULT_GOALS: list[dict[str, Any]] = [
    _goal(
        scope="full_408",
        weak=["操作系统"],
        weeks=8,
        daily=4,
        days=6,
        progress="一轮中",
        concerns=["时间不够", "真题正确率低"],
    ),
    _goal(
        scope="full_408",
        weak=["数据结构", "计算机网络"],
        weeks=16,
        daily=5,
        days=5,
        progress="一轮完",
        concerns=["基础差"],
    ),
    _goal(
        scope="single_subject",
        weak=["操作系统"],
        weeks=30,
        daily=3,
        days=6,
        progress="二轮中",
        concerns=[],
    ),
    _goal(
        scope="full_408",
        weak=["组成原理"],
        weeks=12,
        daily=4,
        days=7,
        progress="未开始",
        concerns=["计算题总算错"],
    ),
]


# ---------------------------------------------------------------------------
# 规则评测
# ---------------------------------------------------------------------------


def evaluate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    golden_total = 0
    golden_fp = 0
    detected = Counter()
    label_total = Counter()
    all_challenges = 0
    matched_challenges = 0
    by_severity = Counter()
    details: list[dict[str, Any]] = []

    for row in rows:
        challenges = review_plan(row["plan"])
        rules = {c.rule for c in challenges}
        by_severity.update(c.severity for c in challenges)
        all_challenges += len(challenges)

        if row["kind"] == "golden":
            golden_total += 1
            if challenges:
                golden_fp += 1
                details.append(
                    {
                        "id": row["id"],
                        "kind": "golden",
                        "false_positive": [c.to_dict() for c in challenges],
                    }
                )
            continue

        labels = row.get("labels") or []
        matched = 0
        for label in labels:
            label_total[label] += 1
            if label in rules:
                detected[label] += 1
                matched += 1
        matched_challenges += matched
        details.append(
            {
                "id": row["id"],
                "kind": "injected",
                "labels": labels,
                "detected_rules": sorted(rules),
                "matched": matched,
            }
        )

    per_rule = {
        label: {
            "n": label_total[label],
            "detected": detected[label],
            "rate": round(detected[label] / label_total[label], 4)
            if label_total[label]
            else 0.0,
        }
        for label in ERROR_TYPES
        if label_total[label]
    }

    return {
        "golden": {
            "n": golden_total,
            "false_positive": golden_fp,
            "false_positive_rate": round(golden_fp / golden_total, 4) if golden_total else 0.0,
        },
        "injected": {
            "n": sum(1 for r in rows if r["kind"] == "injected"),
            "labels": sum(label_total.values()),
            "detected": sum(detected.values()),
            "detection_rate": round(sum(detected.values()) / sum(label_total.values()), 4)
            if label_total
            else 0.0,
            "per_rule": per_rule,
        },
        "challenges": {
            "total": all_challenges,
            "matched": matched_challenges,
            "precision": round(matched_challenges / all_challenges, 4)
            if all_challenges
            else 0.0,
            "by_severity": dict(by_severity),
        },
        "details": details,
    }


# ---------------------------------------------------------------------------
# 单轮 vs 2 轮
# ---------------------------------------------------------------------------


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) >= 20:
        return float(statistics.quantiles(values, n=20)[-1])
    return float(max(values))


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("key"):
            done[row["key"]] = row
    return done


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _timeline_totals(timeline: list[dict[str, Any]]) -> tuple[dict[str, int], float]:
    """按节点累加 token 与耗时。

    单轮臂是同一轮运行的时间前缀，只能这样得到可比的成本；
    直接再跑一次会产生不同的初始 Plan，差异里混进模型随机性。
    """
    prompt = sum(int((e.get("tokens") or {}).get("prompt", 0) or 0) for e in timeline)
    completion = sum(int((e.get("tokens") or {}).get("completion", 0) or 0) for e in timeline)
    ms = sum(float(e.get("took_ms") or 0) for e in timeline)
    return {"prompt": prompt, "completion": completion}, round(ms, 1)


def _row_from_state(
    *,
    key: str,
    arm: str,
    subject: str,
    goal: dict[str, Any],
    state: dict[str, Any],
    role_models: dict[str, str],
    stop_override: str = "",
) -> dict[str, Any]:
    challenges = state.get("challenges") or []
    open_high = [
        c
        for c in challenges
        if c.get("severity") == "high" and c.get("status", "open") == "open"
    ]
    llm_ch = [c for c in challenges if c.get("rule") == "llm_critic"]
    timeline = state.get("timeline") or []
    tokens, ms = _timeline_totals(timeline)
    return {
        "key": key,
        "arm": arm,
        "subject": subject,
        "goal": goal,
        "stop_reason": stop_override or state.get("stop_reason") or "",
        "version": int(state.get("version") or 0),
        "rounds": int(state.get("round") or 0),
        "n_steps": len(timeline),
        "residual_high": len(open_high),
        "n_rule_challenges": sum(1 for c in challenges if c.get("rule") != "llm_critic"),
        "n_llm_challenges": len(llm_ch),
        "llm_challenges": [
            {
                "severity": c.get("severity"),
                "claim": c.get("message"),
                "evidence_ref": c.get("evidence_ref"),
            }
            for c in llm_ch
        ],
        "llm_calls": int(state.get("llm_calls") or 0),
        "llm_errors": int(state.get("llm_errors") or 0),
        "planner_fallback": bool(state.get("planner_fallback")),
        "plan_source": state.get("plan_source"),
        "guide_source": state.get("guide_source"),
        "role_models": role_models,
        "tokens": tokens,
        "took_ms": ms,
        "error": str(state.get("error") or ""),
    }


def collect_plans(
    goals: list[dict[str, Any]],
    *,
    max_rounds: int = 2,
    out_path: Path | None = None,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> list[dict[str, Any]]:
    """每个目标只跑一次 2 轮流程，用第 1 轮快照作为单轮臂。

    这样两条臂共享同一个初始 Plan 与同一轮质疑，差异只来自「是否继续修订」，
    不会被 LLM 每次生成不同 Plan 的随机性污染；LLM 调用也少一半。
    """
    from loguru import logger

    from cs408rag.adversarial.graph import PlanRunner

    path = out_path or RUNS_PATH
    if limit:
        goals = goals[:limit]
    done = _load_done(path)

    keys: list[str] = []
    for gi, goal in enumerate(goals, 1):
        subject = (goal.get("weak_subjects") or ["?"])[0]
        single_key = f"{RUN_KEY_VERSION}:single_round:{subject}"
        two_key = f"{RUN_KEY_VERSION}:two_round:{subject}"
        keys += [single_key, two_key]
        if single_key in done and two_key in done:
            continue

        runner = PlanRunner(use_llm=True, max_rounds=max_rounds)
        after_first: dict[str, Any] | None = None
        final: dict[str, Any] | None = None
        for snapshot in runner.iter_run(goal):
            final = snapshot
            # 第一次 round 变成 1，说明第一轮 advance 已执行完
            if after_first is None and int(snapshot.get("round") or 0) >= 1:
                after_first = snapshot
        if final is None:
            continue
        if after_first is None:
            after_first = final

        # 真实单轮运行（max_rounds=0）会在第 1 轮后以 max_rounds 退出；
        # 2 轮运行里第 1 轮没有 stop_reason，这里补上这个反事实结果。
        single_stop = after_first.get("stop_reason") or ("max_rounds" if max_rounds > 0 else "")
        rows = [
            _row_from_state(
                key=single_key,
                arm="single_round",
                subject=subject,
                goal=goal,
                state=after_first,
                role_models=runner.role_models,
                stop_override=single_stop,
            ),
            _row_from_state(
                key=two_key,
                arm="two_round",
                subject=subject,
                goal=goal,
                state=final,
                role_models=runner.role_models,
            ),
        ]
        for row in rows:
            _append(path, row)
            done[row["key"]] = row
        logger.info(
            f"[{gi}/{len(goals)}] {subject} 单轮 stop={rows[0]['stop_reason']} "
            f"high={rows[0]['residual_high']} | 2 轮 stop={rows[1]['stop_reason']} "
            f"v={rows[1]['version']} high={rows[1]['residual_high']} | "
            f"{rows[1]['took_ms'] / 1000:.1f}s"
        )
        if sleep_s:
            time.sleep(sleep_s)
    # 只返回本次 goals 对应的结果：历史文件里可能有旧版本键或其他科目的记录
    return [done[k] for k in keys if k in done]


def summarize_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "arms": {},
        "paired": [],
        "llm_challenges": [],
        "role_models": {},
    }
    # 每个目标两条臂；--limit 1 时这里是 1，报告要如实写
    out["n_goals"] = max(1, len(rows) // 2)
    for arm in ("single_round", "two_round"):
        rs = [r for r in rows if r.get("arm") == arm]
        if not rs:
            continue
        taken = [float(r.get("took_ms") or 0) for r in rs]
        tokens = [
            int((r.get("tokens") or {}).get("prompt", 0))
            + int((r.get("tokens") or {}).get("completion", 0))
            for r in rs
        ]
        converged = sum(1 for r in rs if r.get("stop_reason") == "converged")
        out["arms"][arm] = {
            "n": len(rs),
            "converged": converged,
            "convergence_rate": round(converged / len(rs), 4),
            "avg_version": round(statistics.mean(r.get("version") or 0 for r in rs), 2),
            "avg_residual_high": round(
                statistics.mean(r.get("residual_high") or 0 for r in rs), 2
            ),
            "avg_llm_calls": round(statistics.mean(r.get("llm_calls") or 0 for r in rs), 2),
            "avg_tokens": round(statistics.mean(tokens), 0) if tokens else 0,
            "p50_ms": round(statistics.median(taken), 1) if taken else 0,
            "p95_ms": round(_p95(taken), 1),
            "avg_llm_challenges": round(
                statistics.mean(r.get("n_llm_challenges") or 0 for r in rs), 2
            ),
            "stop_reasons": dict(Counter(r.get("stop_reason") for r in rs)),
            "planner_fallbacks": sum(1 for r in rs if r.get("planner_fallback")),
            "guide_sources": dict(Counter(r.get("guide_source") for r in rs)),
        }

    by_subject: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_subject.setdefault(str(r.get("subject")), {})[str(r.get("arm"))] = r
    for subject, pair in by_subject.items():
        if "single_round" not in pair or "two_round" not in pair:
            continue
        s, t = pair["single_round"], pair["two_round"]
        out["paired"].append(
            {
                "subject": subject,
                "single_stop": s.get("stop_reason"),
                "two_stop": t.get("stop_reason"),
                "single_versions": s.get("version"),
                "two_versions": t.get("version"),
                "single_residual_high": s.get("residual_high"),
                "two_residual_high": t.get("residual_high"),
                "single_llm_calls": s.get("llm_calls"),
                "two_llm_calls": t.get("llm_calls"),
                "single_ms": s.get("took_ms"),
                "two_ms": t.get("took_ms"),
            }
        )

    for r in rows:
        for name, model in (r.get("role_models") or {}).items():
            if model:
                out["role_models"][name] = model
        for c in r.get("llm_challenges") or []:
            out["llm_challenges"].append(
                {
                    "subject": r.get("subject"),
                    "arm": r.get("arm"),
                    "severity": c.get("severity"),
                    "claim": c.get("claim"),
                    "evidence_ref": c.get("evidence_ref"),
                }
            )
    return out


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _moved(single: float, two: float, unit: str = "") -> str:
    """两轮相对单轮的变化；残留挑战越少越好，成本越高越差。"""
    diff = two - single
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:g}{unit}"


def to_markdown(summary: dict[str, Any]) -> str:
    g, inj, ch = summary["golden"], summary["injected"], summary["challenges"]
    lines = ["# 对抗性 Plan 评测\n"]
    lines.append("规则是纯函数，不调用 LLM；指标可完全离线复现。\n")

    lines.append("## 1. 黄金 Plan 误报\n")
    lines.append("| 项 | 结果 |")
    lines.append("|---|---|")
    lines.append(f"| 黄金 Plan 数 | {g['n']} |")
    lines.append(f"| 误报条数 | {g['false_positive']} |")
    lines.append(f"| **误报率** | **{g['false_positive_rate']:.1%}** |")

    lines.append("\n## 2. 注入错误检出\n")
    lines.append("| 错误类型 | 注入次数 | 检出 | 检出率 |")
    lines.append("|---|---|---|---|")
    for rule, v in inj["per_rule"].items():
        lines.append(f"| {rule} | {v['n']} | {v['detected']} | **{v['rate']:.1%}** |")
    lines.append(
        f"\n整体检出率：**{inj['detection_rate']:.1%}**（{inj['detected']}/{inj['labels']}）"
    )

    lines.append("\n## 3. 挑战质量\n")
    lines.append(f"- 挑战总数：{ch['total']}")
    lines.append(f"- 命中注入标签的挑战：{ch['matched']}")
    lines.append(f"- 挑战精确率：**{ch['precision']:.1%}**")
    lines.append(f"- 按严重度：{ch['by_severity']}")
    lines.append(
        "\n> 精确率不是 100% 是正常的：注入「假引用」会同时导致「高权重知识点无有效引用」，"
        "这类连带挑战是规则设计的预期行为。"
    )

    runs = summary.get("single_vs_two_round")
    if not runs or not runs.get("arms"):
        lines.append("\n## 4. 单轮 vs 2 轮对抗\n")
        lines.append("本次未启用 LLM（`--no-llm`），跳过该对比。规则链路不受影响。")
    else:
        arms = runs["arms"]
        s = arms.get("single_round", {})
        t = arms.get("two_round", {})
        lines.append("\n## 4. 单轮生成 vs 2 轮对抗\n")
        n_goals = runs.get("n_goals", len(runs.get("paired") or []))
        lines.append(
            f"本次完成 **{n_goals} 组**目标：每组只跑一次 2 轮流程，"
            "把第 1 轮结束时的快照当作单轮臂"
            "（等价于 `max_rounds=0`），最终状态当作 2 轮臂。两条臂共享同一个初始 Plan "
            "和同一轮质疑，差异只来自「是否继续修订」，不会被模型每次生成不同 Plan 的"
            "随机性污染。成本按各臂实际经过的节点累加。\n"
        )
        if n_goals < 4:
            lines.append(
                f"> 本次只完成 {n_goals} 组（`--limit` 或 endpoint 限流）；"
                f"单轮/2 轮的结论只按这 {n_goals} 组解读，不外推到默认 4 组。\n"
            )
        models = runs.get("role_models") or {}
        if models:
            lines.append(
                "角色模型：" + " · ".join(f"{k} `{v}`" for k, v in models.items()) + "\n"
            )
        lines.append("| 指标 | 单轮 | 2 轮 | 差值（2 轮 − 单轮） |")
        lines.append("|---|---|---|---|")
        rows = [
            ("收敛率", "convergence_rate", lambda v: f"{v:.0%}"),
            ("平均版本数", "avg_version", lambda v: f"{v:.2f}"),
            ("平均残留 high", "avg_residual_high", lambda v: f"{v:.2f}"),
            ("平均 LLM 调用", "avg_llm_calls", lambda v: f"{v:.2f}"),
            ("平均 token", "avg_tokens", lambda v: f"{v:.0f}"),
            ("P50", "p50_ms", lambda v: f"{v / 1000:.1f}s"),
            ("P95", "p95_ms", lambda v: f"{v / 1000:.1f}s"),
            ("平均残留主观挑战", "avg_llm_challenges", lambda v: f"{v:.2f}"),
        ]
        for label, key, fmt in rows:
            sv, tv = s.get(key, 0), t.get(key, 0)
            if key.endswith("_ms"):
                diff = f"{tv / 1000 - sv / 1000:+.1f}s"
            elif key == "convergence_rate":
                diff = _moved(sv * 100, tv * 100, "pp")
            else:
                diff = _moved(sv, tv)
            lines.append(f"| {label} | {fmt(sv)} | {fmt(tv)} | {diff} |")

        lines.append("\n**逐科目配对**\n")
        lines.append(
            "| 科目 | 单轮 stop / 版本 / 残留 high | 2 轮 stop / 版本 / 残留 high | "
            "单轮耗时 | 2 轮耗时 |"
        )
        lines.append("|---|---|---|---|---|")
        for p in runs.get("paired") or []:
            lines.append(
                f"| {p['subject']} | {p['single_stop']} / {p['single_versions']} / "
                f"{p['single_residual_high']} | {p['two_stop']} / {p['two_versions']} / "
                f"{p['two_residual_high']} | {p['single_ms'] / 1000:.1f}s | "
                f"{p['two_ms'] / 1000:.1f}s |"
            )

        lines.append("\n**stop_reason 分布**\n")
        lines.append(f"- 单轮：{s.get('stop_reasons')}")
        lines.append(f"- 2 轮：{t.get('stop_reasons')}")
        lines.append(
            f"- 经验贴文案来源：单轮 {s.get('guide_sources')}；"
            f"2 轮 {t.get('guide_sources')}"
        )

        d_res = (t.get("avg_residual_high", 0) or 0) - (s.get("avg_residual_high", 0) or 0)
        d_calls = (t.get("avg_llm_calls", 0) or 0) - (s.get("avg_llm_calls", 0) or 0)
        d_ms = (t.get("p50_ms", 0) or 0) - (s.get("p50_ms", 0) or 0)
        lines.append("\n**读数**\n")
        lines.append(f"- 残留 high：2 轮相对单轮 **{d_res:+.2f}**（越负越好）")
        lines.append(f"- LLM 调用：**{d_calls:+.2f}** 次")
        lines.append(f"- P50 节点耗时：**{d_ms / 1000:+.1f}s**")
        lines.append(
            "- 多轮不预设为正收益：若残留 high 没有下降，说明 Critic 的质疑没有被修订"
            "解决，继续修订只是增加成本。"
        )

        lines.append("\n## 5. LLM 主观挑战（辅助，未经人工核对）\n")
        items = runs.get("llm_challenges") or []
        if not items:
            lines.append("本次运行没有产生 LLM 主观挑战。")
        else:
            lines.append(
                "这些条目由 Critic 提出，只用于说明「规则之外还看到了什么」，"
                "**不参与检出率/误报率计算，也未经人工核对**。\n"
            )
            lines.append("| 科目 | 臂 | 严重度 | 挑战 | 证据位置 |")
            lines.append("|---|---|---|---|---|")
            for it in items[:20]:
                claim = str(it.get("claim") or "").replace("|", "／")[:80]
                lines.append(
                    f"| {it.get('subject')} | {it.get('arm')} | {it.get('severity')} | {claim} | "
                    f"`{it.get('evidence_ref')}` |"
                )
            if len(items) > 20:
                lines.append(f"\n（另有 {len(items) - 20} 条未列出）")

    lines.append("\n## 口径说明\n")
    lines.append("- 数据集为**合成注入**：在黄金 Plan 上叠加已知错误，检出率不代表真实 Plan 质量。")
    lines.append("- 每条注入 2 类错误，共 20 条；黄金 10 条。")
    lines.append("- 规则评测不调用 LLM；单轮 vs 2 轮对比才调用，结果写入 "
                 "`reports/adversarial_eval_runs/plans.jsonl`（断点续跑）。")
    lines.append("- 每科目只跑 1 次 2 轮流程，单轮臂取该次运行第 1 轮结束后的快照；"
                 "P50/P95 为描述性统计，不作显著性结论。")
    lines.append("- 耗时是各臂经过节点的耗时合计（含 LLM 等待），不含调度与前端开销。")
    lines.append("- Planner 的「工具」以预注入上下文实现，没有独立工具调用；"
                 "成本用 LLM 调用次数与 token 表示。")
    return "\n".join(lines) + "\n"


def run(
    *,
    limit: int | None = None,
    use_llm: bool | None = None,
    max_rounds: int = 2,
    sleep_s: float = 0.0,
    goals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rows = load()
    if not rows:
        rows = build_dataset()
        save_dataset(rows)
    # 注意：--limit 只限制薄弱科目数量，不截断规则评测集。
    # 截断会把 20 条注入样本切掉，检出率变成 0 且毫无提示。
    summary = evaluate(rows)

    if use_llm is None:
        from cs408rag.generation.llm import get_llm

        use_llm = get_llm().available

    if use_llm:
        print(
            "启用 LLM：将运行单轮/2 轮对比（4 个薄弱科目，约 2–10 分钟）；"
            "加 --no-llm 可全离线只跑规则。"
        )
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        runs = collect_plans(
            goals or DEFAULT_GOALS,
            max_rounds=max_rounds,
            limit=limit,
            sleep_s=sleep_s,
        )
        summary["single_vs_two_round"] = summarize_runs(runs)
    else:
        print("未启用 LLM：只跑规则评测。")
        summary["single_vs_two_round"] = {}

    md = to_markdown(summary)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "adversarial_eval.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "adversarial_eval.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(md)
    return summary


if __name__ == "__main__":
    import argparse

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="对抗 Plan 评测")
    ap.add_argument("--limit", type=int, default=None, help="最多跑几个薄弱科目")
    ap.add_argument("--no-llm", action="store_true", help="完全不调 LLM，只跑规则")
    ap.add_argument("--max-rounds", type=int, default=2, help="2 轮臂的最大修订轮数")
    ap.add_argument("--sleep", type=float, default=0.0, help="每次运行之间休眠秒数")
    args = ap.parse_args()
    run(
        limit=args.limit,
        use_llm=False if args.no_llm else None,
        max_rounds=args.max_rounds,
        sleep_s=args.sleep,
    )