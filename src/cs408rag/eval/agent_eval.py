"""Agent 与单轮 RAG 的分层对比评测。

要回答的问题不是「Agent 是否全面更强」——几乎肯定不是。概念题的答案就在
某一节教材里，多轮检索与工具调用只会引入噪声；Agent 的价值集中在
**单轮检索按定义答不了**的题上：计算题要算，图表题要看图。

因此本评测：

1. 按题型分层（概念 / 计算 / 图表），图表题再按 T1 / T2 / T3 拆开
2. 两条路径用**同一套检索与同一套防泄漏配置**，唯一差别是 Agent 的工具与编排
3. 报告成本：步数、工具调用、token、P50/P95 延迟
4. 结论口径：「用约 N 倍延迟换计算题 X%、图表题 Y% 的提升」

工程约束：LLM 单次 5–15 秒，评测必须**断点续跑**。每题结果即时追加到
jsonl，重跑时跳过已完成项。

用法：
    python -m cs408rag.eval.agent_eval --per-type 10            # 快速验证
    python -m cs408rag.eval.agent_eval --per-type 30            # 正式跑
    python -m cs408rag.eval.agent_eval --paths agent            # 只跑 Agent
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from loguru import logger

from cs408rag.config import EVAL_DIR, REPORTS_DIR, ensure_dirs
from cs408rag.generation.service import DEFAULT_QA_CONFIG

# 评测口径：使用当前 PROD 配置（dense+sparse+BM25、去图谱、zscore、节去重），
# 只额外排除真题解析/试题类 chunk。两条路径必须用同一配置，否则对比不成立。
EVAL_RETRIEVAL_CONFIG = replace(
    DEFAULT_QA_CONFIG,
    exclude_explanations=True,
    exclude_exercise=True,
)

RUNS_DIR = REPORTS_DIR / "agent_eval_runs_prod"


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------


def load_dev_items() -> list[dict[str, Any]]:
    p = EVAL_DIR / "eval_set.json"
    rows = json.loads(p.read_text(encoding="utf-8"))
    return [r for r in rows if not r["is_holdout"]]


def figure_tier(qid: str) -> str:
    from cs408rag.figures.read_figure import read_figure

    try:
        return read_figure(qid).tier
    except Exception:  # noqa: BLE001
        return "unknown"


def stratified_sample(
    items: Sequence[dict[str, Any]],
    *,
    per_type: int,
    question_type: dict[str, str],
    seed: int = 42,
) -> list[dict[str, Any]]:
    """按题型分层抽样，层内按年份均匀分布（等距而非纯随机）。"""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        t = question_type.get(it["qid"], "concept")
        by_type.setdefault(t, []).append(it)

    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for t, rows in by_type.items():
        rows = sorted(rows, key=lambda r: (r["year"], r["qid"]))
        if len(rows) <= per_type:
            picked = rows
        else:
            step = len(rows) / per_type
            picked = [rows[int(i * step)] for i in range(per_type)]
        rng.shuffle(picked)
        for r in picked:
            r["_type"] = t
            if t == "figure":
                r["_tier"] = figure_tier(r["qid"])
        out.extend(picked)
    return out


# ---------------------------------------------------------------------------
# 采集
# ---------------------------------------------------------------------------


def _load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            done[row["qid"]] = row
        except json.JSONDecodeError:
            continue
    return done


def _append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_rag(
    items: Sequence[dict[str, Any]],
    *,
    out_path: Path,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> list[dict[str, Any]]:
    from cs408rag.generation.service import QAService

    done = _load_done(out_path)
    todo = [it for it in items if it["qid"] not in done]
    if limit:
        todo = todo[:limit]
    logger.info(f"[single-turn] 已完成 {len(done)}，待跑 {len(todo)}")

    svc: QAService | None = None
    for i, it in enumerate(todo, 1):
        if svc is None:
            svc = QAService(config=EVAL_RETRIEVAL_CONFIG)
        t0 = time.perf_counter()
        res = svc.answer_quiz(it["question"], it["options"], config=EVAL_RETRIEVAL_CONFIG)
        row = {
            "qid": it["qid"],
            "year": it["year"],
            "subject": it["subject"],
            "type": it.get("_type", "concept"),
            "tier": it.get("_tier", ""),
            "path": "single_turn",
            "choice": res.choice,
            "gold": it["answer"],
            "correct": res.choice == it["answer"],
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "n_steps": 1,
            "n_tools": 0,
            "tokens": res.llm_call.get("prompt_tokens", 0)
            + res.llm_call.get("completion_tokens", 0),
            "route": "single",
            "error": res.error,
            "empty": not (res.answer or "").strip(),
            "answer": (res.answer or "")[:600],
        }
        _append(out_path, row)
        if sleep_s:
            # 部分 endpoint 的 RPM 配额很低（实测连续 10 次只成功 2 次），
            # 主动限速比让客户端反复 429 重试更可控。
            time.sleep(sleep_s)
        done[it["qid"]] = row
        logger.info(
            f"[single-turn] {i}/{len(todo)} {it['qid']} 预测={row['choice']} "
            f"正确={row['correct']} {row['took_ms']:.0f}ms"
        )
    return list(done.values())


def collect_llm_only(
    items: Sequence[dict[str, Any]],
    *,
    out_path: Path,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> list[dict[str, Any]]:
    """LLM-only 基线：不检索、不给上下文，只靠参数化知识。

    这是回答「RAG 到底有没有用」必需的对照组；没有它，单轮 RAG 的准确率
    无法区分是检索贡献还是模型本来就会。
    """
    from cs408rag.generation.llm import get_llm
    from cs408rag.generation.prompt import parse_choice

    done = _load_done(out_path)
    todo = [it for it in items if it["qid"] not in done]
    if limit:
        todo = todo[:limit]
    logger.info(f"[llm-only] 已完成 {len(done)}，待跑 {len(todo)}")

    llm = get_llm()
    for i, it in enumerate(todo, 1):
        opts = "\n".join(f"{k}. {v}" for k, v in (it.get("options") or {}).items())
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 408 考研助手。只能根据你已有的知识作答，不要检索、不要编造来源。"
                    "第一行必须严格写成 `答案：X`，X 为选项字母；之后简要说明依据。"
                ),
            },
            {
                "role": "user",
                "content": f"【题目】\n{it['question']}\n\n【选项】\n{opts}\n\n请作答。",
            },
        ]
        t0 = time.perf_counter()
        resp = llm.chat(messages, max_tokens=1200)
        row = {
            "qid": it["qid"],
            "year": it["year"],
            "subject": it["subject"],
            "type": it.get("_type", "concept"),
            "tier": it.get("_tier", ""),
            "path": "llm_only",
            "choice": parse_choice(resp.text),
            "gold": it["answer"],
            "correct": parse_choice(resp.text) == it["answer"],
            "took_ms": round((time.perf_counter() - t0) * 1000, 1),
            "n_steps": 1,
            "n_tools": 0,
            "tokens": resp.prompt_tokens + resp.completion_tokens,
            "route": "llm_only",
            "error": resp.error,
            "empty": not (resp.text or "").strip(),
            "answer": (resp.text or "")[:600],
        }
        _append(out_path, row)
        if sleep_s:
            time.sleep(sleep_s)
        done[it["qid"]] = row
        logger.info(
            f"[llm-only] {i}/{len(todo)} {it['qid']} 预测={row['choice']} "
            f"正确={row['correct']} {row['took_ms']:.0f}ms"
        )
    return list(done.values())


def collect_agent(
    items: Sequence[dict[str, Any]],
    *,
    out_path: Path,
    limit: int | None = None,
    sleep_s: float = 0.0,
) -> list[dict[str, Any]]:
    from cs408rag.agent.exam_analysis import ExamAnalyzer
    from cs408rag.agent.graph import AgentRunner
    from cs408rag.agent.tools import AgentTools
    from cs408rag.retrieval.pipeline import RetrievalPipeline

    done = _load_done(out_path)
    todo = [it for it in items if it["qid"] not in done]
    if limit:
        todo = todo[:limit]
    logger.info(f"[agent] 已完成 {len(done)}，待跑 {len(todo)}")

    pipeline = RetrievalPipeline()
    # 考点库只保留开发集年份：评测时 Agent 不能引用 held-out 真题，
    # 否则「历年频次」里包含了答案所在年份。
    dev_years = {int(y) for y in range(2009, 2023)}
    analyzer = ExamAnalyzer(year_allowed=dev_years)
    tools = AgentTools(
        pipeline=pipeline,
        analyzer=analyzer,
        config=EVAL_RETRIEVAL_CONFIG,
        eval_mode=True,
    )
    runner = AgentRunner(tools=tools)

    for i, it in enumerate(todo, 1):
        res = runner.run(
            it["question"],
            options=it["options"],
            subject=it["subject"],
            qid=it["qid"],
            mode="quiz",
        )
        row = {
            "qid": it["qid"],
            "year": it["year"],
            "subject": it["subject"],
            "type": it.get("_type", "concept"),
            "tier": it.get("_tier", ""),
            "path": "agent",
            "choice": res.choice,
            "gold": it["answer"],
            "correct": res.choice == it["answer"],
            "took_ms": round(res.took_ms, 1),
            "n_steps": len(res.steps),
            "n_tools": len(res.tool_calls),
            "tools_used": [c["tool"] for c in res.tool_calls],
            "tokens": (res.tokens.get("prompt", 0) + res.tokens.get("completion", 0)),
            "route": res.route,
            "error": res.error,
            "empty": res.empty or not (res.answer or "").strip(),
            "answer": (res.answer or "")[:600],
        }
        _append(out_path, row)
        if sleep_s:
            # 部分 endpoint 的 RPM 配额很低（实测连续 10 次只成功 2 次），
            # 主动限速比让客户端反复 429 重试更可控。
            time.sleep(sleep_s)
        done[it["qid"]] = row
        logger.info(
            f"[agent] {i}/{len(todo)} {it['qid']} route={row['route']} "
            f"预测={row['choice']} 正确={row['correct']} "
            f"{row['took_ms']:.0f}ms tools={row['n_tools']}"
        )
    return list(done.values())


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    n: int = 0
    correct: int = 0
    empty: int = 0
    latencies: list[float] = None  # type: ignore[assignment]
    tokens: list[int] = None  # type: ignore[assignment]
    steps: list[int] = None  # type: ignore[assignment]
    tools: list[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.latencies = [] if self.latencies is None else self.latencies
        self.tokens = [] if self.tokens is None else self.tokens
        self.steps = [] if self.steps is None else self.steps
        self.tools = [] if self.tools is None else self.tools

    def add(self, row: dict[str, Any]) -> None:
        self.n += 1
        self.correct += int(bool(row.get("correct")))
        self.empty += int(bool(row.get("empty")))
        self.latencies.append(float(row.get("took_ms") or 0))
        self.tokens.append(int(row.get("tokens") or 0))
        self.steps.append(int(row.get("n_steps") or 0))
        self.tools.append(int(row.get("n_tools") or 0))

    def to_dict(self) -> dict[str, Any]:
        n = max(self.n, 1)
        p50 = statistics.median(self.latencies) if self.latencies else 0
        p95 = (
            statistics.quantiles(self.latencies, n=20)[-1]
            if len(self.latencies) >= 20
            else (max(self.latencies) if self.latencies else 0)
        )
        return {
            "n": self.n,
            "accuracy": round(self.correct / n, 4),
            "correct": self.correct,
            "empty": self.empty,
            "empty_rate": round(self.empty / n, 4),
            "p50_ms": round(p50, 0),
            "p95_ms": round(p95, 0),
            "avg_tokens": round(sum(self.tokens) / n, 0),
            "avg_steps": round(sum(self.steps) / n, 2),
            "avg_tools": round(sum(self.tools) / n, 2),
        }


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    by_path: dict[str, dict[str, Cell]] = {}
    for r in rows:
        key = r.get("tier") if r.get("type") == "figure" else r.get("type", "concept")
        by_path.setdefault(r["path"], {}).setdefault(key, Cell()).add(r)
        by_path[r["path"]].setdefault("__all__", Cell()).add(r)

    out: dict[str, Any] = {"paths": {}}
    for path, cells in by_path.items():
        out["paths"][path] = {k: v.to_dict() for k, v in cells.items()}
    return out


def paired(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """同一题上两条路径的对错配对，回答「Agent 提升了哪些题、又弄坏了哪些」。"""
    by_qid: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        by_qid.setdefault(r["qid"], {})[r["path"]] = r
    both = {
        q: v for q, v in by_qid.items() if "agent" in v and "single_turn" in v
    }
    out: dict[str, Any] = {"n_paired": len(both)}
    if not both:
        return out

    wins = [
        q
        for q, v in both.items()
        if v["agent"]["correct"] and not v["single_turn"]["correct"]
    ]
    losses = [
        q
        for q, v in both.items()
        if v["single_turn"]["correct"] and not v["agent"]["correct"]
    ]
    by_type: dict[str, dict[str, int]] = {}
    for q, v in both.items():
        t = v["agent"].get("type", "concept")
        by_type.setdefault(t, {"n": 0, "agent_win": 0, "agent_loss": 0})
        by_type[t]["n"] += 1
        if q in wins:
            by_type[t]["agent_win"] += 1
        if q in losses:
            by_type[t]["agent_loss"] += 1
    out.update(
        {
            "agent_wins": len(wins),
            "agent_losses": len(losses),
            "win_qids": wins[:30],
            "loss_qids": losses[:30],
            "by_type": by_type,
        }
    )
    return out


def _bootstrap_mean_ci(
    values: Sequence[float], *, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return (lo, hi)


def bootstrap_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy 的 95% bootstrap CI + 路径间的配对差值 CI。

    25/60 题这种小样本，只看点估计会过度解读；配对 bootstrap 能回答
    「Agent 比单轮低 16pp，这个差值是否稳定」。
    """
    out: dict[str, Any] = {"accuracy": {}, "paired": {}}
    for path in ("llm_only", "single_turn", "agent"):
        vals = [1.0 if r.get("correct") else 0.0 for r in rows if r.get("path") == path]
        if not vals:
            continue
        lo, hi = _bootstrap_mean_ci(vals)
        out["accuracy"][path] = {
            "n": len(vals),
            "accuracy": round(sum(vals) / len(vals), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    by_qid: dict[str, dict[str, bool]] = {}
    for r in rows:
        by_qid.setdefault(r["qid"], {})[r["path"]] = bool(r.get("correct"))
    for a, b in (
        ("agent", "single_turn"),
        ("single_turn", "llm_only"),
        ("agent", "llm_only"),
    ):
        diffs = [
            (1.0 if v.get(a) else 0.0) - (1.0 if v.get(b) else 0.0)
            for v in by_qid.values()
            if a in v and b in v
        ]
        if not diffs:
            continue
        lo, hi = _bootstrap_mean_ci(diffs)
        out["paired"][f"{a}-{b}"] = {
            "n": len(diffs),
            "diff": round(sum(diffs) / len(diffs), 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }
    return out


def to_markdown(
    summary: dict[str, Any],
    pair: dict[str, Any],
    *,
    n_items: int,
    boot: dict[str, Any] | None = None,
) -> str:
    lines = [f"# LLM-only / 单轮 RAG / Agent 分层对比（开发集，{n_items} 题抽样）\n"]
    lines.append(
        "LLM-only 不检索；单轮与 Agent 使用**同一套 PROD 检索配置**"
        "（dense+sparse+BM25、去图谱、zscore、节去重、排除真题解析）。\n"
    )

    labels = {
        "concept": "概念题",
        "calc": "计算题",
        "figure": "图表题（全部）",
        "T1": "图表 T1（拓扑可重建）",
        "T2": "图表 T2（仅文本）",
        "T3": "图表 T3（位图，不可用）",
        "cross_subject": "跨科目题",
        "__all__": "**总体**",
    }
    path_labels = {
        "llm_only": "纯 LLM（无检索）",
        "single_turn": "单轮 RAG",
        "agent": "Agent",
    }
    keys = ["concept", "calc", "figure", "T1", "T2", "T3", "cross_subject", "__all__"]

    lines.append("## 1. 准确率与成本\n")
    lines.append(
        "| 分层 | 路径 | 题数 | 准确率 | 空回答率 | P50 | P95 | 平均 token | 平均步数 | 平均工具调用 |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for k in keys:
        for path in ("llm_only", "single_turn", "agent"):
            cell = (summary["paths"].get(path) or {}).get(k)
            if not cell:
                continue
            lines.append(
                f"| {labels.get(k, k)} | {path_labels.get(path, path)} | {cell['n']} | "
                f"{cell['accuracy']:.1%} | {cell.get('empty_rate', 0):.1%} | "
                f"{cell['p50_ms'] / 1000:.1f}s | {cell['p95_ms'] / 1000:.1f}s | "
                f"{cell['avg_tokens']:.0f} | {cell['avg_steps']:.1f} | {cell['avg_tools']:.1f} |"
            )

    b = boot or summary.get("bootstrap") or {}
    if b.get("accuracy") or b.get("paired"):
        lines.append("\n## 2. 准确率与 RAG 增量（bootstrap 95% CI）\n")
        if b.get("accuracy"):
            lines.append("| 路径 | 题数 | Accuracy | 95% CI |")
            lines.append("|---|---|---|---|")
            for path, v in b["accuracy"].items():
                lo, hi = v["ci95"]
                lines.append(
                    f"| {path_labels.get(path, path)} | {v['n']} | {v['accuracy']:.1%} | "
                    f"[{lo:.1%}, {hi:.1%}] |"
                )
        if b.get("paired"):
            diff_labels = {
                "agent-single_turn": "Agent − 单轮",
                "single_turn-llm_only": "单轮 − LLM-only（RAG 增量）",
                "agent-llm_only": "Agent − LLM-only",
            }
            lines.append("\n| 配对差值 | 配对题数 | 差值 | 95% CI |")
            lines.append("|---|---|---|---|")
            for key, v in b["paired"].items():
                lo, hi = v["ci95"]
                lines.append(
                    f"| {diff_labels.get(key, key)} | {v['n']} | {v['diff']:+.1%} | "
                    f"[{lo:+.1%}, {hi:+.1%}] |"
                )
        lines.append(
            "\n> CI 包含 0 时不能宣称「有提升/有下降」；小样本区间会很宽，"
            "这是结论的一部分，不是需要隐藏的噪声。\n"
        )

    if pair.get("n_paired"):
        lines.append("\n## 3. 配对比较（同一道题上 Agent vs 单轮）\n")
        lines.append(f"- 配对题数：**{pair['n_paired']}**")
        lines.append(f"- Agent 改对：**{pair['agent_wins']}** 题")
        lines.append(f"- Agent 改错：**{pair['agent_losses']}** 题")
        if pair.get("by_type"):
            lines.append("\n| 题型 | 配对数 | Agent 改对 | Agent 改错 |")
            lines.append("|---|---|---|---|")
            for t, v in pair["by_type"].items():
                lines.append(
                    f"| {labels.get(t, t)} | {v['n']} | {v['agent_win']} | {v['agent_loss']} |"
                )
        if pair.get("win_qids"):
            lines.append(f"\nAgent 改对的题：{'、'.join(pair['win_qids'])}")
        if pair.get("loss_qids"):
            lines.append(f"\nAgent 改错的题：{'、'.join(pair['loss_qids'])}")

    lines.append(
        "\n## 4. 口径说明\n"
        "- 开发集（2009–2022），held-out 不参与调参。\n"
        "- 真题解析 chunk 对所有路径都排除：解析逐字含题干，dev 题会自匹配。\n"
        "- LLM-only 不给任何上下文，只靠参数化知识；它是判断「RAG 是否真的有用」的对照组。\n"
        "- Agent 的考点统计只使用开发集年份，评测时 `get_kp_questions` 不返回答案、"
        "且排除当前题。\n"
        "- 空回答单独统计，不算作正确，也不从分母里剔除；CI 用 2000 次 bootstrap。\n"
        "- 图表 T3 的图是位图，当前无可用 VLM，因此 T3 上 Agent 与单轮 RAG 实质相同；"
        "**T1/T2 的增益不得记在检索链路的账上**。\n"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run(
    *,
    per_type: int = 10,
    paths: Sequence[str] = ("llm_only", "single_turn", "agent"),
    limit: int | None = None,
    sleep_s: float = 0.0,
    report_only: bool = False,
) -> dict[str, Any]:
    ensure_dirs()
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    from cs408rag.agent.exam_analysis import ExamAnalyzer

    dev = load_dev_items()
    analyzer = ExamAnalyzer(year_allowed={int(y) for y in range(2009, 2023)})
    items = stratified_sample(dev, per_type=per_type, question_type=analyzer.type_of)
    if limit:
        # limit 必须作用在「同一批题」上；如果每个 collector 各自截断，
        # LLM-only 会跑 4 题、单轮/Agent 只跑 2 题，配对对比直接失效。
        items = items[:limit]
    logger.info(
        "抽样：" + ", ".join(
            f"{t}={sum(1 for i in items if i['_type'] == t)}"
            for t in sorted({i["_type"] for i in items})
        )
    )

    all_rows: list[dict[str, Any]] = []
    if report_only:
        by_path: dict[str, dict[str, dict[str, Any]]] = {}
        for name in ("llm_only", "single_turn", "agent"):
            p = RUNS_DIR / f"{name}.jsonl"
            if p.exists():
                by_path[name] = _load_done(p)
        if not by_path:
            raise SystemExit("没有可复用的逐题结果；先跑一次 --paths ...")
        # 只保留三条路径都做过的题，否则「总体准确率」的分母不同，
        # 不能直接横向比较；各路径多跑的部分仍留在 jsonl 里。
        common: set[str] | None = None
        for rows_map in by_path.values():
            qids = set(rows_map)
            common = qids if common is None else (common & qids)
        common = common or set()
        all_rows = [
            row
            for rows_map in by_path.values()
            for qid, row in rows_map.items()
            if qid in common
        ]
    else:
        if "llm_only" in paths:
            all_rows += collect_llm_only(
                items, out_path=RUNS_DIR / "llm_only.jsonl", limit=None, sleep_s=sleep_s
            )
        if "single_turn" in paths:
            all_rows += collect_rag(
                items, out_path=RUNS_DIR / "single_turn.jsonl", limit=None, sleep_s=sleep_s
            )
        if "agent" in paths:
            all_rows += collect_agent(
                items, out_path=RUNS_DIR / "agent.jsonl", limit=None, sleep_s=sleep_s
            )

    summary = summarize(all_rows)
    pair = paired(all_rows)
    boot = bootstrap_summary(all_rows)
    n_items = len({r["qid"] for r in all_rows}) if all_rows else len(items)
    md = to_markdown(summary, pair, n_items=n_items, boot=boot)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "agent_eval.md").write_text(md, encoding="utf-8")
    (REPORTS_DIR / "agent_eval.json").write_text(
        json.dumps(
            {"summary": summary, "paired": pair, "bootstrap": boot, "n_items": n_items},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(md)
    return {"markdown": md, "summary": summary, "paired": pair, "bootstrap": boot}


if __name__ == "__main__":
    import argparse
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    ap = argparse.ArgumentParser(description="LLM-only / 单轮 RAG / Agent 分层评测")
    ap.add_argument("--per-type", type=int, default=10, help="每层抽多少题")
    ap.add_argument("--paths", default="llm_only,single_turn,agent")
    ap.add_argument("--limit", type=int, default=None, help="本次最多跑多少题（三条路径共用同一批题）")
    ap.add_argument("--sleep", type=float, default=0.0, help="每题之间休眠秒数（避开 RPM 限制）")
    ap.add_argument(
        "--report-only",
        action="store_true",
        help="不调用 LLM，只用已有逐题结果重算报告（endpoint 限流时用）",
    )
    args = ap.parse_args()
    run(
        per_type=args.per_type,
        paths=args.paths.split(","),
        limit=args.limit,
        sleep_s=args.sleep,
        report_only=args.report_only,
    )