"""多轮/多步 Agent 评测（Week 3）。

读取 data/eval/multi_turn_set.jsonl（每行一个会话，含 expect 断言），
逐轮调用 AgentRunner.run，产出指标：

- 澄清率：期望澄清且实际澄清的比例（needs_clarification 命中率）
- 指代消解率：期望改写的问题里 resolved_query 包含关键片段的比例
- 工具失败率：tool_failures / tool_calls
- stop_reason / route 分布、平均轮次、平均成本

用法：
    python -m cs408rag.eval.agent_tasks [--set data/eval/multi_turn_set.jsonl]
        [--limit N] [--out data/eval/multi_turn_report.json]
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from loguru import logger

from cs408rag.agent.graph import AgentRunner

DEFAULT_SET = "data/eval/multi_turn_set.jsonl"
DEFAULT_OUT = "data/eval/multi_turn_report.json"


def load_sessions(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _check_turn(res_dict: dict[str, Any], expect: dict[str, Any]) -> dict[str, bool]:
    """逐条断言，返回 {断言名: 是否通过}。"""
    checks: dict[str, bool] = {}
    if "route" in expect:
        checks["route"] = res_dict.get("route") == expect["route"]
    if expect.get("needs_clarification"):
        checks["clarify"] = bool(res_dict.get("needs_clarification"))
    if "resolved_contains" in expect:
        rq = res_dict.get("resolved_query") or ""
        checks["resolved"] = all(s in rq for s in expect["resolved_contains"])
    if expect.get("no_resolve"):
        checks["no_resolve"] = not res_dict.get("resolved_query")
    if "min_tools" in expect:
        checks["min_tools"] = res_dict.get("n_tool_calls", 0) >= expect["min_tools"]
    if "max_rounds" in expect:
        checks["max_rounds"] = res_dict.get("rounds", 0) <= expect["max_rounds"]
    if "qid_used" in expect:
        # 追问「这题」被消解后，effective qid 落在 turn 记录里
        checks["qid_used"] = True  # 由会话侧断言，见 run_sessions
    return checks


def run_sessions(
    runner: AgentRunner, sessions: list[dict[str, Any]], *, limit: int | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for s in sessions[:limit] if limit else sessions:
        sid = f"eval-{s['id']}-{int(time.time())}"
        for i, turn in enumerate(s["turns"]):
            extra = turn.get("options") or {}
            res = runner.run(
                turn["q"],
                session_id=sid,
                options=extra.get("options"),
                qid=extra.get("qid"),
                mode=extra.get("mode", "qa"),
            )
            d = res.to_dict()
            checks = _check_turn(d, turn.get("expect") or {})
            if (turn.get("expect") or {}).get("qid_used"):
                sess = runner.sessions.get(sid)
                checks["qid_used"] = bool(sess and sess.last_qid == turn["expect"]["qid_used"])
            rows.append(
                {
                    "session": s["id"],
                    "turn": i + 1,
                    "q": turn["q"],
                    "route": d.get("route"),
                    "stop_reason": d.get("stop_reason"),
                    "rounds": d.get("rounds"),
                    "n_tool_calls": d.get("n_tool_calls"),
                    "n_tool_failures": len(d.get("tool_failures") or []),
                    "needs_clarification": d.get("needs_clarification"),
                    "resolved_query": d.get("resolved_query"),
                    "cost_cny": d.get("cost_cny"),
                    "checks": checks,
                    "passed": all(checks.values()) if checks else True,
                }
            )

    n = len(rows) or 1
    clarify_expect = [r for r in rows if "clarify" in r["checks"]]
    resolve_expect = [r for r in rows if "resolved" in r["checks"]]
    total_tools = sum(r["n_tool_calls"] or 0 for r in rows)
    total_fail = sum(r["n_tool_failures"] for r in rows)
    return {
        "n_turns": len(rows),
        "pass_rate": round(sum(1 for r in rows if r["passed"]) / n, 3),
        "clarification_recall": (
            round(sum(1 for r in clarify_expect if r["checks"]["clarify"]) / len(clarify_expect), 3)
            if clarify_expect
            else None
        ),
        "followup_resolve_rate": (
            round(sum(1 for r in resolve_expect if r["checks"]["resolved"]) / len(resolve_expect), 3)
            if resolve_expect
            else None
        ),
        "tool_failure_rate": round(total_fail / total_tools, 3) if total_tools else 0.0,
        "avg_rounds": round(sum(r["rounds"] or 0 for r in rows) / n, 2),
        "total_cost_cny": round(sum(r["cost_cny"] or 0 for r in rows), 4),
        "route_dist": dict(Counter(r["route"] for r in rows)),
        "stop_reason_dist": dict(Counter(r["stop_reason"] for r in rows)),
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default=DEFAULT_SET)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    sessions = load_sessions(Path(args.set))
    logger.info(f"多轮评测：{len(sessions)} 个会话")
    runner = AgentRunner()
    report = run_sessions(runner, sessions, limit=args.limit)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {k: v for k, v in report.items() if k != "rows"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    logger.info(f"报告已写入 {args.out}")


if __name__ == "__main__":
    main()
