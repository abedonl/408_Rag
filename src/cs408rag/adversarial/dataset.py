"""对抗评测数据集：黄金 Plan + 注入错误的 Plan。

黄金 Plan 由真实题库构造（`build_golden_plan`），规则应 0 检出。
注入集在黄金 Plan 上叠加 1–2 类已知错误，用于测检出率。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from cs408rag.adversarial.plan import ERROR_TYPES, build_golden_plan, inject
from cs408rag.config import EVAL_DIR

DATASET_PATH = EVAL_DIR / "adversarial_plans.jsonl"
SUBJECTS = ["数据结构", "组成原理", "操作系统", "计算机网络"]


def build_goldens() -> list[dict[str, Any]]:
    """覆盖全科/单科、8/12/16/30 周、不同进度与用户问题的 10 条黄金 Plan。"""
    specs = [
        (0, 12, 10, "full_408", "mid", []),
        (1, 14, 12, "full_408", "new", []),
        (2, 12, 10, "single_subject", "mid", []),
        (3, 10, 8, "single_subject", "new", []),
        (2, 16, 12, "full_408", "adv", []),
        (0, 16, 10, "single_subject", "mid", []),
        (1, 8, 12, "full_408", "new", ["时间不够"]),
        (3, 30, 8, "full_408", "adv", []),
        (2, 10, 12, "single_subject", "mid", ["真题正确率低"]),
        (0, 12, 14, "full_408", "mid", ["基础差"]),
    ]
    progress_map = {
        "new": {s: "未开始" for s in SUBJECTS},
        "mid": {s: "一轮中" for s in SUBJECTS},
        "adv": {s: "一轮完" for s in SUBJECTS},
    }
    out: list[dict[str, Any]] = []
    for i, (si, weeks, hours, scope, pk, concerns) in enumerate(specs):
        subject = SUBJECTS[si]
        plan = build_golden_plan(
            weak_subject=subject,
            weeks=weeks,
            weekly_hours=hours,
            scope=scope,
            progress=progress_map[pk],
            concerns=concerns,
        )
        out.append(
            {
                "id": f"golden-{i:02d}",
                "kind": "golden",
                "labels": [],
                "weak_subject": subject,
                "scope": scope,
                "progress_kind": pk,
                "plan": plan,
            }
        )
    return out


# 互相覆盖的注入组合：同时注入会导致前一个错误被后一个改回去，
# 检出率会被低估（实测 fabricated_ref 70%、time_overrun 57%）。
CONFLICTS = {
    frozenset(("fabricated_ref", "uncited_priority")),
    frozenset(("fabricated_ref", "missing_high_freq_kp")),
    frozenset(("time_overrun", "missing_high_freq_kp")),
    frozenset(("time_overrun", "weak_subject_underweight")),
    frozenset(("missing_high_freq_kp", "prerequisite_order")),
    frozenset(("missing_high_freq_kp", "uncited_priority")),
}


def _pick_pair(rng: random.Random) -> list[str]:
    for _ in range(50):
        a, b = rng.sample(ERROR_TYPES, 2)
        if frozenset((a, b)) not in CONFLICTS:
            return [a, b]
    return ["fabricated_ref", "time_overrun"]


def build_injected(goldens: list[dict[str, Any]], *, n: int = 20, seed: int = 42) -> list[dict[str, Any]]:
    """在黄金 Plan 上叠加 2 类不互相覆盖的错误。"""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(n):
        base = goldens[i % len(goldens)]["plan"]
        labels = _pick_pair(rng)
        plan = base
        for label in labels:
            plan = inject(plan, label)
        out.append(
            {
                "id": f"injected-{i:02d}",
                "kind": "injected",
                "labels": labels,
                "weak_subject": goldens[i % len(goldens)]["weak_subject"],
                "plan": plan,
            }
        )
    return out


def build_dataset() -> list[dict[str, Any]]:
    goldens = build_goldens()
    return goldens + build_injected(goldens)


def save(rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or DATASET_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def load(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or DATASET_PATH
    if not path.exists():
        return build_dataset()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    rows = build_dataset()
    p = save(rows)
    print(f"dataset -> {p}（{len(rows)} 条：golden {sum(r['kind']=='golden' for r in rows)}，"
          f"injected {sum(r['kind']=='injected' for r in rows)}）")