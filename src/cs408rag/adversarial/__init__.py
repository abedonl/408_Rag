"""对抗性 Plan：规则校验、角色编排与评测。"""

from cs408rag.adversarial.plan import (
    ERROR_TYPES,
    build_golden_plan,
    inject,
)
from cs408rag.adversarial.rules import Challenge, review_plan
from cs408rag.adversarial.schema import plan_diff, plan_digest, validate_plan

__all__ = [
    "ERROR_TYPES",
    "Challenge",
    "build_golden_plan",
    "inject",
    "plan_digest",
    "plan_diff",
    "review_plan",
    "validate_plan",
]