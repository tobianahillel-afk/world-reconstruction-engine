from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

_REQUIRED_TRUE_POLICY_FLAGS = (
    "activation_requires_candidate_refresh",
    "activation_requires_one_run_complexity_gate",
    "activation_requires_performance_review",
    "final_validation_is_validation_only",
)


def validate_roadmap_guardrails(root: Path) -> list[str]:
    """Validate audit-critical roadmap guardrails used by the repository entrypoint."""

    path = root.resolve() / "registry/work-items.yaml"
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return [f"cannot parse registry/work-items.yaml guardrails: {exc}"]

    if not isinstance(data, dict):
        return ["registry/work-items.yaml must contain a YAML mapping for guardrail validation"]

    document = cast(dict[str, Any], data)
    policy = document.get("policy")
    if not isinstance(policy, dict):
        return ["registry/work-items.yaml policy must be a mapping for guardrail validation"]

    errors: list[str] = []
    for flag in _REQUIRED_TRUE_POLICY_FLAGS:
        if policy.get(flag) is not True:
            errors.append(f"roadmap policy must enable {flag}")
    return errors
