from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from wre.roadmap_policy_validation import validate_roadmap_guardrails

ROOT = Path(__file__).resolve().parents[1]


def _copy_repo(tmp_path: Path) -> Path:
    target = tmp_path / "repo"
    shutil.copytree(
        ROOT,
        target,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".pyright",
        ),
    )
    return target


def _load(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_current_roadmap_guardrails_are_enabled() -> None:
    assert validate_roadmap_guardrails(ROOT) == []


def test_each_modernity_sizing_and_performance_guardrail_is_fail_closed(tmp_path: Path) -> None:
    for flag in (
        "activation_requires_candidate_refresh",
        "activation_requires_one_run_complexity_gate",
        "activation_requires_performance_review",
        "final_validation_is_validation_only",
    ):
        repo = _copy_repo(tmp_path / flag)
        path = repo / "registry/work-items.yaml"
        document = _load(path)
        policy = document["policy"]
        assert isinstance(policy, dict)
        policy[flag] = False
        _write(path, document)

        errors = validate_roadmap_guardrails(repo)

        assert errors == [f"roadmap policy must enable {flag}"]
