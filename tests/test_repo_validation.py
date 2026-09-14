from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import yaml

from wre.repo_validation import validate_repository

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


def _item(roadmap: dict[str, Any], item_id: str) -> dict[str, Any]:
    items = roadmap["work_items"]
    assert isinstance(items, list)
    for item in items:
        if isinstance(item, dict) and item.get("id") == item_id:
            return item
    raise AssertionError(f"missing work item {item_id}")


def test_current_repository_metadata_is_valid() -> None:
    assert validate_repository(ROOT) == []


def test_rejects_unknown_active_work_item(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    state_path = repo / "PROJECT_STATE.yaml"
    state = _load(state_path)
    state["active_work_item"] = "L404.1"
    _write(state_path, state)

    errors = validate_repository(repo)

    assert any("active_work_item is unknown" in error for error in errors)


def test_rejects_unknown_dependency(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    roadmap_path = repo / "registry/work-items.yaml"
    roadmap = _load(roadmap_path)
    _item(roadmap, "L0.3")["depends_on"] = ["L0.2", "L404.1"]
    _write(roadmap_path, roadmap)

    errors = validate_repository(repo)

    assert any("references unknown dependency: L404.1" in error for error in errors)


def test_rejects_missing_read_before_path(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    roadmap_path = repo / "registry/work-items.yaml"
    roadmap = _load(roadmap_path)
    _item(roadmap, "L0.3")["read_before"].append("docs/DOES_NOT_EXIST.md")
    _write(roadmap_path, roadmap)

    errors = validate_repository(repo)

    assert any("references missing path: docs/DOES_NOT_EXIST.md" in error for error in errors)


def test_rejects_unfinished_active_dependency(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    state = _load(repo / "PROJECT_STATE.yaml")
    roadmap_path = repo / "registry/work-items.yaml"
    roadmap = _load(roadmap_path)

    active_id = state["active_work_item"]
    assert isinstance(active_id, str)
    active_item = _item(roadmap, active_id)
    dependencies = active_item["depends_on"]
    assert isinstance(dependencies, list)
    assert dependencies
    dependency_id = dependencies[0]
    assert isinstance(dependency_id, str)

    _item(roadmap, dependency_id)["status"] = "ready"
    _write(roadmap_path, roadmap)

    errors = validate_repository(repo)

    assert any(f"depends on unfinished {dependency_id}" in error for error in errors)


def test_rejects_missing_pr_template_section(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    template_path = repo / ".github/PULL_REQUEST_TEMPLATE.md"
    template = template_path.read_text(encoding="utf-8")
    template_path.write_text(template.replace("## Test evidence", "## Evidence"), encoding="utf-8")

    errors = validate_repository(repo)

    assert any(
        "pull request template missing required section: ## Test evidence" in error
        for error in errors
    )


def test_rejects_missing_component_implementation_path(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    components_path = repo / "registry/components.yaml"
    components = _load(components_path)
    development = components["components"]["development_quality_gates"]
    development["implementation"].append("src/wre/does_not_exist.py")
    _write(components_path, components)

    errors = validate_repository(repo)

    assert any("src/wre/does_not_exist.py" in error for error in errors)


def test_prior_lot_review_blocks_advancement(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    roadmap_path = repo / "registry/work-items.yaml"
    state_path = repo / "PROJECT_STATE.yaml"
    reviews_path = repo / "registry/reviews.yaml"

    roadmap = _load(roadmap_path)
    for item in roadmap["work_items"]:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id in {
            "L0.1",
            "L0.2",
            "L0.3",
            "L0.4",
            "L0.5",
            "L0.R",
            "L1.1",
            "L1.2",
            "L1.3",
            "L1.4",
            "L1.5",
        }:
            item["status"] = "done"
        elif item_id == "L2.1":
            item["status"] = "ready"
        else:
            item["status"] = "planned"
    _write(roadmap_path, roadmap)

    state = _load(state_path)
    done_ids = [
        item["id"]
        for item in roadmap["work_items"]
        if isinstance(item, dict) and item.get("status") == "done"
    ]
    state.update(
        {
            "active_work_item": "L2.1",
            "status": "ready",
            "lot": "L2",
            "milestone": "M1",
            "last_completed": done_ids,
        }
    )
    _write(state_path, state)

    reviews = _load(reviews_path)
    reviews["lot_reviews"]["L0"] = {
        "status": "passed",
        "reviewed_at": "2026-09-14",
        "evidence": ["test"],
    }
    reviews["milestone_reviews"]["M0"] = {
        "status": "passed",
        "reviewed_at": "2026-09-14",
        "evidence": ["test"],
    }
    _write(reviews_path, reviews)

    errors = validate_repository(repo)

    assert any("prior lot review L1 is not passed" in error for error in errors)


def test_prior_milestone_review_blocks_advancement(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    roadmap_path = repo / "registry/work-items.yaml"
    state_path = repo / "PROJECT_STATE.yaml"
    reviews_path = repo / "registry/reviews.yaml"

    roadmap = _load(roadmap_path)
    completed_prefixes = ("L0.", "L1.", "L2.", "L3.")
    for item in roadmap["work_items"]:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id.startswith(completed_prefixes):
            item["status"] = "done"
        elif item_id == "L4.1":
            item["status"] = "ready"
        else:
            item["status"] = "planned"
    _write(roadmap_path, roadmap)

    state = _load(state_path)
    done_ids = [
        item["id"]
        for item in roadmap["work_items"]
        if isinstance(item, dict) and item.get("status") == "done"
    ]
    state.update(
        {
            "active_work_item": "L4.1",
            "status": "ready",
            "lot": "L4",
            "milestone": "M2",
            "last_completed": done_ids,
        }
    )
    _write(state_path, state)

    reviews = _load(reviews_path)
    for lot_id in ("L0", "L1", "L2", "L3"):
        reviews["lot_reviews"][lot_id] = {
            "status": "passed",
            "reviewed_at": "2026-09-14",
            "evidence": ["test"],
        }
    reviews["milestone_reviews"]["M0"] = {
        "status": "passed",
        "reviewed_at": "2026-09-14",
        "evidence": ["test"],
    }
    _write(reviews_path, reviews)

    errors = validate_repository(repo)

    assert any("prior milestone review M1 is not passed" in error for error in errors)
