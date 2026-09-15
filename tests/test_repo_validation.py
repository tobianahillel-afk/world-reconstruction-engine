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


def _item_file(repo: Path, item_id: str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    index = _load(repo / "registry/work-items.yaml")
    files = index["work_item_files"]
    assert isinstance(files, list)
    for relative in files:
        assert isinstance(relative, str)
        path = repo / relative
        document = _load(path)
        items = document["work_items"]
        assert isinstance(items, list)
        for item in items:
            if isinstance(item, dict) and item.get("id") == item_id:
                return path, document, item
    raise AssertionError(f"missing work item {item_id}")


def _make_executable(item: dict[str, Any]) -> None:
    item.setdefault("objective", "test executable objective")
    item.setdefault("allowed_scope", ["test-only scope"])
    item.setdefault("out_of_scope", ["everything else"])
    item.setdefault("read_before", ["docs/00_START_HERE.md"])
    item.setdefault("reuse", ["existing test fixture"])
    item.setdefault("acceptance", ["test acceptance"])
    item.setdefault("tests", ["test evidence"])


def test_current_repository_metadata_is_valid() -> None:
    assert validate_repository(ROOT) == []


def test_rejects_unknown_active_work_item(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    state_path = repo / "PROJECT_STATE.yaml"
    state = _load(state_path)
    state["active_work_item"] = "V2L404.1"
    _write(state_path, state)

    errors = validate_repository(repo)

    assert any("active_work_item is unknown" in error for error in errors)


def test_rejects_unknown_dependency_in_split_work_item_file(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    path, document, item = _item_file(repo, "V2L0.2")
    item["depends_on"] = ["V2L0.1", "V2L404.1"]
    _write(path, document)

    errors = validate_repository(repo)

    assert any("references unknown dependency: V2L404.1" in error for error in errors)


def test_rejects_missing_read_before_path(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    path, document, item = _item_file(repo, "V2L0.2")
    read_before = item["read_before"]
    assert isinstance(read_before, list)
    read_before.append("docs/DOES_NOT_EXIST.md")
    _write(path, document)

    errors = validate_repository(repo)

    assert any("references missing path: docs/DOES_NOT_EXIST.md" in error for error in errors)


def test_rejects_unfinished_active_dependency(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    path, document, dependency = _item_file(repo, "V2L0.1")
    dependency["status"] = "planned"
    _write(path, document)

    errors = validate_repository(repo)

    assert any("depends on unfinished V2L0.1" in error for error in errors)


def test_planned_item_may_remain_concise(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    _, _, item = _item_file(repo, "V2L1.1")
    assert item["status"] == "planned"
    assert "objective" not in item

    assert validate_repository(repo) == []


def test_ready_item_requires_full_executable_contract(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    path, document, item = _item_file(repo, "V2L0.2")
    del item["allowed_scope"]
    _write(path, document)

    errors = validate_repository(repo)

    assert any("V2L0.2 requires executable field: allowed_scope" in error for error in errors)


def test_rejects_lot_larger_than_policy_limit(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    path, document, _ = _item_file(repo, "V2L1.1")
    items = document["work_items"]
    assert isinstance(items, list)
    items.append(
        {
            "id": "V2L1.7",
            "lot": "V2L1",
            "title": "oversized lot sentinel",
            "status": "planned",
            "depends_on": ["V2L1.6"],
        }
    )
    _write(path, document)

    errors = validate_repository(repo)

    assert any("V2L1 has 7 work items" in error for error in errors)


def test_rejects_missing_work_item_file(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    index_path = repo / "registry/work-items.yaml"
    index = _load(index_path)
    files = index["work_item_files"]
    assert isinstance(files, list)
    files[0] = "registry/work-items/DOES_NOT_EXIST.yaml"
    _write(index_path, index)

    errors = validate_repository(repo)

    assert any("work item file does not exist" in error for error in errors)


def test_rejects_missing_pr_template_section(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    template_path = repo / ".github/PULL_REQUEST_TEMPLATE.md"
    template = template_path.read_text(encoding="utf-8")
    template_path.write_text(
        template.replace("## Allowed scope", "## Scope"), encoding="utf-8"
    )

    errors = validate_repository(repo)

    assert any(
        "pull request template missing required section: ## Allowed scope" in error
        for error in errors
    )


def test_rejects_missing_component_implementation_path(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    components_path = repo / "registry/components.yaml"
    components = _load(components_path)
    migration = components["components"]["v2_migration_governance"]
    migration["implementation"].append("src/wre/does_not_exist.py")
    _write(components_path, components)

    errors = validate_repository(repo)

    assert any("src/wre/does_not_exist.py" in error for error in errors)


def test_prior_lot_review_blocks_advancement(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)

    m0_path = repo / "registry/work-items/v2m0.yaml"
    m0 = _load(m0_path)
    items = m0["work_items"]
    assert isinstance(items, list)
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        if item_id in {"V2L0.1", "V2L0.2", "V2L0.3"}:
            item["status"] = "done"
            _make_executable(item)
        elif item_id == "V2L1.1":
            item["status"] = "ready"
            _make_executable(item)
        else:
            item["status"] = "planned"
    _write(m0_path, m0)

    state_path = repo / "PROJECT_STATE.yaml"
    state = _load(state_path)
    state.update(
        {
            "active_work_item": "V2L1.1",
            "status": "ready",
            "lot": "V2L1",
            "milestone": "V2M0",
            "last_completed": ["V2L0.1", "V2L0.2", "V2L0.3"],
        }
    )
    _write(state_path, state)

    errors = validate_repository(repo)

    assert any("prior lot review V2L0 is not passed" in error for error in errors)


def test_roadmap_policy_must_remain_deny_by_default(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    index_path = repo / "registry/work-items.yaml"
    index = _load(index_path)
    index["policy"]["deny_by_default_scope"] = False
    _write(index_path, index)

    errors = validate_repository(repo)

    assert any("must enable deny_by_default_scope" in error for error in errors)
