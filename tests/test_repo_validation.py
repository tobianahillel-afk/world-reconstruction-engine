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


def _adapter_registry_path(repo: Path) -> Path:
    return repo / "registry/adapter-models.yaml"


def _valid_adapter_entry(
    adapter_id: str,
    *,
    dependency_ref: str = "exifread",
    with_model: bool = False,
    shipping_status: str = "approved",
    license_review: str = "approved",
) -> dict[str, Any]:
    model: dict[str, Any] | None = None
    checkpoint: dict[str, Any] | None = None
    resume_mode = "unsupported"
    if with_model:
        model = {"name": "example-model", "version": "1.2.3", "revision": "rev-1"}
        checkpoint = {"identifier": "weights-v1", "sha256": "a" * 64}
        resume_mode = "checkpoint"

    return {
        "adapter_id": adapter_id,
        "capability": {
            "name": "geometry.reconstruction",
            "input_kinds": ["image.observation"],
            "output_kinds": ["geometry.sparse"],
        },
        "producer": {
            "implementation": "wre.adapters.example",
            "version": "1.0.0",
            "revision": None,
        },
        "dependency_refs": [dependency_ref],
        "model": model,
        "checkpoint": checkpoint,
        "artifact_key_hardware_policy": "omitted",
        "license": {
            "direct": "BSD-3-Clause",
            "review": license_review,
            "transitive_notes": "Synthetic test fixture only.",
            "redistribution_notes": "Synthetic test fixture only.",
        },
        "shipping_status": shipping_status,
        "reproducibility_notes": "Synthetic validator fixture.",
        "failure_signals": ["input_missing", "solver_failed"],
        "metric_names": ["latency_ms", "reprojection_error"],
        "resume_mode": resume_mode,
    }


def _write_adapter_entries(repo: Path, entries: list[dict[str, Any]]) -> None:
    _write(_adapter_registry_path(repo), {"schema_version": 1, "entries": entries})


def test_current_repository_metadata_is_valid() -> None:
    assert validate_repository(ROOT) == []


def test_adapter_registry_accepts_empty_and_synthetic_exact_entries(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    assert validate_repository(repo) == []

    _write_adapter_entries(
        repo,
        [
            _valid_adapter_entry("alpha.native"),
            _valid_adapter_entry("beta.learned", with_model=True),
        ],
    )

    assert validate_repository(repo) == []


def test_adapter_registry_fails_closed_for_root_and_schema_version(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    registry_path = _adapter_registry_path(repo)

    registry_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")
    errors = validate_repository(repo)
    assert any("adapter-models.yaml must contain a YAML mapping" in error for error in errors)

    _write(registry_path, {"schema_version": 2, "entries": []})
    errors = validate_repository(repo)
    assert any("schema_version must be integer 1" in error for error in errors)

    _write(registry_path, {"schema_version": 1, "entries": {}, "unexpected": True})
    errors = validate_repository(repo)
    assert any("undeclared fields" in error for error in errors)
    assert any("entries must be a list" in error for error in errors)


def test_adapter_registry_rejects_missing_extra_wrong_type_token_enum_blank_and_sha(
    tmp_path: Path,
) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.native", with_model=True)
    del entry["resume_mode"]
    entry["unexpected"] = "scope-creep"
    entry["capability"] = []
    entry["adapter_id"] = "Invalid Token"
    entry["producer"]["implementation"] = " "
    entry["artifact_key_hardware_policy"] = "sometimes"
    entry["shipping_status"] = "maybe"
    entry["license"]["review"] = "unknown"
    entry["checkpoint"]["sha256"] = "A" * 64
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any("missing fields: ['resume_mode']" in error for error in errors)
    assert any("undeclared fields: ['unexpected']" in error for error in errors)
    assert any("capability must be a mapping" in error for error in errors)
    assert any("adapter_id must be a valid registry token" in error for error in errors)
    assert any("producer.implementation must be a non-blank string" in error for error in errors)
    assert any("artifact_key_hardware_policy has invalid value" in error for error in errors)
    assert any("shipping_status has invalid value" in error for error in errors)
    assert any("license.review has invalid value" in error for error in errors)
    assert any("checkpoint.sha256 must be exactly 64 lowercase hex" in error for error in errors)


def test_adapter_registry_rejects_duplicate_and_unsorted_adapter_ids(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    _write_adapter_entries(
        repo,
        [
            _valid_adapter_entry("beta.adapter"),
            _valid_adapter_entry("alpha.adapter"),
            _valid_adapter_entry("alpha.adapter"),
        ],
    )

    errors = validate_repository(repo)

    assert any("adapter_id values must be unique" in error for error in errors)
    assert any("canonical lexicographic adapter_id order" in error for error in errors)


def test_adapter_registry_rejects_duplicate_or_unsorted_set_like_collections(
    tmp_path: Path,
) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.native")
    entry["capability"]["input_kinds"] = ["image.observation", "image.observation"]
    entry["capability"]["output_kinds"] = ["z.output", "a.output"]
    entry["dependency_refs"] = ["ffmpeg", "exifread"]
    entry["failure_signals"] = ["solver_failed", "input_missing"]
    entry["metric_names"] = ["latency_ms", "latency_ms"]
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any("capability.input_kinds must not contain duplicates" in error for error in errors)
    assert any(
        "capability.output_kinds must be in canonical lexicographic order" in error
        for error in errors
    )
    assert any(
        "dependency_refs must be in canonical lexicographic order" in error for error in errors
    )
    assert any(
        "failure_signals must be in canonical lexicographic order" in error for error in errors
    )
    assert any("metric_names must not contain duplicates" in error for error in errors)


def test_adapter_registry_rejects_empty_outputs_but_allows_other_empty_sets(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.native")
    entry["capability"]["input_kinds"] = []
    entry["dependency_refs"] = []
    entry["failure_signals"] = []
    entry["metric_names"] = []
    _write_adapter_entries(repo, [entry])
    assert validate_repository(repo) == []

    entry["capability"]["output_kinds"] = []
    _write_adapter_entries(repo, [entry])
    errors = validate_repository(repo)
    assert any("capability.output_kinds must not be empty" in error for error in errors)


def test_adapter_registry_rejects_unknown_dependency_reference(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.native")
    entry["dependency_refs"] = ["does-not-exist"]
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any("references unknown dependency: does-not-exist" in error for error in errors)
    dependencies = _load(repo / "registry/dependencies.yaml")
    assert "does-not-exist" not in dependencies["dependencies"]


def test_adapter_registry_rejects_floating_latest_versions(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.learned", with_model=True)
    entry["producer"]["version"] = "LATEST"
    entry["model"]["version"] = " latest "
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any("producer.version cannot use floating latest" in error for error in errors)
    assert any("model.version cannot use floating latest" in error for error in errors)


def test_adapter_registry_rejects_checkpoint_without_model(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.learned", with_model=True)
    entry["model"] = None
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any("checkpoint requires model" in error for error in errors)


def test_adapter_registry_enforces_approved_shipping_license_evidence(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry("alpha.native", license_review="pending")
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)
    assert any(
        "approved shipping requires approved entry license review" in error for error in errors
    )

    entry = _valid_adapter_entry("alpha.native", dependency_ref="limap")
    _write_adapter_entries(repo, [entry])
    errors = validate_repository(repo)
    assert any(
        "approved shipping requires approved dependency license review: limap" in error
        for error in errors
    )


def test_adapter_registry_allows_pending_review_for_nonapproved_shipping(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry(
        "alpha.experimental",
        dependency_ref="limap",
        shipping_status="benchmark-only",
        license_review="pending",
    )
    _write_adapter_entries(repo, [entry])

    assert validate_repository(repo) == []


def test_adapter_registry_blocked_license_requires_blocked_shipping(tmp_path: Path) -> None:
    repo = _copy_repo(tmp_path)
    entry = _valid_adapter_entry(
        "alpha.blocked",
        shipping_status="experimental",
        license_review="blocked",
    )
    _write_adapter_entries(repo, [entry])

    errors = validate_repository(repo)

    assert any(
        "blocked entry license review requires blocked shipping_status" in error for error in errors
    )


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
    index = _load(repo / "registry/work-items.yaml")
    files = index["work_item_files"]
    assert isinstance(files, list)

    concise_planned_items: list[dict[str, Any]] = []
    for relative in files:
        assert isinstance(relative, str)
        document = _load(repo / relative)
        items = document["work_items"]
        assert isinstance(items, list)
        concise_planned_items.extend(
            item
            for item in items
            if isinstance(item, dict)
            and item.get("status") == "planned"
            and "objective" not in item
        )

    assert concise_planned_items
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
    template_path.write_text(template.replace("## Allowed scope", "## Scope"), encoding="utf-8")

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

    reviews_path = repo / "registry/reviews.yaml"
    reviews = _load(reviews_path)
    reviews["lot_reviews"]["V2L0"] = {
        "status": "pending",
        "reviewed_at": None,
        "evidence": [],
    }
    _write(reviews_path, reviews)

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
