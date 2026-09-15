from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

ACTIVE_STATUSES = {"ready", "in_progress", "in_review"}
CONTRACT_REQUIRED_STATUSES = ACTIVE_STATUSES | {"done", "blocked"}
REVIEW_STATUSES = {"pending", "in_review", "passed", "failed"}
REQUIRED_PR_SECTIONS = (
    "## Work item",
    "## Objective",
    "## Allowed scope",
    "## Reuse",
    "## Invariants / ownership boundaries",
    "## Acceptance criteria",
    "## Test evidence",
    "## Review checklist",
    "## Known limitations / follow-up",
)
REQUIRED_EXECUTABLE_FIELDS = (
    "objective",
    "allowed_scope",
    "out_of_scope",
    "depends_on",
    "read_before",
    "reuse",
    "acceptance",
    "tests",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _append_missing_path(errors: list[str], root: Path, path_value: object, context: str) -> None:
    if not isinstance(path_value, str):
        errors.append(f"{context} path must be a string: {path_value!r}")
        return
    if not (root / path_value).exists():
        errors.append(f"{context} references missing path: {path_value}")


def _ordered_lots(milestones: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    ordered: list[str] = []
    owner: dict[str, str] = {}
    for milestone_id, milestone in milestones.items():
        if not isinstance(milestone, dict):
            continue
        lot_ids = milestone.get("lots", [])
        if not isinstance(lot_ids, list):
            continue
        for lot_id in lot_ids:
            if isinstance(lot_id, str):
                ordered.append(lot_id)
                owner[lot_id] = milestone_id
    return ordered, owner


def _require_non_blank_text(errors: list[str], value: object, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{context} must be a non-blank string")


def _require_string_list(
    errors: list[str],
    value: object,
    context: str,
    *,
    non_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{context} must be a list")
        return []
    if non_empty and not value:
        errors.append(f"{context} must not be empty")
    normalized: list[str] = []
    for entry in value:
        if not isinstance(entry, str) or not entry.strip():
            errors.append(f"{context} entries must be non-blank strings: {entry!r}")
            continue
        normalized.append(entry)
    return normalized


def _load_work_items(
    root: Path,
    roadmap: dict[str, Any],
    milestones: dict[str, Any],
    lots: dict[str, Any],
    errors: list[str],
) -> list[dict[str, Any]]:
    work_item_files = roadmap.get("work_item_files")
    if not isinstance(work_item_files, list) or not work_item_files:
        errors.append("registry/work-items.yaml must define non-empty work_item_files")
        return []
    normalized_files = [str(item) for item in work_item_files]
    if len(normalized_files) != len(set(normalized_files)):
        errors.append("work_item_files cannot contain duplicates")

    _, lot_to_milestone = _ordered_lots(milestones)
    loaded: list[dict[str, Any]] = []
    seen_file_milestones: set[str] = set()

    for relative in work_item_files:
        if not isinstance(relative, str):
            errors.append(f"work_item_files entry must be a string: {relative!r}")
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"work item file does not exist: {relative}")
            continue
        try:
            document = _load_mapping(path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            errors.append(f"cannot parse work item file {relative}: {exc}")
            continue

        milestone_id = document.get("milestone")
        if not isinstance(milestone_id, str) or milestone_id not in milestones:
            errors.append(f"{relative} references unknown milestone: {milestone_id!r}")
            continue
        if milestone_id in seen_file_milestones:
            errors.append(f"more than one work item file declares milestone {milestone_id}")
        seen_file_milestones.add(milestone_id)

        file_items = document.get("work_items")
        if not isinstance(file_items, list):
            errors.append(f"{relative}.work_items must be a list")
            continue
        for item in file_items:
            if not isinstance(item, dict):
                errors.append(f"work item in {relative} must be a mapping: {item!r}")
                continue
            lot_id = item.get("lot")
            if lot_id not in lots:
                errors.append(f"work item {item.get('id')!r} references unknown lot: {lot_id!r}")
            elif lot_to_milestone.get(str(lot_id)) != milestone_id:
                errors.append(
                    f"work item {item.get('id')!r} belongs to {lot_id!r}, "
                    f"which is not owned by {milestone_id}"
                )
            loaded.append(item)

    missing = set(milestones) - seen_file_milestones
    extra = seen_file_milestones - set(milestones)
    if missing or extra:
        errors.append(
            "work item file milestone coverage mismatch; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return loaded


def _validate_executable_contract(errors: list[str], item_id: str, item: dict[str, Any]) -> None:
    for field in REQUIRED_EXECUTABLE_FIELDS:
        if field not in item:
            errors.append(f"{item_id} requires executable field: {field}")
    _require_non_blank_text(errors, item.get("objective"), f"{item_id}.objective")
    _require_string_list(
        errors,
        item.get("allowed_scope"),
        f"{item_id}.allowed_scope",
        non_empty=True,
    )
    _require_string_list(
        errors,
        item.get("out_of_scope"),
        f"{item_id}.out_of_scope",
    )
    _require_string_list(errors, item.get("reuse"), f"{item_id}.reuse")
    _require_string_list(
        errors,
        item.get("acceptance"),
        f"{item_id}.acceptance",
        non_empty=True,
    )
    _require_string_list(
        errors,
        item.get("tests"),
        f"{item_id}.tests",
        non_empty=True,
    )


def _validate_reviews(
    errors: list[str],
    lots: dict[str, Any],
    milestones: dict[str, Any],
    reviews_doc: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    lot_reviews = reviews_doc.get("lot_reviews", {})
    milestone_reviews = reviews_doc.get("milestone_reviews", {})
    if not isinstance(lot_reviews, dict) or not isinstance(milestone_reviews, dict):
        errors.append(
            "registry/reviews.yaml must define lot_reviews and milestone_reviews mappings"
        )
        return {}, {}
    if set(lot_reviews) != set(lots):
        errors.append("lot review coverage must exactly match roadmap lots")
    if set(milestone_reviews) != set(milestones):
        errors.append("milestone review coverage must exactly match roadmap milestones")

    for review_kind, review_map in (
        ("lot", lot_reviews),
        ("milestone", milestone_reviews),
    ):
        for target, review in review_map.items():
            if not isinstance(review, dict):
                errors.append(f"{review_kind} review {target} must be a mapping")
                continue
            status = review.get("status")
            if status not in REVIEW_STATUSES:
                errors.append(f"{review_kind} review {target} has invalid status: {status!r}")
            evidence = review.get("evidence", [])
            if not isinstance(evidence, list):
                errors.append(f"{review_kind} review {target}.evidence must be a list")
            if status == "passed" and not evidence:
                errors.append(f"passed {review_kind} review {target} must contain evidence")
    return lot_reviews, milestone_reviews


def validate_repository(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []

    required_files = (
        "PROJECT_STATE.yaml",
        "registry/work-items.yaml",
        "registry/components.yaml",
        "registry/reviews.yaml",
        ".github/PULL_REQUEST_TEMPLATE.md",
    )
    for relative in required_files:
        if not (root / relative).is_file():
            errors.append(f"missing required repository file: {relative}")
    if errors:
        return errors

    try:
        state = _load_mapping(root / "PROJECT_STATE.yaml")
        roadmap = _load_mapping(root / "registry/work-items.yaml")
        components_doc = _load_mapping(root / "registry/components.yaml")
        reviews_doc = _load_mapping(root / "registry/reviews.yaml")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"cannot parse repository metadata: {exc}"]

    if roadmap.get("schema_version") != 2:
        errors.append("registry/work-items.yaml schema_version must be 2")
    if state.get("schema_version") != 2:
        errors.append("PROJECT_STATE.yaml schema_version must be 2")
    if state.get("roadmap_version") != roadmap.get("roadmap_version"):
        errors.append("PROJECT_STATE roadmap_version must match registry/work-items.yaml")

    policy = roadmap.get("policy", {})
    if not isinstance(policy, dict):
        errors.append("registry/work-items.yaml policy must be a mapping")
        policy = {}
    if policy.get("deny_by_default_scope") is not True:
        errors.append("roadmap policy must enable deny_by_default_scope")
    if policy.get("ready_requires_full_contract") is not True:
        errors.append("roadmap policy must enable ready_requires_full_contract")

    max_items_per_lot = policy.get("max_work_items_per_lot")
    valid_max = (
        not isinstance(max_items_per_lot, bool)
        and isinstance(max_items_per_lot, int)
        and max_items_per_lot > 0
    )
    if not valid_max:
        errors.append("max_work_items_per_lot must be a positive integer")
        max_items_per_lot = 6
    assert isinstance(max_items_per_lot, int)

    for key in ("canonical_roadmap", "migration_map"):
        _append_missing_path(
            errors,
            root,
            policy.get(key),
            f"roadmap.policy.{key}",
        )

    statuses = set(roadmap.get("status_values", []))
    milestones = roadmap.get("milestones", {})
    lots = roadmap.get("lots", {})
    if not isinstance(milestones, dict) or not isinstance(lots, dict):
        return ["registry/work-items.yaml has invalid milestone/lot structure"]

    work_items = _load_work_items(root, roadmap, milestones, lots, errors)
    items_by_id: dict[str, dict[str, Any]] = {}
    lot_counts: Counter[str] = Counter()

    for item in work_items:
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id.strip():
            errors.append(f"work item missing string id: {item!r}")
            continue
        if item_id in items_by_id:
            errors.append(f"duplicate work item id: {item_id}")
            continue
        items_by_id[item_id] = item

        status = item.get("status")
        if status not in statuses:
            errors.append(f"{item_id} has invalid status: {status!r}")
        lot_id = item.get("lot")
        if isinstance(lot_id, str):
            lot_counts[lot_id] += 1

        if status in CONTRACT_REQUIRED_STATUSES:
            _validate_executable_contract(errors, item_id, item)

        read_before = item.get("read_before", [])
        if not isinstance(read_before, list):
            errors.append(f"{item_id}.read_before must be a list")
        else:
            for relative in read_before:
                _append_missing_path(
                    errors,
                    root,
                    relative,
                    f"{item_id}.read_before",
                )

    for lot_id in lots:
        count = lot_counts.get(lot_id, 0)
        if count == 0:
            errors.append(f"lot {lot_id} has no work items")
        if count > max_items_per_lot:
            errors.append(
                f"lot {lot_id} has {count} work items, exceeding "
                f"max_work_items_per_lot={max_items_per_lot}"
            )

    dependency_graph: dict[str, list[str]] = {}
    for item_id, item in items_by_id.items():
        dependencies = item.get("depends_on", [])
        if not isinstance(dependencies, list):
            errors.append(f"{item_id}.depends_on must be a list")
            dependencies = []
        dependency_graph[item_id] = []
        for dependency in dependencies:
            if not isinstance(dependency, str):
                errors.append(f"{item_id} has non-string dependency: {dependency!r}")
                continue
            dependency_graph[item_id].append(dependency)
            if dependency == item_id:
                errors.append(f"{item_id} depends on itself")
            elif dependency not in items_by_id:
                errors.append(f"{item_id} references unknown dependency: {dependency}")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(item_id: str) -> None:
        if item_id in visited:
            return
        if item_id in visiting:
            errors.append(f"dependency cycle detected at {item_id}")
            return
        visiting.add(item_id)
        for dependency in dependency_graph.get(item_id, []):
            if dependency in items_by_id:
                visit(dependency)
        visiting.remove(item_id)
        visited.add(item_id)

    for item_id in items_by_id:
        visit(item_id)

    active_id = state.get("active_work_item")
    active_item: dict[str, Any] | None
    if active_id not in items_by_id:
        errors.append(f"PROJECT_STATE active_work_item is unknown: {active_id!r}")
        active_item = None
    else:
        active_item = items_by_id[active_id]
        active_status = active_item.get("status")
        if active_status not in ACTIVE_STATUSES:
            errors.append(
                f"active work item {active_id} must have one of "
                f"{sorted(ACTIVE_STATUSES)}, got {active_status!r}"
            )
        if state.get("status") != active_status:
            errors.append(
                f"PROJECT_STATE status {state.get('status')!r} does not "
                f"match {active_id} status {active_status!r}"
            )
        for dependency in dependency_graph.get(str(active_id), []):
            dependency_item = items_by_id.get(dependency)
            if dependency_item is None:
                continue
            dependency_status = dependency_item.get("status")
            if dependency_status != "done":
                errors.append(
                    f"active work item {active_id} depends on unfinished "
                    f"{dependency} ({dependency_status!r})"
                )

    active_candidates = [
        item_id for item_id, item in items_by_id.items() if item.get("status") in ACTIVE_STATUSES
    ]
    if active_id in items_by_id and active_candidates != [active_id]:
        errors.append(
            "exactly the active work item must be ready/in_progress/in_review; "
            f"found {active_candidates}"
        )

    last_completed = state.get("last_completed", [])
    if not isinstance(last_completed, list):
        errors.append("PROJECT_STATE.last_completed must be a list")
        last_completed = []
    for item_id in last_completed:
        if item_id not in items_by_id:
            errors.append(f"PROJECT_STATE.last_completed contains unknown item: {item_id!r}")
        elif items_by_id[item_id].get("status") != "done":
            errors.append(f"PROJECT_STATE.last_completed item is not done: {item_id}")
    done_ids = {item_id for item_id, item in items_by_id.items() if item.get("status") == "done"}
    if set(last_completed) != done_ids:
        errors.append(
            "PROJECT_STATE.last_completed must exactly match done work items; "
            f"state={sorted(set(last_completed))}, roadmap={sorted(done_ids)}"
        )

    ordered_lots, lot_to_milestone = _ordered_lots(milestones)
    if set(ordered_lots) != set(lots):
        errors.append(
            "milestone lot coverage mismatch; "
            f"milestones={sorted(set(ordered_lots))}, lots={sorted(lots)}"
        )
    if len(ordered_lots) != len(set(ordered_lots)):
        errors.append("a lot is assigned to more than one milestone")

    lot_reviews, milestone_reviews = _validate_reviews(
        errors,
        lots,
        milestones,
        reviews_doc,
    )

    if active_item is not None:
        active_lot_value = active_item.get("lot")
        if not isinstance(active_lot_value, str):
            errors.append(f"active work item has invalid lot: {active_lot_value!r}")
            active_lot = ""
        else:
            active_lot = active_lot_value
        if state.get("lot") != active_lot:
            errors.append(
                f"PROJECT_STATE lot {state.get('lot')!r} does not match active lot {active_lot!r}"
            )
        active_milestone = lot_to_milestone.get(active_lot)
        if state.get("milestone") != active_milestone:
            errors.append(
                f"PROJECT_STATE milestone {state.get('milestone')!r} does "
                f"not match active milestone {active_milestone!r}"
            )

        if active_lot in ordered_lots:
            active_lot_index = ordered_lots.index(active_lot)
            for previous_lot in ordered_lots[:active_lot_index]:
                review = lot_reviews.get(previous_lot, {})
                if not isinstance(review, dict) or review.get("status") != "passed":
                    errors.append(
                        f"cannot advance to {active_lot}: prior lot review "
                        f"{previous_lot} is not passed"
                    )

        milestone_order = list(milestones)
        if active_milestone in milestone_order:
            active_index = milestone_order.index(active_milestone)
            for previous_milestone in milestone_order[:active_index]:
                review = milestone_reviews.get(previous_milestone, {})
                if not isinstance(review, dict) or review.get("status") != "passed":
                    errors.append(
                        f"cannot advance to {active_milestone}: prior milestone "
                        f"review {previous_milestone} is not passed"
                    )

    components = components_doc.get("components", {})
    if not isinstance(components, dict):
        errors.append("registry/components.yaml must define a components mapping")
        components = {}
    component_owner_lots: set[str] = set()
    for component_name, component in components.items():
        if not isinstance(component, dict):
            errors.append(f"component {component_name} must be a mapping")
            continue
        owner_lot = component.get("owner_lot")
        if owner_lot not in lots:
            errors.append(f"component {component_name} has unknown owner_lot: {owner_lot!r}")
        elif isinstance(owner_lot, str):
            component_owner_lots.add(owner_lot)

        component_status = component.get("status")
        if component_status not in {"planned", "implementing", "implemented"}:
            errors.append(f"component {component_name} has invalid status: {component_status!r}")
        if component_status in {"implementing", "implemented"}:
            for field in ("implementation", "tests"):
                paths = component.get(field, [])
                if not isinstance(paths, list):
                    errors.append(f"component {component_name}.{field} must be a list")
                    continue
                for relative in paths:
                    _append_missing_path(
                        errors,
                        root,
                        relative,
                        f"component {component_name}.{field}",
                    )
    if component_owner_lots != set(lots):
        errors.append(
            "component registry must cover every roadmap lot; "
            f"covered={sorted(component_owner_lots)}, lots={sorted(lots)}"
        )

    template = (root / ".github/PULL_REQUEST_TEMPLATE.md").read_text(encoding="utf-8")
    for section in REQUIRED_PR_SECTIONS:
        if section not in template:
            errors.append(f"pull request template missing required section: {section}")

    return errors
