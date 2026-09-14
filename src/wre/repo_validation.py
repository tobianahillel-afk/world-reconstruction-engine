from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ACTIVE_STATUSES = {"ready", "in_progress", "in_review"}
REVIEW_STATUSES = {"pending", "in_review", "passed", "failed"}
REQUIRED_PR_SECTIONS = (
    "## Work item",
    "## Objective",
    "## Reuse",
    "## Acceptance criteria",
    "## Test evidence",
    "## Review checklist",
    "## Known limitations / follow-up",
)


def _load_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _append_missing_path(
    errors: list[str], root: Path, path_value: object, context: str
) -> None:
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
        lots = milestone.get("lots", [])
        if not isinstance(lots, list):
            continue
        for lot_id in lots:
            if isinstance(lot_id, str):
                ordered.append(lot_id)
                owner[lot_id] = milestone_id
    return ordered, owner


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

    statuses = set(roadmap.get("status_values", []))
    milestones = roadmap.get("milestones", {})
    lots = roadmap.get("lots", {})
    work_items = roadmap.get("work_items", [])

    if not isinstance(milestones, dict) or not isinstance(lots, dict) or not isinstance(
        work_items, list
    ):
        return ["registry/work-items.yaml has invalid top-level structure"]

    items_by_id: dict[str, dict[str, Any]] = {}
    for item in work_items:
        if not isinstance(item, dict):
            errors.append(f"work item must be a mapping: {item!r}")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str):
            errors.append(f"work item missing string id: {item!r}")
            continue
        if item_id in items_by_id:
            errors.append(f"duplicate work item id: {item_id}")
            continue
        items_by_id[item_id] = item

    dependency_graph: dict[str, list[str]] = {}
    for item_id, item in items_by_id.items():
        status = item.get("status")
        if status not in statuses:
            errors.append(f"{item_id} has invalid status: {status!r}")

        lot_id = item.get("lot")
        if lot_id not in lots:
            errors.append(f"{item_id} references unknown lot: {lot_id!r}")

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

        read_before = item.get("read_before", [])
        if not isinstance(read_before, list):
            errors.append(f"{item_id}.read_before must be a list")
        else:
            for relative in read_before:
                _append_missing_path(errors, root, relative, f"{item_id}.read_before")

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
    if active_id not in items_by_id:
        errors.append(f"PROJECT_STATE active_work_item is unknown: {active_id!r}")
        active_item: dict[str, Any] | None = None
    else:
        active_item = items_by_id[active_id]
        if active_item.get("status") not in ACTIVE_STATUSES:
            errors.append(
                f"active work item {active_id} must have one of {sorted(ACTIVE_STATUSES)}, "
                f"got {active_item.get('status')!r}"
            )
        if state.get("status") != active_item.get("status"):
            errors.append(
                f"PROJECT_STATE status {state.get('status')!r} does not match "
                f"{active_id} status {active_item.get('status')!r}"
            )
        for dependency in dependency_graph.get(active_id, []):
            dependency_item = items_by_id.get(dependency)
            if dependency_item is not None and dependency_item.get("status") != "done":
                errors.append(
                    f"active work item {active_id} depends on unfinished {dependency} "
                    f"({dependency_item.get('status')!r})"
                )

    active_candidates = [
        item_id
        for item_id, item in items_by_id.items()
        if item.get("status") in ACTIVE_STATUSES
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

    lot_reviews = reviews_doc.get("lot_reviews", {})
    milestone_reviews = reviews_doc.get("milestone_reviews", {})
    if not isinstance(lot_reviews, dict) or not isinstance(milestone_reviews, dict):
        errors.append(
            "registry/reviews.yaml must define lot_reviews and milestone_reviews mappings"
        )
        lot_reviews = {}
        milestone_reviews = {}

    if set(lot_reviews) != set(lots):
        errors.append("lot review coverage must exactly match roadmap lots")
    if set(milestone_reviews) != set(milestones):
        errors.append("milestone review coverage must exactly match roadmap milestones")

    for review_kind, review_map in (("lot", lot_reviews), ("milestone", milestone_reviews)):
        for target, review in review_map.items():
            if not isinstance(review, dict):
                errors.append(f"{review_kind} review {target} must be a mapping")
                continue
            if review.get("status") not in REVIEW_STATUSES:
                errors.append(
                    f"{review_kind} review {target} has invalid status: {review.get('status')!r}"
                )
            evidence = review.get("evidence", [])
            if not isinstance(evidence, list):
                errors.append(f"{review_kind} review {target}.evidence must be a list")
            if review.get("status") == "passed" and not evidence:
                errors.append(f"passed {review_kind} review {target} must contain evidence")

    if active_item is not None:
        active_lot = active_item.get("lot")
        if state.get("lot") != active_lot:
            errors.append(
                f"PROJECT_STATE lot {state.get('lot')!r} does not match active lot {active_lot!r}"
            )
        active_milestone = lot_to_milestone.get(active_lot)
        if state.get("milestone") != active_milestone:
            errors.append(
                f"PROJECT_STATE milestone {state.get('milestone')!r} does not match "
                f"active milestone {active_milestone!r}"
            )

        if active_lot in ordered_lots:
            active_lot_index = ordered_lots.index(active_lot)
            for previous_lot in ordered_lots[:active_lot_index]:
                review = lot_reviews.get(previous_lot, {})
                if not isinstance(review, dict) or review.get("status") != "passed":
                    errors.append(
                        f"cannot advance to {active_lot}: prior lot review {previous_lot} "
                        "is not passed"
                    )

        milestone_order = list(milestones)
        if active_milestone in milestone_order:
            active_milestone_index = milestone_order.index(active_milestone)
            for previous_milestone in milestone_order[:active_milestone_index]:
                review = milestone_reviews.get(previous_milestone, {})
                if not isinstance(review, dict) or review.get("status") != "passed":
                    errors.append(
                        f"cannot advance to {active_milestone}: prior milestone review "
                        f"{previous_milestone} is not passed"
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
                        errors, root, relative, f"component {component_name}.{field}"
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
