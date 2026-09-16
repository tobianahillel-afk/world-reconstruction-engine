from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, cast

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


def _require_exact_mapping(
    errors: list[str],
    value: object,
    context: str,
    expected_fields: set[str],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        errors.append(f"{context} must be a mapping")
        return None
    raw_mapping = cast(dict[object, Any], value)
    if not all(isinstance(key, str) for key in raw_mapping):
        errors.append(f"{context} field names must be strings")
        return None
    mapping = cast(dict[str, Any], value)
    actual_fields = set(mapping)
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    if missing:
        errors.append(f"{context} missing fields: {missing}")
    if extra:
        errors.append(f"{context} has undeclared fields: {extra}")
    return mapping


def _schema_object_fields(
    errors: list[str],
    definitions: dict[str, Any],
    name: str,
) -> set[str] | None:
    definition = definitions.get(name)
    if not isinstance(definition, dict):
        errors.append(f"adapter model registry schema missing object definition: {name}")
        return None
    required = definition.get("required")
    properties = definition.get("properties")
    if (
        not isinstance(required, list)
        or not all(isinstance(field, str) for field in required)
        or not isinstance(properties, dict)
    ):
        errors.append(f"adapter model registry schema has invalid object definition: {name}")
        return None
    required_fields = set(cast(list[str], required))
    property_fields = set(cast(dict[str, Any], properties))
    if required_fields != property_fields:
        errors.append(
            f"adapter model registry schema {name} required/properties fields must match"
        )
        return None
    return required_fields


def _schema_enum(
    errors: list[str],
    schema_node: object,
    context: str,
) -> set[str] | None:
    if not isinstance(schema_node, dict):
        errors.append(f"adapter model registry schema missing enum: {context}")
        return None
    values = schema_node.get("enum")
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        errors.append(f"adapter model registry schema has invalid enum: {context}")
        return None
    return set(cast(list[str], values))


def _validate_token(
    errors: list[str],
    value: object,
    context: str,
    pattern: re.Pattern[str],
    max_length: int,
) -> str | None:
    if not isinstance(value, str) or len(value) > max_length or pattern.fullmatch(value) is None:
        errors.append(f"{context} must be a valid registry token")
        return None
    return value


def _validate_nullable_non_blank(errors: list[str], value: object, context: str) -> None:
    if value is not None:
        _require_non_blank_text(errors, value, context)


def _validate_canonical_token_list(
    errors: list[str],
    value: object,
    context: str,
    pattern: re.Pattern[str],
    max_length: int,
    *,
    non_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{context} must be a list")
        return []
    if non_empty and not value:
        errors.append(f"{context} must not be empty")

    tokens: list[str] = []
    valid_collection = True
    for index, entry in enumerate(value):
        token = _validate_token(errors, entry, f"{context}[{index}]", pattern, max_length)
        if token is None:
            valid_collection = False
        else:
            tokens.append(token)
    if valid_collection:
        if len(tokens) != len(set(tokens)):
            errors.append(f"{context} must not contain duplicates")
        if tokens != sorted(tokens):
            errors.append(f"{context} must be in canonical lexicographic order")
    return tokens


def _validate_adapter_model_registry(errors: list[str], root: Path) -> None:
    registry_path = root / "registry/adapter-models.yaml"
    dependencies_path = root / "registry/dependencies.yaml"
    schema_path = root / "registry/schemas/adapter-model-registry.schema.json"

    try:
        registry_data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot parse registry/adapter-models.yaml: {exc}")
        return
    if not isinstance(registry_data, dict):
        errors.append("registry/adapter-models.yaml must contain a YAML mapping")
        return
    registry = cast(dict[str, Any], registry_data)

    try:
        dependencies_doc = _load_mapping(dependencies_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        errors.append(f"cannot parse registry/dependencies.yaml: {exc}")
        return
    try:
        schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot parse adapter model registry schema: {exc}")
        return
    if not isinstance(schema_data, dict):
        errors.append("adapter model registry schema must contain a JSON object")
        return
    schema = cast(dict[str, Any], schema_data)

    root_fields = schema.get("required")
    root_properties = schema.get("properties")
    definitions_value = schema.get("$defs")
    if (
        not isinstance(root_fields, list)
        or not all(isinstance(field, str) for field in root_fields)
        or not isinstance(root_properties, dict)
        or not isinstance(definitions_value, dict)
    ):
        errors.append("adapter model registry schema has invalid root contract")
        return
    expected_root_fields = set(cast(list[str], root_fields))
    if expected_root_fields != set(cast(dict[str, Any], root_properties)):
        errors.append("adapter model registry schema root required/properties fields must match")
        return
    _require_exact_mapping(errors, registry, "registry/adapter-models.yaml", expected_root_fields)

    schema_version = registry.get("schema_version")
    if isinstance(schema_version, bool) or not isinstance(schema_version, int) or schema_version != 1:
        errors.append("registry/adapter-models.yaml schema_version must be integer 1")

    entries_value = registry.get("entries")
    if not isinstance(entries_value, list):
        errors.append("registry/adapter-models.yaml entries must be a list")
        return

    definitions = cast(dict[str, Any], definitions_value)
    entry_fields = _schema_object_fields(errors, definitions, "entry")
    capability_fields = _schema_object_fields(errors, definitions, "capability")
    producer_fields = _schema_object_fields(errors, definitions, "producer")
    model_fields = _schema_object_fields(errors, definitions, "model_identity")
    checkpoint_fields = _schema_object_fields(errors, definitions, "checkpoint_identity")
    license_fields = _schema_object_fields(errors, definitions, "license_metadata")
    object_field_sets = (
        entry_fields,
        capability_fields,
        producer_fields,
        model_fields,
        checkpoint_fields,
        license_fields,
    )
    if any(field_set is None for field_set in object_field_sets):
        return
    assert entry_fields is not None
    assert capability_fields is not None
    assert producer_fields is not None
    assert model_fields is not None
    assert checkpoint_fields is not None
    assert license_fields is not None

    token_schema = definitions.get("token")
    sha_schema = definitions.get("sha256")
    if not isinstance(token_schema, dict) or not isinstance(sha_schema, dict):
        errors.append("adapter model registry schema is missing token or sha256 definitions")
        return
    token_pattern_value = token_schema.get("pattern")
    token_max_length = token_schema.get("maxLength")
    sha_pattern_value = sha_schema.get("pattern")
    if (
        not isinstance(token_pattern_value, str)
        or isinstance(token_max_length, bool)
        or not isinstance(token_max_length, int)
        or token_max_length <= 0
        or not isinstance(sha_pattern_value, str)
    ):
        errors.append("adapter model registry schema has invalid token or sha256 constraints")
        return
    try:
        token_pattern = re.compile(token_pattern_value)
        sha_pattern = re.compile(sha_pattern_value)
    except re.error as exc:
        errors.append(f"adapter model registry schema has invalid regex: {exc}")
        return

    entry_definition = cast(dict[str, Any], definitions["entry"])
    entry_properties = cast(dict[str, Any], entry_definition["properties"])
    license_definition = cast(dict[str, Any], definitions["license_metadata"])
    license_properties = cast(dict[str, Any], license_definition["properties"])
    hardware_values = _schema_enum(
        errors,
        entry_properties.get("artifact_key_hardware_policy"),
        "entry.artifact_key_hardware_policy",
    )
    shipping_values = _schema_enum(
        errors,
        entry_properties.get("shipping_status"),
        "entry.shipping_status",
    )
    resume_values = _schema_enum(errors, entry_properties.get("resume_mode"), "entry.resume_mode")
    license_review_values = _schema_enum(
        errors,
        license_properties.get("review"),
        "license_metadata.review",
    )
    enum_sets = (hardware_values, shipping_values, resume_values, license_review_values)
    if any(enum_set is None for enum_set in enum_sets):
        return
    assert hardware_values is not None
    assert shipping_values is not None
    assert resume_values is not None
    assert license_review_values is not None

    dependencies_value = dependencies_doc.get("dependencies")
    if not isinstance(dependencies_value, dict):
        errors.append("registry/dependencies.yaml must define a dependencies mapping")
        dependencies: dict[str, Any] = {}
    else:
        dependencies = cast(dict[str, Any], dependencies_value)

    adapter_ids: list[str] = []
    for index, entry_value in enumerate(entries_value):
        context = f"registry/adapter-models.yaml entries[{index}]"
        entry = _require_exact_mapping(errors, entry_value, context, entry_fields)
        if entry is None:
            continue

        adapter_id = _validate_token(
            errors,
            entry.get("adapter_id"),
            f"{context}.adapter_id",
            token_pattern,
            token_max_length,
        )
        if adapter_id is not None:
            adapter_ids.append(adapter_id)

        capability = _require_exact_mapping(
            errors,
            entry.get("capability"),
            f"{context}.capability",
            capability_fields,
        )
        if capability is not None:
            _validate_token(
                errors,
                capability.get("name"),
                f"{context}.capability.name",
                token_pattern,
                token_max_length,
            )
            _validate_canonical_token_list(
                errors,
                capability.get("input_kinds"),
                f"{context}.capability.input_kinds",
                token_pattern,
                token_max_length,
            )
            _validate_canonical_token_list(
                errors,
                capability.get("output_kinds"),
                f"{context}.capability.output_kinds",
                token_pattern,
                token_max_length,
                non_empty=True,
            )

        producer = _require_exact_mapping(
            errors,
            entry.get("producer"),
            f"{context}.producer",
            producer_fields,
        )
        if producer is not None:
            _require_non_blank_text(
                errors,
                producer.get("implementation"),
                f"{context}.producer.implementation",
            )
            producer_version = producer.get("version")
            _require_non_blank_text(errors, producer_version, f"{context}.producer.version")
            if isinstance(producer_version, str) and producer_version.strip().lower() == "latest":
                errors.append(f"{context}.producer.version cannot use floating latest")
            _validate_nullable_non_blank(
                errors,
                producer.get("revision"),
                f"{context}.producer.revision",
            )

        dependency_refs = _validate_canonical_token_list(
            errors,
            entry.get("dependency_refs"),
            f"{context}.dependency_refs",
            token_pattern,
            token_max_length,
        )
        for dependency_ref in dependency_refs:
            if dependency_ref not in dependencies:
                errors.append(f"{context}.dependency_refs references unknown dependency: {dependency_ref}")

        model_value = entry.get("model")
        if model_value is not None:
            model = _require_exact_mapping(
                errors,
                model_value,
                f"{context}.model",
                model_fields,
            )
            if model is not None:
                _require_non_blank_text(errors, model.get("name"), f"{context}.model.name")
                model_version = model.get("version")
                _require_non_blank_text(errors, model_version, f"{context}.model.version")
                if isinstance(model_version, str) and model_version.strip().lower() == "latest":
                    errors.append(f"{context}.model.version cannot use floating latest")
                _validate_nullable_non_blank(
                    errors,
                    model.get("revision"),
                    f"{context}.model.revision",
                )

        checkpoint_value = entry.get("checkpoint")
        if checkpoint_value is not None:
            checkpoint = _require_exact_mapping(
                errors,
                checkpoint_value,
                f"{context}.checkpoint",
                checkpoint_fields,
            )
            if model_value is None:
                errors.append(f"{context}.checkpoint requires model")
            if checkpoint is not None:
                _require_non_blank_text(
                    errors,
                    checkpoint.get("identifier"),
                    f"{context}.checkpoint.identifier",
                )
                checkpoint_sha = checkpoint.get("sha256")
                if not isinstance(checkpoint_sha, str) or sha_pattern.fullmatch(checkpoint_sha) is None:
                    errors.append(f"{context}.checkpoint.sha256 must be exactly 64 lowercase hex")

        hardware_policy = entry.get("artifact_key_hardware_policy")
        if hardware_policy not in hardware_values:
            errors.append(
                f"{context}.artifact_key_hardware_policy has invalid value: {hardware_policy!r}"
            )

        license_metadata = _require_exact_mapping(
            errors,
            entry.get("license"),
            f"{context}.license",
            license_fields,
        )
        license_review: object = None
        if license_metadata is not None:
            _require_non_blank_text(
                errors,
                license_metadata.get("direct"),
                f"{context}.license.direct",
            )
            license_review = license_metadata.get("review")
            if license_review not in license_review_values:
                errors.append(f"{context}.license.review has invalid value: {license_review!r}")
            _require_non_blank_text(
                errors,
                license_metadata.get("transitive_notes"),
                f"{context}.license.transitive_notes",
            )
            _require_non_blank_text(
                errors,
                license_metadata.get("redistribution_notes"),
                f"{context}.license.redistribution_notes",
            )

        shipping_status = entry.get("shipping_status")
        if shipping_status not in shipping_values:
            errors.append(f"{context}.shipping_status has invalid value: {shipping_status!r}")

        _require_non_blank_text(
            errors,
            entry.get("reproducibility_notes"),
            f"{context}.reproducibility_notes",
        )
        _validate_canonical_token_list(
            errors,
            entry.get("failure_signals"),
            f"{context}.failure_signals",
            token_pattern,
            token_max_length,
        )
        _validate_canonical_token_list(
            errors,
            entry.get("metric_names"),
            f"{context}.metric_names",
            token_pattern,
            token_max_length,
        )
        resume_mode = entry.get("resume_mode")
        if resume_mode not in resume_values:
            errors.append(f"{context}.resume_mode has invalid value: {resume_mode!r}")

        if shipping_status == "approved":
            if license_review != "approved":
                errors.append(f"{context} approved shipping requires approved entry license review")
            for dependency_ref in dependency_refs:
                dependency = dependencies.get(dependency_ref)
                dependency_review = (
                    dependency.get("license_review") if isinstance(dependency, dict) else None
                )
                if dependency_review != "approved":
                    errors.append(
                        f"{context} approved shipping requires approved dependency license review: "
                        f"{dependency_ref}"
                    )
        if license_review == "blocked" and shipping_status != "blocked":
            errors.append(f"{context} blocked entry license review requires blocked shipping_status")

    if len(adapter_ids) != len(set(adapter_ids)):
        errors.append("registry/adapter-models.yaml adapter_id values must be unique")
    if adapter_ids != sorted(adapter_ids):
        errors.append(
            "registry/adapter-models.yaml entries must be in canonical lexicographic adapter_id order"
        )


def validate_repository(root: Path) -> list[str]:
    root = root.resolve()
    errors: list[str] = []

    required_files = (
        "PROJECT_STATE.yaml",
        "registry/work-items.yaml",
        "registry/components.yaml",
        "registry/reviews.yaml",
        "registry/adapter-models.yaml",
        "registry/dependencies.yaml",
        "registry/schemas/adapter-model-registry.schema.json",
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

    _validate_adapter_model_registry(errors, root)

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

    for item_id, item in items_by_id.items():
        if item.get("status") != "done":
            continue
        for dependency in dependency_graph.get(item_id, []):
            dependency_item = items_by_id.get(dependency)
            if dependency_item is None:
                continue
            dependency_status = dependency_item.get("status")
            if dependency_status != "done":
                errors.append(
                    f"done work item {item_id} depends on unfinished "
                    f"{dependency} ({dependency_status!r})"
                )

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
