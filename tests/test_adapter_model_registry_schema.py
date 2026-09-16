from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA_PATH = _ROOT / "registry" / "schemas" / "adapter-model-registry.schema.json"
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"
_DEPENDENCIES_PATH = _ROOT / "registry" / "dependencies.yaml"


def _schema() -> dict[str, Any]:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def test_empty_registry_envelope_is_versioned_and_contains_no_entries() -> None:
    registry = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))

    assert registry == {"schema_version": 1, "entries": []}


def test_schema_root_is_closed_draft_2020_12() -> None:
    schema = _schema()

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["schema_version", "entries"]
    assert set(schema["properties"]) == {"schema_version", "entries"}
    assert schema["properties"]["schema_version"] == {"const": 1}
    assert schema["properties"]["entries"] == {
        "type": "array",
        "items": {"$ref": "#/$defs/entry"},
    }


def test_entry_schema_is_closed_and_requires_only_v2l3_3_metadata() -> None:
    entry = _schema()["$defs"]["entry"]
    required = {
        "adapter_id",
        "capability",
        "producer",
        "dependency_refs",
        "model",
        "checkpoint",
        "artifact_key_hardware_policy",
        "license",
        "shipping_status",
        "reproducibility_notes",
    }

    assert entry["type"] == "object"
    assert entry["additionalProperties"] is False
    assert set(entry["required"]) == required
    assert set(entry["properties"]) == required


def test_nested_objects_are_closed_and_capability_sets_are_explicit() -> None:
    definitions = _schema()["$defs"]
    for name in (
        "capability",
        "producer",
        "model_identity",
        "checkpoint_identity",
        "license_metadata",
        "entry",
    ):
        assert definitions[name]["additionalProperties"] is False

    capability = definitions["capability"]
    assert capability["required"] == ["name", "input_kinds", "output_kinds"]
    assert capability["properties"]["input_kinds"] == {
        "type": "array",
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/token"},
    }
    assert capability["properties"]["output_kinds"] == {
        "type": "array",
        "minItems": 1,
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/token"},
    }

    dependency_refs = definitions["entry"]["properties"]["dependency_refs"]
    assert dependency_refs == {
        "type": "array",
        "uniqueItems": True,
        "items": {"$ref": "#/$defs/token"},
    }


def test_identity_shipping_hardware_and_license_vocabularies_are_closed() -> None:
    definitions = _schema()["$defs"]
    entry_properties = definitions["entry"]["properties"]

    assert definitions["sha256"] == {
        "type": "string",
        "pattern": "^[0-9a-f]{64}$",
    }
    assert entry_properties["artifact_key_hardware_policy"]["enum"] == [
        "required",
        "omitted",
    ]
    assert entry_properties["shipping_status"]["enum"] == [
        "approved",
        "experimental",
        "benchmark-only",
        "blocked",
    ]
    assert definitions["license_metadata"]["required"] == [
        "direct",
        "review",
        "transitive_notes",
        "redistribution_notes",
    ]
    assert definitions["license_metadata"]["properties"]["review"]["enum"] == [
        "approved",
        "pending",
        "blocked",
    ]


def test_model_checkpoint_are_nullable_explicit_identities_without_defaults() -> None:
    definitions = _schema()["$defs"]
    entry_properties = definitions["entry"]["properties"]

    assert entry_properties["model"] == {
        "oneOf": [
            {"type": "null"},
            {"$ref": "#/$defs/model_identity"},
        ]
    }
    assert entry_properties["checkpoint"] == {
        "oneOf": [
            {"type": "null"},
            {"$ref": "#/$defs/checkpoint_identity"},
        ]
    }
    assert "default" not in json.dumps(definitions, sort_keys=True)
    assert definitions["checkpoint_identity"]["properties"]["sha256"] == {"$ref": "#/$defs/sha256"}


def test_v2l3_4_failure_metric_resume_fields_are_not_in_schema() -> None:
    schema_text = json.dumps(_schema(), sort_keys=True)

    assert '"failure' not in schema_text
    assert '"metric' not in schema_text
    assert '"resume' not in schema_text


def test_dependency_registry_remains_separate_evidence_source() -> None:
    adapter_registry = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    dependency_registry = yaml.safe_load(_DEPENDENCIES_PATH.read_text(encoding="utf-8"))

    assert set(adapter_registry) == {"schema_version", "entries"}
    assert adapter_registry["entries"] == []
    assert "dependencies" in dependency_registry
    assert "colmap" in dependency_registry["dependencies"]
    assert "ffmpeg" in dependency_registry["dependencies"]
    assert "exifread" in dependency_registry["dependencies"]
