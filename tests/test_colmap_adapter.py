from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.colmap_adapter as colmap_adapter
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.reconstruction.colmap_adapter import (
    COLMAP_PRECISION_ADAPTER_ID,
    COLMAP_PRECISION_ARTIFACT_KEY_HARDWARE_POLICY,
    COLMAP_PRECISION_CAPABILITY,
    COLMAP_PRECISION_CAPABILITY_NAME,
    COLMAP_PRECISION_CHECKPOINT,
    COLMAP_PRECISION_COLMAP_VERSION,
    COLMAP_PRECISION_DEPENDENCY_REF,
    COLMAP_PRECISION_INPUT_KIND,
    COLMAP_PRECISION_MODEL,
    COLMAP_PRECISION_OUTPUT_KINDS,
    COLMAP_PRECISION_PRODUCER_IMPLEMENTATION,
    COLMAP_PRECISION_PRODUCER_VERSION,
    COLMAP_PRECISION_PYCOLMAP_VERSION,
    COLMAP_PRECISION_REPRODUCIBILITY_NOTES,
    COLMAP_PRECISION_SHIPPING_STATUS,
    normalize_colmap_precision_inputs,
)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"


def _ref(value: str, kind: str = "image.observation") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(value),
        artifact_kind=ArtifactKind(kind),
    )


def _entries_by_id() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    entries = document["entries"]
    assert isinstance(entries, list)
    return {
        cast(str, entry["adapter_id"]): cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
    }


def test_colmap_precision_descriptor_uses_exact_v2_semantic_kinds() -> None:
    assert COLMAP_PRECISION_ADAPTER_ID == "colmap.precision_geometry"
    assert COLMAP_PRECISION_CAPABILITY_NAME.value == "geometry.precision_sfm"
    assert COLMAP_PRECISION_CAPABILITY.capability is COLMAP_PRECISION_CAPABILITY_NAME
    assert COLMAP_PRECISION_INPUT_KIND == ArtifactKind("image.observation")
    assert COLMAP_PRECISION_CAPABILITY.input_kinds == frozenset({ArtifactKind("image.observation")})
    assert COLMAP_PRECISION_OUTPUT_KINDS == frozenset(
        {
            ArtifactKind("geometry.camera_solution"),
            ArtifactKind("geometry.point_map"),
            ArtifactKind("geometry.solution"),
        }
    )
    assert COLMAP_PRECISION_CAPABILITY.output_kinds is COLMAP_PRECISION_OUTPUT_KINDS
    assert ArtifactKind("geometry.sparse_reconstruction_estimate") not in (
        COLMAP_PRECISION_CAPABILITY.output_kinds
    )


def test_colmap_precision_descriptor_declares_exact_approved_dependency_without_model() -> None:
    assert COLMAP_PRECISION_DEPENDENCY_REF == "colmap"
    assert COLMAP_PRECISION_PYCOLMAP_VERSION == "4.2.0"
    assert COLMAP_PRECISION_COLMAP_VERSION == "COLMAP 4.2.0"
    assert COLMAP_PRECISION_PRODUCER_IMPLEMENTATION == "wre.reconstruction.colmap_adapter"
    assert COLMAP_PRECISION_PRODUCER_VERSION == "1"
    assert COLMAP_PRECISION_MODEL is None
    assert COLMAP_PRECISION_CHECKPOINT is None
    assert COLMAP_PRECISION_ARTIFACT_KEY_HARDWARE_POLICY == "required"
    assert COLMAP_PRECISION_SHIPPING_STATUS == "experimental"
    assert "PyCOLMAP 4.2.0 / COLMAP 4.2.0" in COLMAP_PRECISION_REPRODUCIBILITY_NOTES
    assert "does not execute COLMAP" in COLMAP_PRECISION_REPRODUCIBILITY_NOTES


def test_normalize_colmap_precision_inputs_retains_exact_canonical_tuple() -> None:
    inputs = (_ref("image:a"), _ref("image:b"))

    result = normalize_colmap_precision_inputs(inputs)

    assert result is inputs


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (cast(Any, [_ref("image:a")]), TypeError, "immutable tuple"),
        ((), ValueError, "at least one artifact"),
        (cast(Any, ("not-an-artifact",)), TypeError, "only ArtifactRef"),
        ((_ref("image:a", "video.observation"),), ValueError, "image.observation"),
        ((_ref("image:a"), _ref("image:a")), ValueError, "duplicate"),
        ((_ref("image:b"), _ref("image:a")), ValueError, "canonical ArtifactRef order"),
    ],
)
def test_normalize_colmap_precision_inputs_fails_closed(
    inputs: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        normalize_colmap_precision_inputs(inputs)


def test_registry_v2_colmap_entry_matches_pure_module_contract() -> None:
    entries = _entries_by_id()
    entry = entries[COLMAP_PRECISION_ADAPTER_ID]

    assert entry["capability"] == {
        "name": COLMAP_PRECISION_CAPABILITY_NAME.value,
        "input_kinds": ["image.observation"],
        "output_kinds": [
            "geometry.camera_solution",
            "geometry.point_map",
            "geometry.solution",
        ],
    }
    assert entry["producer"] == {
        "implementation": COLMAP_PRECISION_PRODUCER_IMPLEMENTATION,
        "version": COLMAP_PRECISION_PRODUCER_VERSION,
        "revision": None,
    }
    assert entry["dependency_refs"] == [COLMAP_PRECISION_DEPENDENCY_REF]
    assert entry["model"] is COLMAP_PRECISION_MODEL
    assert entry["checkpoint"] is COLMAP_PRECISION_CHECKPOINT
    assert entry["artifact_key_hardware_policy"] == COLMAP_PRECISION_ARTIFACT_KEY_HARDWARE_POLICY
    assert entry["shipping_status"] == COLMAP_PRECISION_SHIPPING_STATUS
    assert entry["reproducibility_notes"] == COLMAP_PRECISION_REPRODUCIBILITY_NOTES
    assert entry["failure_signals"] == []
    assert entry["metric_names"] == []
    assert entry["resume_mode"] == "unsupported"


def test_registry_preserves_legacy_colmap_sparse_sfm_as_distinct_donor() -> None:
    entries = _entries_by_id()
    donor = entries["colmap.sparse_sfm"]
    v2 = entries[COLMAP_PRECISION_ADAPTER_ID]

    assert donor["capability"] == {
        "name": "geometry.sparse_sfm",
        "input_kinds": ["image.observation"],
        "output_kinds": ["geometry.sparse_reconstruction_estimate"],
    }
    assert donor["producer"] == {
        "implementation": "wre.colmap_reconstruction_importer",
        "version": "2",
        "revision": None,
    }
    assert donor["dependency_refs"] == ["colmap"]
    assert donor["model"] is None
    assert donor["checkpoint"] is None
    assert donor["artifact_key_hardware_policy"] == "required"
    assert donor["shipping_status"] == "approved"
    assert donor["adapter_id"] != v2["adapter_id"]
    assert donor["capability"] != v2["capability"]


def test_colmap_precision_descriptor_module_has_no_execution_or_io_surface() -> None:
    forbidden_names = {
        "Path",
        "pycolmap",
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "open",
        "extract_colmap_features",
        "match_colmap_pairs",
        "verify_colmap_geometry",
        "reconstruct_colmap_incrementally",
        "import_colmap_reconstruction",
        "inspect_colmap_environment",
        "CameraSolution",
        "PointMap",
        "GeometrySolution",
    }

    assert forbidden_names.isdisjoint(vars(colmap_adapter))
