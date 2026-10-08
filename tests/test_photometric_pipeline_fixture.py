from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "synthetic"
    / "photometric-pipeline-controlled"
    / "fixture.json"
)
_RUNNER_PATH = Path(__file__).parents[1] / "scripts" / "run_photometric_pipeline_fixture.py"
_WORKFLOW_PATH = (
    Path(__file__).parents[1] / ".github" / "workflows" / "photometric-pipeline-controlled.yml"
)
_FIXTURE_SHA256 = "a532be3128f254bb9aa212cce6d2de48b14fc6ea2d5ca6a9aa58500b1e6a5308"

_FORBIDDEN_EVIDENCE_KEYS = {
    "appearancemodel",
    "default",
    "inferred_factor",
    "qualitydecision",
    "rank",
    "recovered_color",
    "recovered_highlight",
    "repair",
    "route",
    "score",
    "shipping_promotion",
    "winner",
}


def _load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("wre_v2l17_5_fixture_runner", _RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load V2L17.5 fixture runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fixture_document() -> dict[str, Any]:
    value = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_fixture(path: Path, document: dict[str, Any], *, canonical: bool = True) -> None:
    path.write_text(
        json.dumps(document, indent=2, sort_keys=canonical) + "\n",
        encoding="utf-8",
    )


def _case(document: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(item for item in document["cases"] if item["case_id"] == case_id)


def _record(evidence: dict[str, Any], case_id: str) -> dict[str, Any]:
    return next(item for item in evidence["cases"] if item["case_id"] == case_id)


def _assert_forbidden_keys_absent(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            assert normalized not in _FORBIDDEN_EVIDENCE_KEYS
            _assert_forbidden_keys_absent(child)
    elif isinstance(value, list):
        for child in value:
            _assert_forbidden_keys_absent(child)


def test_fixture_sha_schema_and_ready_endpoint_overrange_evidence() -> None:
    raw = _FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _FIXTURE_SHA256

    runner = _load_runner()
    evidence = runner.run_fixture(_FIXTURE_PATH)
    assert evidence["fixture_sha256"] == _FIXTURE_SHA256
    assert evidence["fixture_id"] == "fixture.photometric-pipeline-controlled.v1"
    assert evidence["schema_version"] == 1
    assert [item["case_id"] for item in evidence["cases"]] == [
        "foreign-normalization",
        "ready-endpoints-overrange",
        "rejected-explicit-factors",
        "unresolved-missing-factors",
        "unsupported-color",
    ]

    ready = _record(evidence, "ready-endpoints-overrange")
    assert ready["color"]["status"] == "ready"
    assert ready["normalization"]["status"] == "ready"
    assert ready["compatibility"]["status"] == "compatible"
    assert ready["compatibility"]["reasons"] == []

    assert ready["normalization"]["normalized_values"] == pytest.approx(
        [0.0, 1.0, 4.0, 0.10253891674808648, 1.0, 0.863442000455597],
        abs=1e-15,
    )
    assert ready["normalization"]["normalized_values"][1] == 1.0
    assert ready["normalization"]["normalized_values"][4] == 1.0
    assert ready["normalization"]["normalized_values"][2] > 1.0

    validity = ready["validity"]
    assert validity["source_low_endpoint_candidate"]["packed_mask"] == [1, 0, 0, 0, 0, 0]
    assert validity["source_high_endpoint_candidate"]["packed_mask"] == [0, 1, 1, 0, 1, 0]
    assert validity["normalized_above_one"]["packed_mask"] == [0, 0, 1, 0, 0, 0]
    assert validity["source_low_endpoint_candidate"]["kind"] == "source_low_endpoint_candidate"
    assert validity["source_high_endpoint_candidate"]["kind"] == "source_high_endpoint_candidate"
    assert "proven" not in validity["source_low_endpoint_candidate"]["kind"]
    assert "sensor" not in validity["source_high_endpoint_candidate"]["kind"]


def test_fail_closed_unresolved_rejected_foreign_and_unsupported_cases() -> None:
    evidence = _load_runner().run_fixture(_FIXTURE_PATH)

    unresolved = _record(evidence, "unresolved-missing-factors")
    assert unresolved["source"]["source_photometry_statuses"] == {
        "color": "resolved",
        "exposure": "resolved",
        "white_balance": "resolved",
    }
    assert unresolved["normalization"]["status"] == "unresolved"
    assert unresolved["normalization"]["plan_identity"] is None
    assert unresolved["normalization"]["normalized_content_sha256"] is None
    assert unresolved["validity"] is None
    assert unresolved["compatibility"]["status"] == "unresolved"
    assert any(
        "photometric normalization is unresolved" in reason
        for reason in unresolved["compatibility"]["reasons"]
    )

    rejected = _record(evidence, "rejected-explicit-factors")
    assert rejected["normalization"]["status"] == "rejected"
    assert rejected["normalization"]["plan_identity"] is None
    assert rejected["validity"] is None
    assert rejected["compatibility"]["status"] == "incompatible"
    assert rejected["compatibility"]["reasons"] == ["photometric normalization is rejected"]

    foreign = _record(evidence, "foreign-normalization")
    assert foreign["normalization"]["status"] == "ready"
    assert foreign["compatibility"]["status"] == "incompatible"
    assert foreign["compatibility"]["color_observation_id"] == "obs:v2l17-5:foreign"
    assert foreign["compatibility"]["normalization_observation_id"] == "obs:v2l17-5:ready"
    assert any("different observations" in reason for reason in foreign["compatibility"]["reasons"])

    unsupported = _record(evidence, "unsupported-color")
    assert unsupported["color"]["status"] == "rejected"
    assert unsupported["normalization"] is None
    assert unsupported["validity"] is None
    assert unsupported["compatibility"]["status"] == "incompatible"
    assert any(
        "color conversion is rejected" in reason
        for reason in unsupported["compatibility"]["reasons"]
    )


def test_runner_rejects_wrong_missing_extra_and_noncanonical_case_fields(
    tmp_path: Path,
) -> None:
    runner = _load_runner()

    missing = _fixture_document()
    del missing["cases"][0]["decoded_encoding"]
    missing_path = tmp_path / "missing.json"
    _write_fixture(missing_path, missing)
    with pytest.raises(ValueError, match="fields mismatch"):
        runner.run_fixture(missing_path)

    extra = _fixture_document()
    extra["cases"][0]["unexpected"] = "not-allowed"
    extra_path = tmp_path / "extra.json"
    _write_fixture(extra_path, extra)
    with pytest.raises(ValueError, match="fields mismatch"):
        runner.run_fixture(extra_path)

    wrong = _fixture_document()
    wrong["cases"][0]["rgb8"] = "32,64,96"
    wrong_path = tmp_path / "wrong.json"
    _write_fixture(wrong_path, wrong)
    with pytest.raises(ValueError, match="must be a list"):
        runner.run_fixture(wrong_path)

    noncanonical = _fixture_document()
    first = noncanonical["cases"][0]
    first["case_id"] = first.pop("case_id")
    noncanonical_path = tmp_path / "noncanonical.json"
    _write_fixture(noncanonical_path, noncanonical, canonical=False)
    with pytest.raises(ValueError, match="canonical"):
        runner.run_fixture(noncanonical_path)


def test_source_raw_factors_and_mask_changes_move_downstream_identities() -> None:
    runner = _load_runner()
    document = _fixture_document()
    original_case = copy.deepcopy(_case(document, "ready-endpoints-overrange"))
    original_before = copy.deepcopy(original_case)
    original = runner._build_stage(original_case)
    assert original_case == original_before
    assert original.converted is not None
    assert original.normalized is not None
    assert original.validity_masks is not None
    assert original.normalization_assessment is not None
    assert original.normalization_assessment.plan is not None

    changed_pixels_case = copy.deepcopy(original_case)
    changed_pixels_case["rgb8"][3] = 65
    changed_pixels = runner._build_stage(changed_pixels_case)
    assert changed_pixels.color_assessment.plan is not None
    assert changed_pixels.color_assessment.plan.identity != original.color_assessment.plan.identity
    assert changed_pixels.converted is not None
    assert changed_pixels.converted.content_sha256 != original.converted.content_sha256
    assert changed_pixels.validity_masks is not None
    assert changed_pixels.validity_masks.identity != original.validity_masks.identity

    changed_raw_case = copy.deepcopy(original_case)
    iso_raw = next(
        item
        for item in changed_raw_case["raw_metadata"]
        if item["namespace"] == "exif" and item["key"] == "iso"
    )
    iso_evidence = next(
        item
        for item in changed_raw_case["source_exposure"]["evidence"]
        if item["namespace"] == "exif" and item["key"] == "iso"
    )
    iso_raw["value"] = "101"
    iso_evidence["value"] = "101"
    changed_raw_case["source_exposure"]["iso_speed"] = 101.0
    changed_raw = runner._build_stage(changed_raw_case)
    assert changed_raw.color_assessment.plan is not None
    assert changed_raw.color_assessment.plan.identity != original.color_assessment.plan.identity

    changed_factors_case = copy.deepcopy(original_case)
    changed_factors_case["normalization_factors"]["exposure_adjustment_ev"] = 0.0
    changed_factors = runner._build_stage(changed_factors_case)
    assert changed_factors.normalization_assessment is not None
    assert changed_factors.normalization_assessment.plan is not None
    assert (
        changed_factors.normalization_assessment.plan.identity
        != original.normalization_assessment.plan.identity
    )
    assert changed_factors.normalized is not None
    assert changed_factors.normalized.derived_sha256 != original.normalized.derived_sha256

    changed_mask_case = copy.deepcopy(original_case)
    changed_mask_case["rgb8"][0] = 1
    changed_mask = runner._build_stage(changed_mask_case)
    assert changed_mask.validity_masks is not None
    assert (
        changed_mask.validity_masks.source_low_endpoint_candidate.packed_mask
        != original.validity_masks.source_low_endpoint_candidate.packed_mask
    )
    assert (
        changed_mask.validity_masks.source_low_endpoint_candidate.identity
        != original.validity_masks.source_low_endpoint_candidate.identity
    )


def test_fixture_execution_preserves_immutable_inputs_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    fixture_before = _FIXTURE_PATH.read_bytes()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = runner.write_evidence(_FIXTURE_PATH, first_path)
    second = runner.write_evidence(_FIXTURE_PATH, second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert _FIXTURE_PATH.read_bytes() == fixture_before

    ready_case = copy.deepcopy(_case(_fixture_document(), "ready-endpoints-overrange"))
    stage = runner._build_stage(ready_case)
    with pytest.raises(FrozenInstanceError):
        stage.source_photometry.color = stage.source_photometry.color  # type: ignore[misc]
    assert stage.converted is not None
    with pytest.raises(FrozenInstanceError):
        stage.converted.packed_rgb_f64_be = b""  # type: ignore[misc]
    assert stage.normalized is not None
    with pytest.raises(FrozenInstanceError):
        stage.normalized.packed_rgb_f64_be = b""  # type: ignore[misc]


def test_retained_json_keeps_separate_statuses_identities_masks_and_no_policy_fields() -> None:
    evidence = _load_runner().run_fixture(_FIXTURE_PATH)
    _assert_forbidden_keys_absent(evidence)

    for item in evidence["cases"]:
        assert set(item) == {
            "case_id",
            "color",
            "compatibility",
            "normalization",
            "source",
            "validity",
        }
        assert item["source"]["observation_id"].startswith("obs:v2l17-5:")
        assert len(item["source"]["decoded_level"]["sha256"]) == 64
        assert len(item["source"]["decoded_artifact_key"]) == 64
        assert len(item["compatibility"]["identity"]) == 64
        if item["color"]["plan_identity"] is not None:
            assert len(item["color"]["plan_identity"]) == 64
        if item["normalization"] is not None and item["normalization"]["plan_identity"] is not None:
            assert len(item["normalization"]["plan_identity"]) == 64
        if item["validity"] is not None:
            assert len(item["validity"]["identity"]) == 64
            assert (
                item["validity"]["source_low_endpoint_candidate"]["identity"]
                != item["validity"]["source_high_endpoint_candidate"]["identity"]
            )
            assert (
                item["validity"]["normalized_above_one"]["identity"]
                != item["validity"]["source_high_endpoint_candidate"]["identity"]
            )


def test_runner_has_no_external_image_network_model_solver_or_accelerator_execution() -> None:
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= {
        "__future__",
        "argparse",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "struct",
        "typing",
        "wre",
    }
    lowered = source.lower()
    for forbidden in (
        "import numpy",
        "import cv2",
        "import pil",
        "import torch",
        "import colour",
        "littlecms",
        "import subprocess",
        "import socket",
        "import requests",
        "import urllib",
        "huggingface",
        "checkpoint",
        "cuda",
        "nvidia",
        "decode_image",
        "auto_exposure",
        "auto_white_balance",
        "tone_map",
        "recover_highlights",
    ):
        assert forbidden not in lowered


def test_workflow_is_cpu_only_and_retains_canonical_fixture_evidence() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in workflow
    assert "scripts/run_photometric_pipeline_fixture.py" in workflow
    assert "tests/test_photometric_pipeline_fixture.py" in workflow
    assert "v2l17-5-photometric-pipeline-controlled" in workflow
    assert "photometric-pipeline-controlled.json" in workflow

    lowered = workflow.lower()
    for forbidden in (
        "cuda",
        "nvidia",
        "model.safetensors",
        "torch.hub",
        "huggingface",
        "open3d",
        "opencv",
        "pillow",
        "colour-science",
    ):
        assert forbidden not in lowered


def test_lifecycle_remains_v2l17_5_until_retained_evidence_and_review_exist() -> None:
    root = Path(__file__).parents[1]
    state = (root / "PROJECT_STATE.yaml").read_text(encoding="utf-8")
    reviews = (root / "registry" / "reviews.yaml").read_text(encoding="utf-8")
    work_items = (root / "registry" / "work-items" / "v2m3.yaml").read_text(encoding="utf-8")

    assert "active_work_item: V2L17.5" in state
    assert "V2L17: {status: pending, reviewed_at: null, evidence: []}" in reviews
    assert "static_appearance:" not in state
    assert "id: V2L17.5" in work_items
    assert "status: ready" in work_items
