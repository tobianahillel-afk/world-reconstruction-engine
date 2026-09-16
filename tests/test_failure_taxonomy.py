from __future__ import annotations

import pytest

from wre.domain import ArtifactMaterializationVerificationStatus, FailureCategory, ProvenanceClass


_EXPECTED_FAILURES = [
    ("UNSUPPORTED_INPUT", "unsupported_input"),
    ("DEPENDENCY_UNAVAILABLE", "dependency_unavailable"),
    ("INSUFFICIENT_OVERLAP", "insufficient_overlap"),
    ("CAMERA_AMBIGUITY", "camera_ambiguity"),
    ("CALIBRATION_FAILURE", "calibration_failure"),
    ("GEOMETRIC_INCONSISTENCY", "geometric_inconsistency"),
    ("DYNAMIC_CONTAMINATION", "dynamic_contamination"),
    ("DEPTH_INCONSISTENCY", "depth_inconsistency"),
    ("MEMORY_EXHAUSTION", "memory_exhaustion"),
    ("TIMEOUT", "timeout"),
    ("CORRUPTED_MEDIA", "corrupted_media"),
    ("CHECKPOINT_INCOMPATIBILITY", "checkpoint_incompatibility"),
    ("QUALITY_GATE_FAILURE", "quality_gate_failure"),
    ("UNKNOWN_INTERNAL_ERROR", "unknown_internal_error"),
]


def test_failure_category_has_exact_closed_member_set() -> None:
    assert [(member.name, member.value) for member in FailureCategory] == _EXPECTED_FAILURES


def test_failure_category_constructs_from_stable_wire_tokens() -> None:
    for name, wire_token in _EXPECTED_FAILURES:
        member = FailureCategory[name]
        assert FailureCategory(wire_token) is member
        assert str(member) == wire_token
        assert hash(member) == hash(FailureCategory(wire_token))


@pytest.mark.parametrize(
    "value",
    [
        "unknown",
        "success",
        "pass",
        "accept_with_warnings",
        "retry",
        "escalate",
        "unresolved",
        "warning",
        "colmap_no_model",
        "ffmpeg_decode_error",
        "solver_specific",
    ],
)
def test_failure_category_rejects_unknown_success_decision_warning_and_solver_tokens(
    value: str,
) -> None:
    with pytest.raises(ValueError):
        FailureCategory(value)


def test_failure_category_is_distinct_from_other_closed_domain_vocabularies() -> None:
    failure_tokens = {member.value for member in FailureCategory}
    materialization_tokens = {
        member.value for member in ArtifactMaterializationVerificationStatus
    }
    provenance_tokens = {member.value for member in ProvenanceClass}

    assert FailureCategory is not ArtifactMaterializationVerificationStatus
    assert FailureCategory is not ProvenanceClass
    assert failure_tokens.isdisjoint(materialization_tokens)
    assert failure_tokens.isdisjoint(provenance_tokens)


def test_unknown_internal_error_is_explicit_not_a_generic_unknown_coercion() -> None:
    assert FailureCategory("unknown_internal_error") is FailureCategory.UNKNOWN_INTERNAL_ERROR
    with pytest.raises(ValueError):
        FailureCategory("some_new_internal_error")
