from typing import Any, cast

from wre.domain import (
    AdapterCapabilityName,
    FailureCategory,
    MetricDirection,
    MetricVector,
    ProvenanceClass,
    QualityDecision,
)
from wre.domain.artifact_materialization import ArtifactMaterializationVerificationStatus

_EXPECTED_DECISIONS = [
    ("PASS", "pass"),
    ("ACCEPT_WITH_WARNINGS", "accept_with_warnings"),
    ("RETRY", "retry"),
    ("ESCALATE", "escalate"),
    ("UNRESOLVED", "unresolved"),
]

_REJECTED_VALUES = [
    "unknown",
    "success",
    "failure",
    "fail",
    "warning",
    "error",
    "quality_gate_failure",
    "colmap_retry",
    "ffmpeg_warning",
    "solver_specific",
]


def _assert_rejected(value: str) -> None:
    try:
        QualityDecision(value)
    except ValueError:
        return
    raise AssertionError(f"QualityDecision unexpectedly accepted {value!r}")


def test_quality_decision_has_exact_closed_member_set() -> None:
    assert [(member.name, member.value) for member in QualityDecision] == _EXPECTED_DECISIONS


def test_quality_decision_constructs_from_stable_wire_tokens() -> None:
    for name, wire_token in _EXPECTED_DECISIONS:
        member = QualityDecision[name]
        assert QualityDecision(wire_token) is member
        assert str(member) == wire_token
        assert hash(member) == hash(QualityDecision(wire_token))


def test_quality_decision_rejects_unknown_failure_warning_and_solver_tokens() -> None:
    for value in _REJECTED_VALUES:
        _assert_rejected(value)


def test_unresolved_is_explicit_not_generic_unknown_coercion() -> None:
    assert QualityDecision("unresolved") is QualityDecision.UNRESOLVED
    _assert_rejected("some_new_unresolved_state")


def test_quality_decision_is_distinct_from_other_domain_vocabularies() -> None:
    assert QualityDecision is not FailureCategory
    assert QualityDecision is not MetricDirection
    assert QualityDecision is not ArtifactMaterializationVerificationStatus
    assert QualityDecision is not ProvenanceClass

    decisions = {member.value for member in QualityDecision}
    failure_tokens = {member.value for member in FailureCategory}
    metric_direction_tokens = {member.value for member in MetricDirection}
    materialization_tokens = {
        member.value for member in ArtifactMaterializationVerificationStatus
    }
    provenance_tokens = {member.value for member in ProvenanceClass}

    assert decisions.isdisjoint(failure_tokens)
    assert decisions.isdisjoint(metric_direction_tokens)
    assert decisions.isdisjoint(materialization_tokens)
    assert decisions.isdisjoint(provenance_tokens)


def test_quality_decision_carries_decision_identity_only() -> None:
    decision = QualityDecision.RETRY

    for attribute in (
        "metric_vector",
        "threshold",
        "policy_revision",
        "reason",
        "escalation_hint",
        "retry_count",
        "route",
        "fallback",
        "timestamp",
        "producer",
        "artifact",
    ):
        assert not hasattr(decision, attribute)


def test_metric_vector_and_adapter_capability_vocabulary_remain_independent() -> None:
    vector = MetricVector(observations=())
    capability = AdapterCapabilityName("quality.pass")

    assert vector.observations == ()
    assert str(capability) == "quality.pass"
    assert not hasattr(vector, "decision")
    assert not hasattr(capability, "decision")
    assert QualityDecision.PASS != cast(Any, capability)
