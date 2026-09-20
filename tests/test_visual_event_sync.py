from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.synchronization.visual as visual_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.temporal_groups import SyncHypothesisDisposition
from wre.synchronization import (
    VisualEventCandidateMultiplicity,
    VisualEventSyncCandidate,
    VisualEventSyncPolicy,
    adapt_visual_event_sync,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _policy(
    *,
    minimum_support_count: int = 3,
    maximum_residual_us: int = 2_000,
) -> VisualEventSyncPolicy:
    return VisualEventSyncPolicy(
        minimum_support_count=minimum_support_count,
        maximum_residual_us=maximum_residual_us,
    )


def _candidate(
    *,
    candidate_offset_us: int = 10_000,
    residual_us: int = 1_000,
    support_count: int = 3,
    multiplicity: VisualEventCandidateMultiplicity = VisualEventCandidateMultiplicity.UNIQUE,
    evidence_refs: tuple[ArtifactRef, ...] | None = None,
) -> VisualEventSyncCandidate:
    return VisualEventSyncCandidate(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        candidate_offset_us=candidate_offset_us,
        residual_us=residual_us,
        support_count=support_count,
        multiplicity=multiplicity,
        evidence_refs=evidence_refs
        if evidence_refs is not None
        else (_ref("sync.visual_event", "artifact:visual-event"),),
    )


def test_visual_event_policy_is_exact_immutable_contract() -> None:
    policy = _policy()

    assert tuple(field.name for field in fields(VisualEventSyncPolicy)) == (
        "minimum_support_count",
        "maximum_residual_us",
    )
    with pytest.raises(FrozenInstanceError):
        policy.minimum_support_count = 4  # type: ignore[misc]

    with pytest.raises(ValueError, match="positive integer"):
        _policy(minimum_support_count=0)
    with pytest.raises(ValueError, match="positive integer"):
        _policy(minimum_support_count=cast(Any, True))
    with pytest.raises(ValueError, match="non-negative integer"):
        _policy(maximum_residual_us=-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        _policy(maximum_residual_us=cast(Any, True))


def test_visual_event_candidate_multiplicity_vocabulary_is_exact() -> None:
    assert tuple(item.value for item in VisualEventCandidateMultiplicity) == (
        "unique",
        "ambiguous",
    )


def test_visual_event_candidate_is_exact_immutable_contract() -> None:
    candidate = _candidate()

    assert tuple(field.name for field in fields(VisualEventSyncCandidate)) == (
        "observation_id1",
        "observation_id2",
        "candidate_offset_us",
        "residual_us",
        "support_count",
        "multiplicity",
        "evidence_refs",
    )
    with pytest.raises(FrozenInstanceError):
        candidate.support_count = 4  # type: ignore[misc]


def test_visual_event_candidate_validates_endpoints_numeric_fields_and_multiplicity() -> None:
    ref = _ref("sync.visual_event", "artifact:visual-event")

    with pytest.raises(TypeError, match="observation_id1"):
        VisualEventSyncCandidate(
            observation_id1=cast(Any, "obs:a"),
            observation_id2=ObservationId("obs:b"),
            candidate_offset_us=0,
            residual_us=0,
            support_count=1,
            multiplicity=VisualEventCandidateMultiplicity.UNIQUE,
            evidence_refs=(ref,),
        )
    with pytest.raises(TypeError, match="observation_id2"):
        VisualEventSyncCandidate(
            observation_id1=ObservationId("obs:a"),
            observation_id2=cast(Any, "obs:b"),
            candidate_offset_us=0,
            residual_us=0,
            support_count=1,
            multiplicity=VisualEventCandidateMultiplicity.UNIQUE,
            evidence_refs=(ref,),
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        VisualEventSyncCandidate(
            observation_id1=ObservationId("obs:b"),
            observation_id2=ObservationId("obs:a"),
            candidate_offset_us=0,
            residual_us=0,
            support_count=1,
            multiplicity=VisualEventCandidateMultiplicity.UNIQUE,
            evidence_refs=(ref,),
        )
    with pytest.raises(TypeError, match="candidate_offset_us"):
        _candidate(candidate_offset_us=cast(Any, True))
    with pytest.raises(ValueError, match="residual_us"):
        _candidate(residual_us=-1)
    with pytest.raises(ValueError, match="residual_us"):
        _candidate(residual_us=cast(Any, True))
    with pytest.raises(ValueError, match="support_count"):
        _candidate(support_count=0)
    with pytest.raises(ValueError, match="support_count"):
        _candidate(support_count=cast(Any, True))
    with pytest.raises(TypeError, match="VisualEventCandidateMultiplicity"):
        _candidate(multiplicity=cast(Any, "unique"))


def test_visual_event_candidate_requires_canonical_non_empty_evidence() -> None:
    ref_a = _ref("sync.a", "artifact:a")
    ref_b = _ref("sync.b", "artifact:b")

    assert _candidate(evidence_refs=(ref_a, ref_b)).evidence_refs == (ref_a, ref_b)

    with pytest.raises(TypeError, match="immutable tuple"):
        _candidate(evidence_refs=cast(Any, [ref_a]))
    with pytest.raises(ValueError, match="must not be empty"):
        _candidate(evidence_refs=())
    with pytest.raises(TypeError, match="ArtifactRef"):
        _candidate(evidence_refs=cast(Any, ("bad",)))
    with pytest.raises(ValueError, match="duplicates"):
        _candidate(evidence_refs=(ref_a, ref_a))
    with pytest.raises(ValueError, match="canonical artifact order"):
        _candidate(evidence_refs=(ref_b, ref_a))


@pytest.mark.parametrize(
    "offset_us",
    [-250_000, 0, 750_000],
)
def test_accepted_visual_event_offsets_preserve_exact_sign_convention(offset_us: int) -> None:
    result = adapt_visual_event_sync(
        _candidate(candidate_offset_us=offset_us),
        _policy(),
    )

    assert result.disposition is SyncHypothesisDisposition.SUPPORTED
    assert result.offset_us == offset_us


def test_visual_event_support_threshold_is_exact() -> None:
    policy = _policy(minimum_support_count=3)

    accepted = adapt_visual_event_sync(_candidate(support_count=3), policy)
    rejected = adapt_visual_event_sync(_candidate(support_count=2), policy)

    assert accepted.disposition is SyncHypothesisDisposition.SUPPORTED
    assert accepted.offset_us == 10_000
    assert rejected.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert rejected.offset_us is None


def test_visual_event_residual_boundary_is_exact() -> None:
    policy = _policy(maximum_residual_us=2_000)

    accepted = adapt_visual_event_sync(_candidate(residual_us=2_000), policy)
    rejected = adapt_visual_event_sync(_candidate(residual_us=2_001), policy)

    assert accepted.disposition is SyncHypothesisDisposition.SUPPORTED
    assert accepted.offset_us == 10_000
    assert rejected.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert rejected.offset_us is None


def test_ambiguous_visual_event_candidate_is_always_unresolved() -> None:
    result = adapt_visual_event_sync(
        _candidate(
            support_count=100,
            residual_us=0,
            multiplicity=VisualEventCandidateMultiplicity.AMBIGUOUS,
        ),
        _policy(minimum_support_count=1, maximum_residual_us=0),
    )

    assert result.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert result.offset_us is None


def test_visual_event_adapter_preserves_every_evidence_ref() -> None:
    refs = (
        _ref("sync.a", "artifact:a"),
        _ref("sync.b", "artifact:b"),
    )

    result = adapt_visual_event_sync(_candidate(evidence_refs=refs), _policy())

    assert result.evidence_refs == refs


def test_visual_event_adapter_never_emits_contradicted() -> None:
    candidates = (
        _candidate(),
        _candidate(support_count=1),
        _candidate(residual_us=10_000),
        _candidate(multiplicity=VisualEventCandidateMultiplicity.AMBIGUOUS),
    )

    results = tuple(adapt_visual_event_sync(candidate, _policy()) for candidate in candidates)

    assert all(
        result.disposition is not SyncHypothesisDisposition.CONTRADICTED
        for result in results
    )


def test_visual_event_adapter_rejects_wrong_input_types() -> None:
    with pytest.raises(TypeError, match="VisualEventSyncCandidate"):
        adapt_visual_event_sync(cast(Any, "bad"), _policy())
    with pytest.raises(TypeError, match="VisualEventSyncPolicy"):
        adapt_visual_event_sync(_candidate(), cast(Any, "bad"))


def test_visual_event_interface_has_no_execution_fusion_or_persistence_surface() -> None:
    forbidden = {
        "Path",
        "subprocess",
        "ffmpeg",
        "pycolmap",
        "requests",
        "socket",
        "torch",
        "numpy",
        "TemporalGroup",
        "SQLiteLocalStore",
        "VideoObservation",
    }
    assert forbidden.isdisjoint(visual_module.__dict__)
