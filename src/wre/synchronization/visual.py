from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.temporal_groups import SyncHypothesis, SyncHypothesisDisposition


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


def _validate_evidence_refs(evidence_refs: object) -> None:
    if not isinstance(evidence_refs, tuple):
        raise TypeError("visual_event_sync.evidence_refs must be an immutable tuple")
    if not evidence_refs:
        raise ValueError("visual_event_sync.evidence_refs must not be empty")
    if any(not isinstance(ref, ArtifactRef) for ref in evidence_refs):
        raise TypeError("visual_event_sync.evidence_refs must contain only ArtifactRef values")

    refs = evidence_refs
    keys = tuple(_artifact_ref_key(ref) for ref in refs)
    if len(keys) != len(set(keys)):
        raise ValueError("visual_event_sync.evidence_refs must not contain duplicates")
    if keys != tuple(sorted(keys)):
        raise ValueError("visual_event_sync.evidence_refs must use canonical artifact order")


@dataclass(frozen=True, slots=True)
class VisualEventSyncPolicy:
    """Explicit acceptance policy for already-produced visual-event evidence."""

    minimum_support_count: int
    maximum_residual_us: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_support_count, bool)
            or not isinstance(self.minimum_support_count, int)
            or self.minimum_support_count <= 0
        ):
            raise ValueError(
                "visual_event_sync_policy.minimum_support_count must be a positive integer"
            )
        if (
            isinstance(self.maximum_residual_us, bool)
            or not isinstance(self.maximum_residual_us, int)
            or self.maximum_residual_us < 0
        ):
            raise ValueError(
                "visual_event_sync_policy.maximum_residual_us must be a non-negative integer"
            )


class VisualEventCandidateMultiplicity(StrEnum):
    """Whether visual evidence identifies one offset or remains ambiguous."""

    UNIQUE = "unique"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class VisualEventSyncCandidate:
    """Solver-private pairwise visual-event alignment evidence."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    candidate_offset_us: int
    residual_us: int
    support_count: int
    multiplicity: VisualEventCandidateMultiplicity
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("visual_event_sync.observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("visual_event_sync.observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "visual_event_sync observation IDs must be distinct and canonically ordered"
            )
        if isinstance(self.candidate_offset_us, bool) or not isinstance(
            self.candidate_offset_us,
            int,
        ):
            raise TypeError("visual_event_sync.candidate_offset_us must be int")
        if (
            isinstance(self.residual_us, bool)
            or not isinstance(self.residual_us, int)
            or self.residual_us < 0
        ):
            raise ValueError("visual_event_sync.residual_us must be a non-negative integer")
        if (
            isinstance(self.support_count, bool)
            or not isinstance(self.support_count, int)
            or self.support_count <= 0
        ):
            raise ValueError("visual_event_sync.support_count must be a positive integer")
        if not isinstance(self.multiplicity, VisualEventCandidateMultiplicity):
            raise TypeError(
                "visual_event_sync.multiplicity must be VisualEventCandidateMultiplicity"
            )

        _validate_evidence_refs(self.evidence_refs)


def adapt_visual_event_sync(
    candidate: VisualEventSyncCandidate,
    policy: VisualEventSyncPolicy,
) -> SyncHypothesis:
    """Map one explicit visual-event candidate into one pairwise stable hypothesis."""

    if not isinstance(candidate, VisualEventSyncCandidate):
        raise TypeError("candidate must be VisualEventSyncCandidate")
    if not isinstance(policy, VisualEventSyncPolicy):
        raise TypeError("policy must be VisualEventSyncPolicy")

    accepted = (
        candidate.multiplicity is VisualEventCandidateMultiplicity.UNIQUE
        and candidate.support_count >= policy.minimum_support_count
        and candidate.residual_us <= policy.maximum_residual_us
    )
    if accepted:
        disposition = SyncHypothesisDisposition.SUPPORTED
        offset_us: int | None = candidate.candidate_offset_us
    else:
        disposition = SyncHypothesisDisposition.UNRESOLVED
        offset_us = None

    return SyncHypothesis(
        observation_id1=candidate.observation_id1,
        observation_id2=candidate.observation_id2,
        disposition=disposition,
        offset_us=offset_us,
        evidence_refs=candidate.evidence_refs,
    )
