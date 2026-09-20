from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.scene_clusters import SceneClusterId

_TEMPORAL_GROUP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


def _validate_evidence_refs(
    evidence_refs: object,
    *,
    context: str,
    allow_empty: bool,
) -> None:
    if not isinstance(evidence_refs, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    if not allow_empty and not evidence_refs:
        raise ValueError(f"{context} must not be empty")
    if any(not isinstance(ref, ArtifactRef) for ref in evidence_refs):
        raise TypeError(f"{context} must contain only ArtifactRef values")

    refs = evidence_refs
    keys = tuple(_artifact_ref_key(ref) for ref in refs)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} must not contain duplicates")
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{context} must use canonical artifact order")


@dataclass(frozen=True, slots=True, order=True)
class TemporalGroupId:
    """Opaque stable identity for one event-time synchronization context."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("temporal_group_id.value must be str")
        if _TEMPORAL_GROUP_ID_RE.fullmatch(self.value) is None:
            raise ValueError(
                "temporal_group_id must be 1-128 lowercase characters using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


class SyncHypothesisDisposition(StrEnum):
    """Evidence-scoped relative-time synchronization state."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SyncHypothesis:
    """Pairwise relative event-time alignment hypothesis with exact evidence."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    disposition: SyncHypothesisDisposition
    offset_us: int | None
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("sync_hypothesis.observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("sync_hypothesis.observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "sync_hypothesis observation IDs must be distinct and canonically ordered"
            )
        if not isinstance(self.disposition, SyncHypothesisDisposition):
            raise TypeError(
                "sync_hypothesis.disposition must be SyncHypothesisDisposition"
            )

        if self.disposition is SyncHypothesisDisposition.SUPPORTED:
            if isinstance(self.offset_us, bool) or not isinstance(self.offset_us, int):
                raise TypeError(
                    "supported sync_hypothesis.offset_us must be an integer microsecond offset"
                )
        elif self.offset_us is not None:
            raise ValueError(
                "contradicted or unresolved sync_hypothesis must not expose offset_us"
            )

        _validate_evidence_refs(
            self.evidence_refs,
            context="sync_hypothesis.evidence_refs",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class TemporalGroup:
    """Immutable event-time synchronization context within one scene cluster."""

    group_id: TemporalGroupId
    scene_cluster_id: SceneClusterId
    observation_ids: tuple[ObservationId, ...]
    evidence_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.group_id, TemporalGroupId):
            raise TypeError("temporal_group.group_id must be TemporalGroupId")
        if not isinstance(self.scene_cluster_id, SceneClusterId):
            raise TypeError("temporal_group.scene_cluster_id must be SceneClusterId")
        if not isinstance(self.observation_ids, tuple):
            raise TypeError("temporal_group.observation_ids must be an immutable tuple")
        if not self.observation_ids:
            raise ValueError("temporal_group.observation_ids must not be empty")
        if any(not isinstance(item, ObservationId) for item in self.observation_ids):
            raise TypeError(
                "temporal_group.observation_ids must contain only ObservationId values"
            )

        observation_values = tuple(item.value for item in self.observation_ids)
        if len(observation_values) != len(set(observation_values)):
            raise ValueError("temporal_group.observation_ids must not contain duplicates")
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError(
                "temporal_group.observation_ids must use canonical lexical order"
            )

        _validate_evidence_refs(
            self.evidence_refs,
            context="temporal_group.evidence_refs",
            allow_empty=True,
        )
        if len(self.observation_ids) > 1 and not self.evidence_refs:
            raise ValueError(
                "multi-observation temporal_group requires explicit supporting evidence_refs"
            )
