from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId

_SCENE_CLUSTER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


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


class SceneRelationDisposition(StrEnum):
    """Evidence-scoped pairwise scene relationship state."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SceneRelationHypothesis:
    """One evidence-backed pairwise relationship assessment, not absolute scene truth."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    disposition: SceneRelationDisposition
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("scene_relation_hypothesis.observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("scene_relation_hypothesis.observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "scene_relation_hypothesis observation IDs must be distinct and canonically ordered"
            )
        if not isinstance(self.disposition, SceneRelationDisposition):
            raise TypeError(
                "scene_relation_hypothesis.disposition must be SceneRelationDisposition"
            )
        _validate_evidence_refs(
            self.evidence_refs,
            context="scene_relation_hypothesis.evidence_refs",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True, order=True)
class SceneClusterId:
    """Opaque stable identity for one hypothesized connected-scene component."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("scene_cluster_id.value must be str")
        if _SCENE_CLUSTER_ID_RE.fullmatch(self.value) is None:
            raise ValueError(
                "scene_cluster_id must be 1-128 lowercase characters using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SceneCluster:
    """Immutable hypothesized connected-scene membership and exact supporting evidence."""

    cluster_id: SceneClusterId
    observation_ids: tuple[ObservationId, ...]
    evidence_refs: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cluster_id, SceneClusterId):
            raise TypeError("scene_cluster.cluster_id must be SceneClusterId")
        if not isinstance(self.observation_ids, tuple):
            raise TypeError("scene_cluster.observation_ids must be an immutable tuple")
        if not self.observation_ids:
            raise ValueError("scene_cluster.observation_ids must not be empty")
        if any(not isinstance(item, ObservationId) for item in self.observation_ids):
            raise TypeError("scene_cluster.observation_ids must contain only ObservationId values")

        observation_values = tuple(item.value for item in self.observation_ids)
        if len(observation_values) != len(set(observation_values)):
            raise ValueError("scene_cluster.observation_ids must not contain duplicates")
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError("scene_cluster.observation_ids must use canonical lexical order")

        _validate_evidence_refs(
            self.evidence_refs,
            context="scene_cluster.evidence_refs",
            allow_empty=True,
        )
        if len(self.observation_ids) > 1 and not self.evidence_refs:
            raise ValueError(
                "multi-observation scene_cluster requires explicit supporting evidence_refs"
            )
