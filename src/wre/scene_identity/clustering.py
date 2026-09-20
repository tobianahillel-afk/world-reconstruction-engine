from __future__ import annotations

import hashlib
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.scene_clusters import (
    SceneCluster,
    SceneClusterId,
    SceneRelationDisposition,
    SceneRelationHypothesis,
)


class SceneClusteringConflictError(RuntimeError):
    """Raised when supported connectivity conflicts with explicit contradiction evidence."""


def _relation_key(relation: SceneRelationHypothesis) -> tuple[str, str]:
    return (relation.observation_id1.value, relation.observation_id2.value)


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


@dataclass(frozen=True, slots=True)
class VerifiedSceneClusteringInput:
    """Explicit scene-membership universe plus canonical pairwise relationship evidence."""

    observation_ids: tuple[ObservationId, ...]
    relations: tuple[SceneRelationHypothesis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_ids, tuple):
            raise TypeError("verified_scene_clustering.observation_ids must be an immutable tuple")
        if not self.observation_ids:
            raise ValueError("verified_scene_clustering.observation_ids must not be empty")
        if any(not isinstance(item, ObservationId) for item in self.observation_ids):
            raise TypeError(
                "verified_scene_clustering.observation_ids must contain only ObservationId values"
            )

        observation_values = tuple(item.value for item in self.observation_ids)
        if len(observation_values) != len(set(observation_values)):
            raise ValueError(
                "verified_scene_clustering.observation_ids must not contain duplicates"
            )
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError(
                "verified_scene_clustering.observation_ids must use canonical lexical order"
            )

        if not isinstance(self.relations, tuple):
            raise TypeError("verified_scene_clustering.relations must be an immutable tuple")
        if any(not isinstance(item, SceneRelationHypothesis) for item in self.relations):
            raise TypeError(
                "verified_scene_clustering.relations must contain only "
                "SceneRelationHypothesis values"
            )

        relation_keys = tuple(_relation_key(item) for item in self.relations)
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError(
                "verified_scene_clustering.relations must contain unique endpoint pairs"
            )
        if relation_keys != tuple(sorted(relation_keys)):
            raise ValueError(
                "verified_scene_clustering.relations must use canonical endpoint-pair order"
            )

        membership = set(observation_values)
        if any(
            relation.observation_id1.value not in membership
            or relation.observation_id2.value not in membership
            for relation in self.relations
        ):
            raise ValueError(
                "verified_scene_clustering relation endpoints must belong to observation_ids"
            )


def _scene_cluster_id(observation_ids: tuple[ObservationId, ...]) -> SceneClusterId:
    payload = "\n".join(item.value for item in observation_ids).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return SceneClusterId(f"scene:{digest}")


def cluster_verified_scene_relations(
    clustering_input: VerifiedSceneClusteringInput,
) -> tuple[SceneCluster, ...]:
    """Build deterministic supported components and fail closed on internal contradictions."""

    if not isinstance(clustering_input, VerifiedSceneClusteringInput):
        raise TypeError("clustering_input must be VerifiedSceneClusteringInput")

    observation_by_value = {
        observation_id.value: observation_id for observation_id in clustering_input.observation_ids
    }
    parent = {value: value for value in observation_by_value}

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(first: str, second: str) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            return
        low, high = sorted((root_first, root_second))
        parent[high] = low

    for relation in clustering_input.relations:
        if relation.disposition is SceneRelationDisposition.SUPPORTED:
            union(relation.observation_id1.value, relation.observation_id2.value)

    component_values: dict[str, list[str]] = {}
    for value in observation_by_value:
        root = find(value)
        component_values.setdefault(root, []).append(value)

    for relation in clustering_input.relations:
        if relation.disposition is not SceneRelationDisposition.CONTRADICTED:
            continue
        if find(relation.observation_id1.value) == find(relation.observation_id2.value):
            raise SceneClusteringConflictError(
                "supported scene component contains an explicitly contradicted pair: "
                f"{relation.observation_id1.value!r}, {relation.observation_id2.value!r}"
            )

    support_evidence_by_root: dict[str, dict[tuple[str, str], ArtifactRef]] = {}
    for relation in clustering_input.relations:
        if relation.disposition is not SceneRelationDisposition.SUPPORTED:
            continue
        root = find(relation.observation_id1.value)
        evidence = support_evidence_by_root.setdefault(root, {})
        for ref in relation.evidence_refs:
            evidence.setdefault(_artifact_ref_key(ref), ref)

    memberships = tuple(
        tuple(observation_by_value[value] for value in sorted(values))
        for _, values in sorted(
            component_values.items(),
            key=lambda item: tuple(sorted(item[1])),
        )
    )

    clusters: list[SceneCluster] = []
    for membership in memberships:
        if len(membership) == 1:
            evidence_refs: tuple[ArtifactRef, ...] = ()
        else:
            root = find(membership[0].value)
            evidence_by_key = support_evidence_by_root.get(root, {})
            evidence_refs = tuple(evidence_by_key[key] for key in sorted(evidence_by_key))
        clusters.append(
            SceneCluster(
                cluster_id=_scene_cluster_id(membership),
                observation_ids=membership,
                evidence_refs=evidence_refs,
            )
        )
    return tuple(clusters)
