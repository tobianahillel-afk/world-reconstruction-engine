from __future__ import annotations

import hashlib
from collections.abc import Iterable
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


@dataclass(frozen=True, slots=True)
class SceneRelationBatch:
    """One canonical non-empty batch of already-produced scene relationships."""

    relations: tuple[SceneRelationHypothesis, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.relations, tuple):
            raise TypeError("scene_relation_batch.relations must be an immutable tuple")
        if not self.relations:
            raise ValueError("scene_relation_batch.relations must not be empty")
        if any(not isinstance(item, SceneRelationHypothesis) for item in self.relations):
            raise TypeError(
                "scene_relation_batch.relations must contain only SceneRelationHypothesis values"
            )

        relation_keys = tuple(_relation_key(item) for item in self.relations)
        if len(relation_keys) != len(set(relation_keys)):
            raise ValueError("scene_relation_batch.relations must contain unique endpoint pairs")
        if relation_keys != tuple(sorted(relation_keys)):
            raise ValueError(
                "scene_relation_batch.relations must use canonical endpoint-pair order"
            )


def _scene_cluster_id(observation_ids: tuple[ObservationId, ...]) -> SceneClusterId:
    payload = "\n".join(item.value for item in observation_ids).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return SceneClusterId(f"scene:{digest}")


def _validate_batched_observation_ids(
    observation_ids: tuple[ObservationId, ...],
) -> dict[str, ObservationId]:
    if not isinstance(observation_ids, tuple):
        raise TypeError("scene_relation_batches.observation_ids must be an immutable tuple")
    if not observation_ids:
        raise ValueError("scene_relation_batches.observation_ids must not be empty")
    if any(not isinstance(item, ObservationId) for item in observation_ids):
        raise TypeError(
            "scene_relation_batches.observation_ids must contain only ObservationId values"
        )

    observation_values = tuple(item.value for item in observation_ids)
    if len(observation_values) != len(set(observation_values)):
        raise ValueError("scene_relation_batches.observation_ids must not contain duplicates")
    if observation_values != tuple(sorted(observation_values)):
        raise ValueError(
            "scene_relation_batches.observation_ids must use canonical lexical order"
        )
    return {observation_id.value: observation_id for observation_id in observation_ids}


def cluster_verified_scene_relation_batches(
    observation_ids: tuple[ObservationId, ...],
    batches: Iterable[SceneRelationBatch],
) -> tuple[SceneCluster, ...]:
    """Cluster a globally canonical one-pass stream of verified relationship batches."""

    observation_by_value = _validate_batched_observation_ids(observation_ids)
    try:
        batch_iterator = iter(batches)
    except TypeError as exc:
        raise TypeError("scene_relation_batches.batches must be iterable") from exc

    parent = {value: value for value in observation_by_value}
    support_evidence_by_root: dict[str, dict[tuple[str, str], ArtifactRef]] = {}
    contradicted_pairs: list[tuple[str, str]] = []
    previous_key: tuple[str, str] | None = None

    def find(value: str) -> str:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def add_support_evidence(root: str, evidence_refs: tuple[ArtifactRef, ...]) -> None:
        evidence = support_evidence_by_root.setdefault(root, {})
        for ref in evidence_refs:
            evidence.setdefault(_artifact_ref_key(ref), ref)

    def union_supported(
        first: str,
        second: str,
        evidence_refs: tuple[ArtifactRef, ...],
    ) -> None:
        root_first = find(first)
        root_second = find(second)
        if root_first == root_second:
            add_support_evidence(root_first, evidence_refs)
            return

        low, high = sorted((root_first, root_second))
        parent[high] = low
        low_evidence = support_evidence_by_root.setdefault(low, {})
        high_evidence = support_evidence_by_root.pop(high, {})
        for evidence_key, ref in high_evidence.items():
            low_evidence.setdefault(evidence_key, ref)
        for ref in evidence_refs:
            low_evidence.setdefault(_artifact_ref_key(ref), ref)

    for batch in batch_iterator:
        if not isinstance(batch, SceneRelationBatch):
            raise TypeError(
                "scene_relation_batches.batches must contain only SceneRelationBatch values"
            )
        for relation in batch.relations:
            relation_key = _relation_key(relation)
            if previous_key is not None and relation_key <= previous_key:
                raise ValueError(
                    "scene relation batches must use one globally strict canonical "
                    "endpoint-pair order"
                )
            previous_key = relation_key

            first, second = relation_key
            if first not in observation_by_value or second not in observation_by_value:
                raise ValueError(
                    "scene_relation_batches relation endpoints must belong to observation_ids"
                )

            if relation.disposition is SceneRelationDisposition.SUPPORTED:
                union_supported(first, second, relation.evidence_refs)
            elif relation.disposition is SceneRelationDisposition.CONTRADICTED:
                contradicted_pairs.append((first, second))

    for first, second in contradicted_pairs:
        if find(first) == find(second):
            raise SceneClusteringConflictError(
                "supported scene component contains an explicitly contradicted pair: "
                f"{first!r}, {second!r}"
            )

    component_values: dict[str, list[str]] = {}
    for value in observation_by_value:
        root = find(value)
        component_values.setdefault(root, []).append(value)

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


def cluster_verified_scene_relations(
    clustering_input: VerifiedSceneClusteringInput,
) -> tuple[SceneCluster, ...]:
    """Build deterministic supported components and fail closed on internal contradictions."""

    if not isinstance(clustering_input, VerifiedSceneClusteringInput):
        raise TypeError("clustering_input must be VerifiedSceneClusteringInput")

    if clustering_input.relations:
        batches: tuple[SceneRelationBatch, ...] = (
            SceneRelationBatch(relations=clustering_input.relations),
        )
    else:
        batches = ()
    return cluster_verified_scene_relation_batches(
        clustering_input.observation_ids,
        batches,
    )
