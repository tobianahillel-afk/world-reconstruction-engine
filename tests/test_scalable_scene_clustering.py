from __future__ import annotations

from collections.abc import Iterator
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.scene_clusters import (
    SceneRelationDisposition,
    SceneRelationHypothesis,
)
from wre.scene_identity import (
    SceneClusteringConflictError,
    SceneRelationBatch,
    VerifiedSceneClusteringInput,
    cluster_verified_scene_relation_batches,
    cluster_verified_scene_relations,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _relation(
    first: str,
    second: str,
    disposition: SceneRelationDisposition,
    artifact_id: str,
) -> SceneRelationHypothesis:
    return SceneRelationHypothesis(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        disposition=disposition,
        evidence_refs=(_ref("scene.relationship", artifact_id),),
    )


def _observations(*values: str) -> tuple[ObservationId, ...]:
    return tuple(ObservationId(value) for value in values)


def _baseline_relations() -> tuple[SceneRelationHypothesis, ...]:
    return (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            "artifact:ab-support",
        ),
        _relation(
            "obs:a",
            "obs:c",
            SceneRelationDisposition.UNRESOLVED,
            "artifact:ac-unresolved",
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:bc-support",
        ),
        _relation(
            "obs:c",
            "obs:d",
            SceneRelationDisposition.CONTRADICTED,
            "artifact:cd-contradiction",
        ),
        _relation(
            "obs:d",
            "obs:e",
            SceneRelationDisposition.SUPPORTED,
            "artifact:de-support",
        ),
    )


def _monolithic(
    observation_ids: tuple[ObservationId, ...],
    relations: tuple[SceneRelationHypothesis, ...],
):
    return cluster_verified_scene_relations(
        VerifiedSceneClusteringInput(
            observation_ids=observation_ids,
            relations=relations,
        )
    )


class _OneShotBatches:
    def __init__(self, batches: tuple[SceneRelationBatch, ...]) -> None:
        self._batches = batches
        self.iteration_count = 0

    def __iter__(self) -> Iterator[SceneRelationBatch]:
        self.iteration_count += 1
        if self.iteration_count != 1:
            raise AssertionError("batch source was iterated more than once")
        yield from self._batches


def test_scene_relation_batch_is_exact_immutable_contract() -> None:
    relation = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ab",
    )
    batch = SceneRelationBatch(relations=(relation,))

    assert tuple(field.name for field in fields(SceneRelationBatch)) == ("relations",)
    with pytest.raises(FrozenInstanceError):
        batch.relations = ()  # type: ignore[misc]

    with pytest.raises(TypeError, match="immutable tuple"):
        SceneRelationBatch(relations=cast(Any, [relation]))
    with pytest.raises(ValueError, match="must not be empty"):
        SceneRelationBatch(relations=())
    with pytest.raises(TypeError, match="SceneRelationHypothesis"):
        SceneRelationBatch(relations=cast(Any, ("not-a-relation",)))


def test_scene_relation_batch_requires_canonical_unique_pairs() -> None:
    ab = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ab",
    )
    ac = _relation(
        "obs:a",
        "obs:c",
        SceneRelationDisposition.UNRESOLVED,
        "artifact:ac",
    )
    ab_other = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.UNRESOLVED,
        "artifact:ab-other",
    )

    with pytest.raises(ValueError, match="unique endpoint pairs"):
        SceneRelationBatch(relations=(ab, ab_other))
    with pytest.raises(ValueError, match="canonical endpoint-pair order"):
        SceneRelationBatch(relations=(ac, ab))


def test_zero_batches_produce_exact_singletons() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c")

    clusters = cluster_verified_scene_relation_batches(observation_ids, ())

    assert clusters == _monolithic(observation_ids, ())
    assert tuple(
        tuple(observation_id.value for observation_id in cluster.observation_ids)
        for cluster in clusters
    ) == (("obs:a",), ("obs:b",), ("obs:c",))
    assert all(cluster.evidence_refs == () for cluster in clusters)


def test_batch_iterable_is_consumed_exactly_once() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c")
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            "artifact:ab",
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:bc",
        ),
    )
    source = _OneShotBatches(
        (
            SceneRelationBatch(relations=(relations[0],)),
            SceneRelationBatch(relations=(relations[1],)),
        )
    )

    clusters = cluster_verified_scene_relation_batches(observation_ids, source)

    assert source.iteration_count == 1
    assert clusters == _monolithic(observation_ids, relations)


def test_valid_batch_partitions_are_byte_for_byte_equivalent_to_monolithic() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c", "obs:d", "obs:e")
    relations = _baseline_relations()
    expected = _monolithic(observation_ids, relations)

    one_batch = cluster_verified_scene_relation_batches(
        observation_ids,
        (SceneRelationBatch(relations=relations),),
    )
    one_edge_per_batch = cluster_verified_scene_relation_batches(
        observation_ids,
        tuple(SceneRelationBatch(relations=(relation,)) for relation in relations),
    )
    mixed_batches = cluster_verified_scene_relation_batches(
        observation_ids,
        (
            SceneRelationBatch(relations=relations[:2]),
            SceneRelationBatch(relations=relations[2:4]),
            SceneRelationBatch(relations=relations[4:]),
        ),
    )

    assert one_batch == expected
    assert one_edge_per_batch == expected
    assert mixed_batches == expected
    assert tuple(cluster.cluster_id for cluster in mixed_batches) == tuple(
        cluster.cluster_id for cluster in expected
    )
    assert tuple(cluster.evidence_refs for cluster in mixed_batches) == tuple(
        cluster.evidence_refs for cluster in expected
    )


def test_cross_batch_duplicate_pair_fails_closed() -> None:
    observation_ids = _observations("obs:a", "obs:b")
    relation = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ab",
    )

    with pytest.raises(ValueError, match="globally strict canonical"):
        cluster_verified_scene_relation_batches(
            observation_ids,
            (
                SceneRelationBatch(relations=(relation,)),
                SceneRelationBatch(relations=(relation,)),
            ),
        )


def test_cross_batch_decreasing_pair_order_fails_closed() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c")
    ab = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ab",
    )
    bc = _relation(
        "obs:b",
        "obs:c",
        SceneRelationDisposition.SUPPORTED,
        "artifact:bc",
    )

    with pytest.raises(ValueError, match="globally strict canonical"):
        cluster_verified_scene_relation_batches(
            observation_ids,
            (
                SceneRelationBatch(relations=(bc,)),
                SceneRelationBatch(relations=(ab,)),
            ),
        )


def test_foreign_endpoint_fails_closed() -> None:
    relation = _relation(
        "obs:a",
        "obs:c",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ac",
    )

    with pytest.raises(ValueError, match="belong to observation_ids"):
        cluster_verified_scene_relation_batches(
            _observations("obs:a", "obs:b"),
            (SceneRelationBatch(relations=(relation,)),),
        )


def test_unresolved_and_separated_contradiction_never_connect_or_add_evidence() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c", "obs:d")
    support_ab = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        "artifact:ab-support",
    )
    unresolved_ac = _relation(
        "obs:a",
        "obs:c",
        SceneRelationDisposition.UNRESOLVED,
        "artifact:ac-unresolved",
    )
    contradiction_bc = _relation(
        "obs:b",
        "obs:c",
        SceneRelationDisposition.CONTRADICTED,
        "artifact:bc-contradiction",
    )
    support_cd = _relation(
        "obs:c",
        "obs:d",
        SceneRelationDisposition.SUPPORTED,
        "artifact:cd-support",
    )
    relations = (support_ab, unresolved_ac, contradiction_bc, support_cd)

    clusters = cluster_verified_scene_relation_batches(
        observation_ids,
        (
            SceneRelationBatch(relations=relations[:2]),
            SceneRelationBatch(relations=relations[2:]),
        ),
    )

    assert clusters == _monolithic(observation_ids, relations)
    assert tuple(
        tuple(observation_id.value for observation_id in cluster.observation_ids)
        for cluster in clusters
    ) == (("obs:a", "obs:b"), ("obs:c", "obs:d"))
    excluded = set(unresolved_ac.evidence_refs + contradiction_bc.evidence_refs)
    assert excluded.isdisjoint(
        ref for cluster in clusters for ref in cluster.evidence_refs
    )


def test_early_contradiction_fails_after_later_support_closes_component() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c")
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.CONTRADICTED,
            "artifact:ab-contradiction",
        ),
        _relation(
            "obs:a",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:ac-support",
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:bc-support",
        ),
    )

    with pytest.raises(
        SceneClusteringConflictError,
        match="explicitly contradicted pair",
    ):
        cluster_verified_scene_relation_batches(
            observation_ids,
            tuple(SceneRelationBatch(relations=(relation,)) for relation in relations),
        )


def test_redundant_supported_edges_keep_every_unique_support_evidence_ref() -> None:
    observation_ids = _observations("obs:a", "obs:b", "obs:c")
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            "artifact:ab",
        ),
        _relation(
            "obs:a",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:ac",
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            "artifact:bc",
        ),
    )

    clusters = cluster_verified_scene_relation_batches(
        observation_ids,
        (
            SceneRelationBatch(relations=relations[:1]),
            SceneRelationBatch(relations=relations[1:]),
        ),
    )

    assert clusters == _monolithic(observation_ids, relations)
    assert clusters[0].evidence_refs == tuple(
        sorted(
            tuple(ref for relation in relations for ref in relation.evidence_refs),
            key=lambda ref: (ref.artifact_kind.value, ref.artifact_id.value),
        )
    )


def test_batch_boundary_validates_observation_universe_and_batch_types() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        cluster_verified_scene_relation_batches(
            cast(Any, [ObservationId("obs:a")]),
            (),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        cluster_verified_scene_relation_batches((), ())
    with pytest.raises(TypeError, match="ObservationId"):
        cluster_verified_scene_relation_batches(cast(Any, ("obs:a",)), ())
    with pytest.raises(ValueError, match="duplicates"):
        cluster_verified_scene_relation_batches(
            _observations("obs:a", "obs:a"),
            (),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        cluster_verified_scene_relation_batches(
            _observations("obs:b", "obs:a"),
            (),
        )
    with pytest.raises(TypeError, match="must be iterable"):
        cluster_verified_scene_relation_batches(
            _observations("obs:a"),
            cast(Any, 123),
        )
    with pytest.raises(TypeError, match="SceneRelationBatch"):
        cluster_verified_scene_relation_batches(
            _observations("obs:a"),
            cast(Any, ("not-a-batch",)),
        )
