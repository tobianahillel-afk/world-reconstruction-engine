from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.scene_identity.clustering as clustering_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.scene_clusters import (
    SceneCluster,
    SceneRelationDisposition,
    SceneRelationHypothesis,
)
from wre.scene_identity import (
    SceneClusteringConflictError,
    VerifiedSceneClusteringInput,
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
    *evidence_refs: ArtifactRef,
) -> SceneRelationHypothesis:
    return SceneRelationHypothesis(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        disposition=disposition,
        evidence_refs=evidence_refs or (_ref("scene.relationship", f"artifact:{first}-{second}"),),
    )


def _input(
    observation_ids: tuple[str, ...],
    relations: tuple[SceneRelationHypothesis, ...] = (),
) -> VerifiedSceneClusteringInput:
    return VerifiedSceneClusteringInput(
        observation_ids=tuple(ObservationId(value) for value in observation_ids),
        relations=relations,
    )


def test_clustering_input_is_exact_immutable_contract() -> None:
    clustering_input = _input(("obs:a", "obs:b"))

    assert tuple(field.name for field in fields(VerifiedSceneClusteringInput)) == (
        "observation_ids",
        "relations",
    )
    with pytest.raises(FrozenInstanceError):
        clustering_input.relations = ()  # type: ignore[misc]


def test_clustering_input_validates_observation_membership_exactly() -> None:
    relation = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
    )

    with pytest.raises(TypeError, match="immutable tuple"):
        VerifiedSceneClusteringInput(
            observation_ids=cast(Any, [ObservationId("obs:a")]),
            relations=(),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        VerifiedSceneClusteringInput(observation_ids=(), relations=())
    with pytest.raises(TypeError, match="ObservationId"):
        VerifiedSceneClusteringInput(
            observation_ids=cast(Any, ("obs:a",)),
            relations=(),
        )
    with pytest.raises(ValueError, match="duplicates"):
        _input(("obs:a", "obs:a"))
    with pytest.raises(ValueError, match="canonical lexical order"):
        _input(("obs:b", "obs:a"))
    with pytest.raises(ValueError, match="belong to observation_ids"):
        _input(("obs:a", "obs:c"), (relation,))


def test_clustering_input_validates_relation_tuple_order_and_pair_uniqueness() -> None:
    ab_supported = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
    )
    ab_unresolved = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.UNRESOLVED,
        _ref("scene.relationship", "artifact:ab-unresolved"),
    )
    ac = _relation(
        "obs:a",
        "obs:c",
        SceneRelationDisposition.SUPPORTED,
    )

    with pytest.raises(TypeError, match="immutable tuple"):
        VerifiedSceneClusteringInput(
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            relations=cast(Any, [ab_supported]),
        )
    with pytest.raises(TypeError, match="SceneRelationHypothesis"):
        VerifiedSceneClusteringInput(
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            relations=cast(Any, ("not-a-relation",)),
        )
    with pytest.raises(ValueError, match="unique endpoint pairs"):
        _input(("obs:a", "obs:b"), (ab_supported, ab_unresolved))
    with pytest.raises(ValueError, match="canonical endpoint-pair order"):
        _input(("obs:a", "obs:b", "obs:c"), (ac, ab_supported))


def test_empty_relations_produce_deterministic_singletons() -> None:
    clusters = cluster_verified_scene_relations(_input(("obs:a", "obs:b", "obs:c")))

    assert tuple(tuple(item.value for item in cluster.observation_ids) for cluster in clusters) == (
        ("obs:a",),
        ("obs:b",),
        ("obs:c",),
    )
    assert all(cluster.evidence_refs == () for cluster in clusters)


def test_supported_edge_produces_multi_observation_cluster_with_exact_evidence() -> None:
    evidence = _ref("scene.support", "artifact:ab")
    relation = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        evidence,
    )

    clusters = cluster_verified_scene_relations(_input(("obs:a", "obs:b"), (relation,)))

    assert len(clusters) == 1
    assert clusters[0].observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )
    assert clusters[0].evidence_refs == (evidence,)


def test_supported_chain_builds_one_component_and_unions_all_support_evidence() -> None:
    evidence_ab = _ref("scene.support", "artifact:ab")
    evidence_bc = _ref("scene.support", "artifact:bc")
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            evidence_ab,
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            evidence_bc,
        ),
    )

    clusters = cluster_verified_scene_relations(_input(("obs:a", "obs:b", "obs:c"), relations))

    assert len(clusters) == 1
    assert tuple(item.value for item in clusters[0].observation_ids) == (
        "obs:a",
        "obs:b",
        "obs:c",
    )
    assert clusters[0].evidence_refs == tuple(
        sorted(
            (evidence_ab, evidence_bc),
            key=lambda ref: (ref.artifact_kind.value, ref.artifact_id.value),
        )
    )


def test_unresolved_relation_neither_merges_nor_contributes_evidence() -> None:
    unresolved_ref = _ref("scene.unresolved", "artifact:ab")
    relation = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.UNRESOLVED,
        unresolved_ref,
    )

    clusters = cluster_verified_scene_relations(_input(("obs:a", "obs:b"), (relation,)))

    assert tuple(tuple(item.value for item in cluster.observation_ids) for cluster in clusters) == (
        ("obs:a",),
        ("obs:b",),
    )
    assert all(unresolved_ref not in cluster.evidence_refs for cluster in clusters)


def test_separated_contradiction_does_not_merge_or_contribute_evidence() -> None:
    evidence_ab = _ref("scene.support", "artifact:ab")
    evidence_cd = _ref("scene.support", "artifact:cd")
    contradiction = _ref("scene.contradiction", "artifact:bc")
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            evidence_ab,
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.CONTRADICTED,
            contradiction,
        ),
        _relation(
            "obs:c",
            "obs:d",
            SceneRelationDisposition.SUPPORTED,
            evidence_cd,
        ),
    )
    clustering_input = _input(("obs:a", "obs:b", "obs:c", "obs:d"), relations)

    clusters = cluster_verified_scene_relations(clustering_input)

    assert tuple(tuple(item.value for item in cluster.observation_ids) for cluster in clusters) == (
        ("obs:a", "obs:b"),
        ("obs:c", "obs:d"),
    )
    assert contradiction not in clusters[0].evidence_refs
    assert contradiction not in clusters[1].evidence_refs
    assert clustering_input.relations == relations


def test_internal_contradiction_fails_closed_without_arbitrary_partition() -> None:
    relations = (
        _relation(
            "obs:a",
            "obs:b",
            SceneRelationDisposition.SUPPORTED,
            _ref("scene.support", "artifact:ab"),
        ),
        _relation(
            "obs:a",
            "obs:c",
            SceneRelationDisposition.CONTRADICTED,
            _ref("scene.contradiction", "artifact:ac"),
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            _ref("scene.support", "artifact:bc"),
        ),
    )

    with pytest.raises(
        SceneClusteringConflictError,
        match="explicitly contradicted pair",
    ):
        cluster_verified_scene_relations(_input(("obs:a", "obs:b", "obs:c"), relations))


def test_cluster_ids_use_exact_membership_sha256_rule() -> None:
    singleton = cluster_verified_scene_relations(_input(("obs:a",)))[0]
    supported = _relation(
        "obs:a",
        "obs:b",
        SceneRelationDisposition.SUPPORTED,
        _ref("scene.support", "artifact:ab"),
    )
    pair = cluster_verified_scene_relations(_input(("obs:a", "obs:b"), (supported,)))[0]

    assert singleton.cluster_id.value == (
        "scene:363205504d24f13a757d850fd7de405322c336a6c04182c17c78237570388eeb"
    )
    assert pair.cluster_id.value == (
        "scene:13591375616eec961dae3167aea62ef43bbb65c2e18fd77144fc03c375ca585e"
    )
    assert cluster_verified_scene_relations(_input(("obs:a", "obs:b"), (supported,))) == (pair,)


def test_every_observation_appears_exactly_once_in_deterministic_cluster_order() -> None:
    relations = (
        _relation(
            "obs:a",
            "obs:d",
            SceneRelationDisposition.SUPPORTED,
            _ref("scene.support", "artifact:ad"),
        ),
        _relation(
            "obs:b",
            "obs:c",
            SceneRelationDisposition.SUPPORTED,
            _ref("scene.support", "artifact:bc"),
        ),
    )

    clusters = cluster_verified_scene_relations(
        _input(("obs:a", "obs:b", "obs:c", "obs:d", "obs:e"), relations)
    )

    assert tuple(tuple(item.value for item in cluster.observation_ids) for cluster in clusters) == (
        ("obs:a", "obs:d"),
        ("obs:b", "obs:c"),
        ("obs:e",),
    )
    flattened = [
        observation_id.value for cluster in clusters for observation_id in cluster.observation_ids
    ]
    assert flattened == ["obs:a", "obs:d", "obs:b", "obs:c", "obs:e"]
    assert sorted(flattened) == ["obs:a", "obs:b", "obs:c", "obs:d", "obs:e"]
    assert len(flattened) == len(set(flattened))


def test_redundant_supported_edges_contribute_all_exact_evidence() -> None:
    evidence_ab = _ref("scene.support", "artifact:ab")
    evidence_ac = _ref("scene.support", "artifact:ac")
    evidence_bc = _ref("scene.support", "artifact:bc")
    relations = (
        _relation("obs:a", "obs:b", SceneRelationDisposition.SUPPORTED, evidence_ab),
        _relation("obs:a", "obs:c", SceneRelationDisposition.SUPPORTED, evidence_ac),
        _relation("obs:b", "obs:c", SceneRelationDisposition.SUPPORTED, evidence_bc),
    )

    cluster = cluster_verified_scene_relations(_input(("obs:a", "obs:b", "obs:c"), relations))[0]

    assert cluster.evidence_refs == tuple(
        sorted(
            (evidence_ab, evidence_ac, evidence_bc),
            key=lambda ref: (ref.artifact_kind.value, ref.artifact_id.value),
        )
    )


def test_clustering_contract_embeds_no_relationship_graph_or_later_lot_state() -> None:
    assert tuple(field.name for field in fields(SceneCluster)) == (
        "cluster_id",
        "observation_ids",
        "evidence_refs",
    )
    forbidden = {
        "PairCandidate",
        "match_colmap_pairs",
        "verify_colmap_geometry",
        "pycolmap",
        "network",
        "filesystem",
        "database",
        "registry",
        "route",
        "geometry",
        "TemporalGroup",
        "Doppelgangers",
        "LightGlue",
    }
    assert forbidden.isdisjoint(clustering_module.__dict__)


def test_clustering_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="VerifiedSceneClusteringInput"):
        cluster_verified_scene_relations(cast(Any, "not-clustering-input"))
