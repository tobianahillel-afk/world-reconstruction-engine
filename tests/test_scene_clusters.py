from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.scene_clusters as scene_clusters_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ObservationId,
    PairCandidate,
    SceneCluster,
    SceneClusterId,
    SceneRelationDisposition,
    SceneRelationHypothesis,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _relation(
    disposition: SceneRelationDisposition = SceneRelationDisposition.SUPPORTED,
) -> SceneRelationHypothesis:
    return SceneRelationHypothesis(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        disposition=disposition,
        evidence_refs=(_ref("scene.relation", "artifact:relation-a"),),
    )


def test_scene_relation_disposition_is_closed_explicit_vocabulary() -> None:
    assert tuple(SceneRelationDisposition) == (
        SceneRelationDisposition.SUPPORTED,
        SceneRelationDisposition.CONTRADICTED,
        SceneRelationDisposition.UNRESOLVED,
    )
    assert tuple(item.value for item in SceneRelationDisposition) == (
        "supported",
        "contradicted",
        "unresolved",
    )
    with pytest.raises(ValueError):
        SceneRelationDisposition("same_scene")
    with pytest.raises(ValueError):
        SceneRelationDisposition(cast(Any, True))
    with pytest.raises(ValueError):
        SceneRelationDisposition(cast(Any, 1))


def test_scene_relation_hypothesis_is_exact_immutable_evidence_backed_contract() -> None:
    relation = _relation()

    assert tuple(field.name for field in fields(SceneRelationHypothesis)) == (
        "observation_id1",
        "observation_id2",
        "disposition",
        "evidence_refs",
    )
    assert hash(relation) == hash(_relation())
    with pytest.raises(FrozenInstanceError):
        relation.disposition = SceneRelationDisposition.UNRESOLVED  # type: ignore[misc]


def test_scene_relation_hypothesis_requires_canonical_endpoints_and_disposition() -> None:
    ref = _ref("scene.relation", "artifact:relation-a")

    with pytest.raises(TypeError, match="observation_id1"):
        SceneRelationHypothesis(
            observation_id1=cast(Any, "obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(ref,),
        )
    with pytest.raises(TypeError, match="observation_id2"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=cast(Any, "obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(ref,),
        )
    with pytest.raises(ValueError, match="canonically ordered"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:b"),
            observation_id2=ObservationId("obs:a"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(ref,),
        )
    with pytest.raises(ValueError, match="canonically ordered"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:a"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(ref,),
        )
    with pytest.raises(TypeError, match="SceneRelationDisposition"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=cast(Any, "supported"),
            evidence_refs=(ref,),
        )
    with pytest.raises(TypeError, match="SceneRelationDisposition"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=cast(Any, True),
            evidence_refs=(ref,),
        )


def test_scene_relation_hypothesis_requires_nonempty_canonical_unique_evidence() -> None:
    first = _ref("scene.relation", "artifact:a")
    second = _ref("scene.relation", "artifact:b")

    relation = SceneRelationHypothesis(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        disposition=SceneRelationDisposition.UNRESOLVED,
        evidence_refs=(first, second),
    )
    assert relation.evidence_refs == (first, second)

    with pytest.raises(TypeError, match="immutable tuple"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=cast(Any, [first]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=cast(Any, ("artifact:a",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(first, first),
        )
    with pytest.raises(ValueError, match="canonical artifact order"):
        SceneRelationHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SceneRelationDisposition.SUPPORTED,
            evidence_refs=(second, first),
        )


def test_relation_dispositions_preserve_same_explicit_endpoints_and_evidence() -> None:
    supported = _relation(SceneRelationDisposition.SUPPORTED)
    contradicted = _relation(SceneRelationDisposition.CONTRADICTED)
    unresolved = _relation(SceneRelationDisposition.UNRESOLVED)

    assert supported.observation_id1 == contradicted.observation_id1 == unresolved.observation_id1
    assert supported.observation_id2 == contradicted.observation_id2 == unresolved.observation_id2
    assert supported.evidence_refs == contradicted.evidence_refs == unresolved.evidence_refs
    assert {
        supported.disposition,
        contradicted.disposition,
        unresolved.disposition,
    } == set(SceneRelationDisposition)

    forbidden = {
        "score",
        "confidence",
        "probability",
        "rank",
        "distance",
        "sequence_distance",
        "gps_distance",
        "geolocation",
        "capture_time",
        "correspondences",
        "match_count",
        "inlier_count",
        "geometry",
        "scene_truth",
        "route",
        "quality_decision",
        "metadata",
    }
    assert forbidden.isdisjoint(SceneRelationHypothesis.__dataclass_fields__)


def test_scene_cluster_id_is_exact_immutable_lowercase_token() -> None:
    cluster_id = SceneClusterId("scene:cluster-01")

    assert str(cluster_id) == "scene:cluster-01"
    assert hash(cluster_id) == hash(SceneClusterId("scene:cluster-01"))
    with pytest.raises(FrozenInstanceError):
        cluster_id.value = "scene:other"  # type: ignore[misc]

    for value in (
        "",
        " ",
        "Scene:cluster",
        "scene cluster",
        " scene:cluster",
        "scene:cluster ",
        "scene/cluster",
        "x" * 129,
    ):
        with pytest.raises(ValueError):
            SceneClusterId(value)
    with pytest.raises(TypeError, match="must be str"):
        SceneClusterId(cast(Any, 7))


def test_scene_cluster_is_exact_immutable_membership_contract() -> None:
    evidence = _ref("scene.cluster.support", "artifact:support")
    cluster = SceneCluster(
        cluster_id=SceneClusterId("scene:ab"),
        observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
        evidence_refs=(evidence,),
    )

    assert tuple(field.name for field in fields(SceneCluster)) == (
        "cluster_id",
        "observation_ids",
        "evidence_refs",
    )
    assert hash(cluster) == hash(
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=(evidence,),
        )
    )
    with pytest.raises(FrozenInstanceError):
        cluster.observation_ids = (ObservationId("obs:a"),)  # type: ignore[misc]


def test_scene_cluster_accepts_explicit_singleton_without_supporting_relation() -> None:
    cluster = SceneCluster(
        cluster_id=SceneClusterId("scene:isolated"),
        observation_ids=(ObservationId("obs:isolated"),),
        evidence_refs=(),
    )

    assert cluster.evidence_refs == ()


def test_scene_cluster_rejects_invalid_membership_and_unsupported_multi_member_state() -> None:
    evidence_a = _ref("scene.cluster.support", "artifact:a")
    evidence_b = _ref("scene.cluster.support", "artifact:b")

    with pytest.raises(TypeError, match="cluster_id"):
        SceneCluster(
            cluster_id=cast(Any, "scene:a"),
            observation_ids=(ObservationId("obs:a"),),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:a"),
            observation_ids=cast(Any, [ObservationId("obs:a")]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:empty"),
            observation_ids=(),
        )
    with pytest.raises(TypeError, match="ObservationId"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:a"),
            observation_ids=cast(Any, ("obs:a",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:a"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:a")),
            evidence_refs=(evidence_a,),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ba"),
            observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
            evidence_refs=(evidence_a,),
        )
    with pytest.raises(ValueError, match="requires explicit supporting evidence"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=cast(Any, [evidence_a]),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=cast(Any, ("artifact:a",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=(evidence_a, evidence_a),
        )
    with pytest.raises(ValueError, match="canonical artifact order"):
        SceneCluster(
            cluster_id=SceneClusterId("scene:ab"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
            evidence_refs=(evidence_b, evidence_a),
        )


def test_scene_contracts_add_no_matching_clustering_or_truth_execution_surface() -> None:
    assert tuple(PairCandidate.__dataclass_fields__) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    for forbidden in (
        "PairCandidate",
        "CorrespondenceSet",
        "LocalMatcher",
        "GeometricVerification",
        "verify_geometry",
        "merge_clusters",
        "build_clusters",
        "transitive_closure",
        "union_find",
        "TemporalGroup",
        "RouteGraph",
        "GeometrySolution",
        "filesystem",
        "network",
    ):
        assert forbidden not in scene_clusters_module.__dict__

    relation = _relation(SceneRelationDisposition.UNRESOLVED)
    cluster = SceneCluster(
        cluster_id=SceneClusterId("scene:single"),
        observation_ids=(relation.observation_id1,),
    )
    assert cluster.observation_ids == (ObservationId("obs:a"),)
