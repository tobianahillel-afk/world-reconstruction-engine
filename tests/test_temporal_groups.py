from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.temporal_groups as temporal_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.scene_clusters import SceneClusterId
from wre.domain.temporal_groups import (
    SyncHypothesis,
    SyncHypothesisDisposition,
    TemporalGroup,
    TemporalGroupId,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _sync(
    disposition: SyncHypothesisDisposition,
    *,
    offset_us: int | None,
    evidence_refs: tuple[ArtifactRef, ...] | None = None,
) -> SyncHypothesis:
    return SyncHypothesis(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        disposition=disposition,
        offset_us=offset_us,
        evidence_refs=evidence_refs
        if evidence_refs is not None
        else (_ref("sync.evidence", "artifact:sync"),),
    )


def test_temporal_group_id_validates_opaque_lowercase_tokens() -> None:
    assert str(TemporalGroupId("group:a-1")) == "group:a-1"
    assert TemporalGroupId("a").value == "a"
    assert TemporalGroupId("a" * 128).value == "a" * 128

    with pytest.raises(TypeError, match="must be str"):
        TemporalGroupId(cast(Any, 1))
    with pytest.raises(ValueError, match="1-128 lowercase"):
        TemporalGroupId("")
    with pytest.raises(ValueError, match="1-128 lowercase"):
        TemporalGroupId("Upper")
    with pytest.raises(ValueError, match="1-128 lowercase"):
        TemporalGroupId("a" * 129)
    with pytest.raises(ValueError, match="1-128 lowercase"):
        TemporalGroupId("group/value")


def test_sync_disposition_vocabulary_is_exact() -> None:
    assert tuple(item.value for item in SyncHypothesisDisposition) == (
        "supported",
        "contradicted",
        "unresolved",
    )


def test_sync_hypothesis_is_exact_immutable_contract() -> None:
    hypothesis = _sync(
        SyncHypothesisDisposition.SUPPORTED,
        offset_us=1250,
    )

    assert tuple(field.name for field in fields(SyncHypothesis)) == (
        "observation_id1",
        "observation_id2",
        "disposition",
        "offset_us",
        "evidence_refs",
    )
    with pytest.raises(FrozenInstanceError):
        hypothesis.offset_us = 0  # type: ignore[misc]


def test_sync_hypothesis_validates_canonical_endpoints_and_disposition() -> None:
    evidence = (_ref("sync.evidence", "artifact:sync"),)

    with pytest.raises(TypeError, match="observation_id1"):
        SyncHypothesis(
            observation_id1=cast(Any, "obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=evidence,
        )
    with pytest.raises(TypeError, match="observation_id2"):
        SyncHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=cast(Any, "obs:b"),
            disposition=SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=evidence,
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        SyncHypothesis(
            observation_id1=ObservationId("obs:b"),
            observation_id2=ObservationId("obs:a"),
            disposition=SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=evidence,
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        SyncHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:a"),
            disposition=SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=evidence,
        )
    with pytest.raises(TypeError, match="SyncHypothesisDisposition"):
        SyncHypothesis(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            disposition=cast(Any, "supported"),
            offset_us=0,
            evidence_refs=evidence,
        )


@pytest.mark.parametrize("offset_us", [-250_000, 0, 750_000])
def test_supported_sync_hypothesis_accepts_signed_microsecond_offsets(
    offset_us: int,
) -> None:
    hypothesis = _sync(
        SyncHypothesisDisposition.SUPPORTED,
        offset_us=offset_us,
    )

    assert hypothesis.offset_us == offset_us


def test_supported_sync_hypothesis_requires_real_integer_offset() -> None:
    with pytest.raises(TypeError, match="integer microsecond offset"):
        _sync(SyncHypothesisDisposition.SUPPORTED, offset_us=None)
    with pytest.raises(TypeError, match="integer microsecond offset"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=cast(Any, True),
        )
    with pytest.raises(TypeError, match="integer microsecond offset"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=cast(Any, 1.5),
        )


@pytest.mark.parametrize(
    "disposition",
    [
        SyncHypothesisDisposition.CONTRADICTED,
        SyncHypothesisDisposition.UNRESOLVED,
    ],
)
def test_non_supported_sync_hypothesis_forbids_fake_offset(
    disposition: SyncHypothesisDisposition,
) -> None:
    assert _sync(disposition, offset_us=None).offset_us is None

    with pytest.raises(ValueError, match="must not expose offset_us"):
        _sync(disposition, offset_us=0)


def test_sync_hypothesis_requires_canonical_non_empty_evidence() -> None:
    ref_a = _ref("sync.a", "artifact:a")
    ref_b = _ref("sync.b", "artifact:b")

    hypothesis = _sync(
        SyncHypothesisDisposition.SUPPORTED,
        offset_us=0,
        evidence_refs=(ref_a, ref_b),
    )
    assert hypothesis.evidence_refs == (ref_a, ref_b)

    with pytest.raises(TypeError, match="immutable tuple"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=cast(Any, [ref_a]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=cast(Any, ("not-a-ref",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=(ref_a, ref_a),
        )
    with pytest.raises(ValueError, match="canonical artifact order"):
        _sync(
            SyncHypothesisDisposition.SUPPORTED,
            offset_us=0,
            evidence_refs=(ref_b, ref_a),
        )


def test_temporal_group_is_exact_immutable_contract() -> None:
    group = TemporalGroup(
        group_id=TemporalGroupId("group:a"),
        scene_cluster_id=SceneClusterId("scene:a"),
        observation_ids=(ObservationId("obs:a"),),
    )

    assert tuple(field.name for field in fields(TemporalGroup)) == (
        "group_id",
        "scene_cluster_id",
        "observation_ids",
        "evidence_refs",
    )
    with pytest.raises(FrozenInstanceError):
        group.observation_ids = ()  # type: ignore[misc]


def test_temporal_group_validates_identity_parent_and_membership() -> None:
    with pytest.raises(TypeError, match="TemporalGroupId"):
        TemporalGroup(
            group_id=cast(Any, "group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=(ObservationId("obs:a"),),
        )
    with pytest.raises(TypeError, match="SceneClusterId"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=cast(Any, "scene:a"),
            observation_ids=(ObservationId("obs:a"),),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=cast(Any, [ObservationId("obs:a")]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=(),
        )
    with pytest.raises(TypeError, match="ObservationId"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=cast(Any, ("obs:a",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:a")),
            evidence_refs=(_ref("sync.evidence", "artifact:sync"),),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
            evidence_refs=(_ref("sync.evidence", "artifact:sync"),),
        )


def test_singleton_temporal_group_may_have_no_edge_evidence() -> None:
    group = TemporalGroup(
        group_id=TemporalGroupId("group:a"),
        scene_cluster_id=SceneClusterId("scene:a"),
        observation_ids=(ObservationId("obs:a"),),
    )

    assert group.evidence_refs == ()


def test_multi_observation_temporal_group_requires_canonical_evidence() -> None:
    ref_a = _ref("sync.a", "artifact:a")
    ref_b = _ref("sync.b", "artifact:b")
    observations = (ObservationId("obs:a"), ObservationId("obs:b"))

    with pytest.raises(ValueError, match="requires explicit supporting evidence_refs"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=observations,
        )

    group = TemporalGroup(
        group_id=TemporalGroupId("group:a"),
        scene_cluster_id=SceneClusterId("scene:a"),
        observation_ids=observations,
        evidence_refs=(ref_a, ref_b),
    )
    assert group.evidence_refs == (ref_a, ref_b)

    with pytest.raises(ValueError, match="duplicates"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=observations,
            evidence_refs=(ref_a, ref_a),
        )
    with pytest.raises(ValueError, match="canonical artifact order"):
        TemporalGroup(
            group_id=TemporalGroupId("group:a"),
            scene_cluster_id=SceneClusterId("scene:a"),
            observation_ids=observations,
            evidence_refs=(ref_b, ref_a),
        )


def test_temporal_contracts_do_not_embed_absolute_or_execution_state() -> None:
    forbidden_fields = {
        "datetime",
        "timestamp",
        "captured_at",
        "publication_time",
        "timezone",
        "epoch",
        "valid_from",
        "valid_to",
        "clock_rate",
        "drift",
        "score",
        "confidence",
        "probability",
        "rank",
        "route",
        "quality_mode",
        "solver",
        "model",
        "path",
        "metadata",
    }
    assert forbidden_fields.isdisjoint(field.name for field in fields(SyncHypothesis))
    assert forbidden_fields.isdisjoint(field.name for field in fields(TemporalGroup))

    forbidden_module_symbols = {
        "datetime",
        "subprocess",
        "Path",
        "FFmpeg",
        "pycolmap",
        "network",
        "requests",
        "TemporalState",
        "ChangeEvent",
    }
    assert forbidden_module_symbols.isdisjoint(temporal_module.__dict__)
