from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ImageObservation,
    MediaAssetRef,
    MediaProfile,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
    VideoObservation,
)

NOW = datetime(2026, 9, 18, 0, 0, tzinfo=UTC)


def _artifact_ref(artifact_id: str, artifact_kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(artifact_kind),
    )


def _asset(uri: str, digest_char: str, mime_type: str) -> MediaAssetRef:
    return MediaAssetRef(
        uri=uri,
        sha256=Sha256Digest(digest_char * 64),
        byte_length=1024,
        mime_type=mime_type,
    )


def _source() -> SourceRef:
    return SourceRef(source_id=SourceId("source:profile-fixture"))


def test_media_profile_empty_evidence_is_explicit_immutable_value() -> None:
    profile = MediaProfile(observation_id=ObservationId("obs:profile:1"))

    assert profile == MediaProfile(observation_id=ObservationId("obs:profile:1"))
    assert hash(profile) == hash(MediaProfile(observation_id=ObservationId("obs:profile:1")))
    assert profile.evidence_artifacts == ()

    with pytest.raises(FrozenInstanceError):
        profile.observation_id = ObservationId("obs:other")  # type: ignore[misc]


def test_media_profile_accepts_canonical_exact_evidence_refs() -> None:
    evidence = (
        _artifact_ref("artifact:evidence:01", "media.metadata"),
        _artifact_ref("artifact:evidence:02", "media.quality"),
    )

    profile = MediaProfile(
        observation_id=ObservationId("obs:profile:2"),
        evidence_artifacts=evidence,
    )

    assert profile.evidence_artifacts == evidence
    assert hash(profile) == hash(
        MediaProfile(
            observation_id=ObservationId("obs:profile:2"),
            evidence_artifacts=evidence,
        )
    )


def test_media_profile_requires_typed_observation_identity() -> None:
    with pytest.raises(TypeError, match=r"media_profile\.observation_id"):
        MediaProfile(observation_id=cast(Any, "obs:profile:1"))


def test_media_profile_requires_immutable_typed_evidence_collection() -> None:
    evidence = _artifact_ref("artifact:evidence:01", "media.metadata")

    with pytest.raises(TypeError, match="immutable tuple"):
        MediaProfile(
            observation_id=ObservationId("obs:profile:1"),
            evidence_artifacts=cast(Any, [evidence]),
        )

    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        MediaProfile(
            observation_id=ObservationId("obs:profile:1"),
            evidence_artifacts=(cast(Any, "artifact:evidence:01"),),
        )


def test_media_profile_rejects_duplicate_and_unsorted_evidence() -> None:
    first = _artifact_ref("artifact:evidence:01", "media.metadata")
    second = _artifact_ref("artifact:evidence:02", "media.quality")

    with pytest.raises(ValueError, match="must be unique"):
        MediaProfile(
            observation_id=ObservationId("obs:profile:1"),
            evidence_artifacts=(first, first),
        )

    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        MediaProfile(
            observation_id=ObservationId("obs:profile:1"),
            evidence_artifacts=(second, first),
        )


def test_media_profile_rejects_conflicting_kind_for_same_artifact_id() -> None:
    with pytest.raises(ValueError, match="conflicting ArtifactKind"):
        MediaProfile(
            observation_id=ObservationId("obs:profile:1"),
            evidence_artifacts=(
                _artifact_ref("artifact:evidence:01", "media.metadata"),
                _artifact_ref("artifact:evidence:01", "media.quality"),
            ),
        )


def test_media_profile_contract_has_no_speculative_signal_or_routing_fields() -> None:
    assert tuple(field.name for field in fields(MediaProfile)) == (
        "observation_id",
        "evidence_artifacts",
    )

    profile = MediaProfile(observation_id=ObservationId("obs:profile:1"))
    for attribute in (
        "asset",
        "uri",
        "sha256",
        "mime_type",
        "observation_kind",
        "blur",
        "exposure",
        "duplicate",
        "diversity",
        "camera_class",
        "dynamic",
        "coverage",
        "metrics",
        "quality_decision",
        "failure",
        "policy",
        "route",
        "signals",
        "metadata",
        "notes",
    ):
        assert not hasattr(profile, attribute)


def test_retained_observation_kinds_are_profiled_only_through_observation_id() -> None:
    source = _source()
    image = ImageObservation(
        observation_id=ObservationId("obs:image:profile"),
        asset=_asset("file:///profile/image.jpg", "a", "image/jpeg"),
        source=source,
        received_at=NOW,
    )
    video = VideoObservation(
        observation_id=ObservationId("obs:video:profile"),
        asset=_asset("file:///profile/video.mp4", "b", "video/mp4"),
        source=source,
        received_at=NOW,
    )
    frame = VideoFrameObservation(
        observation_id=ObservationId("obs:frame:profile"),
        asset=_asset("file:///profile/frame.png", "c", "image/png"),
        source=source,
        received_at=NOW,
        video_asset=video.asset,
        frame_index=0,
        frame_time_us=0,
    )

    assert MediaProfile(observation_id=image.observation_id).observation_id == image.observation_id
    assert MediaProfile(observation_id=video.observation_id).observation_id == video.observation_id
    assert MediaProfile(observation_id=frame.observation_id).observation_id == frame.observation_id
