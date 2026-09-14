from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wre.domain import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    ObservationKind,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
)

NOW = datetime(2026, 9, 14, 19, 0, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64


def _asset(
    uri: str = "file:///observations/image.jpg",
    sha256: str = SHA_A,
    *,
    byte_length: int = 1024,
    mime_type: str | None = "image/jpeg",
) -> MediaAssetRef:
    return MediaAssetRef(
        uri=uri,
        sha256=Sha256Digest(sha256),
        byte_length=byte_length,
        mime_type=mime_type,
    )


def _source() -> SourceRef:
    return SourceRef(source_id=SourceId("upload:local"), locator="incoming/image.jpg")


def test_image_observation_is_raw_solver_independent_record() -> None:
    observation = ImageObservation(
        observation_id=ObservationId("obs:image:0001"),
        asset=_asset(),
        source=_source(),
        received_at=NOW,
        captured_at=None,
    )

    assert observation.kind is ObservationKind.IMAGE
    assert str(observation.observation_id) == "obs:image:0001"
    assert observation.asset.byte_length == 1024
    assert observation.captured_at is None


def test_sha256_is_validated_and_canonicalized() -> None:
    digest = Sha256Digest("ABCDEF" * 10 + "ABCD")

    assert digest.value == ("abcdef" * 10 + "abcd")

    with pytest.raises(ValueError, match="64 hexadecimal"):
        Sha256Digest("not-a-digest")


def test_ids_are_typed_and_reject_ambiguous_text() -> None:
    assert ObservationId("obs:1") != SourceId("obs:1")

    with pytest.raises(ValueError, match="observation_id"):
        ObservationId("bad id")

    with pytest.raises(ValueError, match="source_id"):
        SourceId("/absolute/path")


def test_observation_times_must_be_timezone_aware_when_present() -> None:
    naive = datetime(2026, 9, 14, 19, 0)

    with pytest.raises(ValueError, match="received_at must be timezone-aware"):
        ImageObservation(
            observation_id=ObservationId("obs:image:naive-received"),
            asset=_asset(),
            source=_source(),
            received_at=naive,
        )

    with pytest.raises(ValueError, match="captured_at must be timezone-aware"):
        ImageObservation(
            observation_id=ObservationId("obs:image:naive-captured"),
            asset=_asset(),
            source=_source(),
            received_at=NOW,
            captured_at=naive,
        )


def test_video_frame_keeps_parent_video_identity_and_integer_time_offset() -> None:
    video_asset = _asset(
        uri="file:///observations/source.mp4",
        sha256=SHA_B,
        byte_length=8_000_000,
        mime_type="video/mp4",
    )
    frame_asset = _asset(
        uri="file:///observations/source/frame-0012.png",
        byte_length=40_000,
        mime_type="image/png",
    )
    observation = VideoFrameObservation(
        observation_id=ObservationId("obs:frame:0012"),
        asset=frame_asset,
        source=SourceRef(source_id=SourceId("video:import-1")),
        received_at=NOW,
        video_asset=video_asset,
        frame_index=12,
        frame_time_us=400_000,
    )

    assert observation.kind is ObservationKind.VIDEO_FRAME
    assert observation.video_asset.sha256 == Sha256Digest(SHA_B)
    assert observation.frame_index == 12
    assert observation.frame_time_us == 400_000


@pytest.mark.parametrize(
    ("frame_index", "frame_time_us", "message"),
    [
        (-1, 0, "frame_index"),
        (0, -1, "frame_time_us"),
    ],
)
def test_video_frame_rejects_negative_coordinates_in_source_video(
    frame_index: int, frame_time_us: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        VideoFrameObservation(
            observation_id=ObservationId("obs:frame:invalid"),
            asset=_asset(),
            source=_source(),
            received_at=NOW,
            video_asset=_asset(
                uri="file:///observations/source.mp4",
                sha256=SHA_B,
                mime_type="video/mp4",
            ),
            frame_index=frame_index,
            frame_time_us=frame_time_us,
        )


def test_media_asset_requires_stable_reference_and_non_negative_size() -> None:
    with pytest.raises(ValueError, match=r"asset\.uri"):
        _asset(uri=" ")

    with pytest.raises(ValueError, match="byte_length"):
        _asset(byte_length=-1)

    with pytest.raises(ValueError, match=r"asset\.mime_type"):
        _asset(mime_type=" ")
