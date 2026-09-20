from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

import pytest

import wre.synchronization.metadata_time as metadata_time_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
)
from wre.domain.observations import (
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
)
from wre.domain.temporal_groups import SyncHypothesisDisposition
from wre.synchronization import (
    CaptureTimeSyncAdapterInput,
    SameVideoFrameSyncAdapterInput,
    adapt_capture_time_sync,
    adapt_same_video_frame_sync,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _capture(
    observation_id: str,
    capture_time: CaptureTimeInterpretation,
) -> ObservationMetadataInterpretation:
    return ObservationMetadataInterpretation(
        observation_id=ObservationId(observation_id),
        gps=GpsMetadataInterpretation(status=GpsInterpretationStatus.ABSENT),
        capture_time=capture_time,
    )


def _resolved(instant: datetime) -> CaptureTimeInterpretation:
    return CaptureTimeInterpretation(
        status=CaptureTimeInterpretationStatus.RESOLVED,
        raw_datetime="2026:09:20 12:00:00",
        raw_offset="+00:00",
        instant=instant,
        evidence_keys=("EXIF DateTimeOriginal", "EXIF OffsetTimeOriginal"),
    )


def _absent() -> CaptureTimeInterpretation:
    return CaptureTimeInterpretation(status=CaptureTimeInterpretationStatus.ABSENT)


def _local_ambiguous() -> CaptureTimeInterpretation:
    return CaptureTimeInterpretation(
        status=CaptureTimeInterpretationStatus.LOCAL_AMBIGUOUS,
        raw_datetime="2026:09:20 12:00:00",
        evidence_keys=("EXIF DateTimeOriginal",),
    )


def _invalid() -> CaptureTimeInterpretation:
    return CaptureTimeInterpretation(
        status=CaptureTimeInterpretationStatus.INVALID,
        raw_datetime="2026:09:20 12:00:00",
        raw_offset="+25:00",
        evidence_keys=("EXIF DateTimeOriginal", "EXIF OffsetTimeOriginal"),
        issue="invalid offset",
    )


def _capture_input(
    first: CaptureTimeInterpretation,
    second: CaptureTimeInterpretation,
    *,
    evidence_refs: tuple[ArtifactRef, ...] | None = None,
) -> CaptureTimeSyncAdapterInput:
    return CaptureTimeSyncAdapterInput(
        interpretation1=_capture("obs:a", first),
        interpretation2=_capture("obs:b", second),
        evidence_refs=evidence_refs
        if evidence_refs is not None
        else (_ref("sync.capture_time", "artifact:capture-time"),),
    )


def _asset(uri: str, fill: str) -> MediaAssetRef:
    return MediaAssetRef(
        uri=uri,
        sha256=Sha256Digest(fill * 64),
        byte_length=100,
        mime_type="video/mp4",
    )


def _frame(
    observation_id: str,
    *,
    frame_time_us: int,
    video_asset: MediaAssetRef,
    captured_at: datetime | None = None,
) -> VideoFrameObservation:
    frame_fill = "c" if observation_id.endswith("a") else "d"
    return VideoFrameObservation(
        observation_id=ObservationId(observation_id),
        asset=_asset(f"memory://{observation_id}.png", frame_fill),
        source=SourceRef(source_id=SourceId("video:source")),
        received_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        captured_at=captured_at,
        video_asset=video_asset,
        frame_index=0 if observation_id.endswith("a") else 1,
        frame_time_us=frame_time_us,
    )


def test_capture_time_input_is_exact_immutable_contract() -> None:
    adapter_input = _capture_input(_absent(), _absent())

    assert tuple(field.name for field in fields(CaptureTimeSyncAdapterInput)) == (
        "interpretation1",
        "interpretation2",
        "evidence_refs",
    )
    with pytest.raises(FrozenInstanceError):
        adapter_input.evidence_refs = ()  # type: ignore[misc]


def test_capture_time_input_validates_canonical_endpoints_and_evidence() -> None:
    ref_a = _ref("sync.a", "artifact:a")
    ref_b = _ref("sync.b", "artifact:b")
    first = _capture("obs:a", _absent())
    second = _capture("obs:b", _absent())

    with pytest.raises(TypeError, match="interpretation1"):
        CaptureTimeSyncAdapterInput(
            interpretation1=cast(Any, "bad"),
            interpretation2=second,
            evidence_refs=(ref_a,),
        )
    with pytest.raises(TypeError, match="interpretation2"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=cast(Any, "bad"),
            evidence_refs=(ref_a,),
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        CaptureTimeSyncAdapterInput(
            interpretation1=second,
            interpretation2=first,
            evidence_refs=(ref_a,),
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=first,
            evidence_refs=(ref_a,),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=second,
            evidence_refs=cast(Any, [ref_a]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=second,
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=second,
            evidence_refs=cast(Any, ("bad",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=second,
            evidence_refs=(ref_a, ref_a),
        )
    with pytest.raises(ValueError, match="canonical artifact order"):
        CaptureTimeSyncAdapterInput(
            interpretation1=first,
            interpretation2=second,
            evidence_refs=(ref_b, ref_a),
        )


@pytest.mark.parametrize(
    ("first_us", "second_us", "expected"),
    [
        (750_000, 250_000, -500_000),
        (250_000, 250_000, 0),
        (250_000, 750_000, 500_000),
    ],
)
def test_resolved_capture_times_produce_exact_signed_microsecond_offset(
    first_us: int,
    second_us: int,
    expected: int,
) -> None:
    base = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    result = adapt_capture_time_sync(
        _capture_input(
            _resolved(base + timedelta(microseconds=first_us)),
            _resolved(base + timedelta(microseconds=second_us)),
        )
    )

    assert result is not None
    assert result.disposition is SyncHypothesisDisposition.SUPPORTED
    assert result.offset_us == expected


def test_resolved_capture_times_compare_aware_instants_across_offsets() -> None:
    first = datetime(2026, 9, 20, 14, 0, tzinfo=timezone(timedelta(hours=2)))
    second = datetime(2026, 9, 20, 9, 30, tzinfo=timezone(timedelta(hours=-3)))
    result = adapt_capture_time_sync(_capture_input(_resolved(first), _resolved(second)))

    assert result is not None
    assert result.disposition is SyncHypothesisDisposition.SUPPORTED
    assert result.offset_us == 1_800_000_000


def test_capture_time_adapter_preserves_every_supplied_evidence_ref() -> None:
    base = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    refs = (
        _ref("sync.a", "artifact:a"),
        _ref("sync.b", "artifact:b"),
    )

    result = adapt_capture_time_sync(
        _capture_input(_resolved(base), _resolved(base), evidence_refs=refs)
    )

    assert result is not None
    assert result.evidence_refs == refs


def test_both_absent_capture_times_return_no_hypothesis() -> None:
    assert adapt_capture_time_sync(_capture_input(_absent(), _absent())) is None


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (_resolved(datetime(2026, 9, 20, 12, 0, tzinfo=UTC)), _absent()),
        (_local_ambiguous(), _resolved(datetime(2026, 9, 20, 12, 0, tzinfo=UTC))),
        (_invalid(), _resolved(datetime(2026, 9, 20, 12, 0, tzinfo=UTC))),
        (_local_ambiguous(), _invalid()),
    ],
)
def test_incomplete_capture_time_evidence_remains_unresolved(
    first: CaptureTimeInterpretation,
    second: CaptureTimeInterpretation,
) -> None:
    result = adapt_capture_time_sync(_capture_input(first, second))

    assert result is not None
    assert result.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert result.offset_us is None


def test_capture_time_adapter_never_emits_contradicted() -> None:
    cases = (
        _capture_input(_absent(), _absent()),
        _capture_input(
            _resolved(datetime(2026, 9, 20, 12, 0, tzinfo=UTC)),
            _resolved(datetime(2026, 9, 21, 12, 0, tzinfo=UTC)),
        ),
        _capture_input(_local_ambiguous(), _invalid()),
    )

    results = tuple(adapt_capture_time_sync(case) for case in cases)

    assert all(
        result is None or result.disposition is not SyncHypothesisDisposition.CONTRADICTED
        for result in results
    )


def test_same_video_frame_input_is_exact_immutable_contract() -> None:
    video = _asset("memory://video.mp4", "a")
    adapter_input = SameVideoFrameSyncAdapterInput(
        frame1=_frame("obs:a", frame_time_us=0, video_asset=video),
        frame2=_frame("obs:b", frame_time_us=1, video_asset=video),
        evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
    )

    assert tuple(field.name for field in fields(SameVideoFrameSyncAdapterInput)) == (
        "frame1",
        "frame2",
        "evidence_ref",
    )
    with pytest.raises(FrozenInstanceError):
        adapter_input.evidence_ref = _ref("sync.video_frame", "artifact:other")  # type: ignore[misc]


def test_same_video_frame_input_validates_endpoints_evidence_and_parent_video() -> None:
    video = _asset("memory://video.mp4", "a")
    other_video = _asset("memory://other.mp4", "b")
    frame_a = _frame("obs:a", frame_time_us=0, video_asset=video)
    frame_b = _frame("obs:b", frame_time_us=1, video_asset=video)

    with pytest.raises(TypeError, match="frame1"):
        SameVideoFrameSyncAdapterInput(
            frame1=cast(Any, "bad"),
            frame2=frame_b,
            evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
        )
    with pytest.raises(TypeError, match="frame2"):
        SameVideoFrameSyncAdapterInput(
            frame1=frame_a,
            frame2=cast(Any, "bad"),
            evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
        )
    with pytest.raises(ValueError, match="distinct and canonically ordered"):
        SameVideoFrameSyncAdapterInput(
            frame1=frame_b,
            frame2=frame_a,
            evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        SameVideoFrameSyncAdapterInput(
            frame1=frame_a,
            frame2=frame_b,
            evidence_ref=cast(Any, "bad"),
        )
    with pytest.raises(ValueError, match="exact same parent video_asset"):
        SameVideoFrameSyncAdapterInput(
            frame1=frame_a,
            frame2=_frame("obs:b", frame_time_us=1, video_asset=other_video),
            evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
        )


@pytest.mark.parametrize(
    ("first_us", "second_us", "expected"),
    [
        (900_000, 100_000, -800_000),
        (100_000, 100_000, 0),
        (100_000, 900_000, 800_000),
    ],
)
def test_same_video_frame_timing_produces_exact_signed_offset(
    first_us: int,
    second_us: int,
    expected: int,
) -> None:
    video = _asset("memory://video.mp4", "a")
    evidence = _ref("sync.video_frame", "artifact:frame-time")
    result = adapt_same_video_frame_sync(
        SameVideoFrameSyncAdapterInput(
            frame1=_frame("obs:a", frame_time_us=first_us, video_asset=video),
            frame2=_frame("obs:b", frame_time_us=second_us, video_asset=video),
            evidence_ref=evidence,
        )
    )

    assert result.disposition is SyncHypothesisDisposition.SUPPORTED
    assert result.offset_us == expected
    assert result.evidence_refs == (evidence,)


def test_same_video_frame_sync_ignores_absolute_captured_at() -> None:
    video = _asset("memory://video.mp4", "a")
    result = adapt_same_video_frame_sync(
        SameVideoFrameSyncAdapterInput(
            frame1=_frame(
                "obs:a",
                frame_time_us=10,
                video_asset=video,
                captured_at=datetime(2020, 1, 1, tzinfo=UTC),
            ),
            frame2=_frame(
                "obs:b",
                frame_time_us=25,
                video_asset=video,
                captured_at=datetime(2030, 1, 1, tzinfo=UTC),
            ),
            evidence_ref=_ref("sync.video_frame", "artifact:frame-time"),
        )
    )

    assert result.offset_us == 15


def test_adapters_reject_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="CaptureTimeSyncAdapterInput"):
        adapt_capture_time_sync(cast(Any, "bad"))
    with pytest.raises(TypeError, match="SameVideoFrameSyncAdapterInput"):
        adapt_same_video_frame_sync(cast(Any, "bad"))


def test_metadata_time_adapter_has_no_external_execution_or_persistence_surface() -> None:
    forbidden = {
        "Path",
        "subprocess",
        "ffmpeg",
        "pycolmap",
        "requests",
        "socket",
        "SQLiteLocalStore",
        "TemporalGroup",
        "received_at",
    }
    assert forbidden.isdisjoint(metadata_time_module.__dict__)
