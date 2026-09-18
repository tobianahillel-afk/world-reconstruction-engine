from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import wre.ingestion.keyframes as keyframes_module
from wre.domain import MediaAssetRef, ObservationId, SourceId, SourceRef, VideoObservation
from wre.ingestion import (
    SUPPORTED_FFMPEG_VERSION,
    FFmpegToolchain,
    LocalKeyframeExtractor,
    ProbedVideoFrame,
    SelectedFrameExtractionRequest,
    hash_file_content,
    inspect_toolchain,
)
from wre.persistence import SQLiteLocalStore

NOW = datetime(2026, 9, 18, 17, 15, tzinfo=UTC)


def _source() -> SourceRef:
    return SourceRef(source_id=SourceId("fixture:selected-frame-extraction"))


def _video(path: Path, *, byte_length_delta: int = 0) -> VideoObservation:
    source_hash = hash_file_content(path)
    return VideoObservation(
        observation_id=ObservationId("video:selected-frame-regression"),
        asset=MediaAssetRef(
            uri=path.resolve().as_uri(),
            sha256=source_hash.sha256,
            byte_length=source_hash.byte_length + byte_length_delta,
            mime_type="video/x-matroska",
        ),
        source=_source(),
        received_at=NOW,
        captured_at=NOW,
    )


def _selected_frames() -> tuple[ProbedVideoFrame, ...]:
    return (
        ProbedVideoFrame(frame_index=1, frame_time_us=250_000),
        ProbedVideoFrame(frame_index=4, frame_time_us=1_000_000),
        ProbedVideoFrame(frame_index=7, frame_time_us=1_750_000),
    )


def test_selected_frame_extraction_request_is_exact_typed_and_immutable(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")
    video = _video(source)
    selected = _selected_frames()
    request = SelectedFrameExtractionRequest(
        video=video,
        source_path=source,
        output_dir=tmp_path / "frames",
        selected_frames=selected,
    )

    assert tuple(field.name for field in fields(SelectedFrameExtractionRequest)) == (
        "video",
        "source_path",
        "output_dir",
        "selected_frames",
    )
    assert request.selected_frames is selected
    with pytest.raises(FrozenInstanceError):
        request.output_dir = tmp_path / "other"  # type: ignore[misc]

    with pytest.raises(TypeError, match="video must be VideoObservation"):
        SelectedFrameExtractionRequest(
            video=cast(Any, "video"),
            source_path=source,
            output_dir=tmp_path / "frames",
            selected_frames=selected,
        )
    with pytest.raises(TypeError, match="source_path must be Path"):
        SelectedFrameExtractionRequest(
            video=video,
            source_path=cast(Any, str(source)),
            output_dir=tmp_path / "frames",
            selected_frames=selected,
        )
    with pytest.raises(TypeError, match="output_dir must be Path"):
        SelectedFrameExtractionRequest(
            video=video,
            source_path=source,
            output_dir=cast(Any, str(tmp_path / "frames")),
            selected_frames=selected,
        )


@pytest.mark.parametrize(
    "selected",
    [
        (),
        cast(Any, [_selected_frames()[0]]),
        (cast(Any, "frame"),),
        (
            ProbedVideoFrame(frame_index=1, frame_time_us=0),
            ProbedVideoFrame(frame_index=1, frame_time_us=1),
        ),
        (
            ProbedVideoFrame(frame_index=1, frame_time_us=10),
            ProbedVideoFrame(frame_index=2, frame_time_us=9),
        ),
    ],
)
def test_selected_frame_extraction_request_fails_closed_on_invalid_selection(
    tmp_path: Path,
    selected: object,
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")

    with pytest.raises((TypeError, ValueError)):
        SelectedFrameExtractionRequest(
            video=_video(source),
            source_path=source,
            output_dir=tmp_path / "frames",
            selected_frames=cast(Any, selected),
        )


def test_selected_extraction_verifies_source_bytes_before_toolchain(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"source bytes")
    video = _video(source, byte_length_delta=1)
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")
    extractor = LocalKeyframeExtractor(
        store,
        FFmpegToolchain(ffmpeg_path="must-not-run", ffprobe_path="must-not-run"),
    )

    with pytest.raises(ValueError, match="source video bytes do not match"):
        extractor.extract_selected(
            SelectedFrameExtractionRequest(
                video=video,
                source_path=source,
                output_dir=tmp_path / "frames",
                selected_frames=_selected_frames(),
            )
        )


def _require_pinned_ffmpeg() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("GitHub CI must provide ffmpeg and ffprobe")
        pytest.skip("ffmpeg/ffprobe are not installed")

    identity = inspect_toolchain(
        FFmpegToolchain(
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            required_version=None,
            timeout_seconds=30,
        )
    )
    if identity.canonical_version != SUPPORTED_FFMPEG_VERSION:
        message = (
            "selected-frame extraction integration requires FFmpeg "
            f"{SUPPORTED_FFMPEG_VERSION}, found {identity.canonical_version}"
        )
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(message)
        pytest.skip(message)
    return ffmpeg, ffprobe


def _write_ppm(path: Path, *, frame_number: int, width: int = 16, height: int = 16) -> None:
    header = f"P6\n{width} {height}\n255\n".encode()
    pixel = bytes(
        (
            (frame_number * 29) % 256,
            (frame_number * 53) % 256,
            (frame_number * 97) % 256,
        )
    )
    path.write_bytes(header + pixel * (width * height))


def _make_synthetic_video(tmp_path: Path, ffmpeg: str) -> Path:
    input_dir = tmp_path / "ppm"
    input_dir.mkdir()
    for index in range(8):
        _write_ppm(input_dir / f"frame-{index:03d}.ppm", frame_number=index)

    video_path = tmp_path / "source.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-framerate",
            "4",
            "-start_number",
            "0",
            "-i",
            str(input_dir / "frame-%03d.ppm"),
            "-c:v",
            "ffv1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    return video_path


def test_explicit_selected_extraction_preserves_provenance_timing_and_idempotence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ffmpeg, ffprobe = _require_pinned_ffmpeg()
    video_path = _make_synthetic_video(tmp_path, ffmpeg)
    video = _video(video_path)
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")
    store.put_observation(video)
    selected = _selected_frames()
    request = SelectedFrameExtractionRequest(
        video=video,
        source_path=video_path,
        output_dir=tmp_path / "selected-frames",
        selected_frames=selected,
    )
    extractor = LocalKeyframeExtractor(
        store,
        FFmpegToolchain(
            ffmpeg_path=ffmpeg,
            ffprobe_path=ffprobe,
            timeout_seconds=30,
        ),
    )

    def _unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"explicit selected extraction re-entered selection: {args} {kwargs}")

    monkeypatch.setattr(keyframes_module, "probe_video_frames", _unexpected)
    monkeypatch.setattr(keyframes_module, "select_keyframes", _unexpected)

    first = extractor.extract_selected(request)
    second = extractor.extract_selected(request)

    assert first == second
    assert first.source_observation_id == video.observation_id
    assert first.selected_frames is selected
    assert tuple(frame.frame_index for frame in first.frames) == (1, 4, 7)
    assert tuple(frame.frame_time_us for frame in first.frames) == (
        250_000,
        1_000_000,
        1_750_000,
    )

    for source_frame, observation in zip(selected, first.frames, strict=True):
        assert observation.frame_index == source_frame.frame_index
        assert observation.frame_time_us == source_frame.frame_time_us
        assert observation.video_asset == video.asset
        assert observation.source == video.source
        assert observation.received_at == video.received_at
        assert observation.captured_at is None
        assert observation.asset.mime_type == "image/png"

        output_path = Path(observation.asset.uri.removeprefix("file://"))
        assert output_path.exists()
        output_hash = hash_file_content(output_path)
        assert output_hash.sha256 == observation.asset.sha256
        assert output_hash.byte_length == observation.asset.byte_length
        assert store.get_observation(observation.observation_id) == observation


def test_selected_extraction_module_adds_no_future_selection_or_runtime_surface() -> None:
    for name in (
        "MetricVector",
        "QualityDecision",
        "FrameSelectionDensityBounds",
        "QualityDiversitySelectionConfig",
        "RuntimeTargetProfile",
        "JobSpec",
        "scheduler",
        "nvImageCodec",
        "nvJPEG",
        "NVDEC",
        "libvips",
    ):
        assert name not in keyframes_module.__dict__
