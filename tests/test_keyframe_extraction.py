from __future__ import annotations

import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from wre.domain import MediaAssetRef, ObservationId, SourceId, SourceRef, VideoObservation
from wre.ingestion import (
    FFmpegToolchain,
    KeyframeExtractionRequest,
    KeyframeSelectionPolicy,
    LocalKeyframeExtractor,
    ProbedVideoFrame,
    hash_file_content,
    inspect_toolchain,
    parse_ffprobe_frames,
    select_keyframes,
)
from wre.persistence import SQLiteLocalStore

NOW = datetime(2026, 9, 15, 1, 0, tzinfo=UTC)


def test_select_keyframes_uses_observed_frame_times_and_minimum_spacing() -> None:
    frames = (
        ProbedVideoFrame(frame_index=0, frame_time_us=0),
        ProbedVideoFrame(frame_index=1, frame_time_us=250_000),
        ProbedVideoFrame(frame_index=2, frame_time_us=500_000),
        ProbedVideoFrame(frame_index=3, frame_time_us=750_000),
        ProbedVideoFrame(frame_index=4, frame_time_us=1_000_000),
    )

    selected = select_keyframes(frames, KeyframeSelectionPolicy(min_interval_us=500_000))

    assert selected == (frames[0], frames[2], frames[4])


def test_select_keyframes_rejects_non_monotone_timestamps() -> None:
    frames = (
        ProbedVideoFrame(frame_index=0, frame_time_us=100),
        ProbedVideoFrame(frame_index=1, frame_time_us=99),
    )

    with pytest.raises(ValueError, match="non-decreasing"):
        select_keyframes(frames, KeyframeSelectionPolicy(min_interval_us=1))


def test_parse_ffprobe_frames_preserves_source_indices_and_integer_microseconds() -> None:
    payload = """
    {
      "frames": [
        {"best_effort_timestamp_time": "-0.250000"},
        {"best_effort_timestamp_time": "0.0000004"},
        {},
        {"best_effort_timestamp_time": "0.4999996"},
        {"best_effort_timestamp_time": "1.000000"}
      ]
    }
    """

    frames = parse_ffprobe_frames(payload)

    assert frames == (
        ProbedVideoFrame(frame_index=1, frame_time_us=0),
        ProbedVideoFrame(frame_index=3, frame_time_us=500_000),
        ProbedVideoFrame(frame_index=4, frame_time_us=1_000_000),
    )


def test_parse_ffprobe_frames_rejects_non_monotone_usable_timestamps() -> None:
    payload = """
    {
      "frames": [
        {"best_effort_timestamp_time": "1.000000"},
        {"best_effort_timestamp_time": "0.500000"}
      ]
    }
    """

    with pytest.raises(ValueError, match="non-decreasing"):
        parse_ffprobe_frames(payload)


def test_source_bytes_must_match_persisted_video_before_toolchain_execution(tmp_path: Path) -> None:
    source = tmp_path / "source.mkv"
    source.write_bytes(b"actual bytes")
    source_hash = hash_file_content(source)
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")

    video = VideoObservation(
        observation_id=ObservationId("video:mismatch"),
        asset=MediaAssetRef(
            uri="file:///recorded/source.mkv",
            sha256=source_hash.sha256,
            byte_length=source_hash.byte_length + 1,
        ),
        source=SourceRef(source_id=SourceId("camera:1")),
        received_at=NOW,
    )

    extractor = LocalKeyframeExtractor(
        store,
        FFmpegToolchain(ffmpeg_path="must-not-run", ffprobe_path="must-not-run"),
    )

    with pytest.raises(ValueError, match="do not match"):
        extractor.extract(
            KeyframeExtractionRequest(
                video=video,
                source_path=source,
                output_dir=tmp_path / "frames",
                policy=KeyframeSelectionPolicy(min_interval_us=500_000),
            )
        )


def _require_ffmpeg_for_integration() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is not None and ffprobe is not None:
        return ffmpeg, ffprobe
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail("GitHub CI must provide ffmpeg and ffprobe for the L2.6 integration test")
    pytest.skip("ffmpeg/ffprobe are not installed in this local environment")


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


def _file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_real_ffmpeg_keyframe_extraction_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _require_ffmpeg_for_integration()
    video_path = _make_synthetic_video(tmp_path, ffmpeg)
    source_hash = hash_file_content(video_path)
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")
    video = VideoObservation(
        observation_id=ObservationId("video:synthetic:1"),
        asset=MediaAssetRef(
            uri=video_path.resolve().as_uri(),
            sha256=source_hash.sha256,
            byte_length=source_hash.byte_length,
            mime_type="video/x-matroska",
        ),
        source=SourceRef(source_id=SourceId("fixture:synthetic-video")),
        received_at=NOW,
        captured_at=NOW,
    )
    store.put_observation(video)

    toolchain = FFmpegToolchain(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe, timeout_seconds=30)
    identity = inspect_toolchain(toolchain)
    pinned_toolchain = FFmpegToolchain(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        required_version=identity.canonical_version,
        timeout_seconds=30,
    )
    extractor = LocalKeyframeExtractor(store, pinned_toolchain)
    request = KeyframeExtractionRequest(
        video=video,
        source_path=video_path,
        output_dir=tmp_path / "keyframes",
        policy=KeyframeSelectionPolicy(min_interval_us=500_000),
    )

    first = extractor.extract(request)
    second = extractor.extract(request)

    assert first == second
    assert first.toolchain.canonical_version == identity.canonical_version
    assert tuple(frame.frame_index for frame in first.frames) == (0, 2, 4, 6)
    assert tuple(frame.frame_time_us for frame in first.frames) == (
        0,
        500_000,
        1_000_000,
        1_500_000,
    )
    assert tuple(frame.captured_at for frame in first.frames) == (
        NOW,
        NOW + timedelta(microseconds=500_000),
        NOW + timedelta(seconds=1),
        NOW + timedelta(seconds=1, microseconds=500_000),
    )
    for frame in first.frames:
        assert frame.video_asset == video.asset
        assert frame.asset.mime_type == "image/png"
        assert _file_uri_path(frame.asset.uri).exists()
        assert store.get_observation(frame.observation_id) == frame
