from __future__ import annotations

import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import MediaAssetRef, ObservationId, SourceId, SourceRef, VideoObservation
from wre.ingestion import (
    FFmpegToolchain,
    KeyframeExtractionRequest,
    KeyframeSelectionPolicy,
    LocalKeyframeExtractor,
    hash_file_content,
    inspect_toolchain,
)
from wre.persistence import SQLiteLocalStore

NOW = datetime(2026, 9, 16, 17, 0, tzinfo=UTC)


def _require_ffmpeg_pair() -> tuple[str, str]:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        pytest.skip("ffmpeg/ffprobe are not installed")
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


def test_installed_ffmpeg_pair_can_execute_wre_keyframe_contract(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _require_ffmpeg_pair()
    toolchain = FFmpegToolchain(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        required_version=None,
        timeout_seconds=30,
    )
    identity = inspect_toolchain(toolchain)
    assert identity.canonical_version

    video_path = _make_synthetic_video(tmp_path, ffmpeg)
    source_hash = hash_file_content(video_path)
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")
    video = VideoObservation(
        observation_id=ObservationId("video:ffmpeg-compat"),
        asset=MediaAssetRef(
            uri=video_path.resolve().as_uri(),
            sha256=source_hash.sha256,
            byte_length=source_hash.byte_length,
            mime_type="video/x-matroska",
        ),
        source=SourceRef(source_id=SourceId("fixture:ffmpeg-compat")),
        received_at=NOW,
        captured_at=NOW,
    )
    store.put_observation(video)

    extractor = LocalKeyframeExtractor(store, toolchain)
    request = KeyframeExtractionRequest(
        video=video,
        source_path=video_path,
        output_dir=tmp_path / "keyframes",
        policy=KeyframeSelectionPolicy(min_interval_us=500_000),
    )

    first = extractor.extract(request)
    second = extractor.extract(request)

    assert first == second
    assert first.toolchain == identity
    assert tuple(frame.frame_index for frame in first.frames) == (0, 2, 4, 6)
    assert tuple(frame.frame_time_us for frame in first.frames) == (
        0,
        500_000,
        1_000_000,
        1_500_000,
    )
    assert all(frame.asset.mime_type == "image/png" for frame in first.frames)
    assert all(store.get_observation(frame.observation_id) == frame for frame in first.frames)
