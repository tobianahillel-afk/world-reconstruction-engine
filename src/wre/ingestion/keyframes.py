from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation
from pathlib import Path

from wre.domain.observations import (
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    VideoFrameObservation,
    VideoObservation,
)
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.images import ObservationSink

SUPPORTED_FFMPEG_VERSION = "8.0.1-3ubuntu2"
_VERSION_RE = re.compile(r"^(ffmpeg|ffprobe) version ([^\s]+)")
_MICROSECONDS_PER_SECOND = Decimal(1_000_000)


class FFmpegExecutionError(RuntimeError):
    """Raised when the configured FFmpeg toolchain cannot complete an operation."""


@dataclass(frozen=True, slots=True)
class ProbedVideoFrame:
    frame_index: int
    frame_time_us: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a non-negative integer")
        if (
            isinstance(self.frame_time_us, bool)
            or not isinstance(self.frame_time_us, int)
            or self.frame_time_us < 0
        ):
            raise ValueError("frame_time_us must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class KeyframeSelectionPolicy:
    min_interval_us: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_interval_us, bool)
            or not isinstance(self.min_interval_us, int)
            or self.min_interval_us <= 0
        ):
            raise ValueError("min_interval_us must be a positive integer")


@dataclass(frozen=True, slots=True, kw_only=True)
class FFmpegToolchain:
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
    required_version: str | None = SUPPORTED_FFMPEG_VERSION
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if not self.ffmpeg_path.strip() or not self.ffprobe_path.strip():
            raise ValueError("FFmpeg executable paths must be non-empty")
        if self.required_version is not None and not self.required_version.strip():
            raise ValueError("required_version must be non-empty when provided")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive integer")


@dataclass(frozen=True, slots=True)
class FFmpegToolchainIdentity:
    ffmpeg_version: str
    ffprobe_version: str

    def __post_init__(self) -> None:
        if not self.ffmpeg_version.strip() or not self.ffprobe_version.strip():
            raise ValueError("FFmpeg tool versions must be non-empty")
        if self.ffmpeg_version != self.ffprobe_version:
            raise ValueError("ffmpeg and ffprobe versions must match exactly")

    @property
    def canonical_version(self) -> str:
        return self.ffmpeg_version


@dataclass(frozen=True, slots=True)
class KeyframeExtractionRequest:
    video: VideoObservation
    source_path: Path
    output_dir: Path
    policy: KeyframeSelectionPolicy


@dataclass(frozen=True, slots=True)
class KeyframeExtractionResult:
    source_observation_id: ObservationId
    source_asset: MediaAssetRef
    toolchain: FFmpegToolchainIdentity
    policy: KeyframeSelectionPolicy
    frames: tuple[VideoFrameObservation, ...]


def parse_ffprobe_frames(payload: str) -> tuple[ProbedVideoFrame, ...]:
    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FFmpegExecutionError("ffprobe returned invalid JSON") from exc

    frames = document.get("frames")
    if not isinstance(frames, list):
        raise FFmpegExecutionError("ffprobe JSON does not contain a frame list")

    parsed: list[ProbedVideoFrame] = []
    for source_index, raw_frame in enumerate(frames):
        if not isinstance(raw_frame, dict):
            raise FFmpegExecutionError("ffprobe frame entry must be an object")
        timestamp = raw_frame.get("best_effort_timestamp_time")
        if timestamp is None:
            raise FFmpegExecutionError("ffprobe frame has no usable best-effort timestamp")
        try:
            seconds = Decimal(str(timestamp))
        except (InvalidOperation, ValueError) as exc:
            raise FFmpegExecutionError("invalid ffprobe frame timestamp") from exc
        if not seconds.is_finite():
            raise FFmpegExecutionError("invalid ffprobe frame timestamp")
        microseconds_decimal = (seconds * _MICROSECONDS_PER_SECOND).quantize(
            Decimal("1"), rounding=ROUND_HALF_EVEN
        )
        microseconds = int(microseconds_decimal)
        if microseconds < 0:
            continue
        parsed.append(ProbedVideoFrame(frame_index=source_index, frame_time_us=microseconds))

    if not parsed:
        raise FFmpegExecutionError("ffprobe produced no usable non-negative video frames")
    _validate_monotone_frame_times(parsed)
    return tuple(parsed)


def select_keyframes(
    frames: tuple[ProbedVideoFrame, ...],
    policy: KeyframeSelectionPolicy,
) -> tuple[ProbedVideoFrame, ...]:
    _validate_monotone_frame_times(frames)
    if not frames:
        return ()

    selected = [frames[0]]
    last_selected_time = frames[0].frame_time_us
    for frame in frames[1:]:
        if frame.frame_time_us - last_selected_time >= policy.min_interval_us:
            selected.append(frame)
            last_selected_time = frame.frame_time_us
    return tuple(selected)


def inspect_toolchain(toolchain: FFmpegToolchain) -> FFmpegToolchainIdentity:
    ffmpeg_version = _run_version_command(toolchain.ffmpeg_path, toolchain.timeout_seconds)
    ffprobe_version = _run_version_command(toolchain.ffprobe_path, toolchain.timeout_seconds)
    identity = FFmpegToolchainIdentity(
        ffmpeg_version=ffmpeg_version,
        ffprobe_version=ffprobe_version,
    )
    if toolchain.required_version is not None and identity.canonical_version != toolchain.required_version:
        raise FFmpegExecutionError(
            "FFmpeg toolchain version mismatch: "
            f"expected {toolchain.required_version!r}, got {identity.canonical_version!r}"
        )
    return identity


class LocalKeyframeExtractor:
    def __init__(self, sink: ObservationSink, toolchain: FFmpegToolchain | None = None) -> None:
        self._sink = sink
        self._toolchain = toolchain or FFmpegToolchain()

    def extract(self, request: KeyframeExtractionRequest) -> KeyframeExtractionResult:
        source_path = request.source_path.resolve(strict=True)
        _verify_source_asset(source_path, request.video.asset)
        toolchain_identity = inspect_toolchain(self._toolchain)
        frames = _probe_video_frames(
            source_path,
            self._toolchain.ffprobe_path,
            self._toolchain.timeout_seconds,
        )
        selected = select_keyframes(frames, request.policy)
        request.output_dir.mkdir(parents=True, exist_ok=True)

        materialized: list[VideoFrameObservation] = []
        for frame in selected:
            output_path = request.output_dir / _frame_filename(frame)
            _extract_frame_png(
                source_path,
                output_path,
                frame.frame_index,
                self._toolchain.ffmpeg_path,
                self._toolchain.timeout_seconds,
            )
            output_hash = hash_file_content(output_path)
            observation = VideoFrameObservation(
                observation_id=_frame_observation_id(request.video, toolchain_identity, request.policy, frame),
                asset=MediaAssetRef(
                    uri=output_path.resolve().as_uri(),
                    sha256=output_hash.sha256,
                    byte_length=output_hash.byte_length,
                    mime_type="image/png",
                ),
                video_asset=request.video.asset,
                frame_index=frame.frame_index,
                frame_time_us=frame.frame_time_us,
                source=request.video.source,
                received_at=request.video.received_at,
            )
            self._sink.put_observation(observation)
            materialized.append(observation)

        return KeyframeExtractionResult(
            source_observation_id=request.video.observation_id,
            source_asset=request.video.asset,
            toolchain=toolchain_identity,
            policy=request.policy,
            frames=tuple(materialized),
        )


def _verify_source_asset(source_path: Path, expected_asset: MediaAssetRef) -> None:
    actual = hash_file_content(source_path)
    if actual.sha256 != expected_asset.sha256 or actual.byte_length != expected_asset.byte_length:
        raise ValueError("source bytes do not match the persisted video asset")


def _run_version_command(executable: str, timeout_seconds: int) -> str:
    result = _run_command(
        [executable, "-version"],
        timeout_seconds,
        context=f"inspect {Path(executable).name} version",
    )
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    match = _VERSION_RE.match(first_line)
    if match is None:
        raise FFmpegExecutionError(f"cannot parse {Path(executable).name} version output")
    return match.group(2)


def _probe_video_frames(source_path: Path, ffprobe: str, timeout_seconds: int) -> tuple[ProbedVideoFrame, ...]:
    result = _run_command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source_path),
        ],
        timeout_seconds,
        context="probe video frame timestamps",
    )
    return parse_ffprobe_frames(result.stdout)


def _extract_frame_png(
    source_path: Path,
    output_path: Path,
    frame_index: int,
    ffmpeg: str,
    timeout_seconds: int,
) -> None:
    with tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp.png",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
    try:
        _run_command(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(source_path),
                "-vf",
                f"select=eq(n\\,{frame_index})",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                str(temp_path),
            ],
            timeout_seconds,
            context=f"extract video frame {frame_index}",
        )
        if not temp_path.is_file() or temp_path.stat().st_size <= 0:
            raise FFmpegExecutionError(f"ffmpeg did not materialize frame {frame_index}")
        temp_path.replace(output_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _run_command(
    args: list[str],
    timeout_seconds: int,
    *,
    context: str,
) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FFmpegExecutionError(f"{context} failed") from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
        raise FFmpegExecutionError(f"{context} failed: {detail}")
    return result


def _frame_filename(frame: ProbedVideoFrame) -> str:
    return f"frame-{frame.frame_index:09d}-{frame.frame_time_us:016d}.png"


def _frame_observation_id(
    video: VideoObservation,
    toolchain: FFmpegToolchainIdentity,
    policy: KeyframeSelectionPolicy,
    frame: ProbedVideoFrame,
) -> ObservationId:
    material = "\n".join(
        (
            "wre.video-frame.v1",
            video.asset.sha256.value,
            str(video.asset.byte_length),
            toolchain.canonical_version,
            str(policy.min_interval_us),
            str(frame.frame_index),
            str(frame.frame_time_us),
        )
    ).encode()
    digest = hashlib.sha256(material).hexdigest()
    return ObservationId(f"frame:{digest}")


def _validate_monotone_frame_times(frames: tuple[ProbedVideoFrame, ...] | list[ProbedVideoFrame]) -> None:
    previous: int | None = None
    for frame in frames:
        if previous is not None and frame.frame_time_us < previous:
            raise ValueError("frame_time_us must be non-decreasing")
        previous = frame.frame_time_us
