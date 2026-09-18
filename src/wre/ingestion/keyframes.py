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

SUPPORTED_FFMPEG_VERSION = "6.1.1-3ubuntu5"
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

    @property
    def canonical_version(self) -> str:
        if self.ffmpeg_version != self.ffprobe_version:
            raise ValueError("ffmpeg and ffprobe versions do not match")
        return self.ffmpeg_version


@dataclass(frozen=True, slots=True, kw_only=True)
class KeyframeExtractionRequest:
    video: VideoObservation
    source_path: Path
    output_dir: Path
    policy: KeyframeSelectionPolicy


@dataclass(frozen=True, slots=True)
class KeyframeExtractionResult:
    source_observation_id: ObservationId
    toolchain: FFmpegToolchainIdentity
    policy: KeyframeSelectionPolicy
    frames: tuple[VideoFrameObservation, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedFrameExtractionRequest:
    """Materialize an explicit already-selected source-frame set."""

    video: VideoObservation
    source_path: Path
    output_dir: Path
    selected_frames: tuple[ProbedVideoFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.video, VideoObservation):
            raise TypeError("selected_frame_extraction.video must be VideoObservation")
        if not isinstance(self.source_path, Path):
            raise TypeError("selected_frame_extraction.source_path must be Path")
        if not isinstance(self.output_dir, Path):
            raise TypeError("selected_frame_extraction.output_dir must be Path")
        _validate_selected_frames(self.selected_frames)


@dataclass(frozen=True, slots=True)
class SelectedFrameExtractionResult:
    source_observation_id: ObservationId
    toolchain: FFmpegToolchainIdentity
    selected_frames: tuple[ProbedVideoFrame, ...]
    frames: tuple[VideoFrameObservation, ...]


def _validate_selected_frames(frames: object) -> tuple[ProbedVideoFrame, ...]:
    if not isinstance(frames, tuple):
        raise TypeError("selected_frame_extraction.selected_frames must be an immutable tuple")
    if not frames:
        raise ValueError("selected_frame_extraction.selected_frames must not be empty")
    if any(not isinstance(frame, ProbedVideoFrame) for frame in frames):
        raise TypeError(
            "selected_frame_extraction.selected_frames members must be ProbedVideoFrame"
        )

    previous_index = -1
    previous_time = -1
    for frame in frames:
        if frame.frame_index <= previous_index:
            raise ValueError("selected_frame_extraction frame indices must be strictly increasing")
        if frame.frame_time_us < previous_time:
            raise ValueError("selected_frame_extraction frame timestamps must be non-decreasing")
        previous_index = frame.frame_index
        previous_time = frame.frame_time_us

    return frames


def select_keyframes(
    frames: tuple[ProbedVideoFrame, ...],
    policy: KeyframeSelectionPolicy,
) -> tuple[ProbedVideoFrame, ...]:
    """Select source frames with deterministic minimum temporal spacing."""

    if not frames:
        raise ValueError("at least one timestamped source frame is required")

    previous_index = -1
    previous_time = -1
    selected: list[ProbedVideoFrame] = []
    last_selected_time: int | None = None

    for frame in frames:
        if frame.frame_index <= previous_index:
            raise ValueError("source frame indices must be strictly increasing")
        if frame.frame_time_us < previous_time:
            raise ValueError("source frame timestamps must be non-decreasing")
        previous_index = frame.frame_index
        previous_time = frame.frame_time_us

        enough_spacing = (
            last_selected_time is None
            or frame.frame_time_us - last_selected_time >= policy.min_interval_us
        )
        if enough_spacing:
            selected.append(frame)
            last_selected_time = frame.frame_time_us

    return tuple(selected)


def parse_ffprobe_frames(payload: str) -> tuple[ProbedVideoFrame, ...]:
    """Parse first-video-stream frame timestamps into integer microseconds."""

    try:
        document = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("ffprobe output is not valid JSON") from exc
    if not isinstance(document, dict) or not isinstance(document.get("frames"), list):
        raise ValueError("ffprobe output must contain a frames array")

    frames: list[ProbedVideoFrame] = []
    previous_raw_time: int | None = None
    for frame_index, item in enumerate(document["frames"]):
        if not isinstance(item, dict):
            raise ValueError("ffprobe frame entries must be objects")
        raw_time = item.get("best_effort_timestamp_time")
        if raw_time is None or raw_time == "N/A":
            raise ValueError(f"ffprobe frame {frame_index} has no usable best-effort timestamp")
        if not isinstance(raw_time, str):
            raise ValueError("ffprobe best_effort_timestamp_time must be a string")
        try:
            timestamp = Decimal(raw_time)
        except InvalidOperation as exc:
            raise ValueError("invalid ffprobe frame timestamp") from exc
        if not timestamp.is_finite():
            raise ValueError("invalid ffprobe frame timestamp")
        microseconds = int(
            (timestamp * _MICROSECONDS_PER_SECOND).to_integral_value(rounding=ROUND_HALF_EVEN)
        )
        if previous_raw_time is not None and microseconds < previous_raw_time:
            raise ValueError("ffprobe frame timestamps must be non-decreasing")
        previous_raw_time = microseconds
        if microseconds < 0:
            continue
        frames.append(ProbedVideoFrame(frame_index=frame_index, frame_time_us=microseconds))

    if not frames:
        raise ValueError("ffprobe returned no non-negative timestamped video frames")
    return tuple(frames)


def _run_command(args: list[str], *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FFmpegExecutionError(f"failed to execute {args[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if len(stderr) > 2_000:
            stderr = stderr[-2_000:]
        raise FFmpegExecutionError(f"{args[0]!r} exited with code {completed.returncode}: {stderr}")
    return completed


def _tool_version(executable: str, expected_name: str, *, timeout_seconds: int) -> str:
    completed = _run_command([executable, "-version"], timeout_seconds=timeout_seconds)
    first_line = completed.stdout.splitlines()[0] if completed.stdout else ""
    match = _VERSION_RE.match(first_line)
    if match is None or match.group(1) != expected_name:
        raise FFmpegExecutionError(f"unexpected {expected_name} version output: {first_line!r}")
    return match.group(2)


def inspect_toolchain(toolchain: FFmpegToolchain) -> FFmpegToolchainIdentity:
    identity = FFmpegToolchainIdentity(
        ffmpeg_version=_tool_version(
            toolchain.ffmpeg_path,
            "ffmpeg",
            timeout_seconds=toolchain.timeout_seconds,
        ),
        ffprobe_version=_tool_version(
            toolchain.ffprobe_path,
            "ffprobe",
            timeout_seconds=toolchain.timeout_seconds,
        ),
    )
    version = identity.canonical_version
    if toolchain.required_version is not None and version != toolchain.required_version:
        raise FFmpegExecutionError(
            f"FFmpeg version mismatch: required {toolchain.required_version!r}, found {version!r}"
        )
    return identity


def probe_video_frames(
    source_path: Path,
    toolchain: FFmpegToolchain,
) -> tuple[ProbedVideoFrame, ...]:
    completed = _run_command(
        [
            toolchain.ffprobe_path,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "json",
            str(source_path),
        ],
        timeout_seconds=toolchain.timeout_seconds,
    )
    return parse_ffprobe_frames(completed.stdout)


def _selection_filter(frames: tuple[ProbedVideoFrame, ...]) -> str:
    return "select=" + "+".join(f"eq(n\\,{frame.frame_index})" for frame in frames)


def _frame_observation_id(
    video: VideoObservation,
    frame: ProbedVideoFrame,
) -> ObservationId:
    material = (
        f"{video.observation_id.value}\0{video.asset.sha256.value}\0"
        f"{frame.frame_index}\0{frame.frame_time_us}"
    ).encode()
    return ObservationId(f"vf:{hashlib.sha256(material).hexdigest()}")


def _verified_source_path(video: VideoObservation, source_path: Path) -> Path:
    resolved = source_path.expanduser().resolve(strict=True)
    source_hash = hash_file_content(resolved)
    if (
        source_hash.sha256 != video.asset.sha256
        or source_hash.byte_length != video.asset.byte_length
    ):
        raise ValueError("source video bytes do not match the persisted VideoObservation asset")
    return resolved


class LocalKeyframeExtractor:
    """Extract deterministic source-frame samples through an external FFmpeg CLI."""

    def __init__(self, sink: ObservationSink, toolchain: FFmpegToolchain) -> None:
        self._sink = sink
        self._toolchain = toolchain

    def _materialize_selected_frames(
        self,
        *,
        video: VideoObservation,
        source_path: Path,
        output_dir: Path,
        selected: tuple[ProbedVideoFrame, ...],
    ) -> tuple[VideoFrameObservation, ...]:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="wre-keyframes-", dir=output_dir) as temp_name:
            temp_dir = Path(temp_name)
            output_pattern = temp_dir / "frame-%08d.png"
            _run_command(
                [
                    self._toolchain.ffmpeg_path,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-y",
                    "-i",
                    str(source_path),
                    "-map",
                    "0:v:0",
                    "-vf",
                    _selection_filter(selected),
                    "-fps_mode",
                    "passthrough",
                    "-c:v",
                    "png",
                    "-pix_fmt",
                    "rgb24",
                    str(output_pattern),
                ],
                timeout_seconds=self._toolchain.timeout_seconds,
            )
            extracted_paths = tuple(sorted(temp_dir.glob("frame-*.png")))
            if len(extracted_paths) != len(selected):
                raise FFmpegExecutionError(
                    "FFmpeg extracted an unexpected number of frames: "
                    f"expected {len(selected)}, found {len(extracted_paths)}"
                )

            observations: list[VideoFrameObservation] = []
            for frame, temporary_path in zip(selected, extracted_paths, strict=True):
                final_path = output_dir / (
                    f"frame-{frame.frame_index:012d}-{frame.frame_time_us:016d}.png"
                )
                temporary_path.replace(final_path)
                frame_hash = hash_file_content(final_path)
                observation = VideoFrameObservation(
                    observation_id=_frame_observation_id(video, frame),
                    asset=MediaAssetRef(
                        uri=final_path.as_uri(),
                        sha256=Sha256Digest(frame_hash.sha256.value),
                        byte_length=frame_hash.byte_length,
                        mime_type="image/png",
                    ),
                    source=video.source,
                    received_at=video.received_at,
                    captured_at=None,
                    video_asset=video.asset,
                    frame_index=frame.frame_index,
                    frame_time_us=frame.frame_time_us,
                )
                self._sink.put_observation(observation)
                observations.append(observation)

        return tuple(observations)

    def extract_selected(
        self,
        request: SelectedFrameExtractionRequest,
    ) -> SelectedFrameExtractionResult:
        """Materialize exactly the supplied selected source frames without probing or selection."""

        if not isinstance(request, SelectedFrameExtractionRequest):
            raise TypeError(
                "selected_frame_extraction.request must be SelectedFrameExtractionRequest"
            )

        source_path = _verified_source_path(request.video, request.source_path)
        identity = inspect_toolchain(self._toolchain)
        selected = _validate_selected_frames(request.selected_frames)
        observations = self._materialize_selected_frames(
            video=request.video,
            source_path=source_path,
            output_dir=request.output_dir,
            selected=selected,
        )
        return SelectedFrameExtractionResult(
            source_observation_id=request.video.observation_id,
            toolchain=identity,
            selected_frames=selected,
            frames=observations,
        )

    def extract(self, request: KeyframeExtractionRequest) -> KeyframeExtractionResult:
        source_path = _verified_source_path(request.video, request.source_path)
        identity = inspect_toolchain(self._toolchain)
        probed = probe_video_frames(source_path, self._toolchain)
        selected = select_keyframes(probed, request.policy)
        observations = self._materialize_selected_frames(
            video=request.video,
            source_path=source_path,
            output_dir=request.output_dir,
            selected=selected,
        )
        return KeyframeExtractionResult(
            source_observation_id=request.video.observation_id,
            toolchain=identity,
            policy=request.policy,
            frames=observations,
        )
