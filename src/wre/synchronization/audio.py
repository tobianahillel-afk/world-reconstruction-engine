from __future__ import annotations

import math
import struct
import subprocess
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal
from enum import StrEnum
from pathlib import Path

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import VideoObservation
from wre.domain.temporal_groups import SyncHypothesis, SyncHypothesisDisposition
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.keyframes import (
    FFmpegExecutionError,
    FFmpegToolchain,
    FFmpegToolchainIdentity,
    inspect_toolchain,
)

_MICROSECONDS_PER_SECOND = 1_000_000
_MAX_REFERENCE_CORRELATION_WORK = 5_000_000


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class AudioCorrelationPolicy:
    """Explicit bounded reference-policy parameters; no hidden acceptance threshold."""

    sample_rate_hz: int
    window_duration_us: int
    max_lag_us: int
    minimum_overlap_us: int
    minimum_correlation: float

    def __post_init__(self) -> None:
        sample_rate_hz = _positive_int(self.sample_rate_hz, "audio_policy.sample_rate_hz")
        window_duration_us = _positive_int(
            self.window_duration_us,
            "audio_policy.window_duration_us",
        )
        max_lag_us = _positive_int(self.max_lag_us, "audio_policy.max_lag_us")
        minimum_overlap_us = _positive_int(
            self.minimum_overlap_us,
            "audio_policy.minimum_overlap_us",
        )
        if max_lag_us > window_duration_us:
            raise ValueError("audio_policy.max_lag_us must not exceed window_duration_us")
        if minimum_overlap_us > window_duration_us:
            raise ValueError(
                "audio_policy.minimum_overlap_us must not exceed window_duration_us"
            )
        if (
            isinstance(self.minimum_correlation, bool)
            or not isinstance(self.minimum_correlation, float)
            or not math.isfinite(self.minimum_correlation)
            or not 0.0 < self.minimum_correlation <= 1.0
        ):
            raise ValueError(
                "audio_policy.minimum_correlation must be a finite float within (0, 1]"
            )

        max_samples = (
            sample_rate_hz * window_duration_us + _MICROSECONDS_PER_SECOND - 1
        ) // _MICROSECONDS_PER_SECOND
        max_lag_samples = (
            sample_rate_hz * max_lag_us // _MICROSECONDS_PER_SECOND
        )
        estimated_work = max_samples * (2 * max_lag_samples + 1)
        if estimated_work > _MAX_REFERENCE_CORRELATION_WORK:
            raise ValueError(
                "audio_policy exceeds the bounded reference correlation work limit"
            )


@dataclass(frozen=True, slots=True)
class AudioCorrelationSyncRequest:
    """Two verified-source candidates for bounded audio correlation."""

    video1: VideoObservation
    source_path1: Path
    video2: VideoObservation
    source_path2: Path
    policy: AudioCorrelationPolicy
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.video1, VideoObservation):
            raise TypeError("audio_sync.video1 must be VideoObservation")
        if not isinstance(self.video2, VideoObservation):
            raise TypeError("audio_sync.video2 must be VideoObservation")
        if self.video1.observation_id.value >= self.video2.observation_id.value:
            raise ValueError(
                "audio_sync observation IDs must be distinct and canonically ordered"
            )
        if not isinstance(self.source_path1, Path):
            raise TypeError("audio_sync.source_path1 must be Path")
        if not isinstance(self.source_path2, Path):
            raise TypeError("audio_sync.source_path2 must be Path")
        if not isinstance(self.policy, AudioCorrelationPolicy):
            raise TypeError("audio_sync.policy must be AudioCorrelationPolicy")
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("audio_sync.evidence_ref must be ArtifactRef")


class AudioCorrelationEvidenceStatus(StrEnum):
    ACCEPTED = "accepted"
    WEAK = "weak"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AudioCorrelationEvidence:
    """Solver-private bounded correlation result retained outside SyncHypothesis."""

    sample_rate_hz: int
    lag_samples: int
    overlap_samples: int
    correlation: float
    status: AudioCorrelationEvidenceStatus

    def __post_init__(self) -> None:
        _positive_int(self.sample_rate_hz, "audio_evidence.sample_rate_hz")
        if isinstance(self.lag_samples, bool) or not isinstance(self.lag_samples, int):
            raise TypeError("audio_evidence.lag_samples must be int")
        if (
            isinstance(self.overlap_samples, bool)
            or not isinstance(self.overlap_samples, int)
            or self.overlap_samples < 0
        ):
            raise ValueError("audio_evidence.overlap_samples must be a non-negative integer")
        if (
            isinstance(self.correlation, bool)
            or not isinstance(self.correlation, float)
            or not math.isfinite(self.correlation)
            or not -1.0 <= self.correlation <= 1.0
        ):
            raise ValueError("audio_evidence.correlation must be finite within [-1, 1]")
        if not isinstance(self.status, AudioCorrelationEvidenceStatus):
            raise TypeError("audio_evidence.status must be AudioCorrelationEvidenceStatus")


@dataclass(frozen=True, slots=True)
class AudioCorrelationSyncResult:
    toolchain: FFmpegToolchainIdentity
    evidence: AudioCorrelationEvidence
    hypothesis: SyncHypothesis

    def __post_init__(self) -> None:
        if not isinstance(self.toolchain, FFmpegToolchainIdentity):
            raise TypeError("audio_sync_result.toolchain must be FFmpegToolchainIdentity")
        if not isinstance(self.evidence, AudioCorrelationEvidence):
            raise TypeError("audio_sync_result.evidence must be AudioCorrelationEvidence")
        if not isinstance(self.hypothesis, SyncHypothesis):
            raise TypeError("audio_sync_result.hypothesis must be SyncHypothesis")


def _max_samples(policy: AudioCorrelationPolicy) -> int:
    return (
        policy.sample_rate_hz * policy.window_duration_us
        + _MICROSECONDS_PER_SECOND
        - 1
    ) // _MICROSECONDS_PER_SECOND


def _max_lag_samples(policy: AudioCorrelationPolicy) -> int:
    return policy.sample_rate_hz * policy.max_lag_us // _MICROSECONDS_PER_SECOND


def _minimum_overlap_samples(policy: AudioCorrelationPolicy) -> int:
    return (
        policy.sample_rate_hz * policy.minimum_overlap_us
        + _MICROSECONDS_PER_SECOND
        - 1
    ) // _MICROSECONDS_PER_SECOND


def _duration_seconds_text(duration_us: int) -> str:
    seconds, microseconds = divmod(duration_us, _MICROSECONDS_PER_SECOND)
    return f"{seconds}.{microseconds:06d}"


def _pcm_command(
    source_path: Path,
    policy: AudioCorrelationPolicy,
    toolchain: FFmpegToolchain,
) -> list[str]:
    return [
        toolchain.ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        _duration_seconds_text(policy.window_duration_us),
        "-ac",
        "1",
        "-ar",
        str(policy.sample_rate_hz),
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]


def _verified_source_path(video: VideoObservation, source_path: Path) -> Path:
    resolved = source_path.expanduser().resolve(strict=True)
    content_hash = hash_file_content(resolved)
    if (
        content_hash.sha256 != video.asset.sha256
        or content_hash.byte_length != video.asset.byte_length
    ):
        raise ValueError("audio synchronization source bytes do not match the video asset")
    return resolved


def _extract_pcm_samples(
    source_path: Path,
    policy: AudioCorrelationPolicy,
    toolchain: FFmpegToolchain,
) -> tuple[int, ...]:
    try:
        completed = subprocess.run(
            _pcm_command(source_path, policy, toolchain),
            check=False,
            capture_output=True,
            timeout=toolchain.timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise FFmpegExecutionError(f"failed to execute {toolchain.ffmpeg_path!r}: {exc}") from exc

    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace").strip()
        if len(stderr) > 2_000:
            stderr = stderr[-2_000:]
        raise FFmpegExecutionError(
            f"{toolchain.ffmpeg_path!r} audio extraction exited with "
            f"code {completed.returncode}: {stderr}"
        )
    if not isinstance(completed.stdout, bytes):
        raise FFmpegExecutionError("ffmpeg audio extraction did not return binary PCM")
    if len(completed.stdout) % 2:
        raise FFmpegExecutionError("ffmpeg audio extraction returned misaligned s16le PCM")

    samples = tuple(
        item[0] for item in struct.iter_unpack("<h", completed.stdout)
    )
    bounded = samples[: _max_samples(policy)]
    if not bounded:
        raise FFmpegExecutionError("ffmpeg audio extraction returned no PCM samples")
    return bounded


def _normalized_correlation(
    first: tuple[int, ...],
    second: tuple[int, ...],
    *,
    first_start: int,
    second_start: int,
    length: int,
) -> float | None:
    first_sum = sum(first[first_start : first_start + length])
    second_sum = sum(second[second_start : second_start + length])
    first_mean = first_sum / length
    second_mean = second_sum / length

    covariance = math.fsum(
        (first[first_start + index] - first_mean)
        * (second[second_start + index] - second_mean)
        for index in range(length)
    )
    first_energy = math.fsum(
        (first[first_start + index] - first_mean) ** 2
        for index in range(length)
    )
    second_energy = math.fsum(
        (second[second_start + index] - second_mean) ** 2
        for index in range(length)
    )
    if first_energy <= 0.0 or second_energy <= 0.0:
        return None

    correlation = covariance / math.sqrt(first_energy * second_energy)
    return max(-1.0, min(1.0, correlation))


def _correlate_samples(
    first: tuple[int, ...],
    second: tuple[int, ...],
    policy: AudioCorrelationPolicy,
) -> AudioCorrelationEvidence:
    max_lag = _max_lag_samples(policy)
    minimum_overlap = _minimum_overlap_samples(policy)
    candidates: list[tuple[int, int, float]] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            first_start = 0
            second_start = lag
            overlap = min(len(first), len(second) - lag)
        else:
            first_start = -lag
            second_start = 0
            overlap = min(len(first) + lag, len(second))

        if overlap < minimum_overlap:
            continue
        correlation = _normalized_correlation(
            first,
            second,
            first_start=first_start,
            second_start=second_start,
            length=overlap,
        )
        if correlation is not None:
            candidates.append((lag, overlap, correlation))

    if not candidates:
        return AudioCorrelationEvidence(
            sample_rate_hz=policy.sample_rate_hz,
            lag_samples=0,
            overlap_samples=0,
            correlation=0.0,
            status=AudioCorrelationEvidenceStatus.UNAVAILABLE,
        )

    best_correlation = max(item[2] for item in candidates)
    tied = tuple(item for item in candidates if item[2] == best_correlation)
    selected = min(tied, key=lambda item: (abs(item[0]), item[0]))
    lag_samples, overlap_samples, correlation = selected

    if correlation < policy.minimum_correlation:
        status = AudioCorrelationEvidenceStatus.WEAK
    elif len(tied) > 1:
        status = AudioCorrelationEvidenceStatus.AMBIGUOUS
    else:
        status = AudioCorrelationEvidenceStatus.ACCEPTED

    return AudioCorrelationEvidence(
        sample_rate_hz=policy.sample_rate_hz,
        lag_samples=lag_samples,
        overlap_samples=overlap_samples,
        correlation=float(correlation),
        status=status,
    )


def _lag_offset_us(lag_samples: int, sample_rate_hz: int) -> int:
    value = (
        Decimal(lag_samples)
        * Decimal(_MICROSECONDS_PER_SECOND)
        / Decimal(sample_rate_hz)
    )
    return int(value.to_integral_value(rounding=ROUND_HALF_EVEN))


def _hypothesis_from_evidence(
    request: AudioCorrelationSyncRequest,
    evidence: AudioCorrelationEvidence,
) -> SyncHypothesis:
    if evidence.status is AudioCorrelationEvidenceStatus.ACCEPTED:
        disposition = SyncHypothesisDisposition.SUPPORTED
        offset_us: int | None = _lag_offset_us(
            evidence.lag_samples,
            evidence.sample_rate_hz,
        )
    else:
        disposition = SyncHypothesisDisposition.UNRESOLVED
        offset_us = None

    return SyncHypothesis(
        observation_id1=request.video1.observation_id,
        observation_id2=request.video2.observation_id,
        disposition=disposition,
        offset_us=offset_us,
        evidence_refs=(request.evidence_ref,),
    )


class FFmpegAudioCorrelationSynchronizer:
    """Portable bounded audio-correlation reference using the retained FFmpeg toolchain."""

    def __init__(self, toolchain: FFmpegToolchain) -> None:
        if not isinstance(toolchain, FFmpegToolchain):
            raise TypeError("audio_synchronizer.toolchain must be FFmpegToolchain")
        self._toolchain = toolchain

    def synchronize(
        self,
        request: AudioCorrelationSyncRequest,
    ) -> AudioCorrelationSyncResult:
        if not isinstance(request, AudioCorrelationSyncRequest):
            raise TypeError("request must be AudioCorrelationSyncRequest")

        source1 = _verified_source_path(request.video1, request.source_path1)
        source2 = _verified_source_path(request.video2, request.source_path2)
        identity = inspect_toolchain(self._toolchain)

        try:
            samples1 = _extract_pcm_samples(source1, request.policy, self._toolchain)
            samples2 = _extract_pcm_samples(source2, request.policy, self._toolchain)
            evidence = _correlate_samples(samples1, samples2, request.policy)
        except FFmpegExecutionError:
            evidence = AudioCorrelationEvidence(
                sample_rate_hz=request.policy.sample_rate_hz,
                lag_samples=0,
                overlap_samples=0,
                correlation=0.0,
                status=AudioCorrelationEvidenceStatus.UNAVAILABLE,
            )

        return AudioCorrelationSyncResult(
            toolchain=identity,
            evidence=evidence,
            hypothesis=_hypothesis_from_evidence(request, evidence),
        )
