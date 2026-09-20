from __future__ import annotations

import math
import struct
import wave
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import wre.synchronization.audio as audio_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import (
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoObservation,
)
from wre.domain.temporal_groups import SyncHypothesisDisposition
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.keyframes import (
    FFmpegExecutionError,
    FFmpegToolchain,
    FFmpegToolchainIdentity,
)
from wre.synchronization import (
    AudioCorrelationEvidenceStatus,
    AudioCorrelationPolicy,
    AudioCorrelationSyncRequest,
    FFmpegAudioCorrelationSynchronizer,
)


def _ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact:audio-correlation"),
        artifact_kind=ArtifactKind("sync.audio_correlation"),
    )


def _policy(
    *,
    sample_rate_hz: int = 1_000,
    window_duration_us: int = 200_000,
    max_lag_us: int = 50_000,
    minimum_overlap_us: int = 100_000,
    minimum_correlation: float = 0.8,
) -> AudioCorrelationPolicy:
    return AudioCorrelationPolicy(
        sample_rate_hz=sample_rate_hz,
        window_duration_us=window_duration_us,
        max_lag_us=max_lag_us,
        minimum_overlap_us=minimum_overlap_us,
        minimum_correlation=minimum_correlation,
    )


def _write_bytes(path: Path, payload: bytes) -> MediaAssetRef:
    path.write_bytes(payload)
    content = hash_file_content(path)
    return MediaAssetRef(
        uri=path.resolve().as_uri(),
        sha256=content.sha256,
        byte_length=content.byte_length,
        mime_type="video/mp4",
    )


def _video(observation_id: str, asset: MediaAssetRef) -> VideoObservation:
    return VideoObservation(
        observation_id=ObservationId(observation_id),
        asset=asset,
        source=SourceRef(source_id=SourceId("fixture:audio")),
        received_at=datetime(2026, 9, 20, 15, 0, tzinfo=UTC),
        captured_at=None,
    )


def _request(
    path1: Path,
    path2: Path,
    *,
    policy: AudioCorrelationPolicy | None = None,
) -> AudioCorrelationSyncRequest:
    asset1 = MediaAssetRef(
        uri=path1.resolve().as_uri(),
        sha256=Sha256Digest(hash_file_content(path1).sha256.value),
        byte_length=path1.stat().st_size,
        mime_type="audio/wav",
    )
    asset2 = MediaAssetRef(
        uri=path2.resolve().as_uri(),
        sha256=Sha256Digest(hash_file_content(path2).sha256.value),
        byte_length=path2.stat().st_size,
        mime_type="audio/wav",
    )
    return AudioCorrelationSyncRequest(
        video1=_video("obs:a", asset1),
        source_path1=path1,
        video2=_video("obs:b", asset2),
        source_path2=path2,
        policy=policy or _policy(),
        evidence_ref=_ref(),
    )


def _write_wav(path: Path, samples: tuple[int, ...], sample_rate_hz: int) -> None:
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate_hz)
        stream.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))


def _pattern(length: int) -> tuple[int, ...]:
    value = 17
    samples: list[int] = []
    for _ in range(length):
        value = (value * 1103515245 + 12345) & 0x7FFFFFFF
        samples.append((value % 20_001) - 10_000)
    return tuple(samples)


def _shifted_pair(length: int, lag: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    base = _pattern(length + abs(lag))
    if lag > 0:
        return base[:length], (0,) * lag + base[: length - lag]
    if lag < 0:
        shift = -lag
        return (0,) * shift + base[: length - shift], base[:length]
    return base[:length], base[:length]


def test_audio_policy_is_exact_immutable_bounded_contract() -> None:
    policy = _policy()

    assert tuple(field.name for field in fields(AudioCorrelationPolicy)) == (
        "sample_rate_hz",
        "window_duration_us",
        "max_lag_us",
        "minimum_overlap_us",
        "minimum_correlation",
    )
    with pytest.raises(FrozenInstanceError):
        policy.sample_rate_hz = 2_000  # type: ignore[misc]

    for field_name in (
        "sample_rate_hz",
        "window_duration_us",
        "max_lag_us",
        "minimum_overlap_us",
    ):
        kwargs = {
            "sample_rate_hz": 1_000,
            "window_duration_us": 200_000,
            "max_lag_us": 50_000,
            "minimum_overlap_us": 100_000,
            "minimum_correlation": 0.8,
        }
        kwargs[field_name] = 0
        with pytest.raises(ValueError, match="positive integer"):
            AudioCorrelationPolicy(**kwargs)

    with pytest.raises(ValueError, match="max_lag_us"):
        _policy(max_lag_us=300_000)
    with pytest.raises(ValueError, match="minimum_overlap_us"):
        _policy(minimum_overlap_us=300_000)
    with pytest.raises(ValueError, match="finite float"):
        _policy(minimum_correlation=cast(Any, True))
    with pytest.raises(ValueError, match="finite float"):
        _policy(minimum_correlation=float("nan"))
    with pytest.raises(ValueError, match="finite float"):
        _policy(minimum_correlation=0.0)
    with pytest.raises(ValueError, match="bounded reference"):
        _policy(
            sample_rate_hz=100_000,
            window_duration_us=1_000_000,
            max_lag_us=1_000_000,
            minimum_overlap_us=1,
        )


def test_audio_request_is_exact_immutable_canonical_contract(tmp_path: Path) -> None:
    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    asset1 = _write_bytes(path1, b"a")
    asset2 = _write_bytes(path2, b"b")
    request = AudioCorrelationSyncRequest(
        video1=_video("obs:a", asset1),
        source_path1=path1,
        video2=_video("obs:b", asset2),
        source_path2=path2,
        policy=_policy(),
        evidence_ref=_ref(),
    )

    assert tuple(field.name for field in fields(AudioCorrelationSyncRequest)) == (
        "video1",
        "source_path1",
        "video2",
        "source_path2",
        "policy",
        "evidence_ref",
    )
    with pytest.raises(FrozenInstanceError):
        request.evidence_ref = _ref()  # type: ignore[misc]
    with pytest.raises(ValueError, match="canonically ordered"):
        AudioCorrelationSyncRequest(
            video1=_video("obs:b", asset2),
            source_path1=path2,
            video2=_video("obs:a", asset1),
            source_path2=path1,
            policy=_policy(),
            evidence_ref=_ref(),
        )
    with pytest.raises(TypeError, match="source_path1"):
        AudioCorrelationSyncRequest(
            video1=_video("obs:a", asset1),
            source_path1=cast(Any, str(path1)),
            video2=_video("obs:b", asset2),
            source_path2=path2,
            policy=_policy(),
            evidence_ref=_ref(),
        )


def test_source_mismatch_fails_before_toolchain_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    asset1 = _write_bytes(path1, b"a")
    asset2 = _write_bytes(path2, b"b")
    request = AudioCorrelationSyncRequest(
        video1=_video("obs:a", asset1),
        source_path1=path1,
        video2=_video("obs:b", asset2),
        source_path2=path2,
        policy=_policy(),
        evidence_ref=_ref(),
    )
    path1.write_bytes(b"changed")

    def _unexpected(_: object) -> FFmpegToolchainIdentity:
        raise AssertionError("toolchain inspection ran before source verification")

    monkeypatch.setattr(audio_module, "inspect_toolchain", _unexpected)
    with pytest.raises(ValueError, match="source bytes"):
        FFmpegAudioCorrelationSynchronizer(FFmpegToolchain()).synchronize(request)


def test_pcm_command_is_canonical_and_duration_bounded() -> None:
    policy = _policy(
        sample_rate_hz=8_000,
        window_duration_us=1_234_567,
        max_lag_us=100_000,
        minimum_overlap_us=100_000,
    )
    command = audio_module._pcm_command(
        Path("/tmp/source.mp4"),
        policy,
        FFmpegToolchain(ffmpeg_path="ffmpeg-custom"),
    )

    assert command == [
        "ffmpeg-custom",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-i",
        "/tmp/source.mp4",
        "-map",
        "0:a:0",
        "-vn",
        "-t",
        "1.234567",
        "-ac",
        "1",
        "-ar",
        "8000",
        "-c:a",
        "pcm_s16le",
        "-f",
        "s16le",
        "pipe:1",
    ]


def test_reference_ffmpeg_pcm_extraction_is_deterministic_and_bounded(tmp_path: Path) -> None:
    path = tmp_path / "source.wav"
    samples = tuple((index % 31) * 500 - 7_500 for index in range(300))
    _write_wav(path, samples, 1_000)
    policy = _policy(window_duration_us=100_000, max_lag_us=20_000, minimum_overlap_us=50_000)

    first = audio_module._extract_pcm_samples(path, policy, FFmpegToolchain())
    second = audio_module._extract_pcm_samples(path, policy, FFmpegToolchain())

    assert first == second
    assert 95 <= len(first) <= 100


def test_exact_ffmpeg_identity_is_enforced_before_decode(tmp_path: Path) -> None:
    path1 = tmp_path / "a.wav"
    path2 = tmp_path / "b.wav"
    samples = _pattern(200)
    _write_wav(path1, samples, 1_000)
    _write_wav(path2, samples, 1_000)

    synchronizer = FFmpegAudioCorrelationSynchronizer(
        FFmpegToolchain(required_version="definitely-not-the-retained-version")
    )
    with pytest.raises(FFmpegExecutionError, match="version mismatch"):
        synchronizer.synchronize(_request(path1, path2))


@pytest.mark.parametrize(("lag_samples", "expected_us"), [(-20, -20_000), (0, 0), (20, 20_000)])
def test_known_lag_pcm_maps_to_exact_sync_sign_convention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lag_samples: int,
    expected_us: int,
) -> None:
    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    path1.write_bytes(b"a")
    path2.write_bytes(b"b")
    request = _request(path1, path2)
    first, second = _shifted_pair(200, lag_samples)

    monkeypatch.setattr(
        audio_module,
        "inspect_toolchain",
        lambda _: FFmpegToolchainIdentity("6.1.1-3ubuntu5", "6.1.1-3ubuntu5"),
    )

    def _extract(path: Path, *_: object) -> tuple[int, ...]:
        return first if path == path1.resolve() else second

    monkeypatch.setattr(audio_module, "_extract_pcm_samples", _extract)
    result = FFmpegAudioCorrelationSynchronizer(FFmpegToolchain()).synchronize(request)

    assert result.evidence.status is AudioCorrelationEvidenceStatus.ACCEPTED
    assert result.evidence.lag_samples == lag_samples
    assert result.hypothesis.disposition is SyncHypothesisDisposition.SUPPORTED
    assert result.hypothesis.offset_us == expected_us
    assert result.hypothesis.evidence_refs == (_ref(),)


def test_exact_tie_is_ambiguous_with_canonical_smallest_absolute_lag() -> None:
    samples = tuple(10_000 if index % 2 == 0 else -10_000 for index in range(200))
    evidence = audio_module._correlate_samples(samples, samples, _policy(minimum_correlation=0.9))

    assert evidence.status is AudioCorrelationEvidenceStatus.AMBIGUOUS
    assert evidence.lag_samples == 0
    assert evidence.correlation == pytest.approx(1.0)


def test_weak_correlation_stays_unresolved() -> None:
    first = _pattern(200)
    second = tuple(reversed(first))
    policy = _policy(minimum_correlation=0.999999)
    evidence = audio_module._correlate_samples(first, second, policy)

    assert evidence.status in {
        AudioCorrelationEvidenceStatus.WEAK,
        AudioCorrelationEvidenceStatus.AMBIGUOUS,
    }
    if evidence.status is AudioCorrelationEvidenceStatus.WEAK:
        assert evidence.correlation < policy.minimum_correlation


def test_silence_and_insufficient_overlap_are_unavailable() -> None:
    silence = (0,) * 200
    silent = audio_module._correlate_samples(silence, silence, _policy())
    too_short = audio_module._correlate_samples(
        _pattern(20),
        _pattern(20),
        _policy(minimum_overlap_us=100_000),
    )

    assert silent.status is AudioCorrelationEvidenceStatus.UNAVAILABLE
    assert too_short.status is AudioCorrelationEvidenceStatus.UNAVAILABLE
    assert math.isfinite(silent.correlation)
    assert math.isfinite(too_short.correlation)


def test_extraction_failure_maps_to_unresolved_without_other_fallback(tmp_path: Path) -> None:
    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    path1.write_bytes(b"not media")
    path2.write_bytes(b"also not media")

    result = FFmpegAudioCorrelationSynchronizer(FFmpegToolchain()).synchronize(
        _request(path1, path2)
    )

    assert result.evidence.status is AudioCorrelationEvidenceStatus.UNAVAILABLE
    assert result.hypothesis.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert result.hypothesis.offset_us is None


def test_audio_adapter_never_emits_contradicted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path1 = tmp_path / "a.bin"
    path2 = tmp_path / "b.bin"
    path1.write_bytes(b"a")
    path2.write_bytes(b"b")
    request = _request(path1, path2, policy=_policy(minimum_correlation=1.0))

    monkeypatch.setattr(
        audio_module,
        "inspect_toolchain",
        lambda _: FFmpegToolchainIdentity("6.1.1-3ubuntu5", "6.1.1-3ubuntu5"),
    )
    monkeypatch.setattr(
        audio_module,
        "_extract_pcm_samples",
        lambda *_: _pattern(200),
    )
    result = FFmpegAudioCorrelationSynchronizer(FFmpegToolchain()).synchronize(request)

    assert result.hypothesis.disposition is not SyncHypothesisDisposition.CONTRADICTED


def test_audio_adapter_adds_no_dsp_or_network_dependency_surface() -> None:
    forbidden = {
        "numpy",
        "scipy",
        "librosa",
        "torch",
        "requests",
        "socket",
        "TemporalGroup",
    }
    assert forbidden.isdisjoint(audio_module.__dict__)
