from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import wre.synchronization as synchronization_module
import wre.synchronization.audio as audio_module
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
    VideoObservation,
)
from wre.domain.temporal_groups import SyncHypothesis, SyncHypothesisDisposition
from wre.synchronization import (
    AudioCorrelationEvidence,
    AudioCorrelationEvidenceStatus,
    AudioCorrelationPolicy,
    AudioCorrelationSyncRequest,
    CaptureTimeSyncAdapterInput,
    VisualEventCandidateMultiplicity,
    VisualEventSyncCandidate,
    VisualEventSyncPolicy,
    adapt_capture_time_sync,
    adapt_visual_event_sync,
)

OBSERVATION_ID1 = ObservationId("obs:fixture:a")
OBSERVATION_ID2 = ObservationId("obs:fixture:b")

METADATA_REF = ArtifactRef(
    artifact_id=ArtifactId("artifact:fixture-metadata"),
    artifact_kind=ArtifactKind("sync.capture_time"),
)
AUDIO_REF = ArtifactRef(
    artifact_id=ArtifactId("artifact:fixture-audio"),
    artifact_kind=ArtifactKind("sync.audio_correlation"),
)
VISUAL_REF = ArtifactRef(
    artifact_id=ArtifactId("artifact:fixture-visual"),
    artifact_kind=ArtifactKind("sync.visual_event"),
)
CONTRADICTION_REF = ArtifactRef(
    artifact_id=ArtifactId("artifact:fixture-contradiction"),
    artifact_kind=ArtifactKind("sync.explicit_contradiction"),
)


def _metadata_interpretation(
    observation_id: ObservationId,
    instant: datetime,
) -> ObservationMetadataInterpretation:
    return ObservationMetadataInterpretation(
        observation_id=observation_id,
        gps=GpsMetadataInterpretation(status=GpsInterpretationStatus.ABSENT),
        capture_time=CaptureTimeInterpretation(
            status=CaptureTimeInterpretationStatus.RESOLVED,
            raw_datetime="2026:09:20 12:00:00",
            raw_offset="+00:00",
            instant=instant,
            evidence_keys=("EXIF DateTimeOriginal", "EXIF OffsetTimeOriginal"),
        ),
    )


def _metadata_hypothesis(offset_us: int) -> SyncHypothesis:
    base = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
    result = adapt_capture_time_sync(
        CaptureTimeSyncAdapterInput(
            interpretation1=_metadata_interpretation(OBSERVATION_ID1, base),
            interpretation2=_metadata_interpretation(
                OBSERVATION_ID2,
                base + timedelta(microseconds=offset_us),
            ),
            evidence_refs=(METADATA_REF,),
        )
    )
    assert result is not None
    return result


def _visual_hypothesis(
    offset_us: int,
    *,
    multiplicity: VisualEventCandidateMultiplicity = VisualEventCandidateMultiplicity.UNIQUE,
    support_count: int = 5,
    residual_us: int = 100,
) -> SyncHypothesis:
    return adapt_visual_event_sync(
        VisualEventSyncCandidate(
            observation_id1=OBSERVATION_ID1,
            observation_id2=OBSERVATION_ID2,
            candidate_offset_us=offset_us,
            residual_us=residual_us,
            support_count=support_count,
            multiplicity=multiplicity,
            evidence_refs=(VISUAL_REF,),
        ),
        VisualEventSyncPolicy(
            minimum_support_count=3,
            maximum_residual_us=1_000,
        ),
    )


def _video(observation_id: ObservationId, fill: str) -> VideoObservation:
    return VideoObservation(
        observation_id=observation_id,
        asset=MediaAssetRef(
            uri=f"fixture://{observation_id.value}.mp4",
            sha256=Sha256Digest(fill * 64),
            byte_length=1,
            mime_type="video/mp4",
        ),
        source=SourceRef(source_id=SourceId("fixture:sync")),
        received_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        captured_at=None,
    )


def _unresolved_audio_hypothesis() -> SyncHypothesis:
    request = AudioCorrelationSyncRequest(
        video1=_video(OBSERVATION_ID1, "a"),
        source_path1=Path("fixture-a.mp4"),
        video2=_video(OBSERVATION_ID2, "b"),
        source_path2=Path("fixture-b.mp4"),
        policy=AudioCorrelationPolicy(
            sample_rate_hz=1_000,
            window_duration_us=200_000,
            max_lag_us=50_000,
            minimum_overlap_us=100_000,
            minimum_correlation=0.8,
        ),
        evidence_ref=AUDIO_REF,
    )
    evidence = AudioCorrelationEvidence(
        sample_rate_hz=1_000,
        lag_samples=0,
        overlap_samples=0,
        correlation=0.0,
        status=AudioCorrelationEvidenceStatus.UNAVAILABLE,
    )
    return audio_module._hypothesis_from_evidence(request, evidence)


def _explicit_contradiction() -> SyncHypothesis:
    return SyncHypothesis(
        observation_id1=OBSERVATION_ID1,
        observation_id2=OBSERVATION_ID2,
        disposition=SyncHypothesisDisposition.CONTRADICTED,
        offset_us=None,
        evidence_refs=(CONTRADICTION_REF,),
    )


def test_agreement_keeps_metadata_and_visual_support_as_separate_records() -> None:
    metadata = _metadata_hypothesis(10_000)
    visual = _visual_hypothesis(10_000)
    hypotheses = (metadata, visual)

    assert tuple(item.disposition for item in hypotheses) == (
        SyncHypothesisDisposition.SUPPORTED,
        SyncHypothesisDisposition.SUPPORTED,
    )
    assert tuple(item.offset_us for item in hypotheses) == (10_000, 10_000)
    assert metadata.evidence_refs == (METADATA_REF,)
    assert visual.evidence_refs == (VISUAL_REF,)
    assert metadata is not visual


def test_ambiguous_visual_evidence_stays_unresolved_beside_supported_metadata() -> None:
    metadata = _metadata_hypothesis(10_000)
    visual = _visual_hypothesis(
        10_000,
        multiplicity=VisualEventCandidateMultiplicity.AMBIGUOUS,
    )

    assert metadata.disposition is SyncHypothesisDisposition.SUPPORTED
    assert metadata.offset_us == 10_000
    assert visual.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert visual.offset_us is None
    assert visual.evidence_refs == (VISUAL_REF,)


def test_unavailable_audio_evidence_stays_unresolved_with_audio_only_provenance() -> None:
    audio = _unresolved_audio_hypothesis()

    assert audio.disposition is SyncHypothesisDisposition.UNRESOLVED
    assert audio.offset_us is None
    assert audio.evidence_refs == (AUDIO_REF,)


def test_supported_offset_disagreement_does_not_manufacture_contradiction_or_winner() -> None:
    metadata = _metadata_hypothesis(10_000)
    visual = _visual_hypothesis(25_000)
    hypotheses = (metadata, visual)

    assert tuple(item.disposition for item in hypotheses) == (
        SyncHypothesisDisposition.SUPPORTED,
        SyncHypothesisDisposition.SUPPORTED,
    )
    assert tuple(item.offset_us for item in hypotheses) == (10_000, 25_000)
    assert all(
        item.disposition is not SyncHypothesisDisposition.CONTRADICTED for item in hypotheses
    )
    assert metadata.evidence_refs == (METADATA_REF,)
    assert visual.evidence_refs == (VISUAL_REF,)


def test_explicit_contradiction_coexists_without_overwriting_other_evidence() -> None:
    metadata = _metadata_hypothesis(10_000)
    audio = _unresolved_audio_hypothesis()
    contradiction = _explicit_contradiction()
    hypotheses = (metadata, audio, contradiction)

    assert tuple(item.disposition for item in hypotheses) == (
        SyncHypothesisDisposition.SUPPORTED,
        SyncHypothesisDisposition.UNRESOLVED,
        SyncHypothesisDisposition.CONTRADICTED,
    )
    assert tuple(item.offset_us for item in hypotheses) == (10_000, None, None)
    assert tuple(item.evidence_refs for item in hypotheses) == (
        (METADATA_REF,),
        (AUDIO_REF,),
        (CONTRADICTION_REF,),
    )


def test_fixture_ordering_and_source_provenance_are_deterministic() -> None:
    hypotheses = (
        _metadata_hypothesis(10_000),
        _unresolved_audio_hypothesis(),
        _visual_hypothesis(
            10_000,
            multiplicity=VisualEventCandidateMultiplicity.AMBIGUOUS,
        ),
        _explicit_contradiction(),
    )

    assert tuple(ref.artifact_kind.value for item in hypotheses for ref in item.evidence_refs) == (
        "sync.capture_time",
        "sync.audio_correlation",
        "sync.visual_event",
        "sync.explicit_contradiction",
    )
    assert all(len(item.evidence_refs) == 1 for item in hypotheses)


def test_sync_foundation_exposes_no_fusion_or_temporal_group_construction_surface() -> None:
    forbidden_symbols = {
        "TemporalGroup",
        "fuse_sync_hypotheses",
        "merge_sync_hypotheses",
        "resolve_sync_hypotheses",
        "select_sync_winner",
        "rank_sync_sources",
        "build_temporal_group",
    }

    assert forbidden_symbols.isdisjoint(synchronization_module.__dict__)
