from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.metadata import (
    CaptureTimeInterpretationStatus,
    ObservationMetadataInterpretation,
)
from wre.domain.observations import VideoFrameObservation
from wre.domain.temporal_groups import SyncHypothesis, SyncHypothesisDisposition


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


def _validate_evidence_refs(evidence_refs: object, *, context: str) -> None:
    if not isinstance(evidence_refs, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    if not evidence_refs:
        raise ValueError(f"{context} must not be empty")
    if any(not isinstance(ref, ArtifactRef) for ref in evidence_refs):
        raise TypeError(f"{context} must contain only ArtifactRef values")

    refs = evidence_refs
    keys = tuple(_artifact_ref_key(ref) for ref in refs)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{context} must not contain duplicates")
    if keys != tuple(sorted(keys)):
        raise ValueError(f"{context} must use canonical artifact order")


@dataclass(frozen=True, slots=True)
class CaptureTimeSyncAdapterInput:
    """Pair of already-interpreted capture-time records plus exact provenance evidence."""

    interpretation1: ObservationMetadataInterpretation
    interpretation2: ObservationMetadataInterpretation
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.interpretation1, ObservationMetadataInterpretation):
            raise TypeError(
                "capture_time_sync.interpretation1 must be ObservationMetadataInterpretation"
            )
        if not isinstance(self.interpretation2, ObservationMetadataInterpretation):
            raise TypeError(
                "capture_time_sync.interpretation2 must be ObservationMetadataInterpretation"
            )

        first = self.interpretation1.observation_id.value
        second = self.interpretation2.observation_id.value
        if first >= second:
            raise ValueError(
                "capture_time_sync observation IDs must be distinct and canonically ordered"
            )

        _validate_evidence_refs(
            self.evidence_refs,
            context="capture_time_sync.evidence_refs",
        )


@dataclass(frozen=True, slots=True)
class SameVideoFrameSyncAdapterInput:
    """Two frames sharing one exact parent video timing origin."""

    frame1: VideoFrameObservation
    frame2: VideoFrameObservation
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.frame1, VideoFrameObservation):
            raise TypeError("same_video_frame_sync.frame1 must be VideoFrameObservation")
        if not isinstance(self.frame2, VideoFrameObservation):
            raise TypeError("same_video_frame_sync.frame2 must be VideoFrameObservation")
        if self.frame1.observation_id.value >= self.frame2.observation_id.value:
            raise ValueError(
                "same_video_frame_sync observation IDs must be distinct and canonically ordered"
            )
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("same_video_frame_sync.evidence_ref must be ArtifactRef")
        if self.frame1.video_asset != self.frame2.video_asset:
            raise ValueError(
                "same_video_frame_sync frames must reference the exact same parent video_asset"
            )


def adapt_capture_time_sync(
    adapter_input: CaptureTimeSyncAdapterInput,
) -> SyncHypothesis | None:
    """Adapt interpreted capture time without guessing missing time semantics."""

    if not isinstance(adapter_input, CaptureTimeSyncAdapterInput):
        raise TypeError("adapter_input must be CaptureTimeSyncAdapterInput")

    first = adapter_input.interpretation1
    second = adapter_input.interpretation2
    first_time = first.capture_time
    second_time = second.capture_time

    if (
        first_time.status is CaptureTimeInterpretationStatus.ABSENT
        and second_time.status is CaptureTimeInterpretationStatus.ABSENT
    ):
        return None

    if (
        first_time.status is CaptureTimeInterpretationStatus.RESOLVED
        and second_time.status is CaptureTimeInterpretationStatus.RESOLVED
    ):
        if first_time.instant is None or second_time.instant is None:
            raise ValueError("resolved capture-time interpretation must expose an instant")
        delta = second_time.instant - first_time.instant
        offset_us = delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds
        disposition = SyncHypothesisDisposition.SUPPORTED
    else:
        offset_us = None
        disposition = SyncHypothesisDisposition.UNRESOLVED

    return SyncHypothesis(
        observation_id1=first.observation_id,
        observation_id2=second.observation_id,
        disposition=disposition,
        offset_us=offset_us,
        evidence_refs=adapter_input.evidence_refs,
    )


def adapt_same_video_frame_sync(
    adapter_input: SameVideoFrameSyncAdapterInput,
) -> SyncHypothesis:
    """Adapt exact source-relative timing for two frames of the same parent video."""

    if not isinstance(adapter_input, SameVideoFrameSyncAdapterInput):
        raise TypeError("adapter_input must be SameVideoFrameSyncAdapterInput")

    first = adapter_input.frame1
    second = adapter_input.frame2
    return SyncHypothesis(
        observation_id1=first.observation_id,
        observation_id2=second.observation_id,
        disposition=SyncHypothesisDisposition.SUPPORTED,
        offset_us=second.frame_time_us - first.frame_time_us,
        evidence_refs=(adapter_input.evidence_ref,),
    )
