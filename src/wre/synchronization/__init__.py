"""Synchronization evidence adapters behind stable temporal contracts."""

from wre.synchronization.audio import (
    AudioCorrelationEvidence,
    AudioCorrelationEvidenceStatus,
    AudioCorrelationPolicy,
    AudioCorrelationSyncRequest,
    AudioCorrelationSyncResult,
    FFmpegAudioCorrelationSynchronizer,
)
from wre.synchronization.metadata_time import (
    CaptureTimeSyncAdapterInput,
    SameVideoFrameSyncAdapterInput,
    adapt_capture_time_sync,
    adapt_same_video_frame_sync,
)
from wre.synchronization.visual import (
    VisualEventCandidateMultiplicity,
    VisualEventSyncCandidate,
    VisualEventSyncPolicy,
    adapt_visual_event_sync,
)

__all__ = [
    "AudioCorrelationEvidence",
    "AudioCorrelationEvidenceStatus",
    "AudioCorrelationPolicy",
    "AudioCorrelationSyncRequest",
    "AudioCorrelationSyncResult",
    "CaptureTimeSyncAdapterInput",
    "FFmpegAudioCorrelationSynchronizer",
    "SameVideoFrameSyncAdapterInput",
    "VisualEventCandidateMultiplicity",
    "VisualEventSyncCandidate",
    "VisualEventSyncPolicy",
    "adapt_capture_time_sync",
    "adapt_same_video_frame_sync",
    "adapt_visual_event_sync",
]
