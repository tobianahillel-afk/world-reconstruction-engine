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

__all__ = [
    "AudioCorrelationEvidence",
    "AudioCorrelationEvidenceStatus",
    "AudioCorrelationPolicy",
    "AudioCorrelationSyncRequest",
    "AudioCorrelationSyncResult",
    "CaptureTimeSyncAdapterInput",
    "FFmpegAudioCorrelationSynchronizer",
    "SameVideoFrameSyncAdapterInput",
    "adapt_capture_time_sync",
    "adapt_same_video_frame_sync",
]
