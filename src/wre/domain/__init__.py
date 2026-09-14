"""Core solver-independent domain models."""

from wre.domain.cameras import (
    Camera,
    CameraId,
    ImageDimensions,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    Observation,
    ObservationId,
    ObservationKind,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
)

__all__ = [
    "Camera",
    "CameraId",
    "ImageDimensions",
    "ImageObservation",
    "MediaAssetRef",
    "Observation",
    "ObservationId",
    "ObservationKind",
    "ObservationMetadata",
    "RawMetadataEntry",
    "Sha256Digest",
    "SourceId",
    "SourceRef",
    "VideoFrameObservation",
]
