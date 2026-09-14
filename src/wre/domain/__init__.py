"""Core solver-independent domain models."""

from wre.domain.cameras import (
    Camera,
    CameraId,
    ImageDimensions,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.domain.fragments import LocalFrameId, SpatialFragment, SpatialFragmentId
from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
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
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)

__all__ = [
    "Camera",
    "CameraId",
    "CaptureTimeInterpretation",
    "CaptureTimeInterpretationStatus",
    "DerivedArtifactProvenance",
    "GpsInterpretationStatus",
    "GpsMetadataInterpretation",
    "ImageDimensions",
    "ImageObservation",
    "LocalFrameId",
    "MediaAssetRef",
    "Observation",
    "ObservationId",
    "ObservationKind",
    "ObservationMetadata",
    "ObservationMetadataInterpretation",
    "ProducerRef",
    "RawMetadataEntry",
    "ReconstructionRun",
    "ReconstructionRunId",
    "Sha256Digest",
    "SourceId",
    "SourceRef",
    "SpatialFragment",
    "SpatialFragmentId",
    "VideoFrameObservation",
]
