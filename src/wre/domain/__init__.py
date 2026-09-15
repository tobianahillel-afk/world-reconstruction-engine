"""Core solver-independent domain models."""

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.cameras import (
    Camera,
    CameraId,
    ImageDimensions,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.domain.estimated_geometry import (
    CameraCalibrationEstimate,
    CameraCalibrationEstimateId,
    CameraPoseEstimate,
    EstimatedPoint3DId,
    EstimatedTrackElement,
    LocalScaleStatus,
    Point3DEstimate,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
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
    VideoObservation,
)
from wre.domain.projects import SceneProject, SceneProjectId
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)

__all__ = [
    "ArtifactId",
    "ArtifactKind",
    "ArtifactRef",
    "Camera",
    "CameraCalibrationEstimate",
    "CameraCalibrationEstimateId",
    "CameraId",
    "CameraPoseEstimate",
    "CaptureTimeInterpretation",
    "CaptureTimeInterpretationStatus",
    "DerivedArtifactProvenance",
    "EstimatedPoint3DId",
    "EstimatedTrackElement",
    "GpsInterpretationStatus",
    "GpsMetadataInterpretation",
    "ImageDimensions",
    "ImageObservation",
    "LocalFrameId",
    "LocalScaleStatus",
    "MediaAssetRef",
    "Observation",
    "ObservationId",
    "ObservationKind",
    "ObservationMetadata",
    "ObservationMetadataInterpretation",
    "Point3DEstimate",
    "ProducerRef",
    "RawMetadataEntry",
    "ReconstructionRun",
    "ReconstructionRunId",
    "SceneProject",
    "SceneProjectId",
    "Sha256Digest",
    "SourceId",
    "SourceRef",
    "SparseReconstructionEstimate",
    "SparseReconstructionEstimateId",
    "SpatialFragment",
    "SpatialFragmentId",
    "VideoFrameObservation",
    "VideoObservation",
]
