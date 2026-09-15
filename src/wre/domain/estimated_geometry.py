from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.fragments import LocalFrameId
from wre.domain.observations import ObservationId
from wre.domain.runs import DerivedArtifactProvenance

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROTATION_TOLERANCE = 1e-6

Vector3 = tuple[float, float, float]
RotationMatrix3 = tuple[Vector3, Vector3, Vector3]


def _require_opaque_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{context} must be a finite number")
    return normalized


def _finite_vector3(value: Vector3, context: str) -> Vector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{context} must be an immutable 3-vector")
    return (
        _finite_float(value[0], context),
        _finite_float(value[1], context),
        _finite_float(value[2], context),
    )


def _finite_rotation_matrix(value: RotationMatrix3) -> RotationMatrix3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError("rotation_matrix must be an immutable 3x3 matrix")
    rows: RotationMatrix3 = (
        _finite_vector3(value[0], "rotation_matrix"),
        _finite_vector3(value[1], "rotation_matrix"),
        _finite_vector3(value[2], "rotation_matrix"),
    )

    for row in rows:
        norm_squared = sum(component * component for component in row)
        if abs(norm_squared - 1.0) > _ROTATION_TOLERANCE:
            raise ValueError("rotation_matrix rows must be unit length")
    for left, right in (
        (rows[0], rows[1]),
        (rows[0], rows[2]),
        (rows[1], rows[2]),
    ):
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        if abs(dot) > _ROTATION_TOLERANCE:
            raise ValueError("rotation_matrix rows must be orthogonal")

    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError("rotation_matrix must have determinant +1")
    return rows


@dataclass(frozen=True, slots=True, order=True)
class SparseReconstructionEstimateId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "sparse_reconstruction_estimate_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class CameraCalibrationEstimateId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "camera_calibration_estimate_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class EstimatedPoint3DId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "estimated_point3d_id")

    def __str__(self) -> str:
        return self.value


class LocalScaleStatus(StrEnum):
    """Whether metric scale has independent support in a local estimated frame."""

    UNRESOLVED = "unresolved"
    METRIC = "metric"


@dataclass(frozen=True, slots=True)
class CameraCalibrationEstimate:
    """Producer-labelled camera intrinsics estimate, not physical camera identity."""

    calibration_id: CameraCalibrationEstimateId
    projection_model: str
    width_px: int
    height_px: int
    parameters: tuple[float, ...]
    has_prior_focal_length: bool
    provenance: DerivedArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.projection_model, str) or not self.projection_model.strip():
            raise ValueError("projection_model must be non-blank")
        for name in ("width_px", "height_px"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.parameters, tuple) or not self.parameters:
            raise ValueError("parameters must be a non-empty immutable tuple")
        parameters = tuple(
            _finite_float(value, "camera calibration parameter")
            for value in self.parameters
        )
        object.__setattr__(self, "parameters", parameters)
        if not isinstance(self.has_prior_focal_length, bool):
            raise ValueError("has_prior_focal_length must be a boolean")
        if not isinstance(self.provenance, DerivedArtifactProvenance):
            raise ValueError("provenance must be DerivedArtifactProvenance")


@dataclass(frozen=True, slots=True)
class CameraPoseEstimate:
    """Rigid camera-from-local-frame pose estimate for one raw observation."""

    observation_id: ObservationId
    local_frame_id: LocalFrameId
    calibration_id: CameraCalibrationEstimateId
    rotation_matrix: RotationMatrix3
    translation_xyz: Vector3
    provenance: DerivedArtifactProvenance

    def __post_init__(self) -> None:
        rotation = _finite_rotation_matrix(self.rotation_matrix)
        translation = _finite_vector3(self.translation_xyz, "translation_xyz")
        object.__setattr__(self, "rotation_matrix", rotation)
        object.__setattr__(self, "translation_xyz", translation)
        if not isinstance(self.provenance, DerivedArtifactProvenance):
            raise ValueError("provenance must be DerivedArtifactProvenance")
        if self.provenance.source_observation_ids != (self.observation_id,):
            raise ValueError("camera pose provenance must contain exactly its observation")


@dataclass(frozen=True, slots=True, order=True)
class EstimatedTrackElement:
    observation_id: ObservationId
    feature_index: int

    def __post_init__(self) -> None:
        if isinstance(self.feature_index, bool) or not isinstance(self.feature_index, int):
            raise ValueError("feature_index must be an integer")
        if self.feature_index < 0:
            raise ValueError("feature_index must be non-negative")


@dataclass(frozen=True, slots=True)
class Point3DEstimate:
    point_id: EstimatedPoint3DId
    local_frame_id: LocalFrameId
    position_xyz: Vector3
    reprojection_error_px: float | None
    track: tuple[EstimatedTrackElement, ...]
    provenance: DerivedArtifactProvenance

    def __post_init__(self) -> None:
        position = _finite_vector3(self.position_xyz, "position_xyz")
        object.__setattr__(self, "position_xyz", position)
        if self.reprojection_error_px is not None:
            error = _finite_float(self.reprojection_error_px, "reprojection_error_px")
            if error < 0:
                raise ValueError("reprojection_error_px must be non-negative")
            object.__setattr__(self, "reprojection_error_px", error)
        if not isinstance(self.track, tuple) or not self.track:
            raise ValueError("point track must be a non-empty immutable tuple")
        if not all(isinstance(item, EstimatedTrackElement) for item in self.track):
            raise ValueError("point track must contain only EstimatedTrackElement values")
        canonical = tuple(
            sorted(
                self.track,
                key=lambda item: (item.observation_id.value, item.feature_index),
            )
        )
        if len(canonical) != len(set(canonical)):
            raise ValueError("point track cannot contain duplicate elements")
        observation_values = [item.observation_id.value for item in canonical]
        if len(observation_values) != len(set(observation_values)):
            raise ValueError("point track cannot contain multiple features from one observation")
        object.__setattr__(self, "track", canonical)
        expected_provenance = tuple(item.observation_id for item in canonical)
        if self.provenance.source_observation_ids != expected_provenance:
            raise ValueError("point provenance must exactly match its track observations")


@dataclass(frozen=True, slots=True)
class SparseReconstructionEstimate:
    """One solver-independent local sparse model before fragment acceptance."""

    estimate_id: SparseReconstructionEstimateId
    local_frame_id: LocalFrameId
    scale_status: LocalScaleStatus
    provenance: DerivedArtifactProvenance
    camera_calibrations: tuple[CameraCalibrationEstimate, ...]
    camera_poses: tuple[CameraPoseEstimate, ...]
    points3d: tuple[Point3DEstimate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.scale_status, LocalScaleStatus):
            raise ValueError("scale_status must be a LocalScaleStatus")
        if not isinstance(self.provenance, DerivedArtifactProvenance):
            raise ValueError("provenance must be DerivedArtifactProvenance")
        if not isinstance(self.camera_calibrations, tuple) or not self.camera_calibrations:
            raise ValueError("camera_calibrations must be a non-empty immutable tuple")
        if not isinstance(self.camera_poses, tuple) or not self.camera_poses:
            raise ValueError("camera_poses must be a non-empty immutable tuple")
        if not isinstance(self.points3d, tuple):
            raise ValueError("points3d must be an immutable tuple")
        if not all(
            isinstance(item, CameraCalibrationEstimate)
            for item in self.camera_calibrations
        ):
            raise ValueError("camera_calibrations contains an invalid value")
        if not all(isinstance(item, CameraPoseEstimate) for item in self.camera_poses):
            raise ValueError("camera_poses contains an invalid value")
        if not all(isinstance(item, Point3DEstimate) for item in self.points3d):
            raise ValueError("points3d contains an invalid value")

        calibrations = tuple(
            sorted(
                self.camera_calibrations,
                key=lambda item: item.calibration_id.value,
            )
        )
        poses = tuple(
            sorted(self.camera_poses, key=lambda item: item.observation_id.value)
        )
        points = tuple(sorted(self.points3d, key=lambda item: item.point_id.value))
        object.__setattr__(self, "camera_calibrations", calibrations)
        object.__setattr__(self, "camera_poses", poses)
        object.__setattr__(self, "points3d", points)

        calibration_ids = [item.calibration_id for item in calibrations]
        if len(calibration_ids) != len(set(calibration_ids)):
            raise ValueError("camera calibration estimate IDs must be unique")
        pose_observations = [item.observation_id for item in poses]
        if len(pose_observations) != len(set(pose_observations)):
            raise ValueError("camera poses cannot repeat an observation")
        point_ids = [item.point_id for item in points]
        if len(point_ids) != len(set(point_ids)):
            raise ValueError("estimated point IDs must be unique")

        if tuple(pose_observations) != self.provenance.source_observation_ids:
            raise ValueError("sparse-model provenance must exactly match registered camera poses")
        if any(item.local_frame_id != self.local_frame_id for item in poses):
            raise ValueError("all camera poses must use the sparse model local frame")
        if any(item.local_frame_id != self.local_frame_id for item in points):
            raise ValueError("all 3D points must use the sparse model local frame")

        available_calibrations = set(calibration_ids)
        referenced_calibrations = {item.calibration_id for item in poses}
        if referenced_calibrations != available_calibrations:
            raise ValueError("camera calibration estimates must exactly match pose references")

        model_observations = set(pose_observations)
        for calibration in calibrations:
            if not set(calibration.provenance.source_observation_ids).issubset(
                model_observations
            ):
                raise ValueError(
                    "camera calibration provenance must stay inside model membership"
                )
        for point in points:
            if not set(point.provenance.source_observation_ids).issubset(
                model_observations
            ):
                raise ValueError("3D point provenance must stay inside model membership")

    @property
    def observation_ids(self) -> tuple[ObservationId, ...]:
        return self.provenance.source_observation_ids

    @property
    def observation_count(self) -> int:
        return len(self.camera_poses)

    @property
    def point_count(self) -> int:
        return len(self.points3d)
