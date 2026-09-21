from __future__ import annotations

import math
import re
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.cameras import ImageDimensions
from wre.domain.fragments import LocalFrameId
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_LOWER_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")
_ROTATION_TOLERANCE = 1e-6

RotationMatrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
TranslationVector3 = tuple[float, float, float]


def _require_token(
    value: object,
    pattern: re.Pattern[str],
    context: str,
    message: str,
) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{context} {message}")


def _require_finite_float(value: object, context: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")


def _validate_intrinsic_parameters(values: object) -> None:
    if not isinstance(values, tuple):
        raise TypeError("camera_solution.intrinsic_parameters must be an immutable tuple")
    if not values:
        raise ValueError("camera_solution.intrinsic_parameters must not be empty")
    for value in values:
        _require_finite_float(value, "camera_solution.intrinsic_parameters member")


def _validate_rotation_matrix(value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("camera_solution.rotation_matrix must be an immutable 3x3 tuple")

    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError("camera_solution.rotation_matrix rows must be immutable 3-tuples")
        for member in row:
            _require_finite_float(member, "camera_solution.rotation_matrix member")
        rows.append(row)

    for row in rows:
        norm = math.sqrt(sum(member * member for member in row))
        if abs(norm - 1.0) > _ROTATION_TOLERANCE:
            raise ValueError("camera_solution.rotation_matrix rows must have unit length")

    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            dot = sum(
                rows[left_index][axis] * rows[right_index][axis]
                for axis in range(3)
            )
            if abs(dot) > _ROTATION_TOLERANCE:
                raise ValueError("camera_solution.rotation_matrix rows must be mutually orthogonal")

    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError("camera_solution.rotation_matrix determinant must be +1")


def _validate_translation_xyz(value: object) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("camera_solution.translation_xyz must be an immutable 3-vector tuple")
    for member in value:
        _require_finite_float(member, "camera_solution.translation_xyz member")


def _validate_uncertainty_artifacts(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("camera_solution.uncertainty_artifacts must be an immutable tuple")
    if any(not isinstance(ref, ArtifactRef) for ref in value):
        raise TypeError("camera_solution.uncertainty_artifacts members must be ArtifactRef")

    artifact_keys = tuple(
        (ref.artifact_id.value, ref.artifact_kind.value)
        for ref in value
    )
    if len(artifact_keys) != len(set(artifact_keys)):
        raise ValueError("camera_solution.uncertainty_artifacts must be unique")
    if artifact_keys != tuple(sorted(artifact_keys)):
        raise ValueError(
            "camera_solution.uncertainty_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )

    kinds_by_id: dict[str, str] = {}
    for ref in value:
        previous = kinds_by_id.setdefault(
            ref.artifact_id.value,
            ref.artifact_kind.value,
        )
        if previous != ref.artifact_kind.value:
            raise ValueError(
                "camera_solution.uncertainty_artifacts must not declare conflicting "
                "ArtifactKind values for one ArtifactId"
            )


@dataclass(frozen=True, slots=True, order=True)
class CameraSolutionId:
    """Opaque identity for one canonical per-observation camera solution."""

    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _OPAQUE_ID_RE,
            "camera_solution_id",
            "must be 1-128 characters using letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class CameraProjectionModelName:
    """Open solver-independent projection-model semantic token."""

    value: str

    def __post_init__(self) -> None:
        _require_token(
            self.value,
            _LOWER_TOKEN_RE,
            "camera_projection_model",
            "must be a 1-128 character lowercase token using "
            "letters, digits, '.', '_', ':' or '-'",
        )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class CameraSolution:
    """Canonical per-observation camera calibration and camera-from-local-frame pose.

    Extrinsics use exactly: x_camera = rotation_matrix * x_local + translation_xyz.
    The local frame carries no metric-scale, Earth-alignment or geographic-placement claim.
    """

    solution_id: CameraSolutionId
    observation_id: ObservationId
    local_frame_id: LocalFrameId
    projection_model: CameraProjectionModelName
    dimensions: ImageDimensions
    intrinsic_parameters: tuple[float, ...]
    rotation_matrix: RotationMatrix3x3
    translation_xyz: TranslationVector3
    uncertainty_artifacts: tuple[ArtifactRef, ...]
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.solution_id, CameraSolutionId):
            raise TypeError("camera_solution.solution_id must be CameraSolutionId")
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("camera_solution.observation_id must be ObservationId")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("camera_solution.local_frame_id must be LocalFrameId")
        if not isinstance(self.projection_model, CameraProjectionModelName):
            raise TypeError(
                "camera_solution.projection_model must be CameraProjectionModelName"
            )
        if not isinstance(self.dimensions, ImageDimensions):
            raise TypeError("camera_solution.dimensions must be ImageDimensions")

        _validate_intrinsic_parameters(self.intrinsic_parameters)
        _validate_rotation_matrix(self.rotation_matrix)
        _validate_translation_xyz(self.translation_xyz)
        _validate_uncertainty_artifacts(self.uncertainty_artifacts)

        if not isinstance(self.metrics, MetricVector):
            raise TypeError("camera_solution.metrics must be MetricVector")
