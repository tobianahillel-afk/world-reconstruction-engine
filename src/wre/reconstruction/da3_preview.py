from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    RotationMatrix3x3,
    TranslationVector3,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

DA3_BASE_MODEL_IDENTIFIER = "depth-anything/DA3-BASE"
DA3_BASE_SOURCE_REPOSITORY = "ByteDance-Seed/Depth-Anything-3"
DA3_DEPTH_VALUE_CONVENTION = DepthValueConventionName("da3.relative-depth")

_INTRINSIC_TOLERANCE = 1e-6
_ROTATION_TOLERANCE = 1e-6


PinholeIntrinsicMatrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def _require_finite_float(value: object, context: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    return value


def _validate_intrinsics(value: object) -> PinholeIntrinsicMatrix3x3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("da3_preview.intrinsics must be an immutable 3x3 tuple")

    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError("da3_preview.intrinsics rows must be immutable 3-tuples")
        checked = tuple(
            _require_finite_float(member, "da3_preview.intrinsics member") for member in row
        )
        rows.append((checked[0], checked[1], checked[2]))

    matrix = (rows[0], rows[1], rows[2])
    fx = matrix[0][0]
    fy = matrix[1][1]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("da3_preview pinhole focal lengths must be strictly positive")

    required_zero = (
        matrix[0][1],
        matrix[1][0],
        matrix[2][0],
        matrix[2][1],
    )
    if any(abs(member) > _INTRINSIC_TOLERANCE for member in required_zero):
        raise ValueError("da3_preview intrinsics must have zero skew and homogeneous off-diagonals")
    if abs(matrix[2][2] - 1.0) > _INTRINSIC_TOLERANCE:
        raise ValueError("da3_preview intrinsics homogeneous bottom-right value must be 1.0")

    return matrix


def _validate_rotation(value: object) -> RotationMatrix3x3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("da3_preview.world_to_camera_rotation must be an immutable 3x3 tuple")

    rows: list[tuple[float, float, float]] = []
    for row in value:
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError("da3_preview.world_to_camera_rotation rows must be immutable 3-tuples")
        checked = tuple(
            _require_finite_float(member, "da3_preview.world_to_camera_rotation member")
            for member in row
        )
        rows.append((checked[0], checked[1], checked[2]))

    for row in rows:
        norm = math.sqrt(sum(member * member for member in row))
        if abs(norm - 1.0) > _ROTATION_TOLERANCE:
            raise ValueError("da3_preview world-to-camera rotation rows must have unit length")

    for left_index in range(3):
        for right_index in range(left_index + 1, 3):
            dot = sum(rows[left_index][axis] * rows[right_index][axis] for axis in range(3))
            if abs(dot) > _ROTATION_TOLERANCE:
                raise ValueError(
                    "da3_preview world-to-camera rotation rows must be mutually orthogonal"
                )

    determinant = (
        rows[0][0] * (rows[1][1] * rows[2][2] - rows[1][2] * rows[2][1])
        - rows[0][1] * (rows[1][0] * rows[2][2] - rows[1][2] * rows[2][0])
        + rows[0][2] * (rows[1][0] * rows[2][1] - rows[1][1] * rows[2][0])
    )
    if abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError("da3_preview world-to-camera rotation determinant must be +1")

    return (rows[0], rows[1], rows[2])


def _validate_translation(value: object) -> TranslationVector3:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("da3_preview.world_to_camera_translation must be an immutable 3-vector")
    checked = tuple(
        _require_finite_float(member, "da3_preview.world_to_camera_translation member")
        for member in value
    )
    return (checked[0], checked[1], checked[2])


def _validate_validity(value: object, pixel_count: int) -> tuple[bool, ...]:
    if not isinstance(value, tuple):
        raise TypeError("da3_preview.validity must be an immutable tuple")
    if len(value) != pixel_count:
        raise ValueError("da3_preview.validity length must match image pixel count")
    if any(type(member) is not bool for member in value):
        raise TypeError("da3_preview.validity members must be bool")
    return value


def _validate_depth(
    value: object,
    validity: tuple[bool, ...],
    pixel_count: int,
) -> tuple[float, ...]:
    if not isinstance(value, tuple):
        raise TypeError("da3_preview.depth_values must be an immutable tuple")
    if len(value) != pixel_count:
        raise ValueError("da3_preview.depth_values length must match image pixel count")

    checked: list[float] = []
    for index, member in enumerate(value):
        depth = _require_finite_float(member, "da3_preview.depth_values member")
        if validity[index]:
            if depth <= 0.0:
                raise ValueError("da3_preview valid pixels must have strictly positive depth")
        elif depth != 0.0:
            raise ValueError("da3_preview invalid pixels must use canonical 0.0 depth")
        checked.append(depth)
    return tuple(checked)


def _validate_confidence(
    value: object,
    validity: tuple[bool, ...],
    pixel_count: int,
) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, tuple):
        raise TypeError("da3_preview.confidence must be None or an immutable tuple")
    if len(value) != pixel_count:
        raise ValueError("da3_preview.confidence length must match image pixel count")

    checked: list[float] = []
    for index, member in enumerate(value):
        confidence = _require_finite_float(member, "da3_preview.confidence member")
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError("da3_preview confidence must be in the inclusive range 0.0 to 1.0")
        if not validity[index] and confidence != 0.0:
            raise ValueError("da3_preview invalid pixels must use canonical 0.0 confidence")
        checked.append(confidence)
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class Da3BaseObservationPrediction:
    """Pure candidate-native DA3-BASE geometry for one observation.

    The pose is OpenCV world-to-camera as documented by the reviewed DA3 API.
    WRE treats that candidate-private world as one local reconstruction frame only.
    Depth samples are retained as DA3 relative-depth values without claiming camera-z,
    ray distance, metric scale, Earth alignment, or project-world identity.
    """

    observation_id: ObservationId
    dimensions: ImageDimensions
    world_to_camera_rotation: RotationMatrix3x3
    world_to_camera_translation: TranslationVector3
    intrinsics: PinholeIntrinsicMatrix3x3
    depth_values: tuple[float, ...]
    validity: tuple[bool, ...]
    confidence: tuple[float, ...] | None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("da3_preview.observation_id must be ObservationId")
        if not isinstance(self.dimensions, ImageDimensions):
            raise TypeError("da3_preview.dimensions must be ImageDimensions")

        _validate_rotation(self.world_to_camera_rotation)
        _validate_translation(self.world_to_camera_translation)
        _validate_intrinsics(self.intrinsics)

        pixel_count = self.dimensions.width_px * self.dimensions.height_px
        validity = _validate_validity(self.validity, pixel_count)
        _validate_depth(self.depth_values, validity, pixel_count)
        _validate_confidence(self.confidence, validity, pixel_count)


def _derive_id(
    identity: Sha256Digest,
    role: str,
    observation_id: ObservationId | None = None,
) -> str:
    material = f"da3-base\0{identity.value}\0{role}"
    if observation_id is not None:
        material = f"{material}\0{observation_id.value}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return digest[:48]


def _validate_predictions(
    predictions: tuple[Da3BaseObservationPrediction, ...],
) -> None:
    if not isinstance(predictions, tuple):
        raise TypeError("da3_preview predictions must be an immutable tuple")
    if not predictions:
        raise ValueError("da3_preview predictions must contain at least one observation")
    if any(not isinstance(item, Da3BaseObservationPrediction) for item in predictions):
        raise TypeError(
            "da3_preview predictions must contain only Da3BaseObservationPrediction values"
        )

    observation_values = tuple(item.observation_id.value for item in predictions)
    if len(observation_values) != len(set(observation_values)):
        raise ValueError("da3_preview predictions cannot contain duplicate observations")
    if observation_values != tuple(sorted(observation_values)):
        raise ValueError("da3_preview predictions must be in canonical ObservationId order")


def normalize_da3_base_preview(
    predictions: tuple[Da3BaseObservationPrediction, ...],
    *,
    normalization_identity: Sha256Digest,
) -> FeedForwardGeometryResult:
    """Normalize reviewed DA3-BASE prediction semantics into canonical WRE geometry."""

    _validate_predictions(predictions)
    if not isinstance(normalization_identity, Sha256Digest):
        raise TypeError("da3_preview.normalization_identity must be Sha256Digest")

    frame_id = LocalFrameId(f"frame:da3:{_derive_id(normalization_identity, 'frame')}")
    metrics = MetricVector(observations=())

    cameras: list[CameraSolution] = []
    depths: list[DepthField] = []

    for prediction in predictions:
        camera_id = CameraSolutionId(
            f"camera:da3:{_derive_id(normalization_identity, 'camera', prediction.observation_id)}"
        )
        depth_id = DepthFieldId(
            f"depth:da3:{_derive_id(normalization_identity, 'depth', prediction.observation_id)}"
        )

        fx = prediction.intrinsics[0][0]
        fy = prediction.intrinsics[1][1]
        cx = prediction.intrinsics[0][2]
        cy = prediction.intrinsics[1][2]

        camera = CameraSolution(
            solution_id=camera_id,
            observation_id=prediction.observation_id,
            local_frame_id=frame_id,
            projection_model=CameraProjectionModelName("pinhole"),
            dimensions=prediction.dimensions,
            intrinsic_parameters=(fx, fy, cx, cy),
            rotation_matrix=prediction.world_to_camera_rotation,
            translation_xyz=prediction.world_to_camera_translation,
            uncertainty_artifacts=(),
            metrics=metrics,
        )
        depth = DepthField(
            depth_field_id=depth_id,
            observation_id=prediction.observation_id,
            camera_solution_id=camera.solution_id,
            dimensions=prediction.dimensions,
            depth_value_convention=DA3_DEPTH_VALUE_CONVENTION,
            depth_values=prediction.depth_values,
            validity=prediction.validity,
            confidence=prediction.confidence,
            metrics=metrics,
        )
        cameras.append(camera)
        depths.append(depth)

    camera_tuple = tuple(cameras)
    depth_tuple = tuple(depths)
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(
            f"geometry:da3:{_derive_id(normalization_identity, 'geometry')}"
        ),
        local_frame_id=frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(
            sorted((item.solution_id for item in camera_tuple), key=lambda item: item.value)
        ),
        depth_field_ids=tuple(
            sorted((item.depth_field_id for item in depth_tuple), key=lambda item: item.value)
        ),
        point_map_ids=(),
        metrics=metrics,
    )

    return FeedForwardGeometryResult(
        camera_solutions=camera_tuple,
        depth_fields=depth_tuple,
        point_maps=(),
        geometry_solution=geometry,
    )
