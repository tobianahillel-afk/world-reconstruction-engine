from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.camera_solutions import CameraSolution
from wre.domain.metrics import (
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
)
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

FEED_FORWARD_CAMERA_QUALITY_IMPLEMENTATION = "wre.reconstruction.feed_forward_camera_quality"
FEED_FORWARD_CAMERA_QUALITY_VERSION = "1"
FEED_FORWARD_CAMERA_QUALITY_REVISION = "v2l13.4"

_EVALUATOR_CONFIGURATION_DOCUMENT = {
    "schema_version": 1,
    "matching": "exact_observation_id",
    "relative_rotation": "camera_from_local_pair_geodesic_degrees",
    "relative_translation": "first_camera_baseline_direction_degrees",
    "intrinsics": "pinhole_fx_fy_cx_cy_equal_dimensions",
    "focal_error": "mean_axis_relative_error_then_median_camera",
    "principal_point_error": "euclidean_pixels_over_reference_diagonal_then_median_camera",
    "aggregation": "deterministic_exact_median",
}
_EVALUATOR_CONFIGURATION_BYTES = json.dumps(
    _EVALUATOR_CONFIGURATION_DOCUMENT,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
FEED_FORWARD_CAMERA_QUALITY_EVALUATOR = ArtifactProducerIdentity(
    producer=ProducerRef(
        implementation=FEED_FORWARD_CAMERA_QUALITY_IMPLEMENTATION,
        version=FEED_FORWARD_CAMERA_QUALITY_VERSION,
        revision=FEED_FORWARD_CAMERA_QUALITY_REVISION,
    ),
    configuration=ConfigurationIdentity(
        sha256=Sha256Digest(hashlib.sha256(_EVALUATOR_CONFIGURATION_BYTES).hexdigest())
    ),
)

_GEOMETRY_CAMERA_DIMENSION = MetricDimension("geometry.camera")
_RATIO_UNIT = MetricUnit("ratio")
_DEGREE_UNIT = MetricUnit("degree")
_MEDIAN_CAMERA = MetricAggregation("median_shared_camera")
_MEDIAN_PAIR = MetricAggregation("median_shared_pair")

FOCAL_RELATIVE_ERROR_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.focal_relative_error_median"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=_MEDIAN_CAMERA,
)
OBSERVATION_COVERAGE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.observation_coverage_ratio"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("reference_observation_ratio"),
)
PRINCIPAL_POINT_ERROR_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.principal_point_error_normalized_median"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=_MEDIAN_CAMERA,
)
RELATIVE_ROTATION_ERROR_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.relative_rotation_error_deg_median"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_DEGREE_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=_MEDIAN_PAIR,
)
RELATIVE_TRANSLATION_DIRECTION_ERROR_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.relative_translation_direction_error_deg_median"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_DEGREE_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=_MEDIAN_PAIR,
)
TRANSLATION_PAIR_COVERAGE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.camera.translation_pair_coverage_ratio"),
    dimension=_GEOMETRY_CAMERA_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("eligible_shared_pair_ratio"),
)


@dataclass(frozen=True, slots=True, kw_only=True)
class FeedForwardCameraQualityRequest:
    """Pure camera-quality evidence over one candidate and one trusted reference set."""

    candidate: FeedForwardGeometryResult
    reference_cameras: tuple[CameraSolution, ...]
    input_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, FeedForwardGeometryResult):
            raise TypeError("camera_quality.candidate must be FeedForwardGeometryResult")
        if not isinstance(self.reference_cameras, tuple):
            raise TypeError("camera_quality.reference_cameras must be an immutable tuple")
        if not self.reference_cameras:
            raise ValueError("camera_quality.reference_cameras must be non-empty")
        if any(not isinstance(camera, CameraSolution) for camera in self.reference_cameras):
            raise TypeError("camera_quality.reference_cameras members must be CameraSolution")

        observation_values = tuple(camera.observation_id.value for camera in self.reference_cameras)
        if len(observation_values) != len(set(observation_values)):
            raise ValueError("camera_quality.reference_cameras observations must be unique")
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError(
                "camera_quality.reference_cameras must use canonical ObservationId order"
            )

        local_frame = self.reference_cameras[0].local_frame_id
        if any(camera.local_frame_id != local_frame for camera in self.reference_cameras):
            raise ValueError(
                "camera_quality.reference_cameras must share one reference LocalFrameId"
            )

        if not isinstance(self.input_artifacts, tuple):
            raise TypeError("camera_quality.input_artifacts must be an immutable tuple")
        if any(not isinstance(artifact, ArtifactRef) for artifact in self.input_artifacts):
            raise TypeError("camera_quality.input_artifacts members must be ArtifactRef")


Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Vector3 = tuple[float, float, float]


def _transpose(matrix: Matrix3x3) -> Matrix3x3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(left: Matrix3x3, right: Matrix3x3) -> Matrix3x3:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(3)) for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _matvec(matrix: Matrix3x3, vector: Vector3) -> Vector3:
    return (
        sum(matrix[0][axis] * vector[axis] for axis in range(3)),
        sum(matrix[1][axis] * vector[axis] for axis in range(3)),
        sum(matrix[2][axis] * vector[axis] for axis in range(3)),
    )


def _camera_center(camera: CameraSolution) -> Vector3:
    rotated = _matvec(_transpose(camera.rotation_matrix), camera.translation_xyz)
    return (-rotated[0], -rotated[1], -rotated[2])


def _relative_rotation(first: CameraSolution, second: CameraSolution) -> Matrix3x3:
    return _matmul(second.rotation_matrix, _transpose(first.rotation_matrix))


def _rotation_angle_degrees(left: Matrix3x3, right: Matrix3x3) -> float:
    difference = _matmul(left, _transpose(right))
    trace = difference[0][0] + difference[1][1] + difference[2][2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _unit(vector: Vector3) -> Vector3 | None:
    norm = math.sqrt(sum(member * member for member in vector))
    if norm == 0.0:
        return None
    return (
        vector[0] / norm,
        vector[1] / norm,
        vector[2] / norm,
    )


def _baseline_direction_in_first_camera(
    first: CameraSolution,
    second: CameraSolution,
) -> Vector3 | None:
    baseline_local = _subtract(_camera_center(second), _camera_center(first))
    return _unit(_matvec(first.rotation_matrix, baseline_local))


def _direction_angle_degrees(left: Vector3, right: Vector3) -> float:
    cosine = max(
        -1.0,
        min(1.0, sum(left[axis] * right[axis] for axis in range(3))),
    )
    return math.degrees(math.acos(cosine))


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("camera_quality median requires at least one value")
    ordered = tuple(sorted(values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _validate_shared_pinhole(
    candidate: CameraSolution,
    reference: CameraSolution,
) -> None:
    if candidate.projection_model.value != "pinhole":
        raise ValueError("camera_quality shared candidate camera must use pinhole projection")
    if reference.projection_model.value != "pinhole":
        raise ValueError("camera_quality shared reference camera must use pinhole projection")
    if len(candidate.intrinsic_parameters) != 4:
        raise ValueError("camera_quality shared candidate pinhole camera must have fx fy cx cy")
    if len(reference.intrinsic_parameters) != 4:
        raise ValueError("camera_quality shared reference pinhole camera must have fx fy cx cy")
    if candidate.dimensions != reference.dimensions:
        raise ValueError("camera_quality shared candidate/reference image dimensions must match")
    reference_fx, reference_fy, _, _ = reference.intrinsic_parameters
    if reference_fx <= 0.0 or reference_fy <= 0.0:
        raise ValueError("camera_quality trusted reference focal lengths must be strictly positive")


def _intrinsic_errors(
    candidate: CameraSolution,
    reference: CameraSolution,
) -> tuple[float, float]:
    _validate_shared_pinhole(candidate, reference)

    candidate_fx, candidate_fy, candidate_cx, candidate_cy = candidate.intrinsic_parameters
    reference_fx, reference_fy, reference_cx, reference_cy = reference.intrinsic_parameters

    focal_error = 0.5 * (
        abs(candidate_fx - reference_fx) / reference_fx
        + abs(candidate_fy - reference_fy) / reference_fy
    )
    principal_error_px = math.hypot(
        candidate_cx - reference_cx,
        candidate_cy - reference_cy,
    )
    diagonal = math.hypot(
        float(reference.dimensions.width_px),
        float(reference.dimensions.height_px),
    )
    return focal_error, principal_error_px / diagonal


def _observation(
    descriptor: MetricDescriptor,
    value: float,
    provenance: MetricProvenance,
) -> MetricObservation:
    return MetricObservation(
        descriptor=descriptor,
        value=float(value),
        provenance=provenance,
    )


def evaluate_feed_forward_camera_quality(
    request: FeedForwardCameraQualityRequest,
) -> MetricVector:
    """Measure camera quality without alignment, thresholds, routing or mutation."""

    if not isinstance(request, FeedForwardCameraQualityRequest):
        raise TypeError("request must be FeedForwardCameraQualityRequest")

    provenance = MetricProvenance(
        evaluator=FEED_FORWARD_CAMERA_QUALITY_EVALUATOR,
        input_artifacts=request.input_artifacts,
    )
    candidate_by_observation = {
        camera.observation_id: camera for camera in request.candidate.camera_solutions
    }
    reference_by_observation = {
        camera.observation_id: camera for camera in request.reference_cameras
    }
    shared_ids = tuple(
        sorted(
            set(candidate_by_observation).intersection(reference_by_observation),
            key=lambda observation_id: observation_id.value,
        )
    )

    observations: list[MetricObservation] = [
        _observation(
            OBSERVATION_COVERAGE_DESCRIPTOR,
            len(shared_ids) / len(request.reference_cameras),
            provenance,
        )
    ]

    if shared_ids:
        focal_errors: list[float] = []
        principal_errors: list[float] = []
        for observation_id in shared_ids:
            focal_error, principal_error = _intrinsic_errors(
                candidate_by_observation[observation_id],
                reference_by_observation[observation_id],
            )
            focal_errors.append(focal_error)
            principal_errors.append(principal_error)

        observations.extend(
            (
                _observation(
                    FOCAL_RELATIVE_ERROR_DESCRIPTOR,
                    _median(tuple(focal_errors)),
                    provenance,
                ),
                _observation(
                    PRINCIPAL_POINT_ERROR_DESCRIPTOR,
                    _median(tuple(principal_errors)),
                    provenance,
                ),
            )
        )

    if len(shared_ids) >= 2:
        rotation_errors: list[float] = []
        translation_errors: list[float] = []
        pair_count = 0

        for first_index in range(len(shared_ids) - 1):
            for second_index in range(first_index + 1, len(shared_ids)):
                first_id = shared_ids[first_index]
                second_id = shared_ids[second_index]
                pair_count += 1

                candidate_first = candidate_by_observation[first_id]
                candidate_second = candidate_by_observation[second_id]
                reference_first = reference_by_observation[first_id]
                reference_second = reference_by_observation[second_id]

                rotation_errors.append(
                    _rotation_angle_degrees(
                        _relative_rotation(candidate_first, candidate_second),
                        _relative_rotation(reference_first, reference_second),
                    )
                )

                candidate_direction = _baseline_direction_in_first_camera(
                    candidate_first,
                    candidate_second,
                )
                reference_direction = _baseline_direction_in_first_camera(
                    reference_first,
                    reference_second,
                )
                if candidate_direction is not None and reference_direction is not None:
                    translation_errors.append(
                        _direction_angle_degrees(
                            candidate_direction,
                            reference_direction,
                        )
                    )

        observations.extend(
            (
                _observation(
                    RELATIVE_ROTATION_ERROR_DESCRIPTOR,
                    _median(tuple(rotation_errors)),
                    provenance,
                ),
                _observation(
                    TRANSLATION_PAIR_COVERAGE_DESCRIPTOR,
                    len(translation_errors) / pair_count,
                    provenance,
                ),
            )
        )
        if translation_errors:
            observations.append(
                _observation(
                    RELATIVE_TRANSLATION_DIRECTION_ERROR_DESCRIPTOR,
                    _median(tuple(translation_errors)),
                    provenance,
                )
            )

    return MetricVector(
        observations=tuple(
            sorted(
                observations,
                key=lambda observation: observation.descriptor.name.value,
            )
        )
    )
