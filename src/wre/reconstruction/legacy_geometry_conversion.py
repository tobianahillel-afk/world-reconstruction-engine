from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.estimated_geometry import (
    LocalScaleStatus,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
)
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricVector
from wre.domain.point_maps import PointMap, PointMapId

_ID_SEPARATOR = "\x00"


def _derived_digest(role: str, *source_identifiers: str) -> str:
    material = _ID_SEPARATOR.join((role, *source_identifiers)).encode("utf-8")
    return sha256(material).hexdigest()


def _camera_solution_id(
    estimate_id: SparseReconstructionEstimateId,
    observation_id: str,
) -> CameraSolutionId:
    return CameraSolutionId(
        f"legacy-camera:{_derived_digest('camera-solution', estimate_id.value, observation_id)}"
    )


def _point_map_id(estimate_id: SparseReconstructionEstimateId) -> PointMapId:
    return PointMapId(f"legacy-points:{_derived_digest('point-map', estimate_id.value)}")


def _geometry_solution_id(estimate_id: SparseReconstructionEstimateId) -> GeometrySolutionId:
    return GeometrySolutionId(
        f"legacy-geometry:{_derived_digest('geometry-solution', estimate_id.value)}"
    )


def _empty_metrics() -> MetricVector:
    return MetricVector(observations=())


def _geometry_scale_status(value: LocalScaleStatus) -> GeometryScaleStatus:
    if value is LocalScaleStatus.UNRESOLVED:
        return GeometryScaleStatus.UNRESOLVED
    if value is LocalScaleStatus.METRIC:
        return GeometryScaleStatus.METRIC
    raise TypeError("legacy sparse geometry scale status must be LocalScaleStatus")


@dataclass(frozen=True, slots=True)
class LegacySparseGeometryConversionResult:
    """Canonical V2 values converted one-way from one retained legacy sparse estimate."""

    source_estimate_id: SparseReconstructionEstimateId
    camera_solutions: tuple[CameraSolution, ...]
    point_map: PointMap
    geometry_solution: GeometrySolution

    def __post_init__(self) -> None:
        if not isinstance(self.source_estimate_id, SparseReconstructionEstimateId):
            raise TypeError(
                "legacy_geometry_conversion.source_estimate_id "
                "must be SparseReconstructionEstimateId"
            )
        if not isinstance(self.camera_solutions, tuple):
            raise TypeError(
                "legacy_geometry_conversion.camera_solutions must be an immutable tuple"
            )
        if any(not isinstance(solution, CameraSolution) for solution in self.camera_solutions):
            raise TypeError(
                "legacy_geometry_conversion.camera_solutions members must be CameraSolution"
            )
        if not isinstance(self.point_map, PointMap):
            raise TypeError("legacy_geometry_conversion.point_map must be PointMap")
        if not isinstance(self.geometry_solution, GeometrySolution):
            raise TypeError(
                "legacy_geometry_conversion.geometry_solution must be GeometrySolution"
            )


def convert_sparse_reconstruction_estimate(
    estimate: SparseReconstructionEstimate,
) -> LegacySparseGeometryConversionResult:
    """Convert retained legacy sparse geometry into canonical V2 values without inference."""

    if not isinstance(estimate, SparseReconstructionEstimate):
        raise TypeError("estimate must be SparseReconstructionEstimate")

    calibrations = {
        calibration.calibration_id: calibration for calibration in estimate.camera_calibrations
    }

    camera_solutions = tuple(
        sorted(
            (
                CameraSolution(
                    solution_id=_camera_solution_id(
                        estimate.estimate_id,
                        pose.observation_id.value,
                    ),
                    observation_id=pose.observation_id,
                    local_frame_id=pose.local_frame_id,
                    projection_model=CameraProjectionModelName(
                        calibrations[pose.calibration_id].projection_model.lower()
                    ),
                    dimensions=ImageDimensions(
                        width_px=calibrations[pose.calibration_id].width_px,
                        height_px=calibrations[pose.calibration_id].height_px,
                    ),
                    intrinsic_parameters=calibrations[pose.calibration_id].parameters,
                    rotation_matrix=pose.rotation_matrix,
                    translation_xyz=pose.translation_xyz,
                    uncertainty_artifacts=(),
                    metrics=_empty_metrics(),
                )
                for pose in estimate.camera_poses
            ),
            key=lambda solution: solution.solution_id.value,
        )
    )

    point_map = PointMap(
        point_map_id=_point_map_id(estimate.estimate_id),
        local_frame_id=estimate.local_frame_id,
        source_observation_ids=estimate.observation_ids,
        positions_xyz=tuple(point.position_xyz for point in estimate.points3d),
        confidence=None,
        metrics=_empty_metrics(),
    )

    geometry_solution = GeometrySolution(
        geometry_solution_id=_geometry_solution_id(estimate.estimate_id),
        local_frame_id=estimate.local_frame_id,
        scale_status=_geometry_scale_status(estimate.scale_status),
        camera_solution_ids=tuple(solution.solution_id for solution in camera_solutions),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_empty_metrics(),
    )

    return LegacySparseGeometryConversionResult(
        source_estimate_id=estimate.estimate_id,
        camera_solutions=camera_solutions,
        point_map=point_map,
        geometry_solution=geometry_solution,
    )
