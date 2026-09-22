from __future__ import annotations

from wre.domain import (
    CameraCalibrationEstimate,
    CameraCalibrationEstimateId,
    CameraPoseEstimate,
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    EstimatedPoint3DId,
    EstimatedTrackElement,
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
    ImageDimensions,
    LocalFrameId,
    LocalScaleStatus,
    MetricVector,
    ObservationId,
    Point3DEstimate,
    PointMap,
    PointMapId,
    ReconstructionRunId,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
)
from wre.domain.runs import DerivedArtifactProvenance
from wre.reconstruction.legacy_geometry_conversion import convert_sparse_reconstruction_estimate

_ROTATION = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)
_TRANSLATION = (3.0, -2.0, 5.0)
_LOCAL_POINTS = (
    (2.0, 7.0, -1.0),
    (-3.0, 4.0, 8.0),
)
_EXPECTED_CAMERA_POINT = (-4.0, 0.0, 4.0)


def _empty_metrics() -> MetricVector:
    return MetricVector(observations=())


def _camera_from_local(
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    translation: tuple[float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        rotation[0][0] * point[0]
        + rotation[0][1] * point[1]
        + rotation[0][2] * point[2]
        + translation[0],
        rotation[1][0] * point[0]
        + rotation[1][1] * point[1]
        + rotation[1][2] * point[2]
        + translation[1],
        rotation[2][0] * point[0]
        + rotation[2][1] * point[1]
        + rotation[2][2] * point[2]
        + translation[2],
    )


def _transpose_camera_from_local(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        _ROTATION[0][0] * point[0]
        + _ROTATION[1][0] * point[1]
        + _ROTATION[2][0] * point[2]
        + _TRANSLATION[0],
        _ROTATION[0][1] * point[0]
        + _ROTATION[1][1] * point[1]
        + _ROTATION[2][1] * point[2]
        + _TRANSLATION[1],
        _ROTATION[0][2] * point[0]
        + _ROTATION[1][2] * point[1]
        + _ROTATION[2][2] * point[2]
        + _TRANSLATION[2],
    )


def _translation_sign_flipped_camera_from_local(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _camera_from_local(_ROTATION, (0.0, 0.0, 0.0), point)
    return (
        rotated[0] - _TRANSLATION[0],
        rotated[1] - _TRANSLATION[1],
        rotated[2] - _TRANSLATION[2],
    )


def _inverse_pose_interpretation(
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    shifted = (
        point[0] - _TRANSLATION[0],
        point[1] - _TRANSLATION[1],
        point[2] - _TRANSLATION[2],
    )
    return (
        _ROTATION[0][0] * shifted[0] + _ROTATION[1][0] * shifted[1] + _ROTATION[2][0] * shifted[2],
        _ROTATION[0][1] * shifted[0] + _ROTATION[1][1] * shifted[1] + _ROTATION[2][1] * shifted[2],
        _ROTATION[0][2] * shifted[0] + _ROTATION[1][2] * shifted[1] + _ROTATION[2][2] * shifted[2],
    )


def _canonical_camera(local_frame_id: LocalFrameId) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId("camera-solution:coordinate-fixture"),
        observation_id=ObservationId("obs:coordinate-fixture"),
        local_frame_id=local_frame_id,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=1600, height_px=900),
        intrinsic_parameters=(1200.0, 1200.0, 800.0, 450.0),
        rotation_matrix=_ROTATION,
        translation_xyz=_TRANSLATION,
        uncertainty_artifacts=(),
        metrics=_empty_metrics(),
    )


def _point_map(local_frame_id: LocalFrameId) -> PointMap:
    return PointMap(
        point_map_id=PointMapId("point-map:coordinate-fixture"),
        local_frame_id=local_frame_id,
        source_observation_ids=(ObservationId("obs:coordinate-fixture"),),
        positions_xyz=_LOCAL_POINTS,
        confidence=None,
        metrics=_empty_metrics(),
    )


def _geometry(
    *,
    local_frame_id: LocalFrameId,
    scale_status: GeometryScaleStatus,
    camera_solution_id: CameraSolutionId,
    point_map_id: PointMapId,
) -> GeometrySolution:
    return GeometrySolution(
        geometry_solution_id=GeometrySolutionId(
            f"geometry:coordinate-fixture:{scale_status.value}"
        ),
        local_frame_id=local_frame_id,
        scale_status=scale_status,
        camera_solution_ids=(camera_solution_id,),
        depth_field_ids=(),
        point_map_ids=(point_map_id,),
        metrics=_empty_metrics(),
    )


def _legacy_provenance() -> DerivedArtifactProvenance:
    return DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:coordinate-fixture"),
        source_observation_ids=(ObservationId("obs:coordinate-fixture"),),
    )


def _legacy_estimate(scale_status: LocalScaleStatus) -> SparseReconstructionEstimate:
    local_frame_id = LocalFrameId("frame:coordinate-fixture")
    observation_id = ObservationId("obs:coordinate-fixture")
    calibration_id = CameraCalibrationEstimateId("calibration:coordinate-fixture")
    provenance = _legacy_provenance()

    calibration = CameraCalibrationEstimate(
        calibration_id=calibration_id,
        projection_model="PINHOLE",
        width_px=1600,
        height_px=900,
        parameters=(1200.0, 1200.0, 800.0, 450.0),
        has_prior_focal_length=False,
        provenance=provenance,
    )
    pose = CameraPoseEstimate(
        observation_id=observation_id,
        local_frame_id=local_frame_id,
        calibration_id=calibration_id,
        rotation_matrix=_ROTATION,
        translation_xyz=_TRANSLATION,
        provenance=provenance,
    )
    points = tuple(
        Point3DEstimate(
            point_id=EstimatedPoint3DId(f"point:coordinate-fixture:{index}"),
            local_frame_id=local_frame_id,
            position_xyz=position,
            reprojection_error_px=None,
            track=(EstimatedTrackElement(observation_id, index),),
            provenance=provenance,
        )
        for index, position in enumerate(_LOCAL_POINTS, start=1)
    )
    return SparseReconstructionEstimate(
        estimate_id=SparseReconstructionEstimateId(
            f"sparse:coordinate-fixture:{scale_status.value}"
        ),
        local_frame_id=local_frame_id,
        scale_status=scale_status,
        provenance=provenance,
        camera_calibrations=(calibration,),
        camera_poses=(pose,),
        points3d=points,
    )


def test_camera_from_local_convention_rejects_common_alternative_interpretations() -> None:
    local_frame_id = LocalFrameId("frame:coordinate-fixture")
    camera = _canonical_camera(local_frame_id)
    point = _LOCAL_POINTS[0]

    canonical = _camera_from_local(
        camera.rotation_matrix,
        camera.translation_xyz,
        point,
    )

    assert canonical == _EXPECTED_CAMERA_POINT
    assert _transpose_camera_from_local(point) != canonical
    assert _translation_sign_flipped_camera_from_local(point) != canonical
    assert _inverse_pose_interpretation(point) != canonical
    assert (
        point[0] + _TRANSLATION[0],
        point[1] + _TRANSLATION[1],
        point[2] + _TRANSLATION[2],
    ) != canonical


def test_local_frame_point_order_and_scale_status_remain_explicit_and_non_transforming() -> None:
    local_frame_id = LocalFrameId("frame:coordinate-fixture")
    camera = _canonical_camera(local_frame_id)
    point_map = _point_map(local_frame_id)

    unresolved = _geometry(
        local_frame_id=local_frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_id=camera.solution_id,
        point_map_id=point_map.point_map_id,
    )
    metric = _geometry(
        local_frame_id=local_frame_id,
        scale_status=GeometryScaleStatus.METRIC,
        camera_solution_id=camera.solution_id,
        point_map_id=point_map.point_map_id,
    )

    assert camera.local_frame_id is local_frame_id
    assert point_map.local_frame_id is local_frame_id
    assert unresolved.local_frame_id is local_frame_id
    assert metric.local_frame_id is local_frame_id
    assert point_map.positions_xyz is _LOCAL_POINTS
    assert point_map.positions_xyz == (
        (2.0, 7.0, -1.0),
        (-3.0, 4.0, 8.0),
    )

    assert unresolved.scale_status is GeometryScaleStatus.UNRESOLVED
    assert metric.scale_status is GeometryScaleStatus.METRIC
    assert unresolved.camera_solution_ids == metric.camera_solution_ids
    assert unresolved.point_map_ids == metric.point_map_ids

    assert camera.rotation_matrix == _ROTATION
    assert camera.translation_xyz == _TRANSLATION
    assert not hasattr(camera, "world_transform")
    assert not hasattr(point_map, "world_transform")
    assert not hasattr(unresolved, "world_transform")
    assert not hasattr(metric, "world_transform")
    assert not hasattr(unresolved, "earth_transform")
    assert not hasattr(metric, "earth_transform")
    assert not hasattr(unresolved, "fragment_id")
    assert not hasattr(metric, "fragment_id")
    assert not hasattr(unresolved, "scale_factor")
    assert not hasattr(metric, "scale_factor")


def test_legacy_bridge_preserves_the_same_camera_from_local_equation_and_scale_truth() -> None:
    expected_scale = (
        (LocalScaleStatus.UNRESOLVED, GeometryScaleStatus.UNRESOLVED),
        (LocalScaleStatus.METRIC, GeometryScaleStatus.METRIC),
    )

    for legacy_scale, canonical_scale in expected_scale:
        source = _legacy_estimate(legacy_scale)
        result = convert_sparse_reconstruction_estimate(source)

        assert len(result.camera_solutions) == 1
        camera = result.camera_solutions[0]
        assert camera.local_frame_id is source.local_frame_id
        assert camera.rotation_matrix == _ROTATION
        assert camera.translation_xyz == _TRANSLATION
        assert camera.projection_model.value == "pinhole"
        assert result.point_map.local_frame_id is source.local_frame_id
        assert result.point_map.positions_xyz == _LOCAL_POINTS
        assert result.geometry_solution.local_frame_id is source.local_frame_id
        assert result.geometry_solution.scale_status is canonical_scale
        assert result.geometry_solution.camera_solution_ids == (camera.solution_id,)
        assert result.geometry_solution.point_map_ids == (result.point_map.point_map_id,)
        assert result.geometry_solution.depth_field_ids == ()

        converted_camera_point = _camera_from_local(
            camera.rotation_matrix,
            camera.translation_xyz,
            _LOCAL_POINTS[0],
        )
        assert converted_camera_point == _EXPECTED_CAMERA_POINT
        assert converted_camera_point == _camera_from_local(
            _ROTATION,
            _TRANSLATION,
            _LOCAL_POINTS[0],
        )

        assert tuple(point.position_xyz for point in source.points3d) == _LOCAL_POINTS
        assert not hasattr(result.geometry_solution, "scale_factor")
        assert not hasattr(result.geometry_solution, "world_transform")
        assert not hasattr(result.geometry_solution, "earth_transform")


def test_coordinate_fixture_has_no_external_math_solver_io_or_alignment_surface() -> None:
    forbidden_names = {
        "Path",
        "numpy",
        "np",
        "torch",
        "open3d",
        "pycolmap",
        "subprocess",
        "socket",
        "sqlite3",
        "requests",
        "align_frames",
        "convert_coordinates",
        "world_transform",
        "earth_transform",
        "ecef",
        "enu",
    }

    assert forbidden_names.isdisjoint(globals())
