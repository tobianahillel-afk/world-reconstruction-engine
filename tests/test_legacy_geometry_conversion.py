from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from hashlib import sha256
from typing import Any, cast

import pytest

import wre.reconstruction.legacy_geometry_conversion as conversion_module
from wre.domain import (
    CameraCalibrationEstimate,
    CameraCalibrationEstimateId,
    CameraPoseEstimate,
    CameraSolution,
    EstimatedPoint3DId,
    EstimatedTrackElement,
    GeometryScaleStatus,
    LocalFrameId,
    LocalScaleStatus,
    ObservationId,
    Point3DEstimate,
    ReconstructionRunId,
    SparseReconstructionEstimate,
    SparseReconstructionEstimateId,
)
from wre.domain.runs import DerivedArtifactProvenance
from wre.reconstruction.legacy_geometry_conversion import (
    LegacySparseGeometryConversionResult,
    convert_sparse_reconstruction_estimate,
)


def _provenance(*observation_values: str) -> DerivedArtifactProvenance:
    return DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:legacy-conversion"),
        source_observation_ids=tuple(ObservationId(value) for value in observation_values),
    )


def _calibration(
    *,
    calibration_id: str = "calibration:shared",
    projection_model: str = "SIMPLE_RADIAL",
    observation_values: tuple[str, ...] = ("obs:a", "obs:b"),
    has_prior_focal_length: bool = True,
) -> CameraCalibrationEstimate:
    return CameraCalibrationEstimate(
        calibration_id=CameraCalibrationEstimateId(calibration_id),
        projection_model=projection_model,
        width_px=1280,
        height_px=720,
        parameters=(900.0, 640.0, 360.0, 0.01),
        has_prior_focal_length=has_prior_focal_length,
        provenance=_provenance(*observation_values),
    )


def _pose(
    observation_value: str,
    *,
    calibration_id: str = "calibration:shared",
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraPoseEstimate:
    return CameraPoseEstimate(
        observation_id=ObservationId(observation_value),
        local_frame_id=LocalFrameId("frame:legacy"),
        calibration_id=CameraCalibrationEstimateId(calibration_id),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=translation_xyz,
        provenance=_provenance(observation_value),
    )


def _point(
    point_id: str,
    position_xyz: tuple[float, float, float],
    *,
    observations: tuple[str, ...] = ("obs:a", "obs:b"),
    reprojection_error_px: float | None = 0.25,
) -> Point3DEstimate:
    return Point3DEstimate(
        point_id=EstimatedPoint3DId(point_id),
        local_frame_id=LocalFrameId("frame:legacy"),
        position_xyz=position_xyz,
        reprojection_error_px=reprojection_error_px,
        track=tuple(
            EstimatedTrackElement(ObservationId(value), index + 3)
            for index, value in enumerate(observations)
        ),
        provenance=_provenance(*observations),
    )


def _estimate(
    *,
    estimate_id: str = "sparse:legacy",
    scale_status: LocalScaleStatus = LocalScaleStatus.UNRESOLVED,
    projection_model: str = "SIMPLE_RADIAL",
    points3d: tuple[Point3DEstimate, ...] | None = None,
    has_prior_focal_length: bool = True,
) -> SparseReconstructionEstimate:
    if points3d is None:
        points3d = (
            _point("point:2", (4.0, 5.0, 6.0), reprojection_error_px=0.5),
            _point("point:1", (1.0, 2.0, 3.0), reprojection_error_px=0.125),
        )
    return SparseReconstructionEstimate(
        estimate_id=SparseReconstructionEstimateId(estimate_id),
        local_frame_id=LocalFrameId("frame:legacy"),
        scale_status=scale_status,
        provenance=_provenance("obs:a", "obs:b"),
        camera_calibrations=(
            _calibration(
                projection_model=projection_model,
                has_prior_focal_length=has_prior_focal_length,
            ),
        ),
        camera_poses=(
            _pose("obs:b", translation_xyz=(-1.0, 0.5, 4.0)),
            _pose("obs:a", translation_xyz=(1.0, 2.0, 3.0)),
        ),
        points3d=points3d,
    )


def _expected_digest(role: str, *source_identifiers: str) -> str:
    material = "\x00".join((role, *source_identifiers)).encode("utf-8")
    return sha256(material).hexdigest()


def test_conversion_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="estimate must be SparseReconstructionEstimate"):
        convert_sparse_reconstruction_estimate(cast(Any, "sparse:legacy"))


def test_conversion_result_has_exact_frozen_field_shape() -> None:
    result = convert_sparse_reconstruction_estimate(_estimate())

    assert tuple(field.name for field in fields(LegacySparseGeometryConversionResult)) == (
        "source_estimate_id",
        "camera_solutions",
        "point_map",
        "geometry_solution",
    )

    with pytest.raises(FrozenInstanceError):
        result.source_estimate_id = SparseReconstructionEstimateId(  # type: ignore[misc]
            "sparse:changed"
        )


def test_conversion_ids_are_deterministic_role_separated_and_source_scoped() -> None:
    source = _estimate()
    first = convert_sparse_reconstruction_estimate(source)
    second = convert_sparse_reconstruction_estimate(source)
    other = convert_sparse_reconstruction_estimate(_estimate(estimate_id="sparse:other"))

    assert first == second
    assert first.source_estimate_id is source.estimate_id

    camera_by_observation = {
        solution.observation_id.value: solution for solution in first.camera_solutions
    }
    for observation_value in ("obs:a", "obs:b"):
        expected = _expected_digest(
            "camera-solution",
            source.estimate_id.value,
            observation_value,
        )
        assert (
            camera_by_observation[observation_value].solution_id.value
            == f"legacy-camera:{expected}"
        )

    assert first.point_map.point_map_id.value == (
        "legacy-points:" + _expected_digest("point-map", source.estimate_id.value)
    )
    assert first.geometry_solution.geometry_solution_id.value == (
        "legacy-geometry:" + _expected_digest("geometry-solution", source.estimate_id.value)
    )

    all_ids = {
        *(solution.solution_id.value for solution in first.camera_solutions),
        first.point_map.point_map_id.value,
        first.geometry_solution.geometry_solution_id.value,
    }
    assert len(all_ids) == len(first.camera_solutions) + 2
    assert first.point_map.point_map_id != other.point_map.point_map_id
    assert (
        first.geometry_solution.geometry_solution_id != other.geometry_solution.geometry_solution_id
    )
    assert tuple(solution.solution_id for solution in first.camera_solutions) != tuple(
        solution.solution_id for solution in other.camera_solutions
    )


def test_uppercase_legacy_projection_model_is_lowercased_only() -> None:
    result = convert_sparse_reconstruction_estimate(_estimate(projection_model="SIMPLE_RADIAL"))

    assert {solution.projection_model.value for solution in result.camera_solutions} == {
        "simple_radial"
    }


@pytest.mark.parametrize(
    "projection_model",
    [
        "SIMPLE RADIAL",
        " SIMPLE_RADIAL",
        "SIMPLE_RADIAL ",
        "SIMPLE/RADIAL",
    ],
)
def test_invalid_canonical_projection_model_fails_closed_without_rewriting(
    projection_model: str,
) -> None:
    source = _estimate(projection_model=projection_model)

    with pytest.raises(ValueError, match="camera_projection_model"):
        convert_sparse_reconstruction_estimate(source)


def test_camera_conversion_preserves_per_observation_geometry_and_shared_calibration() -> None:
    source = _estimate()
    calibration = source.camera_calibrations[0]
    result = convert_sparse_reconstruction_estimate(source)

    assert len(result.camera_solutions) == len(source.camera_poses)
    assert tuple(solution.solution_id.value for solution in result.camera_solutions) == tuple(
        sorted(solution.solution_id.value for solution in result.camera_solutions)
    )

    sources_by_observation = {pose.observation_id: pose for pose in source.camera_poses}
    for solution in result.camera_solutions:
        pose = sources_by_observation[solution.observation_id]
        assert solution.local_frame_id is pose.local_frame_id
        assert solution.rotation_matrix is pose.rotation_matrix
        assert solution.translation_xyz is pose.translation_xyz
        assert solution.intrinsic_parameters is calibration.parameters
        assert solution.dimensions.width_px == calibration.width_px
        assert solution.dimensions.height_px == calibration.height_px
        assert solution.uncertainty_artifacts == ()
        assert solution.metrics.observations == ()
        assert not hasattr(solution, "camera_id")
        assert not hasattr(solution, "calibration_id")
        assert not hasattr(solution, "has_prior_focal_length")


@pytest.mark.parametrize(
    ("legacy_status", "canonical_status"),
    [
        (LocalScaleStatus.UNRESOLVED, GeometryScaleStatus.UNRESOLVED),
        (LocalScaleStatus.METRIC, GeometryScaleStatus.METRIC),
    ],
)
def test_scale_mapping_preserves_coordinates_without_rescaling_or_anchoring(
    legacy_status: LocalScaleStatus,
    canonical_status: GeometryScaleStatus,
) -> None:
    source = _estimate(scale_status=legacy_status)
    result = convert_sparse_reconstruction_estimate(source)

    assert result.geometry_solution.scale_status is canonical_status
    assert result.geometry_solution.local_frame_id is source.local_frame_id
    assert result.point_map.local_frame_id is source.local_frame_id

    source_translations = {
        pose.observation_id: pose.translation_xyz for pose in source.camera_poses
    }
    assert {
        solution.observation_id: solution.translation_xyz for solution in result.camera_solutions
    } == source_translations
    assert result.point_map.positions_xyz == tuple(point.position_xyz for point in source.points3d)

    assert not hasattr(result.geometry_solution, "scale_factor")
    assert not hasattr(result.geometry_solution, "world_transform")
    assert not hasattr(result.geometry_solution, "earth_transform")
    assert not hasattr(result.geometry_solution, "gps")
    assert not hasattr(result.point_map, "world_transform")
    assert not hasattr(result.point_map, "scale_factor")


def test_populated_point_map_preserves_legacy_canonical_point_order_and_support() -> None:
    source = _estimate()
    assert tuple(point.point_id.value for point in source.points3d) == (
        "point:1",
        "point:2",
    )

    result = convert_sparse_reconstruction_estimate(source)

    assert result.point_map.source_observation_ids is source.provenance.source_observation_ids
    assert result.point_map.positions_xyz == (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )
    assert result.point_map.confidence is None
    assert result.point_map.metrics.observations == ()


def test_zero_point_estimate_produces_explicit_empty_point_map_and_valid_geometry() -> None:
    source = _estimate(points3d=())

    result = convert_sparse_reconstruction_estimate(source)

    assert result.point_map.positions_xyz == ()
    assert result.point_map.confidence is None
    assert result.geometry_solution.point_map_ids == (result.point_map.point_map_id,)
    assert result.geometry_solution.depth_field_ids == ()


def test_geometry_solution_links_exact_canonical_camera_and_point_ids() -> None:
    source = _estimate()
    result = convert_sparse_reconstruction_estimate(source)

    camera_ids = tuple(solution.solution_id for solution in result.camera_solutions)
    assert result.geometry_solution.camera_solution_ids == camera_ids
    assert tuple(item.value for item in camera_ids) == tuple(
        sorted(item.value for item in camera_ids)
    )
    assert result.geometry_solution.point_map_ids == (result.point_map.point_map_id,)
    assert result.geometry_solution.depth_field_ids == ()
    assert result.geometry_solution.local_frame_id is source.local_frame_id
    assert result.geometry_solution.metrics.observations == ()


def test_conversion_does_not_mutate_legacy_source_or_child_objects() -> None:
    source = _estimate()
    original_calibrations = source.camera_calibrations
    original_poses = source.camera_poses
    original_points = source.points3d
    original_tracks = tuple(point.track for point in source.points3d)
    original_provenances = (
        source.provenance,
        *(calibration.provenance for calibration in source.camera_calibrations),
        *(pose.provenance for pose in source.camera_poses),
        *(point.provenance for point in source.points3d),
    )

    convert_sparse_reconstruction_estimate(source)

    assert source.camera_calibrations is original_calibrations
    assert source.camera_poses is original_poses
    assert source.points3d is original_points
    assert tuple(point.track for point in source.points3d) == original_tracks
    assert (
        source.provenance,
        *(calibration.provenance for calibration in source.camera_calibrations),
        *(pose.provenance for pose in source.camera_poses),
        *(point.provenance for point in source.points3d),
    ) == original_provenances


def test_source_only_legacy_evidence_is_not_promoted_into_canonical_outputs() -> None:
    source = _estimate(has_prior_focal_length=True)
    assert any(point.reprojection_error_px is not None for point in source.points3d)
    assert all(point.track for point in source.points3d)

    result = convert_sparse_reconstruction_estimate(source)

    for solution in result.camera_solutions:
        forbidden_camera_attributes = {
            "camera_id",
            "calibration_id",
            "has_prior_focal_length",
            "provenance",
            "track",
            "feature_index",
            "reprojection_error_px",
            "solver",
            "adapter",
            "model",
            "checkpoint",
            "world_transform",
            "gps",
            "metadata",
        }
        assert all(not hasattr(solution, attribute) for attribute in forbidden_camera_attributes)

    forbidden_point_attributes = {
        "tracks",
        "track",
        "feature_indices",
        "feature_index",
        "reprojection_errors",
        "reprojection_error_px",
        "provenance",
        "point_ids",
        "solver",
        "adapter",
        "model",
        "checkpoint",
        "world_transform",
        "gps",
        "metadata",
    }
    assert all(not hasattr(result.point_map, attribute) for attribute in forbidden_point_attributes)

    forbidden_geometry_attributes = {
        "tracks",
        "correspondences",
        "provenance",
        "scale_factor",
        "world_transform",
        "gps",
        "anchor",
        "fragment_id",
        "surface",
        "mesh",
        "solver",
        "adapter",
        "model",
        "checkpoint",
        "metadata",
    }
    assert all(
        not hasattr(result.geometry_solution, attribute)
        for attribute in forbidden_geometry_attributes
    )


def test_conversion_module_has_no_io_solver_tensor_or_persistence_surface() -> None:
    forbidden_symbols = {
        "Path",
        "os",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "numpy",
        "torch",
        "open3d",
        "pycolmap",
        "CameraId",
        "DepthField",
        "SurfaceModel",
        "SpatialFragment",
        "ArtifactStore",
        "Database",
    }

    assert forbidden_symbols.isdisjoint(conversion_module.__dict__)


def test_conversion_result_contains_only_canonical_target_values() -> None:
    result = convert_sparse_reconstruction_estimate(_estimate())

    assert all(isinstance(solution, CameraSolution) for solution in result.camera_solutions)
    assert result.point_map.confidence is None
    assert result.geometry_solution.depth_field_ids == ()
