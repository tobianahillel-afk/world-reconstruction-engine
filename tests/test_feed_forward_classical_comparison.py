from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    DepthField,
    DepthFieldId,
    DepthValueConventionName,
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
    ImageDimensions,
    LocalFrameId,
    MetricVector,
    ObservationId,
)
from wre.reconstruction.feed_forward_camera_quality import (
    CameraPoseQualityRequest,
    FeedForwardCameraQualityRequest,
    evaluate_camera_pose_quality,
    evaluate_feed_forward_camera_quality,
)
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_POSE_METRIC_NAMES = (
    "geometry.camera.observation_coverage_ratio",
    "geometry.camera.relative_rotation_error_deg_median",
    "geometry.camera.relative_translation_direction_error_deg_median",
    "geometry.camera.translation_pair_coverage_ratio",
)


def _camera(
    observation: str,
    *,
    frame: str,
    center_x: float,
    projection: str = "pinhole",
    intrinsics: tuple[float, ...] = (100.0, 100.0, 50.0, 40.0),
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{frame}:{observation}"),
        observation_id=ObservationId(observation),
        local_frame_id=LocalFrameId(f"frame:{frame}"),
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=100, height_px=80),
        intrinsic_parameters=intrinsics,
        rotation_matrix=_IDENTITY,
        translation_xyz=(-center_x, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )


def _geometry(*cameras: CameraSolution) -> FeedForwardGeometryResult:
    ordered = tuple(sorted(cameras, key=lambda camera: camera.observation_id.value))
    depths = tuple(
        DepthField(
            depth_field_id=DepthFieldId(f"depth:{camera.observation_id.value}"),
            observation_id=camera.observation_id,
            camera_solution_id=camera.solution_id,
            dimensions=camera.dimensions,
            depth_value_convention=DepthValueConventionName("relative-depth"),
            depth_values=(1.0,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            validity=(True,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            confidence=None,
            metrics=MetricVector(observations=()),
        )
        for camera in ordered
    )
    solution = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:comparison"),
        local_frame_id=ordered[0].local_frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(camera.solution_id for camera in ordered),
        depth_field_ids=tuple(depth.depth_field_id for depth in depths),
        point_map_ids=(),
        metrics=MetricVector(observations=()),
    )
    return FeedForwardGeometryResult(
        camera_solutions=ordered,
        depth_fields=depths,
        point_maps=(),
        geometry_solution=solution,
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact:camera-reference"),
        artifact_kind=ArtifactKind("geometry.camera_reference"),
    )


def _values(vector: MetricVector) -> dict[str, float]:
    return {item.descriptor.name.value: item.value for item in vector.observations}


def test_pose_request_is_frozen_and_allows_projection_independent_candidate_cameras() -> None:
    references = (
        _camera("obs:a", frame="reference", center_x=0.0),
        _camera("obs:b", frame="reference", center_x=1.0),
    )
    simple_radial = (
        _camera(
            "obs:a",
            frame="classical",
            center_x=0.0,
            projection="simple_radial",
            intrinsics=(100.0, 50.0, 40.0, 0.01),
        ),
        _camera(
            "obs:b",
            frame="classical",
            center_x=1.0,
            projection="simple_radial",
            intrinsics=(100.0, 50.0, 40.0, 0.01),
        ),
    )
    request = CameraPoseQualityRequest(
        candidate_cameras=simple_radial,
        reference_cameras=references,
        input_artifacts=(_artifact(),),
    )

    result = evaluate_camera_pose_quality(request)

    assert tuple(item.descriptor.name.value for item in result.observations) == _POSE_METRIC_NAMES
    assert all(value == 0.0 or value == 1.0 for value in _values(result).values())
    with pytest.raises(FrozenInstanceError):
        request.input_artifacts = ()  # type: ignore[misc]


def test_shared_pose_evaluator_exactly_matches_feed_forward_pose_subset() -> None:
    references = (
        _camera("obs:a", frame="reference", center_x=0.0),
        _camera("obs:b", frame="reference", center_x=1.0),
        _camera("obs:c", frame="reference", center_x=2.0),
    )
    candidate_cameras = (
        _camera("obs:a", frame="preview", center_x=0.0),
        _camera("obs:b", frame="preview", center_x=1.0),
        _camera("obs:c", frame="preview", center_x=2.0),
    )
    artifact = (_artifact(),)

    pose = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=candidate_cameras,
            reference_cameras=references,
            input_artifacts=artifact,
        )
    )
    feed_forward = evaluate_feed_forward_camera_quality(
        FeedForwardCameraQualityRequest(
            candidate=_geometry(*candidate_cameras),
            reference_cameras=references,
            input_artifacts=artifact,
        )
    )
    feed_forward_pose = {
        item.descriptor.name.value: item
        for item in feed_forward.observations
        if item.descriptor.name.value in _POSE_METRIC_NAMES
    }

    assert tuple(item.descriptor.name.value for item in pose.observations) == _POSE_METRIC_NAMES
    assert {item.descriptor.name.value: item for item in pose.observations} == feed_forward_pose


def test_pose_evaluator_keeps_zero_candidate_as_explicit_zero_coverage() -> None:
    reference = (_camera("obs:a", frame="reference", center_x=0.0),)

    result = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=(),
            reference_cameras=reference,
            input_artifacts=(),
        )
    )

    assert _values(result) == {"geometry.camera.observation_coverage_ratio": 0.0}


def test_pose_request_fails_closed_on_noncanonical_or_cross_frame_camera_sets() -> None:
    reference_a = _camera("obs:a", frame="reference", center_x=0.0)
    reference_b = _camera("obs:b", frame="reference", center_x=1.0)
    candidate_a = _camera("obs:a", frame="candidate", center_x=0.0)
    candidate_b = _camera("obs:b", frame="candidate", center_x=1.0)

    with pytest.raises(ValueError, match="canonical ObservationId"):
        CameraPoseQualityRequest(
            candidate_cameras=(candidate_b, candidate_a),
            reference_cameras=(reference_a, reference_b),
            input_artifacts=(),
        )
    with pytest.raises(ValueError, match="share one LocalFrameId"):
        CameraPoseQualityRequest(
            candidate_cameras=(
                candidate_a,
                _camera("obs:b", frame="other", center_x=1.0),
            ),
            reference_cameras=(reference_a, reference_b),
            input_artifacts=(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        CameraPoseQualityRequest(
            candidate_cameras=(candidate_a, candidate_b),
            reference_cameras=(),
            input_artifacts=(),
        )
