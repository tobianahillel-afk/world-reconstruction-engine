from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.feed_forward_camera_quality as camera_quality
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
    MetricDirection,
    MetricVector,
    ObservationId,
)
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_DIMENSIONS = ImageDimensions(width_px=100, height_px=80)
_INTRINSICS = (120.0, 125.0, 50.0, 40.0)


def _rz(degrees: float) -> tuple[tuple[float, float, float], ...]:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return (
        (cosine, -sine, 0.0),
        (sine, cosine, 0.0),
        (0.0, 0.0, 1.0),
    )


def _matmul(
    left: tuple[tuple[float, float, float], ...],
    right: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _transpose(
    matrix: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matvec(
    matrix: tuple[tuple[float, float, float], ...],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][axis] * vector[axis] for axis in range(3)),
        sum(matrix[1][axis] * vector[axis] for axis in range(3)),
        sum(matrix[2][axis] * vector[axis] for axis in range(3)),
    )


def _camera(
    observation: str,
    *,
    frame: str,
    rotation: tuple[tuple[float, float, float], ...] = _IDENTITY,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    intrinsics: tuple[float, ...] = _INTRINSICS,
    dimensions: ImageDimensions = _DIMENSIONS,
    projection: str = "pinhole",
    metrics: MetricVector | None = None,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{frame}:{observation}"),
        observation_id=ObservationId(observation),
        local_frame_id=LocalFrameId(f"frame:{frame}"),
        projection_model=CameraProjectionModelName(projection),
        dimensions=dimensions,
        intrinsic_parameters=intrinsics,
        rotation_matrix=cast(Any, rotation),
        translation_xyz=translation,
        uncertainty_artifacts=(),
        metrics=metrics if metrics is not None else MetricVector(observations=()),
    )


def _geometry(*cameras: CameraSolution) -> FeedForwardGeometryResult:
    ordered = tuple(sorted(cameras, key=lambda camera: camera.observation_id.value))
    local_frame = ordered[0].local_frame_id
    depths = tuple(
        DepthField(
            depth_field_id=DepthFieldId(f"depth:{camera.observation_id.value}"),
            observation_id=camera.observation_id,
            camera_solution_id=camera.solution_id,
            dimensions=camera.dimensions,
            depth_value_convention=DepthValueConventionName("relative"),
            depth_values=(1.0,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            validity=(True,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            confidence=None,
            metrics=MetricVector(observations=()),
        )
        for camera in ordered
    )
    solution = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:test"),
        local_frame_id=local_frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(
            sorted((camera.solution_id for camera in ordered), key=lambda item: item.value)
        ),
        depth_field_ids=tuple(
            sorted((depth.depth_field_id for depth in depths), key=lambda item: item.value)
        ),
        point_map_ids=(),
        metrics=MetricVector(observations=()),
    )
    return FeedForwardGeometryResult(
        camera_solutions=ordered,
        depth_fields=depths,
        point_maps=(),
        geometry_solution=solution,
    )


def _artifact(identifier: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind("geometry.camera_reference"),
    )


def _request(
    candidate: FeedForwardGeometryResult,
    references: tuple[CameraSolution, ...],
    artifacts: tuple[ArtifactRef, ...] = (),
) -> camera_quality.FeedForwardCameraQualityRequest:
    return camera_quality.FeedForwardCameraQualityRequest(
        candidate=candidate,
        reference_cameras=references,
        input_artifacts=artifacts,
    )


def _metrics(
    result: MetricVector,
) -> dict[str, float]:
    return {
        item.descriptor.name.value: item.value
        for item in result.observations
    }


def _pose_from_center(
    observation: str,
    center: tuple[float, float, float],
    *,
    frame: str,
    rotation: tuple[tuple[float, float, float], ...] = _IDENTITY,
    intrinsics: tuple[float, ...] = _INTRINSICS,
) -> CameraSolution:
    rotated_center = _matvec(rotation, center)
    return _camera(
        observation,
        frame=frame,
        rotation=rotation,
        translation=(
            -rotated_center[0],
            -rotated_center[1],
            -rotated_center[2],
        ),
        intrinsics=intrinsics,
    )


def _transform_local_frame(
    camera: CameraSolution,
    *,
    frame: str,
    scale: float,
    rotation: tuple[tuple[float, float, float], ...],
    translation: tuple[float, float, float],
) -> CameraSolution:
    # New local coordinates are x' = scale * Q * x + a.
    # Camera coordinates are scaled by the same positive scalar, so the equivalent
    # camera-from-local pose is R' = R Q^T and t' = scale*t - R' a.
    new_rotation = _matmul(camera.rotation_matrix, _transpose(rotation))
    rotated_offset = _matvec(new_rotation, translation)
    new_translation = (
        scale * camera.translation_xyz[0] - rotated_offset[0],
        scale * camera.translation_xyz[1] - rotated_offset[1],
        scale * camera.translation_xyz[2] - rotated_offset[2],
    )
    return _camera(
        camera.observation_id.value,
        frame=frame,
        rotation=new_rotation,
        translation=new_translation,
        intrinsics=camera.intrinsic_parameters,
        dimensions=camera.dimensions,
        projection=camera.projection_model.value,
        metrics=camera.metrics,
    )


def test_request_is_exact_frozen_and_preserves_ordered_provenance() -> None:
    candidate = _geometry(_camera("obs:a", frame="candidate"))
    references = (_camera("obs:a", frame="reference"),)
    artifacts = (_artifact("artifact:b"), _artifact("artifact:a"))

    request = _request(candidate, references, artifacts)

    assert tuple(
        field.name for field in fields(camera_quality.FeedForwardCameraQualityRequest)
    ) == (
        "candidate",
        "reference_cameras",
        "input_artifacts",
    )
    assert request.candidate is candidate
    assert request.reference_cameras is references
    assert request.input_artifacts == artifacts
    with pytest.raises(FrozenInstanceError):
        request.input_artifacts = ()  # type: ignore[misc]


def test_request_rejects_invalid_reference_shapes_and_artifact_types() -> None:
    candidate = _geometry(_camera("obs:a", frame="candidate"))
    ref_a = _camera("obs:a", frame="reference")
    ref_b = _camera("obs:b", frame="reference")

    with pytest.raises(ValueError, match="non-empty"):
        _request(candidate, ())
    with pytest.raises(TypeError, match="members must be CameraSolution"):
        camera_quality.FeedForwardCameraQualityRequest(
            candidate=candidate,
            reference_cameras=cast(Any, ("camera",)),
            input_artifacts=(),
        )
    with pytest.raises(ValueError, match="canonical"):
        _request(candidate, (ref_b, ref_a))
    with pytest.raises(ValueError, match="unique"):
        _request(candidate, (ref_a, ref_a))
    with pytest.raises(ValueError, match="share one"):
        _request(
            candidate,
            (
                ref_a,
                _camera("obs:b", frame="other-reference"),
            ),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        camera_quality.FeedForwardCameraQualityRequest(
            candidate=candidate,
            reference_cameras=(ref_a,),
            input_artifacts=cast(Any, [_artifact("artifact:a")]),
        )
    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        camera_quality.FeedForwardCameraQualityRequest(
            candidate=candidate,
            reference_cameras=(ref_a,),
            input_artifacts=cast(Any, ("artifact",)),
        )
    with pytest.raises(TypeError, match="candidate"):
        camera_quality.FeedForwardCameraQualityRequest(
            candidate=cast(Any, "candidate"),
            reference_cameras=(ref_a,),
            input_artifacts=(),
        )


def test_identity_candidate_and_reference_emit_exact_zero_errors() -> None:
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:c", (0.0, 2.0, 0.0), frame="reference"),
    )
    candidate_cameras = tuple(
        _pose_from_center(
            camera.observation_id.value,
            (
                -camera.translation_xyz[0],
                -camera.translation_xyz[1],
                -camera.translation_xyz[2],
            ),
            frame="candidate",
        )
        for camera in references
    )
    result = camera_quality.evaluate_feed_forward_camera_quality(
        _request(_geometry(*candidate_cameras), references)
    )
    values = _metrics(result)

    assert values == {
        "geometry.camera.focal_relative_error_median": 0.0,
        "geometry.camera.observation_coverage_ratio": 1.0,
        "geometry.camera.principal_point_error_normalized_median": 0.0,
        "geometry.camera.relative_rotation_error_deg_median": 0.0,
        "geometry.camera.relative_translation_direction_error_deg_median": 0.0,
        "geometry.camera.translation_pair_coverage_ratio": 1.0,
    }


def test_observation_coverage_uses_reference_denominator_only() -> None:
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:c", (0.0, 1.0, 0.0), frame="reference"),
    )
    candidate = _geometry(
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="candidate"),
        _pose_from_center("obs:c", (0.0, 1.0, 0.0), frame="candidate"),
        _pose_from_center("obs:extra", (3.0, 0.0, 0.0), frame="candidate"),
    )

    values = _metrics(
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(candidate, references)
        )
    )

    assert values["geometry.camera.observation_coverage_ratio"] == pytest.approx(2.0 / 3.0)
    assert "obs:extra" not in {
        camera.observation_id.value for camera in references
    }


def test_asymmetric_pairwise_rotation_uses_deterministic_median() -> None:
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:c", (0.0, 1.0, 0.0), frame="reference"),
    )
    candidate = _geometry(
        _pose_from_center(
            "obs:a",
            (0.0, 0.0, 0.0),
            frame="candidate",
            rotation=_rz(0.0),
        ),
        _pose_from_center(
            "obs:b",
            (1.0, 0.0, 0.0),
            frame="candidate",
            rotation=_rz(10.0),
        ),
        _pose_from_center(
            "obs:c",
            (0.0, 1.0, 0.0),
            frame="candidate",
            rotation=_rz(30.0),
        ),
    )

    values = _metrics(
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(candidate, references)
        )
    )

    assert values["geometry.camera.relative_rotation_error_deg_median"] == pytest.approx(
        20.0,
        abs=1e-9,
    )


def test_relative_pose_metrics_are_invariant_to_rigid_frame_and_positive_scale() -> None:
    references = (
        _pose_from_center(
            "obs:a",
            (0.0, 0.0, 0.0),
            frame="reference",
            rotation=_rz(5.0),
        ),
        _pose_from_center(
            "obs:b",
            (2.0, 1.0, 0.0),
            frame="reference",
            rotation=_rz(-15.0),
        ),
        _pose_from_center(
            "obs:c",
            (-1.0, 3.0, 0.5),
            frame="reference",
            rotation=_rz(25.0),
        ),
    )
    transformed = tuple(
        _transform_local_frame(
            camera,
            frame="candidate",
            scale=7.5,
            rotation=_rz(37.0),
            translation=(11.0, -4.0, 3.0),
        )
        for camera in references
    )

    values = _metrics(
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(_geometry(*transformed), references)
        )
    )

    assert values["geometry.camera.relative_rotation_error_deg_median"] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert values[
        "geometry.camera.relative_translation_direction_error_deg_median"
    ] == pytest.approx(0.0, abs=1e-6)
    assert values["geometry.camera.translation_pair_coverage_ratio"] == 1.0


def test_translation_direction_error_and_coincident_pair_coverage() -> None:
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:c", (0.0, 0.0, 0.0), frame="reference"),
    )
    candidate = _geometry(
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="candidate"),
        _pose_from_center("obs:b", (0.0, 1.0, 0.0), frame="candidate"),
        _pose_from_center("obs:c", (0.0, 0.0, 0.0), frame="candidate"),
    )

    values = _metrics(
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(candidate, references)
        )
    )

    assert values["geometry.camera.translation_pair_coverage_ratio"] == pytest.approx(
        2.0 / 3.0
    )
    assert values[
        "geometry.camera.relative_translation_direction_error_deg_median"
    ] == pytest.approx(90.0)


def test_intrinsic_errors_have_explicit_axis_and_diagonal_semantics() -> None:
    dimensions = ImageDimensions(width_px=100, height_px=100)
    reference = _camera(
        "obs:a",
        frame="reference",
        intrinsics=(100.0, 200.0, 50.0, 50.0),
        dimensions=dimensions,
    )
    candidate = _camera(
        "obs:a",
        frame="candidate",
        intrinsics=(110.0, 180.0, 60.0, 50.0),
        dimensions=dimensions,
    )

    values = _metrics(
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(_geometry(candidate), (reference,))
        )
    )

    assert values["geometry.camera.focal_relative_error_median"] == pytest.approx(0.1)
    assert values[
        "geometry.camera.principal_point_error_normalized_median"
    ] == pytest.approx(10.0 / math.hypot(100.0, 100.0))


@pytest.mark.parametrize(
    ("candidate_kwargs", "reference_kwargs", "message"),
    [
        ({"projection": "fisheye"}, {}, "candidate camera must use pinhole"),
        ({}, {"projection": "fisheye"}, "reference camera must use pinhole"),
        ({"intrinsics": (100.0, 100.0, 50.0)}, {}, "candidate pinhole camera"),
        ({}, {"intrinsics": (100.0, 100.0, 50.0)}, "reference pinhole camera"),
        (
            {"dimensions": ImageDimensions(width_px=101, height_px=80)},
            {},
            "dimensions must match",
        ),
        ({}, {"intrinsics": (0.0, 125.0, 50.0, 40.0)}, "strictly positive"),
    ],
)
def test_shared_intrinsic_contract_fails_closed(
    candidate_kwargs: dict[str, Any],
    reference_kwargs: dict[str, Any],
    message: str,
) -> None:
    candidate = _camera("obs:a", frame="candidate", **candidate_kwargs)
    reference = _camera("obs:a", frame="reference", **reference_kwargs)

    with pytest.raises(ValueError, match=message):
        camera_quality.evaluate_feed_forward_camera_quality(
            _request(_geometry(candidate), (reference,))
        )


def test_zero_shared_emits_only_coverage_and_one_shared_has_no_pair_metrics() -> None:
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
    )

    zero = camera_quality.evaluate_feed_forward_camera_quality(
        _request(
            _geometry(
                _pose_from_center("obs:x", (0.0, 0.0, 0.0), frame="candidate")
            ),
            references,
        )
    )
    assert _metrics(zero) == {"geometry.camera.observation_coverage_ratio": 0.0}

    one = camera_quality.evaluate_feed_forward_camera_quality(
        _request(
            _geometry(
                _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="candidate")
            ),
            references,
        )
    )
    one_values = _metrics(one)
    assert set(one_values) == {
        "geometry.camera.focal_relative_error_median",
        "geometry.camera.observation_coverage_ratio",
        "geometry.camera.principal_point_error_normalized_median",
    }
    assert one_values["geometry.camera.observation_coverage_ratio"] == 0.5


def test_metric_descriptors_and_provenance_are_exact_and_canonical() -> None:
    artifacts = (_artifact("artifact:z"), _artifact("artifact:a"))
    references = (
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="reference"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="reference"),
    )
    candidate = _geometry(
        _pose_from_center("obs:a", (0.0, 0.0, 0.0), frame="candidate"),
        _pose_from_center("obs:b", (1.0, 0.0, 0.0), frame="candidate"),
    )

    result = camera_quality.evaluate_feed_forward_camera_quality(
        _request(candidate, references, artifacts)
    )

    names = tuple(item.descriptor.name.value for item in result.observations)
    assert names == tuple(sorted(names))
    assert names == (
        "geometry.camera.focal_relative_error_median",
        "geometry.camera.observation_coverage_ratio",
        "geometry.camera.principal_point_error_normalized_median",
        "geometry.camera.relative_rotation_error_deg_median",
        "geometry.camera.relative_translation_direction_error_deg_median",
        "geometry.camera.translation_pair_coverage_ratio",
    )
    assert all(
        item.descriptor.dimension.value == "geometry.camera"
        for item in result.observations
    )
    assert all(
        item.provenance.evaluator
        == camera_quality.FEED_FORWARD_CAMERA_QUALITY_EVALUATOR
        for item in result.observations
    )
    assert all(
        item.provenance.input_artifacts == artifacts
        for item in result.observations
    )

    descriptors = {item.descriptor.name.value: item.descriptor for item in result.observations}
    assert descriptors["geometry.camera.observation_coverage_ratio"].direction is (
        MetricDirection.HIGHER_IS_BETTER
    )
    assert descriptors["geometry.camera.translation_pair_coverage_ratio"].direction is (
        MetricDirection.HIGHER_IS_BETTER
    )
    assert descriptors["geometry.camera.relative_rotation_error_deg_median"].unit.value == (
        "degree"
    )


def test_evaluation_does_not_mutate_candidate_reference_or_existing_metrics() -> None:
    existing = MetricVector(observations=())
    candidate_camera = _camera("obs:a", frame="candidate", metrics=existing)
    reference_camera = _camera("obs:a", frame="reference", metrics=existing)
    candidate = _geometry(candidate_camera)
    candidate_before = candidate
    reference_before = reference_camera

    camera_quality.evaluate_feed_forward_camera_quality(
        _request(candidate, (reference_camera,))
    )

    assert candidate == candidate_before
    assert candidate.camera_solutions[0] is candidate_camera
    assert candidate.camera_solutions[0].metrics is existing
    assert reference_camera == reference_before
    assert reference_camera.metrics is existing


def test_evaluator_returns_metrics_only_without_quality_decision_surface() -> None:
    reference = _camera("obs:a", frame="reference")
    candidate = _geometry(_camera("obs:a", frame="candidate"))

    result = camera_quality.evaluate_feed_forward_camera_quality(
        _request(candidate, (reference,))
    )

    assert isinstance(result, MetricVector)
    for attribute in (
        "decision",
        "quality_policy",
        "threshold",
        "winner",
        "route",
        "fallback",
        "retry",
        "colmap",
    ):
        assert not hasattr(result, attribute)


def test_module_surface_contains_no_later_or_candidate_specific_dependencies() -> None:
    source = inspect.getsource(camera_quality)

    for forbidden in (
        "Da3",
        "da3_",
        "COLMAP",
        "colmap",
        "QualityDecision",
        "QualityPolicy",
        "QualityEvaluation",
        "FailureCategory",
        "V2L13.5",
        "benchmark",
        "winner",
        "fallback",
        "retry",
        "torch",
        "numpy",
    ):
        assert forbidden not in source
