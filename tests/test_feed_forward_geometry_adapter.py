from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.feed_forward_geometry as feed_forward_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
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
from wre.domain.observations import ObservationId
from wre.domain.point_maps import PointMap, PointMapId
from wre.reconstruction.feed_forward_geometry import (
    FEED_FORWARD_GEOMETRY_CAPABILITY,
    FEED_FORWARD_GEOMETRY_CAPABILITY_NAME,
    FEED_FORWARD_GEOMETRY_INPUT_KIND,
    FEED_FORWARD_GEOMETRY_OUTPUT_KINDS,
    FeedForwardGeometryResult,
    normalize_feed_forward_geometry_inputs,
)


def _empty_metrics() -> MetricVector:
    return MetricVector(observations=())


def _artifact(value: str, kind: str = "media.decoded_image_pyramid") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(value),
        artifact_kind=ArtifactKind(kind),
    )


def _camera(
    observation: str,
    solution_id: str,
    local_frame: LocalFrameId,
    *,
    dimensions: ImageDimensions | None = None,
    translation_x: float = 0.0,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(solution_id),
        observation_id=ObservationId(observation),
        local_frame_id=local_frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=dimensions or ImageDimensions(width_px=2, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(translation_x, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=_empty_metrics(),
    )


def _depth(
    observation: str,
    depth_id: str,
    camera: CameraSolution,
    *,
    dimensions: ImageDimensions | None = None,
) -> DepthField:
    actual_dimensions = dimensions or camera.dimensions
    pixel_count = actual_dimensions.width_px * actual_dimensions.height_px
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=ObservationId(observation),
        camera_solution_id=camera.solution_id,
        dimensions=actual_dimensions,
        depth_value_convention=DepthValueConventionName("camera-z"),
        depth_values=tuple(float(index + 1) for index in range(pixel_count)),
        validity=tuple(True for _ in range(pixel_count)),
        confidence=None,
        metrics=_empty_metrics(),
    )


def _point_map(
    point_map_id: str,
    local_frame: LocalFrameId,
    observations: tuple[str, ...],
) -> PointMap:
    return PointMap(
        point_map_id=PointMapId(point_map_id),
        local_frame_id=local_frame,
        source_observation_ids=tuple(ObservationId(value) for value in observations),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_empty_metrics(),
    )


def _geometry(
    local_frame: LocalFrameId,
    cameras: tuple[CameraSolution, ...],
    depths: tuple[DepthField, ...],
    points: tuple[PointMap, ...],
    *,
    scale_status: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    camera_ids: tuple[CameraSolutionId, ...] | None = None,
    depth_ids: tuple[DepthFieldId, ...] | None = None,
    point_ids: tuple[PointMapId, ...] | None = None,
) -> GeometrySolution:
    return GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:feed-forward:test"),
        local_frame_id=local_frame,
        scale_status=scale_status,
        camera_solution_ids=(
            camera_ids
            if camera_ids is not None
            else tuple(sorted((item.solution_id for item in cameras), key=lambda item: item.value))
        ),
        depth_field_ids=(
            depth_ids
            if depth_ids is not None
            else tuple(sorted((item.depth_field_id for item in depths), key=lambda item: item.value))
        ),
        point_map_ids=(
            point_ids
            if point_ids is not None
            else tuple(sorted((item.point_map_id for item in points), key=lambda item: item.value))
        ),
        metrics=_empty_metrics(),
    )


def _two_camera_fixture(
    *,
    scale_status: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    include_point_map: bool = False,
) -> tuple[
    LocalFrameId,
    tuple[CameraSolution, ...],
    tuple[DepthField, ...],
    tuple[PointMap, ...],
    GeometrySolution,
]:
    local_frame = LocalFrameId("frame:feed-forward:test")
    camera_a = _camera("obs:a", "camera:z", local_frame)
    camera_b = _camera("obs:b", "camera:a", local_frame, translation_x=1.0)
    cameras = (camera_a, camera_b)
    depths = (
        _depth("obs:a", "depth:z", camera_a),
        _depth("obs:b", "depth:a", camera_b),
    )
    points = (
        (_point_map("points:a", local_frame, ("obs:a", "obs:b")),)
        if include_point_map
        else ()
    )
    geometry = _geometry(
        local_frame,
        cameras,
        depths,
        points,
        scale_status=scale_status,
    )
    return local_frame, cameras, depths, points, geometry


def test_feed_forward_capability_is_pure_and_uses_canonical_geometry_kinds() -> None:
    assert FEED_FORWARD_GEOMETRY_CAPABILITY_NAME.value == "geometry.feed_forward"
    assert FEED_FORWARD_GEOMETRY_INPUT_KIND == ArtifactKind("media.decoded_image_pyramid")
    assert FEED_FORWARD_GEOMETRY_CAPABILITY.capability is (
        FEED_FORWARD_GEOMETRY_CAPABILITY_NAME
    )
    assert FEED_FORWARD_GEOMETRY_CAPABILITY.input_kinds == frozenset(
        {ArtifactKind("media.decoded_image_pyramid")}
    )
    assert FEED_FORWARD_GEOMETRY_OUTPUT_KINDS == frozenset(
        {
            ArtifactKind("geometry.camera_solution"),
            ArtifactKind("geometry.depth_field"),
            ArtifactKind("geometry.point_map"),
            ArtifactKind("geometry.solution"),
        }
    )
    assert FEED_FORWARD_GEOMETRY_CAPABILITY.output_kinds is (
        FEED_FORWARD_GEOMETRY_OUTPUT_KINDS
    )
    assert ArtifactKind("geometry.sparse_reconstruction_estimate") not in (
        FEED_FORWARD_GEOMETRY_OUTPUT_KINDS
    )


def test_normalize_inputs_retains_exact_canonical_tuple() -> None:
    inputs = (_artifact("decoded:a"), _artifact("decoded:b"))

    result = normalize_feed_forward_geometry_inputs(inputs)

    assert result is inputs


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (cast(Any, [_artifact("decoded:a")]), TypeError, "immutable tuple"),
        ((), ValueError, "at least one artifact"),
        (cast(Any, ("raw",)), TypeError, "only ArtifactRef"),
        ((_artifact("decoded:a", "image.observation"),), ValueError, "decoded_image_pyramid"),
        ((_artifact("decoded:a"), _artifact("decoded:a")), ValueError, "duplicate"),
        ((_artifact("decoded:b"), _artifact("decoded:a")), ValueError, "canonical ArtifactRef"),
    ],
)
def test_normalize_inputs_fails_closed_without_sorting_or_coercion(
    inputs: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        normalize_feed_forward_geometry_inputs(inputs)


def test_result_has_exact_frozen_field_shape() -> None:
    assert tuple(field.name for field in fields(FeedForwardGeometryResult)) == (
        "camera_solutions",
        "depth_fields",
        "point_maps",
        "geometry_solution",
    )

    _, cameras, depths, points, geometry = _two_camera_fixture()
    result = FeedForwardGeometryResult(cameras, depths, points, geometry)

    with pytest.raises(FrozenInstanceError):
        result.geometry_solution = geometry  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("camera_solutions", cast(Any, []), "camera_solutions"),
        ("camera_solutions", cast(Any, ("camera",)), "CameraSolution"),
        ("depth_fields", cast(Any, []), "depth_fields"),
        ("depth_fields", cast(Any, ("depth",)), "DepthField"),
        ("point_maps", cast(Any, []), "point_maps"),
        ("point_maps", cast(Any, ("points",)), "PointMap"),
        ("geometry_solution", cast(Any, "geometry"), "GeometrySolution"),
    ],
)
def test_result_rejects_wrong_child_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    _, cameras, depths, points, geometry = _two_camera_fixture()
    kwargs: dict[str, Any] = {
        "camera_solutions": cameras,
        "depth_fields": depths,
        "point_maps": points,
        "geometry_solution": geometry,
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        FeedForwardGeometryResult(**kwargs)


def test_unresolved_depth_only_result_preserves_exact_children_and_references() -> None:
    local_frame, cameras, depths, points, geometry = _two_camera_fixture()

    result = FeedForwardGeometryResult(
        camera_solutions=cameras,
        depth_fields=depths,
        point_maps=points,
        geometry_solution=geometry,
    )

    assert result.camera_solutions is cameras
    assert result.depth_fields is depths
    assert result.point_maps is points
    assert result.geometry_solution is geometry
    assert result.geometry_solution.local_frame_id is local_frame
    assert result.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.geometry_solution.camera_solution_ids == (
        CameraSolutionId("camera:a"),
        CameraSolutionId("camera:z"),
    )
    assert result.geometry_solution.depth_field_ids == (
        DepthFieldId("depth:a"),
        DepthFieldId("depth:z"),
    )
    assert result.geometry_solution.point_map_ids == ()


def test_optional_point_map_is_supported_without_becoming_mandatory() -> None:
    local_frame, cameras, depths, points, geometry = _two_camera_fixture(
        include_point_map=True
    )

    result = FeedForwardGeometryResult(cameras, depths, points, geometry)

    assert result.point_maps is points
    assert result.point_maps[0].local_frame_id is local_frame
    assert result.point_maps[0].source_observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )
    assert result.geometry_solution.point_map_ids == (PointMapId("points:a"),)


def test_zero_emitted_geometry_carriers_fail_closed() -> None:
    local_frame = LocalFrameId("frame:feed-forward:test")
    camera = _camera("obs:a", "camera:a", local_frame)
    geometry = _geometry(
        local_frame,
        (camera,),
        (),
        (),
        point_ids=(PointMapId("points:foreign"),),
    )

    with pytest.raises(ValueError, match="at least one DepthField or PointMap"):
        FeedForwardGeometryResult((camera,), (), (), geometry)


def test_camera_observations_must_be_unique_and_canonically_ordered() -> None:
    local_frame = LocalFrameId("frame:feed-forward:test")
    camera_a = _camera("obs:a", "camera:a", local_frame)
    camera_b = _camera("obs:b", "camera:b", local_frame)
    depth_a = _depth("obs:a", "depth:a", camera_a)
    depth_b = _depth("obs:b", "depth:b", camera_b)
    geometry = _geometry(local_frame, (camera_a, camera_b), (depth_a, depth_b), ())

    with pytest.raises(ValueError, match="canonical ObservationId order"):
        FeedForwardGeometryResult(
            (camera_b, camera_a),
            (depth_a, depth_b),
            (),
            geometry,
        )

    duplicate_camera = _camera("obs:a", "camera:c", local_frame)
    with pytest.raises(ValueError, match="camera observations must be unique"):
        FeedForwardGeometryResult(
            (camera_a, duplicate_camera),
            (depth_a,),
            (),
            _geometry(local_frame, (camera_a, duplicate_camera), (depth_a,), ()),
        )


def test_camera_and_point_local_frames_must_match_geometry_frame() -> None:
    local_frame, cameras, depths, _, geometry = _two_camera_fixture()
    foreign_frame = LocalFrameId("frame:feed-forward:foreign")
    foreign_camera = _camera("obs:b", "camera:a", foreign_frame, translation_x=1.0)

    with pytest.raises(ValueError, match="cameras and GeometrySolution"):
        FeedForwardGeometryResult(
            (cameras[0], foreign_camera),
            depths,
            (),
            geometry,
        )

    foreign_point = _point_map(
        "points:a",
        foreign_frame,
        ("obs:a",),
    )
    geometry_with_point = _geometry(
        local_frame,
        cameras,
        depths,
        (foreign_point,),
    )
    with pytest.raises(ValueError, match="PointMaps and GeometrySolution"):
        FeedForwardGeometryResult(
            cameras,
            depths,
            (foreign_point,),
            geometry_with_point,
        )


def test_depth_must_match_exact_emitted_camera_and_dimensions() -> None:
    local_frame = LocalFrameId("frame:feed-forward:test")
    camera_a = _camera("obs:a", "camera:a", local_frame)
    camera_b = _camera("obs:b", "camera:b", local_frame)
    foreign_observation_depth = DepthField(
        depth_field_id=DepthFieldId("depth:c"),
        observation_id=ObservationId("obs:c"),
        camera_solution_id=camera_a.solution_id,
        dimensions=camera_a.dimensions,
        depth_value_convention=DepthValueConventionName("camera-z"),
        depth_values=(1.0, 2.0),
        validity=(True, True),
        confidence=None,
        metrics=_empty_metrics(),
    )
    geometry_foreign = _geometry(
        local_frame,
        (camera_a, camera_b),
        (foreign_observation_depth,),
        (),
    )
    with pytest.raises(ValueError, match="observation must have an emitted CameraSolution"):
        FeedForwardGeometryResult(
            (camera_a, camera_b),
            (foreign_observation_depth,),
            (),
            geometry_foreign,
        )

    wrong_reference = DepthField(
        depth_field_id=DepthFieldId("depth:a"),
        observation_id=camera_a.observation_id,
        camera_solution_id=camera_b.solution_id,
        dimensions=camera_a.dimensions,
        depth_value_convention=DepthValueConventionName("camera-z"),
        depth_values=(1.0, 2.0),
        validity=(True, True),
        confidence=None,
        metrics=_empty_metrics(),
    )
    with pytest.raises(ValueError, match="reference its emitted CameraSolution"):
        FeedForwardGeometryResult(
            (camera_a, camera_b),
            (wrong_reference,),
            (),
            _geometry(local_frame, (camera_a, camera_b), (wrong_reference,), ()),
        )

    wrong_dimensions = _depth(
        "obs:a",
        "depth:a",
        camera_a,
        dimensions=ImageDimensions(width_px=1, height_px=1),
    )
    with pytest.raises(ValueError, match="dimensions must match"):
        FeedForwardGeometryResult(
            (camera_a, camera_b),
            (wrong_dimensions,),
            (),
            _geometry(local_frame, (camera_a, camera_b), (wrong_dimensions,), ()),
        )


def test_point_map_support_must_stay_inside_camera_membership() -> None:
    local_frame, cameras, depths, _, _ = _two_camera_fixture()
    foreign_point = _point_map(
        "points:a",
        local_frame,
        ("obs:a", "obs:foreign"),
    )
    geometry = _geometry(local_frame, cameras, depths, (foreign_point,))

    with pytest.raises(ValueError, match="support must stay inside camera membership"):
        FeedForwardGeometryResult(cameras, depths, (foreign_point,), geometry)


def test_geometry_solution_must_reference_exact_emitted_children() -> None:
    local_frame, cameras, depths, points, _ = _two_camera_fixture(include_point_map=True)

    missing_camera = _geometry(
        local_frame,
        cameras,
        depths,
        points,
        camera_ids=(CameraSolutionId("camera:a"),),
    )
    with pytest.raises(ValueError, match="exactly all emitted cameras"):
        FeedForwardGeometryResult(cameras, depths, points, missing_camera)

    missing_depth = _geometry(
        local_frame,
        cameras,
        depths,
        points,
        depth_ids=(DepthFieldId("depth:a"),),
    )
    with pytest.raises(ValueError, match="exactly all emitted depths"):
        FeedForwardGeometryResult(cameras, depths, points, missing_depth)

    foreign_point = _geometry(
        local_frame,
        cameras,
        depths,
        points,
        point_ids=(PointMapId("points:foreign"),),
    )
    with pytest.raises(ValueError, match="exactly all emitted point maps"):
        FeedForwardGeometryResult(cameras, depths, points, foreign_point)


def test_metric_status_is_preserved_without_rescaling_or_conversion() -> None:
    local_frame, cameras, depths, points, geometry = _two_camera_fixture(
        scale_status=GeometryScaleStatus.METRIC,
        include_point_map=True,
    )
    original_translations = tuple(item.translation_xyz for item in cameras)
    original_depth_values = tuple(item.depth_values for item in depths)
    original_positions = tuple(item.positions_xyz for item in points)

    result = FeedForwardGeometryResult(cameras, depths, points, geometry)

    assert result.geometry_solution.scale_status is GeometryScaleStatus.METRIC
    assert result.geometry_solution.local_frame_id is local_frame
    assert tuple(item.translation_xyz for item in result.camera_solutions) == (
        original_translations
    )
    assert tuple(item.depth_values for item in result.depth_fields) == original_depth_values
    assert tuple(item.positions_xyz for item in result.point_maps) == original_positions


def test_result_validation_does_not_reorder_or_clone_children() -> None:
    _, cameras, depths, points, geometry = _two_camera_fixture(include_point_map=True)

    first = FeedForwardGeometryResult(cameras, depths, points, geometry)
    second = FeedForwardGeometryResult(cameras, depths, points, geometry)

    assert first == second
    assert first.camera_solutions is cameras
    assert first.depth_fields is depths
    assert first.point_maps is points
    assert first.geometry_solution is geometry


def test_feed_forward_interface_module_has_no_model_execution_or_later_product_surface() -> None:
    forbidden_names = {
        "Path",
        "torch",
        "cuda",
        "tensorrt",
        "onnx",
        "jax",
        "numpy",
        "np",
        "requests",
        "socket",
        "subprocess",
        "open",
        "Model",
        "Checkpoint",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
        "QualityDecision",
        "Router",
        "reconstruct_colmap_incrementally",
        "reconstruct_colmap_globally",
    }

    assert forbidden_names.isdisjoint(vars(feed_forward_module))
