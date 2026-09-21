from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.geometry_solutions as geometry_solutions_module
from wre.domain import (
    CameraSolutionId,
    DepthFieldId,
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
    LocalFrameId,
    LocalScaleStatus,
    MetricVector,
    PointMapId,
)


def _geometry_solution(
    *,
    geometry_solution_id: GeometrySolutionId | None = None,
    local_frame_id: LocalFrameId | None = None,
    scale_status: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    camera_solution_ids: tuple[CameraSolutionId, ...] = (
        CameraSolutionId("camera:1"),
        CameraSolutionId("camera:2"),
    ),
    depth_field_ids: tuple[DepthFieldId, ...] = (DepthFieldId("depth:1"),),
    point_map_ids: tuple[PointMapId, ...] = (),
    metrics: MetricVector | None = None,
) -> GeometrySolution:
    return GeometrySolution(
        geometry_solution_id=geometry_solution_id or GeometrySolutionId("geometry:1"),
        local_frame_id=local_frame_id or LocalFrameId("local:1"),
        scale_status=scale_status,
        camera_solution_ids=camera_solution_ids,
        depth_field_ids=depth_field_ids,
        point_map_ids=point_map_ids,
        metrics=metrics or MetricVector(observations=()),
    )


def test_geometry_solution_id_is_typed_immutable_hashable_and_stringable() -> None:
    first = GeometrySolutionId("geometry:A-1")
    second = GeometrySolutionId("geometry:A-1")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "geometry:A-1"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", ".leading", " leading", "has/slash", "has space", "a" * 129, 1, True],
)
def test_geometry_solution_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        GeometrySolutionId(cast(Any, value))


def test_geometry_scale_status_has_exact_stable_members_and_values() -> None:
    assert tuple((member.name, member.value) for member in GeometryScaleStatus) == (
        ("UNRESOLVED", "unresolved"),
        ("METRIC", "metric"),
    )


def test_geometry_scale_status_is_v2_specific_and_rejects_legacy_enum() -> None:
    assert GeometryScaleStatus is not LocalScaleStatus

    with pytest.raises(TypeError, match="scale_status must be GeometryScaleStatus"):
        _geometry_solution(scale_status=cast(Any, LocalScaleStatus.UNRESOLVED))


def test_geometry_solution_has_exact_frozen_field_shape_and_retains_caller_objects() -> None:
    geometry_solution_id = GeometrySolutionId("geometry:retained")
    local_frame_id = LocalFrameId("local:retained")
    scale_status = GeometryScaleStatus.METRIC
    camera_solution_ids = (
        CameraSolutionId("camera:a"),
        CameraSolutionId("camera:b"),
    )
    depth_field_ids = (DepthFieldId("depth:a"),)
    point_map_ids = (PointMapId("points:a"),)
    metrics = MetricVector(observations=())

    solution = _geometry_solution(
        geometry_solution_id=geometry_solution_id,
        local_frame_id=local_frame_id,
        scale_status=scale_status,
        camera_solution_ids=camera_solution_ids,
        depth_field_ids=depth_field_ids,
        point_map_ids=point_map_ids,
        metrics=metrics,
    )

    assert tuple(field.name for field in fields(GeometrySolution)) == (
        "geometry_solution_id",
        "local_frame_id",
        "scale_status",
        "camera_solution_ids",
        "depth_field_ids",
        "point_map_ids",
        "metrics",
    )
    assert solution.geometry_solution_id is geometry_solution_id
    assert solution.local_frame_id is local_frame_id
    assert solution.scale_status is scale_status
    assert solution.camera_solution_ids is camera_solution_ids
    assert solution.depth_field_ids is depth_field_ids
    assert solution.point_map_ids is point_map_ids
    assert solution.metrics is metrics

    with pytest.raises(FrozenInstanceError):
        solution.local_frame_id = LocalFrameId("other")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        (
            "geometry_solution_id",
            "geometry",
            "geometry_solution_id must be GeometrySolutionId",
        ),
        ("local_frame_id", "local", "local_frame_id must be LocalFrameId"),
        ("scale_status", "metric", "scale_status must be GeometryScaleStatus"),
        ("metrics", (), "metrics must be MetricVector"),
    ],
)
def test_geometry_solution_rejects_wrong_core_types(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "geometry_solution_id": GeometrySolutionId("geometry:typed"),
        "local_frame_id": LocalFrameId("local:typed"),
        "scale_status": GeometryScaleStatus.UNRESOLVED,
        "camera_solution_ids": (CameraSolutionId("camera:1"),),
        "depth_field_ids": (DepthFieldId("depth:1"),),
        "point_map_ids": (),
        "metrics": MetricVector(observations=()),
    }
    values[field_name] = replacement

    with pytest.raises(TypeError, match=message):
        GeometrySolution(**cast(Any, values))


def test_camera_solution_ids_must_be_non_empty_immutable_and_canonical() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _geometry_solution(camera_solution_ids=())

    with pytest.raises(TypeError, match="immutable tuple"):
        _geometry_solution(camera_solution_ids=cast(Any, [CameraSolutionId("camera:1")]))

    with pytest.raises(TypeError, match="members must be CameraSolutionId"):
        _geometry_solution(
            camera_solution_ids=cast(Any, (CameraSolutionId("camera:1"), "camera:2"))
        )

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _geometry_solution(
            camera_solution_ids=(
                CameraSolutionId("camera:1"),
                CameraSolutionId("camera:1"),
            )
        )

    with pytest.raises(ValueError, match="canonical CameraSolutionId order"):
        _geometry_solution(
            camera_solution_ids=(
                CameraSolutionId("camera:2"),
                CameraSolutionId("camera:1"),
            )
        )


def test_camera_solution_ids_are_retained_without_silent_sorting() -> None:
    camera_solution_ids = (
        CameraSolutionId("camera:1"),
        CameraSolutionId("camera:2"),
        CameraSolutionId("camera:3"),
    )

    solution = _geometry_solution(camera_solution_ids=camera_solution_ids)

    assert solution.camera_solution_ids is camera_solution_ids


@pytest.mark.parametrize(
    ("field_name", "mutable_value", "wrong_value", "duplicate_value", "decreasing_value"),
    [
        (
            "depth_field_ids",
            [DepthFieldId("depth:1")],
            (DepthFieldId("depth:1"), "depth:2"),
            (DepthFieldId("depth:1"), DepthFieldId("depth:1")),
            (DepthFieldId("depth:2"), DepthFieldId("depth:1")),
        ),
        (
            "point_map_ids",
            [PointMapId("points:1")],
            (PointMapId("points:1"), "points:2"),
            (PointMapId("points:1"), PointMapId("points:1")),
            (PointMapId("points:2"), PointMapId("points:1")),
        ),
    ],
)
def test_optional_geometry_reference_tuples_are_immutable_unique_and_canonical(
    field_name: str,
    mutable_value: object,
    wrong_value: object,
    duplicate_value: object,
    decreasing_value: object,
) -> None:
    carrier_defaults: dict[str, object] = {
        "depth_field_ids": (DepthFieldId("depth:base"),),
        "point_map_ids": (PointMapId("points:base"),),
    }

    values = carrier_defaults.copy()
    values[field_name] = mutable_value
    with pytest.raises(TypeError, match="immutable tuple"):
        _geometry_solution(**cast(Any, values))

    values = carrier_defaults.copy()
    values[field_name] = wrong_value
    with pytest.raises(TypeError, match="members must be"):
        _geometry_solution(**cast(Any, values))

    values = carrier_defaults.copy()
    values[field_name] = duplicate_value
    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _geometry_solution(**cast(Any, values))

    values = carrier_defaults.copy()
    values[field_name] = decreasing_value
    with pytest.raises(ValueError, match="canonical"):
        _geometry_solution(**cast(Any, values))


def test_depth_and_point_reference_tuples_are_retained_without_rewriting() -> None:
    depth_field_ids = (DepthFieldId("depth:1"), DepthFieldId("depth:2"))
    point_map_ids = (PointMapId("points:1"), PointMapId("points:2"))

    solution = _geometry_solution(
        depth_field_ids=depth_field_ids,
        point_map_ids=point_map_ids,
    )

    assert solution.depth_field_ids is depth_field_ids
    assert solution.point_map_ids is point_map_ids


def test_geometry_solution_requires_at_least_one_geometry_carrier_beyond_cameras() -> None:
    with pytest.raises(ValueError, match="at least one DepthFieldId or PointMapId"):
        _geometry_solution(depth_field_ids=(), point_map_ids=())


def test_depth_only_point_only_and_mixed_geometry_carriers_are_valid() -> None:
    depth_only = _geometry_solution(
        geometry_solution_id=GeometrySolutionId("geometry:depth"),
        depth_field_ids=(DepthFieldId("depth:1"),),
        point_map_ids=(),
    )
    point_only = _geometry_solution(
        geometry_solution_id=GeometrySolutionId("geometry:points"),
        depth_field_ids=(),
        point_map_ids=(PointMapId("points:1"),),
    )
    mixed = _geometry_solution(
        geometry_solution_id=GeometrySolutionId("geometry:mixed"),
        depth_field_ids=(DepthFieldId("depth:1"),),
        point_map_ids=(PointMapId("points:1"),),
    )

    assert depth_only.depth_field_ids
    assert depth_only.point_map_ids == ()
    assert point_only.depth_field_ids == ()
    assert point_only.point_map_ids
    assert mixed.depth_field_ids and mixed.point_map_ids


def test_scale_status_is_descriptive_without_scale_resolution_or_world_claims() -> None:
    unresolved = _geometry_solution(
        geometry_solution_id=GeometrySolutionId("geometry:unresolved"),
        scale_status=GeometryScaleStatus.UNRESOLVED,
    )
    metric = _geometry_solution(
        geometry_solution_id=GeometrySolutionId("geometry:metric"),
        scale_status=GeometryScaleStatus.METRIC,
    )

    assert unresolved.scale_status is GeometryScaleStatus.UNRESOLVED
    assert metric.scale_status is GeometryScaleStatus.METRIC

    for solution in (unresolved, metric):
        assert not hasattr(solution, "scale_factor")
        assert not hasattr(solution, "world_transform")
        assert not hasattr(solution, "earth_transform")
        assert not hasattr(solution, "origin")
        assert not hasattr(solution, "orientation")
        assert not hasattr(solution, "latitude")
        assert not hasattr(solution, "longitude")
        assert not hasattr(solution, "gps")


def test_metrics_are_retained_without_derived_score_confidence_or_preference() -> None:
    metrics = MetricVector(observations=())

    solution = _geometry_solution(metrics=metrics)

    assert solution.metrics is metrics
    assert not hasattr(solution, "score")
    assert not hasattr(solution, "confidence")
    assert not hasattr(solution, "preferred")
    assert not hasattr(solution, "rank")
    assert not hasattr(solution, "quality_decision")
    assert not hasattr(solution, "route")


def test_geometry_solution_has_no_embedded_child_or_future_surface() -> None:
    solution = _geometry_solution()

    forbidden_attributes = {
        "observation_id",
        "observation_ids",
        "camera_id",
        "camera_solution",
        "camera_solutions",
        "intrinsic_parameters",
        "rotation_matrix",
        "translation_xyz",
        "depth_field",
        "depth_fields",
        "depth_values",
        "validity",
        "point_map",
        "point_maps",
        "positions_xyz",
        "tracks",
        "correspondences",
        "fragment_id",
        "spatial_fragment_id",
        "scale_factor",
        "world_transform",
        "ecef",
        "enu",
        "latitude",
        "longitude",
        "gps",
        "anchor",
        "surface",
        "mesh",
        "sdf",
        "appearance",
        "material",
        "environment",
        "dynamic_state",
        "historical_state",
        "solver",
        "adapter",
        "model",
        "checkpoint",
        "route",
        "quality_mode",
        "metadata",
        "timestamp",
        "persistence",
    }

    assert all(not hasattr(solution, attribute) for attribute in forbidden_attributes)


def test_geometry_solutions_module_has_no_io_legacy_tensor_or_solver_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "numpy",
        "torch",
        "open3d",
        "pycolmap",
        "LocalScaleStatus",
        "SparseReconstructionEstimate",
        "ObservationId",
        "CameraId",
        "CameraSolution",
        "DepthField",
        "PointMap",
        "SurfaceModel",
    }

    assert forbidden_symbols.isdisjoint(geometry_solutions_module.__dict__)
