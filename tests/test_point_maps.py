from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.point_maps as point_maps_module
from wre.domain import LocalFrameId, MetricVector, ObservationId, PointMap, PointMapId


def _point_map(
    *,
    point_map_id: PointMapId | None = None,
    local_frame_id: LocalFrameId | None = None,
    source_observation_ids: tuple[ObservationId, ...] = (
        ObservationId("obs:1"),
        ObservationId("obs:2"),
    ),
    positions_xyz: tuple[tuple[float, float, float], ...] = (
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    ),
    confidence: tuple[float, ...] | None = (0.9, 0.8),
    metrics: MetricVector | None = None,
) -> PointMap:
    return PointMap(
        point_map_id=point_map_id or PointMapId("point-map:1"),
        local_frame_id=local_frame_id or LocalFrameId("local:1"),
        source_observation_ids=source_observation_ids,
        positions_xyz=positions_xyz,
        confidence=confidence,
        metrics=metrics or MetricVector(observations=()),
    )


def test_point_map_id_is_typed_immutable_hashable_and_stringable() -> None:
    first = PointMapId("points:A-1")
    second = PointMapId("points:A-1")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "points:A-1"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", ".leading", " leading", "has/slash", "has space", "a" * 129, 1, True],
)
def test_point_map_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        PointMapId(cast(Any, value))


def test_point_map_has_exact_frozen_field_shape_and_retains_caller_objects() -> None:
    point_map_id = PointMapId("points:retained")
    local_frame_id = LocalFrameId("local:retained")
    source_observation_ids = (ObservationId("obs:a"), ObservationId("obs:b"))
    positions_xyz = ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0))
    confidence = (0.0, 1.0)
    metrics = MetricVector(observations=())

    point_map = _point_map(
        point_map_id=point_map_id,
        local_frame_id=local_frame_id,
        source_observation_ids=source_observation_ids,
        positions_xyz=positions_xyz,
        confidence=confidence,
        metrics=metrics,
    )

    assert tuple(field.name for field in fields(PointMap)) == (
        "point_map_id",
        "local_frame_id",
        "source_observation_ids",
        "positions_xyz",
        "confidence",
        "metrics",
    )
    assert point_map.point_map_id is point_map_id
    assert point_map.local_frame_id is local_frame_id
    assert point_map.source_observation_ids is source_observation_ids
    assert point_map.positions_xyz is positions_xyz
    assert point_map.confidence is confidence
    assert point_map.metrics is metrics

    with pytest.raises(FrozenInstanceError):
        point_map.local_frame_id = LocalFrameId("other")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("point_map_id", "points", "point_map_id must be PointMapId"),
        ("local_frame_id", "local", "local_frame_id must be LocalFrameId"),
        ("metrics", (), "metrics must be MetricVector"),
    ],
)
def test_point_map_rejects_wrong_core_types(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "point_map_id": PointMapId("points:typed"),
        "local_frame_id": LocalFrameId("local:typed"),
        "source_observation_ids": (ObservationId("obs:1"),),
        "positions_xyz": ((1.0, 2.0, 3.0),),
        "confidence": (0.5,),
        "metrics": MetricVector(observations=()),
    }
    values[field_name] = replacement

    with pytest.raises(TypeError, match=message):
        PointMap(**cast(Any, values))


def test_source_observation_ids_must_be_non_empty_immutable_and_canonical() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _point_map(source_observation_ids=())

    with pytest.raises(TypeError, match="immutable tuple"):
        _point_map(source_observation_ids=cast(Any, [ObservationId("obs:1")]))

    with pytest.raises(TypeError, match="members must be ObservationId"):
        _point_map(source_observation_ids=cast(Any, (ObservationId("obs:1"), "obs:2")))

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        _point_map(source_observation_ids=(ObservationId("obs:1"), ObservationId("obs:1")))

    with pytest.raises(ValueError, match="canonical ObservationId order"):
        _point_map(source_observation_ids=(ObservationId("obs:2"), ObservationId("obs:1")))


def test_source_observation_ids_are_retained_without_silent_sorting() -> None:
    source_observation_ids = (
        ObservationId("obs:1"),
        ObservationId("obs:2"),
        ObservationId("obs:3"),
    )

    point_map = _point_map(source_observation_ids=source_observation_ids)

    assert point_map.source_observation_ids is source_observation_ids


def test_positions_may_be_empty_without_fabricating_geometry_or_failure() -> None:
    positions_xyz: tuple[tuple[float, float, float], ...] = ()
    confidence: tuple[float, ...] = ()

    point_map = _point_map(positions_xyz=positions_xyz, confidence=confidence)

    assert point_map.positions_xyz is positions_xyz
    assert point_map.confidence is confidence


def test_positions_require_immutable_outer_and_nested_tuples() -> None:
    with pytest.raises(TypeError, match="positions_xyz must be an immutable tuple"):
        _point_map(positions_xyz=cast(Any, [(1.0, 2.0, 3.0)]))

    with pytest.raises(TypeError, match="members must be immutable tuples"):
        _point_map(positions_xyz=cast(Any, ([1.0, 2.0, 3.0],)))


@pytest.mark.parametrize(
    "position",
    [
        (),
        (1.0,),
        (1.0, 2.0),
        (1.0, 2.0, 3.0, 4.0),
    ],
)
def test_positions_require_exact_three_vector_shape(position: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="must be 3-vectors"):
        _point_map(
            positions_xyz=cast(Any, (position,)),
            confidence=(0.5,),
        )


@pytest.mark.parametrize("value", [1, True, "1.0"])
def test_position_components_require_exact_float(value: object) -> None:
    with pytest.raises(TypeError, match="positions_xyz component must be float"):
        _point_map(
            positions_xyz=cast(Any, ((value, 2.0, 3.0),)),
            confidence=(0.5,),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_position_components_require_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="positions_xyz component must be finite"):
        _point_map(positions_xyz=((value, 2.0, 3.0),), confidence=(0.5,))


def test_point_sequence_retains_order_and_duplicate_coordinates_exactly() -> None:
    positions_xyz = (
        (4.0, 5.0, 6.0),
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
    )
    confidence = (0.7, 0.8, 0.9)

    point_map = _point_map(positions_xyz=positions_xyz, confidence=confidence)

    assert point_map.positions_xyz is positions_xyz
    assert point_map.positions_xyz == positions_xyz
    assert point_map.confidence is confidence


def test_confidence_may_be_absent_without_synthesizing_a_claim() -> None:
    point_map = _point_map(confidence=None)

    assert point_map.confidence is None


def test_confidence_requires_immutable_tuple_and_exact_point_count() -> None:
    with pytest.raises(TypeError, match="immutable tuple"):
        _point_map(confidence=cast(Any, [0.9, 0.8]))

    with pytest.raises(ValueError, match="length must match point count"):
        _point_map(confidence=(0.9,))

    with pytest.raises(ValueError, match="length must match point count"):
        _point_map(confidence=(0.9, 0.8, 0.7))


@pytest.mark.parametrize("value", [0, 1, True, "0.5"])
def test_confidence_requires_exact_float_members(value: object) -> None:
    with pytest.raises(TypeError, match="confidence member must be float"):
        _point_map(confidence=cast(Any, (value, 0.8)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_requires_finite_members(value: float) -> None:
    with pytest.raises(ValueError, match="confidence member must be finite"):
        _point_map(confidence=(value, 0.8))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_must_be_in_closed_zero_to_one_range(value: float) -> None:
    with pytest.raises(ValueError, match=r"inclusive range 0\.0 to 1\.0"):
        _point_map(confidence=(value, 0.8))


def test_confidence_endpoints_are_allowed_and_retained_exactly() -> None:
    confidence = (0.0, 1.0)

    point_map = _point_map(confidence=confidence)

    assert point_map.confidence is confidence


def test_metrics_are_retained_without_derived_count_confidence_or_quality() -> None:
    metrics = MetricVector(observations=())

    point_map = _point_map(metrics=metrics)

    assert point_map.metrics is metrics
    assert not hasattr(point_map, "point_count")
    assert not hasattr(point_map, "aggregate_confidence")
    assert not hasattr(point_map, "quality_decision")


def test_point_map_has_no_future_or_noncanonical_surface() -> None:
    point_map = _point_map()

    forbidden_attributes = {
        "camera_id",
        "camera_solution_id",
        "depth_field_id",
        "dimensions",
        "pixel_coordinates",
        "pixel_indices",
        "tracks",
        "correspondences",
        "feature_indices",
        "reprojection_error_px",
        "normals",
        "colors",
        "descriptors",
        "labels",
        "semantics",
        "materials",
        "faces",
        "topology",
        "scale",
        "scale_status",
        "world_transform",
        "ecef",
        "enu",
        "latitude",
        "longitude",
        "gps",
        "geometry_solution",
        "surface",
        "appearance",
        "dynamic_state",
        "historical_state",
        "solver",
        "adapter",
        "model",
        "checkpoint",
        "route",
        "quality_mode",
        "metadata",
        "persistence",
    }

    assert all(not hasattr(point_map, attribute) for attribute in forbidden_attributes)


def test_point_maps_module_has_no_io_tensor_storage_or_solver_execution_surface() -> None:
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
        "CameraId",
        "CameraSolutionId",
        "DepthFieldId",
        "ImageDimensions",
        "GeometrySolution",
        "SurfaceModel",
    }

    assert forbidden_symbols.isdisjoint(point_maps_module.__dict__)
