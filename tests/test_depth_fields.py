from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.depth_fields as depth_fields_module
from wre.domain import (
    CameraSolutionId,
    DepthField,
    DepthFieldId,
    DepthValueConventionName,
    ImageDimensions,
    MetricVector,
    ObservationId,
)


def _depth_field(
    *,
    depth_field_id: DepthFieldId | None = None,
    observation_id: ObservationId | None = None,
    camera_solution_id: CameraSolutionId | None = None,
    dimensions: ImageDimensions | None = None,
    depth_value_convention: DepthValueConventionName | None = None,
    depth_values: tuple[float, ...] = (1.0, 2.0, 0.0, 4.0),
    validity: tuple[bool, ...] = (True, True, False, True),
    confidence: tuple[float, ...] | None = (0.9, 0.8, 0.0, 0.7),
    metrics: MetricVector | None = None,
) -> DepthField:
    return DepthField(
        depth_field_id=depth_field_id or DepthFieldId("depth-field:1"),
        observation_id=observation_id or ObservationId("obs:1"),
        camera_solution_id=camera_solution_id or CameraSolutionId("camera-solution:1"),
        dimensions=dimensions or ImageDimensions(width_px=2, height_px=2),
        depth_value_convention=depth_value_convention or DepthValueConventionName("camera-z"),
        depth_values=depth_values,
        validity=validity,
        confidence=confidence,
        metrics=metrics or MetricVector(observations=()),
    )


def test_depth_field_id_is_typed_immutable_hashable_and_stringable() -> None:
    first = DepthFieldId("depth:A-1")
    second = DepthFieldId("depth:A-1")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "depth:A-1"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", ".leading", " leading", "has/slash", "has space", "a" * 129, 1, True],
)
def test_depth_field_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        DepthFieldId(cast(Any, value))


def test_depth_value_convention_is_open_lowercase_immutable_token() -> None:
    first = DepthValueConventionName("ray-distance")
    second = DepthValueConventionName("ray-distance")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "ray-distance"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", "CAMERA-Z", "Ray_Distance", ".leading", "has/slash", "has space", "a" * 129, 1],
)
def test_depth_value_convention_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        DepthValueConventionName(cast(Any, value))


def test_depth_field_has_exact_frozen_field_shape_and_retains_caller_objects() -> None:
    depth_field_id = DepthFieldId("depth:retained")
    observation_id = ObservationId("obs:retained")
    camera_solution_id = CameraSolutionId("camera:retained")
    dimensions = ImageDimensions(width_px=2, height_px=2)
    convention = DepthValueConventionName("inverse-depth")
    depth_values = (1.0, 2.0, 0.0, 4.0)
    validity = (True, True, False, True)
    confidence = (0.9, 0.8, 0.0, 0.7)
    metrics = MetricVector(observations=())

    depth_field = _depth_field(
        depth_field_id=depth_field_id,
        observation_id=observation_id,
        camera_solution_id=camera_solution_id,
        dimensions=dimensions,
        depth_value_convention=convention,
        depth_values=depth_values,
        validity=validity,
        confidence=confidence,
        metrics=metrics,
    )

    assert tuple(field.name for field in fields(DepthField)) == (
        "depth_field_id",
        "observation_id",
        "camera_solution_id",
        "dimensions",
        "depth_value_convention",
        "depth_values",
        "validity",
        "confidence",
        "metrics",
    )
    assert depth_field.depth_field_id is depth_field_id
    assert depth_field.observation_id is observation_id
    assert depth_field.camera_solution_id is camera_solution_id
    assert depth_field.dimensions is dimensions
    assert depth_field.depth_value_convention is convention
    assert depth_field.depth_values is depth_values
    assert depth_field.validity is validity
    assert depth_field.confidence is confidence
    assert depth_field.metrics is metrics

    with pytest.raises(FrozenInstanceError):
        depth_field.observation_id = ObservationId("other")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("depth_field_id", "depth", "depth_field_id must be DepthFieldId"),
        ("observation_id", "obs", "observation_id must be ObservationId"),
        ("camera_solution_id", "camera", "camera_solution_id must be CameraSolutionId"),
        ("dimensions", (2, 2), "dimensions must be ImageDimensions"),
        (
            "depth_value_convention",
            "camera-z",
            "depth_value_convention must be DepthValueConventionName",
        ),
        ("metrics", (), "metrics must be MetricVector"),
    ],
)
def test_depth_field_rejects_wrong_core_types(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "depth_field_id": DepthFieldId("depth:typed"),
        "observation_id": ObservationId("obs:typed"),
        "camera_solution_id": CameraSolutionId("camera:typed"),
        "dimensions": ImageDimensions(width_px=2, height_px=2),
        "depth_value_convention": DepthValueConventionName("camera-z"),
        "depth_values": (1.0, 2.0, 0.0, 4.0),
        "validity": (True, True, False, True),
        "confidence": (0.9, 0.8, 0.0, 0.7),
        "metrics": MetricVector(observations=()),
    }
    values[field_name] = replacement

    with pytest.raises(TypeError, match=message):
        DepthField(**cast(Any, values))


@pytest.mark.parametrize("field_name", ["depth_values", "validity", "confidence"])
def test_raster_collections_reject_mutable_lists(field_name: str) -> None:
    values: dict[str, object] = {
        "depth_values": (1.0, 2.0, 0.0, 4.0),
        "validity": (True, True, False, True),
        "confidence": (0.9, 0.8, 0.0, 0.7),
    }
    values[field_name] = list(cast(tuple[object, ...], values[field_name]))

    with pytest.raises(TypeError, match="immutable tuple"):
        _depth_field(**cast(Any, values))


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("depth_values", (1.0, 2.0, 0.0)),
        ("depth_values", (1.0, 2.0, 0.0, 4.0, 5.0)),
        ("validity", (True, True, False)),
        ("validity", (True, True, False, True, False)),
        ("confidence", (0.9, 0.8, 0.0)),
        ("confidence", (0.9, 0.8, 0.0, 0.7, 0.6)),
    ],
)
def test_raster_lengths_must_match_dimensions(field_name: str, value: object) -> None:
    values: dict[str, object] = {
        "depth_values": (1.0, 2.0, 0.0, 4.0),
        "validity": (True, True, False, True),
        "confidence": (0.9, 0.8, 0.0, 0.7),
    }
    values[field_name] = value

    with pytest.raises(ValueError, match="length must match image pixel count"):
        _depth_field(**cast(Any, values))


@pytest.mark.parametrize("value", [1, True, "1.0"])
def test_depth_values_require_exact_float_members(value: object) -> None:
    with pytest.raises(TypeError, match="depth_values member must be float"):
        _depth_field(depth_values=cast(Any, (value, 2.0, 0.0, 4.0)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_depth_values_require_finite_members(value: float) -> None:
    with pytest.raises(ValueError, match="depth_values member must be finite"):
        _depth_field(depth_values=(value, 2.0, 0.0, 4.0))


@pytest.mark.parametrize("value", [0.0, -1.0])
def test_valid_pixels_require_strictly_positive_depth(value: float) -> None:
    with pytest.raises(ValueError, match="valid pixels must have strictly positive depth"):
        _depth_field(depth_values=(value, 2.0, 0.0, 4.0))


@pytest.mark.parametrize("value", [1.0, -1.0])
def test_invalid_pixels_require_canonical_zero_depth(value: float) -> None:
    with pytest.raises(ValueError, match=r"invalid pixels must use canonical 0\.0 depth"):
        _depth_field(depth_values=(1.0, 2.0, value, 4.0))


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_validity_requires_exact_bool_members(value: object) -> None:
    with pytest.raises(TypeError, match="validity members must be bool"):
        _depth_field(validity=cast(Any, (True, value, False, True)))


def test_confidence_may_be_absent_without_synthesizing_a_claim() -> None:
    depth_field = _depth_field(confidence=None)

    assert depth_field.confidence is None


@pytest.mark.parametrize("value", [0, 1, True, "0.5"])
def test_confidence_requires_exact_float_members(value: object) -> None:
    with pytest.raises(TypeError, match="confidence member must be float"):
        _depth_field(confidence=cast(Any, (0.9, value, 0.0, 0.7)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_confidence_requires_finite_members(value: float) -> None:
    with pytest.raises(ValueError, match="confidence member must be finite"):
        _depth_field(confidence=(0.9, value, 0.0, 0.7))


@pytest.mark.parametrize("value", [-0.1, 1.1])
def test_confidence_must_be_in_closed_zero_to_one_range(value: float) -> None:
    with pytest.raises(ValueError, match=r"inclusive range 0\.0 to 1\.0"):
        _depth_field(confidence=(0.9, value, 0.0, 0.7))


@pytest.mark.parametrize("value", [0.1, 1.0])
def test_invalid_pixels_require_canonical_zero_confidence(value: float) -> None:
    with pytest.raises(ValueError, match=r"invalid pixels must use canonical 0\.0 confidence"):
        _depth_field(confidence=(0.9, 0.8, value, 0.7))


def test_confidence_endpoints_are_allowed_for_valid_pixels() -> None:
    depth_field = _depth_field(
        validity=(True, True, True, True),
        depth_values=(1.0, 2.0, 3.0, 4.0),
        confidence=(0.0, 1.0, 0.25, 0.75),
    )

    assert depth_field.confidence == (0.0, 1.0, 0.25, 0.75)


@pytest.mark.parametrize("token", ["camera-z", "ray-distance", "inverse-depth"])
def test_depth_conventions_are_retained_without_conversion_or_normalization(token: str) -> None:
    convention = DepthValueConventionName(token)
    depth_values = (1.0, 2.0, 0.0, 4.0)

    depth_field = _depth_field(
        depth_value_convention=convention,
        depth_values=depth_values,
    )

    assert depth_field.depth_value_convention is convention
    assert depth_field.depth_values is depth_values


def test_distinct_depth_conventions_remain_distinct_semantics() -> None:
    camera_z = _depth_field(
        depth_field_id=DepthFieldId("depth:camera-z"),
        depth_value_convention=DepthValueConventionName("camera-z"),
    )
    ray_distance = _depth_field(
        depth_field_id=DepthFieldId("depth:ray-distance"),
        depth_value_convention=DepthValueConventionName("ray-distance"),
    )

    assert camera_z.depth_value_convention != ray_distance.depth_value_convention
    assert camera_z.depth_values == ray_distance.depth_values


def test_depth_field_has_no_future_or_noncanonical_surface() -> None:
    depth_field = _depth_field()

    forbidden_attributes = {
        "local_frame_id",
        "camera_id",
        "intrinsic_parameters",
        "rotation_matrix",
        "translation_xyz",
        "projection_model",
        "scale",
        "scale_status",
        "world_transform",
        "ecef",
        "enu",
        "latitude",
        "longitude",
        "gps",
        "point_map",
        "points",
        "point_coordinates",
        "geometry_solution",
        "geometry",
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

    assert all(not hasattr(depth_field, attribute) for attribute in forbidden_attributes)


def test_depth_fields_module_has_no_io_persistence_tensor_or_solver_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "numpy",
        "torch",
        "pycolmap",
        "LocalFrameId",
        "CameraId",
        "PointMap",
        "GeometrySolution",
        "SurfaceModel",
    }

    assert forbidden_symbols.isdisjoint(depth_fields_module.__dict__)
