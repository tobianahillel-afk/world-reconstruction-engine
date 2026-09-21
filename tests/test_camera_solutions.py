from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.camera_solutions as camera_solutions_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    ImageDimensions,
    LocalFrameId,
    MetricVector,
    ObservationId,
)


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _solution(
    *,
    solution_id: CameraSolutionId | None = None,
    observation_id: ObservationId | None = None,
    local_frame_id: LocalFrameId | None = None,
    projection_model: CameraProjectionModelName | None = None,
    dimensions: ImageDimensions | None = None,
    intrinsic_parameters: tuple[float, ...] = (1000.0, 1000.0, 640.0, 360.0),
    rotation_matrix: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    uncertainty_artifacts: tuple[ArtifactRef, ...] = (),
    metrics: MetricVector | None = None,
) -> CameraSolution:
    return CameraSolution(
        solution_id=solution_id or CameraSolutionId("camera-solution:1"),
        observation_id=observation_id or ObservationId("obs:1"),
        local_frame_id=local_frame_id or LocalFrameId("frame:1"),
        projection_model=projection_model or CameraProjectionModelName("pinhole"),
        dimensions=dimensions or ImageDimensions(width_px=1280, height_px=720),
        intrinsic_parameters=intrinsic_parameters,
        rotation_matrix=rotation_matrix,
        translation_xyz=translation_xyz,
        uncertainty_artifacts=uncertainty_artifacts,
        metrics=metrics or MetricVector(observations=()),
    )


def test_camera_solution_id_is_typed_immutable_hashable_and_stringable() -> None:
    first = CameraSolutionId("solution:A-1")
    second = CameraSolutionId("solution:A-1")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "solution:A-1"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", ".leading", " leading", "has/slash", "has space", "a" * 129, 1, True],
)
def test_camera_solution_id_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        CameraSolutionId(cast(Any, value))


def test_projection_model_name_is_open_lowercase_immutable_token() -> None:
    first = CameraProjectionModelName("opencv.radial-3")
    second = CameraProjectionModelName("opencv.radial-3")

    assert first == second
    assert hash(first) == hash(second)
    assert str(first) == "opencv.radial-3"

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "value",
    ["", "PINHOLE", "Simple_Radial", ".leading", "has/slash", "has space", "a" * 129, 1],
)
def test_projection_model_name_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        CameraProjectionModelName(cast(Any, value))


def test_camera_solution_has_exact_frozen_field_shape_and_retains_objects() -> None:
    solution_id = CameraSolutionId("solution:retained")
    observation_id = ObservationId("obs:retained")
    local_frame_id = LocalFrameId("frame:retained")
    projection_model = CameraProjectionModelName("pinhole")
    dimensions = ImageDimensions(width_px=1920, height_px=1080)
    uncertainty = (_artifact("covariance:1", "camera.uncertainty"),)
    metrics = MetricVector(observations=())

    solution = _solution(
        solution_id=solution_id,
        observation_id=observation_id,
        local_frame_id=local_frame_id,
        projection_model=projection_model,
        dimensions=dimensions,
        uncertainty_artifacts=uncertainty,
        metrics=metrics,
    )

    assert tuple(field.name for field in fields(CameraSolution)) == (
        "solution_id",
        "observation_id",
        "local_frame_id",
        "projection_model",
        "dimensions",
        "intrinsic_parameters",
        "rotation_matrix",
        "translation_xyz",
        "uncertainty_artifacts",
        "metrics",
    )
    assert solution.solution_id is solution_id
    assert solution.observation_id is observation_id
    assert solution.local_frame_id is local_frame_id
    assert solution.projection_model is projection_model
    assert solution.dimensions is dimensions
    assert solution.uncertainty_artifacts is uncertainty
    assert solution.metrics is metrics

    with pytest.raises(FrozenInstanceError):
        solution.observation_id = ObservationId("other")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("solution_id", "solution", "solution_id must be CameraSolutionId"),
        ("observation_id", "obs", "observation_id must be ObservationId"),
        ("local_frame_id", "frame", "local_frame_id must be LocalFrameId"),
        (
            "projection_model",
            "pinhole",
            "projection_model must be CameraProjectionModelName",
        ),
        ("dimensions", (1280, 720), "dimensions must be ImageDimensions"),
        ("metrics", (), "metrics must be MetricVector"),
    ],
)
def test_camera_solution_rejects_wrong_core_types(
    field_name: str,
    replacement: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "solution_id": CameraSolutionId("solution:typed"),
        "observation_id": ObservationId("obs:typed"),
        "local_frame_id": LocalFrameId("frame:typed"),
        "projection_model": CameraProjectionModelName("pinhole"),
        "dimensions": ImageDimensions(width_px=1280, height_px=720),
        "intrinsic_parameters": (1000.0, 1000.0, 640.0, 360.0),
        "rotation_matrix": (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        "translation_xyz": (0.0, 0.0, 0.0),
        "uncertainty_artifacts": (),
        "metrics": MetricVector(observations=()),
    }
    values[field_name] = replacement

    with pytest.raises(TypeError, match=message):
        CameraSolution(**cast(Any, values))


def test_intrinsic_parameters_require_nonempty_immutable_exact_finite_floats() -> None:
    intrinsics = (1000.0, 1001.0, 640.0, 360.0)
    solution = _solution(intrinsic_parameters=intrinsics)

    assert solution.intrinsic_parameters is intrinsics

    with pytest.raises(TypeError, match="immutable tuple"):
        _solution(intrinsic_parameters=cast(Any, [1000.0]))
    with pytest.raises(ValueError, match="must not be empty"):
        _solution(intrinsic_parameters=())

    for value in (1, True, "1.0"):
        with pytest.raises(TypeError, match="member must be float"):
            _solution(intrinsic_parameters=cast(Any, (value,)))
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="member must be finite"):
            _solution(intrinsic_parameters=(value,))


def test_rotation_accepts_identity_and_representative_non_identity_proper_rotation() -> None:
    identity = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    quarter_turn_z = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    assert _solution(rotation_matrix=identity).rotation_matrix is identity
    assert _solution(rotation_matrix=quarter_turn_z).rotation_matrix is quarter_turn_z


def test_rotation_rejects_malformed_mutable_wrong_type_and_non_finite_values() -> None:
    with pytest.raises(TypeError, match="immutable 3x3 tuple"):
        _solution(
            rotation_matrix=cast(
                Any,
                [
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ],
            )
        )
    with pytest.raises(TypeError, match="immutable 3-tuples"):
        _solution(
            rotation_matrix=cast(
                Any,
                (
                    [1.0, 0.0, 0.0],
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )
        )
    with pytest.raises(TypeError, match="immutable 3x3 tuple"):
        _solution(rotation_matrix=cast(Any, ((1.0, 0.0, 0.0),)))
    with pytest.raises(TypeError, match="member must be float"):
        _solution(
            rotation_matrix=cast(
                Any,
                (
                    (1, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                ),
            )
        )
    with pytest.raises(ValueError, match="member must be finite"):
        _solution(
            rotation_matrix=(
                (float("nan"), 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )


def test_rotation_rejects_non_unit_non_orthogonal_and_reflection_matrices() -> None:
    with pytest.raises(ValueError, match="unit length"):
        _solution(
            rotation_matrix=(
                (2.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    with pytest.raises(ValueError, match="mutually orthogonal"):
        _solution(
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.6, 0.8, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
    with pytest.raises(ValueError, match=r"determinant must be \+1"):
        _solution(
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
            )
        )


def test_translation_requires_immutable_exact_finite_float_three_vector() -> None:
    translation = (1.0, -2.0, 3.5)
    solution = _solution(translation_xyz=translation)

    assert solution.translation_xyz is translation

    with pytest.raises(TypeError, match="immutable 3-vector tuple"):
        _solution(translation_xyz=cast(Any, [1.0, 2.0, 3.0]))
    with pytest.raises(TypeError, match="immutable 3-vector tuple"):
        _solution(translation_xyz=cast(Any, (1.0, 2.0)))

    for value in (1, True, "1.0"):
        with pytest.raises(TypeError, match="member must be float"):
            _solution(translation_xyz=cast(Any, (value, 0.0, 0.0)))
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="member must be finite"):
            _solution(translation_xyz=(value, 0.0, 0.0))


def test_extrinsics_use_camera_from_local_frame_sign_convention() -> None:
    rotation = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    translation = (1.0, 2.0, 3.0)
    solution = _solution(rotation_matrix=rotation, translation_xyz=translation)
    local_point = (2.0, 3.0, 4.0)

    camera_point = tuple(
        sum(solution.rotation_matrix[row][axis] * local_point[axis] for axis in range(3))
        + solution.translation_xyz[row]
        for row in range(3)
    )

    assert camera_point == (-2.0, 4.0, 7.0)


def test_uncertainty_artifacts_accept_empty_and_canonical_populated_tuple() -> None:
    assert _solution().uncertainty_artifacts == ()

    refs = (
        _artifact("a", "camera.covariance"),
        _artifact("b", "camera.uncertainty"),
    )
    solution = _solution(uncertainty_artifacts=refs)

    assert solution.uncertainty_artifacts is refs


def test_uncertainty_artifacts_fail_closed_on_identity_order_and_kind_conflicts() -> None:
    ref_a = _artifact("a", "camera.covariance")
    ref_b = _artifact("b", "camera.uncertainty")

    with pytest.raises(TypeError, match="immutable tuple"):
        _solution(uncertainty_artifacts=cast(Any, [ref_a]))
    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        _solution(uncertainty_artifacts=cast(Any, ("a",)))
    with pytest.raises(ValueError, match="must be unique"):
        _solution(uncertainty_artifacts=(ref_a, ref_a))
    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        _solution(uncertainty_artifacts=(ref_b, ref_a))
    with pytest.raises(ValueError, match="conflicting ArtifactKind"):
        _solution(
            uncertainty_artifacts=(
                _artifact("same", "camera.covariance"),
                _artifact("same", "camera.uncertainty"),
            )
        )


def test_camera_solution_has_no_future_or_noncanonical_surface() -> None:
    solution = _solution()

    forbidden_attributes = {
        "camera_id",
        "physical_camera_id",
        "scale",
        "scale_status",
        "world_transform",
        "world_from_camera",
        "camera_center",
        "ecef",
        "enu",
        "latitude",
        "longitude",
        "gps",
        "timestamp",
        "depth",
        "depth_field",
        "point_map",
        "points",
        "geometry",
        "surface",
        "appearance",
        "dynamic_state",
        "historical_state",
        "route",
        "quality_mode",
        "adapter",
        "model",
        "checkpoint",
        "metadata",
    }

    assert all(not hasattr(solution, attribute) for attribute in forbidden_attributes)


def test_camera_solutions_module_has_no_io_persistence_or_external_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "pycolmap",
        "numpy",
        "scipy",
        "CameraId",
        "CameraCalibrationEstimate",
        "CameraPoseEstimate",
        "SparseReconstructionEstimate",
    }

    assert forbidden_symbols.isdisjoint(camera_solutions_module.__dict__)
