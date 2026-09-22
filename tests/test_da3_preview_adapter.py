from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.da3_preview as da3_module
from wre.domain import (
    CameraSolutionId,
    DepthFieldId,
    GeometryScaleStatus,
    GeometrySolutionId,
    ImageDimensions,
    LocalFrameId,
    ObservationId,
    Sha256Digest,
)
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

_UNSET = object()


def _rotation() -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def _intrinsics() -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    return (
        (500.0, 0.0, 1.25),
        (0.0, 510.0, 0.75),
        (0.0, 0.0, 1.0),
    )


def _prediction(
    observation: str = "obs:a",
    *,
    dimensions: ImageDimensions | None = None,
    rotation: Any = None,
    translation: Any = None,
    intrinsics: Any = None,
    depth_values: Any = None,
    validity: Any = None,
    confidence: Any = _UNSET,
) -> da3_module.Da3BaseObservationPrediction:
    actual_dimensions = dimensions or ImageDimensions(width_px=2, height_px=2)
    return da3_module.Da3BaseObservationPrediction(
        observation_id=ObservationId(observation),
        dimensions=actual_dimensions,
        world_to_camera_rotation=_rotation() if rotation is None else rotation,
        world_to_camera_translation=(
            (1.0, 2.0, 3.0) if translation is None else translation
        ),
        intrinsics=_intrinsics() if intrinsics is None else intrinsics,
        depth_values=(
            (1.0, 2.0, 0.0, 4.0) if depth_values is None else depth_values
        ),
        validity=(True, True, False, True) if validity is None else validity,
        confidence=(0.9, 0.8, 0.0, 0.7) if confidence is _UNSET else confidence,
    )


def _identity(value: str = "1") -> Sha256Digest:
    return Sha256Digest(value * 64)


def _apply_pose(
    rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    translation: tuple[float, float, float],
    point: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        sum(rotation[row][axis] * point[axis] for axis in range(3))
        + translation[row]
        for row in range(3)
    )  # type: ignore[return-value]


def test_candidate_identity_constants_are_descriptive_only() -> None:
    assert da3_module.DA3_BASE_MODEL_IDENTIFIER == "depth-anything/DA3-BASE"
    assert da3_module.DA3_BASE_SOURCE_REPOSITORY == "ByteDance-Seed/Depth-Anything-3"
    assert da3_module.DA3_DEPTH_VALUE_CONVENTION.value == "da3.relative-depth"


def test_candidate_prediction_has_exact_frozen_field_shape() -> None:
    assert tuple(field.name for field in fields(da3_module.Da3BaseObservationPrediction)) == (
        "observation_id",
        "dimensions",
        "world_to_camera_rotation",
        "world_to_camera_translation",
        "intrinsics",
        "depth_values",
        "validity",
        "confidence",
    )

    prediction = _prediction()
    with pytest.raises(FrozenInstanceError):
        prediction.depth_values = (1.0,)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"rotation": cast(Any, [[1.0, 0.0, 0.0]])}, TypeError, "immutable 3x3"),
        (
            {
                "rotation": (
                    (1.0, 0.0, 0.0),
                    (0.0, 2.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            },
            ValueError,
            "unit length",
        ),
        (
            {
                "rotation": (
                    (1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            },
            ValueError,
            "mutually orthogonal",
        ),
        (
            {
                "rotation": (
                    (-1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            },
            ValueError,
            "determinant",
        ),
        ({"translation": cast(Any, [1.0, 2.0, 3.0])}, TypeError, "immutable 3-vector"),
        (
            {"translation": (float("nan"), 0.0, 0.0)},
            ValueError,
            "must be finite",
        ),
        ({"intrinsics": cast(Any, [[1.0]])}, TypeError, "immutable 3x3"),
        (
            {
                "intrinsics": (
                    (0.0, 0.0, 1.0),
                    (0.0, 2.0, 1.0),
                    (0.0, 0.0, 1.0),
                )
            },
            ValueError,
            "focal lengths",
        ),
        (
            {
                "intrinsics": (
                    (2.0, 0.1, 1.0),
                    (0.0, 2.0, 1.0),
                    (0.0, 0.0, 1.0),
                )
            },
            ValueError,
            "zero skew",
        ),
        (
            {
                "intrinsics": (
                    (2.0, 0.0, 1.0),
                    (0.0, 2.0, 1.0),
                    (0.0, 0.0, 2.0),
                )
            },
            ValueError,
            "bottom-right",
        ),
    ],
)
def test_candidate_prediction_rejects_malformed_camera_values(
    kwargs: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _prediction(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error_type", "message"),
    [
        ({"validity": cast(Any, [True, True, False, True])}, TypeError, "immutable tuple"),
        ({"validity": (True,)}, ValueError, "pixel count"),
        ({"validity": (True, True, 0, True)}, TypeError, "members must be bool"),
        ({"depth_values": cast(Any, [1.0, 2.0, 0.0, 4.0])}, TypeError, "immutable tuple"),
        ({"depth_values": (1.0,)}, ValueError, "pixel count"),
        (
            {"depth_values": (1.0, float("inf"), 0.0, 4.0)},
            ValueError,
            "must be finite",
        ),
        (
            {"depth_values": (1.0, 0.0, 0.0, 4.0)},
            ValueError,
            "strictly positive",
        ),
        (
            {"depth_values": (1.0, 2.0, 3.0, 4.0)},
            ValueError,
            "invalid pixels must use canonical 0.0 depth",
        ),
        ({"confidence": cast(Any, [0.9, 0.8, 0.0, 0.7])}, TypeError, "immutable tuple"),
        ({"confidence": (0.9,)}, ValueError, "pixel count"),
        (
            {"confidence": (0.9, float("nan"), 0.0, 0.7)},
            ValueError,
            "must be finite",
        ),
        (
            {"confidence": (0.9, 1.1, 0.0, 0.7)},
            ValueError,
            "inclusive range",
        ),
        (
            {"confidence": (0.9, 0.8, 0.1, 0.7)},
            ValueError,
            "invalid pixels must use canonical 0.0 confidence",
        ),
    ],
)
def test_candidate_prediction_rejects_malformed_depth_support(
    kwargs: dict[str, Any],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _prediction(**kwargs)


def test_candidate_prediction_accepts_missing_confidence_without_inventing_it() -> None:
    prediction = _prediction(confidence=None)

    assert prediction.confidence is None


def test_normalization_requires_immutable_nonempty_canonical_unique_predictions() -> None:
    prediction_a = _prediction("obs:a")
    prediction_b = _prediction("obs:b")

    with pytest.raises(TypeError, match="immutable tuple"):
        da3_module.normalize_da3_base_preview(
            cast(Any, [prediction_a]),
            normalization_identity=_identity(),
        )
    with pytest.raises(ValueError, match="at least one"):
        da3_module.normalize_da3_base_preview((), normalization_identity=_identity())
    with pytest.raises(TypeError, match="Da3BaseObservationPrediction"):
        da3_module.normalize_da3_base_preview(
            cast(Any, ("prediction",)),
            normalization_identity=_identity(),
        )
    with pytest.raises(ValueError, match="duplicate"):
        da3_module.normalize_da3_base_preview(
            (prediction_a, prediction_a),
            normalization_identity=_identity(),
        )
    with pytest.raises(ValueError, match="canonical ObservationId order"):
        da3_module.normalize_da3_base_preview(
            (prediction_b, prediction_a),
            normalization_identity=_identity(),
        )
    with pytest.raises(TypeError, match="Sha256Digest"):
        da3_module.normalize_da3_base_preview(
            (prediction_a,),
            normalization_identity=cast(Any, "identity"),
        )


def test_normalization_preserves_exact_pinhole_intrinsics_and_dimensions() -> None:
    prediction = _prediction()

    result = da3_module.normalize_da3_base_preview(
        (prediction,),
        normalization_identity=_identity(),
    )
    camera = result.camera_solutions[0]

    assert camera.projection_model.value == "pinhole"
    assert camera.dimensions is prediction.dimensions
    assert camera.intrinsic_parameters == (500.0, 510.0, 1.25, 0.75)


def test_world_to_camera_pose_is_copied_as_camera_from_local_without_transform() -> None:
    prediction = _prediction()
    result = da3_module.normalize_da3_base_preview(
        (prediction,),
        normalization_identity=_identity(),
    )
    camera = result.camera_solutions[0]

    assert camera.rotation_matrix is prediction.world_to_camera_rotation
    assert camera.translation_xyz is prediction.world_to_camera_translation

    local_point = (2.0, 7.0, -1.0)
    canonical_camera_point = _apply_pose(
        camera.rotation_matrix,
        camera.translation_xyz,
        local_point,
    )
    assert canonical_camera_point == (-6.0, 4.0, 2.0)

    transposed_rotation = tuple(
        tuple(prediction.world_to_camera_rotation[column][row] for column in range(3))
        for row in range(3)
    )
    transpose_point = _apply_pose(
        cast(Any, transposed_rotation),
        prediction.world_to_camera_translation,
        local_point,
    )
    sign_flipped_translation = tuple(
        -member for member in prediction.world_to_camera_translation
    )
    sign_flipped_point = _apply_pose(
        prediction.world_to_camera_rotation,
        cast(Any, sign_flipped_translation),
        local_point,
    )

    assert transpose_point != canonical_camera_point
    assert sign_flipped_point != canonical_camera_point


def test_depth_is_preserved_as_explicit_relative_depth_without_point_unprojection() -> None:
    prediction = _prediction()

    result = da3_module.normalize_da3_base_preview(
        (prediction,),
        normalization_identity=_identity(),
    )
    depth = result.depth_fields[0]

    assert depth.depth_value_convention is da3_module.DA3_DEPTH_VALUE_CONVENTION
    assert depth.depth_values is prediction.depth_values
    assert depth.validity is prediction.validity
    assert depth.confidence is prediction.confidence
    assert result.point_maps == ()
    assert result.geometry_solution.point_map_ids == ()


def test_multi_observation_result_has_exact_linkage_and_one_local_frame() -> None:
    prediction_a = _prediction("obs:a")
    prediction_b = _prediction(
        "obs:b",
        translation=(4.0, 5.0, 6.0),
        depth_values=(2.0, 3.0, 0.0, 5.0),
        confidence=(0.6, 0.7, 0.0, 0.8),
    )

    result = da3_module.normalize_da3_base_preview(
        (prediction_a, prediction_b),
        normalization_identity=_identity(),
    )

    assert isinstance(result, FeedForwardGeometryResult)
    assert tuple(item.observation_id for item in result.camera_solutions) == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )
    assert tuple(item.observation_id for item in result.depth_fields) == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )

    frame = result.geometry_solution.local_frame_id
    assert isinstance(frame, LocalFrameId)
    assert all(item.local_frame_id is frame for item in result.camera_solutions)
    for camera, depth in zip(result.camera_solutions, result.depth_fields, strict=True):
        assert depth.camera_solution_id == camera.solution_id
        assert depth.dimensions == camera.dimensions

    assert result.geometry_solution.camera_solution_ids == tuple(
        sorted(
            (item.solution_id for item in result.camera_solutions),
            key=lambda item: item.value,
        )
    )
    assert result.geometry_solution.depth_field_ids == tuple(
        sorted(
            (item.depth_field_id for item in result.depth_fields),
            key=lambda item: item.value,
        )
    )


def test_da3_base_never_promotes_scale_or_candidate_world_to_project_truth() -> None:
    result = da3_module.normalize_da3_base_preview(
        (_prediction(),),
        normalization_identity=_identity(),
    )

    assert result.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.geometry_solution.local_frame_id.value.startswith("frame:da3:")
    assert "world" not in result.geometry_solution.local_frame_id.value
    assert "earth" not in result.geometry_solution.local_frame_id.value
    assert "ecef" not in result.geometry_solution.local_frame_id.value
    assert "enu" not in result.geometry_solution.local_frame_id.value
    assert result.point_maps == ()


def test_normalization_identity_deterministically_separates_output_roles() -> None:
    predictions = (_prediction("obs:a"), _prediction("obs:b"))
    identity = _identity("2")

    first = da3_module.normalize_da3_base_preview(
        predictions,
        normalization_identity=identity,
    )
    second = da3_module.normalize_da3_base_preview(
        predictions,
        normalization_identity=identity,
    )

    assert first == second
    assert first.geometry_solution.geometry_solution_id == (
        second.geometry_solution.geometry_solution_id
    )
    assert first.geometry_solution.local_frame_id == second.geometry_solution.local_frame_id

    camera_ids = tuple(item.solution_id for item in first.camera_solutions)
    depth_ids = tuple(item.depth_field_id for item in first.depth_fields)
    assert len({item.value for item in camera_ids}) == 2
    assert len({item.value for item in depth_ids}) == 2
    assert all(item.value.startswith("camera:da3:") for item in camera_ids)
    assert all(item.value.startswith("depth:da3:") for item in depth_ids)
    assert first.geometry_solution.geometry_solution_id.value.startswith("geometry:da3:")

    other = da3_module.normalize_da3_base_preview(
        predictions,
        normalization_identity=_identity("3"),
    )
    assert other.geometry_solution.local_frame_id != first.geometry_solution.local_frame_id
    assert other.geometry_solution.geometry_solution_id != (
        first.geometry_solution.geometry_solution_id
    )
    assert tuple(item.solution_id for item in other.camera_solutions) != camera_ids
    assert tuple(item.depth_field_id for item in other.depth_fields) != depth_ids


def test_output_ids_have_distinct_canonical_types() -> None:
    result = da3_module.normalize_da3_base_preview(
        (_prediction(),),
        normalization_identity=_identity("4"),
    )

    assert isinstance(result.camera_solutions[0].solution_id, CameraSolutionId)
    assert isinstance(result.depth_fields[0].depth_field_id, DepthFieldId)
    assert isinstance(
        result.geometry_solution.geometry_solution_id,
        GeometrySolutionId,
    )


def test_normalization_does_not_mutate_or_reorder_source_values() -> None:
    predictions = (_prediction("obs:a"), _prediction("obs:b"))
    original_depth = tuple(item.depth_values for item in predictions)
    original_rotation = tuple(item.world_to_camera_rotation for item in predictions)
    original_intrinsics = tuple(item.intrinsics for item in predictions)

    result = da3_module.normalize_da3_base_preview(
        predictions,
        normalization_identity=_identity("5"),
    )

    assert tuple(item.depth_values for item in predictions) == original_depth
    assert tuple(item.world_to_camera_rotation for item in predictions) == original_rotation
    assert tuple(item.intrinsics for item in predictions) == original_intrinsics
    assert result.depth_fields[0].depth_values is predictions[0].depth_values
    assert result.camera_solutions[0].rotation_matrix is (
        predictions[0].world_to_camera_rotation
    )


def test_da3_preview_module_has_no_external_execution_or_later_layer_surface() -> None:
    forbidden_names = {
        "Path",
        "open",
        "os",
        "subprocess",
        "socket",
        "requests",
        "numpy",
        "np",
        "torch",
        "xformers",
        "huggingface_hub",
        "safetensors",
        "PIL",
        "cv2",
        "Model",
        "Checkpoint",
        "Router",
        "QualityDecision",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
    }

    assert forbidden_names.isdisjoint(vars(da3_module))
