from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.geometry_consensus as consensus_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    RotationMatrix3x3,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricDirection, MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.geometry_consensus import (
    CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR,
    GEOMETRY_CONSENSUS_EVALUATOR,
    RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR,
    RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR,
    SHARED_OBSERVATION_COVERAGE_DESCRIPTOR,
    GeometryConsensusRequest,
    GeometryConsensusResult,
    GeometryPairDisagreement,
    evaluate_geometry_consensus,
)
from wre.reconstruction.geometry_solution_comparison import (
    CompetingGeometrySolutions,
    GeometrySolutionCandidate,
    GeometrySolutionPair,
)

IDENTITY: RotationMatrix3x3 = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
ROTATE_Z_90: RotationMatrix3x3 = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.consensus.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "geometry.solution") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _matvec(
    matrix: RotationMatrix3x3,
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        sum(matrix[0][axis] * vector[axis] for axis in range(3)),
        sum(matrix[1][axis] * vector[axis] for axis in range(3)),
        sum(matrix[2][axis] * vector[axis] for axis in range(3)),
    )


def _transpose(matrix: RotationMatrix3x3) -> RotationMatrix3x3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(
    left: RotationMatrix3x3,
    right: RotationMatrix3x3,
) -> RotationMatrix3x3:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(3)) for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _camera(
    *,
    token: str,
    observation: str,
    frame: LocalFrameId,
    center: tuple[float, float, float],
    rotation: RotationMatrix3x3 = IDENTITY,
    projection: str = "pinhole",
) -> CameraSolution:
    rotated_center = _matvec(rotation, center)
    translation = (
        -rotated_center[0],
        -rotated_center[1],
        -rotated_center[2],
    )
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}:{observation}"),
        observation_id=ObservationId(f"obs:{observation}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=4, height_px=3),
        intrinsic_parameters=(3.0, 3.0, 2.0, 1.5),
        rotation_matrix=rotation,
        translation_xyz=translation,
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


CameraSpec = tuple[
    str,
    tuple[float, float, float],
    RotationMatrix3x3,
]


def _candidate(
    token: str,
    specs: tuple[CameraSpec, ...],
    *,
    frame: str | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    representation: str = "point",
    projection: str = "pinhole",
) -> GeometrySolutionCandidate:
    local_frame = LocalFrameId(frame or f"frame:{token}")
    cameras = tuple(
        sorted(
            (
                _camera(
                    token=token,
                    observation=observation,
                    frame=local_frame,
                    center=center,
                    rotation=rotation,
                    projection=projection,
                )
                for observation, center, rotation in specs
            ),
            key=lambda item: item.solution_id.value,
        )
    )

    if representation == "point":
        point_map = PointMap(
            point_map_id=PointMapId(f"points:{token}"),
            local_frame_id=local_frame,
            source_observation_ids=tuple(
                sorted(
                    (camera.observation_id for camera in cameras),
                    key=lambda item: item.value,
                )
            ),
            positions_xyz=((1.0, 2.0, 3.0),),
            confidence=None,
            metrics=_metrics(),
        )
        depths: tuple[DepthField, ...] = ()
        points: tuple[PointMap, ...] = (point_map,)
    elif representation == "depth":
        first_camera = cameras[0]
        pixel_count = first_camera.dimensions.width_px * first_camera.dimensions.height_px
        depth = DepthField(
            depth_field_id=DepthFieldId(f"depth:{token}"),
            observation_id=first_camera.observation_id,
            camera_solution_id=first_camera.solution_id,
            dimensions=first_camera.dimensions,
            depth_value_convention=DepthValueConventionName("relative-depth"),
            depth_values=tuple(1.0 for _ in range(pixel_count)),
            validity=tuple(True for _ in range(pixel_count)),
            confidence=None,
            metrics=_metrics(),
        )
        depths = (depth,)
        points = ()
    else:
        raise ValueError("unsupported test representation")

    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=local_frame,
        scale_status=scale,
        camera_solution_ids=tuple(camera.solution_id for camera in cameras),
        depth_field_ids=tuple(item.depth_field_id for item in depths),
        point_map_ids=tuple(item.point_map_id for item in points),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=cameras,
        depth_fields=depths,
        point_maps=points,
        producer=_producer(token),
        source_artifacts=(_artifact(f"source:{token}"),),
    )


def _competing(*candidates: GeometrySolutionCandidate) -> CompetingGeometrySolutions:
    return CompetingGeometrySolutions(
        candidates=tuple(sorted(candidates, key=lambda item: item.geometry_solution_id.value))
    )


def _request(*candidates: GeometrySolutionCandidate) -> GeometryConsensusRequest:
    return GeometryConsensusRequest(
        competing=_competing(*candidates),
        input_artifacts=(
            _artifact("evidence:a", "evidence.geometry"),
            _artifact("evidence:b", "evidence.geometry"),
        ),
    )


def _metric_values(result: GeometryConsensusResult, pair_index: int = 0) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value
        for observation in result.pair_disagreements[pair_index].metrics.observations
    }


def _transform_specs(
    specs: tuple[CameraSpec, ...],
    *,
    old_from_new_rotation: RotationMatrix3x3,
    old_from_new_translation: tuple[float, float, float],
) -> tuple[CameraSpec, ...]:
    new_from_old_rotation = _transpose(old_from_new_rotation)
    transformed: list[CameraSpec] = []
    for observation, center_old, rotation_old in specs:
        centered = _subtract(center_old, old_from_new_translation)
        center_new = _matvec(new_from_old_rotation, centered)
        rotation_new = _matmul(rotation_old, old_from_new_rotation)
        transformed.append((observation, center_new, rotation_new))
    return tuple(transformed)


def test_request_and_result_have_exact_frozen_field_shapes() -> None:
    left = _candidate("a", (("0", (0.0, 0.0, 0.0), IDENTITY),))
    right = _candidate("b", (("0", (0.0, 0.0, 0.0), IDENTITY),))
    request = _request(left, right)
    result = evaluate_geometry_consensus(request)

    assert tuple(field.name for field in fields(GeometryConsensusRequest)) == (
        "competing",
        "input_artifacts",
    )
    assert tuple(field.name for field in fields(GeometryPairDisagreement)) == (
        "pair",
        "metrics",
    )
    assert tuple(field.name for field in fields(GeometryConsensusResult)) == (
        "request",
        "pair_disagreements",
    )
    with pytest.raises(FrozenInstanceError):
        request.input_artifacts = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.pair_disagreements = ()  # type: ignore[misc]


def test_request_rejects_wrong_mutable_empty_duplicate_and_noncanonical_evidence() -> None:
    left = _candidate("a", (("0", (0.0, 0.0, 0.0), IDENTITY),))
    right = _candidate("b", (("0", (0.0, 0.0, 0.0), IDENTITY),))
    competing = _competing(left, right)
    first = _artifact("evidence:a", "evidence.geometry")
    second = _artifact("evidence:b", "evidence.geometry")

    with pytest.raises(TypeError, match="CompetingGeometrySolutions"):
        GeometryConsensusRequest(
            competing=cast(Any, (left, right)),
            input_artifacts=(first,),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        GeometryConsensusRequest(
            competing=competing,
            input_artifacts=cast(Any, [first]),
        )
    with pytest.raises(ValueError, match="non-empty"):
        GeometryConsensusRequest(competing=competing, input_artifacts=())
    with pytest.raises(TypeError, match="ArtifactRef"):
        GeometryConsensusRequest(
            competing=competing,
            input_artifacts=cast(Any, ("artifact",)),
        )
    with pytest.raises(ValueError, match="unique"):
        GeometryConsensusRequest(
            competing=competing,
            input_artifacts=(first, first),
        )
    with pytest.raises(ValueError, match="canonical"):
        GeometryConsensusRequest(
            competing=competing,
            input_artifacts=(second, first),
        )


def test_result_contains_every_canonical_pair_for_four_candidates() -> None:
    candidates = tuple(
        _candidate(
            token,
            (
                ("0", (0.0, 0.0, 0.0), IDENTITY),
                ("1", (1.0, 0.0, 0.0), IDENTITY),
            ),
        )
        for token in ("a", "b", "c", "d")
    )

    result = evaluate_geometry_consensus(_request(*candidates))

    assert tuple(item.pair for item in result.pair_disagreements) == (
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:b")),
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:c")),
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:d")),
        GeometrySolutionPair(GeometrySolutionId("geometry:b"), GeometrySolutionId("geometry:c")),
        GeometrySolutionPair(GeometrySolutionId("geometry:b"), GeometrySolutionId("geometry:d")),
        GeometrySolutionPair(GeometrySolutionId("geometry:c"), GeometrySolutionId("geometry:d")),
    )


def test_result_rejects_missing_or_noncanonical_pair_evidence() -> None:
    candidates = tuple(
        _candidate(
            token,
            (
                ("0", (0.0, 0.0, 0.0), IDENTITY),
                ("1", (1.0, 0.0, 0.0), IDENTITY),
            ),
        )
        for token in ("a", "b", "c")
    )
    request = _request(*candidates)
    valid = evaluate_geometry_consensus(request)

    with pytest.raises(ValueError, match="exactly one metric vector"):
        GeometryConsensusResult(
            request=request,
            pair_disagreements=valid.pair_disagreements[:-1],
        )
    with pytest.raises(ValueError, match="exactly one metric vector"):
        GeometryConsensusResult(
            request=request,
            pair_disagreements=tuple(reversed(valid.pair_disagreements)),
        )


def test_pairwise_metrics_are_symmetric_when_candidate_contents_are_swapped() -> None:
    first_specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (1.0, 0.0, 0.0), IDENTITY),
    )
    second_specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (0.0, 1.0, 0.0), ROTATE_Z_90),
    )

    first = evaluate_geometry_consensus(
        _request(
            _candidate("a", first_specs),
            _candidate("b", second_specs),
        )
    )
    swapped = evaluate_geometry_consensus(
        _request(
            _candidate("a", second_specs),
            _candidate("b", first_specs),
        )
    )

    assert _metric_values(first) == pytest.approx(_metric_values(swapped))
    assert _metric_values(first)[
        RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR.name.value
    ] == pytest.approx(90.0)
    assert _metric_values(first)[
        RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR.name.value
    ] == pytest.approx(90.0)


def test_independent_rigid_local_frame_change_leaves_relative_disagreement_zero() -> None:
    specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (1.0, 0.0, 0.0), ROTATE_Z_90),
        ("2", (1.0, 1.0, 0.0), IDENTITY),
    )
    transformed = _transform_specs(
        specs,
        old_from_new_rotation=ROTATE_Z_90,
        old_from_new_translation=(3.0, -2.0, 1.0),
    )
    result = evaluate_geometry_consensus(
        _request(
            _candidate("a", specs, frame="frame:a"),
            _candidate("b", transformed, frame="frame:b"),
        )
    )
    values = _metric_values(result)

    assert values[SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value] == 1.0
    assert values[CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR.name.value] == 1.0
    assert values[RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR.name.value] == pytest.approx(
        0.0, abs=1e-10
    )
    assert values[
        RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR.name.value
    ] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(
    ("left_observations", "right_observations", "expected_ratio"),
    [
        (("0", "1"), ("0", "1"), 1.0),
        (("0", "1", "2"), ("1", "2", "3"), 0.5),
        (("0",), ("1",), 0.0),
        (("0", "1"), ("1", "2"), 1.0 / 3.0),
    ],
)
def test_shared_observation_coverage_uses_union(
    left_observations: tuple[str, ...],
    right_observations: tuple[str, ...],
    expected_ratio: float,
) -> None:
    def specs(observations: tuple[str, ...]) -> tuple[CameraSpec, ...]:
        return tuple(
            (
                observation,
                (float(index), 0.0, 0.0),
                IDENTITY,
            )
            for index, observation in enumerate(observations)
        )

    result = evaluate_geometry_consensus(
        _request(
            _candidate("a", specs(left_observations)),
            _candidate("b", specs(right_observations)),
        )
    )
    values = _metric_values(result)

    assert values[SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value] == pytest.approx(
        expected_ratio
    )


def test_zero_or_single_shared_observation_emits_only_coverage() -> None:
    zero = evaluate_geometry_consensus(
        _request(
            _candidate("a", (("0", (0.0, 0.0, 0.0), IDENTITY),)),
            _candidate("b", (("1", (0.0, 0.0, 0.0), IDENTITY),)),
        )
    )
    one = evaluate_geometry_consensus(
        _request(
            _candidate(
                "a",
                (
                    ("0", (0.0, 0.0, 0.0), IDENTITY),
                    ("1", (1.0, 0.0, 0.0), IDENTITY),
                ),
            ),
            _candidate(
                "b",
                (
                    ("1", (1.0, 0.0, 0.0), IDENTITY),
                    ("2", (2.0, 0.0, 0.0), IDENTITY),
                ),
            ),
        )
    )

    assert tuple(_metric_values(zero)) == (SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value,)
    assert tuple(_metric_values(one)) == (SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value,)


def test_degenerate_baselines_keep_rotation_and_zero_translation_coverage() -> None:
    specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (0.0, 0.0, 0.0), ROTATE_Z_90),
    )
    result = evaluate_geometry_consensus(
        _request(
            _candidate("a", specs),
            _candidate("b", specs),
        )
    )
    values = _metric_values(result)

    assert values[SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value] == 1.0
    assert values[RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR.name.value] == pytest.approx(0.0)
    assert values[CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR.name.value] == 0.0
    assert RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR.name.value not in values


def test_different_frame_scale_representation_and_projection_preserve_pose_evidence() -> None:
    specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (1.0, 0.0, 0.0), IDENTITY),
    )
    result = evaluate_geometry_consensus(
        _request(
            _candidate(
                "a",
                specs,
                frame="frame:depth",
                scale=GeometryScaleStatus.UNRESOLVED,
                representation="depth",
                projection="pinhole",
            ),
            _candidate(
                "b",
                specs,
                frame="frame:point",
                scale=GeometryScaleStatus.METRIC,
                representation="point",
                projection="simple_radial",
            ),
        )
    )
    values = _metric_values(result)

    assert values[SHARED_OBSERVATION_COVERAGE_DESCRIPTOR.name.value] == 1.0
    assert values[RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR.name.value] == pytest.approx(0.0)
    assert values[CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR.name.value] == 1.0
    assert values[
        RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR.name.value
    ] == pytest.approx(0.0)


def test_metrics_are_informational_and_retain_exact_provenance() -> None:
    specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (1.0, 0.0, 0.0), IDENTITY),
    )
    request = _request(_candidate("a", specs), _candidate("b", specs))
    result = evaluate_geometry_consensus(request)

    assert result.request is request
    for observation in result.pair_disagreements[0].metrics.observations:
        assert observation.descriptor.direction is MetricDirection.INFORMATIONAL
        assert observation.provenance.evaluator is GEOMETRY_CONSENSUS_EVALUATOR
        assert observation.provenance.input_artifacts == request.input_artifacts


def test_consensus_does_not_mutate_candidates_or_add_selection_semantics() -> None:
    specs: tuple[CameraSpec, ...] = (
        ("0", (0.0, 0.0, 0.0), IDENTITY),
        ("1", (1.0, 0.0, 0.0), IDENTITY),
    )
    left = _candidate("a", specs)
    right = _candidate("b", specs)
    request = _request(left, right)
    left_before = left
    right_before = right

    result = evaluate_geometry_consensus(request)

    assert left == left_before
    assert right == right_before
    assert request.competing.candidates[0] is left
    assert request.competing.candidates[1] is right
    assert not hasattr(result, "winner")
    assert not hasattr(result, "preferred_candidate")
    assert not hasattr(result, "decision")
    assert not hasattr(result, "score")
    assert not hasattr(result.pair_disagreements[0], "transform")


def test_descriptor_names_and_configuration_do_not_imply_quality_or_winner() -> None:
    descriptors = (
        SHARED_OBSERVATION_COVERAGE_DESCRIPTOR,
        RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR,
        CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR,
        RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR,
    )
    forbidden_tokens = ("winner", "preferred", "quality", "score", "threshold", "pass")

    for descriptor in descriptors:
        assert descriptor.direction is MetricDirection.INFORMATIONAL
        assert not any(token in descriptor.name.value for token in forbidden_tokens)


def test_module_surface_is_cpu_only_and_solver_independent() -> None:
    forbidden = {
        "numpy",
        "torch",
        "pycolmap",
        "cuda",
        "GLUEMAP",
        "VGGT",
        "Da3ExecutionResult",
        "ColmapSparseModelArtifact",
        "ColmapGeometryOutcome",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Sim3",
        "SE3",
        "ICP",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
        "subprocess",
        "socket",
        "requests",
        "pathlib",
    }

    assert forbidden.isdisjoint(vars(consensus_module))
