from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.geometry_solution_comparison as comparison_module
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
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.geometry_solution_comparison import (
    CompetingGeometrySolutions,
    GeometrySolutionCandidate,
    GeometrySolutionPair,
    derive_geometry_solution_pairs,
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.geometry.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest(token * 64)),
    )


def _artifact(identifier: str, kind: str = "geometry.solution") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _camera(
    *,
    frame: LocalFrameId,
    observation: str,
    camera_id: str,
    projection: str = "pinhole",
    width_px: int = 2,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(camera_id),
        observation_id=ObservationId(observation),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=width_px, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _depth(
    *,
    camera: CameraSolution,
    depth_id: str,
    observation: str | None = None,
    width_px: int | None = None,
) -> DepthField:
    dimensions = ImageDimensions(
        width_px=width_px if width_px is not None else camera.dimensions.width_px,
        height_px=camera.dimensions.height_px,
    )
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=ObservationId(observation or camera.observation_id.value),
        camera_solution_id=camera.solution_id,
        dimensions=dimensions,
        depth_value_convention=DepthValueConventionName("relative-depth"),
        depth_values=tuple(1.0 for _ in range(dimensions.width_px * dimensions.height_px)),
        validity=tuple(True for _ in range(dimensions.width_px * dimensions.height_px)),
        confidence=None,
        metrics=_metrics(),
    )


def _point_map(
    *,
    frame: LocalFrameId,
    point_map_id: str,
    observation: str,
) -> PointMap:
    return PointMap(
        point_map_id=PointMapId(point_map_id),
        local_frame_id=frame,
        source_observation_ids=(ObservationId(observation),),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )


def _candidate(
    token: str,
    *,
    frame: str | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    projection: str = "pinhole",
    use_depth: bool = True,
    use_point_map: bool = False,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    local_frame = LocalFrameId(frame or f"frame:{token}")
    camera = _camera(
        frame=local_frame,
        observation=f"obs:{token}",
        camera_id=f"camera:{token}",
        projection=projection,
    )
    depth = _depth(camera=camera, depth_id=f"depth:{token}") if use_depth else None
    point_map = (
        _point_map(
            frame=local_frame,
            point_map_id=f"points:{token}",
            observation=f"obs:{token}",
        )
        if use_point_map
        else None
    )
    depths = (depth,) if depth is not None else ()
    point_maps = (point_map,) if point_map is not None else ()
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=local_frame,
        scale_status=scale,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=tuple(item.depth_field_id for item in depths),
        point_map_ids=tuple(item.point_map_id for item in point_maps),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera,),
        depth_fields=depths,
        point_maps=point_maps,
        producer=_producer(token[0]),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (_artifact(f"artifact:{token}"),)
        ),
    )


def test_candidate_has_exact_frozen_field_shape_and_preserves_objects() -> None:
    candidate = _candidate("a")

    assert tuple(field.name for field in fields(GeometrySolutionCandidate)) == (
        "geometry_solution",
        "camera_solutions",
        "depth_fields",
        "point_maps",
        "producer",
        "source_artifacts",
    )
    assert candidate.geometry_solution_id is candidate.geometry_solution.geometry_solution_id
    assert candidate.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    with pytest.raises(FrozenInstanceError):
        candidate.depth_fields = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("geometry_solution", cast(Any, "geometry"), "GeometrySolution"),
        ("camera_solutions", cast(Any, []), "immutable tuple"),
        ("camera_solutions", cast(Any, ("camera",)), "CameraSolution"),
        ("depth_fields", cast(Any, []), "immutable tuple"),
        ("depth_fields", cast(Any, ("depth",)), "DepthField"),
        ("point_maps", cast(Any, []), "immutable tuple"),
        ("point_maps", cast(Any, ("points",)), "PointMap"),
        ("producer", cast(Any, "producer"), "ArtifactProducerIdentity"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
        ("source_artifacts", cast(Any, ("artifact",)), "ArtifactRef"),
    ],
)
def test_candidate_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    candidate = _candidate("a")
    kwargs: dict[str, Any] = {
        field.name: getattr(candidate, field.name)
        for field in fields(GeometrySolutionCandidate)
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        GeometrySolutionCandidate(**kwargs)


def test_feed_forward_style_depth_candidate_preserves_exact_semantics() -> None:
    candidate = _candidate("feed")

    assert len(candidate.camera_solutions) == 1
    assert len(candidate.depth_fields) == 1
    assert candidate.point_maps == ()
    assert candidate.depth_fields[0].camera_solution_id == candidate.camera_solutions[0].solution_id
    assert candidate.geometry_solution.depth_field_ids == (
        candidate.depth_fields[0].depth_field_id,
    )
    assert candidate.geometry_solution.point_map_ids == ()
    assert candidate.geometry_solution.local_frame_id == LocalFrameId("frame:feed")


def test_classical_style_point_candidate_requires_no_depth_or_pycolmap() -> None:
    candidate = _candidate("classical", use_depth=False, use_point_map=True)

    assert candidate.depth_fields == ()
    assert len(candidate.point_maps) == 1
    assert candidate.geometry_solution.depth_field_ids == ()
    assert candidate.geometry_solution.point_map_ids == (
        candidate.point_maps[0].point_map_id,
    )
    assert "pycolmap" not in comparison_module.__dict__


def test_candidate_accepts_empty_source_artifacts_but_requires_producer() -> None:
    candidate = _candidate("memory", source_artifacts=())

    assert candidate.source_artifacts == ()
    assert isinstance(candidate.producer, ArtifactProducerIdentity)


def test_source_artifacts_must_be_unique_and_canonically_ordered() -> None:
    first = _artifact("artifact:a", "evidence.alpha")
    second = _artifact("artifact:b", "evidence.beta")

    assert _candidate("a", source_artifacts=(first, second)).source_artifacts == (
        first,
        second,
    )
    with pytest.raises(ValueError, match="unique"):
        _candidate("a", source_artifacts=(first, first))
    with pytest.raises(ValueError, match="canonical"):
        _candidate("a", source_artifacts=(second, first))


def test_candidate_requires_exact_camera_membership_and_order() -> None:
    candidate = _candidate("a")
    geometry = candidate.geometry_solution
    camera = candidate.camera_solutions[0]
    extra = _camera(
        frame=geometry.local_frame_id,
        observation="obs:extra",
        camera_id="camera:extra",
    )

    with pytest.raises(ValueError, match="exactly match"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(),
            depth_fields=candidate.depth_fields,
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )
    with pytest.raises(ValueError, match="exactly match"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(camera, extra),
            depth_fields=candidate.depth_fields,
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    frame = LocalFrameId("frame:order")
    camera_a = _camera(frame=frame, observation="obs:a", camera_id="camera:a")
    camera_b = _camera(frame=frame, observation="obs:b", camera_id="camera:b")
    depth_a = _depth(camera=camera_a, depth_id="depth:a")
    geometry_two = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:order"),
        local_frame_id=frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=(camera_a.solution_id, camera_b.solution_id),
        depth_field_ids=(depth_a.depth_field_id,),
        point_map_ids=(),
        metrics=_metrics(),
    )
    with pytest.raises(ValueError, match="canonical CameraSolutionId"):
        GeometrySolutionCandidate(
            geometry_solution=geometry_two,
            camera_solutions=(camera_b, camera_a),
            depth_fields=(depth_a,),
            point_maps=(),
            producer=_producer("o"),
            source_artifacts=(),
        )


def test_candidate_requires_exact_depth_membership_order_and_camera_linkage() -> None:
    candidate = _candidate("a")
    camera = candidate.camera_solutions[0]
    geometry = candidate.geometry_solution

    with pytest.raises(ValueError, match="exactly match"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(camera,),
            depth_fields=(),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    wrong_observation = _depth(
        camera=camera,
        depth_id=candidate.depth_fields[0].depth_field_id.value,
        observation="obs:foreign",
    )
    with pytest.raises(ValueError, match="observation"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(camera,),
            depth_fields=(wrong_observation,),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    wrong_dimensions = _depth(
        camera=camera,
        depth_id=candidate.depth_fields[0].depth_field_id.value,
        width_px=3,
    )
    with pytest.raises(ValueError, match="dimensions"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(camera,),
            depth_fields=(wrong_dimensions,),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_candidate_rejects_foreign_camera_and_point_local_frames() -> None:
    candidate = _candidate("a", use_point_map=True)
    geometry = candidate.geometry_solution
    foreign = LocalFrameId("frame:foreign")
    foreign_camera = _camera(
        frame=foreign,
        observation=candidate.camera_solutions[0].observation_id.value,
        camera_id=candidate.camera_solutions[0].solution_id.value,
    )
    with pytest.raises(ValueError, match="LocalFrameId"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=(foreign_camera,),
            depth_fields=candidate.depth_fields,
            point_maps=candidate.point_maps,
            producer=candidate.producer,
            source_artifacts=(),
        )

    foreign_point = _point_map(
        frame=foreign,
        point_map_id=candidate.point_maps[0].point_map_id.value,
        observation=candidate.camera_solutions[0].observation_id.value,
    )
    with pytest.raises(ValueError, match="LocalFrameId"):
        GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=candidate.camera_solutions,
            depth_fields=candidate.depth_fields,
            point_maps=(foreign_point,),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_competing_aggregate_accepts_semantically_different_alternatives() -> None:
    first = _candidate(
        "a",
        frame="frame:shared",
        projection="pinhole",
        scale=GeometryScaleStatus.UNRESOLVED,
    )
    second = _candidate(
        "b",
        frame="frame:other",
        projection="simple_radial",
        scale=GeometryScaleStatus.METRIC,
        use_depth=False,
        use_point_map=True,
    )

    competing = CompetingGeometrySolutions(candidates=(first, second))

    assert competing.candidates == (first, second)
    assert competing.candidates[0].geometry_solution.local_frame_id != (
        competing.candidates[1].geometry_solution.local_frame_id
    )
    assert competing.candidates[0].geometry_solution.scale_status is (
        GeometryScaleStatus.UNRESOLVED
    )
    assert competing.candidates[1].geometry_solution.scale_status is GeometryScaleStatus.METRIC
    assert competing.candidates[0].camera_solutions[0].projection_model.value == "pinhole"
    assert competing.candidates[1].camera_solutions[0].projection_model.value == "simple_radial"


def test_competing_aggregate_rejects_single_mutable_duplicate_and_noncanonical_input() -> None:
    first = _candidate("a")
    second = _candidate("b")

    with pytest.raises(TypeError, match="immutable tuple"):
        CompetingGeometrySolutions(candidates=cast(Any, [first, second]))
    with pytest.raises(ValueError, match="at least two"):
        CompetingGeometrySolutions(candidates=(first,))
    with pytest.raises(ValueError, match="unique"):
        CompetingGeometrySolutions(candidates=(first, first))
    with pytest.raises(ValueError, match="canonical GeometrySolutionId"):
        CompetingGeometrySolutions(candidates=(second, first))


def test_same_frame_alternatives_remain_distinct_without_merge() -> None:
    first = _candidate("a", frame="frame:shared")
    second = _candidate("b", frame="frame:shared")

    competing = CompetingGeometrySolutions(candidates=(first, second))

    assert competing.candidates[0] is first
    assert competing.candidates[1] is second
    assert first.geometry_solution.local_frame_id == second.geometry_solution.local_frame_id
    assert first.geometry_solution_id != second.geometry_solution_id


def test_pair_has_exact_frozen_identity_only_shape_and_canonical_order() -> None:
    pair = GeometrySolutionPair(
        left_geometry_solution_id=GeometrySolutionId("geometry:a"),
        right_geometry_solution_id=GeometrySolutionId("geometry:b"),
    )

    assert tuple(field.name for field in fields(GeometrySolutionPair)) == (
        "left_geometry_solution_id",
        "right_geometry_solution_id",
    )
    with pytest.raises(FrozenInstanceError):
        pair.left_geometry_solution_id = GeometrySolutionId("geometry:c")  # type: ignore[misc]
    with pytest.raises(ValueError, match="distinct"):
        GeometrySolutionPair(
            left_geometry_solution_id=GeometrySolutionId("geometry:a"),
            right_geometry_solution_id=GeometrySolutionId("geometry:a"),
        )
    with pytest.raises(ValueError, match="canonical"):
        GeometrySolutionPair(
            left_geometry_solution_id=GeometrySolutionId("geometry:b"),
            right_geometry_solution_id=GeometrySolutionId("geometry:a"),
        )


def test_pair_derivation_is_complete_deterministic_and_identity_only() -> None:
    candidates = tuple(_candidate(token) for token in ("a", "b", "c", "d"))
    competing = CompetingGeometrySolutions(candidates=candidates)

    pairs = derive_geometry_solution_pairs(competing)

    assert pairs == (
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:b")),
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:c")),
        GeometrySolutionPair(GeometrySolutionId("geometry:a"), GeometrySolutionId("geometry:d")),
        GeometrySolutionPair(GeometrySolutionId("geometry:b"), GeometrySolutionId("geometry:c")),
        GeometrySolutionPair(GeometrySolutionId("geometry:b"), GeometrySolutionId("geometry:d")),
        GeometrySolutionPair(GeometrySolutionId("geometry:c"), GeometrySolutionId("geometry:d")),
    )
    assert len(pairs) == 6
    for pair in pairs:
        assert not hasattr(pair, "transform")
        assert not hasattr(pair, "metric")
        assert not hasattr(pair, "score")
        assert not hasattr(pair, "winner")
        assert not hasattr(pair, "preferred_candidate")
        assert not hasattr(pair, "decision")


def test_pair_derivation_rejects_untyped_aggregate() -> None:
    with pytest.raises(TypeError, match="CompetingGeometrySolutions"):
        derive_geometry_solution_pairs(cast(Any, ("geometry:a", "geometry:b")))


def test_comparison_construction_does_not_mutate_source_candidates() -> None:
    first = _candidate("a")
    second = _candidate("b")
    first_before = first
    second_before = second
    first_camera_before = first.camera_solutions[0]
    second_geometry_before = second.geometry_solution

    competing = CompetingGeometrySolutions(candidates=(first, second))
    pairs = derive_geometry_solution_pairs(competing)

    assert first == first_before
    assert second == second_before
    assert competing.candidates[0] is first
    assert competing.candidates[1] is second
    assert first.camera_solutions[0] is first_camera_before
    assert second.geometry_solution is second_geometry_before
    assert len(pairs) == 1


def test_comparison_module_has_no_alignment_solver_metric_or_later_layer_surface() -> None:
    forbidden = {
        "torch",
        "numpy",
        "pycolmap",
        "Da3ExecutionResult",
        "CanonicalColmapSparseModel",
        "ColmapGeometryOutcome",
        "QualityDecision",
        "QualityPolicy",
        "MetricObservation",
        "MetricDescriptor",
        "Sim3",
        "SE3",
        "BundleAdjustment",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
        "Router",
    }

    assert forbidden.isdisjoint(vars(comparison_module))
