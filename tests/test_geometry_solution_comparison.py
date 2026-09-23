from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast

import pytest

import wre.reconstruction.geometry_solution_comparison as comparison
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


def _producer(seed: str = "a") -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.geometry.{seed}",
            version="1",
            revision=f"revision:{seed}",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest(seed * 64)),
    )


def _source(identifier: str, kind: str = "geometry.solution") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _camera(
    candidate: str,
    index: int,
    *,
    frame: LocalFrameId,
    projection: str = "pinhole",
    observation: str | None = None,
    dimensions: ImageDimensions | None = None,
) -> CameraSolution:
    obs = observation or f"obs:{index}"
    dims = dimensions or ImageDimensions(width_px=4, height_px=2)
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{candidate}:{index}"),
        observation_id=ObservationId(obs),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=dims,
        intrinsic_parameters=(500.0, 510.0, 2.0, 1.0),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(float(index), 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )


def _depth(candidate: str, index: int, camera: CameraSolution) -> DepthField:
    pixel_count = camera.dimensions.width_px * camera.dimensions.height_px
    return DepthField(
        depth_field_id=DepthFieldId(f"depth:{candidate}:{index}"),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName("relative_depth"),
        depth_values=(1.0,) * pixel_count,
        validity=(True,) * pixel_count,
        confidence=None,
        metrics=MetricVector(observations=()),
    )


def _point_map(
    candidate: str,
    *,
    frame: LocalFrameId,
    observations: tuple[ObservationId, ...],
) -> PointMap:
    return PointMap(
        point_map_id=PointMapId(f"points:{candidate}"),
        local_frame_id=frame,
        source_observation_ids=observations,
        positions_xyz=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
        confidence=None,
        metrics=MetricVector(observations=()),
    )


def _candidate(
    candidate: str,
    *,
    frame_value: str | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    with_depth: bool = True,
    with_points: bool = False,
    projection: str = "pinhole",
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> comparison.GeometrySolutionCandidate:
    frame = LocalFrameId(frame_value or f"frame:{candidate}")
    cameras = (
        _camera(candidate, 1, frame=frame, projection=projection),
        _camera(candidate, 2, frame=frame, projection=projection),
    )
    depths = (
        tuple(_depth(candidate, index, camera) for index, camera in enumerate(cameras, 1))
        if with_depth
        else ()
    )
    point_maps = (
        (
            _point_map(
                candidate,
                frame=frame,
                observations=tuple(camera.observation_id for camera in cameras),
            ),
        )
        if with_points
        else ()
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{candidate}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=tuple(camera.solution_id for camera in cameras),
        depth_field_ids=tuple(depth.depth_field_id for depth in depths),
        point_map_ids=tuple(point.point_map_id for point in point_maps),
        metrics=MetricVector(observations=()),
    )
    return comparison.GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=cameras,
        depth_fields=depths,
        point_maps=point_maps,
        producer=_producer(candidate[0]),
        source_artifacts=source_artifacts
        if source_artifacts is not None
        else (_source(f"source:{candidate}"),),
    )


def test_candidate_has_exact_frozen_field_shape_and_retains_identity() -> None:
    candidate = _candidate("a", with_depth=True, with_points=True)

    assert tuple(field.name for field in fields(comparison.GeometrySolutionCandidate)) == (
        "geometry_solution",
        "camera_solutions",
        "depth_fields",
        "point_maps",
        "producer",
        "source_artifacts",
    )
    assert candidate.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert (
        candidate.camera_solutions[0].local_frame_id
        == candidate.geometry_solution.local_frame_id
    )
    assert candidate.point_maps[0].local_frame_id == candidate.geometry_solution.local_frame_id
    with pytest.raises(FrozenInstanceError):
        candidate.producer = _producer("z")  # type: ignore[misc]


def test_feed_forward_style_candidate_preserves_depth_without_point_map() -> None:
    candidate = _candidate("a", with_depth=True, with_points=False)

    assert len(candidate.camera_solutions) == 2
    assert len(candidate.depth_fields) == 2
    assert candidate.point_maps == ()
    assert candidate.geometry_solution.depth_field_ids == tuple(
        item.depth_field_id for item in candidate.depth_fields
    )
    assert candidate.geometry_solution.point_map_ids == ()


def test_classical_style_candidate_preserves_point_map_without_depth() -> None:
    candidate = _candidate("b", with_depth=False, with_points=True)

    assert len(candidate.camera_solutions) == 2
    assert candidate.depth_fields == ()
    assert len(candidate.point_maps) == 1
    assert candidate.geometry_solution.depth_field_ids == ()
    assert candidate.geometry_solution.point_map_ids == (
        candidate.point_maps[0].point_map_id,
    )


@pytest.mark.parametrize("field_name", ["camera_solutions", "depth_fields", "point_maps"])
def test_candidate_rejects_mutable_child_collections(field_name: str) -> None:
    candidate = _candidate("a", with_depth=True, with_points=True)
    kwargs: dict[str, Any] = {
        "geometry_solution": candidate.geometry_solution,
        "camera_solutions": candidate.camera_solutions,
        "depth_fields": candidate.depth_fields,
        "point_maps": candidate.point_maps,
        "producer": candidate.producer,
        "source_artifacts": candidate.source_artifacts,
    }
    kwargs[field_name] = list(kwargs[field_name])

    with pytest.raises(TypeError, match="immutable tuple"):
        comparison.GeometrySolutionCandidate(**kwargs)


def test_candidate_requires_exact_camera_membership_and_canonical_order() -> None:
    candidate = _candidate("a")

    with pytest.raises(ValueError, match="exactly all supplied cameras"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions[:1],
            depth_fields=candidate.depth_fields,
            point_maps=candidate.point_maps,
            producer=candidate.producer,
            source_artifacts=candidate.source_artifacts,
        )

    with pytest.raises(ValueError, match="canonical CameraSolutionId order"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=tuple(reversed(candidate.camera_solutions)),
            depth_fields=candidate.depth_fields,
            point_maps=candidate.point_maps,
            producer=candidate.producer,
            source_artifacts=candidate.source_artifacts,
        )

    with pytest.raises(ValueError, match="unique IDs"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=replace(
                candidate.geometry_solution,
                camera_solution_ids=(candidate.camera_solutions[0].solution_id,),
            ),
            camera_solutions=(candidate.camera_solutions[0], candidate.camera_solutions[0]),
            depth_fields=(),
            point_maps=(
                _point_map(
                    "duplicate-camera",
                    frame=candidate.geometry_solution.local_frame_id,
                    observations=(candidate.camera_solutions[0].observation_id,),
                ),
            ),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_candidate_requires_exact_depth_membership_order_and_camera_linkage() -> None:
    candidate = _candidate("a")

    with pytest.raises(ValueError, match="exactly all supplied depths"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=candidate.depth_fields[:1],
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    with pytest.raises(ValueError, match="canonical DepthFieldId order"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=tuple(reversed(candidate.depth_fields)),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    foreign = replace(
        candidate.depth_fields[0],
        camera_solution_id=CameraSolutionId("camera:foreign"),
    )
    geometry = replace(
        candidate.geometry_solution,
        depth_field_ids=(foreign.depth_field_id, candidate.depth_fields[1].depth_field_id),
    )
    with pytest.raises(ValueError, match="reference a supplied CameraSolution"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(foreign, candidate.depth_fields[1]),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_candidate_rejects_depth_observation_or_dimensions_mismatch() -> None:
    candidate = _candidate("a")
    depth = candidate.depth_fields[0]

    wrong_observation = replace(depth, observation_id=ObservationId("obs:foreign"))
    geometry = replace(
        candidate.geometry_solution,
        depth_field_ids=(
            wrong_observation.depth_field_id,
            candidate.depth_fields[1].depth_field_id,
        ),
    )
    with pytest.raises(ValueError, match="observation must match"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=geometry,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(wrong_observation, candidate.depth_fields[1]),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    wrong_dimensions = replace(
        depth,
        dimensions=ImageDimensions(width_px=2, height_px=4),
        depth_values=(1.0,) * 8,
        validity=(True,) * 8,
    )
    with pytest.raises(ValueError, match="dimensions must match"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(wrong_dimensions, candidate.depth_fields[1]),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_candidate_requires_exact_point_membership_frame_and_support() -> None:
    candidate = _candidate("a", with_depth=False, with_points=True)
    point = candidate.point_maps[0]

    with pytest.raises(ValueError, match="exactly all supplied point maps"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(),
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )

    foreign_frame = replace(point, local_frame_id=LocalFrameId("frame:foreign"))
    with pytest.raises(ValueError, match="share one LocalFrameId"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(),
            point_maps=(foreign_frame,),
            producer=candidate.producer,
            source_artifacts=(),
        )

    foreign_support = replace(
        point,
        source_observation_ids=(ObservationId("obs:foreign"),),
    )
    with pytest.raises(ValueError, match="camera membership"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=(),
            point_maps=(foreign_support,),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_candidate_rejects_foreign_camera_frame() -> None:
    candidate = _candidate("a")
    foreign = replace(
        candidate.camera_solutions[0],
        local_frame_id=LocalFrameId("frame:foreign"),
    )

    with pytest.raises(ValueError, match="share one LocalFrameId"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=(foreign, candidate.camera_solutions[1]),
            depth_fields=candidate.depth_fields,
            point_maps=(),
            producer=candidate.producer,
            source_artifacts=(),
        )


def test_source_artifacts_may_be_empty_but_must_be_unique_and_canonical() -> None:
    empty = _candidate("a", source_artifacts=())
    assert empty.source_artifacts == ()

    first = _source("artifact:a", "evidence.a")
    second = _source("artifact:b", "evidence.b")
    candidate = _candidate("b", source_artifacts=(first, second))
    assert candidate.source_artifacts == (first, second)

    with pytest.raises(ValueError, match="canonical"):
        _candidate("c", source_artifacts=(second, first))
    with pytest.raises(ValueError, match="unique"):
        _candidate("d", source_artifacts=(first, first))
    with pytest.raises(TypeError, match="immutable tuple"):
        comparison.GeometrySolutionCandidate(
            geometry_solution=candidate.geometry_solution,
            camera_solutions=candidate.camera_solutions,
            depth_fields=candidate.depth_fields,
            point_maps=candidate.point_maps,
            producer=candidate.producer,
            source_artifacts=cast(Any, [first]),
        )


def test_mixed_candidates_preserve_incompatible_frames_scale_and_projection() -> None:
    first = _candidate(
        "a",
        frame_value="frame:a",
        scale=GeometryScaleStatus.UNRESOLVED,
        with_depth=True,
        projection="pinhole",
    )
    second = _candidate(
        "b",
        frame_value="frame:b",
        scale=GeometryScaleStatus.METRIC,
        with_depth=False,
        with_points=True,
        projection="fisheye",
    )

    competing = comparison.CompetingGeometrySolutions(candidates=(first, second))

    assert competing.candidates == (first, second)
    assert first.geometry_solution.local_frame_id != second.geometry_solution.local_frame_id
    assert first.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert second.geometry_solution.scale_status is GeometryScaleStatus.METRIC
    assert first.camera_solutions[0].projection_model != second.camera_solutions[0].projection_model
    assert not hasattr(competing, "alignment")
    assert not hasattr(competing, "score")
    assert not hasattr(competing, "winner")


def test_competing_candidates_require_two_unique_canonical_geometry_ids() -> None:
    first = _candidate("a")
    second = _candidate("b")

    with pytest.raises(ValueError, match="at least two"):
        comparison.CompetingGeometrySolutions(candidates=(first,))
    with pytest.raises(TypeError, match="immutable tuple"):
        comparison.CompetingGeometrySolutions(candidates=cast(Any, [first, second]))
    with pytest.raises(ValueError, match="canonical GeometrySolutionId order"):
        comparison.CompetingGeometrySolutions(candidates=(second, first))

    duplicate = replace(
        second,
        geometry_solution=replace(
            second.geometry_solution,
            geometry_solution_id=first.geometry_solution.geometry_solution_id,
        ),
    )
    with pytest.raises(ValueError, match="unique GeometrySolutionIds"):
        comparison.CompetingGeometrySolutions(candidates=(first, duplicate))


def test_pair_identity_is_frozen_distinct_and_canonical() -> None:
    left = GeometrySolutionId("geometry:a")
    right = GeometrySolutionId("geometry:b")
    pair = comparison.GeometrySolutionPair(
        left_geometry_solution_id=left,
        right_geometry_solution_id=right,
    )

    assert tuple(field.name for field in fields(comparison.GeometrySolutionPair)) == (
        "left_geometry_solution_id",
        "right_geometry_solution_id",
    )
    with pytest.raises(FrozenInstanceError):
        pair.left_geometry_solution_id = right  # type: ignore[misc]
    with pytest.raises(ValueError, match="distinct"):
        comparison.GeometrySolutionPair(
            left_geometry_solution_id=left,
            right_geometry_solution_id=left,
        )
    with pytest.raises(ValueError, match="canonical lexical"):
        comparison.GeometrySolutionPair(
            left_geometry_solution_id=right,
            right_geometry_solution_id=left,
        )


@pytest.mark.parametrize("count", [2, 3, 5])
def test_pair_derivation_is_complete_deterministic_and_lexical(count: int) -> None:
    candidates = tuple(_candidate(chr(ord("a") + index)) for index in range(count))
    competing = comparison.CompetingGeometrySolutions(candidates=candidates)

    first = comparison.derive_geometry_solution_pairs(competing)
    second = comparison.derive_geometry_solution_pairs(competing)

    assert first == second
    assert len(first) == count * (count - 1) // 2
    assert first == tuple(sorted(first))
    assert all(
        pair.left_geometry_solution_id.value < pair.right_geometry_solution_id.value
        for pair in first
    )


def test_same_frame_alternatives_are_not_merged_and_different_frames_are_not_aligned() -> None:
    shared_frame = "frame:shared"
    first = _candidate("a", frame_value=shared_frame)
    second = _candidate("b", frame_value=shared_frame, with_depth=False, with_points=True)
    third = _candidate("c", frame_value="frame:other")
    competing = comparison.CompetingGeometrySolutions(candidates=(first, second, third))

    assert competing.candidates[0] is first
    assert competing.candidates[1] is second
    assert competing.candidates[2] is third
    assert first.geometry_solution.local_frame_id == second.geometry_solution.local_frame_id
    assert third.geometry_solution.local_frame_id != first.geometry_solution.local_frame_id
    pairs = comparison.derive_geometry_solution_pairs(competing)
    assert len(pairs) == 3
    assert all(not hasattr(pair, "transform") for pair in pairs)
    assert all(not hasattr(pair, "preferred_candidate") for pair in pairs)


def test_aggregate_and_pair_derivation_do_not_mutate_source_values() -> None:
    first = _candidate("a", with_depth=True, with_points=True)
    second = _candidate("b", with_depth=False, with_points=True)
    before = (first, second)

    competing = comparison.CompetingGeometrySolutions(candidates=before)
    pairs = comparison.derive_geometry_solution_pairs(competing)

    assert competing.candidates == before
    assert competing.candidates[0] is first
    assert competing.candidates[1] is second
    assert first == before[0]
    assert second == before[1]
    assert pairs == (
        comparison.GeometrySolutionPair(
            left_geometry_solution_id=first.geometry_solution.geometry_solution_id,
            right_geometry_solution_id=second.geometry_solution.geometry_solution_id,
        ),
    )


def test_module_surface_has_no_solver_alignment_metric_or_decision_dependencies() -> None:
    forbidden = {
        "torch",
        "numpy",
        "pycolmap",
        "Da3ExecutionResult",
        "CanonicalColmapSparseModel",
        "QualityDecision",
        "QualityPolicy",
        "MetricObservation",
        "alignment",
        "refinement",
        "winner",
        "score",
        "route",
    }
    assert forbidden.isdisjoint(vars(comparison))
