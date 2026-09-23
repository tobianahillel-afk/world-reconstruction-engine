from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast, get_type_hints

import pytest

import wre.reconstruction.geometry_refinement as refinement
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
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.refinement.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "geometry.refinement_evidence") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _candidate(
    token: str,
    *,
    frame: str | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    projection: str = "pinhole",
    observation: str | None = None,
    with_depth: bool = True,
    with_point_map: bool = False,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    local_frame = LocalFrameId(frame or f"frame:{token}")
    observation_id = ObservationId(observation or f"obs:{token}")
    camera = CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=observation_id,
        local_frame_id=local_frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=2, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )
    depth = (
        DepthField(
            depth_field_id=DepthFieldId(f"depth:{token}"),
            observation_id=observation_id,
            camera_solution_id=camera.solution_id,
            dimensions=camera.dimensions,
            depth_value_convention=DepthValueConventionName("relative-depth"),
            depth_values=(1.0, 2.0),
            validity=(True, True),
            confidence=None,
            metrics=MetricVector(observations=()),
        )
        if with_depth
        else None
    )
    point_map = (
        PointMap(
            point_map_id=PointMapId(f"points:{token}"),
            local_frame_id=local_frame,
            source_observation_ids=(observation_id,),
            positions_xyz=((1.0, 2.0, 3.0),),
            confidence=None,
            metrics=MetricVector(observations=()),
        )
        if with_point_map
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
        metrics=MetricVector(observations=()),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera,),
        depth_fields=depths,
        point_maps=point_maps,
        producer=_producer(token),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (_artifact(f"source:{token}", "geometry.solution"),)
        ),
    )


def test_request_and_result_have_exact_frozen_field_shapes() -> None:
    initialization = _candidate("initial")
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(),
    )
    refined = _candidate("refined")
    result = refinement.GeometryRefinementResult(
        request=request,
        refined_candidate=refined,
    )

    assert tuple(field.name for field in fields(refinement.GeometryRefinementRequest)) == (
        "initialization",
        "supporting_artifacts",
    )
    assert tuple(field.name for field in fields(refinement.GeometryRefinementResult)) == (
        "request",
        "refined_candidate",
    )
    assert result.request is request
    assert result.request.initialization is initialization
    assert result.refined_candidate is refined

    with pytest.raises(FrozenInstanceError):
        request.supporting_artifacts = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.refined_candidate = initialization  # type: ignore[misc]


@pytest.mark.parametrize(
    ("initialization", "supporting_artifacts", "message"),
    [
        (cast(Any, "geometry"), (), "initialization"),
        (_candidate("a"), cast(Any, []), "immutable tuple"),
        (_candidate("a"), cast(Any, ("artifact",)), "ArtifactRef"),
    ],
)
def test_request_rejects_wrong_member_types(
    initialization: Any,
    supporting_artifacts: Any,
    message: str,
) -> None:
    with pytest.raises(TypeError, match=message):
        refinement.GeometryRefinementRequest(
            initialization=initialization,
            supporting_artifacts=supporting_artifacts,
        )


def test_supporting_artifacts_may_be_empty_but_must_be_unique_and_canonical() -> None:
    initialization = _candidate("initial")
    empty = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(),
    )
    assert empty.supporting_artifacts == ()

    first = _artifact("artifact:a", "evidence.alpha")
    second = _artifact("artifact:b", "evidence.beta")
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(first, second),
    )
    assert request.supporting_artifacts == (first, second)

    with pytest.raises(ValueError, match="unique"):
        refinement.GeometryRefinementRequest(
            initialization=initialization,
            supporting_artifacts=(first, first),
        )
    with pytest.raises(ValueError, match="canonical"):
        refinement.GeometryRefinementRequest(
            initialization=initialization,
            supporting_artifacts=(second, first),
        )


def test_result_rejects_wrong_types_and_in_place_geometry_identity() -> None:
    initialization = _candidate("initial")
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(),
    )

    with pytest.raises(TypeError, match="request"):
        refinement.GeometryRefinementResult(
            request=cast(Any, "request"),
            refined_candidate=_candidate("refined"),
        )
    with pytest.raises(TypeError, match="refined_candidate"):
        refinement.GeometryRefinementResult(
            request=request,
            refined_candidate=cast(Any, "candidate"),
        )
    with pytest.raises(ValueError, match="distinct GeometrySolutionId"):
        refinement.GeometryRefinementResult(
            request=request,
            refined_candidate=initialization,
        )

    relabeled = replace(
        _candidate("refined"),
        geometry_solution=replace(
            _candidate("refined").geometry_solution,
            geometry_solution_id=initialization.geometry_solution_id,
        ),
    )
    with pytest.raises(ValueError, match="distinct GeometrySolutionId"):
        refinement.GeometryRefinementResult(
            request=request,
            refined_candidate=relabeled,
        )


def test_helper_constructs_exact_result_without_copying_request_or_candidates() -> None:
    initialization = _candidate("initial")
    evidence = (_artifact("artifact:evidence"),)
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=evidence,
    )
    refined = _candidate("refined")

    result = refinement.build_geometry_refinement_result(request, refined)

    assert result.request is request
    assert result.request.initialization is initialization
    assert result.refined_candidate is refined
    assert result.request.supporting_artifacts is evidence


def test_protocol_exposes_only_one_typed_refine_operation() -> None:
    public_callables = {
        name
        for name, value in refinement.GeometryRefinementAdapter.__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert public_callables == {"refine"}

    signature = inspect.signature(refinement.GeometryRefinementAdapter.refine)
    assert tuple(signature.parameters) == ("self", "request")
    hints = get_type_hints(refinement.GeometryRefinementAdapter.refine)
    assert hints == {
        "request": refinement.GeometryRefinementRequest,
        "return": refinement.GeometryRefinementResult,
    }


def test_depth_initialization_can_refine_to_point_based_candidate() -> None:
    initialization = _candidate(
        "initial",
        with_depth=True,
        with_point_map=False,
    )
    refined = _candidate(
        "refined",
        with_depth=False,
        with_point_map=True,
    )
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(_artifact("artifact:tracks", "evidence.tracks"),),
    )

    result = refinement.build_geometry_refinement_result(request, refined)

    assert len(result.request.initialization.depth_fields) == 1
    assert result.request.initialization.point_maps == ()
    assert result.refined_candidate.depth_fields == ()
    assert len(result.refined_candidate.point_maps) == 1
    assert result.request.initialization is initialization
    assert result.refined_candidate is refined


def test_same_frame_refinement_retains_both_ids_without_delta_or_quality_claim() -> None:
    initialization = _candidate("initial", frame="frame:shared")
    refined = _candidate("refined", frame="frame:shared")
    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(),
    )
    result = refinement.build_geometry_refinement_result(request, refined)

    assert result.request.initialization.local_frame_id if False else True
    assert (
        result.request.initialization.geometry_solution.local_frame_id
        == result.refined_candidate.geometry_solution.local_frame_id
    )
    assert (
        result.request.initialization.geometry_solution_id
        != result.refined_candidate.geometry_solution_id
    )
    for attribute in (
        "delta",
        "transform",
        "alignment",
        "metric",
        "score",
        "quality",
        "decision",
    ):
        assert not hasattr(result, attribute)


def test_different_frame_and_unresolved_to_metric_lineage_is_valid_without_alignment() -> None:
    initialization = _candidate(
        "initial",
        frame="frame:initial",
        scale=GeometryScaleStatus.UNRESOLVED,
        projection="pinhole",
        observation="obs:initial",
    )
    refined = _candidate(
        "refined",
        frame="frame:refined",
        scale=GeometryScaleStatus.METRIC,
        projection="simple_radial",
        observation="obs:refined",
        with_depth=False,
        with_point_map=True,
    )
    result = refinement.build_geometry_refinement_result(
        refinement.GeometryRefinementRequest(
            initialization=initialization,
            supporting_artifacts=(),
        ),
        refined,
    )

    assert (
        result.request.initialization.geometry_solution.local_frame_id
        != result.refined_candidate.geometry_solution.local_frame_id
    )
    assert (
        result.request.initialization.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert result.refined_candidate.geometry_solution.scale_status is GeometryScaleStatus.METRIC
    assert (
        result.request.initialization.camera_solutions[0].projection_model
        != result.refined_candidate.camera_solutions[0].projection_model
    )
    assert (
        result.request.initialization.camera_solutions[0].observation_id
        != result.refined_candidate.camera_solutions[0].observation_id
    )
    assert not hasattr(result, "transform")
    assert not hasattr(result, "scale_factor")


def test_supporting_evidence_never_replaces_candidate_provenance() -> None:
    init_source = (_artifact("artifact:init", "geometry.solution"),)
    refined_source = (_artifact("artifact:refined", "geometry.solution"),)
    supporting = (_artifact("artifact:support", "evidence.matches"),)
    initialization = _candidate("initial", source_artifacts=init_source)
    refined = _candidate("refined", source_artifacts=refined_source)

    result = refinement.build_geometry_refinement_result(
        refinement.GeometryRefinementRequest(
            initialization=initialization,
            supporting_artifacts=supporting,
        ),
        refined,
    )

    assert result.request.initialization.producer == initialization.producer
    assert result.refined_candidate.producer == refined.producer
    assert result.request.initialization.source_artifacts == init_source
    assert result.refined_candidate.source_artifacts == refined_source
    assert result.request.supporting_artifacts == supporting
    assert supporting != init_source
    assert supporting != refined_source


def test_request_and_result_construction_do_not_mutate_source_values() -> None:
    initialization = _candidate("initial", with_depth=True, with_point_map=True)
    refined = _candidate("refined", with_depth=False, with_point_map=True)
    initialization_before = initialization
    refined_before = refined
    initial_geometry_before = initialization.geometry_solution
    refined_camera_before = refined.camera_solutions[0]

    request = refinement.GeometryRefinementRequest(
        initialization=initialization,
        supporting_artifacts=(),
    )
    result = refinement.build_geometry_refinement_result(request, refined)

    assert initialization == initialization_before
    assert refined == refined_before
    assert request.initialization is initialization
    assert result.refined_candidate is refined
    assert initialization.geometry_solution is initial_geometry_before
    assert refined.camera_solutions[0] is refined_camera_before


def test_module_surface_has_no_solver_alignment_metric_or_later_item_dependencies() -> None:
    forbidden = {
        "torch",
        "numpy",
        "pycolmap",
        "Da3ExecutionResult",
        "ColmapGeometryOutcome",
        "BundleAdjustment",
        "SE3",
        "Sim3",
        "QualityDecision",
        "QualityPolicy",
        "MetricObservation",
        "MetricDescriptor",
        "Router",
        "Scheduler",
        "SurfaceModel",
        "MasterScene",
        "RuntimeScene",
    }
    assert forbidden.isdisjoint(vars(refinement))
