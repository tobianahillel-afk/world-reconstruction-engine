from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast, get_type_hints

import pytest

import wre.reconstruction.hybrid_surface_candidate as hybrid_module
from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
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
from wre.domain.surfaces import (
    SURFACE_MODEL_ARTIFACT_KIND,
    SurfaceIntendedUse,
    SurfaceModel,
    SurfaceRepresentationName,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate
from wre.reconstruction.hybrid_surface_candidate import (
    HybridSurfaceCandidateAdapter,
    HybridSurfaceCandidateRequest,
    HybridSurfaceCandidateResult,
    build_hybrid_surface_candidate_result,
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.hybrid_surface.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "evidence.source") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _canonical_artifacts(*values: ArtifactRef) -> tuple[ArtifactRef, ...]:
    by_key = {(item.artifact_id.value, item.artifact_kind.value): item for item in values}
    return tuple(by_key[key] for key in sorted(by_key))


def _camera(token: str, *, frame: LocalFrameId) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=2, height_px=1),
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


def _candidate(
    token: str,
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    camera = _camera(token, frame=frame)
    point_map = PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=(camera.observation_id,),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera,),
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (_artifact(f"artifact:{token}:source", "geometry.input"),)
        ),
    )


def _request(
    token: str = "request",
    *,
    source: GeometrySolutionCandidate | None = None,
    supporting: tuple[ArtifactRef, ...] = (),
    representation: str = "mesh",
    intended_uses: tuple[SurfaceIntendedUse, ...] = (SurfaceIntendedUse.COLLISION,),
) -> HybridSurfaceCandidateRequest:
    return HybridSurfaceCandidateRequest(
        source_geometry=source or _candidate(token),
        supporting_artifacts=supporting,
        output_representation=SurfaceRepresentationName(representation),
        intended_uses=intended_uses,
    )


def _surface(
    token: str,
    *,
    source: GeometrySolutionCandidate,
    source_artifacts: tuple[ArtifactRef, ...],
    representation: SurfaceRepresentationName | None = None,
    intended_uses: tuple[SurfaceIntendedUse, ...] = (SurfaceIntendedUse.COLLISION,),
    artifact_ref: ArtifactRef | None = None,
    producer: ArtifactProducerIdentity | None = None,
) -> SurfaceModel:
    return SurfaceModel(
        artifact_ref=artifact_ref
        or ArtifactRef(
            artifact_id=ArtifactId(f"surface:{token}"),
            artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
        ),
        source_geometry=source,
        representation=representation or SurfaceRepresentationName("mesh"),
        intended_uses=intended_uses,
        local_frame_id=source.geometry_solution.local_frame_id,
        scale_status=source.geometry_solution.scale_status,
        producer=producer or _producer(f"surface:{token}"),
        source_artifacts=source_artifacts,
    )


def _materialization(
    surface: SurfaceModel,
    *,
    artifact_ref: ArtifactRef | None = None,
) -> ArtifactMaterializationMetadata:
    return ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref or surface.artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path="candidate/surface.ply",
                sha256=Sha256Digest("1" * 64),
                byte_length=1234,
            ),
        ),
    )


def _valid_parts(
    token: str = "valid",
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    supporting: tuple[ArtifactRef, ...] | None = None,
    representation: str = "mesh",
    intended_uses: tuple[SurfaceIntendedUse, ...] = (SurfaceIntendedUse.COLLISION,),
) -> tuple[
    HybridSurfaceCandidateRequest,
    SurfaceModel,
    ArtifactMaterializationMetadata,
]:
    source = _candidate(token, scale=scale)
    retained_support = (
        supporting
        if supporting is not None
        else (_artifact(f"artifact:{token}:support", "evidence.depth"),)
    )
    request = _request(
        token,
        source=source,
        supporting=retained_support,
        representation=representation,
        intended_uses=intended_uses,
    )
    ancestry = _canonical_artifacts(
        *source.source_artifacts,
        *retained_support,
    )
    candidate = _surface(
        token,
        source=source,
        source_artifacts=ancestry,
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    return request, candidate, _materialization(candidate)


def test_contract_has_exact_frozen_shapes_protocol_and_public_surface() -> None:
    request, candidate, materialization = _valid_parts("shape")
    result = HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=materialization,
    )

    assert tuple(field.name for field in fields(HybridSurfaceCandidateRequest)) == (
        "source_geometry",
        "supporting_artifacts",
        "output_representation",
        "intended_uses",
    )
    assert tuple(field.name for field in fields(HybridSurfaceCandidateResult)) == (
        "request",
        "candidate",
        "materialization",
    )

    signature = inspect.signature(HybridSurfaceCandidateAdapter.derive_candidate)
    assert tuple(signature.parameters) == ("self", "request")
    hints = get_type_hints(HybridSurfaceCandidateAdapter.derive_candidate)
    assert hints["request"] is HybridSurfaceCandidateRequest
    assert hints["return"] is HybridSurfaceCandidateResult

    assert set(hybrid_module.__all__) == {
        "HybridSurfaceCandidateAdapter",
        "HybridSurfaceCandidateRequest",
        "HybridSurfaceCandidateResult",
        "build_hybrid_surface_candidate_result",
    }

    with pytest.raises(FrozenInstanceError):
        request.supporting_artifacts = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidate = candidate  # type: ignore[misc]


@pytest.mark.parametrize(
    ("source", "supporting", "representation", "intended_uses", "error_type", "message"),
    [
        (
            cast(Any, "geometry"),
            (),
            SurfaceRepresentationName("mesh"),
            (SurfaceIntendedUse.COLLISION,),
            TypeError,
            "source_geometry",
        ),
        (
            _candidate("list-support"),
            cast(Any, []),
            SurfaceRepresentationName("mesh"),
            (SurfaceIntendedUse.COLLISION,),
            TypeError,
            "immutable tuple",
        ),
        (
            _candidate("wrong-support"),
            cast(Any, ("artifact",)),
            SurfaceRepresentationName("mesh"),
            (SurfaceIntendedUse.COLLISION,),
            TypeError,
            "ArtifactRef",
        ),
        (
            _candidate("wrong-representation"),
            (),
            cast(Any, "mesh"),
            (SurfaceIntendedUse.COLLISION,),
            TypeError,
            "SurfaceRepresentationName",
        ),
        (
            _candidate("list-uses"),
            (),
            SurfaceRepresentationName("mesh"),
            cast(Any, [SurfaceIntendedUse.COLLISION]),
            TypeError,
            "immutable tuple",
        ),
        (
            _candidate("empty-uses"),
            (),
            SurfaceRepresentationName("mesh"),
            (),
            ValueError,
            "non-empty",
        ),
        (
            _candidate("wrong-uses"),
            (),
            SurfaceRepresentationName("mesh"),
            cast(Any, ("collision",)),
            TypeError,
            "SurfaceIntendedUse",
        ),
    ],
)
def test_request_rejects_wrong_member_shapes(
    source: Any,
    supporting: Any,
    representation: Any,
    intended_uses: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        HybridSurfaceCandidateRequest(
            source_geometry=source,
            supporting_artifacts=supporting,
            output_representation=representation,
            intended_uses=intended_uses,
        )


def test_supporting_artifacts_allow_empty_and_require_unique_canonical_order() -> None:
    source = _candidate("support")
    empty = HybridSurfaceCandidateRequest(
        source_geometry=source,
        supporting_artifacts=(),
        output_representation=SurfaceRepresentationName("mesh"),
        intended_uses=(SurfaceIntendedUse.COLLISION,),
    )
    assert empty.supporting_artifacts == ()

    first = _artifact("artifact:a", "evidence.alpha")
    second = _artifact("artifact:b", "evidence.beta")
    supporting = (first, second)
    request = _request("support", source=source, supporting=supporting)
    assert request.supporting_artifacts is supporting

    with pytest.raises(ValueError, match="unique"):
        _request("duplicate-support", source=source, supporting=(first, first))
    with pytest.raises(ValueError, match="canonical"):
        _request("unordered-support", source=source, supporting=(second, first))


def test_supporting_artifact_may_overlap_source_ancestry_without_replacing_it() -> None:
    source = _candidate("overlap")
    request = _request(
        "overlap",
        source=source,
        supporting=(source.source_artifacts[0],),
    )
    candidate = _surface(
        "overlap",
        source=source,
        source_artifacts=source.source_artifacts,
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    result = HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=_materialization(candidate),
    )

    assert result.candidate.source_artifacts == source.source_artifacts


def test_intended_uses_must_be_non_empty_unique_and_canonical_without_coercion() -> None:
    source = _candidate("uses")
    uses = (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.MEASUREMENT,
        SurfaceIntendedUse.NAVIGATION,
    )
    request = _request("uses", source=source, intended_uses=uses)
    assert request.intended_uses is uses

    with pytest.raises(ValueError, match="unique"):
        _request(
            "duplicate-uses",
            source=source,
            intended_uses=(
                SurfaceIntendedUse.COLLISION,
                SurfaceIntendedUse.COLLISION,
            ),
        )
    with pytest.raises(ValueError, match="canonical"):
        _request(
            "unordered-uses",
            source=source,
            intended_uses=(
                SurfaceIntendedUse.NAVIGATION,
                SurfaceIntendedUse.COLLISION,
            ),
        )


@pytest.mark.parametrize(
    ("request_value", "candidate_value", "materialization_value", "message"),
    [
        (cast(Any, "request"), None, None, "request"),
        (None, cast(Any, "candidate"), None, "candidate"),
        (None, None, cast(Any, "materialization"), "materialization"),
    ],
)
def test_result_rejects_wrong_member_types(
    request_value: Any,
    candidate_value: Any,
    materialization_value: Any,
    message: str,
) -> None:
    request, candidate, materialization = _valid_parts("wrong-result-types")

    with pytest.raises(TypeError, match=message):
        HybridSurfaceCandidateResult(
            request=request if request_value is None else request_value,
            candidate=candidate if candidate_value is None else candidate_value,
            materialization=(
                materialization if materialization_value is None else materialization_value
            ),
        )


def test_result_preserves_source_geometry_frame_scale_representation_and_intent_exactly() -> None:
    uses = (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.NAVIGATION,
    )
    request, candidate, materialization = _valid_parts(
        "preserve",
        scale=GeometryScaleStatus.METRIC,
        representation="surfels",
        intended_uses=uses,
    )
    result = HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=materialization,
    )

    assert result.candidate.source_geometry is request.source_geometry
    assert (
        result.candidate.local_frame_id == request.source_geometry.geometry_solution.local_frame_id
    )
    assert result.candidate.scale_status is GeometryScaleStatus.METRIC
    assert result.candidate.representation is request.output_representation
    assert result.candidate.intended_uses is request.intended_uses

    other_source = _candidate("other-source", scale=GeometryScaleStatus.METRIC)
    foreign = _surface(
        "foreign",
        source=other_source,
        source_artifacts=other_source.source_artifacts,
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    with pytest.raises(ValueError, match="exact source"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=foreign,
            materialization=_materialization(foreign),
        )

    wrong_representation = _surface(
        "wrong-representation",
        source=request.source_geometry,
        source_artifacts=_canonical_artifacts(
            *request.source_geometry.source_artifacts,
            *request.supporting_artifacts,
        ),
        representation=SurfaceRepresentationName("sdf"),
        intended_uses=request.intended_uses,
    )
    with pytest.raises(ValueError, match="SurfaceRepresentationName"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=wrong_representation,
            materialization=_materialization(wrong_representation),
        )

    wrong_uses = _surface(
        "wrong-uses",
        source=request.source_geometry,
        source_artifacts=_canonical_artifacts(
            *request.source_geometry.source_artifacts,
            *request.supporting_artifacts,
        ),
        representation=request.output_representation,
        intended_uses=(SurfaceIntendedUse.MEASUREMENT,),
    )
    with pytest.raises(ValueError, match="intended uses"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=wrong_uses,
            materialization=_materialization(wrong_uses),
        )


def test_output_artifact_ref_must_be_distinct_from_all_inputs() -> None:
    source = _candidate("distinct")
    support = ArtifactRef(
        artifact_id=ArtifactId("surface:reused-support"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )
    request = _request("distinct", source=source, supporting=(support,))

    candidate = _surface(
        "ignored",
        source=source,
        artifact_ref=support,
        source_artifacts=source.source_artifacts,
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )

    with pytest.raises(ValueError, match="distinct from all inputs"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=candidate,
            materialization=_materialization(candidate),
        )

    source_surface_ref = ArtifactRef(
        artifact_id=ArtifactId("surface:source-collision"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )
    colliding_source = _candidate(
        "source-collision",
        source_artifacts=(source_surface_ref,),
    )
    with pytest.raises(ValueError, match="must not reuse"):
        _surface(
            "source-collision",
            source=colliding_source,
            artifact_ref=source_surface_ref,
            source_artifacts=colliding_source.source_artifacts,
        )


def test_candidate_ancestry_must_equal_exact_canonical_union() -> None:
    ancestor_a = _artifact("artifact:a", "evidence.alpha")
    ancestor_b = _artifact("artifact:b", "geometry.solution")
    support_c = _artifact("artifact:c", "evidence.depth")
    support_d = _artifact("artifact:d", "evidence.mask")
    source = _candidate(
        "lineage",
        source_artifacts=(ancestor_a, ancestor_b),
    )
    request = _request(
        "lineage",
        source=source,
        supporting=(support_c, support_d),
    )
    exact = (ancestor_a, ancestor_b, support_c, support_d)

    candidate = _surface(
        "lineage",
        source=source,
        source_artifacts=exact,
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    result = HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=_materialization(candidate),
    )
    assert result.candidate.source_artifacts is exact

    missing_support = _surface(
        "missing-support",
        source=source,
        source_artifacts=(ancestor_a, ancestor_b, support_c),
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    with pytest.raises(ValueError, match="canonical union"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=missing_support,
            materialization=_materialization(missing_support),
        )

    extra = _artifact("artifact:e", "evidence.extra")
    extra_candidate = _surface(
        "extra",
        source=source,
        source_artifacts=(ancestor_a, ancestor_b, support_c, support_d, extra),
        representation=request.output_representation,
        intended_uses=request.intended_uses,
    )
    with pytest.raises(ValueError, match="canonical union"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=extra_candidate,
            materialization=_materialization(extra_candidate),
        )

    with pytest.raises(ValueError, match="unique"):
        _surface(
            "duplicate-lineage",
            source=source,
            source_artifacts=(
                ancestor_a,
                ancestor_b,
                support_c,
                support_c,
            ),
        )
    with pytest.raises(ValueError, match="canonical"):
        _surface(
            "unordered-lineage",
            source=source,
            source_artifacts=(ancestor_b, ancestor_a, support_c),
        )


def test_materialization_must_bind_exact_candidate_artifact() -> None:
    request, candidate, _ = _valid_parts("materialization")
    wrong_ref = ArtifactRef(
        artifact_id=ArtifactId("surface:other"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )

    with pytest.raises(ValueError, match="exactly match"):
        HybridSurfaceCandidateResult(
            request=request,
            candidate=candidate,
            materialization=_materialization(candidate, artifact_ref=wrong_ref),
        )


def test_producer_is_adapter_owned_and_boundary_invents_no_model_or_checkpoint() -> None:
    request, _, _ = _valid_parts("producer")
    producer = _producer("caller-adapter")
    candidate = _surface(
        "producer",
        source=request.source_geometry,
        source_artifacts=_canonical_artifacts(
            *request.source_geometry.source_artifacts,
            *request.supporting_artifacts,
        ),
        representation=request.output_representation,
        intended_uses=request.intended_uses,
        producer=producer,
    )
    result = build_hybrid_surface_candidate_result(
        request,
        candidate,
        _materialization(candidate),
    )

    assert result.candidate.producer is producer
    assert result.candidate.producer.model is None
    assert result.candidate.producer.checkpoint is None


@pytest.mark.parametrize(
    "scale",
    [GeometryScaleStatus.UNRESOLVED, GeometryScaleStatus.METRIC],
)
def test_scale_status_is_preserved_without_inferred_accuracy(
    scale: GeometryScaleStatus,
) -> None:
    request, candidate, materialization = _valid_parts(
        f"scale-{scale.value}",
        scale=scale,
        intended_uses=(SurfaceIntendedUse.MEASUREMENT,),
    )
    result = HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=materialization,
    )

    assert result.candidate.scale_status is scale
    forbidden = {
        "accuracy",
        "tolerance",
        "measurement_accuracy",
        "measurement_ready",
        "measurement_suitable",
        "quality_decision",
        "decision",
    }
    assert forbidden.isdisjoint(field.name for field in fields(HybridSurfaceCandidateResult))
    assert forbidden.isdisjoint(field.name for field in fields(HybridSurfaceCandidateRequest))


def test_build_helper_preserves_exact_objects_and_does_not_mutate_inputs() -> None:
    request, candidate, materialization = _valid_parts("immutability")
    source_artifacts_before = request.source_geometry.source_artifacts
    supporting_before = request.supporting_artifacts
    intended_uses_before = request.intended_uses
    candidate_ancestry_before = candidate.source_artifacts
    entries_before = materialization.entries

    result = build_hybrid_surface_candidate_result(
        request,
        candidate,
        materialization,
    )

    assert result.request is request
    assert result.candidate is candidate
    assert result.materialization is materialization
    assert request.source_geometry.source_artifacts is source_artifacts_before
    assert request.supporting_artifacts is supporting_before
    assert request.intended_uses is intended_uses_before
    assert candidate.source_artifacts is candidate_ancestry_before
    assert materialization.entries is entries_before
    assert request.source_geometry.point_maps[0].positions_xyz == ((1.0, 2.0, 3.0),)


def test_boundary_exposes_no_appearance_selection_repair_fill_or_runtime_semantics() -> None:
    request_fields = {field.name for field in fields(HybridSurfaceCandidateRequest)}
    result_fields = {field.name for field in fields(HybridSurfaceCandidateResult)}
    candidate_fields = {field.name for field in fields(SurfaceModel)}

    forbidden_fields = {
        "appearance",
        "appearance_model",
        "radiance",
        "gaussians",
        "splats",
        "texture",
        "material",
        "render_quality",
        "rank",
        "score",
        "quality_decision",
        "winner",
        "selected",
        "merge",
        "replace",
        "repair",
        "watertight",
        "safe",
        "accuracy",
        "generated_completion",
        "unsupported_region",
        "holes",
        "default",
        "runtime",
        "benchmark",
    }

    assert forbidden_fields.isdisjoint(request_fields)
    assert forbidden_fields.isdisjoint(result_fields)
    assert forbidden_fields.isdisjoint(candidate_fields)

    for name in (
        "select_candidate",
        "rank_candidates",
        "repair_mesh",
        "fill_holes",
        "merge_surfaces",
        "replace_surface",
        "evaluate_quality",
        "benchmark",
        "create_runtime_asset",
    ):
        assert not hasattr(hybrid_module, name)


def test_module_surface_is_pure_and_contains_no_concrete_hybrid_technology() -> None:
    forbidden_names = {
        "numpy",
        "torch",
        "open3d",
        "trimesh",
        "pymeshlab",
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "AppearanceModel",
        "QualityDecision",
        "Open3dTsdfSurfaceAdapter",
        "SurfaceMeshInspection",
        "UnsupportedRegion",
        "RuntimeScene",
        "MasterScene",
        "TwoDGS",
        "OMeGa",
        "SurfaceSplat",
        "MeshSplatting",
    }
    assert forbidden_names.isdisjoint(vars(hybrid_module))

    module_source = inspect.getsource(hybrid_module).lower()
    for token in (
        "2dgs",
        "omega",
        "surfacesplat",
        "meshsplatting",
        "open3d",
        "trimesh",
        "pymeshlab",
        "numpy",
        "torch",
        "filesystem",
        "network",
        "subprocess",
    ):
        assert token not in module_source
