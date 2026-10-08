from __future__ import annotations

import ast
import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast, get_type_hints

import pytest

import wre.reconstruction.static_appearance_candidate as module
from wre.domain.appearance import (
    APPEARANCE_MODEL_ARTIFACT_KIND,
    AppearanceModel,
    AppearanceRepresentationName,
)
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
from wre.reconstruction import (
    StaticAppearanceCandidateAdapter,
    StaticAppearanceCandidateRequest,
    StaticAppearanceCandidateResult,
    build_static_appearance_candidate_result,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _digest(value: str) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(value.encode("utf-8")).hexdigest())


def _producer(value: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(implementation=f"test.appearance_adapter.{value}", version="1"),
        configuration=ConfigurationIdentity(sha256=_digest(value)),
    )


def _artifact(value: str, kind: str = "evidence.source") -> ArtifactRef:
    return ArtifactRef(artifact_id=ArtifactId(value), artifact_kind=ArtifactKind(kind))


def _ordered(*items: ArtifactRef) -> tuple[ArtifactRef, ...]:
    by_key = {(item.artifact_id.value, item.artifact_kind.value): item for item in items}
    return tuple(by_key[key] for key in sorted(by_key))


def _geometry(
    value: str, *, scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{value}")
    camera = CameraSolution(
        solution_id=CameraSolutionId(f"camera:{value}"),
        observation_id=ObservationId(f"obs:{value}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=2, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )
    points = PointMap(
        point_map_id=PointMapId(f"points:{value}"),
        local_frame_id=frame,
        source_observation_ids=(camera.observation_id,),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=MetricVector(observations=()),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{value}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=(),
        point_map_ids=(points.point_map_id,),
        metrics=MetricVector(observations=()),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera,),
        depth_fields=(),
        point_maps=(points,),
        producer=_producer(f"geometry:{value}"),
        source_artifacts=(_artifact(f"artifact:{value}"),),
    )


def _surface(geometry: GeometrySolutionCandidate, *, value: str = "one") -> SurfaceModel:
    return SurfaceModel(
        artifact_ref=ArtifactRef(ArtifactId(f"surface:{value}"), SURFACE_MODEL_ARTIFACT_KIND),
        source_geometry=geometry,
        representation=SurfaceRepresentationName("mesh"),
        intended_uses=(SurfaceIntendedUse.COLLISION,),
        local_frame_id=geometry.geometry_solution.local_frame_id,
        scale_status=geometry.geometry_solution.scale_status,
        producer=_producer(f"surface:{value}"),
        source_artifacts=_ordered(
            *geometry.source_artifacts, _artifact(f"surface:evidence:{value}")
        ),
    )


def _parts(
    value: str = "valid",
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    with_surface: bool = False,
    representation: str = "radiance.static",
) -> tuple[
    StaticAppearanceCandidateRequest, AppearanceModel, ArtifactMaterializationMetadata
]:
    geometry = _geometry(value, scale=scale)
    surface = _surface(geometry, value=value) if with_surface else None
    supporting = (_artifact(f"support:{value}", "evidence.photometry"),)
    observations = (ObservationId(f"obs:{value}"),)
    rep = AppearanceRepresentationName(representation)
    request = StaticAppearanceCandidateRequest(
        source_geometry=geometry,
        source_surface=surface,
        source_observation_ids=observations,
        output_representation=rep,
        supporting_artifacts=supporting,
    )
    ancestry = _ordered(
        *geometry.source_artifacts,
        *(surface.source_artifacts if surface is not None else ()),
        *((surface.artifact_ref,) if surface is not None else ()),
        *supporting,
    )
    artifact_ref = ArtifactRef(ArtifactId(f"appearance:{value}"), APPEARANCE_MODEL_ARTIFACT_KIND)
    candidate = AppearanceModel(
        artifact_ref=artifact_ref,
        source_geometry=geometry,
        source_surface=surface,
        source_observation_ids=observations,
        representation=rep,
        local_frame_id=geometry.geometry_solution.local_frame_id,
        scale_status=geometry.geometry_solution.scale_status,
        producer=_producer(f"appearance:{value}"),
        source_artifacts=ancestry,
    )
    manifest = ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path="appearance/part.bin", sha256=_digest(value), byte_length=19
            ),
        ),
    )
    return request, candidate, manifest


def test_shapes_exports_frozen_slots_and_protocol() -> None:
    request, candidate, materialization = _parts()
    result = build_static_appearance_candidate_result(request, candidate, materialization)
    assert tuple(field.name for field in fields(StaticAppearanceCandidateRequest)) == (
        "source_geometry",
        "source_surface",
        "source_observation_ids",
        "output_representation",
        "supporting_artifacts",
    )
    assert tuple(field.name for field in fields(StaticAppearanceCandidateResult)) == (
        "request",
        "candidate",
        "materialization",
    )
    signature = inspect.signature(StaticAppearanceCandidateAdapter.derive_candidate)
    assert tuple(signature.parameters) == ("self", "request")
    hints = get_type_hints(StaticAppearanceCandidateAdapter.derive_candidate)
    assert hints["request"] is StaticAppearanceCandidateRequest
    assert hints["return"] is StaticAppearanceCandidateResult
    assert result.request is request
    assert result.candidate is candidate
    assert result.materialization is materialization
    assert set(module.__all__) == {
        "StaticAppearanceCandidateAdapter",
        "StaticAppearanceCandidateRequest",
        "StaticAppearanceCandidateResult",
        "build_static_appearance_candidate_result",
    }
    with pytest.raises(FrozenInstanceError):
        request.source_surface = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidate = candidate  # type: ignore[misc]
    with pytest.raises(AttributeError):
        request.undeclared = True  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("field_name", "bad_value", "message"),
    [
        ("source_geometry", "geometry", "source_geometry"),
        ("source_surface", "surface", "source_surface"),
        ("source_observation_ids", [], "immutable tuple"),
        ("source_observation_ids", (), "non-empty"),
        ("source_observation_ids", ("obs:x",), "ObservationId"),
        ("output_representation", "gaussian", "AppearanceRepresentationName"),
        ("supporting_artifacts", [], "immutable tuple"),
        ("supporting_artifacts", ("ref",), "ArtifactRef"),
    ],
)
def test_request_rejects_invalid_types_and_containers(
    field_name: str, bad_value: Any, message: str
) -> None:
    request, _, _ = _parts()
    with pytest.raises((TypeError, ValueError), match=message):
        replace(request, **{field_name: bad_value})


def test_observation_membership_is_canonical_and_not_reordered() -> None:
    request, _, _ = _parts()
    observations = (ObservationId("obs:a"), ObservationId("obs:z"))
    good = replace(request, source_observation_ids=observations)
    assert good.source_observation_ids is observations
    with pytest.raises(ValueError, match="unique"):
        replace(request, source_observation_ids=(observations[0], observations[0]))
    with pytest.raises(ValueError, match="canonical"):
        replace(request, source_observation_ids=tuple(reversed(observations)))


def test_supporting_artifacts_can_be_empty_but_must_be_unique_canonical() -> None:
    request, _, _ = _parts()
    assert replace(request, supporting_artifacts=()).supporting_artifacts == ()
    a = _artifact("support:a", "input.a")
    z = _artifact("support:z", "input.z")
    values = (a, z)
    assert replace(request, supporting_artifacts=values).supporting_artifacts is values
    with pytest.raises(ValueError, match="unique"):
        replace(request, supporting_artifacts=(a, a))
    with pytest.raises(ValueError, match="canonical"):
        replace(request, supporting_artifacts=(z, a))


def test_optional_surface_must_retain_complete_geometry_even_when_ids_match() -> None:
    request, candidate, materialization = _parts(with_surface=True)
    result = build_static_appearance_candidate_result(request, candidate, materialization)
    assert result.candidate.source_surface is request.source_surface

    same_frame = replace(
        request.source_geometry,
        producer=_producer("different:producer"),
    )
    assert same_frame.geometry_solution == request.source_geometry.geometry_solution
    foreign_surface = _surface(same_frame)
    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        replace(request, source_surface=foreign_surface)

    other_frame = _surface(_geometry("other-frame"))
    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        replace(request, source_surface=other_frame)

    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        replace(
            request,
            source_surface=_surface(_geometry("other-scale", scale=GeometryScaleStatus.METRIC)),
        )


@pytest.mark.parametrize("scale", (GeometryScaleStatus.UNRESOLVED, GeometryScaleStatus.METRIC))
def test_scale_frame_and_representation_remain_open_and_unchanged(
    scale: GeometryScaleStatus,
) -> None:
    request, candidate, manifest = _parts(
        f"scale:{scale.value}", scale=scale, representation="future:appearance.v3"
    )
    result = build_static_appearance_candidate_result(request, candidate, manifest)
    assert result.candidate.scale_status is scale
    assert (
        result.candidate.local_frame_id
        is request.source_geometry.geometry_solution.local_frame_id
    )
    assert result.candidate.representation == AppearanceRepresentationName("future:appearance.v3")
    with pytest.raises(ValueError, match="LocalFrameId"):
        replace(candidate, local_frame_id=LocalFrameId("frame:elsewhere"))
    with pytest.raises(ValueError, match="scale status"):
        other = (
            GeometryScaleStatus.METRIC
            if scale is GeometryScaleStatus.UNRESOLVED
            else GeometryScaleStatus.UNRESOLVED
        )
        replace(candidate, scale_status=other)


def test_result_rejects_missing_extra_and_reordered_ancestry() -> None:
    request, candidate, manifest = _parts("ancestry", with_surface=True)
    assert request.source_surface is not None
    assert len(candidate.source_artifacts) == 4
    assert (
        build_static_appearance_candidate_result(request, candidate, manifest).candidate
        is candidate
    )

    for invalid in (
        tuple(item for item in candidate.source_artifacts if item not in request.supporting_artifacts),
        _ordered(*candidate.source_artifacts, _artifact("artifact:surprise")),
    ):
        # A canonical AppearanceModel permits extra ancestry; the adapter boundary does not.
        altered = replace(candidate, source_artifacts=invalid)
        with pytest.raises(ValueError, match="canonical union"):
            build_static_appearance_candidate_result(request, altered, manifest)

    with pytest.raises(ValueError, match="canonical"):
        replace(candidate, source_artifacts=tuple(reversed(candidate.source_artifacts)))
    with pytest.raises(ValueError, match="unique"):
        replace(
            candidate,
            source_artifacts=candidate.source_artifacts + candidate.source_artifacts[-1:],
        )

    same_id_different_kind = _artifact("appearance:ancestry", "model.other")
    request_with_support = replace(
        request,
        supporting_artifacts=_ordered(*request.supporting_artifacts, same_id_different_kind),
    )
    with pytest.raises(ValueError, match="canonical union"):
        build_static_appearance_candidate_result(request_with_support, candidate, manifest)

    # The canonical AppearanceModel rejects output/source aliasing before this boundary.
    with pytest.raises(ValueError, match="not reuse"):
        replace(
            candidate,
            source_artifacts=_ordered(*candidate.source_artifacts, candidate.artifact_ref),
        )


def test_result_rejects_candidate_request_materialization_mismatches() -> None:
    request, candidate, manifest = _parts("mismatch")
    other_req, other_candidate, _ = _parts("foreign")
    for bad in (
        replace(candidate, source_geometry=other_req.source_geometry,
                local_frame_id=other_req.source_geometry.geometry_solution.local_frame_id,
                scale_status=other_req.source_geometry.geometry_solution.scale_status,
                source_artifacts=other_req.source_geometry.source_artifacts),
        replace(candidate, source_observation_ids=(ObservationId("obs:other"),)),
        replace(candidate, representation=AppearanceRepresentationName("other")),
    ):
        with pytest.raises(ValueError):
            build_static_appearance_candidate_result(request, bad, manifest)

    foreign_optional_surface = _surface(request.source_geometry)
    complete_with_surface = replace(
        candidate,
        source_surface=foreign_optional_surface,
        source_artifacts=_ordered(
            *candidate.source_artifacts,
            *foreign_optional_surface.source_artifacts,
            foreign_optional_surface.artifact_ref,
        ),
    )
    with pytest.raises(ValueError, match="optional SurfaceModel"):
        build_static_appearance_candidate_result(
            request, complete_with_surface, manifest
        )
    with pytest.raises(ValueError, match="ArtifactRef"):
        build_static_appearance_candidate_result(
            request, candidate, replace(manifest, artifact_ref=other_candidate.artifact_ref)
        )

    with pytest.raises(TypeError, match="request"):
        StaticAppearanceCandidateResult(cast(Any, "request"), candidate, manifest)
    with pytest.raises(TypeError, match="candidate"):
        StaticAppearanceCandidateResult(request, cast(Any, "appearance"), manifest)
    with pytest.raises(TypeError, match="materialization"):
        StaticAppearanceCandidateResult(request, candidate, cast(Any, "manifest"))


def test_producer_payload_materialization_and_input_identity_are_not_modified() -> None:
    request, candidate, manifest = _parts("unchanged", with_surface=True)
    saved = (request.source_geometry, request.source_surface, request.source_observation_ids,
             request.supporting_artifacts, candidate.producer, candidate.source_artifacts,
             manifest.entries)
    result = build_static_appearance_candidate_result(request, candidate, manifest)
    assert result.candidate.producer is candidate.producer
    assert result.materialization is manifest
    assert result.candidate.artifact_ref.artifact_kind == APPEARANCE_MODEL_ARTIFACT_KIND
    assert (
        request.source_geometry, request.source_surface, request.source_observation_ids,
        request.supporting_artifacts, candidate.producer, candidate.source_artifacts,
        manifest.entries
    ) == saved
    assert all(
        isinstance(entry, ArtifactMaterializationEntry)
        for entry in result.materialization.entries
    )


def test_pure_module_has_no_runtime_solver_rendering_or_payload_io() -> None:
    source = Path(module.__file__).read_text(encoding="utf-8")
    imports: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.update(part.name.split(".")[0] for part in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "dataclasses", "typing", "wre"}
    for name in ("open(", "read_bytes(", "read_text(", "subprocess", "torch",
                 "numpy", "cuda", "gsplat", "nerfstudio", "opensplat",
                 "render(", "train(", "model_loader", "MasterScene", "RuntimeScene"):
        assert name not in source
    assert set(module.__all__) == {
        "StaticAppearanceCandidateAdapter",
        "StaticAppearanceCandidateRequest",
        "StaticAppearanceCandidateResult",
        "build_static_appearance_candidate_result",
    }
