from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.surfaces as surface_module
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


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.surface.{token}",
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
            else (_artifact(f"artifact:{token}"),)
        ),
    )


def _surface(
    token: str,
    *,
    source_geometry: GeometrySolutionCandidate,
    representation: SurfaceRepresentationName | None = None,
    intended_uses: tuple[SurfaceIntendedUse, ...] = (SurfaceIntendedUse.COLLISION,),
    local_frame_id: LocalFrameId | None = None,
    scale_status: GeometryScaleStatus | None = None,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
    artifact_ref: ArtifactRef | None = None,
) -> SurfaceModel:
    return SurfaceModel(
        artifact_ref=artifact_ref
        or ArtifactRef(
            artifact_id=ArtifactId(f"surface:{token}"),
            artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
        ),
        source_geometry=source_geometry,
        representation=representation or SurfaceRepresentationName("mesh"),
        intended_uses=intended_uses,
        local_frame_id=local_frame_id or source_geometry.geometry_solution.local_frame_id,
        scale_status=scale_status or source_geometry.geometry_solution.scale_status,
        producer=_producer(token),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else source_geometry.source_artifacts
        ),
    )


def test_surface_model_has_exact_frozen_shape_kind_and_intended_use_values() -> None:
    source = _candidate("shape")
    model = _surface(
        "shape",
        source_geometry=source,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )

    assert SURFACE_MODEL_ARTIFACT_KIND == ArtifactKind("geometry.surface")
    assert tuple(item.value for item in SurfaceIntendedUse) == (
        "collision",
        "measurement",
        "navigation",
    )
    assert tuple(field.name for field in fields(SurfaceModel)) == (
        "artifact_ref",
        "source_geometry",
        "representation",
        "intended_uses",
        "local_frame_id",
        "scale_status",
        "producer",
        "source_artifacts",
    )
    assert model.source_geometry is source

    with pytest.raises(FrozenInstanceError):
        model.intended_uses = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("artifact_ref", cast(Any, "artifact"), "artifact_ref"),
        ("source_geometry", cast(Any, "geometry"), "source_geometry"),
        ("representation", cast(Any, "mesh"), "representation"),
        ("intended_uses", cast(Any, []), "immutable tuple"),
        ("local_frame_id", cast(Any, "frame"), "local_frame_id"),
        ("scale_status", cast(Any, "metric"), "scale_status"),
        ("producer", cast(Any, "producer"), "producer"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
    ],
)
def test_surface_model_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    source = _candidate("types")
    kwargs: dict[str, Any] = {
        "artifact_ref": ArtifactRef(ArtifactId("surface:types"), SURFACE_MODEL_ARTIFACT_KIND),
        "source_geometry": source,
        "representation": SurfaceRepresentationName("mesh"),
        "intended_uses": (SurfaceIntendedUse.COLLISION,),
        "local_frame_id": source.geometry_solution.local_frame_id,
        "scale_status": source.geometry_solution.scale_status,
        "producer": _producer("types"),
        "source_artifacts": source.source_artifacts,
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        SurfaceModel(**kwargs)


def test_surface_model_rejects_wrong_artifact_kind() -> None:
    source = _candidate("kind")

    with pytest.raises(ValueError, match=r"geometry\.surface"):
        _surface(
            "kind",
            source_geometry=source,
            artifact_ref=_artifact("surface:kind", "geometry.dense_depth"),
        )


@pytest.mark.parametrize("value", ["mesh", "sdf", "surfels", "mesh.triangles", "tsdf:v1"])
def test_surface_representation_name_remains_open_for_valid_lower_tokens(value: str) -> None:
    assert SurfaceRepresentationName(value).value == value


@pytest.mark.parametrize(
    "value",
    ["", "Mesh", "tri mesh", " mesh", "mesh ", "-mesh", "mesh/triangles"],
)
def test_surface_representation_name_rejects_invalid_tokens(value: str) -> None:
    with pytest.raises(ValueError, match="lowercase token"):
        SurfaceRepresentationName(value)


def test_surface_intended_uses_are_non_empty_unique_immutable_and_canonical() -> None:
    source = _candidate("uses")
    uses = (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.MEASUREMENT,
        SurfaceIntendedUse.NAVIGATION,
    )
    model = _surface("uses", source_geometry=source, intended_uses=uses)
    assert model.intended_uses is uses

    with pytest.raises(ValueError, match="non-empty"):
        _surface("empty", source_geometry=source, intended_uses=())
    with pytest.raises(TypeError, match="SurfaceIntendedUse"):
        _surface(
            "wrong",
            source_geometry=source,
            intended_uses=cast(Any, ("collision",)),
        )
    with pytest.raises(ValueError, match="unique"):
        _surface(
            "duplicate",
            source_geometry=source,
            intended_uses=(
                SurfaceIntendedUse.COLLISION,
                SurfaceIntendedUse.COLLISION,
            ),
        )
    with pytest.raises(ValueError, match="canonical"):
        _surface(
            "unordered",
            source_geometry=source,
            intended_uses=(
                SurfaceIntendedUse.NAVIGATION,
                SurfaceIntendedUse.COLLISION,
            ),
        )


def test_surface_model_preserves_exact_local_frame_and_scale_status() -> None:
    unresolved = _candidate("unresolved", scale=GeometryScaleStatus.UNRESOLVED)
    model = _surface("unresolved", source_geometry=unresolved)

    assert model.local_frame_id == unresolved.geometry_solution.local_frame_id
    assert model.scale_status is GeometryScaleStatus.UNRESOLVED

    with pytest.raises(ValueError, match="LocalFrameId"):
        _surface(
            "wrong-frame",
            source_geometry=unresolved,
            local_frame_id=LocalFrameId("frame:other"),
        )
    with pytest.raises(ValueError, match="scale status"):
        _surface(
            "wrong-scale",
            source_geometry=unresolved,
            scale_status=GeometryScaleStatus.METRIC,
        )

    metric = _candidate("metric", scale=GeometryScaleStatus.METRIC)
    metric_model = _surface("metric", source_geometry=metric)
    assert metric_model.scale_status is GeometryScaleStatus.METRIC


def test_source_artifacts_are_non_empty_unique_canonical_and_retain_source_ancestry() -> None:
    ancestor_a = _artifact("artifact:a", "evidence.images")
    ancestor_b = _artifact("artifact:b", "geometry.solution")
    extra = _artifact("artifact:c", "geometry.dense_depth")
    source = _candidate(
        "ancestry",
        source_artifacts=(ancestor_a, ancestor_b),
    )

    evidence = (ancestor_a, ancestor_b, extra)
    model = _surface(
        "ancestry",
        source_geometry=source,
        source_artifacts=evidence,
    )
    assert model.source_artifacts is evidence

    with pytest.raises(ValueError, match="non-empty"):
        _surface("empty", source_geometry=source, source_artifacts=())
    with pytest.raises(TypeError, match="ArtifactRef"):
        _surface(
            "wrong-member",
            source_geometry=source,
            source_artifacts=cast(Any, ("artifact",)),
        )
    with pytest.raises(ValueError, match="unique"):
        _surface(
            "duplicate",
            source_geometry=source,
            source_artifacts=(ancestor_a, ancestor_b, ancestor_b),
        )
    with pytest.raises(ValueError, match="canonical"):
        _surface(
            "unordered",
            source_geometry=source,
            source_artifacts=(ancestor_b, ancestor_a, extra),
        )
    with pytest.raises(ValueError, match="retain every source"):
        _surface(
            "missing",
            source_geometry=source,
            source_artifacts=(ancestor_a, extra),
        )


def test_surface_artifact_identity_cannot_reuse_source_artifact_identity() -> None:
    surface_ref = ArtifactRef(
        artifact_id=ArtifactId("surface:collision"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )
    source = _candidate("collision", source_artifacts=(surface_ref,))

    with pytest.raises(ValueError, match="must not reuse"):
        _surface(
            "collision",
            source_geometry=source,
            artifact_ref=surface_ref,
            source_artifacts=(surface_ref,),
        )


def test_empty_source_geometry_ancestry_still_requires_explicit_surface_evidence() -> None:
    source = _candidate("empty-source", source_artifacts=())
    evidence = (_artifact("artifact:dense", "geometry.dense_depth"),)

    model = _surface(
        "evidence",
        source_geometry=source,
        source_artifacts=evidence,
    )
    assert model.source_artifacts is evidence

    with pytest.raises(ValueError, match="non-empty"):
        _surface("missing-evidence", source_geometry=source, source_artifacts=())


def test_surface_model_preserves_source_payload_and_caller_tuples_without_mutation() -> None:
    ancestor = _artifact("artifact:source", "geometry.solution")
    extra = _artifact("artifact:depth", "geometry.dense_depth")
    source = _candidate("immutable", source_artifacts=(ancestor,))
    uses = (SurfaceIntendedUse.COLLISION, SurfaceIntendedUse.NAVIGATION)
    evidence = (extra, ancestor)
    canonical_evidence = tuple(
        sorted(evidence, key=lambda item: (item.artifact_id.value, item.artifact_kind.value))
    )

    model = _surface(
        "immutable",
        source_geometry=source,
        intended_uses=uses,
        source_artifacts=canonical_evidence,
    )

    assert model.source_geometry is source
    assert model.intended_uses is uses
    assert model.source_artifacts is canonical_evidence
    assert model.source_geometry.point_maps[0].positions_xyz == ((1.0, 2.0, 3.0),)
    assert source.source_artifacts == (ancestor,)


def test_intended_use_is_not_suitability_or_runtime_asset_evidence() -> None:
    source = _candidate("intent")
    model = _surface(
        "intent",
        source_geometry=source,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )

    for attribute in (
        "collision_suitable",
        "measurement_suitable",
        "navigation_suitable",
        "suitability",
        "suitability_score",
        "threshold",
        "decision",
        "quality_decision",
        "watertight",
        "collision_mesh",
        "navmesh",
        "runtime_lod",
        "generated_regions",
        "hole_fill",
        "appearance",
    ):
        assert not hasattr(model, attribute)


def test_surface_module_exposes_no_generation_solver_runtime_or_v2l16_2_behavior() -> None:
    forbidden = {
        "open3d",
        "pymeshlab",
        "numpy",
        "torch",
        "pycolmap",
        "TSDF",
        "SDF",
        "Poisson",
        "Meshing",
        "DepthFusion",
        "DenseDepthArtifact",
        "SurfaceAdapter",
        "CollisionMesh",
        "NavMesh",
        "QualityDecision",
        "MasterScene",
        "RuntimeScene",
    }

    assert forbidden.isdisjoint(vars(surface_module))
    assert not hasattr(SurfaceModel, "run")
    assert not hasattr(SurfaceModel, "generate")
    assert not hasattr(SurfaceModel, "fuse")
    assert not hasattr(SurfaceModel, "evaluate_suitability")
