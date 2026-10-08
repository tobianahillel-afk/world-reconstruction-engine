from __future__ import annotations

import ast
import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import wre.domain.appearance as appearance_module
from wre.domain import APPEARANCE_MODEL_ARTIFACT_KIND, AppearanceModel, AppearanceRepresentationName
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


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(implementation=f"test.appearance.{token}", version="1"),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "evidence.source") -> ArtifactRef:
    return ArtifactRef(ArtifactId(identifier), ArtifactKind(kind))


def _geometry(
    token: str,
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    ancestry: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    camera = CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=2, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )
    point_map = PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=(camera.observation_id,),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=MetricVector(observations=()),
    )
    geometry_solution = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=MetricVector(observations=()),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry_solution,
        camera_solutions=(camera,),
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=(ancestry if ancestry is not None else (_artifact(f"artifact:{token}"),)),
    )


def _surface(
    token: str,
    geometry: GeometrySolutionCandidate,
    *,
    artifact_ref: ArtifactRef | None = None,
    ancestry: tuple[ArtifactRef, ...] | None = None,
) -> SurfaceModel:
    return SurfaceModel(
        artifact_ref=artifact_ref
        if artifact_ref is not None
        else ArtifactRef(ArtifactId(f"surface:{token}"), SURFACE_MODEL_ARTIFACT_KIND),
        source_geometry=geometry,
        representation=SurfaceRepresentationName("mesh"),
        intended_uses=(SurfaceIntendedUse.COLLISION,),
        local_frame_id=geometry.geometry_solution.local_frame_id,
        scale_status=geometry.geometry_solution.scale_status,
        producer=_producer(f"surface:{token}"),
        source_artifacts=ancestry if ancestry is not None else geometry.source_artifacts,
    )


def _model(
    token: str,
    *,
    source_geometry: GeometrySolutionCandidate | None = None,
    source_surface: SurfaceModel | None = None,
    source_observation_ids: tuple[ObservationId, ...] | None = None,
    representation: AppearanceRepresentationName | None = None,
    local_frame_id: LocalFrameId | None = None,
    scale_status: GeometryScaleStatus | None = None,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
    artifact_ref: ArtifactRef | None = None,
) -> AppearanceModel:
    geometry = source_geometry if source_geometry is not None else _geometry(token)
    required = set(geometry.source_artifacts)
    if source_surface is not None:
        required.update(source_surface.source_artifacts)
        required.add(source_surface.artifact_ref)
    default_ancestry = tuple(
        sorted(required, key=lambda item: (item.artifact_id.value, item.artifact_kind.value))
    )
    return AppearanceModel(
        artifact_ref=artifact_ref
        if artifact_ref is not None
        else ArtifactRef(ArtifactId(f"appearance:{token}"), APPEARANCE_MODEL_ARTIFACT_KIND),
        source_geometry=geometry,
        source_surface=source_surface,
        source_observation_ids=source_observation_ids
        if source_observation_ids is not None
        else (ObservationId(f"obs:{token}"),),
        representation=representation
        if representation is not None
        else AppearanceRepresentationName("static.visual"),
        local_frame_id=local_frame_id
        if local_frame_id is not None
        else geometry.geometry_solution.local_frame_id,
        scale_status=(
            scale_status if scale_status is not None else geometry.geometry_solution.scale_status
        ),
        producer=_producer(token),
        source_artifacts=source_artifacts if source_artifacts is not None else default_ancestry,
    )


def test_canonical_appearance_envelope_exact_frozen_shape_and_kind() -> None:
    source = _geometry("shape")
    model = _model("shape", source_geometry=source)
    assert APPEARANCE_MODEL_ARTIFACT_KIND == ArtifactKind("appearance.static")
    assert tuple(field.name for field in fields(AppearanceModel)) == (
        "artifact_ref",
        "source_geometry",
        "source_surface",
        "source_observation_ids",
        "representation",
        "local_frame_id",
        "scale_status",
        "producer",
        "source_artifacts",
    )
    assert model.source_geometry is source
    assert model.source_surface is None
    assert model.source_artifacts == source.source_artifacts
    with pytest.raises(FrozenInstanceError):
        model.representation = AppearanceRepresentationName("changed")  # type: ignore[misc]


@pytest.mark.parametrize(
    "token",
    ("gaussian", "radiance", "neural.v2", "future:representation", "textures_2d"),
)
def test_representation_token_is_open_without_solver_enum(token: str) -> None:
    assert str(AppearanceRepresentationName(token)) == token


@pytest.mark.parametrize(
    "token",
    ("", "Gaussian", "visual space", " visual", "visual ", "-visual", "visual/mesh", "x" * 129),
)
def test_representation_token_rejects_invalid_values(token: str) -> None:
    with pytest.raises(ValueError, match="lowercase token"):
        AppearanceRepresentationName(token)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("artifact_ref", cast(Any, "appearance"), "artifact_ref"),
        ("source_geometry", cast(Any, "geometry"), "source_geometry"),
        ("source_surface", cast(Any, "surface"), "source_surface"),
        ("source_observation_ids", cast(Any, []), "immutable tuple"),
        ("representation", cast(Any, "radiance"), "representation"),
        ("local_frame_id", cast(Any, "frame"), "local_frame_id"),
        ("scale_status", cast(Any, "metric"), "scale_status"),
        ("producer", cast(Any, "producer"), "producer"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
    ],
)
def test_rejects_wrong_field_types(field_name: str, replacement: Any, message: str) -> None:
    source = _geometry("types")
    model = _model("types", source_geometry=source)
    kwargs = {field.name: getattr(model, field.name) for field in fields(model)}
    kwargs[field_name] = replacement
    with pytest.raises((TypeError, ValueError), match=message):
        AppearanceModel(**kwargs)


def test_artifact_kind_is_exact_and_never_surface_kind() -> None:
    with pytest.raises(ValueError, match=r"appearance\.static"):
        _model("wrong-kind", artifact_ref=_artifact("appearance:wrong", "geometry.surface"))


def test_observations_nonempty_canonical_unique_no_mutation() -> None:
    ids = (ObservationId("obs:a"), ObservationId("obs:z"))
    model = _model("observations", source_observation_ids=ids)
    assert model.source_observation_ids is ids

    with pytest.raises(ValueError, match="non-empty"):
        _model("empty-observations", source_observation_ids=())
    with pytest.raises(TypeError, match="ObservationId"):
        _model("wrong-observations", source_observation_ids=cast(Any, ("obs:a",)))
    with pytest.raises(ValueError, match="unique"):
        _model("duplicate-observations", source_observation_ids=(ids[0], ids[0]))
    with pytest.raises(ValueError, match="canonical"):
        _model("reverse-observations", source_observation_ids=tuple(reversed(ids)))


@pytest.mark.parametrize("scale", [GeometryScaleStatus.UNRESOLVED, GeometryScaleStatus.METRIC])
def test_local_frame_and_scale_preserve_source_geometry_exactly(
    scale: GeometryScaleStatus,
) -> None:
    geometry = _geometry(f"scale:{scale.value}", scale=scale)
    model = _model(f"scale:{scale.value}", source_geometry=geometry)
    assert model.local_frame_id is geometry.geometry_solution.local_frame_id
    assert model.scale_status is scale

    with pytest.raises(ValueError, match="LocalFrameId"):
        _model(
            "foreign-frame", source_geometry=geometry, local_frame_id=LocalFrameId("frame:alien")
        )
    other_scale = (
        GeometryScaleStatus.METRIC
        if scale is GeometryScaleStatus.UNRESOLVED
        else GeometryScaleStatus.UNRESOLVED
    )
    with pytest.raises(ValueError, match="scale status"):
        _model("foreign-scale", source_geometry=geometry, scale_status=other_scale)


def test_optional_surface_requires_same_complete_geometry_and_exact_frame_scale() -> None:
    geometry = _geometry("with-surface")
    surface = _surface("with-surface", geometry)
    model = _model("with-surface", source_geometry=geometry, source_surface=surface)
    assert model.source_surface is surface
    assert surface.artifact_ref in model.source_artifacts

    absent = _model("without-surface", source_geometry=geometry)
    assert absent.source_surface is None

    other = _geometry("foreign")
    with pytest.raises(ValueError, match="exact source geometry"):
        _model(
            "foreign-surface", source_geometry=geometry, source_surface=_surface("foreign", other)
        )

    # Matching frame and scale alone do not establish identical geometry ancestry.
    alternative_solution = replace(
        geometry.geometry_solution,
        geometry_solution_id=GeometrySolutionId("geometry:distinct"),
    )
    alternate = replace(geometry, geometry_solution=alternative_solution)
    with pytest.raises(ValueError, match="exact source geometry"):
        _model(
            "foreign-same-frame",
            source_geometry=geometry,
            source_surface=_surface("other", alternate),
        )


def test_source_artifact_ancestry_is_complete_unique_canonical_and_non_aliasing() -> None:
    source_a = _artifact("artifact:a", "evidence.source")
    source_b = _artifact("artifact:b", "geometry.points")
    geometry = _geometry("lineage", ancestry=(source_a, source_b))
    surface_art = ArtifactRef(ArtifactId("artifact:surface"), SURFACE_MODEL_ARTIFACT_KIND)
    surface_extra = _artifact("artifact:c", "evidence.surface")
    surface = _surface(
        "lineage",
        geometry,
        artifact_ref=surface_art,
        ancestry=(source_a, source_b, surface_extra),
    )
    all_sources = (source_a, source_b, surface_extra, surface_art)
    model = _model(
        "lineage", source_geometry=geometry, source_surface=surface, source_artifacts=all_sources
    )
    assert model.source_artifacts is all_sources

    with pytest.raises(ValueError, match="non-empty"):
        _model("empty-lineage", source_geometry=geometry, source_artifacts=())
    with pytest.raises(TypeError, match="ArtifactRef"):
        _model("wrong-lineage", source_geometry=geometry, source_artifacts=cast(Any, ("x",)))
    with pytest.raises(ValueError, match="unique"):
        _model("duplicate-lineage", source_geometry=geometry, source_artifacts=(source_a, source_a))
    with pytest.raises(ValueError, match="canonical"):
        _model("unsorted-lineage", source_geometry=geometry, source_artifacts=(source_b, source_a))
    with pytest.raises(ValueError, match="ancestry"):
        _model("missing-geometry", source_geometry=geometry, source_artifacts=(source_a,))
    with pytest.raises(ValueError, match="ancestry"):
        _model(
            "missing-surface",
            source_geometry=geometry,
            source_surface=surface,
            source_artifacts=(source_a, source_b, surface_extra),
        )
    with pytest.raises(ValueError, match="ancestry"):
        _model(
            "missing-surface-parent",
            source_geometry=geometry,
            source_surface=surface,
            source_artifacts=(source_a, source_b, surface_art),
        )
    with pytest.raises(ValueError, match="not reuse"):
        _model(
            "alias",
            source_geometry=geometry,
            source_artifacts=(source_a, source_b),
            artifact_ref=source_a,
        )


def test_metadata_only_surface_and_geometry_immutability_and_separation() -> None:
    geometry = _geometry("immutability")
    surface = _surface("immutability", geometry)
    observations = (ObservationId("obs:a"), ObservationId("obs:b"))
    source_artifacts = tuple(
        sorted(
            (*geometry.source_artifacts, surface.artifact_ref),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    model = _model(
        "immutability",
        source_geometry=geometry,
        source_surface=surface,
        source_observation_ids=observations,
        source_artifacts=source_artifacts,
    )
    assert model.source_geometry is geometry
    assert model.source_surface is surface
    assert model.source_observation_ids is observations
    assert model.source_artifacts is source_artifacts
    assert model.producer is not geometry.producer
    with pytest.raises(FrozenInstanceError):
        geometry.geometry_solution.scale_status = GeometryScaleStatus.METRIC  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        surface.source_artifacts = ()  # type: ignore[misc]


def test_module_has_no_solver_renderer_payload_runtime_or_generated_content() -> None:
    source_path = Path(appearance_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    assert imports <= {"__future__", "re", "dataclasses", "typing", "wre"}

    field_names = {field.name for field in fields(AppearanceModel)}
    for forbidden_field in (
        "positions",
        "covariance",
        "spherical_harmonics",
        "weights",
        "pixels",
        "material",
        "environment",
        "renderer",
        "training",
        "routing",
        "quality",
        "completion",
        "runtime",
        "geometry_payload",
        "collision",
    ):
        assert forbidden_field not in field_names

    assert "numpy" not in source.lower()
    assert "torch" not in source.lower()
    assert "cuda" not in source.lower()
    assert "gsplat" not in source.lower()
    assert "nerfstudio" not in source.lower()
    assert "opensplat" not in source.lower()
    assert "subprocess" not in source.lower()
    assert "checkpoint" not in source.lower()
    assert "QualityDecision" not in source
    assert "MaterialModel" not in source
    assert "EnvironmentModel" not in source
    assert "MasterScene" not in source
    assert "RuntimeScene" not in source
    docstring = AppearanceModel.__doc__
    assert docstring is not None
    assert "generated-completion truth" in docstring
