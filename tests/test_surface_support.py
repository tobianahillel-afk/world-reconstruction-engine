from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.surface_support as support_module
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
from wre.domain.surface_support import (
    SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
    SurfaceRegionSupportStatus,
    SurfaceSupportMap,
    SurfaceSupportRegion,
)
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
            implementation=f"test.surface_support.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "evidence.region") -> ArtifactRef:
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
        source_artifacts=(_artifact(f"geometry-source:{token}", "geometry.solution"),),
    )


def _surface(
    token: str,
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    intended_uses: tuple[SurfaceIntendedUse, ...] = (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.MEASUREMENT,
        SurfaceIntendedUse.NAVIGATION,
    ),
) -> SurfaceModel:
    source = _candidate(token, scale=scale)
    return SurfaceModel(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(f"surface:{token}"),
            artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
        ),
        source_geometry=source,
        representation=SurfaceRepresentationName("mesh"),
        intended_uses=intended_uses,
        local_frame_id=source.geometry_solution.local_frame_id,
        scale_status=source.geometry_solution.scale_status,
        producer=_producer(f"surface:{token}"),
        source_artifacts=source.source_artifacts,
    )


def _region(
    token: str,
    *,
    status: SurfaceRegionSupportStatus = SurfaceRegionSupportStatus.UNSUPPORTED,
    kind: str = "evidence.region",
) -> SurfaceSupportRegion:
    return SurfaceSupportRegion(
        selector_artifact=_artifact(f"region:{token}", kind),
        status=status,
    )


def _support_map(
    token: str,
    *,
    surface: SurfaceModel | None = None,
    regions: tuple[SurfaceSupportRegion, ...] | None = None,
    artifact_ref: ArtifactRef | None = None,
    local_frame_id: LocalFrameId | None = None,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
    producer: ArtifactProducerIdentity | None = None,
) -> SurfaceSupportMap:
    retained_surface = surface or _surface(token)
    retained_regions = regions or (
        _region(f"{token}:a", status=SurfaceRegionSupportStatus.UNSUPPORTED),
        _region(f"{token}:b", status=SurfaceRegionSupportStatus.HOLE),
    )
    retained_ref = artifact_ref or ArtifactRef(
        artifact_id=ArtifactId(f"surface-support:{token}"),
        artifact_kind=SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
    )
    expected_sources = tuple(
        sorted(
            (
                retained_surface.artifact_ref,
                *(region.selector_artifact for region in retained_regions),
            ),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    return SurfaceSupportMap(
        artifact_ref=retained_ref,
        surface_model=retained_surface,
        local_frame_id=local_frame_id or retained_surface.local_frame_id,
        regions=retained_regions,
        producer=producer or _producer(f"support:{token}"),
        source_artifacts=source_artifacts if source_artifacts is not None else expected_sources,
    )


def test_frozen_shapes_artifact_kind_and_negative_statuses_are_exact() -> None:
    region = _region("shape")
    support = _support_map("shape")

    assert SURFACE_SUPPORT_MAP_ARTIFACT_KIND == ArtifactKind("geometry.surface_support")
    assert tuple(item.value for item in SurfaceRegionSupportStatus) == (
        "unsupported",
        "hole",
    )
    assert tuple(field.name for field in fields(SurfaceSupportRegion)) == (
        "selector_artifact",
        "status",
    )
    assert tuple(field.name for field in fields(SurfaceSupportMap)) == (
        "artifact_ref",
        "surface_model",
        "local_frame_id",
        "regions",
        "producer",
        "source_artifacts",
    )

    with pytest.raises(FrozenInstanceError):
        region.status = SurfaceRegionSupportStatus.HOLE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        support.regions = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("selector_artifact", cast(Any, "selector"), "selector_artifact"),
        ("status", cast(Any, "hole"), "status"),
    ],
)
def test_surface_support_region_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    kwargs: dict[str, Any] = {
        "selector_artifact": _artifact("region:type"),
        "status": SurfaceRegionSupportStatus.UNSUPPORTED,
    }
    kwargs[field_name] = replacement

    with pytest.raises(TypeError, match=message):
        SurfaceSupportRegion(**kwargs)


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("artifact_ref", cast(Any, "artifact"), "artifact_ref"),
        ("surface_model", cast(Any, "surface"), "surface_model"),
        ("local_frame_id", cast(Any, "frame"), "local_frame_id"),
        ("regions", cast(Any, []), "immutable tuple"),
        ("producer", cast(Any, "producer"), "producer"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
    ],
)
def test_surface_support_map_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    surface = _surface("types")
    regions = (_region("types"),)
    source_artifacts = tuple(
        sorted(
            (surface.artifact_ref, regions[0].selector_artifact),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    kwargs: dict[str, Any] = {
        "artifact_ref": ArtifactRef(
            ArtifactId("surface-support:types"),
            SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
        ),
        "surface_model": surface,
        "local_frame_id": surface.local_frame_id,
        "regions": regions,
        "producer": _producer("support:types"),
        "source_artifacts": source_artifacts,
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        SurfaceSupportMap(**kwargs)


def test_support_map_requires_exact_artifact_kind_and_local_frame() -> None:
    surface = _surface("kind-frame")
    region = (_region("kind-frame"),)

    with pytest.raises(ValueError, match=r"geometry\.surface_support"):
        _support_map(
            "wrong-kind",
            surface=surface,
            regions=region,
            artifact_ref=_artifact("support:wrong-kind", "geometry.surface"),
        )

    with pytest.raises(ValueError, match="LocalFrameId"):
        _support_map(
            "wrong-frame",
            surface=surface,
            regions=region,
            local_frame_id=LocalFrameId("frame:other"),
        )


def test_regions_are_non_empty_immutable_unique_and_canonically_ordered() -> None:
    surface = _surface("regions")
    first = _region("a")
    second = _region("b", status=SurfaceRegionSupportStatus.HOLE)

    support = _support_map("regions", surface=surface, regions=(first, second))
    assert support.regions == (first, second)

    with pytest.raises(ValueError, match="non-empty"):
        _support_map("empty", surface=surface, regions=cast(Any, ()))

    with pytest.raises(TypeError, match="SurfaceSupportRegion"):
        _support_map(
            "wrong-member",
            surface=surface,
            regions=cast(Any, (_artifact("region:wrong"),)),
        )

    duplicate = SurfaceSupportRegion(
        selector_artifact=first.selector_artifact,
        status=SurfaceRegionSupportStatus.HOLE,
    )
    with pytest.raises(ValueError, match="unique"):
        _support_map("duplicate", surface=surface, regions=(first, duplicate))

    with pytest.raises(ValueError, match="canonical"):
        _support_map("unordered", surface=surface, regions=(second, first))


def test_selector_artifacts_must_be_distinct_from_surface_and_support_map() -> None:
    surface = _surface("selector-collision")
    support_ref = ArtifactRef(
        ArtifactId("surface-support:selector-collision"),
        SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
    )

    surface_collision = SurfaceSupportRegion(
        selector_artifact=surface.artifact_ref,
        status=SurfaceRegionSupportStatus.UNSUPPORTED,
    )
    with pytest.raises(ValueError, match="distinct"):
        _support_map(
            "surface-collision",
            surface=surface,
            regions=(surface_collision,),
            artifact_ref=support_ref,
        )

    support_collision = SurfaceSupportRegion(
        selector_artifact=support_ref,
        status=SurfaceRegionSupportStatus.HOLE,
    )
    with pytest.raises(ValueError, match="distinct"):
        _support_map(
            "support-collision",
            surface=surface,
            regions=(support_collision,),
            artifact_ref=support_ref,
        )


def test_source_artifacts_must_equal_exact_canonical_surface_selector_union() -> None:
    surface = _surface("ancestry")
    first = _region("ancestry:a", kind="evidence.mask")
    second = _region(
        "ancestry:b",
        status=SurfaceRegionSupportStatus.HOLE,
        kind="evidence.polygon",
    )
    regions = (first, second)
    expected = tuple(
        sorted(
            (
                surface.artifact_ref,
                first.selector_artifact,
                second.selector_artifact,
            ),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )

    support = _support_map(
        "ancestry",
        surface=surface,
        regions=regions,
        source_artifacts=expected,
    )
    assert support.source_artifacts is expected

    with pytest.raises(ValueError, match="canonical union"):
        _support_map(
            "missing",
            surface=surface,
            regions=regions,
            source_artifacts=expected[:-1],
        )

    extra = _artifact("region:extra", "evidence.mask")
    with pytest.raises(ValueError, match="canonical union"):
        _support_map(
            "extra",
            surface=surface,
            regions=regions,
            source_artifacts=tuple(
                sorted(
                    (*expected, extra),
                    key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
                )
            ),
        )

    with pytest.raises(ValueError, match="unique"):
        _support_map(
            "duplicate-source",
            surface=surface,
            regions=regions,
            source_artifacts=(*expected, expected[-1]),
        )

    with pytest.raises(ValueError, match="canonical"):
        _support_map(
            "unordered-source",
            surface=surface,
            regions=regions,
            source_artifacts=tuple(reversed(expected)),
        )

    relabeled = ArtifactRef(
        artifact_id=first.selector_artifact.artifact_id,
        artifact_kind=ArtifactKind("evidence.relabelled"),
    )
    with pytest.raises(ValueError, match="canonical union"):
        _support_map(
            "relabeled",
            surface=surface,
            regions=regions,
            source_artifacts=tuple(
                sorted(
                    (surface.artifact_ref, relabeled, second.selector_artifact),
                    key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
                )
            ),
        )


def test_source_artifacts_reject_wrong_members_and_map_identity_collision() -> None:
    surface = _surface("source-types")
    region = _region("source-types")
    support_ref = ArtifactRef(
        ArtifactId("surface-support:source-types"),
        SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
    )

    with pytest.raises(TypeError, match="ArtifactRef"):
        _support_map(
            "wrong-source-member",
            surface=surface,
            regions=(region,),
            source_artifacts=cast(Any, ("artifact",)),
        )

    collision_region = SurfaceSupportRegion(
        selector_artifact=support_ref,
        status=SurfaceRegionSupportStatus.UNSUPPORTED,
    )
    with pytest.raises(ValueError, match="distinct"):
        _support_map(
            "map-source-collision",
            surface=surface,
            regions=(collision_region,),
            artifact_ref=support_ref,
        )


@pytest.mark.parametrize("scale", [GeometryScaleStatus.UNRESOLVED, GeometryScaleStatus.METRIC])
def test_surface_identity_frame_scale_representation_and_intended_uses_are_preserved(
    scale: GeometryScaleStatus,
) -> None:
    surface = _surface(
        f"preserve:{scale.value}",
        scale=scale,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )
    support = _support_map(f"preserve:{scale.value}", surface=surface)

    assert support.surface_model is surface
    assert support.local_frame_id is surface.local_frame_id
    assert support.surface_model.scale_status is scale
    assert support.surface_model.representation == SurfaceRepresentationName("mesh")
    assert support.surface_model.intended_uses == (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.NAVIGATION,
    )
    assert support.surface_model.producer is surface.producer
    assert support.surface_model.source_artifacts is surface.source_artifacts


def test_negative_evidence_does_not_create_positive_support_or_completeness_claims() -> None:
    support = _support_map("negative-only")

    forbidden = {
        "supported",
        "support_complete",
        "complete",
        "completeness",
        "watertight",
        "safe",
        "accurate",
        "collision_safe",
        "navigation_ready",
        "measurement_accurate",
        "suitable",
        "suitability",
        "quality_decision",
        "decision",
        "score",
        "rank",
        "confidence",
    }
    assert forbidden.isdisjoint(field.name for field in fields(SurfaceSupportMap))
    assert forbidden.isdisjoint(field.name for field in fields(SurfaceSupportRegion))
    for attribute in forbidden:
        assert not hasattr(support, attribute)
        assert not hasattr(support.regions[0], attribute)


def test_statuses_neither_request_fill_nor_authorize_generated_completion() -> None:
    unsupported = _region(
        "unsupported",
        status=SurfaceRegionSupportStatus.UNSUPPORTED,
    )
    hole = _region("hole", status=SurfaceRegionSupportStatus.HOLE)

    assert unsupported.status.value == "unsupported"
    assert hole.status.value == "hole"

    forbidden = (
        "fill",
        "filled",
        "repair",
        "inpaint",
        "completion",
        "complete",
        "generated",
        "synthetic",
        "geometry",
        "provenance_class",
        "collision_truth",
        "measurement_truth",
        "navigation_truth",
    )
    for region in (unsupported, hole):
        for attribute in forbidden:
            assert not hasattr(region, attribute)


def test_intended_uses_remain_caller_intent_and_are_not_reinterpreted() -> None:
    surface = _surface(
        "intended-use",
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )
    before = surface.intended_uses
    support = _support_map("intended-use", surface=surface)

    assert support.surface_model.intended_uses is before
    assert surface.intended_uses == (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.MEASUREMENT,
        SurfaceIntendedUse.NAVIGATION,
    )
    for attribute in (
        "collision_suitable",
        "measurement_suitable",
        "navigation_suitable",
        "collision_ready",
        "measurement_ready",
        "navigation_ready",
    ):
        assert not hasattr(support, attribute)
        assert not hasattr(support.surface_model, attribute)


def test_annotation_construction_does_not_mutate_surface_selectors_or_payload_metadata() -> None:
    surface = _surface("immutable")
    first = _region("immutable:a")
    second = _region(
        "immutable:b",
        status=SurfaceRegionSupportStatus.HOLE,
    )
    regions = (first, second)
    surface_source_artifacts_before = surface.source_artifacts
    intended_uses_before = surface.intended_uses
    geometry_positions_before = surface.source_geometry.point_maps[0].positions_xyz
    selector_refs_before = tuple(region.selector_artifact for region in regions)

    support = _support_map("immutable", surface=surface, regions=regions)

    assert support.surface_model is surface
    assert support.regions is regions
    assert surface.source_artifacts is surface_source_artifacts_before
    assert surface.intended_uses is intended_uses_before
    assert surface.source_geometry.point_maps[0].positions_xyz is geometry_positions_before
    assert tuple(region.selector_artifact for region in regions) == selector_refs_before
    assert support.producer != surface.producer


def test_module_surface_is_pure_negative_annotation_only() -> None:
    assert set(support_module.__all__) == {
        "SURFACE_SUPPORT_MAP_ARTIFACT_KIND",
        "SurfaceRegionSupportStatus",
        "SurfaceSupportMap",
        "SurfaceSupportRegion",
    }

    forbidden_names = {
        "numpy",
        "np",
        "open3d",
        "trimesh",
        "pymeshlab",
        "torch",
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "urllib",
        "QualityDecision",
        "ProvenanceClass",
        "CompletionArtifact",
        "RuntimeScene",
        "MasterScene",
        "BenchmarkRecord",
        "SurfaceAdapter",
    }
    assert forbidden_names.isdisjoint(vars(support_module))

    for name in (
        "parse_selector",
        "read_selector",
        "generate_selector",
        "rasterize_selector",
        "fill_holes",
        "repair_surface",
        "generate_completion",
        "complete_surface",
        "evaluate_quality",
        "benchmark",
        "route",
        "select_candidate",
        "compile_runtime",
    ):
        assert not hasattr(support_module, name)
