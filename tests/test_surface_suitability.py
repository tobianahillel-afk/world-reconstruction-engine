from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.surface_suitability as suitability_module
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
from wre.domain.metrics import (
    MetricAggregation,
    MetricDimension,
    MetricDirection,
    MetricUnit,
    MetricVector,
)
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
from wre.reconstruction.surface_suitability import (
    BOUNDARY_EDGE_RATIO_DESCRIPTOR,
    CONNECTED_COMPONENT_COUNT_DESCRIPTOR,
    DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR,
    FINITE_VERTEX_RATIO_DESCRIPTOR,
    LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR,
    NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR,
    SURFACE_SUITABILITY_EVALUATOR,
    SurfaceMeshInspection,
    SurfaceSuitabilityReport,
    evaluate_surface_suitability,
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.surface_suitability.{token}",
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
        source_artifacts=(_artifact(f"artifact:{token}"),),
    )


def _surface(
    token: str,
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    representation: SurfaceRepresentationName | None = None,
    intended_uses: tuple[SurfaceIntendedUse, ...] = (SurfaceIntendedUse.COLLISION,),
    artifact_ref: ArtifactRef | None = None,
) -> SurfaceModel:
    source = _candidate(token, scale=scale)
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
        producer=_producer(f"surface:{token}"),
        source_artifacts=source.source_artifacts,
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
                relative_path="surface.ply",
                sha256=Sha256Digest("1" * 64),
                byte_length=1234,
            ),
        ),
    )


def _inspection(
    token: str = "inspection",
    *,
    surface: SurfaceModel | None = None,
    materialization: ArtifactMaterializationMetadata | None = None,
    producer: ArtifactProducerIdentity | None = None,
    vertex_count: int = 10,
    triangle_count: int = 8,
    finite_vertex_count: int = 8,
    degenerate_triangle_count: int = 2,
    unique_edge_count: int = 12,
    boundary_edge_count: int = 3,
    non_manifold_edge_count: int = 1,
    connected_component_count: int = 2,
    largest_component_triangle_count: int = 6,
) -> SurfaceMeshInspection:
    retained_surface = surface or _surface(token)
    return SurfaceMeshInspection(
        surface_model=retained_surface,
        materialization=materialization or _materialization(retained_surface),
        producer=producer or _producer(f"inspection:{token}"),
        vertex_count=vertex_count,
        triangle_count=triangle_count,
        finite_vertex_count=finite_vertex_count,
        degenerate_triangle_count=degenerate_triangle_count,
        unique_edge_count=unique_edge_count,
        boundary_edge_count=boundary_edge_count,
        non_manifold_edge_count=non_manifold_edge_count,
        connected_component_count=connected_component_count,
        largest_component_triangle_count=largest_component_triangle_count,
    )


def _metric_map(report: SurfaceSuitabilityReport) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value
        for observation in report.metrics.observations
    }


def test_frozen_shapes_are_exact_and_evaluator_identity_is_stable() -> None:
    inspection = _inspection("shape")
    report = evaluate_surface_suitability(inspection)

    assert tuple(field.name for field in fields(SurfaceMeshInspection)) == (
        "surface_model",
        "materialization",
        "producer",
        "vertex_count",
        "triangle_count",
        "finite_vertex_count",
        "degenerate_triangle_count",
        "unique_edge_count",
        "boundary_edge_count",
        "non_manifold_edge_count",
        "connected_component_count",
        "largest_component_triangle_count",
    )
    assert tuple(field.name for field in fields(SurfaceSuitabilityReport)) == (
        "inspection",
        "metrics",
    )
    assert SURFACE_SUITABILITY_EVALUATOR.producer.implementation == (
        "wre.reconstruction.surface_suitability"
    )
    assert SURFACE_SUITABILITY_EVALUATOR.producer.version == "1"
    assert SURFACE_SUITABILITY_EVALUATOR.producer.revision == "v2l16.3"
    assert report.inspection is inspection

    with pytest.raises(FrozenInstanceError):
        inspection.vertex_count = 11  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.metrics = MetricVector(observations=())  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("surface_model", cast(Any, "surface"), "surface_model"),
        ("materialization", cast(Any, "materialization"), "materialization"),
        ("producer", cast(Any, "producer"), "producer"),
        ("vertex_count", cast(Any, 1.0), "vertex_count"),
        ("triangle_count", cast(Any, True), "triangle_count"),
        ("finite_vertex_count", cast(Any, 1.0), "finite_vertex_count"),
        ("degenerate_triangle_count", cast(Any, 1.0), "degenerate_triangle_count"),
        ("unique_edge_count", cast(Any, 1.0), "unique_edge_count"),
        ("boundary_edge_count", cast(Any, 1.0), "boundary_edge_count"),
        ("non_manifold_edge_count", cast(Any, 1.0), "non_manifold_edge_count"),
        ("connected_component_count", cast(Any, 1.0), "connected_component_count"),
        (
            "largest_component_triangle_count",
            cast(Any, 1.0),
            "largest_component_triangle_count",
        ),
    ],
)
def test_inspection_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    surface = _surface("types")
    kwargs: dict[str, Any] = {
        "surface_model": surface,
        "materialization": _materialization(surface),
        "producer": _producer("inspection:types"),
        "vertex_count": 10,
        "triangle_count": 8,
        "finite_vertex_count": 8,
        "degenerate_triangle_count": 2,
        "unique_edge_count": 12,
        "boundary_edge_count": 3,
        "non_manifold_edge_count": 1,
        "connected_component_count": 2,
        "largest_component_triangle_count": 6,
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        SurfaceMeshInspection(**kwargs)


def test_materialization_must_bind_exact_surface_artifact() -> None:
    surface = _surface("materialization")
    wrong_ref = ArtifactRef(
        artifact_id=ArtifactId("surface:other"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )

    with pytest.raises(ValueError, match="exactly match"):
        _inspection(
            "materialization",
            surface=surface,
            materialization=_materialization(surface, artifact_ref=wrong_ref),
        )


def test_only_mesh_representation_is_accepted() -> None:
    surface = _surface(
        "representation",
        representation=SurfaceRepresentationName("sdf"),
    )

    with pytest.raises(ValueError, match="representation mesh"):
        _inspection("representation", surface=surface)


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("vertex_count", 0, "vertex_count"),
        ("triangle_count", 0, "triangle_count"),
        ("finite_vertex_count", -1, "finite_vertex_count"),
        ("degenerate_triangle_count", -1, "degenerate_triangle_count"),
        ("unique_edge_count", -1, "unique_edge_count"),
        ("boundary_edge_count", -1, "boundary_edge_count"),
        ("non_manifold_edge_count", -1, "non_manifold_edge_count"),
        ("connected_component_count", 0, "connected_component_count"),
        ("largest_component_triangle_count", 0, "largest_component_triangle_count"),
    ],
)
def test_topology_counts_require_positive_or_non_negative_values(
    field_name: str,
    value: int,
    message: str,
) -> None:
    kwargs: dict[str, Any] = {
        "vertex_count": 10,
        "triangle_count": 8,
        "finite_vertex_count": 8,
        "degenerate_triangle_count": 2,
        "unique_edge_count": 12,
        "boundary_edge_count": 3,
        "non_manifold_edge_count": 1,
        "connected_component_count": 2,
        "largest_component_triangle_count": 6,
    }
    kwargs[field_name] = value

    with pytest.raises(ValueError, match=message):
        _inspection("counts", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vertex_count": 4, "finite_vertex_count": 5}, "finite_vertex_count"),
        ({"triangle_count": 4, "degenerate_triangle_count": 5}, "degenerate_triangle_count"),
        ({"unique_edge_count": 4, "boundary_edge_count": 5}, "boundary_edge_count"),
        ({"unique_edge_count": 4, "non_manifold_edge_count": 5}, "non_manifold_edge_count"),
        ({"triangle_count": 4, "connected_component_count": 5}, "connected_component_count"),
        (
            {"triangle_count": 4, "largest_component_triangle_count": 5},
            "largest_component_triangle_count",
        ),
    ],
)
def test_topology_counts_fail_closed_when_bounded_counts_exceed_denominators(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _inspection("bounds", **kwargs)


def test_metric_descriptors_values_units_directions_and_aggregations_are_exact() -> None:
    report = evaluate_surface_suitability(_inspection("metrics"))
    observations = report.metrics.observations

    assert tuple(item.descriptor.name.value for item in observations) == (
        "geometry.surface.boundary_edge_ratio",
        "geometry.surface.connected_component_count",
        "geometry.surface.degenerate_triangle_ratio",
        "geometry.surface.finite_vertex_ratio",
        "geometry.surface.largest_component_triangle_ratio",
        "geometry.surface.non_manifold_edge_ratio",
    )
    assert _metric_map(report) == {
        "geometry.surface.boundary_edge_ratio": 3 / 12,
        "geometry.surface.connected_component_count": 2.0,
        "geometry.surface.degenerate_triangle_ratio": 2 / 8,
        "geometry.surface.finite_vertex_ratio": 8 / 10,
        "geometry.surface.largest_component_triangle_ratio": 6 / 8,
        "geometry.surface.non_manifold_edge_ratio": 1 / 12,
    }

    assert FINITE_VERTEX_RATIO_DESCRIPTOR.dimension == MetricDimension("geometry.surface")
    assert FINITE_VERTEX_RATIO_DESCRIPTOR.unit == MetricUnit("ratio")
    assert FINITE_VERTEX_RATIO_DESCRIPTOR.direction is MetricDirection.HIGHER_IS_BETTER
    assert FINITE_VERTEX_RATIO_DESCRIPTOR.aggregation == MetricAggregation("vertex_ratio")

    assert DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR.unit == MetricUnit("ratio")
    assert DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR.direction is MetricDirection.LOWER_IS_BETTER
    assert DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR.aggregation == MetricAggregation("triangle_ratio")

    assert BOUNDARY_EDGE_RATIO_DESCRIPTOR.unit == MetricUnit("ratio")
    assert BOUNDARY_EDGE_RATIO_DESCRIPTOR.direction is MetricDirection.LOWER_IS_BETTER
    assert BOUNDARY_EDGE_RATIO_DESCRIPTOR.aggregation == MetricAggregation("unique_edge_ratio")

    assert NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR.unit == MetricUnit("ratio")
    assert NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR.direction is MetricDirection.LOWER_IS_BETTER
    assert NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR.aggregation == MetricAggregation("unique_edge_ratio")

    assert LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR.unit == MetricUnit("ratio")
    assert LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR.direction is MetricDirection.HIGHER_IS_BETTER
    assert LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR.aggregation == MetricAggregation(
        "triangle_ratio"
    )

    assert CONNECTED_COMPONENT_COUNT_DESCRIPTOR.unit == MetricUnit("count")
    assert CONNECTED_COMPONENT_COUNT_DESCRIPTOR.direction is MetricDirection.INFORMATIONAL
    assert CONNECTED_COMPONENT_COUNT_DESCRIPTOR.aggregation == MetricAggregation("count")


def test_zero_unique_edges_omits_edge_ratios_instead_of_fabricating_values() -> None:
    report = evaluate_surface_suitability(
        _inspection(
            "zero-edges",
            unique_edge_count=0,
            boundary_edge_count=0,
            non_manifold_edge_count=0,
        )
    )
    names = tuple(item.descriptor.name.value for item in report.metrics.observations)

    assert "geometry.surface.boundary_edge_ratio" not in names
    assert "geometry.surface.non_manifold_edge_ratio" not in names
    assert names == (
        "geometry.surface.connected_component_count",
        "geometry.surface.degenerate_triangle_ratio",
        "geometry.surface.finite_vertex_ratio",
        "geometry.surface.largest_component_triangle_ratio",
    )


def test_unresolved_scale_keeps_topology_metrics_descriptive_without_metric_claims() -> None:
    surface = _surface("unresolved", scale=GeometryScaleStatus.UNRESOLVED)
    inspection = _inspection("unresolved", surface=surface)
    report = evaluate_surface_suitability(inspection)

    assert report.inspection.surface_model.scale_status is GeometryScaleStatus.UNRESOLVED
    assert all(
        observation.descriptor.unit.value in {"ratio", "count"}
        for observation in report.metrics.observations
    )
    assert all(
        token not in observation.descriptor.name.value
        for observation in report.metrics.observations
        for token in ("meter", "length", "area", "volume", "tolerance", "accuracy")
    )


def test_metric_scale_does_not_create_measurement_accuracy_or_suitability_decisions() -> None:
    surface = _surface(
        "metric",
        scale=GeometryScaleStatus.METRIC,
        intended_uses=(SurfaceIntendedUse.MEASUREMENT,),
    )
    report = evaluate_surface_suitability(_inspection("metric", surface=surface))

    assert report.inspection.surface_model.scale_status is GeometryScaleStatus.METRIC
    forbidden = (
        "measurement_accuracy",
        "measurement_suitable",
        "measurement_ready",
        "accuracy",
        "tolerance",
        "quality_decision",
        "decision",
    )
    assert all(
        token not in observation.descriptor.name.value
        for observation in report.metrics.observations
        for token in forbidden
    )
    for attribute in forbidden:
        assert not hasattr(report, attribute)


def test_intended_use_metadata_does_not_change_metric_vector() -> None:
    shared_ref = ArtifactRef(
        artifact_id=ArtifactId("surface:intended-use-invariant"),
        artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
    )
    collision = _surface(
        "uses",
        intended_uses=(SurfaceIntendedUse.COLLISION,),
        artifact_ref=shared_ref,
    )
    all_uses = SurfaceModel(
        artifact_ref=shared_ref,
        source_geometry=collision.source_geometry,
        representation=collision.representation,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
        local_frame_id=collision.local_frame_id,
        scale_status=collision.scale_status,
        producer=collision.producer,
        source_artifacts=collision.source_artifacts,
    )

    first = evaluate_surface_suitability(_inspection("uses-a", surface=collision))
    second = evaluate_surface_suitability(_inspection("uses-b", surface=all_uses))

    assert first.metrics == second.metrics
    assert first.inspection.surface_model.intended_uses != second.inspection.surface_model.intended_uses


def test_inspection_producer_and_metric_provenance_are_distinct_and_exact() -> None:
    surface = _surface("provenance")
    inspection_producer = _producer("caller-topology-inspection")
    inspection = _inspection(
        "provenance",
        surface=surface,
        producer=inspection_producer,
    )
    report = evaluate_surface_suitability(inspection)

    assert report.inspection.producer is inspection_producer
    for observation in report.metrics.observations:
        assert observation.provenance.evaluator == SURFACE_SUITABILITY_EVALUATOR
        assert observation.provenance.evaluator != inspection_producer
        assert observation.provenance.input_artifacts == (surface.artifact_ref,)


def test_evaluator_preserves_exact_surface_materialization_and_ancestry_without_mutation() -> None:
    surface = _surface(
        "immutable",
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )
    materialization = _materialization(surface)
    producer = _producer("immutable-inspection")
    inspection = _inspection(
        "immutable",
        surface=surface,
        materialization=materialization,
        producer=producer,
    )

    source_artifacts_before = surface.source_artifacts
    entries_before = materialization.entries
    report = evaluate_surface_suitability(inspection)

    assert report.inspection is inspection
    assert report.inspection.surface_model is surface
    assert report.inspection.materialization is materialization
    assert report.inspection.producer is producer
    assert surface.source_artifacts is source_artifacts_before
    assert materialization.entries is entries_before
    assert surface.source_geometry.point_maps[0].positions_xyz == ((1.0, 2.0, 3.0),)


def test_report_rejects_noncanonical_metrics() -> None:
    inspection = _inspection("report")

    with pytest.raises(ValueError, match="canonical descriptive surface metrics"):
        SurfaceSuitabilityReport(
            inspection=inspection,
            metrics=MetricVector(observations=()),
        )

    with pytest.raises(TypeError, match="inspection"):
        SurfaceSuitabilityReport(
            inspection=cast(Any, "inspection"),
            metrics=MetricVector(observations=()),
        )

    with pytest.raises(TypeError, match="metrics"):
        SurfaceSuitabilityReport(
            inspection=inspection,
            metrics=cast(Any, "metrics"),
        )


def test_perfect_looking_topology_still_has_no_watertightness_or_safety_verdict() -> None:
    report = evaluate_surface_suitability(
        _inspection(
            "perfect",
            vertex_count=8,
            triangle_count=12,
            finite_vertex_count=8,
            degenerate_triangle_count=0,
            unique_edge_count=18,
            boundary_edge_count=0,
            non_manifold_edge_count=0,
            connected_component_count=1,
            largest_component_triangle_count=12,
        )
    )
    values = _metric_map(report)
    assert values["geometry.surface.boundary_edge_ratio"] == 0.0
    assert values["geometry.surface.non_manifold_edge_ratio"] == 0.0
    assert values["geometry.surface.largest_component_triangle_ratio"] == 1.0

    forbidden = {
        "collision_suitable",
        "measurement_suitable",
        "navigation_suitable",
        "watertight",
        "safe",
        "accurate",
        "ready",
        "pass",
        "fail",
        "threshold",
        "score",
        "rank",
        "decision",
        "quality_decision",
    }
    assert forbidden.isdisjoint(field.name for field in fields(SurfaceSuitabilityReport))
    assert forbidden.isdisjoint(field.name for field in fields(SurfaceMeshInspection))
    assert all(
        token not in observation.descriptor.name.value
        for observation in report.metrics.observations
        for token in forbidden
    )


def test_module_surface_contains_only_descriptive_v2l16_3_behavior() -> None:
    assert set(suitability_module.__all__) == {
        "BOUNDARY_EDGE_RATIO_DESCRIPTOR",
        "CONNECTED_COMPONENT_COUNT_DESCRIPTOR",
        "DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR",
        "FINITE_VERTEX_RATIO_DESCRIPTOR",
        "LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR",
        "NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR",
        "SURFACE_SUITABILITY_EVALUATOR",
        "SurfaceMeshInspection",
        "SurfaceSuitabilityReport",
        "evaluate_surface_suitability",
    }

    forbidden_names = {
        "open3d",
        "numpy",
        "trimesh",
        "pymeshlab",
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "QualityDecision",
        "SurfaceAdapter",
        "HybridSurface",
        "UnsupportedRegion",
        "BenchmarkRecord",
        "RuntimeScene",
        "MasterScene",
    }
    assert forbidden_names.isdisjoint(vars(suitability_module))

    for name in (
        "repair_mesh",
        "fill_holes",
        "generate",
        "route",
        "select_candidate",
        "evaluate_quality",
        "benchmark",
        "is_watertight",
        "collision_suitable",
        "measurement_ready",
    ):
        assert not hasattr(suitability_module, name)


def test_evaluate_surface_suitability_rejects_wrong_input_type() -> None:
    with pytest.raises(TypeError, match="SurfaceMeshInspection"):
        evaluate_surface_suitability(cast(Any, "inspection"))
