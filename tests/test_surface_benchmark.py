from __future__ import annotations

import builtins
import hashlib
import inspect
import json
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import wre.reconstruction.surface_benchmark as surface_benchmark_module
from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.benchmarks import BenchmarkFixtureId, BenchmarkFixtureIdentity
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
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.quality import QualityMode
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
from wre.reconstruction.surface_benchmark import (
    SurfaceBenchmarkCandidate,
    SurfaceBenchmarkRequest,
    SurfaceBenchmarkResult,
    SurfaceCandidateBenchmarkResult,
    SurfaceSupportAnnotationState,
    benchmark_surface_candidates,
)
from wre.reconstruction.surface_suitability import (
    SURFACE_SUITABILITY_EVALUATOR,
    SurfaceMeshInspection,
    evaluate_surface_suitability,
)

_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "synthetic"
    / "physical-surface-controlled"
    / "fixture.json"
)
_WORKFLOW_PATH = (
    Path(__file__).parents[1]
    / ".github"
    / "workflows"
    / "physical-surface-controlled-benchmark.yml"
)
_EXPECTED_FIXTURE_SHA256 = "f4d1cac810d929a3bfbdd6afae728c3a91eea8887e10b10b7610925d0328b03d"
_FORBIDDEN_RECORD_KEYS = {
    "accuracy",
    "collision_ready",
    "completeness",
    "default",
    "default_route",
    "fallback",
    "generated_completion",
    "measurement_ready",
    "navigation_ready",
    "overall_score",
    "preferred_route",
    "qualitydecision",
    "rank",
    "retry",
    "safety",
    "score",
    "shipping_promotion",
    "supported_area",
    "supported_fraction",
    "threshold",
    "watertight",
    "winner",
}


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.surface_benchmark.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "benchmark.surface_source") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _artifact_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_id.value, ref.artifact_kind.value)


def _canonical_artifacts(*refs: ArtifactRef) -> tuple[ArtifactRef, ...]:
    by_identity = {_artifact_key(ref): ref for ref in refs}
    return tuple(by_identity[key] for key in sorted(by_identity))


def _source_geometry(
    token: str,
    *,
    scale: GeometryScaleStatus,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    camera = CameraSolution(
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
        source_artifacts=(_artifact(f"source:{token}"),),
    )


def _fixture_document() -> dict[str, Any]:
    raw = _FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_FIXTURE_SHA256
    document = json.loads(raw)
    assert document["schema_version"] == 1
    return cast(dict[str, Any], document)


def _surface(entry: dict[str, Any]) -> SurfaceModel:
    artifact_id = cast(str, entry["artifact_id"])
    token = cast(str, entry["producer_token"])
    scale = GeometryScaleStatus(cast(str, entry["scale_status"]))
    source = _source_geometry(artifact_id.replace("surface:", ""), scale=scale)
    intended_uses = tuple(
        SurfaceIntendedUse(value) for value in cast(list[str], entry["intended_uses"])
    )
    return SurfaceModel(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(artifact_id),
            artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
        ),
        source_geometry=source,
        representation=SurfaceRepresentationName("mesh"),
        intended_uses=intended_uses,
        local_frame_id=source.geometry_solution.local_frame_id,
        scale_status=source.geometry_solution.scale_status,
        producer=_producer(token),
        source_artifacts=source.source_artifacts,
    )


def _inspection(surface: SurfaceModel, entry: dict[str, Any]) -> SurfaceMeshInspection:
    materialization_raw = cast(dict[str, Any], entry["materialization"])
    inspection_raw = cast(dict[str, Any], entry["inspection"])
    materialization = ArtifactMaterializationMetadata(
        artifact_ref=surface.artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path=cast(str, materialization_raw["relative_path"]),
                sha256=Sha256Digest(cast(str, materialization_raw["sha256"])),
                byte_length=cast(int, materialization_raw["byte_length"]),
            ),
        ),
    )
    return SurfaceMeshInspection(
        surface_model=surface,
        materialization=materialization,
        producer=_producer(cast(str, inspection_raw["producer_token"])),
        vertex_count=cast(int, inspection_raw["vertex_count"]),
        triangle_count=cast(int, inspection_raw["triangle_count"]),
        finite_vertex_count=cast(int, inspection_raw["finite_vertex_count"]),
        degenerate_triangle_count=cast(int, inspection_raw["degenerate_triangle_count"]),
        unique_edge_count=cast(int, inspection_raw["unique_edge_count"]),
        boundary_edge_count=cast(int, inspection_raw["boundary_edge_count"]),
        non_manifold_edge_count=cast(int, inspection_raw["non_manifold_edge_count"]),
        connected_component_count=cast(int, inspection_raw["connected_component_count"]),
        largest_component_triangle_count=cast(
            int,
            inspection_raw["largest_component_triangle_count"],
        ),
    )


def _support_map(
    surface: SurfaceModel,
    support_raw: dict[str, Any],
) -> SurfaceSupportMap:
    regions = tuple(
        SurfaceSupportRegion(
            selector_artifact=ArtifactRef(
                artifact_id=ArtifactId(cast(str, region["selector_artifact_id"])),
                artifact_kind=ArtifactKind(cast(str, region["selector_artifact_kind"])),
            ),
            status=SurfaceRegionSupportStatus(cast(str, region["status"])),
        )
        for region in cast(list[dict[str, Any]], support_raw["regions"])
    )
    return SurfaceSupportMap(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(cast(str, support_raw["artifact_id"])),
            artifact_kind=SURFACE_SUPPORT_MAP_ARTIFACT_KIND,
        ),
        surface_model=surface,
        local_frame_id=surface.local_frame_id,
        regions=regions,
        producer=_producer(cast(str, support_raw["producer_token"])),
        source_artifacts=_canonical_artifacts(
            surface.artifact_ref,
            *(region.selector_artifact for region in regions),
        ),
    )


def _candidate(entry: dict[str, Any]) -> SurfaceBenchmarkCandidate:
    surface = _surface(entry)
    inspection = _inspection(surface, entry)
    support_raw = entry["support"]
    support_map = (
        None if support_raw is None else _support_map(surface, cast(dict[str, Any], support_raw))
    )
    return SurfaceBenchmarkCandidate(
        inspection=inspection,
        support_map=support_map,
    )


def _request() -> SurfaceBenchmarkRequest:
    document = _fixture_document()
    candidates = tuple(
        _candidate(cast(dict[str, Any], entry))
        for entry in cast(list[dict[str, Any]], document["candidates"])
    )
    return SurfaceBenchmarkRequest(
        fixture=BenchmarkFixtureIdentity(
            fixture_id=BenchmarkFixtureId(cast(str, document["fixture_id"])),
            sha256=Sha256Digest(_EXPECTED_FIXTURE_SHA256),
        ),
        hardware=HardwareRuntimeIdentity(
            sha256=Sha256Digest(hashlib.sha256(b"v2l16.6-controlled-cpu-locked-wre").hexdigest())
        ),
        candidates=candidates,
    )


def _artifact_record(ref: ArtifactRef) -> dict[str, str]:
    return {
        "artifact_id": ref.artifact_id.value,
        "artifact_kind": ref.artifact_kind.value,
    }


def _producer_record(producer: ArtifactProducerIdentity) -> dict[str, Any]:
    return {
        "producer": {
            "implementation": producer.producer.implementation,
            "version": producer.producer.version,
            "revision": producer.producer.revision,
        },
        "configuration_sha256": producer.configuration.sha256.value,
        "model": (
            None
            if producer.model is None
            else {
                "name": producer.model.name,
                "version": producer.model.version,
                "revision": producer.model.revision,
            }
        ),
        "checkpoint": (
            None
            if producer.checkpoint is None
            else {
                "identifier": producer.checkpoint.identifier,
                "sha256": producer.checkpoint.sha256.value,
            }
        ),
    }


def _metric_vector_record(vector: MetricVector) -> list[dict[str, Any]]:
    return [
        {
            "name": observation.descriptor.name.value,
            "dimension": observation.descriptor.dimension.value,
            "unit": observation.descriptor.unit.value,
            "direction": observation.descriptor.direction.value,
            "aggregation": observation.descriptor.aggregation.value,
            "value": observation.value,
            "provenance": {
                "evaluator": _producer_record(observation.provenance.evaluator),
                "input_artifacts": [
                    _artifact_record(ref) for ref in observation.provenance.input_artifacts
                ],
            },
        }
        for observation in vector.observations
    ]


def _materialization_record(
    materialization: ArtifactMaterializationMetadata,
) -> dict[str, Any]:
    return {
        "artifact": _artifact_record(materialization.artifact_ref),
        "entries": [
            {
                "relative_path": entry.relative_path,
                "sha256": entry.sha256.value,
                "byte_length": entry.byte_length,
            }
            for entry in materialization.entries
        ],
    }


def _result_document(result: SurfaceBenchmarkResult) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for item in result.candidates:
        surface = item.candidate.inspection.surface_model
        support_map = item.candidate.support_map
        support_annotation: dict[str, Any]
        if support_map is None:
            support_annotation = {
                "state": SurfaceSupportAnnotationState.UNKNOWN.value,
                "support_map_artifact": None,
                "regions": None,
            }
        else:
            support_annotation = {
                "state": SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS.value,
                "support_map_artifact": _artifact_record(support_map.artifact_ref),
                "producer": _producer_record(support_map.producer),
                "regions": [
                    {
                        "selector_artifact": _artifact_record(region.selector_artifact),
                        "status": region.status.value,
                    }
                    for region in item.support_regions or ()
                ],
            }

        inspection = item.candidate.inspection
        candidates.append(
            {
                "surface_model": {
                    "artifact": _artifact_record(surface.artifact_ref),
                    "producer": _producer_record(surface.producer),
                    "geometry_solution_id": surface.source_geometry.geometry_solution_id.value,
                    "local_frame_id": surface.local_frame_id.value,
                    "scale_status": surface.scale_status.value,
                    "representation": surface.representation.value,
                    "intended_uses": [value.value for value in surface.intended_uses],
                },
                "materialization": _materialization_record(inspection.materialization),
                "inspection": {
                    "producer": _producer_record(inspection.producer),
                    "vertex_count": inspection.vertex_count,
                    "triangle_count": inspection.triangle_count,
                    "finite_vertex_count": inspection.finite_vertex_count,
                    "degenerate_triangle_count": inspection.degenerate_triangle_count,
                    "unique_edge_count": inspection.unique_edge_count,
                    "boundary_edge_count": inspection.boundary_edge_count,
                    "non_manifold_edge_count": inspection.non_manifold_edge_count,
                    "connected_component_count": inspection.connected_component_count,
                    "largest_component_triangle_count": (
                        inspection.largest_component_triangle_count
                    ),
                },
                "benchmark_record_id": item.benchmark_record.record_id.value,
                "topology_metrics": _metric_vector_record(item.benchmark_record.metrics),
                "support_annotation": support_annotation,
            }
        )

    return {
        "schema_version": 1,
        "fixture": {
            "fixture_id": result.request.fixture.fixture_id.value,
            "sha256": result.request.fixture.sha256.value,
        },
        "hardware_runtime_sha256": result.request.hardware.sha256.value,
        "quality_mode": QualityMode.QUALITY.value,
        "candidates": candidates,
        "descriptive_only": True,
        "unavailable_dimensions": [
            "trusted_geometric_reference_error",
            "supported_area_or_fraction",
            "surface_completeness",
            "collision_navigation_measurement_certification",
            "production_surface_backend_selection",
        ],
    }


def _assert_no_forbidden_record_keys(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            assert normalized not in _FORBIDDEN_RECORD_KEYS
            _assert_no_forbidden_record_keys(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_forbidden_record_keys(child)


def test_candidate_request_and_result_have_exact_frozen_shapes() -> None:
    request = _request()
    result = benchmark_surface_candidates(request)
    item = result.candidates[0]

    assert tuple(field.name for field in fields(SurfaceBenchmarkCandidate)) == (
        "inspection",
        "support_map",
    )
    assert tuple(field.name for field in fields(SurfaceBenchmarkRequest)) == (
        "fixture",
        "hardware",
        "candidates",
    )
    assert tuple(field.name for field in fields(SurfaceCandidateBenchmarkResult)) == (
        "candidate",
        "suitability",
        "support_annotation_state",
        "support_map_artifact_ref",
        "support_regions",
        "benchmark_record",
    )
    assert tuple(field.name for field in fields(SurfaceBenchmarkResult)) == (
        "request",
        "candidates",
    )

    with pytest.raises(FrozenInstanceError):
        request.candidates = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        item.support_regions = None  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]


def test_request_rejects_wrong_types_minimum_duplicates_and_noncanonical_order() -> None:
    request = _request()
    first, second, third = request.candidates

    with pytest.raises(TypeError, match="fixture"):
        replace(request, fixture=cast(Any, "fixture"))
    with pytest.raises(TypeError, match="hardware"):
        replace(request, hardware=cast(Any, "hardware"))
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(request, candidates=cast(Any, list(request.candidates)))
    with pytest.raises(ValueError, match="at least three"):
        replace(request, candidates=(first, second))
    with pytest.raises(ValueError, match="unique"):
        replace(request, candidates=(first, first, third))
    with pytest.raises(ValueError, match="canonical"):
        replace(request, candidates=tuple(reversed(request.candidates)))


def test_candidate_rejects_foreign_support_surface_and_wrong_types() -> None:
    request = _request()
    first, second, _third = request.candidates
    assert first.support_map is not None

    with pytest.raises(TypeError, match="inspection"):
        SurfaceBenchmarkCandidate(
            inspection=cast(Any, "inspection"),
            support_map=None,
        )
    with pytest.raises(TypeError, match="support_map"):
        replace(first, support_map=cast(Any, "support"))
    with pytest.raises(ValueError, match="exact SurfaceModel"):
        SurfaceBenchmarkCandidate(
            inspection=second.inspection,
            support_map=first.support_map,
        )


def test_absent_support_annotation_is_explicit_unknown_not_zero_or_positive_support() -> None:
    result = benchmark_surface_candidates(_request())
    unknown = next(
        item
        for item in result.candidates
        if item.candidate.inspection.surface_model.artifact_ref.artifact_id.value
        == "surface:candidate-unknown"
    )

    assert unknown.candidate.support_map is None
    assert unknown.support_annotation_state is SurfaceSupportAnnotationState.UNKNOWN
    assert unknown.support_map_artifact_ref is None
    assert unknown.support_regions is None
    for attribute in (
        "supported_regions",
        "supported_area",
        "supported_fraction",
        "completeness",
        "watertight",
        "safety",
    ):
        assert not hasattr(unknown, attribute)


def test_negative_support_regions_and_identity_are_retained_exactly() -> None:
    result = benchmark_surface_candidates(_request())
    by_id = {
        item.candidate.inspection.surface_model.artifact_ref.artifact_id.value: item
        for item in result.candidates
    }

    hole = by_id["surface:candidate-hole"]
    assert hole.support_annotation_state is SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS
    assert hole.candidate.support_map is not None
    assert hole.support_map_artifact_ref == hole.candidate.support_map.artifact_ref
    assert hole.support_regions == hole.candidate.support_map.regions
    assert tuple(region.status for region in hole.support_regions or ()) == (
        SurfaceRegionSupportStatus.HOLE,
    )

    unsupported = by_id["surface:candidate-unsupported"]
    assert unsupported.candidate.support_map is not None
    assert unsupported.support_map_artifact_ref == unsupported.candidate.support_map.artifact_ref
    assert unsupported.support_regions == unsupported.candidate.support_map.regions
    assert tuple(region.status for region in unsupported.support_regions or ()) == (
        SurfaceRegionSupportStatus.UNSUPPORTED,
        SurfaceRegionSupportStatus.UNSUPPORTED,
    )
    assert tuple(
        region.selector_artifact.artifact_id.value for region in unsupported.support_regions or ()
    ) == (
        "selector:unsupported:facade",
        "selector:unsupported:ground",
    )


def test_v2l16_3_metric_vector_and_provenance_are_retained_unchanged() -> None:
    result = benchmark_surface_candidates(_request())

    for item in result.candidates:
        expected = evaluate_surface_suitability(item.candidate.inspection)
        assert item.suitability == expected
        assert item.benchmark_record.metrics == expected.metrics
        assert item.benchmark_record.producer == item.candidate.inspection.surface_model.producer
        assert item.benchmark_record.quality_mode is QualityMode.QUALITY
        assert item.benchmark_record.performance_evidence == ()
        assert item.benchmark_record.comparison_baseline is None
        for observation in item.benchmark_record.metrics.observations:
            assert observation.provenance.evaluator == SURFACE_SUITABILITY_EVALUATOR
            assert observation.provenance.input_artifacts == (
                item.candidate.inspection.surface_model.artifact_ref,
            )


def test_support_annotation_cannot_change_topology_metrics_or_producer_provenance() -> None:
    request = _request()
    first = request.candidates[0]
    assert first.support_map is not None

    annotated = benchmark_surface_candidates(request).candidates[0]
    without_annotation = SurfaceBenchmarkCandidate(
        inspection=first.inspection,
        support_map=None,
    )
    alternate_request = replace(
        request,
        candidates=(without_annotation, *request.candidates[1:]),
    )
    unknown = benchmark_surface_candidates(alternate_request).candidates[0]

    assert annotated.benchmark_record.metrics == unknown.benchmark_record.metrics
    assert annotated.benchmark_record.producer == unknown.benchmark_record.producer
    assert annotated.benchmark_record.record_id == unknown.benchmark_record.record_id
    assert annotated.support_annotation_state is SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS
    assert unknown.support_annotation_state is SurfaceSupportAnnotationState.UNKNOWN


def test_unresolved_and_metric_scale_remain_descriptive_without_accuracy_claims() -> None:
    result = benchmark_surface_candidates(_request())
    by_id = {
        item.candidate.inspection.surface_model.artifact_ref.artifact_id.value: item
        for item in result.candidates
    }

    assert (
        by_id["surface:candidate-unknown"].candidate.inspection.surface_model.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert (
        by_id["surface:candidate-hole"].candidate.inspection.surface_model.scale_status
        is GeometryScaleStatus.METRIC
    )

    for item in result.candidates:
        for observation in item.benchmark_record.metrics.observations:
            assert observation.descriptor.unit.value in {"ratio", "count"}
            assert all(
                token not in observation.descriptor.name.value
                for token in ("meter", "distance", "area", "volume", "tolerance", "accuracy")
            )


def test_result_records_reject_inconsistent_composition_and_support_state() -> None:
    result = benchmark_surface_candidates(_request())
    first = result.candidates[0]
    unknown = result.candidates[1]

    with pytest.raises(ValueError, match=r"exact V2L16\.3"):
        replace(
            first,
            benchmark_record=replace(
                first.benchmark_record,
                metrics=MetricVector(observations=()),
            ),
        )
    with pytest.raises(ValueError, match="negative_annotations"):
        replace(
            first,
            support_annotation_state=SurfaceSupportAnnotationState.UNKNOWN,
        )
    with pytest.raises(ValueError, match="remain unknown"):
        replace(
            unknown,
            support_annotation_state=SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS,
        )
    with pytest.raises(ValueError, match="request order"):
        replace(result, candidates=tuple(reversed(result.candidates)))


def test_fixture_hash_retained_json_and_benchmark_are_deterministic(tmp_path: Path) -> None:
    request = _request()
    first = benchmark_surface_candidates(request)
    second = benchmark_surface_candidates(request)
    assert first == second

    document = _result_document(first)
    _assert_no_forbidden_record_keys(document)
    assert document["fixture"]["sha256"] == _EXPECTED_FIXTURE_SHA256
    assert document["descriptive_only"] is True

    candidates = cast(list[dict[str, Any]], document["candidates"])
    unknown = next(
        item
        for item in candidates
        if item["surface_model"]["artifact"]["artifact_id"] == "surface:candidate-unknown"
    )
    assert unknown["support_annotation"] == {
        "state": "unknown",
        "support_map_artifact": None,
        "regions": None,
    }

    hole = next(
        item
        for item in candidates
        if item["surface_model"]["artifact"]["artifact_id"] == "surface:candidate-hole"
    )
    assert hole["support_annotation"]["regions"][0]["status"] == "hole"

    unsupported = next(
        item
        for item in candidates
        if item["surface_model"]["artifact"]["artifact_id"] == "surface:candidate-unsupported"
    )
    assert [region["status"] for region in unsupported["support_annotation"]["regions"]] == [
        "unsupported",
        "unsupported",
    ]

    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    output = tmp_path / "physical-surface-controlled.json"
    output.write_text(encoded, encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8")) == document

    requested_output = os.environ.get("WRE_PHYSICAL_SURFACE_CONTROLLED_BENCHMARK_OUTPUT")
    if requested_output:
        Path(requested_output).write_text(encoded, encoding="utf-8")


def test_benchmark_does_not_open_or_interpret_mesh_or_selector_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request()

    def _forbidden_open(*args: object, **kwargs: object) -> object:
        raise AssertionError(f"benchmark attempted payload/file access: {args!r} {kwargs!r}")

    monkeypatch.setattr(builtins, "open", _forbidden_open)
    result = benchmark_surface_candidates(request)
    assert len(result.candidates) == 3

    source = inspect.getsource(surface_benchmark_module).lower()
    for forbidden in (
        "import numpy",
        "import open3d",
        "import trimesh",
        "import pymeshlab",
        "import torch",
        "import pathlib",
        "import subprocess",
        "import socket",
        "import requests",
    ):
        assert forbidden not in source


def test_workflow_is_cpu_only_and_retains_machine_readable_evidence() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in workflow
    assert "WRE_PHYSICAL_SURFACE_CONTROLLED_BENCHMARK_OUTPUT" in workflow
    assert "v2l16-6-controlled-physical-surface-benchmark" in workflow
    assert "tests/test_surface_benchmark.py" in workflow
    for forbidden in (
        "cuda",
        "nvidia",
        "model.safetensors",
        "torch.hub",
        "huggingface",
        "open3d",
        "trimesh",
        "pymeshlab",
    ):
        assert forbidden not in workflow.lower()
