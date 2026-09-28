from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.benchmarks import BenchmarkFixtureId, BenchmarkFixtureIdentity
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
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.quality import QualityMode
from wre.domain.runs import ProducerRef
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact
from wre.reconstruction.dense_depth_benchmark import (
    DENSE_DEPTH_BENCHMARK_EVALUATOR,
    DenseDepthBenchmarkRequest,
    DenseDepthBenchmarkResult,
    DenseDepthCandidateBenchmarkResult,
    benchmark_dense_depth_candidates,
)
from wre.reconstruction.dense_depth_coverage import (
    DENSE_DEPTH_COVERAGE_EVALUATOR,
    evaluate_dense_depth_coverage,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "synthetic" / "dense-depth-controlled" / "fixture.json"
)
_WORKFLOW_PATH = (
    Path(__file__).parents[1] / ".github" / "workflows" / "dense-depth-controlled-benchmark.yml"
)
_EXPECTED_FIXTURE_SHA256 = "7356108b43fe0be5b1c206afa5f14d35262ba5c7d8b51fb093a969dc3dad3f1a"
_FORBIDDEN_RECORD_KEYS = {
    "default",
    "default_route",
    "fallback",
    "overall_score",
    "preferred_route",
    "qualitydecision",
    "rank",
    "retry",
    "score",
    "shipping_promotion",
    "threshold",
    "winner",
}


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.dense_depth_benchmark.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _source_geometry() -> GeometrySolutionCandidate:
    frame = LocalFrameId("frame:dense-depth-controlled")
    cameras = tuple(
        CameraSolution(
            solution_id=CameraSolutionId(f"camera:{token}"),
            observation_id=ObservationId(f"obs:{token}"),
            local_frame_id=frame,
            projection_model=CameraProjectionModelName("pinhole"),
            dimensions=ImageDimensions(width_px=2, height_px=2),
            intrinsic_parameters=(2.0, 2.0, 1.0, 1.0),
            rotation_matrix=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            ),
            translation_xyz=(float(index), 0.0, 0.0),
            uncertainty_artifacts=(),
            metrics=_metrics(),
        )
        for index, token in enumerate(("a", "b", "c"))
    )
    point_map = PointMap(
        point_map_id=PointMapId("points:dense-depth-controlled"),
        local_frame_id=frame,
        source_observation_ids=tuple(camera.observation_id for camera in cameras),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:dense-depth-controlled"),
        local_frame_id=frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(camera.solution_id for camera in cameras),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=cameras,
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer("source-geometry"),
        source_artifacts=(
            _artifact(
                "artifact:source:dense-depth-controlled",
                "benchmark.geometry_source",
            ),
        ),
    )


def _fixture_document() -> dict[str, Any]:
    raw = _FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _EXPECTED_FIXTURE_SHA256
    document = json.loads(raw)
    assert document["schema_version"] == 1
    return cast(dict[str, Any], document)


def _depth_field(
    *,
    source: GeometrySolutionCandidate,
    owner_token: str,
    field: dict[str, Any],
    convention: str,
) -> DepthField:
    token = cast(str, field["camera"])
    camera = next(
        item
        for item in source.camera_solutions
        if item.solution_id == CameraSolutionId(f"camera:{token}")
    )
    confidence_raw = field["confidence"]
    confidence = (
        None
        if confidence_raw is None
        else tuple(float(value) for value in cast(list[float], confidence_raw))
    )
    return DepthField(
        depth_field_id=DepthFieldId(f"depth:{owner_token}:{token}"),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=tuple(float(value) for value in cast(list[float], field["depth_values"])),
        validity=tuple(bool(value) for value in cast(list[bool], field["validity"])),
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense_artifact(
    *,
    source: GeometrySolutionCandidate,
    entry: dict[str, Any],
    convention: str,
) -> DenseDepthArtifact:
    token = cast(str, entry["producer_token"])
    depth_fields = tuple(
        _depth_field(
            source=source,
            owner_token=token,
            field=cast(dict[str, Any], field),
            convention=convention,
        )
        for field in cast(list[dict[str, Any]], entry["fields"])
    )
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(cast(str, entry["artifact_id"])),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer(token),
        source_artifacts=source.source_artifacts,
    )


def _request() -> DenseDepthBenchmarkRequest:
    document = _fixture_document()
    source = _source_geometry()
    convention = cast(str, document["depth_value_convention"])
    reference = _dense_artifact(
        source=source,
        entry=cast(dict[str, Any], document["reference"]),
        convention=convention,
    )
    candidates = tuple(
        _dense_artifact(
            source=source,
            entry=cast(dict[str, Any], entry),
            convention=convention,
        )
        for entry in cast(list[dict[str, Any]], document["candidates"])
    )
    fixture_sha = hashlib.sha256(_FIXTURE_PATH.read_bytes()).hexdigest()
    return DenseDepthBenchmarkRequest(
        fixture=BenchmarkFixtureIdentity(
            fixture_id=BenchmarkFixtureId(cast(str, document["fixture_id"])),
            sha256=Sha256Digest(fixture_sha),
        ),
        hardware=HardwareRuntimeIdentity(
            sha256=Sha256Digest(hashlib.sha256(b"v2l15.6-controlled-cpu-locked-wre").hexdigest())
        ),
        reference=reference,
        candidates=candidates,
    )


def _metric_values(vector: MetricVector) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value for observation in vector.observations
    }


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


def _result_document(result: DenseDepthBenchmarkResult) -> dict[str, Any]:
    request = result.request
    candidates: list[dict[str, Any]] = []
    error_metric_names = {
        "geometry.dense_depth.reference_absolute_error_median",
        "geometry.dense_depth.reference_absolute_relative_error_median",
    }
    for item in result.candidates:
        observed_names = {
            observation.descriptor.name.value for observation in item.reference_metrics.observations
        }
        candidates.append(
            {
                "artifact": _artifact_record(item.candidate.artifact_ref),
                "producer": _producer_record(item.candidate.producer),
                "benchmark_record_id": item.benchmark_record.record_id.value,
                "support_fields": [
                    {
                        "depth_field_id": field.depth_field_id.value,
                        "camera_solution_id": field.camera_solution_id.value,
                        "observation_id": field.observation_id.value,
                        "total_pixel_count": field.total_pixel_count,
                        "valid_pixel_count": field.valid_pixel_count,
                        "hole_pixel_count": field.hole_pixel_count,
                        "valid_pixel_ratio": field.valid_pixel_ratio,
                        "confidence_available": field.confidence_available,
                        "valid_confidence_mean": field.valid_confidence_mean,
                    }
                    for field in item.coverage.field_coverages
                ],
                "support_metrics": _metric_vector_record(item.coverage.metrics),
                "reference_metrics": _metric_vector_record(item.reference_metrics),
                "benchmark_metrics": _metric_vector_record(item.benchmark_record.metrics),
                "omitted_reference_metrics": sorted(error_metric_names - observed_names),
            }
        )

    return {
        "schema_version": 1,
        "fixture": {
            "fixture_id": request.fixture.fixture_id.value,
            "sha256": request.fixture.sha256.value,
        },
        "hardware_runtime_sha256": request.hardware.sha256.value,
        "quality_mode": QualityMode.QUALITY.value,
        "reference": {
            "artifact": _artifact_record(request.reference.artifact_ref),
            "producer": _producer_record(request.reference.producer),
            "geometry_solution_id": request.reference.source_geometry.geometry_solution_id.value,
        },
        "candidates": candidates,
        "unavailable_dimensions": [
            "learned_dense_depth_runtime_execution",
            "production_dense_depth_backend_selection",
            "representative_solver_performance",
        ],
        "deferred_evidence": {
            "v2l15_2_real_cuda_patchmatch_execution": "deferred_unverified",
            "learned_depth_prior": "model_independent_boundary_only",
        },
        "descriptive_only": True,
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


def test_request_candidate_and_result_have_exact_frozen_shapes() -> None:
    request = _request()
    result = benchmark_dense_depth_candidates(request)
    candidate = result.candidates[0]

    assert tuple(item.name for item in fields(DenseDepthBenchmarkRequest)) == (
        "fixture",
        "hardware",
        "reference",
        "candidates",
    )
    assert tuple(item.name for item in fields(DenseDepthCandidateBenchmarkResult)) == (
        "candidate",
        "coverage",
        "reference_metrics",
        "benchmark_record",
    )
    assert tuple(item.name for item in fields(DenseDepthBenchmarkResult)) == (
        "request",
        "candidates",
    )

    with pytest.raises(FrozenInstanceError):
        request.candidates = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        candidate.reference_metrics = _metrics()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]


def test_request_rejects_wrong_types_order_duplicates_and_reference_identity() -> None:
    request = _request()
    first, second, _third = request.candidates

    with pytest.raises(TypeError, match="fixture"):
        replace(request, fixture=cast(Any, "fixture"))
    with pytest.raises(TypeError, match="hardware"):
        replace(request, hardware=cast(Any, "hardware"))
    with pytest.raises(TypeError, match="reference"):
        replace(request, reference=cast(Any, "reference"))
    with pytest.raises(TypeError, match="immutable tuple"):
        replace(request, candidates=cast(Any, list(request.candidates)))
    with pytest.raises(ValueError, match="at least two"):
        replace(request, candidates=(first,))
    with pytest.raises(ValueError, match="canonical"):
        replace(request, candidates=tuple(reversed(request.candidates)))
    with pytest.raises(ValueError, match="unique"):
        replace(request, candidates=(first, first))

    same_identity = replace(first, artifact_ref=request.reference.artifact_ref)
    with pytest.raises(ValueError, match="reference artifact"):
        replace(request, candidates=(same_identity, second))


def test_request_requires_exact_source_geometry_and_valid_reference_support() -> None:
    request = _request()
    first, second, _third = request.candidates
    foreign_source = replace(
        request.reference.source_geometry,
        producer=_producer("foreign-source"),
    )
    foreign_candidate = replace(first, source_geometry=foreign_source)

    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        replace(request, candidates=(foreign_candidate, second))

    invalid_reference_fields = tuple(
        replace(
            field,
            depth_values=(0.0, 0.0, 0.0, 0.0),
            validity=(False, False, False, False),
            confidence=None,
        )
        for field in request.reference.depth_fields
    )
    invalid_reference = replace(
        request.reference,
        depth_fields=invalid_reference_fields,
    )
    with pytest.raises(ValueError, match="valid depth support"):
        replace(request, reference=invalid_reference)


def test_overlapping_field_semantics_fail_closed_without_conversion() -> None:
    request = _request()
    first, second, _third = request.candidates
    changed_field = replace(
        first.depth_fields[0],
        depth_value_convention=DepthValueConventionName("euclidean-ray"),
    )
    changed_candidate = replace(
        first,
        depth_fields=(changed_field, *first.depth_fields[1:]),
    )
    changed_request = replace(
        request,
        candidates=(changed_candidate, second),
    )

    with pytest.raises(ValueError, match="DepthValueConventionName"):
        benchmark_dense_depth_candidates(changed_request)


def test_controlled_reference_metrics_use_exact_valid_support_and_medians() -> None:
    result = benchmark_dense_depth_candidates(_request())
    by_id = {item.candidate.artifact_ref.artifact_id.value: item for item in result.candidates}

    full = _metric_values(by_id["dense:candidate-full"].reference_metrics)
    assert full["geometry.dense_depth.reference_valid_coverage_ratio"] == 1.0
    assert full["geometry.dense_depth.reference_absolute_error_median"] == 0.5
    assert full["geometry.dense_depth.reference_absolute_relative_error_median"] == pytest.approx(
        0.25
    )

    partial = _metric_values(by_id["dense:candidate-partial"].reference_metrics)
    assert partial["geometry.dense_depth.reference_valid_coverage_ratio"] == pytest.approx(
        1.0 / 3.0
    )
    assert partial["geometry.dense_depth.reference_absolute_error_median"] == 0.5
    assert partial[
        "geometry.dense_depth.reference_absolute_relative_error_median"
    ] == pytest.approx(5.0 / 24.0)

    zero = _metric_values(by_id["dense:candidate-zero-overlap"].reference_metrics)
    assert zero == {
        "geometry.dense_depth.reference_valid_coverage_ratio": 0.0,
    }


def test_v2l15_5_support_metrics_and_provenance_are_retained_unchanged() -> None:
    request = _request()
    result = benchmark_dense_depth_candidates(request)

    for item in result.candidates:
        expected = evaluate_dense_depth_coverage(item.candidate)
        assert item.coverage == expected
        record_by_name = {
            observation.descriptor.name.value: observation
            for observation in item.benchmark_record.metrics.observations
        }
        for support_observation in expected.metrics.observations:
            retained = record_by_name[support_observation.descriptor.name.value]
            assert retained == support_observation
            assert retained.provenance.evaluator == DENSE_DEPTH_COVERAGE_EVALUATOR
            assert retained.provenance.input_artifacts == (item.candidate.artifact_ref,)


def test_reference_metric_provenance_uses_evaluator_and_canonical_artifact_pair() -> None:
    request = _request()
    result = benchmark_dense_depth_candidates(request)

    for item in result.candidates:
        expected_pair = tuple(
            sorted(
                (item.candidate.artifact_ref, request.reference.artifact_ref),
                key=lambda ref: (ref.artifact_id.value, ref.artifact_kind.value),
            )
        )
        for observation in item.reference_metrics.observations:
            assert observation.provenance.evaluator == DENSE_DEPTH_BENCHMARK_EVALUATOR
            assert observation.provenance.input_artifacts == expected_pair


def test_confidence_is_informational_and_cannot_change_reference_error_values() -> None:
    request = _request()
    first, second, _third = request.candidates
    changed_fields = tuple(
        replace(
            field,
            confidence=(
                None
                if field.confidence is None
                else tuple(1.0 if supported else 0.0 for supported in field.validity)
            ),
        )
        for field in first.depth_fields
    )
    changed = replace(
        first,
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId("dense:candidate-full-confidence-alt"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        depth_fields=changed_fields,
        producer=_producer("candidate-full-confidence-alt"),
    )
    alternate_request = replace(request, candidates=(changed, second))

    original = benchmark_dense_depth_candidates(request).candidates[0]
    alternate = benchmark_dense_depth_candidates(alternate_request).candidates[0]

    assert _metric_values(original.reference_metrics) == _metric_values(alternate.reference_metrics)
    support_original = _metric_values(original.coverage.metrics)
    support_alternate = _metric_values(alternate.coverage.metrics)
    assert (
        support_original["geometry.dense_depth.valid_confidence_mean"]
        != support_alternate["geometry.dense_depth.valid_confidence_mean"]
    )


def test_benchmark_records_are_descriptive_quality_records_without_selection_semantics() -> None:
    request = _request()
    result = benchmark_dense_depth_candidates(request)

    for item in result.candidates:
        record = item.benchmark_record
        assert record.fixture == request.fixture
        assert record.hardware == request.hardware
        assert record.producer == item.candidate.producer
        assert record.quality_mode is QualityMode.QUALITY
        assert record.performance_evidence == ()
        assert record.comparison_baseline is None
        names = tuple(
            observation.descriptor.name.value for observation in record.metrics.observations
        )
        assert names == tuple(sorted(names))
        for attribute in (
            "winner",
            "rank",
            "score",
            "threshold",
            "decision",
            "route",
            "default",
            "retry",
            "fallback",
        ):
            assert not hasattr(record, attribute)


def test_candidate_and_result_records_reject_inconsistent_composition() -> None:
    request = _request()
    result = benchmark_dense_depth_candidates(request)
    first = result.candidates[0]

    with pytest.raises(ValueError, match="retain support and reference metrics"):
        replace(
            first,
            benchmark_record=replace(
                first.benchmark_record,
                metrics=first.reference_metrics,
            ),
        )
    with pytest.raises(ValueError, match="request order"):
        replace(result, candidates=tuple(reversed(result.candidates)))
    with pytest.raises(TypeError, match="request"):
        DenseDepthBenchmarkResult(
            request=cast(Any, "request"),
            candidates=result.candidates,
        )


def test_fixture_hash_and_retained_document_are_deterministic_and_nonselecting(
    tmp_path: Path,
) -> None:
    request = _request()
    first = benchmark_dense_depth_candidates(request)
    second = benchmark_dense_depth_candidates(request)
    assert first == second

    document = _result_document(first)
    _assert_no_forbidden_record_keys(document)
    assert document["fixture"]["sha256"] == _EXPECTED_FIXTURE_SHA256
    assert document["deferred_evidence"]["v2l15_2_real_cuda_patchmatch_execution"] == (
        "deferred_unverified"
    )
    assert document["descriptive_only"] is True
    candidates = cast(list[dict[str, Any]], document["candidates"])
    zero = next(
        item
        for item in candidates
        if item["artifact"]["artifact_id"] == "dense:candidate-zero-overlap"
    )
    assert zero["omitted_reference_metrics"] == [
        "geometry.dense_depth.reference_absolute_error_median",
        "geometry.dense_depth.reference_absolute_relative_error_median",
    ]

    encoded = json.dumps(document, indent=2, sort_keys=True) + "\n"
    output = tmp_path / "dense-depth-controlled.json"
    output.write_text(encoded, encoding="utf-8")
    assert json.loads(output.read_text(encoding="utf-8")) == document

    requested_output = os.environ.get("WRE_DENSE_DEPTH_CONTROLLED_BENCHMARK_OUTPUT")
    if requested_output:
        Path(requested_output).write_text(encoded, encoding="utf-8")


def test_workflow_is_cpu_only_and_retains_machine_readable_evidence() -> None:
    workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-24.04" in workflow
    assert "WRE_DENSE_DEPTH_CONTROLLED_BENCHMARK_OUTPUT" in workflow
    assert "v2l15-6-controlled-dense-depth-benchmark" in workflow
    assert "tests/test_dense_depth_benchmark.py" in workflow
    for forbidden in (
        "cuda",
        "nvidia",
        "huggingface",
        "model.safetensors",
        "torch.hub",
        "patch_match_stereo",
    ):
        assert forbidden not in workflow.lower()
