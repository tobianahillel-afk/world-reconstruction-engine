from __future__ import annotations

import hashlib
import json
import os
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any, cast

import pytest

import wre.reconstruction.dense_depth_benchmark as benchmark
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.benchmarks import (
    BenchmarkFixtureId,
    BenchmarkFixtureIdentity,
)
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
    DenseDepthBenchmarkCandidateResult,
    DenseDepthBenchmarkRequest,
    DenseDepthBenchmarkResult,
    benchmark_dense_depth_candidates,
)
from wre.reconstruction.dense_depth_coverage import evaluate_dense_depth_coverage
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "synthetic"
    / "dense-depth-controlled"
    / "fixture.json"
)
_FIXTURE_SHA256 = "91546e57df846da3ad8d6011c40476f875588ac5e629eb10b4f8ec33d84c98e4"


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


def _source_artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact:source:dense-depth-controlled"),
        artifact_kind=ArtifactKind("geometry.solution"),
    )


def _fixture_document() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(_FIXTURE_PATH.read_text(encoding="utf-8")))


def _fixture_identity() -> BenchmarkFixtureIdentity:
    fixture_bytes = _FIXTURE_PATH.read_bytes()
    digest = hashlib.sha256(fixture_bytes).hexdigest()
    assert digest == _FIXTURE_SHA256
    return BenchmarkFixtureIdentity(
        fixture_id=BenchmarkFixtureId("fixture.dense-depth-controlled.v1"),
        sha256=Sha256Digest(digest),
    )


def _hardware() -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(
        sha256=Sha256Digest(hashlib.sha256(b"dense-depth-controlled-cpu").hexdigest())
    )


def _camera(entry: dict[str, Any], frame: LocalFrameId) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(str(entry["solution_id"])),
        observation_id=ObservationId(str(entry["observation_id"])),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(
            width_px=int(entry["width_px"]),
            height_px=int(entry["height_px"]),
        ),
        intrinsic_parameters=(2.0, 2.0, 1.0, 1.0),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _source_geometry(document: dict[str, Any]) -> GeometrySolutionCandidate:
    source = cast(dict[str, Any], document["source_geometry"])
    frame = LocalFrameId(str(source["local_frame_id"]))
    cameras = tuple(
        _camera(cast(dict[str, Any], item), frame)
        for item in cast(list[dict[str, Any]], source["cameras"])
    )
    point_map = PointMap(
        point_map_id=PointMapId("points:dense-depth-controlled"),
        local_frame_id=frame,
        source_observation_ids=tuple(camera.observation_id for camera in cameras),
        positions_xyz=((0.0, 0.0, 1.0),),
        confidence=None,
        metrics=_metrics(),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(str(source["geometry_solution_id"])),
        local_frame_id=frame,
        scale_status=GeometryScaleStatus.METRIC,
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
        source_artifacts=(_source_artifact(),),
    )


def _depth_field(
    entry: dict[str, Any],
    cameras_by_id: dict[str, CameraSolution],
) -> DepthField:
    camera = cameras_by_id[str(entry["camera_solution_id"])]
    confidence_value = entry["confidence"]
    confidence = (
        None
        if confidence_value is None
        else tuple(float(item) for item in cast(list[float], confidence_value))
    )
    return DepthField(
        depth_field_id=DepthFieldId(str(entry["depth_field_id"])),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName(str(entry["depth_value_convention"])),
        depth_values=tuple(float(item) for item in cast(list[float], entry["depth_values"])),
        validity=tuple(bool(item) for item in cast(list[bool], entry["validity"])),
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense_artifact(
    entry: dict[str, Any],
    source: GeometrySolutionCandidate,
) -> DenseDepthArtifact:
    cameras_by_id = {camera.solution_id.value: camera for camera in source.camera_solutions}
    depth_fields = tuple(
        _depth_field(cast(dict[str, Any], item), cameras_by_id)
        for item in cast(list[dict[str, Any]], entry["fields"])
    )
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(str(entry["artifact_id"])),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer(str(entry["producer_token"])),
        source_artifacts=source.source_artifacts,
    )


def _controlled_request() -> DenseDepthBenchmarkRequest:
    document = _fixture_document()
    source = _source_geometry(document)
    reference = _dense_artifact(
        cast(dict[str, Any], document["reference"]),
        source,
    )
    candidates = tuple(
        _dense_artifact(cast(dict[str, Any], item), source)
        for item in cast(list[dict[str, Any]], document["candidates"])
    )
    return DenseDepthBenchmarkRequest(
        fixture=_fixture_identity(),
        hardware=_hardware(),
        reference_depth=reference,
        candidates=candidates,
    )


def _metric_values(vector: MetricVector) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value for observation in vector.observations
    }


def _artifact_document(ref: ArtifactRef) -> dict[str, str]:
    return {
        "artifact_id": ref.artifact_id.value,
        "artifact_kind": ref.artifact_kind.value,
    }


def _producer_document(value: ArtifactProducerIdentity) -> dict[str, Any]:
    return {
        "implementation": value.producer.implementation,
        "version": value.producer.version,
        "revision": value.producer.revision,
        "configuration_sha256": value.configuration.sha256.value,
    }


def _result_document(result: DenseDepthBenchmarkResult) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixture": {
            "fixture_id": result.request.fixture.fixture_id.value,
            "sha256": result.request.fixture.sha256.value,
        },
        "hardware_sha256": result.request.hardware.sha256.value,
        "reference_artifact": _artifact_document(result.request.reference_depth.artifact_ref),
        "candidates": [
            {
                "candidate_artifact": _artifact_document(item.candidate.artifact_ref),
                "producer": _producer_document(item.candidate.producer),
                "benchmark_record_id": item.benchmark_record.record_id.value,
                "quality_mode": item.benchmark_record.quality_mode.value,
                "reference_valid_pixel_count": item.reference_valid_pixel_count,
                "jointly_valid_pixel_count": item.jointly_valid_pixel_count,
                "support_metrics": _metric_values(item.coverage.metrics),
                "reference_metrics": _metric_values(item.reference_metrics),
            }
            for item in result.candidates
        ],
        "deferred_evidence": {
            "v2l15_2_real_cuda_patchmatch": "deferred_unverified",
        },
        "unavailable_dimensions": [
            "cross_depth_convention_conversion",
            "confidence_truth_calibration",
            "real_cuda_patchmatch_execution",
        ],
    }


def _assert_no_selection_fields(value: object) -> None:
    forbidden = {
        "winner",
        "rank",
        "score",
        "preferred_route",
        "threshold",
        "default",
        "retry",
        "fallback",
        "quality_decision",
        "shipping_promotion",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_no_selection_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_selection_fields(child)


def test_request_candidate_result_and_result_have_exact_frozen_shapes() -> None:
    request = _controlled_request()
    result = benchmark_dense_depth_candidates(request)
    candidate = result.candidates[0]

    assert tuple(item.name for item in fields(DenseDepthBenchmarkRequest)) == (
        "fixture",
        "hardware",
        "reference_depth",
        "candidates",
    )
    assert tuple(item.name for item in fields(DenseDepthBenchmarkCandidateResult)) == (
        "candidate",
        "coverage",
        "reference_valid_pixel_count",
        "jointly_valid_pixel_count",
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
        candidate.jointly_valid_pixel_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.candidates = ()  # type: ignore[misc]


def test_request_rejects_wrong_type_order_duplicates_and_reference_identity() -> None:
    request = _controlled_request()
    complete, partial, zero = request.candidates

    with pytest.raises(TypeError, match="fixture"):
        DenseDepthBenchmarkRequest(
            fixture=cast(Any, "fixture"),
            hardware=request.hardware,
            reference_depth=request.reference_depth,
            candidates=request.candidates,
        )
    with pytest.raises(ValueError, match="at least two"):
        replace(request, candidates=(complete,))
    with pytest.raises(ValueError, match="canonical"):
        replace(request, candidates=(zero, partial, complete))
    with pytest.raises(ValueError, match="unique"):
        replace(request, candidates=(complete, complete))
    duplicate_reference = replace(
        complete,
        artifact_ref=request.reference_depth.artifact_ref,
    )
    with pytest.raises(ValueError, match="reference ArtifactRef"):
        replace(request, candidates=(partial, duplicate_reference))


def test_request_rejects_different_source_geometry_without_alignment() -> None:
    request = _controlled_request()
    candidate = request.candidates[0]
    source = candidate.source_geometry
    other_geometry = replace(
        source.geometry_solution,
        geometry_solution_id=GeometrySolutionId("geometry:dense-depth-other"),
    )
    other_source = GeometrySolutionCandidate(
        geometry_solution=other_geometry,
        camera_solutions=source.camera_solutions,
        depth_fields=source.depth_fields,
        point_maps=source.point_maps,
        producer=source.producer,
        source_artifacts=source.source_artifacts,
    )
    changed = replace(candidate, source_geometry=other_source)

    with pytest.raises(ValueError, match="exact same"):
        replace(request, candidates=(changed, request.candidates[1]))


def test_request_rejects_overlapping_depth_convention_mismatch() -> None:
    request = _controlled_request()
    complete = request.candidates[0]
    first = complete.depth_fields[0]
    mismatched = replace(
        first,
        depth_value_convention=DepthValueConventionName("ray-distance"),
    )
    changed = replace(
        complete,
        depth_fields=(mismatched, *complete.depth_fields[1:]),
    )

    with pytest.raises(ValueError, match="DepthValueConventionName"):
        replace(request, candidates=(changed, request.candidates[1]))


def test_reference_must_have_at_least_one_valid_pixel() -> None:
    request = _controlled_request()
    reference = request.reference_depth
    fields_without_support = tuple(
        replace(
            field,
            depth_values=tuple(0.0 for _ in field.depth_values),
            validity=tuple(False for _ in field.validity),
        )
        for field in reference.depth_fields
    )
    empty_reference = replace(reference, depth_fields=fields_without_support)

    with pytest.raises(ValueError, match="at least one valid"):
        replace(request, reference_depth=empty_reference)


def test_controlled_fixture_reports_exact_reference_and_support_metrics() -> None:
    result = benchmark_dense_depth_candidates(_controlled_request())
    complete, partial, zero = result.candidates

    complete_reference = _metric_values(complete.reference_metrics)
    assert complete.reference_valid_pixel_count == 8
    assert complete.jointly_valid_pixel_count == 8
    assert complete_reference["geometry.dense_depth.reference_valid_coverage_ratio"] == 1.0
    assert complete_reference[
        "geometry.dense_depth.reference_absolute_error_median"
    ] == pytest.approx(0.15)
    assert complete_reference[
        "geometry.dense_depth.reference_absolute_relative_error_median"
    ] == pytest.approx(0.058333333333333334)

    complete_support = _metric_values(complete.coverage.metrics)
    assert complete_support["geometry.dense_depth.camera_support_ratio"] == 1.0
    assert complete_support["geometry.dense_depth.valid_pixel_ratio"] == 1.0
    assert complete_support["geometry.dense_depth.hole_pixel_ratio"] == 0.0
    assert complete_support["geometry.dense_depth.confidence_field_ratio"] == 0.5
    assert complete_support["geometry.dense_depth.valid_confidence_mean"] == 0.75

    partial_reference = _metric_values(partial.reference_metrics)
    assert partial.reference_valid_pixel_count == 8
    assert partial.jointly_valid_pixel_count == 2
    assert partial_reference["geometry.dense_depth.reference_valid_coverage_ratio"] == 0.25
    assert partial_reference[
        "geometry.dense_depth.reference_absolute_error_median"
    ] == pytest.approx(0.25)
    assert partial_reference[
        "geometry.dense_depth.reference_absolute_relative_error_median"
    ] == pytest.approx(1.0 / 12.0)

    partial_support = _metric_values(partial.coverage.metrics)
    assert partial_support["geometry.dense_depth.camera_support_ratio"] == 0.5
    assert partial_support["geometry.dense_depth.valid_pixel_ratio"] == 0.5
    assert partial_support["geometry.dense_depth.hole_pixel_ratio"] == 0.5
    assert partial_support["geometry.dense_depth.confidence_field_ratio"] == 1.0
    assert partial_support["geometry.dense_depth.valid_confidence_mean"] == pytest.approx(0.7)

    zero_reference = _metric_values(zero.reference_metrics)
    assert zero.reference_valid_pixel_count == 8
    assert zero.jointly_valid_pixel_count == 0
    assert zero_reference == {
        "geometry.dense_depth.reference_valid_coverage_ratio": 0.0,
    }

    zero_support = _metric_values(zero.coverage.metrics)
    assert zero_support["geometry.dense_depth.camera_support_ratio"] == 0.5
    assert zero_support["geometry.dense_depth.valid_pixel_ratio"] == 0.0
    assert zero_support["geometry.dense_depth.hole_pixel_ratio"] == 1.0
    assert zero_support["geometry.dense_depth.confidence_field_ratio"] == 0.0


def test_support_metric_observations_and_provenance_are_retained_unchanged() -> None:
    request = _controlled_request()
    result = benchmark_dense_depth_candidates(request)
    item = result.candidates[0]
    expected = evaluate_dense_depth_coverage(item.candidate)

    assert item.coverage == expected
    coverage_by_name = {
        observation.descriptor.name.value: observation
        for observation in item.coverage.metrics.observations
    }
    record_by_name = {
        observation.descriptor.name.value: observation
        for observation in item.benchmark_record.metrics.observations
    }
    for name, observation in coverage_by_name.items():
        assert record_by_name[name] is observation

    reference_names = {
        observation.descriptor.name.value for observation in item.reference_metrics.observations
    }
    for observation in item.benchmark_record.metrics.observations:
        if observation.descriptor.name.value not in reference_names:
            continue
        assert observation.provenance.evaluator == benchmark.DENSE_DEPTH_BENCHMARK_EVALUATOR
        assert observation.provenance.input_artifacts == tuple(
            sorted(
                (
                    item.candidate.artifact_ref,
                    request.reference_depth.artifact_ref,
                ),
                key=lambda ref: (ref.artifact_id.value, ref.artifact_kind.value),
            )
        )


def test_confidence_changes_do_not_change_reference_error_values() -> None:
    request = _controlled_request()
    complete = request.candidates[0]
    first = complete.depth_fields[0]
    changed_first = replace(
        first,
        confidence=(0.1, 0.2, 0.3, 0.4),
    )
    variant = replace(
        complete,
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId("dense:candidate-confidence-variant"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        depth_fields=(changed_first, *complete.depth_fields[1:]),
        producer=_producer("candidate-confidence-variant"),
    )
    changed_request = replace(
        request,
        candidates=(complete, variant),
    )
    result = benchmark_dense_depth_candidates(changed_request)

    first_values = _metric_values(result.candidates[0].reference_metrics)
    second_values = _metric_values(result.candidates[1].reference_metrics)
    assert first_values == second_values

    first_support = _metric_values(result.candidates[0].coverage.metrics)
    second_support = _metric_values(result.candidates[1].coverage.metrics)
    assert first_support["geometry.dense_depth.valid_confidence_mean"] == 0.75
    assert second_support["geometry.dense_depth.valid_confidence_mean"] == pytest.approx(0.25)


def test_benchmark_records_are_deterministic_quality_records_without_selection_semantics() -> None:
    request = _controlled_request()
    first = benchmark_dense_depth_candidates(request)
    second = benchmark_dense_depth_candidates(request)

    assert first == second
    assert tuple(item.candidate for item in first.candidates) == request.candidates
    record_ids = tuple(item.benchmark_record.record_id for item in first.candidates)
    assert len(record_ids) == len(set(record_ids))

    for item in first.candidates:
        record = item.benchmark_record
        assert record.fixture == request.fixture
        assert record.producer == item.candidate.producer
        assert record.quality_mode is QualityMode.QUALITY
        assert record.hardware == request.hardware
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
            "preferred_route",
            "threshold",
            "decision",
            "route",
            "default",
            "retry",
            "fallback",
            "shipping_promotion",
        ):
            assert not hasattr(record, attribute)


def test_candidate_result_rejects_invalid_count_and_record_relationships() -> None:
    result = benchmark_dense_depth_candidates(_controlled_request())
    item = result.candidates[0]

    with pytest.raises(ValueError, match="non-negative"):
        replace(item, jointly_valid_pixel_count=-1)
    with pytest.raises(ValueError, match="cannot exceed"):
        replace(
            item,
            jointly_valid_pixel_count=item.reference_valid_pixel_count + 1,
        )
    with pytest.raises(ValueError, match="producer"):
        replace(
            item,
            benchmark_record=replace(
                item.benchmark_record,
                producer=_producer("wrong-producer"),
            ),
        )


def test_source_and_depth_evidence_remain_immutable_after_benchmark() -> None:
    request = _controlled_request()
    reference_fields = request.reference_depth.depth_fields
    candidate_fields = tuple(item.depth_fields for item in request.candidates)
    candidate_values = tuple(
        tuple(field.depth_values for field in item.depth_fields) for item in request.candidates
    )

    result = benchmark_dense_depth_candidates(request)

    assert result.request is request
    assert request.reference_depth.depth_fields is reference_fields
    assert tuple(item.depth_fields for item in request.candidates) == candidate_fields
    assert (
        tuple(
            tuple(field.depth_values for field in item.depth_fields) for item in request.candidates
        )
        == candidate_values
    )
    assert all(item.coverage.source_depth is item.candidate for item in result.candidates)


def test_fixture_identity_and_retained_document_are_deterministic_and_nonselective() -> None:
    request = _controlled_request()
    assert request.fixture.fixture_id == BenchmarkFixtureId("fixture.dense-depth-controlled.v1")
    assert request.fixture.sha256 == Sha256Digest(_FIXTURE_SHA256)

    document = _result_document(benchmark_dense_depth_candidates(request))
    assert document["deferred_evidence"] == {
        "v2l15_2_real_cuda_patchmatch": "deferred_unverified",
    }
    assert document["unavailable_dimensions"] == [
        "cross_depth_convention_conversion",
        "confidence_truth_calibration",
        "real_cuda_patchmatch_execution",
    ]
    assert len(cast(list[Any], document["candidates"])) == 3
    _assert_no_selection_fields(document)

    encoded = (
        json.dumps(
            document,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    assert encoded == (
        json.dumps(
            _result_document(benchmark_dense_depth_candidates(_controlled_request())),
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


@pytest.mark.skipif(
    os.environ.get("WRE_DENSE_DEPTH_CONTROLLED_BENCHMARK") != "1",
    reason="retained dense-depth benchmark output is enabled only in its dedicated lane",
)
def test_retained_controlled_dense_depth_benchmark() -> None:
    output = Path(os.environ["WRE_DENSE_DEPTH_CONTROLLED_BENCHMARK_OUTPUT"])
    result = benchmark_dense_depth_candidates(_controlled_request())
    document = _result_document(result)
    _assert_no_selection_fields(document)
    output.write_text(
        json.dumps(
            document,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def test_module_surface_has_no_solver_filling_selection_or_surface_dependencies() -> None:
    forbidden = {
        "numpy",
        "torch",
        "pycolmap",
        "PatchMatch",
        "StereoFusion",
        "DepthConsistencyFusionAdapter",
        "LearnedDepthPriorAdapter",
        "SurfaceModel",
        "TSDF",
        "SDF",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Scheduler",
        "MasterScene",
        "RuntimeScene",
    }
    assert forbidden.isdisjoint(vars(benchmark))
    for attribute in (
        "fill",
        "inpaint",
        "interpolate",
        "fuse",
        "select",
        "rank",
        "decide",
        "route",
    ):
        assert not hasattr(DenseDepthBenchmarkResult, attribute)


def test_benchmark_rejects_wrong_request_type() -> None:
    with pytest.raises(TypeError, match="DenseDepthBenchmarkRequest"):
        benchmark_dense_depth_candidates(cast(Any, "request"))
