from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.benchmarks import (
    BenchmarkFixtureIdentity,
    BenchmarkRecord,
    BenchmarkRecordId,
)
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.metrics import (
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
)
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.quality import QualityMode
from wre.domain.runs import ProducerRef
from wre.reconstruction.dense_depth import DenseDepthArtifact
from wre.reconstruction.dense_depth_coverage import (
    DenseDepthCoverageReport,
    evaluate_dense_depth_coverage,
)

DENSE_DEPTH_BENCHMARK_IMPLEMENTATION = "wre.reconstruction.dense_depth_benchmark"
DENSE_DEPTH_BENCHMARK_VERSION = "1"
DENSE_DEPTH_BENCHMARK_REVISION = "v2l15.6"

_EVALUATOR_CONFIGURATION_DOCUMENT = {
    "schema_version": 1,
    "reference_coverage_denominator": "valid_reference_pixels",
    "comparison_support": "jointly_valid_pixels_only",
    "absolute_error": "median_absolute_depth_error_unchanged_depth_units",
    "absolute_relative_error": "median_absolute_error_over_positive_reference_depth",
    "confidence_comparison": "none",
    "depth_conversion": "none",
    "selection": "none",
}
_EVALUATOR_CONFIGURATION_BYTES = json.dumps(
    _EVALUATOR_CONFIGURATION_DOCUMENT,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
DENSE_DEPTH_BENCHMARK_EVALUATOR = ArtifactProducerIdentity(
    producer=ProducerRef(
        implementation=DENSE_DEPTH_BENCHMARK_IMPLEMENTATION,
        version=DENSE_DEPTH_BENCHMARK_VERSION,
        revision=DENSE_DEPTH_BENCHMARK_REVISION,
    ),
    configuration=ConfigurationIdentity(
        sha256=Sha256Digest(hashlib.sha256(_EVALUATOR_CONFIGURATION_BYTES).hexdigest())
    ),
)

_REFERENCE_DIMENSION = MetricDimension("geometry.dense_depth.reference")

REFERENCE_VALID_COVERAGE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_valid_coverage_ratio"),
    dimension=_REFERENCE_DIMENSION,
    unit=MetricUnit("ratio"),
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("valid_reference_pixel_ratio"),
)
REFERENCE_ABSOLUTE_ERROR_MEDIAN_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_absolute_error_median"),
    dimension=_REFERENCE_DIMENSION,
    unit=MetricUnit("depth_units"),
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("median_joint_valid"),
)
REFERENCE_ABSOLUTE_RELATIVE_ERROR_MEDIAN_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_absolute_relative_error_median"),
    dimension=_REFERENCE_DIMENSION,
    unit=MetricUnit("ratio"),
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("median_joint_valid"),
)


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _canonical_artifact_pair(
    left: ArtifactRef,
    right: ArtifactRef,
) -> tuple[ArtifactRef, ArtifactRef]:
    return tuple(sorted((left, right), key=_artifact_key))  # type: ignore[return-value]


def _validate_shared_source_geometry(
    reference_depth: DenseDepthArtifact,
    candidates: tuple[DenseDepthArtifact, ...],
) -> None:
    for candidate in candidates:
        if candidate.source_geometry != reference_depth.source_geometry:
            raise ValueError(
                "dense depth benchmark candidate and reference must use the exact same "
                "source GeometrySolutionCandidate"
            )


def _validate_overlapping_field_semantics(
    reference_depth: DenseDepthArtifact,
    candidate: DenseDepthArtifact,
) -> None:
    reference_by_camera = {
        field.camera_solution_id: field for field in reference_depth.depth_fields
    }
    for candidate_field in candidate.depth_fields:
        reference_field = reference_by_camera.get(candidate_field.camera_solution_id)
        if reference_field is None:
            continue
        if candidate_field.observation_id != reference_field.observation_id:
            raise ValueError(
                "dense depth benchmark overlapping fields must preserve ObservationId"
            )
        if candidate_field.dimensions != reference_field.dimensions:
            raise ValueError(
                "dense depth benchmark overlapping fields must preserve ImageDimensions"
            )
        if candidate_field.depth_value_convention != reference_field.depth_value_convention:
            raise ValueError(
                "dense depth benchmark overlapping fields must preserve "
                "DepthValueConventionName"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class DenseDepthBenchmarkRequest:
    fixture: BenchmarkFixtureIdentity
    hardware: HardwareRuntimeIdentity
    reference_depth: DenseDepthArtifact
    candidates: tuple[DenseDepthArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, BenchmarkFixtureIdentity):
            raise TypeError("dense_depth_benchmark.fixture must be BenchmarkFixtureIdentity")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("dense_depth_benchmark.hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.reference_depth, DenseDepthArtifact):
            raise TypeError("dense_depth_benchmark.reference_depth must be DenseDepthArtifact")
        if not isinstance(self.candidates, tuple):
            raise TypeError("dense_depth_benchmark.candidates must be an immutable tuple")
        if len(self.candidates) < 2:
            raise ValueError("dense_depth_benchmark requires at least two candidates")
        if any(not isinstance(item, DenseDepthArtifact) for item in self.candidates):
            raise TypeError(
                "dense_depth_benchmark.candidates members must be DenseDepthArtifact"
            )

        candidate_keys = tuple(_artifact_key(item.artifact_ref) for item in self.candidates)
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("dense_depth_benchmark candidate ArtifactRefs must be unique")
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError(
                "dense_depth_benchmark candidates must use canonical ArtifactId/ArtifactKind order"
            )
        reference_key = _artifact_key(self.reference_depth.artifact_ref)
        if reference_key in set(candidate_keys):
            raise ValueError(
                "dense_depth_benchmark reference ArtifactRef must differ from every candidate"
            )

        _validate_shared_source_geometry(self.reference_depth, self.candidates)
        for candidate in self.candidates:
            _validate_overlapping_field_semantics(self.reference_depth, candidate)

        reference_valid_pixels = sum(
            sum(field.validity) for field in self.reference_depth.depth_fields
        )
        if reference_valid_pixels <= 0:
            raise ValueError(
                "dense_depth_benchmark reference must contain at least one valid depth pixel"
            )


@dataclass(frozen=True, slots=True)
class DenseDepthBenchmarkCandidateResult:
    candidate: DenseDepthArtifact
    coverage: DenseDepthCoverageReport
    reference_valid_pixel_count: int
    jointly_valid_pixel_count: int
    reference_metrics: MetricVector
    benchmark_record: BenchmarkRecord

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DenseDepthArtifact):
            raise TypeError("dense_depth_benchmark_candidate.candidate must be DenseDepthArtifact")
        if not isinstance(self.coverage, DenseDepthCoverageReport):
            raise TypeError(
                "dense_depth_benchmark_candidate.coverage must be DenseDepthCoverageReport"
            )
        if self.coverage.source_depth is not self.candidate:
            raise ValueError(
                "dense_depth_benchmark_candidate.coverage must retain the exact candidate object"
            )
        for field_name, value in (
            ("reference_valid_pixel_count", self.reference_valid_pixel_count),
            ("jointly_valid_pixel_count", self.jointly_valid_pixel_count),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"dense_depth_benchmark_candidate.{field_name} must be non-negative int"
                )
        if self.reference_valid_pixel_count <= 0:
            raise ValueError(
                "dense_depth_benchmark_candidate.reference_valid_pixel_count must be positive"
            )
        if self.jointly_valid_pixel_count > self.reference_valid_pixel_count:
            raise ValueError(
                "dense_depth_benchmark_candidate jointly valid pixels cannot exceed "
                "reference valid pixels"
            )
        if not isinstance(self.reference_metrics, MetricVector):
            raise TypeError(
                "dense_depth_benchmark_candidate.reference_metrics must be MetricVector"
            )
        if not isinstance(self.benchmark_record, BenchmarkRecord):
            raise TypeError(
                "dense_depth_benchmark_candidate.benchmark_record must be BenchmarkRecord"
            )
        if self.benchmark_record.producer != self.candidate.producer:
            raise ValueError(
                "dense_depth_benchmark_candidate benchmark producer must match candidate producer"
            )
        if self.benchmark_record.quality_mode is not QualityMode.QUALITY:
            raise ValueError(
                "dense_depth_benchmark_candidate benchmark quality mode must be QUALITY"
            )


@dataclass(frozen=True, slots=True)
class DenseDepthBenchmarkResult:
    request: DenseDepthBenchmarkRequest
    candidates: tuple[DenseDepthBenchmarkCandidateResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, DenseDepthBenchmarkRequest):
            raise TypeError("dense_depth_benchmark_result.request must be DenseDepthBenchmarkRequest")
        if not isinstance(self.candidates, tuple):
            raise TypeError(
                "dense_depth_benchmark_result.candidates must be an immutable tuple"
            )
        if any(
            not isinstance(item, DenseDepthBenchmarkCandidateResult) for item in self.candidates
        ):
            raise TypeError(
                "dense_depth_benchmark_result.candidates members must be "
                "DenseDepthBenchmarkCandidateResult"
            )
        if tuple(item.candidate for item in self.candidates) != self.request.candidates:
            raise ValueError(
                "dense_depth_benchmark_result candidates must exactly match request order"
            )
        if any(
            item.benchmark_record.fixture != self.request.fixture for item in self.candidates
        ):
            raise ValueError(
                "dense_depth_benchmark_result benchmark records must use request fixture"
            )
        if any(
            item.benchmark_record.hardware != self.request.hardware for item in self.candidates
        ):
            raise ValueError(
                "dense_depth_benchmark_result benchmark records must use request hardware"
            )


def _reference_metric_provenance(
    candidate: DenseDepthArtifact,
    reference_depth: DenseDepthArtifact,
) -> MetricProvenance:
    return MetricProvenance(
        evaluator=DENSE_DEPTH_BENCHMARK_EVALUATOR,
        input_artifacts=_canonical_artifact_pair(
            candidate.artifact_ref,
            reference_depth.artifact_ref,
        ),
    )


def _reference_metrics(
    candidate: DenseDepthArtifact,
    reference_depth: DenseDepthArtifact,
) -> tuple[int, int, MetricVector]:
    reference_by_camera = {
        field.camera_solution_id: field for field in reference_depth.depth_fields
    }
    candidate_by_camera = {
        field.camera_solution_id: field for field in candidate.depth_fields
    }

    reference_valid_pixel_count = sum(
        sum(field.validity) for field in reference_depth.depth_fields
    )
    jointly_valid_pixel_count = 0
    absolute_errors: list[float] = []
    absolute_relative_errors: list[float] = []

    for camera_solution_id, reference_field in reference_by_camera.items():
        candidate_field = candidate_by_camera.get(camera_solution_id)
        if candidate_field is None:
            continue

        for candidate_value, candidate_valid, reference_value, reference_valid in zip(
            candidate_field.depth_values,
            candidate_field.validity,
            reference_field.depth_values,
            reference_field.validity,
            strict=True,
        ):
            if not (candidate_valid and reference_valid):
                continue
            jointly_valid_pixel_count += 1
            absolute_error = abs(candidate_value - reference_value)
            absolute_errors.append(absolute_error)
            absolute_relative_errors.append(absolute_error / reference_value)

    provenance = _reference_metric_provenance(candidate, reference_depth)
    observations = [
        MetricObservation(
            descriptor=REFERENCE_VALID_COVERAGE_RATIO_DESCRIPTOR,
            value=float(jointly_valid_pixel_count / reference_valid_pixel_count),
            provenance=provenance,
        )
    ]
    if absolute_errors:
        observations.extend(
            (
                MetricObservation(
                    descriptor=REFERENCE_ABSOLUTE_ERROR_MEDIAN_DESCRIPTOR,
                    value=float(statistics.median(absolute_errors)),
                    provenance=provenance,
                ),
                MetricObservation(
                    descriptor=REFERENCE_ABSOLUTE_RELATIVE_ERROR_MEDIAN_DESCRIPTOR,
                    value=float(statistics.median(absolute_relative_errors)),
                    provenance=provenance,
                ),
            )
        )
    observations.sort(key=lambda item: item.descriptor.name.value)
    return (
        reference_valid_pixel_count,
        jointly_valid_pixel_count,
        MetricVector(observations=tuple(observations)),
    )


def _record_id(
    request: DenseDepthBenchmarkRequest,
    candidate: DenseDepthArtifact,
) -> BenchmarkRecordId:
    document = {
        "schema_version": 1,
        "fixture_id": request.fixture.fixture_id.value,
        "fixture_sha256": request.fixture.sha256.value,
        "hardware_sha256": request.hardware.sha256.value,
        "reference_artifact": {
            "artifact_id": request.reference_depth.artifact_ref.artifact_id.value,
            "artifact_kind": request.reference_depth.artifact_ref.artifact_kind.value,
        },
        "candidate_artifact": {
            "artifact_id": candidate.artifact_ref.artifact_id.value,
            "artifact_kind": candidate.artifact_ref.artifact_kind.value,
        },
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return BenchmarkRecordId(f"benchmark:dense-depth:{digest}")


def _combined_metrics(
    coverage: DenseDepthCoverageReport,
    reference_metrics: MetricVector,
) -> MetricVector:
    observations = (
        *coverage.metrics.observations,
        *reference_metrics.observations,
    )
    return MetricVector(
        observations=tuple(
            sorted(observations, key=lambda item: item.descriptor.name.value)
        )
    )


def benchmark_dense_depth_candidates(
    request: DenseDepthBenchmarkRequest,
) -> DenseDepthBenchmarkResult:
    if not isinstance(request, DenseDepthBenchmarkRequest):
        raise TypeError("request must be DenseDepthBenchmarkRequest")

    results: list[DenseDepthBenchmarkCandidateResult] = []
    for candidate in request.candidates:
        coverage = evaluate_dense_depth_coverage(candidate)
        (
            reference_valid_pixel_count,
            jointly_valid_pixel_count,
            reference_metrics,
        ) = _reference_metrics(candidate, request.reference_depth)
        benchmark_record = BenchmarkRecord(
            record_id=_record_id(request, candidate),
            fixture=request.fixture,
            producer=candidate.producer,
            quality_mode=QualityMode.QUALITY,
            hardware=request.hardware,
            metrics=_combined_metrics(coverage, reference_metrics),
            performance_evidence=(),
            comparison_baseline=None,
        )
        results.append(
            DenseDepthBenchmarkCandidateResult(
                candidate=candidate,
                coverage=coverage,
                reference_valid_pixel_count=reference_valid_pixel_count,
                jointly_valid_pixel_count=jointly_valid_pixel_count,
                reference_metrics=reference_metrics,
                benchmark_record=benchmark_record,
            )
        )

    return DenseDepthBenchmarkResult(
        request=request,
        candidates=tuple(results),
    )
