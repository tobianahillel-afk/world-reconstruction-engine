from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from statistics import median

from wre.domain.artifacts import ArtifactRef
from wre.domain.benchmarks import (
    BenchmarkFixtureIdentity,
    BenchmarkRecord,
    BenchmarkRecordId,
)
from wre.domain.depth_fields import DepthField
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
    "reference_support_denominator": "trusted_reference_valid_pixels",
    "comparison_support": "jointly_valid_candidate_reference_pixels",
    "depth_conversion": "none",
    "absolute_error_aggregation": "median",
    "absolute_relative_error_aggregation": "median",
    "relative_error_denominator": "positive_valid_reference_depth",
    "confidence_comparison": "none",
    "quality_thresholds": "none",
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

_DENSE_DEPTH_DIMENSION = MetricDimension("geometry.dense_depth")
_RATIO_UNIT = MetricUnit("ratio")
_DEPTH_VALUE_UNIT = MetricUnit("depth_value")

REFERENCE_VALID_COVERAGE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_valid_coverage_ratio"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("reference_valid_pixel_ratio"),
)
REFERENCE_ABSOLUTE_ERROR_MEDIAN_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_absolute_error_median"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_DEPTH_VALUE_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("joint_valid_pixel_median"),
)
REFERENCE_ABSOLUTE_RELATIVE_ERROR_MEDIAN_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.reference_absolute_relative_error_median"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("joint_valid_pixel_median"),
)


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _canonical_artifact_pair(left: ArtifactRef, right: ArtifactRef) -> tuple[ArtifactRef, ...]:
    return tuple(sorted((left, right), key=_artifact_key))


def _reference_valid_pixel_count(reference: DenseDepthArtifact) -> int:
    return sum(
        sum(1 for supported in depth_field.validity if supported)
        for depth_field in reference.depth_fields
    )


@dataclass(frozen=True, slots=True)
class DenseDepthBenchmarkRequest:
    """Controlled comparison request over immutable dense-depth evidence."""

    fixture: BenchmarkFixtureIdentity
    hardware: HardwareRuntimeIdentity
    reference: DenseDepthArtifact
    candidates: tuple[DenseDepthArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, BenchmarkFixtureIdentity):
            raise TypeError("dense_depth_benchmark.fixture must be BenchmarkFixtureIdentity")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("dense_depth_benchmark.hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.reference, DenseDepthArtifact):
            raise TypeError("dense_depth_benchmark.reference must be DenseDepthArtifact")
        if not isinstance(self.candidates, tuple):
            raise TypeError("dense_depth_benchmark.candidates must be an immutable tuple")
        if len(self.candidates) < 2:
            raise ValueError("dense_depth_benchmark requires at least two candidates")
        if any(not isinstance(item, DenseDepthArtifact) for item in self.candidates):
            raise TypeError("dense_depth_benchmark.candidates members must be DenseDepthArtifact")

        identities = tuple(_artifact_key(item.artifact_ref) for item in self.candidates)
        if len(identities) != len(set(identities)):
            raise ValueError("dense_depth_benchmark candidate artifacts must be unique")
        if _artifact_key(self.reference.artifact_ref) in set(identities):
            raise ValueError(
                "dense_depth_benchmark reference artifact must differ from every candidate"
            )
        if identities != tuple(sorted(identities)):
            raise ValueError(
                "dense_depth_benchmark candidates must use canonical ArtifactId/ArtifactKind order"
            )

        for candidate in self.candidates:
            if candidate.source_geometry != self.reference.source_geometry:
                raise ValueError(
                    "dense_depth_benchmark candidate and reference must preserve the exact "
                    "source GeometrySolutionCandidate"
                )

        if _reference_valid_pixel_count(self.reference) == 0:
            raise ValueError(
                "dense_depth_benchmark trusted reference must contain valid depth support"
            )


def _validate_matching_field_semantics(
    candidate: DepthField,
    reference: DepthField,
) -> None:
    if candidate.camera_solution_id != reference.camera_solution_id:
        raise ValueError(
            "dense_depth_benchmark overlapping fields must preserve CameraSolutionId"
        )
    if candidate.observation_id != reference.observation_id:
        raise ValueError(
            "dense_depth_benchmark overlapping fields must preserve ObservationId"
        )
    if candidate.dimensions != reference.dimensions:
        raise ValueError(
            "dense_depth_benchmark overlapping fields must preserve ImageDimensions"
        )
    if candidate.depth_value_convention != reference.depth_value_convention:
        raise ValueError(
            "dense_depth_benchmark overlapping fields must preserve "
            "DepthValueConventionName"
        )


def _reference_metrics(
    candidate: DenseDepthArtifact,
    reference: DenseDepthArtifact,
) -> MetricVector:
    reference_by_camera = {
        field.camera_solution_id: field for field in reference.depth_fields
    }
    reference_by_observation = {
        field.observation_id: field for field in reference.depth_fields
    }

    candidate_by_camera: dict[object, DepthField] = {}
    for field in candidate.depth_fields:
        reference_field = reference_by_camera.get(field.camera_solution_id)
        reference_observation_field = reference_by_observation.get(field.observation_id)
        if reference_field is not None:
            _validate_matching_field_semantics(field, reference_field)
        elif reference_observation_field is not None:
            _validate_matching_field_semantics(field, reference_observation_field)
        candidate_by_camera[field.camera_solution_id] = field

    reference_valid_count = _reference_valid_pixel_count(reference)
    jointly_valid_count = 0
    absolute_errors: list[float] = []
    absolute_relative_errors: list[float] = []

    for reference_field in reference.depth_fields:
        candidate_field = candidate_by_camera.get(reference_field.camera_solution_id)
        if candidate_field is None:
            continue
        _validate_matching_field_semantics(candidate_field, reference_field)

        for candidate_depth, candidate_valid, reference_depth, reference_valid in zip(
            candidate_field.depth_values,
            candidate_field.validity,
            reference_field.depth_values,
            reference_field.validity,
            strict=True,
        ):
            if not reference_valid or not candidate_valid:
                continue
            jointly_valid_count += 1
            absolute_error = abs(candidate_depth - reference_depth)
            absolute_errors.append(absolute_error)
            absolute_relative_errors.append(absolute_error / reference_depth)

    provenance = MetricProvenance(
        evaluator=DENSE_DEPTH_BENCHMARK_EVALUATOR,
        input_artifacts=_canonical_artifact_pair(
            candidate.artifact_ref,
            reference.artifact_ref,
        ),
    )
    observations = [
        MetricObservation(
            descriptor=REFERENCE_VALID_COVERAGE_RATIO_DESCRIPTOR,
            value=float(jointly_valid_count / reference_valid_count),
            provenance=provenance,
        )
    ]
    if jointly_valid_count:
        observations.extend(
            (
                MetricObservation(
                    descriptor=REFERENCE_ABSOLUTE_ERROR_MEDIAN_DESCRIPTOR,
                    value=float(median(absolute_errors)),
                    provenance=provenance,
                ),
                MetricObservation(
                    descriptor=REFERENCE_ABSOLUTE_RELATIVE_ERROR_MEDIAN_DESCRIPTOR,
                    value=float(median(absolute_relative_errors)),
                    provenance=provenance,
                ),
            )
        )

    return MetricVector(
        observations=tuple(
            sorted(
                observations,
                key=lambda item: item.descriptor.name.value,
            )
        )
    )


def _benchmark_record_id(
    request: DenseDepthBenchmarkRequest,
    candidate: DenseDepthArtifact,
) -> BenchmarkRecordId:
    document = {
        "schema_version": 1,
        "fixture_id": request.fixture.fixture_id.value,
        "fixture_sha256": request.fixture.sha256.value,
        "hardware_sha256": request.hardware.sha256.value,
        "candidate": {
            "artifact_id": candidate.artifact_ref.artifact_id.value,
            "artifact_kind": candidate.artifact_ref.artifact_kind.value,
        },
        "reference": {
            "artifact_id": request.reference.artifact_ref.artifact_id.value,
            "artifact_kind": request.reference.artifact_ref.artifact_kind.value,
        },
        "evaluator": {
            "implementation": DENSE_DEPTH_BENCHMARK_IMPLEMENTATION,
            "version": DENSE_DEPTH_BENCHMARK_VERSION,
            "revision": DENSE_DEPTH_BENCHMARK_REVISION,
            "configuration_sha256": (
                DENSE_DEPTH_BENCHMARK_EVALUATOR.configuration.sha256.value
            ),
        },
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return BenchmarkRecordId(
        f"dense-depth:{hashlib.sha256(encoded).hexdigest()[:32]}"
    )


@dataclass(frozen=True, slots=True)
class DenseDepthCandidateBenchmarkResult:
    """One descriptive benchmark record for one existing candidate artifact."""

    candidate: DenseDepthArtifact
    coverage: DenseDepthCoverageReport
    reference_metrics: MetricVector
    benchmark_record: BenchmarkRecord

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, DenseDepthArtifact):
            raise TypeError(
                "dense_depth_candidate_benchmark.candidate must be DenseDepthArtifact"
            )
        if not isinstance(self.coverage, DenseDepthCoverageReport):
            raise TypeError(
                "dense_depth_candidate_benchmark.coverage must be DenseDepthCoverageReport"
            )
        if not isinstance(self.reference_metrics, MetricVector):
            raise TypeError(
                "dense_depth_candidate_benchmark.reference_metrics must be MetricVector"
            )
        if not isinstance(self.benchmark_record, BenchmarkRecord):
            raise TypeError(
                "dense_depth_candidate_benchmark.benchmark_record must be BenchmarkRecord"
            )
        if self.coverage.source_depth != self.candidate:
            raise ValueError(
                "dense_depth_candidate_benchmark coverage must summarize the candidate"
            )

        expected_metrics = tuple(
            sorted(
                self.coverage.metrics.observations
                + self.reference_metrics.observations,
                key=lambda item: item.descriptor.name.value,
            )
        )
        if self.benchmark_record.metrics.observations != expected_metrics:
            raise ValueError(
                "dense_depth_candidate_benchmark BenchmarkRecord must retain support and "
                "reference metrics in canonical MetricName order"
            )
        if self.benchmark_record.producer != self.candidate.producer:
            raise ValueError(
                "dense_depth_candidate_benchmark BenchmarkRecord producer must match candidate"
            )
        if self.benchmark_record.quality_mode is not QualityMode.QUALITY:
            raise ValueError(
                "dense_depth_candidate_benchmark BenchmarkRecord must use QualityMode.QUALITY"
            )
        if self.benchmark_record.comparison_baseline is not None:
            raise ValueError(
                "dense_depth_candidate_benchmark must not encode a comparison baseline"
            )


@dataclass(frozen=True, slots=True)
class DenseDepthBenchmarkResult:
    """Deterministic controlled benchmark result with no selection semantics."""

    request: DenseDepthBenchmarkRequest
    candidates: tuple[DenseDepthCandidateBenchmarkResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, DenseDepthBenchmarkRequest):
            raise TypeError("dense_depth_benchmark_result.request must be DenseDepthBenchmarkRequest")
        if not isinstance(self.candidates, tuple):
            raise TypeError(
                "dense_depth_benchmark_result.candidates must be an immutable tuple"
            )
        if any(
            not isinstance(item, DenseDepthCandidateBenchmarkResult)
            for item in self.candidates
        ):
            raise TypeError(
                "dense_depth_benchmark_result.candidates members must be "
                "DenseDepthCandidateBenchmarkResult"
            )
        if tuple(item.candidate for item in self.candidates) != self.request.candidates:
            raise ValueError(
                "dense_depth_benchmark_result candidates must exactly match request order"
            )
        for item in self.candidates:
            if item.benchmark_record.fixture != self.request.fixture:
                raise ValueError(
                    "dense_depth_benchmark_result records must preserve shared fixture identity"
                )
            if item.benchmark_record.hardware != self.request.hardware:
                raise ValueError(
                    "dense_depth_benchmark_result records must preserve shared hardware identity"
                )


def benchmark_dense_depth_candidates(
    request: DenseDepthBenchmarkRequest,
) -> DenseDepthBenchmarkResult:
    """Describe candidate support and trusted-reference depth error without selection."""

    if not isinstance(request, DenseDepthBenchmarkRequest):
        raise TypeError("request must be DenseDepthBenchmarkRequest")

    results: list[DenseDepthCandidateBenchmarkResult] = []
    for candidate in request.candidates:
        coverage = evaluate_dense_depth_coverage(candidate)
        reference_metrics = _reference_metrics(candidate, request.reference)
        combined_metrics = MetricVector(
            observations=tuple(
                sorted(
                    coverage.metrics.observations + reference_metrics.observations,
                    key=lambda item: item.descriptor.name.value,
                )
            )
        )
        record = BenchmarkRecord(
            record_id=_benchmark_record_id(request, candidate),
            fixture=request.fixture,
            producer=candidate.producer,
            quality_mode=QualityMode.QUALITY,
            hardware=request.hardware,
            metrics=combined_metrics,
        )
        results.append(
            DenseDepthCandidateBenchmarkResult(
                candidate=candidate,
                coverage=coverage,
                reference_metrics=reference_metrics,
                benchmark_record=record,
            )
        )

    return DenseDepthBenchmarkResult(
        request=request,
        candidates=tuple(results),
    )


__all__ = [
    "DENSE_DEPTH_BENCHMARK_EVALUATOR",
    "DENSE_DEPTH_BENCHMARK_IMPLEMENTATION",
    "DENSE_DEPTH_BENCHMARK_REVISION",
    "DENSE_DEPTH_BENCHMARK_VERSION",
    "REFERENCE_ABSOLUTE_ERROR_MEDIAN_DESCRIPTOR",
    "REFERENCE_ABSOLUTE_RELATIVE_ERROR_MEDIAN_DESCRIPTOR",
    "REFERENCE_VALID_COVERAGE_RATIO_DESCRIPTOR",
    "DenseDepthBenchmarkRequest",
    "DenseDepthBenchmarkResult",
    "DenseDepthCandidateBenchmarkResult",
    "benchmark_dense_depth_candidates",
]
