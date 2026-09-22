from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
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
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.quality import QualityMode
from wre.domain.runs import ProducerRef
from wre.reconstruction.da3_runtime import (
    DA3_PROFILE_STAGE_NAMES,
    Da3ExecutionResult,
)

DA3_EXECUTION_BENCHMARK_IMPLEMENTATION = "wre.reconstruction.da3_execution_benchmark"
DA3_EXECUTION_BENCHMARK_VERSION = "1"
DA3_EXECUTION_BENCHMARK_REVISION = "v2l13.6"
DA3_EXECUTION_END_TO_END_STAGE = "end_to_end"
DA3_EXECUTION_PROFILE_EVIDENCE_KIND = ArtifactKind("performance.execution_profile")
DA3_REFERENCE_PROFILE = "da3.cpu.float32.eager"
DA3_CUDA_EAGER_PROFILE = "da3.cuda.float32.eager"
DA3_CAMERA_METRIC_ABS_TOLERANCE = 1e-5


class Da3ExecutionProfileStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class Da3ExecutionSampleKind(StrEnum):
    COLD = "cold"
    STEADY = "steady"


@dataclass(frozen=True, slots=True, order=True)
class Da3ExecutionProfileId:
    value: str

    def __post_init__(self) -> None:
        if self.value not in (DA3_REFERENCE_PROFILE, DA3_CUDA_EAGER_PROFILE):
            raise ValueError("unsupported DA3 execution profile")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Da3ExecutionProfileAvailability:
    profile: Da3ExecutionProfileId
    status: Da3ExecutionProfileStatus
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Da3ExecutionProfileId):
            raise TypeError("profile availability profile must be Da3ExecutionProfileId")
        if not isinstance(self.status, Da3ExecutionProfileStatus):
            raise TypeError("profile availability status must be Da3ExecutionProfileStatus")
        if self.status is Da3ExecutionProfileStatus.AVAILABLE:
            if self.reason is not None:
                raise ValueError("available profile must not carry an unavailability reason")
        elif not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("unavailable or failed profile requires a non-blank reason")


@dataclass(frozen=True, slots=True)
class Da3StageTiming:
    stage: str
    elapsed_seconds: float

    def __post_init__(self) -> None:
        allowed = (*DA3_PROFILE_STAGE_NAMES, DA3_EXECUTION_END_TO_END_STAGE)
        if self.stage not in allowed:
            raise ValueError("unsupported DA3 benchmark stage")
        if type(self.elapsed_seconds) is not float:
            raise TypeError("stage timing elapsed_seconds must be float")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0.0:
            raise ValueError("stage timing elapsed_seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class Da3GeometrySemantics:
    observation_ids: tuple[ObservationId, ...]
    projection_models: tuple[str, ...]
    image_dimensions: tuple[tuple[int, int], ...]
    depth_dimensions: tuple[tuple[int, int], ...]
    depth_valid_counts: tuple[int, ...]
    depth_all_valid_samples_positive: tuple[bool, ...]
    scale_status: str
    point_map_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation_ids, tuple) or not self.observation_ids:
            raise ValueError("geometry semantics observation_ids must be a non-empty tuple")
        if any(not isinstance(item, ObservationId) for item in self.observation_ids):
            raise TypeError("geometry semantics observation_ids members must be ObservationId")
        values = tuple(item.value for item in self.observation_ids)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("geometry semantics observation_ids must be unique canonical order")
        count = len(self.observation_ids)
        for name, values_tuple in (
            ("projection_models", self.projection_models),
            ("image_dimensions", self.image_dimensions),
            ("depth_dimensions", self.depth_dimensions),
            ("depth_valid_counts", self.depth_valid_counts),
            (
                "depth_all_valid_samples_positive",
                self.depth_all_valid_samples_positive,
            ),
        ):
            if not isinstance(values_tuple, tuple) or len(values_tuple) != count:
                raise ValueError(f"geometry semantics {name} must match observation count")
        if type(self.point_map_count) is not int or self.point_map_count < 0:
            raise ValueError("geometry semantics point_map_count must be a non-negative integer")
        if not isinstance(self.scale_status, str) or not self.scale_status:
            raise ValueError("geometry semantics scale_status must be non-blank")


@dataclass(frozen=True, slots=True)
class Da3ProfileExecutionObservation:
    profile: Da3ExecutionProfileId
    producer: ArtifactProducerIdentity
    hardware: HardwareRuntimeIdentity
    normalization_identity: Sha256Digest
    stage_timings: tuple[Da3StageTiming, ...]
    peak_process_rss_bytes: int
    peak_gpu_allocated_bytes: int | None
    peak_gpu_reserved_bytes: int | None
    semantics: Da3GeometrySemantics
    camera_quality: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Da3ExecutionProfileId):
            raise TypeError("profile observation profile must be Da3ExecutionProfileId")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("profile observation producer must be ArtifactProducerIdentity")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("profile observation hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.normalization_identity, Sha256Digest):
            raise TypeError("profile observation normalization_identity must be Sha256Digest")
        if not isinstance(self.stage_timings, tuple) or not self.stage_timings:
            raise ValueError("profile observation stage_timings must be a non-empty tuple")
        if any(not isinstance(item, Da3StageTiming) for item in self.stage_timings):
            raise TypeError("profile observation stage_timings members must be Da3StageTiming")
        stages = tuple(item.stage for item in self.stage_timings)
        if len(stages) != len(set(stages)):
            raise ValueError("profile observation stage timings must be unique")
        if DA3_EXECUTION_END_TO_END_STAGE not in stages:
            raise ValueError("profile observation requires end_to_end timing")
        if type(self.peak_process_rss_bytes) is not int or self.peak_process_rss_bytes <= 0:
            raise ValueError("profile observation peak_process_rss_bytes must be positive integer")
        for field_name, value in (
            ("peak_gpu_allocated_bytes", self.peak_gpu_allocated_bytes),
            ("peak_gpu_reserved_bytes", self.peak_gpu_reserved_bytes),
        ):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"profile observation {field_name} must be non-negative integer")
        if not isinstance(self.semantics, Da3GeometrySemantics):
            raise TypeError("profile observation semantics must be Da3GeometrySemantics")
        if not isinstance(self.camera_quality, MetricVector):
            raise TypeError("profile observation camera_quality must be MetricVector")


@dataclass(frozen=True, slots=True)
class Da3ExecutionSample:
    kind: Da3ExecutionSampleKind
    sample_index: int
    observation: Da3ProfileExecutionObservation

    def __post_init__(self) -> None:
        if not isinstance(self.kind, Da3ExecutionSampleKind):
            raise TypeError("execution sample kind must be Da3ExecutionSampleKind")
        if type(self.sample_index) is not int or self.sample_index < 0:
            raise ValueError("execution sample index must be a non-negative integer")
        if self.kind is Da3ExecutionSampleKind.COLD and self.sample_index != 0:
            raise ValueError("cold execution sample index must be zero")
        if self.kind is Da3ExecutionSampleKind.STEADY and self.sample_index < 1:
            raise ValueError("steady execution sample index must be positive")
        if not isinstance(self.observation, Da3ProfileExecutionObservation):
            raise TypeError("execution sample observation must be Da3ProfileExecutionObservation")


@dataclass(frozen=True, slots=True, kw_only=True)
class Da3ExecutionBenchmarkRequest:
    fixture: BenchmarkFixtureIdentity
    hardware: HardwareRuntimeIdentity
    input_artifacts: tuple[ArtifactRef, ...]
    steady_state_samples: int = 2
    reference_profile: Da3ExecutionProfileId = Da3ExecutionProfileId(DA3_REFERENCE_PROFILE)
    candidate_profiles: tuple[Da3ExecutionProfileId, ...] = (
        Da3ExecutionProfileId(DA3_CUDA_EAGER_PROFILE),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, BenchmarkFixtureIdentity):
            raise TypeError("benchmark request fixture must be BenchmarkFixtureIdentity")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("benchmark request hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.input_artifacts, tuple):
            raise TypeError("benchmark request input_artifacts must be an immutable tuple")
        if any(not isinstance(item, ArtifactRef) for item in self.input_artifacts):
            raise TypeError("benchmark request input_artifacts members must be ArtifactRef")
        artifact_keys = tuple(
            (item.artifact_id.value, item.artifact_kind.value) for item in self.input_artifacts
        )
        if artifact_keys != tuple(sorted(artifact_keys)):
            raise ValueError("benchmark request input_artifacts must be canonical order")
        if len(artifact_keys) != len(set(artifact_keys)):
            raise ValueError("benchmark request input_artifacts must be unique")
        if type(self.steady_state_samples) is not int or self.steady_state_samples < 1:
            raise ValueError("benchmark request steady_state_samples must be positive integer")
        if self.reference_profile != Da3ExecutionProfileId(DA3_REFERENCE_PROFILE):
            raise ValueError("benchmark request reference_profile must be exact CPU reference")
        if not isinstance(self.candidate_profiles, tuple):
            raise TypeError("benchmark request candidate_profiles must be immutable tuple")
        if any(not isinstance(item, Da3ExecutionProfileId) for item in self.candidate_profiles):
            raise TypeError("benchmark request candidate_profiles members have wrong type")
        candidate_values = tuple(item.value for item in self.candidate_profiles)
        if candidate_values != tuple(sorted(candidate_values)):
            raise ValueError("benchmark request candidate_profiles must be canonical order")
        if len(candidate_values) != len(set(candidate_values)):
            raise ValueError("benchmark request candidate_profiles must be unique")
        if self.reference_profile in self.candidate_profiles:
            raise ValueError("benchmark request candidates must not repeat reference profile")


@dataclass(frozen=True, slots=True)
class Da3MeasuredExecutionProfile:
    profile: Da3ExecutionProfileId
    cold_sample: Da3ExecutionSample
    steady_samples: tuple[Da3ExecutionSample, ...]
    benchmark_record: BenchmarkRecord

    def __post_init__(self) -> None:
        if not isinstance(self.profile, Da3ExecutionProfileId):
            raise TypeError("measured profile profile must be Da3ExecutionProfileId")
        if not isinstance(self.cold_sample, Da3ExecutionSample):
            raise TypeError("measured profile cold_sample must be Da3ExecutionSample")
        if self.cold_sample.kind is not Da3ExecutionSampleKind.COLD:
            raise ValueError("measured profile cold_sample must be cold")
        if not isinstance(self.steady_samples, tuple) or not self.steady_samples:
            raise ValueError("measured profile steady_samples must be non-empty tuple")
        if any(item.kind is not Da3ExecutionSampleKind.STEADY for item in self.steady_samples):
            raise ValueError("measured profile steady samples must all be steady")
        if any(
            item.observation.profile != self.profile
            for item in (self.cold_sample, *self.steady_samples)
        ):
            raise ValueError("measured profile samples must use the profile identity")
        if not isinstance(self.benchmark_record, BenchmarkRecord):
            raise TypeError("measured profile benchmark_record must be BenchmarkRecord")


@dataclass(frozen=True, slots=True)
class Da3CandidateExecutionOutcome:
    availability: Da3ExecutionProfileAvailability
    measured: Da3MeasuredExecutionProfile | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.availability, Da3ExecutionProfileAvailability):
            raise TypeError("candidate outcome availability has wrong type")
        if self.availability.status is Da3ExecutionProfileStatus.AVAILABLE:
            if not isinstance(self.measured, Da3MeasuredExecutionProfile):
                raise ValueError("available candidate outcome requires measured profile")
            if self.measured.profile != self.availability.profile:
                raise ValueError("candidate outcome measured profile identity mismatch")
        elif self.measured is not None:
            raise ValueError("unavailable or failed candidate must not carry measured profile")


@dataclass(frozen=True, slots=True)
class Da3ExecutionBenchmarkResult:
    reference: Da3MeasuredExecutionProfile
    candidates: tuple[Da3CandidateExecutionOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Da3MeasuredExecutionProfile):
            raise TypeError("benchmark result reference must be measured profile")
        if self.reference.profile != Da3ExecutionProfileId(DA3_REFERENCE_PROFILE):
            raise ValueError("benchmark result reference must be CPU reference profile")
        if not isinstance(self.candidates, tuple):
            raise TypeError("benchmark result candidates must be immutable tuple")
        if any(not isinstance(item, Da3CandidateExecutionOutcome) for item in self.candidates):
            raise TypeError("benchmark result candidates members have wrong type")
        values = tuple(item.availability.profile.value for item in self.candidates)
        if values != tuple(sorted(values)) or len(values) != len(set(values)):
            raise ValueError("benchmark result candidates must be unique canonical profile order")


class Da3ExecutionBenchmarkExecutor(Protocol):
    def probe(self, profile: Da3ExecutionProfileId) -> Da3ExecutionProfileAvailability: ...

    def execute(
        self,
        profile: Da3ExecutionProfileId,
        *,
        kind: Da3ExecutionSampleKind,
        sample_index: int,
    ) -> Da3ProfileExecutionObservation: ...


def geometry_semantics(result: Da3ExecutionResult) -> Da3GeometrySemantics:
    cameras = result.geometry.camera_solutions
    depths = result.geometry.depth_fields
    depth_by_observation = {item.observation_id: item for item in depths}
    observation_ids = tuple(item.observation_id for item in cameras)
    if set(depth_by_observation) != set(observation_ids):
        raise ValueError("DA3 benchmark requires one depth field for every camera observation")

    ordered_depths = tuple(depth_by_observation[item] for item in observation_ids)
    return Da3GeometrySemantics(
        observation_ids=observation_ids,
        projection_models=tuple(item.projection_model.value for item in cameras),
        image_dimensions=tuple(
            (item.dimensions.width_px, item.dimensions.height_px) for item in cameras
        ),
        depth_dimensions=tuple(
            (item.dimensions.width_px, item.dimensions.height_px) for item in ordered_depths
        ),
        depth_valid_counts=tuple(sum(item.validity) for item in ordered_depths),
        depth_all_valid_samples_positive=tuple(
            all(
                value > 0.0
                for value, valid in zip(item.depth_values, item.validity, strict=True)
                if valid
            )
            for item in ordered_depths
        ),
        scale_status=result.geometry.geometry_solution.scale_status.value,
        point_map_count=len(result.geometry.point_maps),
    )


def _benchmark_configuration(
    request: Da3ExecutionBenchmarkRequest,
    profile: Da3ExecutionProfileId,
) -> ConfigurationIdentity:
    document = {
        "schema_version": 1,
        "profile": profile.value,
        "steady_state_samples": request.steady_state_samples,
        "fixture_id": request.fixture.fixture_id.value,
        "fixture_sha256": request.fixture.sha256.value,
        "hardware_sha256": request.hardware.sha256.value,
        "input_artifacts": [
            {
                "artifact_id": item.artifact_id.value,
                "artifact_kind": item.artifact_kind.value,
            }
            for item in request.input_artifacts
        ],
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return ConfigurationIdentity(sha256=Sha256Digest(hashlib.sha256(encoded).hexdigest()))


def _benchmark_evaluator(
    request: Da3ExecutionBenchmarkRequest,
    profile: Da3ExecutionProfileId,
) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=DA3_EXECUTION_BENCHMARK_IMPLEMENTATION,
            version=DA3_EXECUTION_BENCHMARK_VERSION,
            revision=DA3_EXECUTION_BENCHMARK_REVISION,
        ),
        configuration=_benchmark_configuration(request, profile),
    )


def _descriptor(name: str, unit: str, aggregation: str) -> MetricDescriptor:
    return MetricDescriptor(
        name=MetricName(name),
        dimension=MetricDimension("runtime"),
        unit=MetricUnit(unit),
        direction=MetricDirection.LOWER_IS_BETTER,
        aggregation=MetricAggregation(aggregation),
    )


def _performance_metric(
    *,
    evaluator: ArtifactProducerIdentity,
    input_artifacts: tuple[ArtifactRef, ...],
    name: str,
    unit: str,
    aggregation: str,
    value: float,
) -> MetricObservation:
    return MetricObservation(
        descriptor=_descriptor(name, unit, aggregation),
        value=value,
        provenance=MetricProvenance(
            evaluator=evaluator,
            input_artifacts=input_artifacts,
        ),
    )


def _timing_map(sample: Da3ExecutionSample) -> dict[str, float]:
    return {item.stage: item.elapsed_seconds for item in sample.observation.stage_timings}


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("cannot aggregate empty benchmark values")
    result = float(statistics.median(values))
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("benchmark median must be finite and non-negative")
    return result


def _performance_metrics(
    request: Da3ExecutionBenchmarkRequest,
    profile: Da3ExecutionProfileId,
    cold: Da3ExecutionSample,
    steady: tuple[Da3ExecutionSample, ...],
) -> MetricVector:
    evaluator = _benchmark_evaluator(request, profile)
    observations: list[MetricObservation] = []
    cold_timings = _timing_map(cold)
    steady_timings = tuple(_timing_map(item) for item in steady)
    stages = tuple(sorted(cold_timings))
    if any(tuple(sorted(item)) != stages for item in steady_timings):
        raise ValueError("all benchmark samples must expose the same stage timing set")

    for stage in stages:
        observations.append(
            _performance_metric(
                evaluator=evaluator,
                input_artifacts=request.input_artifacts,
                name=f"runtime.da3.{stage}.cold_seconds",
                unit="second",
                aggregation="cold",
                value=cold_timings[stage],
            )
        )
        observations.append(
            _performance_metric(
                evaluator=evaluator,
                input_artifacts=request.input_artifacts,
                name=f"runtime.da3.{stage}.steady_median_seconds",
                unit="second",
                aggregation="median",
                value=_median(tuple(item[stage] for item in steady_timings)),
            )
        )

    peak_rss = float(
        max(
            (
                cold.observation.peak_process_rss_bytes,
                *(item.observation.peak_process_rss_bytes for item in steady),
            )
        )
    )
    observations.append(
        _performance_metric(
            evaluator=evaluator,
            input_artifacts=request.input_artifacts,
            name="runtime.da3.peak_process_rss_bytes",
            unit="byte",
            aggregation="peak",
            value=peak_rss,
        )
    )

    gpu_allocated = tuple(
        item
        for item in (
            cold.observation.peak_gpu_allocated_bytes,
            *(sample.observation.peak_gpu_allocated_bytes for sample in steady),
        )
        if item is not None
    )
    gpu_reserved = tuple(
        item
        for item in (
            cold.observation.peak_gpu_reserved_bytes,
            *(sample.observation.peak_gpu_reserved_bytes for sample in steady),
        )
        if item is not None
    )
    if gpu_allocated:
        if len(gpu_allocated) != len(steady) + 1:
            raise ValueError("GPU allocated-memory telemetry must exist for every sample or none")
        observations.append(
            _performance_metric(
                evaluator=evaluator,
                input_artifacts=request.input_artifacts,
                name="runtime.da3.peak_gpu_allocated_bytes",
                unit="byte",
                aggregation="peak",
                value=float(max(gpu_allocated)),
            )
        )
    if gpu_reserved:
        if len(gpu_reserved) != len(steady) + 1:
            raise ValueError("GPU reserved-memory telemetry must exist for every sample or none")
        observations.append(
            _performance_metric(
                evaluator=evaluator,
                input_artifacts=request.input_artifacts,
                name="runtime.da3.peak_gpu_reserved_bytes",
                unit="byte",
                aggregation="peak",
                value=float(max(gpu_reserved)),
            )
        )

    return MetricVector(
        observations=tuple(sorted(observations, key=lambda item: item.descriptor.name.value))
    )


def _combine_metrics(performance: MetricVector, camera_quality: MetricVector) -> MetricVector:
    observations = (*performance.observations, *camera_quality.observations)
    names = tuple(item.descriptor.name.value for item in observations)
    if len(names) != len(set(names)):
        raise ValueError("benchmark performance and camera-quality metric names must be distinct")
    return MetricVector(
        observations=tuple(sorted(observations, key=lambda item: item.descriptor.name.value))
    )


def _record_identity(
    request: Da3ExecutionBenchmarkRequest,
    profile: Da3ExecutionProfileId,
) -> Sha256Digest:
    configuration = _benchmark_configuration(request, profile)
    material = (
        f"wre.da3-execution-benchmark\0{configuration.sha256.value}\0"
        f"{profile.value}\0{request.hardware.sha256.value}"
    )
    return Sha256Digest(hashlib.sha256(material.encode("utf-8")).hexdigest())


def _performance_evidence(
    request: Da3ExecutionBenchmarkRequest,
    profile: Da3ExecutionProfileId,
) -> ArtifactRef:
    identity = _record_identity(request, profile)
    return ArtifactRef(
        artifact_id=ArtifactId(f"performance:da3:{identity.value[:48]}"),
        artifact_kind=DA3_EXECUTION_PROFILE_EVIDENCE_KIND,
    )


def _quality_values(vector: MetricVector) -> dict[str, float]:
    return {item.descriptor.name.value: item.value for item in vector.observations}


def _assert_reference_samples_consistent(
    cold: Da3ExecutionSample,
    steady: tuple[Da3ExecutionSample, ...],
) -> None:
    semantics = cold.observation.semantics
    quality = _quality_values(cold.observation.camera_quality)
    for sample in steady:
        if sample.observation.semantics != semantics:
            raise ValueError("reference benchmark samples changed geometry semantics")
        if _quality_values(sample.observation.camera_quality) != quality:
            raise ValueError("reference benchmark samples changed camera-quality metrics")


def _assert_candidate_compatible(
    reference: Da3MeasuredExecutionProfile,
    candidate: Da3MeasuredExecutionProfile,
) -> None:
    reference_semantics = reference.cold_sample.observation.semantics
    candidate_semantics = candidate.cold_sample.observation.semantics
    if candidate_semantics != reference_semantics:
        raise ValueError("candidate execution profile changed canonical geometry semantics")

    reference_quality = _quality_values(reference.cold_sample.observation.camera_quality)
    candidate_quality = _quality_values(candidate.cold_sample.observation.camera_quality)
    if set(candidate_quality) != set(reference_quality):
        raise ValueError("candidate execution profile changed camera-quality metric membership")
    for name in sorted(reference_quality):
        if abs(candidate_quality[name] - reference_quality[name]) > DA3_CAMERA_METRIC_ABS_TOLERANCE:
            raise ValueError("candidate execution profile changed camera quality beyond tolerance")


def _measure_profile(
    request: Da3ExecutionBenchmarkRequest,
    executor: Da3ExecutionBenchmarkExecutor,
    profile: Da3ExecutionProfileId,
    *,
    comparison_baseline: BenchmarkRecordId | None,
) -> Da3MeasuredExecutionProfile:
    cold_observation = executor.execute(
        profile,
        kind=Da3ExecutionSampleKind.COLD,
        sample_index=0,
    )
    cold = Da3ExecutionSample(
        kind=Da3ExecutionSampleKind.COLD,
        sample_index=0,
        observation=cold_observation,
    )
    steady = tuple(
        Da3ExecutionSample(
            kind=Da3ExecutionSampleKind.STEADY,
            sample_index=index,
            observation=executor.execute(
                profile,
                kind=Da3ExecutionSampleKind.STEADY,
                sample_index=index,
            ),
        )
        for index in range(1, request.steady_state_samples + 1)
    )

    samples = (cold, *steady)
    if any(item.observation.profile != profile for item in samples):
        raise ValueError("executor returned a different profile than requested")
    if any(item.observation.hardware != request.hardware for item in samples):
        raise ValueError("benchmark sample hardware identity differs from request")
    producer = cold.observation.producer
    if any(item.observation.producer != producer for item in samples):
        raise ValueError("benchmark profile producer identity changed across samples")
    _assert_reference_samples_consistent(cold, steady)

    performance = _performance_metrics(request, profile, cold, steady)
    metrics = _combine_metrics(performance, cold.observation.camera_quality)
    identity = _record_identity(request, profile)
    record = BenchmarkRecord(
        record_id=BenchmarkRecordId(f"benchmark:da3:{identity.value[:48]}"),
        fixture=request.fixture,
        producer=producer,
        quality_mode=QualityMode.PREVIEW,
        hardware=request.hardware,
        metrics=metrics,
        performance_evidence=(_performance_evidence(request, profile),),
        comparison_baseline=comparison_baseline,
    )
    return Da3MeasuredExecutionProfile(
        profile=profile,
        cold_sample=cold,
        steady_samples=steady,
        benchmark_record=record,
    )


def benchmark_da3_execution(
    request: Da3ExecutionBenchmarkRequest,
    *,
    executor: Da3ExecutionBenchmarkExecutor,
) -> Da3ExecutionBenchmarkResult:
    if not isinstance(request, Da3ExecutionBenchmarkRequest):
        raise TypeError("request must be Da3ExecutionBenchmarkRequest")

    reference_availability = executor.probe(request.reference_profile)
    if reference_availability.profile != request.reference_profile:
        raise ValueError("reference profile probe returned wrong profile")
    if reference_availability.status is not Da3ExecutionProfileStatus.AVAILABLE:
        raise ValueError("reference DA3 execution profile must be available")

    reference = _measure_profile(
        request,
        executor,
        request.reference_profile,
        comparison_baseline=None,
    )

    candidates: list[Da3CandidateExecutionOutcome] = []
    for profile in request.candidate_profiles:
        availability = executor.probe(profile)
        if availability.profile != profile:
            raise ValueError("candidate profile probe returned wrong profile")
        if availability.status is not Da3ExecutionProfileStatus.AVAILABLE:
            candidates.append(Da3CandidateExecutionOutcome(availability=availability))
            continue
        try:
            measured = _measure_profile(
                request,
                executor,
                profile,
                comparison_baseline=reference.benchmark_record.record_id,
            )
            _assert_candidate_compatible(reference, measured)
        except Exception as exc:
            candidates.append(
                Da3CandidateExecutionOutcome(
                    availability=Da3ExecutionProfileAvailability(
                        profile=profile,
                        status=Da3ExecutionProfileStatus.FAILED,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                )
            )
            continue
        candidates.append(
            Da3CandidateExecutionOutcome(
                availability=availability,
                measured=measured,
            )
        )

    return Da3ExecutionBenchmarkResult(
        reference=reference,
        candidates=tuple(candidates),
    )
