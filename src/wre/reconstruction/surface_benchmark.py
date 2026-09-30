from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.benchmarks import (
    BenchmarkFixtureIdentity,
    BenchmarkRecord,
    BenchmarkRecordId,
)
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.quality import QualityMode
from wre.domain.surface_support import SurfaceSupportMap
from wre.reconstruction.surface_suitability import (
    SurfaceMeshInspection,
    SurfaceSuitabilityReport,
    evaluate_surface_suitability,
)

SURFACE_BENCHMARK_IMPLEMENTATION = "wre.reconstruction.surface_benchmark"
SURFACE_BENCHMARK_VERSION = "1"
SURFACE_BENCHMARK_REVISION = "v2l16.6"


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkCandidate:
    """One immutable physical-surface evidence bundle for controlled benchmarking."""

    inspection: SurfaceMeshInspection
    support_map: SurfaceSupportMap | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.inspection, SurfaceMeshInspection):
            raise TypeError("surface_benchmark_candidate.inspection must be SurfaceMeshInspection")
        if self.support_map is not None and not isinstance(self.support_map, SurfaceSupportMap):
            raise TypeError(
                "surface_benchmark_candidate.support_map must be SurfaceSupportMap when present"
            )
        if self.support_map is not None:
            if self.support_map.surface_model != self.inspection.surface_model:
                raise ValueError(
                    "surface_benchmark_candidate support map must reference the exact "
                    "inspection SurfaceModel"
                )
            if self.support_map.local_frame_id != self.inspection.surface_model.local_frame_id:
                raise ValueError(
                    "surface_benchmark_candidate support map must preserve the inspection "
                    "SurfaceModel LocalFrameId"
                )


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkRequest:
    """Canonical controlled benchmark request with no surface-selection semantics."""

    fixture: BenchmarkFixtureIdentity
    hardware: HardwareRuntimeIdentity
    candidates: tuple[SurfaceBenchmarkCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, BenchmarkFixtureIdentity):
            raise TypeError("surface_benchmark.fixture must be BenchmarkFixtureIdentity")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("surface_benchmark.hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.candidates, tuple):
            raise TypeError("surface_benchmark.candidates must be an immutable tuple")
        if len(self.candidates) < 3:
            raise ValueError("surface_benchmark requires at least three candidates")
        if any(not isinstance(item, SurfaceBenchmarkCandidate) for item in self.candidates):
            raise TypeError(
                "surface_benchmark.candidates members must be SurfaceBenchmarkCandidate"
            )

        identities = tuple(
            _artifact_key(item.inspection.surface_model.artifact_ref) for item in self.candidates
        )
        if len(identities) != len(set(identities)):
            raise ValueError("surface_benchmark SurfaceModel artifacts must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError(
                "surface_benchmark candidates must use canonical SurfaceModel "
                "ArtifactId/ArtifactKind order"
            )


def _benchmark_record_id(
    request: SurfaceBenchmarkRequest,
    candidate: SurfaceBenchmarkCandidate,
) -> BenchmarkRecordId:
    support_ref = None
    if candidate.support_map is not None:
        support_ref = {
            "artifact_id": candidate.support_map.artifact_ref.artifact_id.value,
            "artifact_kind": candidate.support_map.artifact_ref.artifact_kind.value,
        }
    surface_ref = candidate.inspection.surface_model.artifact_ref
    document = {
        "schema_version": 1,
        "implementation": SURFACE_BENCHMARK_IMPLEMENTATION,
        "version": SURFACE_BENCHMARK_VERSION,
        "revision": SURFACE_BENCHMARK_REVISION,
        "fixture_id": request.fixture.fixture_id.value,
        "fixture_sha256": request.fixture.sha256.value,
        "hardware_sha256": request.hardware.sha256.value,
        "surface": {
            "artifact_id": surface_ref.artifact_id.value,
            "artifact_kind": surface_ref.artifact_kind.value,
        },
        "support_map": support_ref,
    }
    encoded = json.dumps(
        document,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return BenchmarkRecordId(f"surface:{hashlib.sha256(encoded).hexdigest()[:32]}")


@dataclass(frozen=True, slots=True)
class SurfaceCandidateBenchmarkResult:
    """Descriptive benchmark evidence for one existing physical surface."""

    candidate: SurfaceBenchmarkCandidate
    suitability: SurfaceSuitabilityReport
    benchmark_record: BenchmarkRecord

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SurfaceBenchmarkCandidate):
            raise TypeError(
                "surface_candidate_benchmark.candidate must be SurfaceBenchmarkCandidate"
            )
        if not isinstance(self.suitability, SurfaceSuitabilityReport):
            raise TypeError(
                "surface_candidate_benchmark.suitability must be SurfaceSuitabilityReport"
            )
        if not isinstance(self.benchmark_record, BenchmarkRecord):
            raise TypeError(
                "surface_candidate_benchmark.benchmark_record must be BenchmarkRecord"
            )
        if self.suitability.inspection != self.candidate.inspection:
            raise ValueError(
                "surface_candidate_benchmark suitability must summarize the exact candidate "
                "inspection"
            )
        if self.benchmark_record.metrics != self.suitability.metrics:
            raise ValueError(
                "surface_candidate_benchmark BenchmarkRecord must retain the exact V2L16.3 "
                "MetricVector"
            )
        surface = self.candidate.inspection.surface_model
        if self.benchmark_record.producer != surface.producer:
            raise ValueError(
                "surface_candidate_benchmark BenchmarkRecord producer must match SurfaceModel"
            )
        if self.benchmark_record.quality_mode is not QualityMode.QUALITY:
            raise ValueError(
                "surface_candidate_benchmark BenchmarkRecord must use QualityMode.QUALITY"
            )
        if self.benchmark_record.performance_evidence:
            raise ValueError(
                "surface_candidate_benchmark must not encode support annotations as "
                "performance evidence"
            )
        if self.benchmark_record.comparison_baseline is not None:
            raise ValueError(
                "surface_candidate_benchmark must not encode a comparison baseline"
            )


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkResult:
    """Deterministic controlled surface benchmark with no ranking or certification."""

    request: SurfaceBenchmarkRequest
    candidates: tuple[SurfaceCandidateBenchmarkResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, SurfaceBenchmarkRequest):
            raise TypeError("surface_benchmark_result.request must be SurfaceBenchmarkRequest")
        if not isinstance(self.candidates, tuple):
            raise TypeError("surface_benchmark_result.candidates must be an immutable tuple")
        if any(not isinstance(item, SurfaceCandidateBenchmarkResult) for item in self.candidates):
            raise TypeError(
                "surface_benchmark_result.candidates members must be "
                "SurfaceCandidateBenchmarkResult"
            )
        if tuple(item.candidate for item in self.candidates) != self.request.candidates:
            raise ValueError(
                "surface_benchmark_result candidates must exactly match request order"
            )
        for item in self.candidates:
            if item.benchmark_record.fixture != self.request.fixture:
                raise ValueError(
                    "surface_benchmark_result records must preserve shared fixture identity"
                )
            if item.benchmark_record.hardware != self.request.hardware:
                raise ValueError(
                    "surface_benchmark_result records must preserve shared hardware identity"
                )


def benchmark_surface_candidates(request: SurfaceBenchmarkRequest) -> SurfaceBenchmarkResult:
    """Retain descriptive topology and negative support evidence without selection."""

    if not isinstance(request, SurfaceBenchmarkRequest):
        raise TypeError("request must be SurfaceBenchmarkRequest")

    results: list[SurfaceCandidateBenchmarkResult] = []
    for candidate in request.candidates:
        suitability = evaluate_surface_suitability(candidate.inspection)
        surface = candidate.inspection.surface_model
        record = BenchmarkRecord(
            record_id=_benchmark_record_id(request, candidate),
            fixture=request.fixture,
            producer=surface.producer,
            quality_mode=QualityMode.QUALITY,
            hardware=request.hardware,
            metrics=suitability.metrics,
        )
        results.append(
            SurfaceCandidateBenchmarkResult(
                candidate=candidate,
                suitability=suitability,
                benchmark_record=record,
            )
        )

    return SurfaceBenchmarkResult(
        request=request,
        candidates=tuple(results),
    )


__all__ = [
    "SURFACE_BENCHMARK_IMPLEMENTATION",
    "SURFACE_BENCHMARK_REVISION",
    "SURFACE_BENCHMARK_VERSION",
    "SurfaceBenchmarkCandidate",
    "SurfaceBenchmarkRequest",
    "SurfaceBenchmarkResult",
    "SurfaceCandidateBenchmarkResult",
    "benchmark_surface_candidates",
]
