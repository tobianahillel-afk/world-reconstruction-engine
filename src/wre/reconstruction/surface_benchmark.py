from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactRef
from wre.domain.benchmarks import (
    BenchmarkFixtureIdentity,
    BenchmarkRecord,
    BenchmarkRecordId,
)
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.quality import QualityMode
from wre.domain.surface_support import SurfaceSupportMap, SurfaceSupportRegion
from wre.reconstruction.surface_suitability import (
    SurfaceMeshInspection,
    SurfaceSuitabilityReport,
    evaluate_surface_suitability,
)


class SurfaceSupportAnnotationState(StrEnum):
    """Presence state for explicit negative support evidence in one benchmark result."""

    UNKNOWN = "unknown"
    NEGATIVE_ANNOTATIONS = "negative_annotations"


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkCandidate:
    """One immutable physical-surface benchmark candidate.

    The optional support map is negative evidence only. A missing map means unknown,
    never zero unsupported regions or positive support completeness.
    """

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
                    "surface_benchmark_candidate support_map must reference the exact "
                    "SurfaceModel value retained by inspection"
                )
            if self.support_map.local_frame_id != self.inspection.surface_model.local_frame_id:
                raise ValueError(
                    "surface_benchmark_candidate support_map must preserve the exact "
                    "SurfaceModel LocalFrameId"
                )


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkRequest:
    """Controlled benchmark request over immutable canonical physical-surface evidence."""

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
            raise ValueError("surface_benchmark candidate SurfaceModel artifacts must be unique")
        if identities != tuple(sorted(identities)):
            raise ValueError(
                "surface_benchmark candidates must use canonical SurfaceModel "
                "ArtifactId/ArtifactKind order"
            )


def _benchmark_record_id(
    request: SurfaceBenchmarkRequest,
    candidate: SurfaceBenchmarkCandidate,
) -> BenchmarkRecordId:
    surface = candidate.inspection.surface_model
    document = {
        "fixture_id": request.fixture.fixture_id.value,
        "fixture_sha256": request.fixture.sha256.value,
        "hardware_sha256": request.hardware.sha256.value,
        "surface_artifact_id": surface.artifact_ref.artifact_id.value,
        "surface_artifact_kind": surface.artifact_ref.artifact_kind.value,
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
    """Descriptive controlled evidence for one retained physical SurfaceModel."""

    candidate: SurfaceBenchmarkCandidate
    suitability: SurfaceSuitabilityReport
    support_annotation_state: SurfaceSupportAnnotationState
    support_map_artifact_ref: ArtifactRef | None
    support_regions: tuple[SurfaceSupportRegion, ...] | None
    benchmark_record: BenchmarkRecord

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, SurfaceBenchmarkCandidate):
            raise TypeError(
                "surface_candidate_benchmark_result.candidate must be SurfaceBenchmarkCandidate"
            )
        if not isinstance(self.suitability, SurfaceSuitabilityReport):
            raise TypeError(
                "surface_candidate_benchmark_result.suitability must be SurfaceSuitabilityReport"
            )
        if not isinstance(self.support_annotation_state, SurfaceSupportAnnotationState):
            raise TypeError(
                "surface_candidate_benchmark_result.support_annotation_state must be "
                "SurfaceSupportAnnotationState"
            )
        if self.support_map_artifact_ref is not None and not isinstance(
            self.support_map_artifact_ref, ArtifactRef
        ):
            raise TypeError(
                "surface_candidate_benchmark_result.support_map_artifact_ref must be "
                "ArtifactRef when present"
            )
        if self.support_regions is not None:
            if not isinstance(self.support_regions, tuple):
                raise TypeError(
                    "surface_candidate_benchmark_result.support_regions must be an "
                    "immutable tuple when present"
                )
            if any(not isinstance(item, SurfaceSupportRegion) for item in self.support_regions):
                raise TypeError(
                    "surface_candidate_benchmark_result.support_regions members must be "
                    "SurfaceSupportRegion"
                )
        if not isinstance(self.benchmark_record, BenchmarkRecord):
            raise TypeError(
                "surface_candidate_benchmark_result.benchmark_record must be BenchmarkRecord"
            )

        if self.suitability.inspection != self.candidate.inspection:
            raise ValueError(
                "surface_candidate_benchmark_result suitability must summarize the exact "
                "candidate inspection"
            )

        support_map = self.candidate.support_map
        if support_map is None:
            if self.support_annotation_state is not SurfaceSupportAnnotationState.UNKNOWN:
                raise ValueError(
                    "surface_candidate_benchmark_result missing support annotation must "
                    "remain unknown"
                )
            if self.support_map_artifact_ref is not None or self.support_regions is not None:
                raise ValueError(
                    "surface_candidate_benchmark_result unknown support annotation must "
                    "retain no support-map artifact or region tuple"
                )
        else:
            if (
                self.support_annotation_state
                is not SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS
            ):
                raise ValueError(
                    "surface_candidate_benchmark_result present support map must remain "
                    "negative_annotations"
                )
            if self.support_map_artifact_ref != support_map.artifact_ref:
                raise ValueError(
                    "surface_candidate_benchmark_result must retain exact support-map "
                    "ArtifactRef identity"
                )
            if self.support_regions != support_map.regions:
                raise ValueError(
                    "surface_candidate_benchmark_result must retain exact ordered "
                    "negative support regions"
                )

        surface = self.candidate.inspection.surface_model
        if self.benchmark_record.producer != surface.producer:
            raise ValueError(
                "surface_candidate_benchmark_result BenchmarkRecord producer must match "
                "SurfaceModel producer"
            )
        if self.benchmark_record.quality_mode is not QualityMode.QUALITY:
            raise ValueError(
                "surface_candidate_benchmark_result BenchmarkRecord must use "
                "QualityMode.QUALITY"
            )
        if self.benchmark_record.metrics != self.suitability.metrics:
            raise ValueError(
                "surface_candidate_benchmark_result BenchmarkRecord must retain the exact "
                "V2L16.3 descriptive MetricVector"
            )
        if self.benchmark_record.performance_evidence != ():
            raise ValueError(
                "surface_candidate_benchmark_result must not synthesize performance evidence"
            )
        if self.benchmark_record.comparison_baseline is not None:
            raise ValueError(
                "surface_candidate_benchmark_result must not encode a comparison baseline"
            )


@dataclass(frozen=True, slots=True)
class SurfaceBenchmarkResult:
    """Deterministic controlled surface benchmark result with no selection semantics."""

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


def benchmark_surface_candidates(
    request: SurfaceBenchmarkRequest,
) -> SurfaceBenchmarkResult:
    """Compose descriptive topology and explicit negative support evidence without selection."""

    if not isinstance(request, SurfaceBenchmarkRequest):
        raise TypeError("request must be SurfaceBenchmarkRequest")

    results: list[SurfaceCandidateBenchmarkResult] = []
    for candidate in request.candidates:
        suitability = evaluate_surface_suitability(candidate.inspection)
        support_map = candidate.support_map
        if support_map is None:
            support_state = SurfaceSupportAnnotationState.UNKNOWN
            support_map_artifact_ref = None
            support_regions = None
        else:
            support_state = SurfaceSupportAnnotationState.NEGATIVE_ANNOTATIONS
            support_map_artifact_ref = support_map.artifact_ref
            support_regions = support_map.regions

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
                support_annotation_state=support_state,
                support_map_artifact_ref=support_map_artifact_ref,
                support_regions=support_regions,
                benchmark_record=record,
            )
        )

    return SurfaceBenchmarkResult(
        request=request,
        candidates=tuple(results),
    )


__all__ = [
    "SurfaceBenchmarkCandidate",
    "SurfaceBenchmarkRequest",
    "SurfaceBenchmarkResult",
    "SurfaceCandidateBenchmarkResult",
    "SurfaceSupportAnnotationState",
    "benchmark_surface_candidates",
]
