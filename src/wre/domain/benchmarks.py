from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.metrics import MetricVector
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.quality import QualityMode

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: object, context: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


@dataclass(frozen=True, slots=True, order=True)
class BenchmarkRecordId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "benchmark_record_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class BenchmarkFixtureId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "benchmark_fixture_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class BenchmarkFixtureIdentity:
    fixture_id: BenchmarkFixtureId
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.fixture_id, BenchmarkFixtureId):
            raise TypeError("benchmark_fixture.fixture_id must be BenchmarkFixtureId")
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("benchmark_fixture.sha256 must be Sha256Digest")


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    record_id: BenchmarkRecordId
    fixture: BenchmarkFixtureIdentity
    producer: ArtifactProducerIdentity
    quality_mode: QualityMode
    hardware: HardwareRuntimeIdentity
    metrics: MetricVector
    performance_evidence: tuple[ArtifactRef, ...] = ()
    comparison_baseline: BenchmarkRecordId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_id, BenchmarkRecordId):
            raise TypeError("benchmark_record.record_id must be BenchmarkRecordId")
        if not isinstance(self.fixture, BenchmarkFixtureIdentity):
            raise TypeError("benchmark_record.fixture must be BenchmarkFixtureIdentity")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("benchmark_record.producer must be ArtifactProducerIdentity")
        if not isinstance(self.quality_mode, QualityMode):
            raise TypeError("benchmark_record.quality_mode must be QualityMode")
        if not isinstance(self.hardware, HardwareRuntimeIdentity):
            raise TypeError("benchmark_record.hardware must be HardwareRuntimeIdentity")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("benchmark_record.metrics must be MetricVector")
        if not isinstance(self.performance_evidence, tuple):
            raise TypeError("benchmark_record.performance_evidence must be an immutable tuple")
        if any(not isinstance(item, ArtifactRef) for item in self.performance_evidence):
            raise TypeError("benchmark_record.performance_evidence members must be ArtifactRef")

        evidence_keys = [
            (item.artifact_id.value, item.artifact_kind.value) for item in self.performance_evidence
        ]
        if len(set(self.performance_evidence)) != len(self.performance_evidence):
            raise ValueError("benchmark_record.performance_evidence members must be unique")
        if evidence_keys != sorted(evidence_keys):
            raise ValueError(
                "benchmark_record.performance_evidence must use canonical "
                "ArtifactId/ArtifactKind order"
            )

        if self.comparison_baseline is not None and not isinstance(
            self.comparison_baseline, BenchmarkRecordId
        ):
            raise TypeError(
                "benchmark_record.comparison_baseline must be BenchmarkRecordId when present"
            )
        if self.comparison_baseline == self.record_id:
            raise ValueError(
                "benchmark_record.comparison_baseline must not reference its own record"
            )
