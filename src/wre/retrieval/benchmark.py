from __future__ import annotations

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
from wre.domain.observations import ObservationId
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.quality import QualityMode

RETRIEVAL_CANDIDATE_COUNT = MetricDescriptor(
    name=MetricName("retrieval.candidate_count"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("total"),
)
RETRIEVAL_CANDIDATE_RECALL = MetricDescriptor(
    name=MetricName("retrieval.candidate_recall"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("fraction"),
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("ratio"),
)
RETRIEVAL_INCREMENTAL_CANDIDATE_RECALL = MetricDescriptor(
    name=MetricName("retrieval.incremental_candidate_recall"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("fraction"),
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("difference"),
)
RETRIEVAL_RELEVANT_PAIR_COUNT = MetricDescriptor(
    name=MetricName("retrieval.relevant_pair_count"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("total"),
)
RETRIEVAL_RETRIEVED_RELEVANT_PAIR_COUNT = MetricDescriptor(
    name=MetricName("retrieval.retrieved_relevant_pair_count"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("total"),
)
RETRIEVAL_SOURCE_COUNT = MetricDescriptor(
    name=MetricName("retrieval.source_count"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("total"),
)
RETRIEVAL_UNION_CANDIDATE_COUNT = MetricDescriptor(
    name=MetricName("retrieval.union_candidate_count"),
    dimension=MetricDimension("retrieval"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("total"),
)


def _pair_key(
    observation_id1: ObservationId,
    observation_id2: ObservationId,
) -> tuple[str, str]:
    return (observation_id1.value, observation_id2.value)


def _candidate_key(candidate: PairCandidate) -> tuple[str, str]:
    return _pair_key(candidate.observation_id1, candidate.observation_id2)


@dataclass(frozen=True, slots=True)
class RetrievalRelevantPair:
    """Benchmark-only relevant unordered observation pair."""

    observation_id1: ObservationId
    observation_id2: ObservationId

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("retrieval_relevant_pair.observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("retrieval_relevant_pair.observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "retrieval_relevant_pair endpoints must be distinct and canonically ordered"
            )


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkFixture:
    """Explicit benchmark identity, membership and benchmark-only relevant-pair labels."""

    identity: BenchmarkFixtureIdentity
    observation_ids: tuple[ObservationId, ...]
    relevant_pairs: tuple[RetrievalRelevantPair, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, BenchmarkFixtureIdentity):
            raise TypeError("retrieval_benchmark_fixture.identity must be BenchmarkFixtureIdentity")
        if not isinstance(self.observation_ids, tuple):
            raise TypeError(
                "retrieval_benchmark_fixture.observation_ids must be an immutable tuple"
            )
        if not self.observation_ids:
            raise ValueError("retrieval_benchmark_fixture.observation_ids must not be empty")
        if any(not isinstance(item, ObservationId) for item in self.observation_ids):
            raise TypeError(
                "retrieval_benchmark_fixture.observation_ids must contain ObservationId values"
            )
        observation_values = tuple(item.value for item in self.observation_ids)
        if len(observation_values) != len(set(observation_values)):
            raise ValueError(
                "retrieval_benchmark_fixture.observation_ids must not contain duplicates"
            )
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError(
                "retrieval_benchmark_fixture.observation_ids must use canonical lexical order"
            )

        if not isinstance(self.relevant_pairs, tuple):
            raise TypeError(
                "retrieval_benchmark_fixture.relevant_pairs must be an immutable tuple"
            )
        if not self.relevant_pairs:
            raise ValueError("retrieval_benchmark_fixture.relevant_pairs must not be empty")
        if any(not isinstance(item, RetrievalRelevantPair) for item in self.relevant_pairs):
            raise TypeError(
                "retrieval_benchmark_fixture.relevant_pairs must contain "
                "RetrievalRelevantPair values"
            )
        pair_keys = tuple(
            _pair_key(item.observation_id1, item.observation_id2)
            for item in self.relevant_pairs
        )
        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError(
                "retrieval_benchmark_fixture.relevant_pairs must not contain duplicates"
            )
        if pair_keys != tuple(sorted(pair_keys)):
            raise ValueError(
                "retrieval_benchmark_fixture.relevant_pairs must use canonical pair order"
            )
        membership = set(observation_values)
        if any(left not in membership or right not in membership for left, right in pair_keys):
            raise ValueError(
                "retrieval_benchmark_fixture relevant-pair endpoints must belong to observations"
            )


@dataclass(frozen=True, slots=True)
class RetrievalSourceCandidateSet:
    """One already-produced source-specific canonical PairCandidate set."""

    source_id: PairCandidateSourceId
    candidates: tuple[PairCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, PairCandidateSourceId):
            raise TypeError("retrieval_source_candidate_set.source_id must be PairCandidateSourceId")
        if not isinstance(self.candidates, tuple):
            raise TypeError(
                "retrieval_source_candidate_set.candidates must be an immutable tuple"
            )
        if any(not isinstance(item, PairCandidate) for item in self.candidates):
            raise TypeError(
                "retrieval_source_candidate_set.candidates must contain PairCandidate values"
            )
        keys = tuple(_candidate_key(item) for item in self.candidates)
        if len(keys) != len(set(keys)):
            raise ValueError("retrieval_source_candidate_set.candidates must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError(
                "retrieval_source_candidate_set.candidates must use canonical pair order"
            )
        for candidate in self.candidates:
            if len(candidate.sources) != 1:
                raise ValueError(
                    "source-specific retrieval candidates must contain exactly one source"
                )
            if candidate.sources[0].source_id != self.source_id:
                raise ValueError(
                    "source-specific candidate source ID must match candidate-set source ID"
                )


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkRequest:
    fixture: RetrievalBenchmarkFixture
    source_sets: tuple[RetrievalSourceCandidateSet, ...]
    metric_provenance: MetricProvenance
    comparison_source_id: PairCandidateSourceId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, RetrievalBenchmarkFixture):
            raise TypeError("retrieval_benchmark_request.fixture must be RetrievalBenchmarkFixture")
        if not isinstance(self.source_sets, tuple):
            raise TypeError("retrieval_benchmark_request.source_sets must be an immutable tuple")
        if not self.source_sets:
            raise ValueError("retrieval_benchmark_request.source_sets must not be empty")
        if any(not isinstance(item, RetrievalSourceCandidateSet) for item in self.source_sets):
            raise TypeError(
                "retrieval_benchmark_request.source_sets must contain "
                "RetrievalSourceCandidateSet values"
            )
        source_values = tuple(item.source_id.value for item in self.source_sets)
        if len(source_values) != len(set(source_values)):
            raise ValueError("retrieval_benchmark_request source IDs must be unique")
        if source_values != tuple(sorted(source_values)):
            raise ValueError(
                "retrieval_benchmark_request source sets must use canonical source-ID order"
            )
        if not isinstance(self.metric_provenance, MetricProvenance):
            raise TypeError(
                "retrieval_benchmark_request.metric_provenance must be MetricProvenance"
            )
        if self.comparison_source_id is not None:
            if not isinstance(self.comparison_source_id, PairCandidateSourceId):
                raise TypeError(
                    "retrieval_benchmark_request.comparison_source_id must be "
                    "PairCandidateSourceId when present"
                )
            if self.comparison_source_id.value not in set(source_values):
                raise ValueError(
                    "retrieval_benchmark_request comparison source must be supplied"
                )

        membership = {item.value for item in self.fixture.observation_ids}
        for source_set in self.source_sets:
            for candidate in source_set.candidates:
                if (
                    candidate.observation_id1.value not in membership
                    or candidate.observation_id2.value not in membership
                ):
                    raise ValueError(
                        "retrieval source candidate endpoints must belong to benchmark fixture"
                    )


@dataclass(frozen=True, slots=True)
class RetrievalSourceBenchmarkResult:
    source_id: PairCandidateSourceId
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, PairCandidateSourceId):
            raise TypeError(
                "retrieval_source_benchmark_result.source_id must be PairCandidateSourceId"
            )
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("retrieval_source_benchmark_result.metrics must be MetricVector")


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkComparison:
    source_id: PairCandidateSourceId
    incremental_candidate_recall: float

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, PairCandidateSourceId):
            raise TypeError(
                "retrieval_benchmark_comparison.source_id must be PairCandidateSourceId"
            )
        if type(self.incremental_candidate_recall) is not float:
            raise TypeError(
                "retrieval_benchmark_comparison.incremental_candidate_recall must be float"
            )
        if not 0.0 <= self.incremental_candidate_recall <= 1.0:
            raise ValueError(
                "retrieval_benchmark_comparison incremental recall must be within 0..1"
            )


@dataclass(frozen=True, slots=True)
class RetrievalBenchmarkResult:
    fixture: RetrievalBenchmarkFixture
    source_results: tuple[RetrievalSourceBenchmarkResult, ...]
    union_candidates: tuple[PairCandidate, ...]
    union_metrics: MetricVector
    comparison: RetrievalBenchmarkComparison | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, RetrievalBenchmarkFixture):
            raise TypeError("retrieval_benchmark_result.fixture must be RetrievalBenchmarkFixture")
        if not isinstance(self.source_results, tuple):
            raise TypeError(
                "retrieval_benchmark_result.source_results must be an immutable tuple"
            )
        if any(not isinstance(item, RetrievalSourceBenchmarkResult) for item in self.source_results):
            raise TypeError(
                "retrieval_benchmark_result.source_results must contain "
                "RetrievalSourceBenchmarkResult values"
            )
        source_ids = tuple(item.source_id.value for item in self.source_results)
        if not source_ids:
            raise ValueError("retrieval_benchmark_result.source_results must not be empty")
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("retrieval_benchmark_result source results must be unique")
        if source_ids != tuple(sorted(source_ids)):
            raise ValueError(
                "retrieval_benchmark_result source results must use canonical source order"
            )
        if not isinstance(self.union_candidates, tuple):
            raise TypeError(
                "retrieval_benchmark_result.union_candidates must be an immutable tuple"
            )
        if any(not isinstance(item, PairCandidate) for item in self.union_candidates):
            raise TypeError(
                "retrieval_benchmark_result.union_candidates must contain PairCandidate values"
            )
        union_keys = tuple(_candidate_key(item) for item in self.union_candidates)
        if len(union_keys) != len(set(union_keys)):
            raise ValueError("retrieval_benchmark_result.union_candidates must be unique")
        if union_keys != tuple(sorted(union_keys)):
            raise ValueError(
                "retrieval_benchmark_result.union_candidates must use canonical pair order"
            )
        if not isinstance(self.union_metrics, MetricVector):
            raise TypeError("retrieval_benchmark_result.union_metrics must be MetricVector")
        if self.comparison is not None:
            if not isinstance(self.comparison, RetrievalBenchmarkComparison):
                raise TypeError(
                    "retrieval_benchmark_result.comparison must be "
                    "RetrievalBenchmarkComparison when present"
                )
            if self.comparison.source_id.value not in set(source_ids):
                raise ValueError(
                    "retrieval_benchmark_result comparison source must have a source result"
                )


def union_retrieval_candidate_sets(
    source_sets: tuple[RetrievalSourceCandidateSet, ...],
) -> tuple[PairCandidate, ...]:
    """Union already-produced source candidates only through canonical PairCandidate merge."""

    if not isinstance(source_sets, tuple):
        raise TypeError("source_sets must be an immutable tuple")
    if any(not isinstance(item, RetrievalSourceCandidateSet) for item in source_sets):
        raise TypeError("source_sets must contain RetrievalSourceCandidateSet values")
    source_ids = tuple(item.source_id.value for item in source_sets)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("source_sets must not repeat source IDs")
    ordered_sets = tuple(sorted(source_sets, key=lambda item: item.source_id.value))
    return merge_pair_candidates(
        tuple(candidate for source_set in ordered_sets for candidate in source_set.candidates)
    )


def _metric(
    descriptor: MetricDescriptor,
    value: float,
    provenance: MetricProvenance,
) -> MetricObservation:
    return MetricObservation(
        descriptor=descriptor,
        value=value,
        provenance=provenance,
    )


def _recall_metrics(
    *,
    candidates: tuple[PairCandidate, ...],
    fixture: RetrievalBenchmarkFixture,
    provenance: MetricProvenance,
) -> tuple[MetricVector, float]:
    relevant_keys = {
        _pair_key(item.observation_id1, item.observation_id2)
        for item in fixture.relevant_pairs
    }
    candidate_keys = {_candidate_key(item) for item in candidates}
    retrieved_count = len(relevant_keys.intersection(candidate_keys))
    relevant_count = len(relevant_keys)
    recall = retrieved_count / relevant_count
    metrics = MetricVector(
        observations=(
            _metric(RETRIEVAL_CANDIDATE_COUNT, float(len(candidates)), provenance),
            _metric(RETRIEVAL_CANDIDATE_RECALL, float(recall), provenance),
            _metric(RETRIEVAL_RELEVANT_PAIR_COUNT, float(relevant_count), provenance),
            _metric(
                RETRIEVAL_RETRIEVED_RELEVANT_PAIR_COUNT,
                float(retrieved_count),
                provenance,
            ),
        )
    )
    return metrics, float(recall)


def benchmark_retrieval_candidates(
    request: RetrievalBenchmarkRequest,
) -> RetrievalBenchmarkResult:
    """Measure source-specific and union candidate recall against explicit fixture labels."""

    if not isinstance(request, RetrievalBenchmarkRequest):
        raise TypeError("request must be RetrievalBenchmarkRequest")

    source_results: list[RetrievalSourceBenchmarkResult] = []
    source_recalls: dict[str, float] = {}
    for source_set in request.source_sets:
        metrics, recall = _recall_metrics(
            candidates=source_set.candidates,
            fixture=request.fixture,
            provenance=request.metric_provenance,
        )
        source_results.append(
            RetrievalSourceBenchmarkResult(
                source_id=source_set.source_id,
                metrics=metrics,
            )
        )
        source_recalls[source_set.source_id.value] = recall

    union_candidates = union_retrieval_candidate_sets(request.source_sets)
    base_union_metrics, union_recall = _recall_metrics(
        candidates=union_candidates,
        fixture=request.fixture,
        provenance=request.metric_provenance,
    )
    observations = list(base_union_metrics.observations)
    observations.extend(
        (
            _metric(
                RETRIEVAL_SOURCE_COUNT,
                float(len(request.source_sets)),
                request.metric_provenance,
            ),
            _metric(
                RETRIEVAL_UNION_CANDIDATE_COUNT,
                float(len(union_candidates)),
                request.metric_provenance,
            ),
        )
    )

    comparison: RetrievalBenchmarkComparison | None = None
    if request.comparison_source_id is not None:
        baseline_recall = source_recalls[request.comparison_source_id.value]
        incremental_recall = union_recall - baseline_recall
        if incremental_recall < 0.0:
            raise RuntimeError("retrieval union recall cannot be below a constituent source")
        observations.append(
            _metric(
                RETRIEVAL_INCREMENTAL_CANDIDATE_RECALL,
                float(incremental_recall),
                request.metric_provenance,
            )
        )
        comparison = RetrievalBenchmarkComparison(
            source_id=request.comparison_source_id,
            incremental_candidate_recall=float(incremental_recall),
        )

    observations.sort(key=lambda item: item.descriptor.name.value)
    return RetrievalBenchmarkResult(
        fixture=request.fixture,
        source_results=tuple(source_results),
        union_candidates=union_candidates,
        union_metrics=MetricVector(observations=tuple(observations)),
        comparison=comparison,
    )


def build_retrieval_benchmark_record(
    result: RetrievalBenchmarkResult,
    *,
    record_id: BenchmarkRecordId,
    producer: ArtifactProducerIdentity,
    quality_mode: QualityMode,
    hardware: HardwareRuntimeIdentity,
    source_id: PairCandidateSourceId | None = None,
    performance_evidence: tuple[ArtifactRef, ...] = (),
    comparison_baseline: BenchmarkRecordId | None = None,
) -> BenchmarkRecord:
    """Wrap explicit retrieval benchmark metrics in the existing BenchmarkRecord contract."""

    if not isinstance(result, RetrievalBenchmarkResult):
        raise TypeError("result must be RetrievalBenchmarkResult")
    if not isinstance(record_id, BenchmarkRecordId):
        raise TypeError("record_id must be BenchmarkRecordId")
    if not isinstance(producer, ArtifactProducerIdentity):
        raise TypeError("producer must be ArtifactProducerIdentity")
    if not isinstance(quality_mode, QualityMode):
        raise TypeError("quality_mode must be QualityMode")
    if not isinstance(hardware, HardwareRuntimeIdentity):
        raise TypeError("hardware must be HardwareRuntimeIdentity")
    if source_id is not None and not isinstance(source_id, PairCandidateSourceId):
        raise TypeError("source_id must be PairCandidateSourceId when present")

    if source_id is None:
        metrics = result.union_metrics
    else:
        matching = tuple(
            item.metrics for item in result.source_results if item.source_id == source_id
        )
        if len(matching) != 1:
            raise ValueError("source_id must identify exactly one benchmark source result")
        metrics = matching[0]

    return BenchmarkRecord(
        record_id=record_id,
        fixture=result.fixture.identity,
        producer=producer,
        quality_mode=quality_mode,
        hardware=hardware,
        metrics=metrics,
        performance_evidence=performance_evidence,
        comparison_baseline=comparison_baseline,
    )
