from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import pytest

import wre.retrieval.benchmark as benchmark_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    BenchmarkFixtureId,
    BenchmarkFixtureIdentity,
    BenchmarkRecordId,
    ConfigurationIdentity,
    HardwareRuntimeIdentity,
    MetricAggregation,
    MetricDirection,
    MetricProvenance,
    ObservationId,
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    ProducerRef,
    QualityMode,
    Sha256Digest,
)
from wre.retrieval import (
    GPS_PAIR_CANDIDATE_SOURCE_ID,
    SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
    SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
    VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID,
)
from wre.retrieval.benchmark import (
    RETRIEVAL_CANDIDATE_COUNT,
    RETRIEVAL_CANDIDATE_RECALL,
    RETRIEVAL_INCREMENTAL_CANDIDATE_RECALL,
    RETRIEVAL_RELEVANT_PAIR_COUNT,
    RETRIEVAL_RETRIEVED_RELEVANT_PAIR_COUNT,
    RETRIEVAL_SOURCE_COUNT,
    RETRIEVAL_UNION_CANDIDATE_COUNT,
    RetrievalBenchmarkComparison,
    RetrievalBenchmarkFixture,
    RetrievalBenchmarkRequest,
    RetrievalBenchmarkResult,
    RetrievalRelevantPair,
    RetrievalSourceBenchmarkResult,
    RetrievalSourceCandidateSet,
    benchmark_retrieval_candidates,
    build_retrieval_benchmark_record,
    union_retrieval_candidate_sets,
)


def _artifact(source: str, suffix: str = "evidence") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:{source}:{suffix}"),
        artifact_kind=ArtifactKind("retrieval.evidence"),
    )


def _candidate(
    left: str,
    right: str,
    source_id: PairCandidateSourceId,
    *refs: ArtifactRef,
) -> PairCandidate:
    actual_refs = refs or (_artifact(source_id.value),)
    return PairCandidate(
        observation_id1=ObservationId(left),
        observation_id2=ObservationId(right),
        sources=(
            PairCandidateSource(
                source_id=source_id,
                evidence_refs=tuple(actual_refs),
            ),
        ),
    )


def _relevant(left: str, right: str) -> RetrievalRelevantPair:
    return RetrievalRelevantPair(
        observation_id1=ObservationId(left),
        observation_id2=ObservationId(right),
    )


def _fixture(
    *,
    observation_ids: tuple[str, ...] = ("obs:a", "obs:b", "obs:c", "obs:d"),
    relevant_pairs: tuple[tuple[str, str], ...] = (
        ("obs:a", "obs:b"),
        ("obs:a", "obs:c"),
        ("obs:c", "obs:d"),
    ),
) -> RetrievalBenchmarkFixture:
    return RetrievalBenchmarkFixture(
        identity=BenchmarkFixtureIdentity(
            fixture_id=BenchmarkFixtureId("fixture.retrieval.v1"),
            sha256=Sha256Digest("a" * 64),
        ),
        observation_ids=tuple(ObservationId(value) for value in observation_ids),
        relevant_pairs=tuple(_relevant(left, right) for left, right in relevant_pairs),
    )


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.retrieval.benchmark",
            version="1",
            revision="fixture",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("b" * 64)),
    )


def _provenance() -> MetricProvenance:
    return MetricProvenance(
        evaluator=_producer(),
        input_artifacts=(_artifact("benchmark", "fixture"),),
    )


def _hardware() -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(sha256=Sha256Digest("c" * 64))


def _set(
    source_id: PairCandidateSourceId,
    *candidates: PairCandidate,
) -> RetrievalSourceCandidateSet:
    return RetrievalSourceCandidateSet(
        source_id=source_id,
        candidates=tuple(candidates),
    )


def _metric_value(result_metrics: Any, name: str) -> float:
    matches = [
        observation.value
        for observation in result_metrics.observations
        if observation.descriptor.name.value == name
    ]
    assert len(matches) == 1
    return matches[0]


def _request(
    source_sets: tuple[RetrievalSourceCandidateSet, ...],
    *,
    fixture: RetrievalBenchmarkFixture | None = None,
    comparison_source_id: PairCandidateSourceId | None = None,
) -> RetrievalBenchmarkRequest:
    return RetrievalBenchmarkRequest(
        fixture=fixture or _fixture(),
        source_sets=source_sets,
        metric_provenance=_provenance(),
        comparison_source_id=comparison_source_id,
    )


def test_relevant_pair_is_exact_immutable_and_contains_no_truth_extensions() -> None:
    pair = _relevant("obs:a", "obs:b")

    assert tuple(field.name for field in fields(RetrievalRelevantPair)) == (
        "observation_id1",
        "observation_id2",
    )
    assert hash(pair) == hash(_relevant("obs:a", "obs:b"))
    with pytest.raises(FrozenInstanceError):
        pair.observation_id1 = ObservationId("obs:x")  # type: ignore[misc]

    for forbidden in (
        "score",
        "confidence",
        "source",
        "geometry",
        "scene",
        "time",
        "geolocation",
        "quality_decision",
    ):
        assert forbidden not in RetrievalRelevantPair.__dataclass_fields__

    with pytest.raises(TypeError, match="observation_id1"):
        RetrievalRelevantPair(
            observation_id1=cast(Any, "obs:a"),
            observation_id2=ObservationId("obs:b"),
        )
    with pytest.raises(TypeError, match="observation_id2"):
        RetrievalRelevantPair(
            observation_id1=ObservationId("obs:a"),
            observation_id2=cast(Any, "obs:b"),
        )
    with pytest.raises(ValueError, match="canonically ordered"):
        _relevant("obs:b", "obs:a")
    with pytest.raises(ValueError, match="canonically ordered"):
        _relevant("obs:a", "obs:a")


def test_fixture_requires_canonical_membership_and_nonempty_relevant_truth() -> None:
    fixture = _fixture()

    assert tuple(field.name for field in fields(RetrievalBenchmarkFixture)) == (
        "identity",
        "observation_ids",
        "relevant_pairs",
    )
    assert fixture.observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
        ObservationId("obs:c"),
        ObservationId("obs:d"),
    )
    assert len(fixture.relevant_pairs) == 3

    with pytest.raises(ValueError, match="observation_ids must not be empty"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=(),
            relevant_pairs=fixture.relevant_pairs,
        )
    with pytest.raises(ValueError, match="relevant_pairs must not be empty"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=fixture.observation_ids,
            relevant_pairs=(),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
            relevant_pairs=(_relevant("obs:a", "obs:b"),),
        )
    with pytest.raises(ValueError, match="duplicates"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=(ObservationId("obs:a"), ObservationId("obs:a")),
            relevant_pairs=(_relevant("obs:a", "obs:b"),),
        )
    with pytest.raises(ValueError, match="canonical pair order"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=fixture.observation_ids,
            relevant_pairs=(
                _relevant("obs:c", "obs:d"),
                _relevant("obs:a", "obs:b"),
            ),
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=fixture.observation_ids,
            relevant_pairs=(
                _relevant("obs:a", "obs:b"),
                _relevant("obs:a", "obs:b"),
            ),
        )
    with pytest.raises(ValueError, match="must belong to observations"):
        RetrievalBenchmarkFixture(
            identity=fixture.identity,
            observation_ids=fixture.observation_ids,
            relevant_pairs=(_relevant("obs:a", "obs:z"),),
        )


def test_source_candidate_set_requires_one_matching_source_and_canonical_pairs() -> None:
    source_id = PairCandidateSourceId("source.a")
    first = _candidate("obs:a", "obs:b", source_id)
    second = _candidate("obs:a", "obs:c", source_id)
    source_set = _set(source_id, first, second)

    assert tuple(field.name for field in fields(RetrievalSourceCandidateSet)) == (
        "source_id",
        "candidates",
    )
    assert source_set.candidates == (first, second)
    assert _set(source_id).candidates == ()

    with pytest.raises(ValueError, match="must be unique"):
        _set(source_id, first, first)
    with pytest.raises(ValueError, match="canonical pair order"):
        _set(source_id, second, first)
    with pytest.raises(ValueError, match="source ID must match"):
        _set(source_id, _candidate("obs:a", "obs:b", PairCandidateSourceId("other")))

    multi_source = PairCandidate(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        sources=(
            PairCandidateSource(
                source_id=PairCandidateSourceId("source.a"),
                evidence_refs=(_artifact("source.a"),),
            ),
            PairCandidateSource(
                source_id=PairCandidateSourceId("source.b"),
                evidence_refs=(_artifact("source.b"),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="exactly one source"):
        _set(source_id, multi_source)


def test_request_requires_unique_canonical_sources_and_fixture_membership() -> None:
    sequential = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID),
    )
    learned = _set(
        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:c", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
    )
    request = _request((learned, sequential))

    assert tuple(item.source_id.value for item in request.source_sets) == (
        "selavpr_plus",
        "sequential",
    )

    with pytest.raises(ValueError, match="must not be empty"):
        _request(())
    with pytest.raises(ValueError, match="source IDs must be unique"):
        _request((learned, learned))
    with pytest.raises(ValueError, match="canonical source-ID order"):
        _request((sequential, learned))
    with pytest.raises(ValueError, match="endpoints must belong"):
        _request(
            (
                _set(
                    SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
                    _candidate(
                        "obs:a",
                        "obs:z",
                        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
                    ),
                ),
            )
        )
    with pytest.raises(ValueError, match="comparison source must be supplied"):
        _request(
            (learned,),
            comparison_source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        )


def test_union_delegates_canonical_merge_and_preserves_all_source_evidence() -> None:
    sequential_ref = _artifact("sequential", "a")
    gps_ref = _artifact("gps", "a")
    learned_ref = _artifact("selavpr_plus", "a")
    sequential = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate(
            "obs:a",
            "obs:b",
            SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
            sequential_ref,
        ),
    )
    gps = _set(
        GPS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", GPS_PAIR_CANDIDATE_SOURCE_ID, gps_ref),
    )
    learned = _set(
        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate(
            "obs:a",
            "obs:c",
            SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
            learned_ref,
        ),
    )

    first = union_retrieval_candidate_sets((sequential, learned, gps))
    second = union_retrieval_candidate_sets((gps, sequential, learned))

    assert first == second
    assert tuple(
        (item.observation_id1.value, item.observation_id2.value) for item in first
    ) == (("obs:a", "obs:b"), ("obs:a", "obs:c"))
    assert tuple(source.source_id.value for source in first[0].sources) == (
        "gps",
        "sequential",
    )
    assert first[0].sources[0].evidence_refs == (gps_ref,)
    assert first[0].sources[1].evidence_refs == (sequential_ref,)
    assert first[1].sources[0].evidence_refs == (learned_ref,)


def test_union_collapses_duplicate_evidence_only_through_pair_merge() -> None:
    ref_a = _artifact("sequential", "a")
    ref_b = _artifact("sequential", "b")
    first_set = RetrievalSourceCandidateSet(
        source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        candidates=(
            _candidate(
                "obs:a",
                "obs:b",
                SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
                ref_a,
                ref_b,
            ),
        ),
    )

    merged = union_retrieval_candidate_sets((first_set,))

    assert merged[0].sources[0].evidence_refs == (ref_a, ref_b)


def test_benchmark_measures_zero_partial_full_and_union_recall() -> None:
    sequential = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID),
    )
    gps = _set(
        GPS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:b", "obs:c", GPS_PAIR_CANDIDATE_SOURCE_ID),
    )
    learned = _set(
        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:c", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
        _candidate("obs:c", "obs:d", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
    )
    request = _request((gps, learned, sequential))

    result = benchmark_retrieval_candidates(request)

    by_source = {item.source_id.value: item.metrics for item in result.source_results}
    assert _metric_value(by_source["gps"], "retrieval.candidate_count") == 1.0
    assert _metric_value(by_source["gps"], "retrieval.retrieved_relevant_pair_count") == 0.0
    assert _metric_value(by_source["gps"], "retrieval.candidate_recall") == 0.0
    assert _metric_value(by_source["sequential"], "retrieval.candidate_recall") == pytest.approx(
        1.0 / 3.0
    )
    assert _metric_value(by_source["selavpr_plus"], "retrieval.candidate_recall") == pytest.approx(
        2.0 / 3.0
    )
    assert _metric_value(result.union_metrics, "retrieval.candidate_recall") == 1.0
    assert _metric_value(result.union_metrics, "retrieval.retrieved_relevant_pair_count") == 3.0
    assert _metric_value(result.union_metrics, "retrieval.source_count") == 3.0
    assert _metric_value(result.union_metrics, "retrieval.union_candidate_count") == 4.0


def test_false_positive_candidates_change_count_not_recall_truth() -> None:
    baseline = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID),
    )
    with_false_positive = _set(
        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
        _candidate("obs:b", "obs:d", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
    )
    fixture = _fixture(relevant_pairs=(("obs:a", "obs:b"),))

    result = benchmark_retrieval_candidates(
        _request((with_false_positive,), fixture=fixture)
    )
    metrics = result.source_results[0].metrics

    assert _metric_value(metrics, "retrieval.candidate_count") == 2.0
    assert _metric_value(metrics, "retrieval.retrieved_relevant_pair_count") == 1.0
    assert _metric_value(metrics, "retrieval.candidate_recall") == 1.0
    assert not hasattr(result, "false_scene_pairs")
    assert baseline.candidates[0].observation_id1 == ObservationId("obs:a")


def test_empty_candidate_set_is_valid_and_yields_zero_recall() -> None:
    source_set = _set(GPS_PAIR_CANDIDATE_SOURCE_ID)

    result = benchmark_retrieval_candidates(_request((source_set,)))
    metrics = result.source_results[0].metrics

    assert result.union_candidates == ()
    assert _metric_value(metrics, "retrieval.candidate_count") == 0.0
    assert _metric_value(metrics, "retrieval.retrieved_relevant_pair_count") == 0.0
    assert _metric_value(metrics, "retrieval.candidate_recall") == 0.0
    assert _metric_value(result.union_metrics, "retrieval.candidate_recall") == 0.0


def test_explicit_comparison_records_nonnegative_incremental_recall_only() -> None:
    sequential = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID),
    )
    learned = _set(
        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:c", SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID),
    )

    result = benchmark_retrieval_candidates(
        _request(
            (learned, sequential),
            comparison_source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        )
    )

    assert result.comparison is not None
    assert result.comparison.source_id == SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID
    assert result.comparison.incremental_candidate_recall == pytest.approx(1.0 / 3.0)
    assert _metric_value(
        result.union_metrics,
        "retrieval.incremental_candidate_recall",
    ) == pytest.approx(1.0 / 3.0)
    assert not hasattr(result.comparison, "winner")
    assert not hasattr(result.comparison, "default_source")


def test_metric_descriptors_are_stable_and_preserve_explicit_provenance() -> None:
    descriptors = (
        RETRIEVAL_CANDIDATE_COUNT,
        RETRIEVAL_CANDIDATE_RECALL,
        RETRIEVAL_INCREMENTAL_CANDIDATE_RECALL,
        RETRIEVAL_RELEVANT_PAIR_COUNT,
        RETRIEVAL_RETRIEVED_RELEVANT_PAIR_COUNT,
        RETRIEVAL_SOURCE_COUNT,
        RETRIEVAL_UNION_CANDIDATE_COUNT,
    )

    assert tuple(item.name.value for item in descriptors) == (
        "retrieval.candidate_count",
        "retrieval.candidate_recall",
        "retrieval.incremental_candidate_recall",
        "retrieval.relevant_pair_count",
        "retrieval.retrieved_relevant_pair_count",
        "retrieval.source_count",
        "retrieval.union_candidate_count",
    )
    assert RETRIEVAL_CANDIDATE_COUNT.unit.value == "count"
    assert RETRIEVAL_CANDIDATE_COUNT.direction is MetricDirection.INFORMATIONAL
    assert RETRIEVAL_CANDIDATE_COUNT.aggregation == MetricAggregation("total")
    assert RETRIEVAL_CANDIDATE_RECALL.unit.value == "fraction"
    assert RETRIEVAL_CANDIDATE_RECALL.direction is MetricDirection.HIGHER_IS_BETTER
    assert RETRIEVAL_CANDIDATE_RECALL.aggregation == MetricAggregation("ratio")

    source_set = _set(SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID)
    provenance = _provenance()
    result = benchmark_retrieval_candidates(
        RetrievalBenchmarkRequest(
            fixture=_fixture(),
            source_sets=(source_set,),
            metric_provenance=provenance,
        )
    )
    assert all(
        observation.provenance == provenance
        for observation in result.union_metrics.observations
    )
    assert tuple(
        observation.descriptor.name.value
        for observation in result.union_metrics.observations
    ) == tuple(
        sorted(
            observation.descriptor.name.value
            for observation in result.union_metrics.observations
        )
    )


def test_result_contract_is_canonical_and_contains_no_production_decision() -> None:
    source_set = _set(SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID)
    result = benchmark_retrieval_candidates(_request((source_set,)))

    assert tuple(field.name for field in fields(RetrievalSourceBenchmarkResult)) == (
        "source_id",
        "metrics",
    )
    assert tuple(field.name for field in fields(RetrievalBenchmarkResult)) == (
        "fixture",
        "source_results",
        "union_candidates",
        "union_metrics",
        "comparison",
    )
    for forbidden in (
        "scene_cluster",
        "same_scene",
        "quality_decision",
        "route",
        "winner",
        "default_source",
        "rank",
        "weight",
    ):
        assert forbidden not in RetrievalBenchmarkResult.__dataclass_fields__


def test_build_benchmark_record_preserves_only_caller_supplied_record_context() -> None:
    sequential = _set(
        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
        _candidate("obs:a", "obs:b", SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID),
    )
    result = benchmark_retrieval_candidates(_request((sequential,)))
    producer = _producer()
    hardware = _hardware()
    evidence = (_artifact("performance", "trace"),)
    baseline_id = BenchmarkRecordId("benchmark.retrieval.baseline")

    union_record = build_retrieval_benchmark_record(
        result,
        record_id=BenchmarkRecordId("benchmark.retrieval.union"),
        producer=producer,
        quality_mode=QualityMode.FAST,
        hardware=hardware,
        performance_evidence=evidence,
        comparison_baseline=baseline_id,
    )
    source_record = build_retrieval_benchmark_record(
        result,
        record_id=BenchmarkRecordId("benchmark.retrieval.sequential"),
        producer=producer,
        quality_mode=QualityMode.FAST,
        hardware=hardware,
        source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
    )

    assert union_record.fixture == result.fixture.identity
    assert union_record.producer is producer
    assert union_record.quality_mode is QualityMode.FAST
    assert union_record.hardware is hardware
    assert union_record.metrics is result.union_metrics
    assert union_record.performance_evidence == evidence
    assert union_record.comparison_baseline == baseline_id
    assert source_record.metrics is result.source_results[0].metrics

    with pytest.raises(ValueError, match="exactly one benchmark source result"):
        build_retrieval_benchmark_record(
            result,
            record_id=BenchmarkRecordId("benchmark.retrieval.unknown"),
            producer=producer,
            quality_mode=QualityMode.FAST,
            hardware=hardware,
            source_id=PairCandidateSourceId("unknown"),
        )


def test_all_completed_v2l7_source_ids_compose_without_external_engines() -> None:
    source_sets = tuple(
        sorted(
            (
                _set(
                    SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
                    _candidate(
                        "obs:a",
                        "obs:b",
                        SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
                    ),
                ),
                _set(
                    GPS_PAIR_CANDIDATE_SOURCE_ID,
                    _candidate("obs:a", "obs:b", GPS_PAIR_CANDIDATE_SOURCE_ID),
                ),
                _set(
                    VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID,
                    _candidate(
                        "obs:a",
                        "obs:c",
                        VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID,
                    ),
                ),
                _set(
                    SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
                    _candidate(
                        "obs:c",
                        "obs:d",
                        SELAVPR_PLUS_PAIR_CANDIDATE_SOURCE_ID,
                    ),
                ),
            ),
            key=lambda item: item.source_id.value,
        )
    )

    result = benchmark_retrieval_candidates(_request(source_sets))

    assert tuple(item.source_id.value for item in result.source_results) == (
        "colmap.vocab_tree",
        "gps",
        "selavpr_plus",
        "sequential",
    )
    ab = result.union_candidates[0]
    assert tuple(source.source_id.value for source in ab.sources) == (
        "gps",
        "sequential",
    )
    assert _metric_value(result.union_metrics, "retrieval.candidate_recall") == 1.0


def test_benchmark_module_has_no_later_scope_or_external_execution_surface() -> None:
    for name in (
        "torch",
        "pycolmap",
        "faiss",
        "requests",
        "urllib",
        "subprocess",
        "SceneCluster",
        "CorrespondenceSet",
        "QualityDecision",
        "RouteGraph",
        "GeometrySolution",
    ):
        assert name not in benchmark_module.__dict__

    source = Path(benchmark_module.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "torch.",
        "pycolmap",
        "faiss.",
        "requests.",
        "urllib.",
        "subprocess",
        "SceneCluster",
        "CorrespondenceSet",
        "QualityDecision",
        "rank_fusion",
        "reciprocal_rank",
        "default_source",
        "geometric_verification",
    ):
        assert forbidden not in source
