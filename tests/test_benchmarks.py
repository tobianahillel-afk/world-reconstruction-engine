from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    BenchmarkFixtureId,
    BenchmarkFixtureIdentity,
    BenchmarkRecord,
    BenchmarkRecordId,
    ConfigurationIdentity,
    FailureCategory,
    HardwareRuntimeIdentity,
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
    ProducerRef,
    QualityDecision,
    QualityMode,
    Sha256Digest,
)


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.benchmark.subject",
            version="1.2.3",
            revision="adapter:4",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
    )


def _hardware() -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(sha256=Sha256Digest("b" * 64))


def _fixture() -> BenchmarkFixtureIdentity:
    return BenchmarkFixtureIdentity(
        fixture_id=BenchmarkFixtureId("fixture.static-room.v1"),
        sha256=Sha256Digest("c" * 64),
    )


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(ArtifactId(identifier), ArtifactKind(kind))


def _metrics() -> MetricVector:
    descriptor = MetricDescriptor(
        name=MetricName("runtime.wall_time"),
        dimension=MetricDimension("runtime"),
        unit=MetricUnit("second"),
        direction=MetricDirection.LOWER_IS_BETTER,
        aggregation=MetricAggregation("total"),
    )
    observation = MetricObservation(
        descriptor=descriptor,
        value=12.5,
        provenance=MetricProvenance(
            evaluator=_producer(),
            input_artifacts=(_artifact("geometry", "geometry.solution"),),
        ),
    )
    return MetricVector(observations=(observation,))


def _record(**overrides: Any) -> BenchmarkRecord:
    values: dict[str, Any] = {
        "record_id": BenchmarkRecordId("benchmark.static-room.preview"),
        "fixture": _fixture(),
        "producer": _producer(),
        "quality_mode": QualityMode.PREVIEW,
        "hardware": _hardware(),
        "metrics": _metrics(),
        "performance_evidence": (),
        "comparison_baseline": None,
    }
    values.update(overrides)
    return BenchmarkRecord(**values)


def test_quality_mode_has_exact_closed_vocabulary_and_is_distinct_from_decision() -> None:
    assert [(member.name, member.value) for member in QualityMode] == [
        ("PREVIEW", "preview"),
        ("FAST", "fast"),
        ("QUALITY", "quality"),
        ("MASTER", "master"),
    ]
    for member in QualityMode:
        assert QualityMode(member.value) is member
        assert str(member) == member.value
    with pytest.raises(ValueError):
        QualityMode("standard")

    assert QualityMode is not QualityDecision
    assert {member.value for member in QualityMode}.isdisjoint(
        {member.value for member in QualityDecision}
    )


@pytest.mark.parametrize("identity_type", [BenchmarkRecordId, BenchmarkFixtureId])
def test_benchmark_ids_are_immutable_hashable_and_use_wre_opaque_id_syntax(
    identity_type: type[BenchmarkRecordId] | type[BenchmarkFixtureId],
) -> None:
    first = identity_type("Benchmark:Fixture_01.v1")
    second = identity_type("Benchmark:Fixture_01.v1")

    assert str(first) == "Benchmark:Fixture_01.v1"
    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]

    for invalid in ("", " leading", "trailing ", ".leading", "has/slash", "a" * 129):
        with pytest.raises(ValueError):
            identity_type(invalid)


def test_benchmark_id_types_remain_distinct_from_each_other_and_artifact_identity() -> None:
    value = "shared-id"

    assert BenchmarkRecordId(value) != cast(Any, BenchmarkFixtureId(value))
    assert BenchmarkRecordId(value) != cast(Any, ArtifactId(value))
    assert BenchmarkFixtureId(value) != cast(Any, ArtifactId(value))


def test_fixture_identity_requires_exact_typed_content_identity_and_is_immutable() -> None:
    fixture = _fixture()
    same = _fixture()

    assert fixture == same
    assert hash(fixture) == hash(same)
    assert fixture.sha256 == Sha256Digest("c" * 64)
    with pytest.raises(FrozenInstanceError):
        fixture.sha256 = Sha256Digest("d" * 64)  # type: ignore[misc]

    with pytest.raises(TypeError, match="fixture_id must be BenchmarkFixtureId"):
        BenchmarkFixtureIdentity(
            fixture_id=cast(Any, "fixture.static-room.v1"),
            sha256=Sha256Digest("c" * 64),
        )
    with pytest.raises(TypeError, match="sha256 must be Sha256Digest"):
        BenchmarkFixtureIdentity(
            fixture_id=BenchmarkFixtureId("fixture.static-room.v1"),
            sha256=cast(Any, "c" * 64),
        )


def test_benchmark_record_is_immutable_hashable_and_preserves_exact_inputs() -> None:
    record = _record()
    same = _record()

    assert record == same
    assert hash(record) == hash(same)
    assert record.fixture == _fixture()
    assert record.producer == _producer()
    assert record.quality_mode is QualityMode.PREVIEW
    assert record.hardware == _hardware()
    assert record.metrics == _metrics()
    assert record.performance_evidence == ()
    assert record.comparison_baseline is None
    with pytest.raises(FrozenInstanceError):
        record.quality_mode = QualityMode.MASTER  # type: ignore[misc]


def test_benchmark_record_preserves_canonical_performance_evidence_and_baseline_reference() -> None:
    evidence = (
        _artifact("perf-a", "performance.stage-timing"),
        _artifact("perf-b", "performance.trace"),
    )
    baseline = BenchmarkRecordId("benchmark.static-room.fast-baseline")
    metrics = _metrics()
    record = _record(
        metrics=metrics,
        performance_evidence=evidence,
        comparison_baseline=baseline,
    )

    assert record.metrics is metrics
    assert record.performance_evidence == evidence
    assert record.comparison_baseline == baseline
    assert not hasattr(record, "ranking")
    assert not hasattr(record, "winner")
    assert not hasattr(record, "default")
    assert not hasattr(record, "trace_payload")


def test_performance_evidence_requires_immutable_unique_canonical_typed_refs() -> None:
    first = _artifact("a", "performance.trace")
    second = _artifact("b", "performance.trace")

    with pytest.raises(TypeError, match="must be an immutable tuple"):
        _record(performance_evidence=cast(Any, [first]))
    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        _record(performance_evidence=cast(Any, ("performance.trace",)))
    with pytest.raises(ValueError, match="members must be unique"):
        _record(performance_evidence=(first, first))
    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        _record(performance_evidence=(second, first))


def test_comparison_baseline_is_optional_typed_and_cannot_reference_self() -> None:
    record_id = BenchmarkRecordId("benchmark.record")
    baseline = BenchmarkRecordId("benchmark.baseline")

    record = _record(record_id=record_id, comparison_baseline=baseline)
    assert record.comparison_baseline == baseline
    with pytest.raises(TypeError, match="must be BenchmarkRecordId when present"):
        _record(comparison_baseline=cast(Any, "benchmark.baseline"))
    with pytest.raises(ValueError, match="must not reference its own record"):
        _record(record_id=record_id, comparison_baseline=record_id)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("record_id", "record", "record_id must be BenchmarkRecordId"),
        ("fixture", "fixture", "fixture must be BenchmarkFixtureIdentity"),
        ("producer", "producer", "producer must be ArtifactProducerIdentity"),
        ("quality_mode", "preview", "quality_mode must be QualityMode"),
        ("hardware", "hardware", "hardware must be HardwareRuntimeIdentity"),
        ("metrics", "metrics", "metrics must be MetricVector"),
    ],
)
def test_benchmark_record_rejects_untyped_core_fields(
    field: str, replacement: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        _record(**{field: replacement})


def test_benchmark_record_does_not_conflate_measurement_decision_or_failure_semantics() -> None:
    record = _record()

    assert record.metrics == _metrics()
    assert QualityDecision.PASS != cast(Any, QualityMode.PREVIEW)
    assert FailureCategory.QUALITY_GATE_FAILURE != cast(Any, QualityMode.PREVIEW)
    for attribute in (
        "decision",
        "failure_category",
        "threshold",
        "policy_id",
        "policy_revision",
        "route",
        "score",
        "rank",
        "winner",
        "default",
        "promotion_status",
        "timestamp",
        "environment",
        "notes",
    ):
        assert not hasattr(record, attribute)
