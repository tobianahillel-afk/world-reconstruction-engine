from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    FailureCategory,
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
    Sha256Digest,
)


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.quality.reference",
            version="1.0.0",
            revision="metric:1",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
    )


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=ArtifactId(identifier), artifact_kind=ArtifactKind(kind))


def _descriptor(
    name: str,
    *,
    dimension: str = "geometry",
    unit: str = "pixel",
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
    aggregation: str = "mean",
) -> MetricDescriptor:
    return MetricDescriptor(
        name=MetricName(name),
        dimension=MetricDimension(dimension),
        unit=MetricUnit(unit),
        direction=direction,
        aggregation=MetricAggregation(aggregation),
    )


def _provenance(*artifacts: ArtifactRef) -> MetricProvenance:
    return MetricProvenance(evaluator=_producer(), input_artifacts=tuple(artifacts))


def _observation(name: str, value: float, **descriptor_kwargs: Any) -> MetricObservation:
    return MetricObservation(
        descriptor=_descriptor(name, **descriptor_kwargs),
        value=value,
        provenance=_provenance(_artifact(f"input-{name}", "geometry.solution")),
    )


@pytest.mark.parametrize(
    ("token_type", "value"),
    [
        (MetricName, "geometry.reprojection_error"),
        (MetricDimension, "runtime.latency"),
        (MetricUnit, "millisecond"),
        (MetricAggregation, "p95"),
    ],
)
def test_open_metric_tokens_are_typed_immutable_and_hashable(
    token_type: type[Any], value: str
) -> None:
    first = token_type(value)
    second = token_type(value)

    assert str(first) == value
    assert first == second
    assert hash(first) == hash(second)
    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]


def test_open_metric_token_types_remain_distinct() -> None:
    value = "shared.token"

    assert MetricName(value) != cast(Any, MetricDimension(value))
    assert MetricName(value) != cast(Any, MetricUnit(value))
    assert MetricName(value) != cast(Any, MetricAggregation(value))


@pytest.mark.parametrize(
    "value",
    ["", "UPPER", " leading", ".leading", "has/slash", "has space", "a" * 129],
)
@pytest.mark.parametrize(
    "token_type",
    [MetricName, MetricDimension, MetricUnit, MetricAggregation],
)
def test_open_metric_tokens_reject_invalid_values(token_type: type[Any], value: str) -> None:
    with pytest.raises(ValueError):
        token_type(value)


def test_metric_direction_has_exact_closed_wire_vocabulary() -> None:
    assert [(member.name, member.value) for member in MetricDirection] == [
        ("HIGHER_IS_BETTER", "higher_is_better"),
        ("LOWER_IS_BETTER", "lower_is_better"),
        ("INFORMATIONAL", "informational"),
    ]
    assert str(MetricDirection.HIGHER_IS_BETTER) == "higher_is_better"
    with pytest.raises(ValueError):
        MetricDirection("maximize")


def test_metric_descriptor_is_explicit_immutable_and_hashable() -> None:
    descriptor = _descriptor("geometry.reprojection_error")
    same = _descriptor("geometry.reprojection_error")

    assert descriptor == same
    assert hash(descriptor) == hash(same)
    with pytest.raises(FrozenInstanceError):
        descriptor.name = MetricName("changed")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("name", "metric", "name must be MetricName"),
        ("dimension", "geometry", "dimension must be MetricDimension"),
        ("unit", "pixel", "unit must be MetricUnit"),
        ("direction", "lower_is_better", "direction must be MetricDirection"),
        ("aggregation", "mean", "aggregation must be MetricAggregation"),
    ],
)
def test_metric_descriptor_rejects_untyped_members(
    field: str, replacement: object, message: str
) -> None:
    kwargs: dict[str, object] = {
        "name": MetricName("geometry.reprojection_error"),
        "dimension": MetricDimension("geometry"),
        "unit": MetricUnit("pixel"),
        "direction": MetricDirection.LOWER_IS_BETTER,
        "aggregation": MetricAggregation("mean"),
    }
    kwargs[field] = replacement

    with pytest.raises(TypeError, match=message):
        MetricDescriptor(**cast(Any, kwargs))


def test_metric_provenance_preserves_order_and_allows_empty_inputs() -> None:
    first = _artifact("a", "geometry.solution")
    second = _artifact("b", "camera.solution")
    provenance = _provenance(first, second)
    reversed_provenance = _provenance(second, first)
    empty = _provenance()

    assert provenance.input_artifacts == (first, second)
    assert provenance != reversed_provenance
    assert len({provenance, reversed_provenance}) == 2
    assert empty.input_artifacts == ()
    with pytest.raises(FrozenInstanceError):
        provenance.input_artifacts = ()  # type: ignore[misc]


def test_metric_provenance_rejects_untyped_evaluator_collection_and_members() -> None:
    artifact = _artifact("a", "geometry.solution")

    with pytest.raises(TypeError, match="evaluator must be ArtifactProducerIdentity"):
        MetricProvenance(evaluator=cast(Any, "evaluator"), input_artifacts=())
    with pytest.raises(TypeError, match="input_artifacts must be an immutable tuple"):
        MetricProvenance(evaluator=_producer(), input_artifacts=cast(Any, [artifact]))
    with pytest.raises(TypeError, match="input_artifacts members must be ArtifactRef"):
        MetricProvenance(evaluator=_producer(), input_artifacts=cast(Any, ("artifact",)))


def test_metric_observation_accepts_only_finite_float_values() -> None:
    descriptor = _descriptor("geometry.reprojection_error")
    provenance = _provenance()
    observation = MetricObservation(descriptor=descriptor, value=1.25, provenance=provenance)

    assert observation.value == 1.25
    assert hash(observation) == hash(MetricObservation(descriptor, 1.25, provenance))
    with pytest.raises(FrozenInstanceError):
        observation.value = 2.0  # type: ignore[misc]

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            MetricObservation(descriptor=descriptor, value=value, provenance=provenance)

    for value in (1, True, "1.25"):
        with pytest.raises(TypeError, match="value must be float"):
            MetricObservation(descriptor=descriptor, value=cast(Any, value), provenance=provenance)


def test_metric_observation_rejects_untyped_descriptor_and_provenance() -> None:
    with pytest.raises(TypeError, match="descriptor must be MetricDescriptor"):
        MetricObservation(
            descriptor=cast(Any, "metric"),
            value=1.0,
            provenance=_provenance(),
        )
    with pytest.raises(TypeError, match="provenance must be MetricProvenance"):
        MetricObservation(
            descriptor=_descriptor("geometry.reprojection_error"),
            value=1.0,
            provenance=cast(Any, "provenance"),
        )


def test_metric_vector_accepts_empty_single_and_multidimensional_canonical_order() -> None:
    geometry = _observation("geometry.reprojection_error", 0.8)
    quality = _observation(
        "appearance.psnr",
        31.0,
        dimension="appearance",
        unit="decibel",
        direction=MetricDirection.HIGHER_IS_BETTER,
    )
    runtime = _observation(
        "runtime.wall_time",
        12.5,
        dimension="runtime",
        unit="second",
        direction=MetricDirection.LOWER_IS_BETTER,
        aggregation="total",
    )

    empty = MetricVector(observations=())
    single = MetricVector(observations=(geometry,))
    multidimensional = MetricVector(observations=(quality, geometry, runtime))

    assert empty.observations == ()
    assert single.observations == (geometry,)
    assert [str(item.descriptor.name) for item in multidimensional.observations] == [
        "appearance.psnr",
        "geometry.reprojection_error",
        "runtime.wall_time",
    ]
    with pytest.raises(FrozenInstanceError):
        multidimensional.observations = ()  # type: ignore[misc]


def test_metric_vector_rejects_unsorted_duplicate_mutable_and_untyped_observations() -> None:
    first = _observation("a.metric", 1.0)
    second = _observation("b.metric", 2.0)
    duplicate = _observation("a.metric", 3.0)

    with pytest.raises(ValueError, match="canonical MetricName order"):
        MetricVector(observations=(second, first))
    with pytest.raises(ValueError, match="metric names must be unique"):
        MetricVector(observations=(first, duplicate))
    with pytest.raises(TypeError, match="observations must be an immutable tuple"):
        MetricVector(observations=cast(Any, [first]))
    with pytest.raises(TypeError, match="observations members must be MetricObservation"):
        MetricVector(observations=cast(Any, ("a.metric",)))


def test_failure_and_adapter_local_metric_vocabularies_remain_independent() -> None:
    metric = MetricName("quality_gate_failure")

    assert metric.value == FailureCategory.QUALITY_GATE_FAILURE.value
    assert metric != cast(Any, FailureCategory.QUALITY_GATE_FAILURE)
    assert not hasattr(metric, "failure_category")
    assert not hasattr(metric, "decision")
    assert not hasattr(metric, "threshold")
