from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactRef
from wre.domain.producer_identity import ArtifactProducerIdentity

_METRIC_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


def _require_metric_token(value: object, context: str) -> None:
    if not isinstance(value, str) or _METRIC_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be a 1-128 character lowercase token using "
            "letters, digits, '.', '_', ':' or '-'"
        )


@dataclass(frozen=True, slots=True, order=True)
class MetricName:
    """Open solver-independent identity for one metric."""

    value: str

    def __post_init__(self) -> None:
        _require_metric_token(self.value, "metric_name")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class MetricDimension:
    """Open semantic dimension associated with a metric value."""

    value: str

    def __post_init__(self) -> None:
        _require_metric_token(self.value, "metric_dimension")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class MetricUnit:
    """Open unit identity associated with a metric value."""

    value: str

    def __post_init__(self) -> None:
        _require_metric_token(self.value, "metric_unit")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class MetricAggregation:
    """Open descriptive identity for how a supplied scalar was aggregated."""

    value: str

    def __post_init__(self) -> None:
        _require_metric_token(self.value, "metric_aggregation")

    def __str__(self) -> str:
        return self.value


class MetricDirection(StrEnum):
    """Stable interpretation of whether larger or smaller values are preferable."""

    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    INFORMATIONAL = "informational"


@dataclass(frozen=True, slots=True)
class MetricDescriptor:
    """Explicit semantic description of one scalar metric."""

    name: MetricName
    dimension: MetricDimension
    unit: MetricUnit
    direction: MetricDirection
    aggregation: MetricAggregation

    def __post_init__(self) -> None:
        if not isinstance(self.name, MetricName):
            raise TypeError("metric_descriptor.name must be MetricName")
        if not isinstance(self.dimension, MetricDimension):
            raise TypeError("metric_descriptor.dimension must be MetricDimension")
        if not isinstance(self.unit, MetricUnit):
            raise TypeError("metric_descriptor.unit must be MetricUnit")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("metric_descriptor.direction must be MetricDirection")
        if not isinstance(self.aggregation, MetricAggregation):
            raise TypeError("metric_descriptor.aggregation must be MetricAggregation")


@dataclass(frozen=True, slots=True)
class MetricProvenance:
    """Exact evaluator and ordered artifact inputs for a metric observation."""

    evaluator: ArtifactProducerIdentity
    input_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evaluator, ArtifactProducerIdentity):
            raise TypeError("metric_provenance.evaluator must be ArtifactProducerIdentity")
        if not isinstance(self.input_artifacts, tuple):
            raise TypeError("metric_provenance.input_artifacts must be an immutable tuple")
        if any(not isinstance(artifact, ArtifactRef) for artifact in self.input_artifacts):
            raise TypeError("metric_provenance.input_artifacts members must be ArtifactRef")


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One finite scalar measurement with explicit semantics and provenance."""

    descriptor: MetricDescriptor
    value: float
    provenance: MetricProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, MetricDescriptor):
            raise TypeError("metric_observation.descriptor must be MetricDescriptor")
        if type(self.value) is not float:
            raise TypeError("metric_observation.value must be float")
        if not math.isfinite(self.value):
            raise ValueError("metric_observation.value must be finite")
        if not isinstance(self.provenance, MetricProvenance):
            raise TypeError("metric_observation.provenance must be MetricProvenance")


@dataclass(frozen=True, slots=True)
class MetricVector:
    """Canonical ordered collection of unique named metric observations."""

    observations: tuple[MetricObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observations, tuple):
            raise TypeError("metric_vector.observations must be an immutable tuple")
        if any(not isinstance(observation, MetricObservation) for observation in self.observations):
            raise TypeError("metric_vector.observations members must be MetricObservation")

        names = tuple(observation.descriptor.name.value for observation in self.observations)
        if len(names) != len(set(names)):
            raise ValueError("metric_vector metric names must be unique")
        if names != tuple(sorted(names)):
            raise ValueError("metric_vector observations must be in canonical MetricName order")
