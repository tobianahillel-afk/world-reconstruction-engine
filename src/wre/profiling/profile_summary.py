from __future__ import annotations

import math
from dataclasses import dataclass

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

_DISTINCT_SOURCE_COUNT_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.collection.distinct_source_count"),
    dimension=MetricDimension("collection"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("count"),
)
_OBSERVATION_COUNT_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.collection.observation_count"),
    dimension=MetricDimension("collection"),
    unit=MetricUnit("count"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("count"),
)
_COVERAGE_DIVERSITY_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.coverage.grid_luma_diversity_mean"),
    dimension=MetricDimension("coverage"),
    unit=MetricUnit("normalized_luma"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("mean"),
)
_SEQUENCE_DURATION_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.sequence.duration_seconds"),
    dimension=MetricDimension("sequence"),
    unit=MetricUnit("second"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("duration"),
)
_TEMPORAL_CHANGE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.temporal.grid_luma_change_mean"),
    dimension=MetricDimension("temporal"),
    unit=MetricUnit("normalized_luma"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("mean"),
)


def _validate_distance_tuple(value: object, context: str) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{context} must be an immutable tuple")
    for member in value:
        if type(member) is not float:
            raise TypeError(f"{context} members must be float")
        if not math.isfinite(member) or not 0.0 <= member <= 1.0:
            raise ValueError(f"{context} members must be finite within [0, 1]")


@dataclass(frozen=True, slots=True)
class ProfileSummaryInput:
    observation_count: int
    distinct_source_count: int
    video_duration_us: int | None = None
    temporal_grid_luma_changes: tuple[float, ...] = ()
    coverage_grid_luma_distances: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ValueError("profile_summary.observation_count must be a positive integer")
        if type(self.distinct_source_count) is not int or self.distinct_source_count <= 0:
            raise ValueError(
                "profile_summary.distinct_source_count must be a positive integer"
            )
        if self.distinct_source_count > self.observation_count:
            raise ValueError(
                "profile_summary.distinct_source_count must not exceed observation_count"
            )
        if self.video_duration_us is not None and (
            type(self.video_duration_us) is not int or self.video_duration_us < 0
        ):
            raise ValueError(
                "profile_summary.video_duration_us must be a non-negative integer or None"
            )

        _validate_distance_tuple(
            self.temporal_grid_luma_changes,
            "profile_summary.temporal_grid_luma_changes",
        )
        _validate_distance_tuple(
            self.coverage_grid_luma_distances,
            "profile_summary.coverage_grid_luma_distances",
        )


def evaluate_profile_summary(
    summary: ProfileSummaryInput,
    provenance: MetricProvenance,
) -> MetricVector:
    """Aggregate explicit threshold-free profiling evidence into canonical metrics."""

    if not isinstance(summary, ProfileSummaryInput):
        raise TypeError("profile_summary.summary must be ProfileSummaryInput")
    if not isinstance(provenance, MetricProvenance):
        raise TypeError("profile_summary.provenance must be MetricProvenance")

    observations: list[MetricObservation] = [
        MetricObservation(
            descriptor=_DISTINCT_SOURCE_COUNT_DESCRIPTOR,
            value=float(summary.distinct_source_count),
            provenance=provenance,
        ),
        MetricObservation(
            descriptor=_OBSERVATION_COUNT_DESCRIPTOR,
            value=float(summary.observation_count),
            provenance=provenance,
        ),
    ]

    if summary.coverage_grid_luma_distances:
        observations.append(
            MetricObservation(
                descriptor=_COVERAGE_DIVERSITY_DESCRIPTOR,
                value=float(
                    math.fsum(summary.coverage_grid_luma_distances)
                    / len(summary.coverage_grid_luma_distances)
                ),
                provenance=provenance,
            )
        )

    if summary.video_duration_us is not None:
        observations.append(
            MetricObservation(
                descriptor=_SEQUENCE_DURATION_DESCRIPTOR,
                value=float(summary.video_duration_us / 1_000_000.0),
                provenance=provenance,
            )
        )

    if summary.temporal_grid_luma_changes:
        observations.append(
            MetricObservation(
                descriptor=_TEMPORAL_CHANGE_DESCRIPTOR,
                value=float(
                    math.fsum(summary.temporal_grid_luma_changes)
                    / len(summary.temporal_grid_luma_changes)
                ),
                provenance=provenance,
            )
        )

    return MetricVector(observations=tuple(observations))
