from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.profiling.profile_summary as profile_summary_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    MetricDirection,
    MetricProvenance,
    ProducerRef,
    Sha256Digest,
)
from wre.profiling import ProfileSummaryInput, evaluate_profile_summary


def _provenance() -> MetricProvenance:
    return MetricProvenance(
        evaluator=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="wre.profiling.profile_summary",
                version="1.0.0",
                revision="v2l5.5",
            ),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("c" * 64)),
        ),
        input_artifacts=(
            ArtifactRef(
                artifact_id=ArtifactId("artifact:profile-summary-evidence"),
                artifact_kind=ArtifactKind("media.profile_evidence"),
            ),
        ),
    )


def test_profile_summary_input_is_exact_immutable_hashable_value() -> None:
    summary = ProfileSummaryInput(
        observation_count=4,
        distinct_source_count=2,
        video_duration_us=1_500_000,
        temporal_grid_luma_changes=(0.2, 0.2, 0.1),
        coverage_grid_luma_distances=(0.8, 0.4),
    )

    assert tuple(field.name for field in fields(ProfileSummaryInput)) == (
        "observation_count",
        "distinct_source_count",
        "video_duration_us",
        "temporal_grid_luma_changes",
        "coverage_grid_luma_distances",
    )
    assert hash(summary) == hash(
        ProfileSummaryInput(
            observation_count=4,
            distinct_source_count=2,
            video_duration_us=1_500_000,
            temporal_grid_luma_changes=(0.2, 0.2, 0.1),
            coverage_grid_luma_distances=(0.8, 0.4),
        )
    )
    assert summary.temporal_grid_luma_changes == (0.2, 0.2, 0.1)

    with pytest.raises(FrozenInstanceError):
        summary.observation_count = 5  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_observation_count_must_be_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="observation_count must be a positive integer"):
        ProfileSummaryInput(
            observation_count=cast(Any, value),
            distinct_source_count=1,
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_distinct_source_count_must_be_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="distinct_source_count must be a positive integer"):
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=cast(Any, value),
        )


def test_distinct_source_count_cannot_exceed_observation_count() -> None:
    with pytest.raises(ValueError, match="must not exceed observation_count"):
        ProfileSummaryInput(observation_count=2, distinct_source_count=3)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_video_duration_must_be_non_negative_integer_when_present(value: object) -> None:
    with pytest.raises(ValueError, match="video_duration_us"):
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=1,
            video_duration_us=cast(Any, value),
        )

    assert (
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=1,
            video_duration_us=0,
        ).video_duration_us
        == 0
    )


@pytest.mark.parametrize(
    "field_name",
    ["temporal_grid_luma_changes", "coverage_grid_luma_distances"],
)
def test_distance_collections_must_be_immutable_tuples(field_name: str) -> None:
    kwargs: dict[str, Any] = {field_name: [0.1, 0.2]}

    with pytest.raises(TypeError, match="immutable tuple"):
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=1,
            **kwargs,
        )


@pytest.mark.parametrize("value", [0, True, "0.5"])
@pytest.mark.parametrize(
    "field_name",
    ["temporal_grid_luma_changes", "coverage_grid_luma_distances"],
)
def test_distance_members_must_be_exact_floats(
    field_name: str,
    value: object,
) -> None:
    kwargs: dict[str, Any] = {field_name: (value,)}

    with pytest.raises(TypeError, match="members must be float"):
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=1,
            **kwargs,
        )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf"), -0.01, 1.01],
)
@pytest.mark.parametrize(
    "field_name",
    ["temporal_grid_luma_changes", "coverage_grid_luma_distances"],
)
def test_distance_members_must_be_finite_normalized_values(
    field_name: str,
    value: float,
) -> None:
    kwargs: dict[str, Any] = {field_name: (value,)}

    with pytest.raises(ValueError, match=r"finite within \[0, 1\]"):
        ProfileSummaryInput(
            observation_count=2,
            distinct_source_count=1,
            **kwargs,
        )


def test_empty_optional_evidence_emits_only_exact_collection_counts() -> None:
    provenance = _provenance()
    result = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=5,
            distinct_source_count=2,
        ),
        provenance,
    )

    assert tuple(item.descriptor.name.value for item in result.observations) == (
        "media.collection.distinct_source_count",
        "media.collection.observation_count",
    )
    assert tuple(item.value for item in result.observations) == (2.0, 5.0)
    assert tuple(item.descriptor.dimension.value for item in result.observations) == (
        "collection",
        "collection",
    )
    assert tuple(item.descriptor.unit.value for item in result.observations) == (
        "count",
        "count",
    )
    assert tuple(item.descriptor.aggregation.value for item in result.observations) == (
        "count",
        "count",
    )
    assert all(
        item.descriptor.direction is MetricDirection.INFORMATIONAL for item in result.observations
    )
    assert all(item.provenance is provenance for item in result.observations)


def test_video_duration_is_exactly_converted_and_absent_duration_is_omitted() -> None:
    provenance = _provenance()
    present = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=3,
            distinct_source_count=1,
            video_duration_us=1_500_000,
        ),
        provenance,
    )
    absent = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=3,
            distinct_source_count=1,
        ),
        provenance,
    )

    by_name = {item.descriptor.name.value: item for item in present.observations}
    assert by_name["media.sequence.duration_seconds"].value == 1.5
    assert by_name["media.sequence.duration_seconds"].descriptor.unit.value == "second"
    assert by_name["media.sequence.duration_seconds"].descriptor.aggregation.value == "duration"
    assert "media.sequence.duration_seconds" not in {
        item.descriptor.name.value for item in absent.observations
    }


def test_temporal_and_coverage_aggregates_are_exact_threshold_free_means() -> None:
    provenance = _provenance()
    result = evaluate_profile_summary(
        ProfileSummaryInput(
            observation_count=4,
            distinct_source_count=3,
            temporal_grid_luma_changes=(0.0, 0.5, 1.0),
            coverage_grid_luma_distances=(0.25, 0.75),
        ),
        provenance,
    )

    by_name = {item.descriptor.name.value: item for item in result.observations}
    assert by_name["media.temporal.grid_luma_change_mean"].value == 0.5
    assert by_name["media.coverage.grid_luma_diversity_mean"].value == 0.5
    assert (
        by_name["media.temporal.grid_luma_change_mean"].descriptor.unit.value == "normalized_luma"
    )
    assert (
        by_name["media.coverage.grid_luma_diversity_mean"].descriptor.unit.value
        == "normalized_luma"
    )
    assert by_name["media.temporal.grid_luma_change_mean"].descriptor.aggregation.value == "mean"
    assert by_name["media.coverage.grid_luma_diversity_mean"].descriptor.aggregation.value == "mean"


def test_all_metrics_use_canonical_name_order_finite_values_and_shared_provenance() -> None:
    provenance = _provenance()
    summary = ProfileSummaryInput(
        observation_count=8,
        distinct_source_count=3,
        video_duration_us=2_000_000,
        temporal_grid_luma_changes=(0.1, 0.9),
        coverage_grid_luma_distances=(0.2, 0.6),
    )

    first = evaluate_profile_summary(summary, provenance)
    second = evaluate_profile_summary(summary, provenance)

    assert first == second
    assert tuple(item.descriptor.name.value for item in first.observations) == (
        "media.collection.distinct_source_count",
        "media.collection.observation_count",
        "media.coverage.grid_luma_diversity_mean",
        "media.sequence.duration_seconds",
        "media.temporal.grid_luma_change_mean",
    )
    assert all(
        type(item.value) is float and math.isfinite(item.value) for item in first.observations
    )
    assert all(item.provenance is provenance for item in first.observations)
    assert all(
        item.descriptor.direction is MetricDirection.INFORMATIONAL for item in first.observations
    )


def test_evaluator_requires_exact_typed_inputs() -> None:
    summary = ProfileSummaryInput(
        observation_count=2,
        distinct_source_count=1,
    )
    provenance = _provenance()

    assert tuple(signature(evaluate_profile_summary).parameters) == (
        "summary",
        "provenance",
    )

    with pytest.raises(TypeError, match="summary must be ProfileSummaryInput"):
        evaluate_profile_summary(cast(Any, {}), provenance)

    with pytest.raises(TypeError, match="provenance must be MetricProvenance"):
        evaluate_profile_summary(summary, cast(Any, None))


def test_public_surface_has_no_labels_thresholds_routes_or_external_io() -> None:
    for name in (
        "is_dynamic",
        "is_static",
        "is_long_sequence",
        "is_sparse_coverage",
        "overlap",
        "threshold",
        "route",
        "QualityDecision",
        "select_frames",
        "retrieve",
        "cluster",
        "track",
        "segment",
        "geometry",
        "solver",
        "Path",
        "open",
        "subprocess",
        "ffprobe",
        "ffmpeg",
    ):
        assert name not in profile_summary_module.__dict__
