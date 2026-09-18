from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.frame_selection.interval as interval_module
from wre.domain import (
    FrameSelectionPolicy,
    FrameSelectionPolicyId,
    FrameSelectionPolicyRevision,
    MetricName,
)
from wre.frame_selection import IntervalFrameSelectionConfig, select_interval_frames
from wre.ingestion.keyframes import (
    KeyframeSelectionPolicy,
    ProbedVideoFrame,
    select_keyframes,
)


def _policy(
    *,
    required: tuple[MetricName, ...] = (),
    optional: tuple[MetricName, ...] = (),
) -> FrameSelectionPolicy:
    return FrameSelectionPolicy(
        policy_id=FrameSelectionPolicyId("interval-baseline"),
        revision=FrameSelectionPolicyRevision(1),
        required_metric_names=required,
        optional_metric_names=optional,
    )


def _frames(*times_us: int) -> tuple[ProbedVideoFrame, ...]:
    return tuple(
        ProbedVideoFrame(frame_index=index, frame_time_us=time_us)
        for index, time_us in enumerate(times_us)
    )


def test_interval_config_is_exact_immutable_hashable_value() -> None:
    config = IntervalFrameSelectionConfig(min_interval_us=500_000)

    assert tuple(field.name for field in fields(IntervalFrameSelectionConfig)) == (
        "min_interval_us",
    )
    assert config == IntervalFrameSelectionConfig(min_interval_us=500_000)
    assert hash(config) == hash(IntervalFrameSelectionConfig(min_interval_us=500_000))

    with pytest.raises(FrozenInstanceError):
        config.min_interval_us = 1  # type: ignore[misc]


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1000"])
def test_interval_config_requires_positive_exact_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        IntervalFrameSelectionConfig(min_interval_us=cast(Any, value))


def test_adapter_signature_and_types_are_exact() -> None:
    assert tuple(signature(select_interval_frames).parameters) == (
        "frames",
        "policy",
        "config",
    )

    frames = _frames(0, 1_000_000)
    policy = _policy()
    config = IntervalFrameSelectionConfig(min_interval_us=500_000)

    with pytest.raises(TypeError, match="immutable tuple"):
        select_interval_frames(cast(Any, list(frames)), policy, config)

    with pytest.raises(TypeError, match="members must be ProbedVideoFrame"):
        select_interval_frames((cast(Any, (0, 0)),), policy, config)

    with pytest.raises(TypeError, match="policy must be FrameSelectionPolicy"):
        select_interval_frames(frames, cast(Any, "interval-baseline"), config)

    with pytest.raises(TypeError, match="config must be IntervalFrameSelectionConfig"):
        select_interval_frames(frames, policy, cast(Any, 500_000))


@pytest.mark.parametrize(
    ("required", "optional"),
    [
        ((MetricName("media.sharpness.laplacian_variance"),), ()),
        ((), (MetricName("media.visual.grid_luma_mae"),)),
    ],
)
def test_interval_adapter_rejects_metric_aware_policies(
    required: tuple[MetricName, ...],
    optional: tuple[MetricName, ...],
) -> None:
    with pytest.raises(ValueError, match="no metric declarations"):
        select_interval_frames(
            _frames(0, 1_000_000),
            _policy(required=required, optional=optional),
            IntervalFrameSelectionConfig(min_interval_us=500_000),
        )


@pytest.mark.parametrize(
    ("times_us", "interval_us"),
    [
        ((0, 250_000, 500_000, 750_000, 1_000_000), 500_000),
        ((0, 500_000, 1_000_000), 500_000),
        ((0, 100_000, 900_000, 1_000_000, 2_100_000), 1_000_000),
        ((125_000, 125_001, 225_000, 425_000, 925_000), 300_000),
    ],
)
def test_adapter_matches_retained_donor_exactly(
    times_us: tuple[int, ...],
    interval_us: int,
) -> None:
    frames = _frames(*times_us)
    config = IntervalFrameSelectionConfig(min_interval_us=interval_us)

    selected = select_interval_frames(frames, _policy(), config)
    donor_selected = select_keyframes(
        frames,
        KeyframeSelectionPolicy(min_interval_us=interval_us),
    )

    assert selected == donor_selected
    assert all(
        selected_frame is donor_frame
        for selected_frame, donor_frame in zip(selected, donor_selected, strict=True)
    )


def test_empty_input_preserves_donor_failure_semantics() -> None:
    config = IntervalFrameSelectionConfig(min_interval_us=1)

    with pytest.raises(ValueError, match="at least one timestamped source frame"):
        select_interval_frames((), _policy(), config)

    with pytest.raises(ValueError, match="at least one timestamped source frame"):
        select_keyframes((), KeyframeSelectionPolicy(min_interval_us=1))


def test_non_monotone_index_preserves_donor_failure_semantics() -> None:
    frames = (
        ProbedVideoFrame(frame_index=1, frame_time_us=0),
        ProbedVideoFrame(frame_index=1, frame_time_us=1_000),
    )
    config = IntervalFrameSelectionConfig(min_interval_us=1)

    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        select_interval_frames(frames, _policy(), config)

    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        select_keyframes(frames, KeyframeSelectionPolicy(min_interval_us=1))


def test_non_monotone_timestamp_preserves_donor_failure_semantics() -> None:
    frames = (
        ProbedVideoFrame(frame_index=0, frame_time_us=1_000),
        ProbedVideoFrame(frame_index=1, frame_time_us=999),
    )
    config = IntervalFrameSelectionConfig(min_interval_us=1)

    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        select_interval_frames(frames, _policy(), config)

    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        select_keyframes(frames, KeyframeSelectionPolicy(min_interval_us=1))


def test_adapter_returns_only_original_probed_frame_values() -> None:
    frames = _frames(0, 100, 500, 1_000)
    selected = select_interval_frames(
        frames,
        _policy(),
        IntervalFrameSelectionConfig(min_interval_us=500),
    )

    assert selected == (frames[0], frames[2], frames[3])
    assert selected[0] is frames[0]
    assert selected[1] is frames[2]
    assert selected[2] is frames[3]
    assert all(isinstance(frame, ProbedVideoFrame) for frame in selected)


def test_public_surface_has_no_extraction_quality_budget_or_route_behavior() -> None:
    for name in (
        "Path",
        "open",
        "subprocess",
        "FFmpegToolchain",
        "LocalKeyframeExtractor",
        "KeyframeExtractionRequest",
        "MetricVector",
        "MetricObservation",
        "QualityDecision",
        "threshold",
        "weight",
        "score",
        "budget",
        "route",
        "selected_frame_artifact",
        "persist",
        "registry",
    ):
        assert name not in interval_module.__dict__
