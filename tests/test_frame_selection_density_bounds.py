from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature
from itertools import pairwise
from typing import Any, cast

import pytest

import wre.frame_selection.density_bounds as density_bounds_module
from wre.frame_selection import (
    FrameSelectionDensityBounds,
    apply_frame_selection_density_bounds,
)
from wre.ingestion.keyframes import ProbedVideoFrame


def _frames(*times_us: int) -> tuple[ProbedVideoFrame, ...]:
    return tuple(
        ProbedVideoFrame(frame_index=index, frame_time_us=time_us)
        for index, time_us in enumerate(times_us)
    )


def test_density_bounds_are_exact_immutable_hashable_value() -> None:
    bounds = FrameSelectionDensityBounds(
        maximum_selected_frames=10,
        minimum_selected_interval_us=500_000,
    )

    assert tuple(field.name for field in fields(FrameSelectionDensityBounds)) == (
        "maximum_selected_frames",
        "minimum_selected_interval_us",
    )
    assert hash(bounds) == hash(
        FrameSelectionDensityBounds(
            maximum_selected_frames=10,
            minimum_selected_interval_us=500_000,
        )
    )
    with pytest.raises(FrozenInstanceError):
        bounds.maximum_selected_frames = 5  # type: ignore[misc]


def test_density_bounds_require_at_least_one_exact_positive_integer_bound() -> None:
    with pytest.raises(ValueError, match="at least one bound"):
        FrameSelectionDensityBounds()

    for field_name in (
        "maximum_selected_frames",
        "minimum_selected_interval_us",
    ):
        for value in (0, -1, True, 1.5, "1"):
            kwargs: dict[str, Any] = {field_name: value}
            with pytest.raises(ValueError, match="positive integer or None"):
                FrameSelectionDensityBounds(**kwargs)


def test_public_reducer_signature_and_input_types_are_exact() -> None:
    assert tuple(signature(apply_frame_selection_density_bounds).parameters) == (
        "frames",
        "bounds",
    )
    frames = _frames(0, 100)
    bounds = FrameSelectionDensityBounds(maximum_selected_frames=1)

    with pytest.raises(TypeError, match="immutable tuple"):
        apply_frame_selection_density_bounds(cast(Any, list(frames)), bounds)
    with pytest.raises(TypeError, match="members must be ProbedVideoFrame"):
        apply_frame_selection_density_bounds((cast(Any, (0, 0)),), bounds)
    with pytest.raises(TypeError, match="bounds must be FrameSelectionDensityBounds"):
        apply_frame_selection_density_bounds(frames, cast(Any, {}))


def test_source_order_must_be_strict_by_index_and_non_decreasing_by_time() -> None:
    bounds = FrameSelectionDensityBounds(maximum_selected_frames=1)

    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        apply_frame_selection_density_bounds(
            (
                ProbedVideoFrame(frame_index=1, frame_time_us=0),
                ProbedVideoFrame(frame_index=1, frame_time_us=100),
            ),
            bounds,
        )

    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        apply_frame_selection_density_bounds(
            (
                ProbedVideoFrame(frame_index=0, frame_time_us=100),
                ProbedVideoFrame(frame_index=1, frame_time_us=99),
            ),
            bounds,
        )


def test_empty_singleton_and_noop_bounds_preserve_original_values() -> None:
    count_bound = FrameSelectionDensityBounds(maximum_selected_frames=10)
    interval_bound = FrameSelectionDensityBounds(minimum_selected_interval_us=100)

    assert apply_frame_selection_density_bounds((), count_bound) == ()

    singleton = _frames(123)
    assert apply_frame_selection_density_bounds(singleton, count_bound) is singleton

    frames = _frames(0, 100, 200)
    selected = apply_frame_selection_density_bounds(frames, interval_bound)
    assert selected == frames
    assert selected[0] is frames[0]
    assert selected[1] is frames[1]
    assert selected[2] is frames[2]


def test_minimum_interval_is_greedy_inclusive_and_preserves_source_order() -> None:
    frames = _frames(0, 100, 500, 999, 1_000, 1_500)
    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(minimum_selected_interval_us=500),
    )

    assert selected == (
        frames[0],
        frames[2],
        frames[4],
        frames[5],
    )


def test_positive_minimum_interval_drops_equal_timestamp_followers() -> None:
    frames = (
        ProbedVideoFrame(frame_index=0, frame_time_us=0),
        ProbedVideoFrame(frame_index=1, frame_time_us=0),
        ProbedVideoFrame(frame_index=2, frame_time_us=100),
    )

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(minimum_selected_interval_us=100),
    )

    assert selected == (frames[0], frames[2])


def test_maximum_one_returns_exact_first_frame() -> None:
    frames = _frames(0, 100, 200)

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(maximum_selected_frames=1),
    )

    assert selected == (frames[0],)
    assert selected[0] is frames[0]


def test_hard_cap_uses_deterministic_floor_quantiles_and_preserves_endpoints() -> None:
    frames = _frames(*range(10))

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(maximum_selected_frames=4),
    )

    assert selected == (
        frames[0],
        frames[3],
        frames[6],
        frames[9],
    )
    assert selected[0] is frames[0]
    assert selected[-1] is frames[-1]


@pytest.mark.parametrize(
    ("frame_count", "maximum", "expected_indices"),
    [
        (6, 4, (0, 1, 3, 5)),
        (5, 3, (0, 2, 4)),
        (4, 2, (0, 3)),
    ],
)
def test_floor_quantile_rule_is_exact(
    frame_count: int,
    maximum: int,
    expected_indices: tuple[int, ...],
) -> None:
    frames = _frames(*range(frame_count))

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(maximum_selected_frames=maximum),
    )

    assert tuple(frame.frame_index for frame in selected) == expected_indices


def test_combined_bounds_apply_spacing_before_hard_cap() -> None:
    frames = _frames(0, 100, 500, 900, 1_000, 1_500)

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(
            maximum_selected_frames=3,
            minimum_selected_interval_us=500,
        ),
    )

    assert selected == (
        frames[0],
        frames[2],
        frames[5],
    )
    assert len(selected) == 3
    assert all(
        right.frame_time_us - left.frame_time_us >= 500 for left, right in pairwise(selected)
    )


def test_reducer_preserves_original_frame_identity_and_timestamps() -> None:
    frames = (
        ProbedVideoFrame(frame_index=10, frame_time_us=0),
        ProbedVideoFrame(frame_index=20, frame_time_us=500),
        ProbedVideoFrame(frame_index=30, frame_time_us=1_000),
        ProbedVideoFrame(frame_index=40, frame_time_us=1_500),
    )

    selected = apply_frame_selection_density_bounds(
        frames,
        FrameSelectionDensityBounds(maximum_selected_frames=2),
    )

    assert selected == (frames[0], frames[3])
    assert selected[0] is frames[0]
    assert selected[1] is frames[3]
    assert selected[0].frame_index == 10
    assert selected[0].frame_time_us == 0
    assert selected[1].frame_index == 40
    assert selected[1].frame_time_us == 1_500


def test_public_surface_has_no_quality_hardware_route_decode_or_persistence_behavior() -> None:
    for name in (
        "MetricVector",
        "MetricObservation",
        "FrameSelectionPolicy",
        "QualityDiversitySelectionConfig",
        "score",
        "rank",
        "threshold",
        "memory_budget",
        "vram",
        "scheduler",
        "hardware",
        "QualityDecision",
        "route",
        "Path",
        "open",
        "subprocess",
        "FFmpegToolchain",
        "LocalKeyframeExtractor",
        "VideoFrameObservation",
        "ArtifactRef",
        "persist",
        "registry",
    ):
        assert name not in density_bounds_module.__dict__
