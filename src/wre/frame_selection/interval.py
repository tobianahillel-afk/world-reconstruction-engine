from __future__ import annotations

from dataclasses import dataclass

from wre.domain.frame_selection import FrameSelectionPolicy
from wre.ingestion.keyframes import (
    KeyframeSelectionPolicy,
    ProbedVideoFrame,
    select_keyframes,
)


@dataclass(frozen=True, slots=True)
class IntervalFrameSelectionConfig:
    """Explicit configuration for the retained deterministic interval donor."""

    min_interval_us: int

    def __post_init__(self) -> None:
        if type(self.min_interval_us) is not int or self.min_interval_us <= 0:
            raise ValueError("interval_frame_selection.min_interval_us must be a positive integer")


def select_interval_frames(
    frames: tuple[ProbedVideoFrame, ...],
    policy: FrameSelectionPolicy,
    config: IntervalFrameSelectionConfig,
) -> tuple[ProbedVideoFrame, ...]:
    """Delegate interval-only frame selection to the retained donor."""

    if not isinstance(frames, tuple):
        raise TypeError("interval_frame_selection.frames must be an immutable tuple")
    if any(not isinstance(frame, ProbedVideoFrame) for frame in frames):
        raise TypeError("interval_frame_selection.frames members must be ProbedVideoFrame")
    if not isinstance(policy, FrameSelectionPolicy):
        raise TypeError("interval_frame_selection.policy must be FrameSelectionPolicy")
    if not isinstance(config, IntervalFrameSelectionConfig):
        raise TypeError("interval_frame_selection.config must be IntervalFrameSelectionConfig")
    if policy.required_metric_names or policy.optional_metric_names:
        raise ValueError("interval frame selection requires a policy with no metric declarations")

    return select_keyframes(
        frames,
        KeyframeSelectionPolicy(min_interval_us=config.min_interval_us),
    )
