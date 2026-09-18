from __future__ import annotations

from dataclasses import dataclass

from wre.ingestion.keyframes import ProbedVideoFrame


@dataclass(frozen=True, slots=True)
class FrameSelectionDensityBounds:
    """Explicit deterministic post-selection density/resource bounds."""

    maximum_selected_frames: int | None = None
    minimum_selected_interval_us: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_selected_frames is not None and (
            type(self.maximum_selected_frames) is not int
            or self.maximum_selected_frames <= 0
        ):
            raise ValueError(
                "frame_selection_density.maximum_selected_frames must be a positive integer or None"
            )
        if self.minimum_selected_interval_us is not None and (
            type(self.minimum_selected_interval_us) is not int
            or self.minimum_selected_interval_us <= 0
        ):
            raise ValueError(
                "frame_selection_density.minimum_selected_interval_us must be a positive integer or None"
            )
        if self.maximum_selected_frames is None and self.minimum_selected_interval_us is None:
            raise ValueError("frame_selection_density requires at least one bound")


def _validate_frames(frames: object) -> tuple[ProbedVideoFrame, ...]:
    if not isinstance(frames, tuple):
        raise TypeError("frame_selection_density.frames must be an immutable tuple")
    if any(not isinstance(frame, ProbedVideoFrame) for frame in frames):
        raise TypeError("frame_selection_density.frames members must be ProbedVideoFrame")

    previous_index = -1
    previous_time = -1
    for frame in frames:
        if frame.frame_index <= previous_index:
            raise ValueError(
                "frame_selection_density frame indices must be strictly increasing"
            )
        if frame.frame_time_us < previous_time:
            raise ValueError(
                "frame_selection_density frame timestamps must be non-decreasing"
            )
        previous_index = frame.frame_index
        previous_time = frame.frame_time_us

    return frames


def _apply_minimum_interval(
    frames: tuple[ProbedVideoFrame, ...],
    minimum_interval_us: int,
) -> tuple[ProbedVideoFrame, ...]:
    if not frames:
        return ()

    selected = [frames[0]]
    last_selected_time = frames[0].frame_time_us
    for frame in frames[1:]:
        if frame.frame_time_us - last_selected_time >= minimum_interval_us:
            selected.append(frame)
            last_selected_time = frame.frame_time_us
    return tuple(selected)


def _apply_maximum_count(
    frames: tuple[ProbedVideoFrame, ...],
    maximum_selected_frames: int,
) -> tuple[ProbedVideoFrame, ...]:
    if len(frames) <= maximum_selected_frames:
        return frames
    if maximum_selected_frames == 1:
        return (frames[0],)

    last_index = len(frames) - 1
    denominator = maximum_selected_frames - 1
    indices = tuple(
        (selection_index * last_index) // denominator
        for selection_index in range(maximum_selected_frames)
    )
    return tuple(frames[index] for index in indices)


def apply_frame_selection_density_bounds(
    frames: tuple[ProbedVideoFrame, ...],
    bounds: FrameSelectionDensityBounds,
) -> tuple[ProbedVideoFrame, ...]:
    """Reduce already-selected frames using only explicit density/count bounds."""

    validated_frames = _validate_frames(frames)
    if not isinstance(bounds, FrameSelectionDensityBounds):
        raise TypeError(
            "frame_selection_density.bounds must be FrameSelectionDensityBounds"
        )

    bounded = validated_frames
    if bounds.minimum_selected_interval_us is not None:
        bounded = _apply_minimum_interval(
            bounded,
            bounds.minimum_selected_interval_us,
        )

    if bounds.maximum_selected_frames is not None:
        bounded = _apply_maximum_count(
            bounded,
            bounds.maximum_selected_frames,
        )

    return bounded
