from __future__ import annotations

import math
from dataclasses import dataclass

from wre.domain.frame_selection import FrameSelectionPolicy
from wre.domain.metrics import (
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricVector,
)
from wre.ingestion.keyframes import ProbedVideoFrame

_BLACK_CLIP = MetricName("media.exposure.black_clip_fraction")
_WHITE_CLIP = MetricName("media.exposure.white_clip_fraction")
_SHARPNESS = MetricName("media.sharpness.laplacian_variance")
_GRID_LUMA_MAE = MetricName("media.visual.grid_luma_mae")


@dataclass(frozen=True, slots=True)
class FrameSelectionCandidate:
    """One timestamped source-frame candidate with explicit precomputed metrics."""

    frame: ProbedVideoFrame
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.frame, ProbedVideoFrame):
            raise TypeError("frame_selection_candidate.frame must be ProbedVideoFrame")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("frame_selection_candidate.metrics must be MetricVector")


@dataclass(frozen=True, slots=True)
class AdjacentFrameDiversityEvidence:
    """Pairwise metric evidence for one consecutive source-candidate pair."""

    left_frame_index: int
    right_frame_index: int
    metrics: MetricVector

    def __post_init__(self) -> None:
        if type(self.left_frame_index) is not int or self.left_frame_index < 0:
            raise ValueError(
                "adjacent_frame_diversity.left_frame_index must be a non-negative integer"
            )
        if type(self.right_frame_index) is not int or self.right_frame_index < 0:
            raise ValueError(
                "adjacent_frame_diversity.right_frame_index must be a non-negative integer"
            )
        if self.left_frame_index >= self.right_frame_index:
            raise ValueError(
                "adjacent_frame_diversity left_frame_index must be less than right_frame_index"
            )
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("adjacent_frame_diversity.metrics must be MetricVector")


@dataclass(frozen=True, slots=True)
class QualityDiversitySelectionConfig:
    """Explicit deterministic thresholds for the first quality/diversity baseline."""

    minimum_sharpness_laplacian_variance: float | None = None
    maximum_black_clip_fraction: float | None = None
    maximum_white_clip_fraction: float | None = None
    minimum_adjacent_grid_luma_mae: float | None = None

    def __post_init__(self) -> None:
        self._validate_threshold(
            self.minimum_sharpness_laplacian_variance,
            "minimum_sharpness_laplacian_variance",
            lower_bound=0.0,
            upper_bound=None,
        )
        self._validate_threshold(
            self.maximum_black_clip_fraction,
            "maximum_black_clip_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        self._validate_threshold(
            self.maximum_white_clip_fraction,
            "maximum_white_clip_fraction",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        self._validate_threshold(
            self.minimum_adjacent_grid_luma_mae,
            "minimum_adjacent_grid_luma_mae",
            lower_bound=0.0,
            upper_bound=1.0,
        )
        if all(
            threshold is None
            for threshold in (
                self.minimum_sharpness_laplacian_variance,
                self.maximum_black_clip_fraction,
                self.maximum_white_clip_fraction,
                self.minimum_adjacent_grid_luma_mae,
            )
        ):
            raise ValueError("quality_diversity_selection requires at least one threshold")

    @staticmethod
    def _validate_threshold(
        value: object,
        name: str,
        *,
        lower_bound: float,
        upper_bound: float | None,
    ) -> None:
        if value is None:
            return
        if type(value) is not float:
            raise TypeError(f"quality_diversity_selection.{name} must be float or None")
        if not math.isfinite(value):
            raise ValueError(f"quality_diversity_selection.{name} must be finite")
        if value < lower_bound or (upper_bound is not None and value > upper_bound):
            if upper_bound is None:
                raise ValueError(f"quality_diversity_selection.{name} must be >= {lower_bound}")
            raise ValueError(
                f"quality_diversity_selection.{name} must be within [{lower_bound}, {upper_bound}]"
            )


def _required_metric_names(
    config: QualityDiversitySelectionConfig,
) -> tuple[MetricName, ...]:
    names: list[MetricName] = []
    if config.maximum_black_clip_fraction is not None:
        names.append(_BLACK_CLIP)
    if config.maximum_white_clip_fraction is not None:
        names.append(_WHITE_CLIP)
    if config.minimum_sharpness_laplacian_variance is not None:
        names.append(_SHARPNESS)
    if config.minimum_adjacent_grid_luma_mae is not None:
        names.append(_GRID_LUMA_MAE)
    return tuple(sorted(names, key=lambda name: name.value))


def _metric(
    metrics: MetricVector,
    name: MetricName,
    expected_direction: MetricDirection,
) -> MetricObservation:
    observation = next(
        (item for item in metrics.observations if item.descriptor.name == name),
        None,
    )
    if observation is None:
        raise ValueError(f"required frame-selection metric is missing: {name.value}")
    if observation.descriptor.direction is not expected_direction:
        raise ValueError(
            f"frame-selection metric {name.value} has incompatible direction "
            f"{observation.descriptor.direction.value}"
        )
    return observation


def _normalized_metric(
    metrics: MetricVector,
    name: MetricName,
    expected_direction: MetricDirection,
) -> float:
    value = _metric(metrics, name, expected_direction).value
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"frame-selection metric {name.value} must be within [0, 1]")
    return value


def _passes_quality(
    candidate: FrameSelectionCandidate,
    config: QualityDiversitySelectionConfig,
) -> bool:
    if config.minimum_sharpness_laplacian_variance is not None:
        sharpness = _metric(
            candidate.metrics,
            _SHARPNESS,
            MetricDirection.HIGHER_IS_BETTER,
        ).value
        if sharpness < 0.0:
            raise ValueError(
                "frame-selection metric media.sharpness.laplacian_variance must be non-negative"
            )
        if sharpness < config.minimum_sharpness_laplacian_variance:
            return False

    if config.maximum_black_clip_fraction is not None:
        black_clip = _normalized_metric(
            candidate.metrics,
            _BLACK_CLIP,
            MetricDirection.LOWER_IS_BETTER,
        )
        if black_clip > config.maximum_black_clip_fraction:
            return False

    if config.maximum_white_clip_fraction is not None:
        white_clip = _normalized_metric(
            candidate.metrics,
            _WHITE_CLIP,
            MetricDirection.LOWER_IS_BETTER,
        )
        if white_clip > config.maximum_white_clip_fraction:
            return False

    return True


def _validate_candidates(
    candidates: object,
) -> tuple[FrameSelectionCandidate, ...]:
    if not isinstance(candidates, tuple):
        raise TypeError("quality_diversity_selection.candidates must be an immutable tuple")
    if not candidates:
        raise ValueError("quality_diversity_selection.candidates must not be empty")
    if any(not isinstance(candidate, FrameSelectionCandidate) for candidate in candidates):
        raise TypeError(
            "quality_diversity_selection.candidates members must be FrameSelectionCandidate"
        )

    previous_index = -1
    previous_time = -1
    for candidate in candidates:
        if candidate.frame.frame_index <= previous_index:
            raise ValueError(
                "quality_diversity_selection candidate frame indices must be strictly increasing"
            )
        if candidate.frame.frame_time_us < previous_time:
            raise ValueError(
                "quality_diversity_selection candidate timestamps must be non-decreasing"
            )
        previous_index = candidate.frame.frame_index
        previous_time = candidate.frame.frame_time_us

    return candidates


def _validate_adjacent_diversity(
    candidates: tuple[FrameSelectionCandidate, ...],
    evidence: object,
    config: QualityDiversitySelectionConfig,
) -> tuple[float, ...]:
    if not isinstance(evidence, tuple):
        raise TypeError(
            "quality_diversity_selection.adjacent_diversity_evidence must be an immutable tuple"
        )
    if any(not isinstance(item, AdjacentFrameDiversityEvidence) for item in evidence):
        raise TypeError(
            "quality_diversity_selection.adjacent_diversity_evidence members "
            "must be AdjacentFrameDiversityEvidence"
        )

    if config.minimum_adjacent_grid_luma_mae is None:
        if evidence:
            raise ValueError(
                "adjacent diversity evidence must be empty when no diversity "
                "threshold is configured"
            )
        return ()

    expected_pair_count = len(candidates) - 1
    if len(evidence) != expected_pair_count:
        raise ValueError(
            "adjacent diversity evidence must contain exactly one entry "
            "for every consecutive candidate pair"
        )

    values: list[float] = []
    for index, item in enumerate(evidence):
        expected_left = candidates[index].frame.frame_index
        expected_right = candidates[index + 1].frame.frame_index
        if item.left_frame_index != expected_left or item.right_frame_index != expected_right:
            raise ValueError(
                "adjacent diversity evidence must match consecutive candidate "
                "frame-index pairs exactly"
            )
        values.append(
            _normalized_metric(
                item.metrics,
                _GRID_LUMA_MAE,
                MetricDirection.INFORMATIONAL,
            )
        )
    return tuple(values)


def select_quality_diversity_frames(
    candidates: tuple[FrameSelectionCandidate, ...],
    policy: FrameSelectionPolicy,
    config: QualityDiversitySelectionConfig,
    adjacent_diversity_evidence: tuple[AdjacentFrameDiversityEvidence, ...] = (),
) -> tuple[ProbedVideoFrame, ...]:
    """Select source frames using explicit quality and adjacent-diversity thresholds."""

    validated_candidates = _validate_candidates(candidates)
    if not isinstance(policy, FrameSelectionPolicy):
        raise TypeError("quality_diversity_selection.policy must be FrameSelectionPolicy")
    if not isinstance(config, QualityDiversitySelectionConfig):
        raise TypeError(
            "quality_diversity_selection.config must be QualityDiversitySelectionConfig"
        )
    if policy.optional_metric_names:
        raise ValueError("quality-diversity selection requires empty optional_metric_names")

    expected_metric_names = _required_metric_names(config)
    if policy.required_metric_names != expected_metric_names:
        raise ValueError(
            "frame-selection policy required_metric_names must exactly match "
            "configured quality-diversity metrics"
        )

    adjacent_values = _validate_adjacent_diversity(
        validated_candidates,
        adjacent_diversity_evidence,
        config,
    )

    selected: list[ProbedVideoFrame] = []
    for index, candidate in enumerate(validated_candidates):
        if not _passes_quality(candidate, config):
            continue
        if not selected:
            selected.append(candidate.frame)
            continue
        if config.minimum_adjacent_grid_luma_mae is None:
            selected.append(candidate.frame)
            continue
        if adjacent_values[index - 1] >= config.minimum_adjacent_grid_luma_mae:
            selected.append(candidate.frame)

    return tuple(selected)
