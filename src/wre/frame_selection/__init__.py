"""V2 frame-selection adapters."""

from wre.frame_selection.interval import IntervalFrameSelectionConfig, select_interval_frames
from wre.frame_selection.quality_diversity import (
    AdjacentFrameDiversityEvidence,
    FrameSelectionCandidate,
    QualityDiversitySelectionConfig,
    select_quality_diversity_frames,
)

__all__ = [
    "AdjacentFrameDiversityEvidence",
    "FrameSelectionCandidate",
    "IntervalFrameSelectionConfig",
    "QualityDiversitySelectionConfig",
    "select_interval_frames",
    "select_quality_diversity_frames",
]
