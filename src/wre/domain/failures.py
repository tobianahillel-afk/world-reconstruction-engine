from __future__ import annotations

from enum import StrEnum


class FailureCategory(StrEnum):
    """Stable cross-adapter category for an explicit WRE failure outcome."""

    UNSUPPORTED_INPUT = "unsupported_input"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    INSUFFICIENT_OVERLAP = "insufficient_overlap"
    CAMERA_AMBIGUITY = "camera_ambiguity"
    CALIBRATION_FAILURE = "calibration_failure"
    GEOMETRIC_INCONSISTENCY = "geometric_inconsistency"
    DYNAMIC_CONTAMINATION = "dynamic_contamination"
    DEPTH_INCONSISTENCY = "depth_inconsistency"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT = "timeout"
    CORRUPTED_MEDIA = "corrupted_media"
    CHECKPOINT_INCOMPATIBILITY = "checkpoint_incompatibility"
    QUALITY_GATE_FAILURE = "quality_gate_failure"
    UNKNOWN_INTERNAL_ERROR = "unknown_internal_error"
