from __future__ import annotations

from enum import StrEnum


class QualityDecision(StrEnum):
    """Stable explicit decision emitted by WRE quality evaluation."""

    PASS = "pass"
    ACCEPT_WITH_WARNINGS = "accept_with_warnings"
    RETRY = "retry"
    ESCALATE = "escalate"
    UNRESOLVED = "unresolved"
