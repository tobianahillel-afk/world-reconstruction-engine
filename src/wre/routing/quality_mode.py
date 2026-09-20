from __future__ import annotations

from dataclasses import dataclass

from wre.domain import QualityMode


@dataclass(frozen=True, slots=True)
class RouteQualityRequest:
    """Minimal route-facing request for one canonical WRE product quality mode."""

    quality_mode: QualityMode

    def __post_init__(self) -> None:
        if not isinstance(self.quality_mode, QualityMode):
            raise TypeError("route_quality_request.quality_mode must be QualityMode")
