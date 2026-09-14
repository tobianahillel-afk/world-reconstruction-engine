from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from wre.domain.observations import ObservationId


def _canonical_evidence_keys(value: tuple[str, ...], context: str) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{context} must be an immutable tuple")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"{context} must contain only non-blank strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{context} must not contain duplicates")
    return tuple(sorted(value))


def _optional_non_blank(value: str | None, context: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{context} must be non-blank when present")


class GpsInterpretationStatus(StrEnum):
    ABSENT = "absent"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class GpsMetadataInterpretation:
    """Deterministic interpretation of EXIF GPS coordinate metadata."""

    status: GpsInterpretationStatus
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    map_datum: str | None = None
    evidence_keys: tuple[str, ...] = ()
    issue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_keys",
            _canonical_evidence_keys(self.evidence_keys, "gps.evidence_keys"),
        )
        _optional_non_blank(self.map_datum, "gps.map_datum")
        _optional_non_blank(self.issue, "gps.issue")

        if self.status is GpsInterpretationStatus.ABSENT:
            if (
                self.latitude_deg is not None
                or self.longitude_deg is not None
                or self.map_datum is not None
                or self.evidence_keys
                or self.issue is not None
            ):
                raise ValueError("absent GPS interpretation must not carry evidence or values")
            return

        if self.status is GpsInterpretationStatus.RESOLVED:
            if self.latitude_deg is None or self.longitude_deg is None:
                raise ValueError("resolved GPS interpretation requires latitude and longitude")
            if self.issue is not None:
                raise ValueError("resolved GPS interpretation must not carry an issue")
            if not self.evidence_keys:
                raise ValueError("resolved GPS interpretation requires evidence keys")
            if not math.isfinite(self.latitude_deg) or not -90.0 <= self.latitude_deg <= 90.0:
                raise ValueError("gps.latitude_deg must be finite and within [-90, 90]")
            if not math.isfinite(self.longitude_deg) or not -180.0 <= self.longitude_deg <= 180.0:
                raise ValueError("gps.longitude_deg must be finite and within [-180, 180]")
            return

        if self.latitude_deg is not None or self.longitude_deg is not None:
            raise ValueError("unresolved GPS interpretation must not expose coordinates")
        if not self.evidence_keys:
            raise ValueError("unresolved GPS interpretation requires evidence keys")
        if self.issue is None:
            raise ValueError("incomplete or invalid GPS interpretation requires an issue")


class CaptureTimeInterpretationStatus(StrEnum):
    ABSENT = "absent"
    LOCAL_AMBIGUOUS = "local_ambiguous"
    INVALID = "invalid"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class CaptureTimeInterpretation:
    """Interpretation of EXIF capture time without guessing a missing timezone."""

    status: CaptureTimeInterpretationStatus
    raw_datetime: str | None = None
    raw_offset: str | None = None
    instant: datetime | None = None
    evidence_keys: tuple[str, ...] = ()
    issue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_keys",
            _canonical_evidence_keys(self.evidence_keys, "capture_time.evidence_keys"),
        )
        _optional_non_blank(self.raw_datetime, "capture_time.raw_datetime")
        _optional_non_blank(self.raw_offset, "capture_time.raw_offset")
        _optional_non_blank(self.issue, "capture_time.issue")

        if self.instant is not None and (
            self.instant.tzinfo is None or self.instant.utcoffset() is None
        ):
            raise ValueError("capture_time.instant must be timezone-aware")

        if self.status is CaptureTimeInterpretationStatus.ABSENT:
            if (
                self.raw_datetime is not None
                or self.raw_offset is not None
                or self.instant is not None
                or self.evidence_keys
                or self.issue is not None
            ):
                raise ValueError("absent capture-time interpretation must not carry evidence")
            return

        if self.status is CaptureTimeInterpretationStatus.LOCAL_AMBIGUOUS:
            if self.raw_datetime is None or self.raw_offset is not None or self.instant is not None:
                raise ValueError(
                    "local-ambiguous capture time requires only a raw local datetime"
                )
            if not self.evidence_keys or self.issue is not None:
                raise ValueError("local-ambiguous capture time requires evidence and no issue")
            return

        if self.status is CaptureTimeInterpretationStatus.RESOLVED:
            if self.raw_datetime is None or self.raw_offset is None or self.instant is None:
                raise ValueError("resolved capture time requires raw datetime, offset and instant")
            if not self.evidence_keys or self.issue is not None:
                raise ValueError("resolved capture time requires evidence and no issue")
            return

        if self.instant is not None:
            raise ValueError("invalid capture time must not expose an instant")
        if not self.evidence_keys or self.issue is None:
            raise ValueError("invalid capture time requires evidence keys and an issue")


@dataclass(frozen=True, slots=True)
class ObservationMetadataInterpretation:
    """Derived GPS/time semantics for one immutable raw observation metadata record."""

    observation_id: ObservationId
    gps: GpsMetadataInterpretation
    capture_time: CaptureTimeInterpretation
