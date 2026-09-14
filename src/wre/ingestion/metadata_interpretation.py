from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta, timezone
from fractions import Fraction
from typing import Protocol

from wre.domain.cameras import ObservationMetadata
from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
)
from wre.ingestion.exif import EXIF_NAMESPACE

_GPS_LATITUDE = "GPS GPSLatitude"
_GPS_LATITUDE_REF = "GPS GPSLatitudeRef"
_GPS_LONGITUDE = "GPS GPSLongitude"
_GPS_LONGITUDE_REF = "GPS GPSLongitudeRef"
_GPS_MAP_DATUM = "GPS GPSMapDatum"
_GPS_REQUIRED_KEYS = (
    _GPS_LATITUDE,
    _GPS_LATITUDE_REF,
    _GPS_LONGITUDE,
    _GPS_LONGITUDE_REF,
)

_EXIF_DATETIME_ORIGINAL = "EXIF DateTimeOriginal"
_EXIF_OFFSET_TIME_ORIGINAL = "EXIF OffsetTimeOriginal"
_EXIF_CAPTURE_TIME_KEYS = (_EXIF_DATETIME_ORIGINAL, _EXIF_OFFSET_TIME_ORIGINAL)
_OFFSET_RE = re.compile(r"^(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$")


class MetadataInterpretationSink(Protocol):
    """Persistence boundary for deterministic metadata interpretations."""

    def put_metadata_interpretation(
        self,
        interpretation: ObservationMetadataInterpretation,
    ) -> None:
        """Persist one interpretation or reject an identity conflict."""
        ...


def _exif_values(metadata: ObservationMetadata) -> tuple[dict[str, str], frozenset[str]]:
    values: dict[str, str] = {}
    duplicates: set[str] = set()
    for entry in metadata.raw_entries:
        if entry.namespace != EXIF_NAMESPACE:
            continue
        if entry.key in values:
            duplicates.add(entry.key)
            continue
        values[entry.key] = entry.value
    return values, frozenset(duplicates)


def _fraction(text: str) -> Fraction:
    try:
        return Fraction(text.strip())
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"invalid numeric component {text!r}") from exc


def _fraction_list(value: str, *, expected: int, context: str) -> tuple[Fraction, ...]:
    text = value.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"{context} must be a bracketed list")
    inner = text[1:-1].strip()
    parts = [] if not inner else [item.strip() for item in inner.split(",")]
    if len(parts) != expected:
        raise ValueError(f"{context} must contain exactly {expected} components")
    return tuple(_fraction(item) for item in parts)


def _dms_to_degrees(value: str, *, context: str) -> float:
    degrees, minutes, seconds = _fraction_list(value, expected=3, context=context)
    if degrees < 0 or minutes < 0 or seconds < 0:
        raise ValueError(f"{context} components must be non-negative")
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"{context} minutes and seconds must be below 60")
    return float(degrees + minutes / 60 + seconds / 3600)


def _invalid_gps(evidence_keys: tuple[str, ...], issue: str) -> GpsMetadataInterpretation:
    return GpsMetadataInterpretation(
        status=GpsInterpretationStatus.INVALID,
        evidence_keys=evidence_keys,
        issue=issue,
    )


def _interpret_gps(
    values: dict[str, str],
    duplicates: frozenset[str],
) -> GpsMetadataInterpretation:
    all_gps_keys = tuple(sorted(key for key in values if key.startswith("GPS ")))
    duplicate_gps_keys = tuple(sorted(key for key in duplicates if key.startswith("GPS ")))
    evidence_keys = tuple(sorted(set(all_gps_keys) | set(duplicate_gps_keys)))
    if not evidence_keys:
        return GpsMetadataInterpretation(status=GpsInterpretationStatus.ABSENT)

    relevant_duplicates = sorted(
        key for key in duplicates if key in {*_GPS_REQUIRED_KEYS, _GPS_MAP_DATUM}
    )
    if relevant_duplicates:
        return _invalid_gps(
            evidence_keys,
            "duplicate GPS metadata keys: " + ", ".join(relevant_duplicates),
        )

    missing = [key for key in _GPS_REQUIRED_KEYS if key not in values]
    if missing:
        return GpsMetadataInterpretation(
            status=GpsInterpretationStatus.INCOMPLETE,
            map_datum=values.get(_GPS_MAP_DATUM),
            evidence_keys=evidence_keys,
            issue="missing required GPS metadata keys: " + ", ".join(missing),
        )

    try:
        latitude = _dms_to_degrees(values[_GPS_LATITUDE], context=_GPS_LATITUDE)
        longitude = _dms_to_degrees(values[_GPS_LONGITUDE], context=_GPS_LONGITUDE)
        latitude_ref = values[_GPS_LATITUDE_REF].strip().upper()
        longitude_ref = values[_GPS_LONGITUDE_REF].strip().upper()
        if latitude_ref not in {"N", "S"}:
            raise ValueError("GPS latitude reference must be N or S")
        if longitude_ref not in {"E", "W"}:
            raise ValueError("GPS longitude reference must be E or W")
        if latitude_ref == "S":
            latitude = -latitude
        if longitude_ref == "W":
            longitude = -longitude
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("GPS latitude is outside [-90, 90]")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("GPS longitude is outside [-180, 180]")
    except ValueError as exc:
        return _invalid_gps(evidence_keys, str(exc))

    return GpsMetadataInterpretation(
        status=GpsInterpretationStatus.RESOLVED,
        latitude_deg=latitude,
        longitude_deg=longitude,
        map_datum=values.get(_GPS_MAP_DATUM),
        evidence_keys=tuple(
            sorted(key for key in (*_GPS_REQUIRED_KEYS, _GPS_MAP_DATUM) if key in values)
        ),
    )


def _invalid_capture_time(
    *,
    raw_datetime: str | None,
    raw_offset: str | None,
    evidence_keys: tuple[str, ...],
    issue: str,
) -> CaptureTimeInterpretation:
    return CaptureTimeInterpretation(
        status=CaptureTimeInterpretationStatus.INVALID,
        raw_datetime=raw_datetime,
        raw_offset=raw_offset,
        evidence_keys=evidence_keys,
        issue=issue,
    )


def _parse_offset(value: str) -> timezone:
    match = _OFFSET_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError("EXIF OffsetTimeOriginal must use ±HH:MM")
    hours = int(match.group("hours"))
    minutes = int(match.group("minutes"))
    if hours >= 24 or minutes >= 60:
        raise ValueError("EXIF OffsetTimeOriginal is outside a valid UTC offset range")
    offset = timedelta(hours=hours, minutes=minutes)
    if match.group("sign") == "-":
        offset = -offset
    return timezone(offset)


def _interpret_capture_time(
    values: dict[str, str],
    duplicates: frozenset[str],
) -> CaptureTimeInterpretation:
    evidence_keys = tuple(
        sorted(
            key for key in _EXIF_CAPTURE_TIME_KEYS if key in values or key in duplicates
        )
    )
    if not evidence_keys:
        return CaptureTimeInterpretation(status=CaptureTimeInterpretationStatus.ABSENT)

    raw_datetime = values.get(_EXIF_DATETIME_ORIGINAL)
    raw_offset = values.get(_EXIF_OFFSET_TIME_ORIGINAL)
    duplicate_keys = sorted(key for key in duplicates if key in _EXIF_CAPTURE_TIME_KEYS)
    if duplicate_keys:
        return _invalid_capture_time(
            raw_datetime=raw_datetime,
            raw_offset=raw_offset,
            evidence_keys=evidence_keys,
            issue="duplicate capture-time metadata keys: " + ", ".join(duplicate_keys),
        )
    if raw_datetime is None:
        return _invalid_capture_time(
            raw_datetime=None,
            raw_offset=raw_offset,
            evidence_keys=evidence_keys,
            issue="EXIF OffsetTimeOriginal is present without EXIF DateTimeOriginal",
        )

    try:
        local_datetime = datetime.strptime(raw_datetime.strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return _invalid_capture_time(
            raw_datetime=raw_datetime,
            raw_offset=raw_offset,
            evidence_keys=evidence_keys,
            issue="EXIF DateTimeOriginal must use YYYY:MM:DD HH:MM:SS",
        )

    if raw_offset is None:
        return CaptureTimeInterpretation(
            status=CaptureTimeInterpretationStatus.LOCAL_AMBIGUOUS,
            raw_datetime=raw_datetime,
            evidence_keys=evidence_keys,
        )

    try:
        source_timezone = _parse_offset(raw_offset)
    except ValueError as exc:
        return _invalid_capture_time(
            raw_datetime=raw_datetime,
            raw_offset=raw_offset,
            evidence_keys=evidence_keys,
            issue=str(exc),
        )

    instant = local_datetime.replace(tzinfo=source_timezone).astimezone(UTC)
    return CaptureTimeInterpretation(
        status=CaptureTimeInterpretationStatus.RESOLVED,
        raw_datetime=raw_datetime,
        raw_offset=raw_offset,
        instant=instant,
        evidence_keys=evidence_keys,
    )


def interpret_observation_metadata(
    metadata: ObservationMetadata,
) -> ObservationMetadataInterpretation:
    """Interpret GPS and capture-time semantics without modifying the raw observation."""

    values, duplicates = _exif_values(metadata)
    return ObservationMetadataInterpretation(
        observation_id=metadata.observation_id,
        gps=_interpret_gps(values, duplicates),
        capture_time=_interpret_capture_time(values, duplicates),
    )


class MetadataInterpreter:
    """Interpret and persist GPS/time semantics derived from immutable raw metadata."""

    def __init__(self, sink: MetadataInterpretationSink) -> None:
        self._sink = sink

    def ingest(self, metadata: ObservationMetadata) -> ObservationMetadataInterpretation:
        interpretation = interpret_observation_metadata(metadata)
        self._sink.put_metadata_interpretation(interpretation)
        return interpretation
