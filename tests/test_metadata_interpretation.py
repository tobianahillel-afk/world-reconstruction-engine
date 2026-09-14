from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import (
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    ObservationId,
    ObservationMetadata,
    RawMetadataEntry,
)
from wre.ingestion import MetadataInterpreter, interpret_observation_metadata
from wre.persistence import PersistenceConflictError, SQLiteLocalStore

OBSERVATION_ID = ObservationId("obs:metadata-interpretation")


def _metadata(*entries: tuple[str, str]) -> ObservationMetadata:
    return ObservationMetadata(
        observation_id=OBSERVATION_ID,
        raw_entries=tuple(
            RawMetadataEntry(namespace="exif", key=key, value=value) for key, value in entries
        ),
    )


def _resolved_entries() -> tuple[tuple[str, str], ...]:
    return (
        ("GPS GPSLatitude", "[40, 26, 30]"),
        ("GPS GPSLatitudeRef", "N"),
        ("GPS GPSLongitude", "[73, 59, 0]"),
        ("GPS GPSLongitudeRef", "W"),
        ("GPS GPSMapDatum", "WGS-84"),
        ("EXIF DateTimeOriginal", "2026:09:14 20:00:00"),
        ("EXIF OffsetTimeOriginal", "+02:00"),
    )


def test_resolved_gps_and_offset_capture_time_are_deterministic() -> None:
    result = interpret_observation_metadata(_metadata(*_resolved_entries()))

    assert result.observation_id == OBSERVATION_ID
    assert result.gps.status is GpsInterpretationStatus.RESOLVED
    assert result.gps.latitude_deg == pytest.approx(40.4416666667)
    assert result.gps.longitude_deg == pytest.approx(-73.9833333333)
    assert result.gps.map_datum == "WGS-84"
    assert result.gps.evidence_keys == (
        "GPS GPSLatitude",
        "GPS GPSLatitudeRef",
        "GPS GPSLongitude",
        "GPS GPSLongitudeRef",
        "GPS GPSMapDatum",
    )
    assert result.capture_time.status is CaptureTimeInterpretationStatus.RESOLVED
    assert result.capture_time.raw_datetime == "2026:09:14 20:00:00"
    assert result.capture_time.raw_offset == "+02:00"
    assert result.capture_time.instant == datetime(2026, 9, 14, 18, 0, tzinfo=UTC)


def test_south_and_east_references_control_coordinate_signs() -> None:
    metadata = _metadata(
        ("GPS GPSLatitude", "[33, 52, 0]"),
        ("GPS GPSLatitudeRef", "S"),
        ("GPS GPSLongitude", "[151, 12, 0]"),
        ("GPS GPSLongitudeRef", "E"),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.RESOLVED
    assert result.gps.latitude_deg == pytest.approx(-33.8666666667)
    assert result.gps.longitude_deg == pytest.approx(151.2)
    assert result.gps.map_datum is None


def test_fractional_dms_components_are_supported_without_float_parsing_guesswork() -> None:
    metadata = _metadata(
        ("GPS GPSLatitude", "[48, 51, 144/10]"),
        ("GPS GPSLatitudeRef", "N"),
        ("GPS GPSLongitude", "[2, 21, 180/10]"),
        ("GPS GPSLongitudeRef", "E"),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.RESOLVED
    assert result.gps.latitude_deg == pytest.approx(48.854)
    assert result.gps.longitude_deg == pytest.approx(2.355)


def test_incomplete_gps_never_exposes_partial_coordinates() -> None:
    metadata = _metadata(
        ("GPS GPSLatitude", "[40, 26, 30]"),
        ("GPS GPSLatitudeRef", "N"),
        ("GPS GPSMapDatum", "WGS-84"),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.INCOMPLETE
    assert result.gps.latitude_deg is None
    assert result.gps.longitude_deg is None
    assert result.gps.map_datum == "WGS-84"
    assert "GPS GPSLongitude" in result.gps.issue
    assert "GPS GPSLongitudeRef" in result.gps.issue


def test_invalid_gps_is_explicit_and_never_exposes_coordinates() -> None:
    metadata = _metadata(
        ("GPS GPSLatitude", "[40, 61, 0]"),
        ("GPS GPSLatitudeRef", "N"),
        ("GPS GPSLongitude", "[73, 59, 0]"),
        ("GPS GPSLongitudeRef", "W"),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.INVALID
    assert result.gps.latitude_deg is None
    assert result.gps.longitude_deg is None
    assert "below 60" in result.gps.issue


def test_duplicate_coordinate_key_is_invalid_instead_of_last_write_wins() -> None:
    metadata = ObservationMetadata(
        observation_id=OBSERVATION_ID,
        raw_entries=(
            RawMetadataEntry(namespace="exif", key="GPS GPSLatitude", value="[40, 0, 0]"),
            RawMetadataEntry(namespace="exif", key="GPS GPSLatitude", value="[41, 0, 0]"),
            RawMetadataEntry(namespace="exif", key="GPS GPSLatitudeRef", value="N"),
            RawMetadataEntry(namespace="exif", key="GPS GPSLongitude", value="[2, 0, 0]"),
            RawMetadataEntry(namespace="exif", key="GPS GPSLongitudeRef", value="E"),
        ),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.INVALID
    assert result.gps.latitude_deg is None
    assert "duplicate GPS metadata keys" in result.gps.issue


def test_local_capture_time_without_offset_stays_ambiguous() -> None:
    result = interpret_observation_metadata(
        _metadata(("EXIF DateTimeOriginal", "2026:09:14 20:00:00"))
    )

    assert result.capture_time.status is CaptureTimeInterpretationStatus.LOCAL_AMBIGUOUS
    assert result.capture_time.raw_datetime == "2026:09:14 20:00:00"
    assert result.capture_time.raw_offset is None
    assert result.capture_time.instant is None


def test_invalid_capture_time_offset_is_explicit() -> None:
    result = interpret_observation_metadata(
        _metadata(
            ("EXIF DateTimeOriginal", "2026:09:14 20:00:00"),
            ("EXIF OffsetTimeOriginal", "+25:00"),
        )
    )

    assert result.capture_time.status is CaptureTimeInterpretationStatus.INVALID
    assert result.capture_time.instant is None
    assert "UTC offset range" in result.capture_time.issue


def test_offset_without_original_datetime_is_invalid() -> None:
    result = interpret_observation_metadata(
        _metadata(("EXIF OffsetTimeOriginal", "+02:00"))
    )

    assert result.capture_time.status is CaptureTimeInterpretationStatus.INVALID
    assert result.capture_time.instant is None
    assert "without EXIF DateTimeOriginal" in result.capture_time.issue


def test_non_exif_entries_do_not_create_gps_or_capture_time_semantics() -> None:
    metadata = ObservationMetadata(
        observation_id=OBSERVATION_ID,
        raw_entries=(RawMetadataEntry(namespace="xmp", key="GPS GPSLatitude", value="[1, 2, 3]"),),
    )

    result = interpret_observation_metadata(metadata)

    assert result.gps.status is GpsInterpretationStatus.ABSENT
    assert result.capture_time.status is CaptureTimeInterpretationStatus.ABSENT


def test_metadata_interpretation_round_trips_and_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "wre.db")
    metadata = _metadata(*_resolved_entries())
    interpreter = MetadataInterpreter(store)

    first = interpreter.ingest(metadata)
    second = interpreter.ingest(metadata)

    assert first == second
    assert store.get_metadata_interpretation(OBSERVATION_ID) == first


def test_changed_interpretation_for_same_observation_is_rejected(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "wre.db")
    interpreter = MetadataInterpreter(store)
    original = _metadata(*_resolved_entries())
    changed = _metadata(
        ("GPS GPSLatitude", "[41, 0, 0]"),
        ("GPS GPSLatitudeRef", "N"),
        ("GPS GPSLongitude", "[73, 59, 0]"),
        ("GPS GPSLongitudeRef", "W"),
        ("EXIF DateTimeOriginal", "2026:09:14 20:00:00"),
        ("EXIF OffsetTimeOriginal", "+02:00"),
    )

    stored = interpreter.ingest(original)

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        interpreter.ingest(changed)

    assert store.get_metadata_interpretation(OBSERVATION_ID) == stored
