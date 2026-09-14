from __future__ import annotations

import struct
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import ObservationId, SourceId, SourceRef
from wre.ingestion import (
    ExifMetadataIngestor,
    ExifMetadataIngestRequest,
    LocalImageIngestRequest,
    LocalImageIngestor,
    extract_exif_entries,
)
from wre.persistence import PersistenceConflictError, SQLiteLocalStore

RECEIVED_AT = datetime(2026, 9, 14, 21, 0, tzinfo=UTC)


def _ifd_entry(tag: int, field_type: int, count: int, value_or_offset: bytes) -> bytes:
    assert len(value_or_offset) == 4
    return struct.pack("<HHI", tag, field_type, count) + value_or_offset


def _write_exif_tiff(path: Path, *, make: str = "ACME") -> None:
    make_bytes = make.encode("ascii") + b"\x00"
    model_bytes = b"Model-X\x00"
    captured_bytes = b"2026:09:14 20:00:00\x00"

    ifd0_offset = 8
    ifd0_size = 2 + (4 * 12) + 4
    make_offset = ifd0_offset + ifd0_size
    model_offset = make_offset + len(make_bytes)
    exif_ifd_offset = model_offset + len(model_bytes)
    if exif_ifd_offset % 2:
        exif_ifd_offset += 1

    exif_ifd_size = 2 + 12 + 4
    captured_offset = exif_ifd_offset + exif_ifd_size
    gps_ifd_offset = captured_offset + len(captured_bytes)
    gps_ifd_size = 2 + (2 * 12) + 4
    gps_rationals_offset = gps_ifd_offset + gps_ifd_size

    payload = bytearray(struct.pack("<2sHI", b"II", 42, ifd0_offset))
    payload += struct.pack("<H", 4)
    payload += _ifd_entry(0x010F, 2, len(make_bytes), struct.pack("<I", make_offset))
    payload += _ifd_entry(0x0110, 2, len(model_bytes), struct.pack("<I", model_offset))
    payload += _ifd_entry(0x8769, 4, 1, struct.pack("<I", exif_ifd_offset))
    payload += _ifd_entry(0x8825, 4, 1, struct.pack("<I", gps_ifd_offset))
    payload += struct.pack("<I", 0)
    assert len(payload) == make_offset

    payload += make_bytes
    assert len(payload) == model_offset
    payload += model_bytes
    while len(payload) < exif_ifd_offset:
        payload += b"\x00"

    payload += struct.pack("<H", 1)
    payload += _ifd_entry(0x9003, 2, len(captured_bytes), struct.pack("<I", captured_offset))
    payload += struct.pack("<I", 0)
    assert len(payload) == captured_offset
    payload += captured_bytes
    assert len(payload) == gps_ifd_offset

    payload += struct.pack("<H", 2)
    payload += _ifd_entry(0x0001, 2, 2, b"N\x00\x00\x00")
    payload += _ifd_entry(0x0002, 5, 3, struct.pack("<I", gps_rationals_offset))
    payload += struct.pack("<I", 0)
    assert len(payload) == gps_rationals_offset
    payload += struct.pack("<IIIIII", 40, 1, 26, 1, 3000, 100)

    path.write_bytes(payload)


def _write_empty_tiff(path: Path) -> None:
    path.write_bytes(struct.pack("<2sHIHI", b"II", 42, 8, 0, 0))


def _local_request(path: Path, observation_id: ObservationId) -> LocalImageIngestRequest:
    return LocalImageIngestRequest(
        path=path,
        observation_id=observation_id,
        source=SourceRef(
            source_id=SourceId("source:exif-fixture"),
            locator=path.name,
        ),
        received_at=RECEIVED_AT,
        mime_type="image/tiff",
    )


def test_extract_exif_entries_preserves_standard_tags_without_semantic_conversion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.tiff"
    _write_exif_tiff(path)

    entries = extract_exif_entries(path)
    by_key = {entry.key: entry for entry in entries}

    assert [entry.key for entry in entries] == sorted(entry.key for entry in entries)
    assert all(entry.namespace == "exif" for entry in entries)
    assert by_key["Image Make"].value == "ACME"
    assert by_key["Image Model"].value == "Model-X"
    assert by_key["EXIF DateTimeOriginal"].value == "2026:09:14 20:00:00"
    assert by_key["GPS GPSLatitudeRef"].value == "N"
    assert "40" in by_key["GPS GPSLatitude"].value
    assert "26" in by_key["GPS GPSLatitude"].value


def test_exif_metadata_ingestion_persists_raw_tags_without_updating_observation_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.tiff"
    _write_exif_tiff(path)
    store = SQLiteLocalStore(tmp_path / "wre.db")
    observation_id = ObservationId("obs:exif-001")
    observation = LocalImageIngestor(store).ingest(_local_request(path, observation_id)).observation

    metadata = ExifMetadataIngestor(store).ingest(
        ExifMetadataIngestRequest(path=path, observation_id=observation_id)
    )

    assert metadata.camera_id is None
    assert metadata.dimensions is None
    assert store.get_observation_metadata(observation_id) == metadata
    assert store.get_observation(observation_id) == observation
    assert observation.captured_at is None


def test_exif_metadata_ingestion_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "metadata.tiff"
    _write_exif_tiff(path)
    store = SQLiteLocalStore(tmp_path / "wre.db")
    observation_id = ObservationId("obs:exif-retry")
    ingestor = ExifMetadataIngestor(store)
    request = ExifMetadataIngestRequest(path=path, observation_id=observation_id)

    first = ingestor.ingest(request)
    second = ingestor.ingest(request)

    assert first == second
    assert store.get_observation_metadata(observation_id) == first


def test_changed_exif_for_same_metadata_identity_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "metadata.tiff"
    _write_exif_tiff(path, make="ACME")
    store = SQLiteLocalStore(tmp_path / "wre.db")
    observation_id = ObservationId("obs:exif-conflict")
    ingestor = ExifMetadataIngestor(store)
    request = ExifMetadataIngestRequest(path=path, observation_id=observation_id)

    original = ingestor.ingest(request)
    _write_exif_tiff(path, make="OTHER")

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        ingestor.ingest(request)

    assert store.get_observation_metadata(observation_id) == original


def test_image_without_exif_produces_empty_raw_metadata(tmp_path: Path) -> None:
    path = tmp_path / "empty.tiff"
    _write_empty_tiff(path)
    store = SQLiteLocalStore(tmp_path / "wre.db")
    observation_id = ObservationId("obs:no-exif")

    metadata = ExifMetadataIngestor(store).ingest(
        ExifMetadataIngestRequest(path=path, observation_id=observation_id)
    )

    assert metadata.raw_entries == ()
    assert store.get_observation_metadata(observation_id) == metadata


def test_missing_exif_file_fails_before_metadata_persistence(tmp_path: Path) -> None:
    path = tmp_path / "missing.tiff"
    store = SQLiteLocalStore(tmp_path / "wre.db")
    observation_id = ObservationId("obs:missing-exif")

    with pytest.raises(FileNotFoundError):
        ExifMetadataIngestor(store).ingest(
            ExifMetadataIngestRequest(path=path, observation_id=observation_id)
        )

    assert store.get_observation_metadata(observation_id) is None
