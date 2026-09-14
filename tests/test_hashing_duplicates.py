from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import (
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.ingestion import ImageIngestor, ImageIngestRequest
from wre.ingestion.hashing import (
    FileContentHash,
    LocalImageIngestor,
    LocalImageIngestRequest,
    hash_file_content,
)
from wre.persistence import SQLiteLocalStore

ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
RECEIVED_AT = datetime(2026, 9, 14, 20, 30, tzinfo=UTC)


def _source(value: str) -> SourceRef:
    return SourceRef(source_id=SourceId(f"source:{value}"), locator=f"{value}.jpg")


def _request(path: Path, observation_id: str) -> LocalImageIngestRequest:
    suffix = observation_id.split(":", maxsplit=1)[-1]
    return LocalImageIngestRequest(
        path=path,
        observation_id=ObservationId(observation_id),
        source=_source(suffix),
        received_at=RECEIVED_AT,
    )


def test_hash_file_content_streams_sha256_and_exact_byte_length(tmp_path: Path) -> None:
    path = tmp_path / "image.bin"
    path.write_bytes(b"abc")

    result = hash_file_content(path, chunk_size=1)

    assert result == FileContentHash(
        sha256=Sha256Digest(ABC_SHA256),
        byte_length=3,
    )


@pytest.mark.parametrize("chunk_size", [0, -1, True])
def test_hash_file_content_rejects_invalid_chunk_size(
    tmp_path: Path,
    chunk_size: int,
) -> None:
    path = tmp_path / "image.bin"
    path.write_bytes(b"abc")

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        hash_file_content(path, chunk_size=chunk_size)


def test_local_image_ingestion_hashes_and_persists_asset(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"abc")
    store = SQLiteLocalStore(tmp_path / "wre.db")

    result = LocalImageIngestor(store, chunk_size=2).ingest(_request(path, "obs:first"))

    assert result.observation.asset == MediaAssetRef(
        uri=path.resolve().as_uri(),
        sha256=Sha256Digest(ABC_SHA256),
        byte_length=3,
        mime_type=None,
    )
    assert result.duplicate_observation_ids == ()
    assert result.has_duplicate_content is False
    assert store.get_observation(ObservationId("obs:first")) == result.observation
    assert store.get_observation_metadata(ObservationId("obs:first")) is None


def test_duplicate_content_is_reported_without_dropping_observations(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"same bytes")
    store = SQLiteLocalStore(tmp_path / "wre.db")
    ingestor = LocalImageIngestor(store)

    first = ingestor.ingest(_request(path, "obs:z"))
    second = ingestor.ingest(_request(path, "obs:a"))
    third = ingestor.ingest(_request(path, "obs:m"))

    assert first.duplicate_observation_ids == ()
    assert second.duplicate_observation_ids == (ObservationId("obs:z"),)
    assert third.duplicate_observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:z"),
    )
    assert third.has_duplicate_content is True
    assert store.get_observation(ObservationId("obs:z")) == first.observation
    assert store.get_observation(ObservationId("obs:a")) == second.observation
    assert store.get_observation(ObservationId("obs:m")) == third.observation


def test_retry_excludes_its_own_observation_from_duplicate_matches(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"retry bytes")
    store = SQLiteLocalStore(tmp_path / "wre.db")
    ingestor = LocalImageIngestor(store)
    request = _request(path, "obs:retry")

    first = ingestor.ingest(request)
    second = ingestor.ingest(request)

    assert first.observation == second.observation
    assert first.duplicate_observation_ids == ()
    assert second.duplicate_observation_ids == ()


def test_duplicate_lookup_requires_matching_byte_length(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"abc")
    store = SQLiteLocalStore(tmp_path / "wre.db")
    fake_asset = MediaAssetRef(
        uri="file:///fake.jpg",
        sha256=Sha256Digest(ABC_SHA256),
        byte_length=999,
        mime_type="image/jpeg",
    )
    ImageIngestor(store).ingest(
        ImageIngestRequest(
            observation_id=ObservationId("obs:wrong-length"),
            asset=fake_asset,
            source=_source("fake"),
            received_at=RECEIVED_AT,
        )
    )

    result = LocalImageIngestor(store).ingest(_request(path, "obs:real"))

    assert result.duplicate_observation_ids == ()
    assert result.observation.asset.byte_length == 3


def test_explicit_asset_uri_and_mime_type_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "photo.jpg"
    path.write_bytes(b"asset bytes")
    store = SQLiteLocalStore(tmp_path / "wre.db")
    request = LocalImageIngestRequest(
        path=path,
        observation_id=ObservationId("obs:remote-uri"),
        source=_source("remote"),
        received_at=RECEIVED_AT,
        asset_uri="https://media.example.invalid/photo.jpg",
        mime_type="image/jpeg",
    )

    result = LocalImageIngestor(store).ingest(request)

    assert result.observation.asset.uri == "https://media.example.invalid/photo.jpg"
    assert result.observation.asset.mime_type == "image/jpeg"


def test_missing_file_fails_before_any_observation_is_persisted(tmp_path: Path) -> None:
    path = tmp_path / "missing.jpg"
    store = SQLiteLocalStore(tmp_path / "wre.db")
    request = _request(path, "obs:missing")

    with pytest.raises(FileNotFoundError):
        LocalImageIngestor(store).ingest(request)

    assert store.get_observation(request.observation_id) is None
