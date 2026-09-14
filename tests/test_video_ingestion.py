from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import (
    MediaAssetRef,
    ObservationId,
    ObservationKind,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoObservation,
)
from wre.ingestion import (
    LocalVideoIngestRequest,
    LocalVideoIngestor,
    VideoIngestRequest,
    VideoIngestor,
)
from wre.persistence import SQLiteLocalStore

NOW = datetime(2026, 9, 15, 1, 0, tzinfo=UTC)


def _source(locator: str = "incoming/source.mp4") -> SourceRef:
    return SourceRef(source_id=SourceId("upload:video"), locator=locator)


def _store(tmp_path: Path) -> SQLiteLocalStore:
    return SQLiteLocalStore(tmp_path / "state" / "wre.sqlite3")


def test_preaddressed_video_ingestion_persists_raw_video_observation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    asset = MediaAssetRef(
        uri="s3://bucket/source.mp4",
        sha256=Sha256Digest("a" * 64),
        byte_length=12_345,
        mime_type="video/mp4",
    )
    request = VideoIngestRequest(
        observation_id=ObservationId("obs:video:0001"),
        asset=asset,
        source=_source(),
        received_at=NOW,
        captured_at=None,
    )

    observation = VideoIngestor(store).ingest(request)

    assert isinstance(observation, VideoObservation)
    assert observation.kind is ObservationKind.VIDEO
    assert observation.asset == asset
    assert observation.captured_at is None
    assert store.get_observation(observation.observation_id) == observation


def test_local_video_ingestion_hashes_bytes_and_preserves_explicit_fields(tmp_path: Path) -> None:
    path = tmp_path / "clip.bin"
    payload = (b"not-decoded-video-bytes" * 17) + b"tail"
    path.write_bytes(payload)
    store = _store(tmp_path)

    result = LocalVideoIngestor(store, chunk_size=7).ingest(
        LocalVideoIngestRequest(
            path=path,
            observation_id=ObservationId("obs:video:local"),
            source=_source("drop/clip.mov"),
            received_at=NOW,
            asset_uri="archive://videos/clip.mov",
            mime_type="video/quicktime",
        )
    )

    observation = result.observation
    assert observation.asset.sha256 == Sha256Digest(hashlib.sha256(payload).hexdigest())
    assert observation.asset.byte_length == len(payload)
    assert observation.asset.uri == "archive://videos/clip.mov"
    assert observation.asset.mime_type == "video/quicktime"
    assert observation.source == _source("drop/clip.mov")
    assert not result.has_duplicate_content
    assert store.get_observation(observation.observation_id) == observation


def test_local_video_ingestion_does_not_infer_mime_or_capture_time(tmp_path: Path) -> None:
    path = tmp_path / "looks-like-video.mp4"
    path.write_bytes(b"opaque source bytes")
    store = _store(tmp_path)

    result = LocalVideoIngestor(store).ingest(
        LocalVideoIngestRequest(
            path=path,
            observation_id=ObservationId("obs:video:no-inference"),
            source=_source(),
            received_at=NOW,
        )
    )

    assert result.observation.asset.uri == path.resolve().as_uri()
    assert result.observation.asset.mime_type is None
    assert result.observation.captured_at is None


def test_exact_content_duplicates_preserve_distinct_video_observations(tmp_path: Path) -> None:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"same opaque video bytes")
    store = _store(tmp_path)
    ingestor = LocalVideoIngestor(store)

    first = ingestor.ingest(
        LocalVideoIngestRequest(
            path=path,
            observation_id=ObservationId("obs:video:first"),
            source=_source("first/source.mp4"),
            received_at=NOW,
        )
    )
    second = ingestor.ingest(
        LocalVideoIngestRequest(
            path=path,
            observation_id=ObservationId("obs:video:second"),
            source=_source("second/source.mp4"),
            received_at=NOW,
        )
    )

    assert not first.has_duplicate_content
    assert second.duplicate_observation_ids == (ObservationId("obs:video:first"),)
    assert store.get_observation(ObservationId("obs:video:first")) == first.observation
    assert store.get_observation(ObservationId("obs:video:second")) == second.observation


def test_retry_of_same_video_observation_is_idempotent_and_excludes_self(tmp_path: Path) -> None:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"retry bytes")
    store = _store(tmp_path)
    ingestor = LocalVideoIngestor(store)
    request = LocalVideoIngestRequest(
        path=path,
        observation_id=ObservationId("obs:video:retry"),
        source=_source(),
        received_at=NOW,
    )

    first = ingestor.ingest(request)
    second = ingestor.ingest(request)

    assert first.observation == second.observation
    assert second.duplicate_observation_ids == ()


def test_missing_video_file_fails_before_persistence(tmp_path: Path) -> None:
    store = _store(tmp_path)
    observation_id = ObservationId("obs:video:missing")

    with pytest.raises(FileNotFoundError):
        LocalVideoIngestor(store).ingest(
            LocalVideoIngestRequest(
                path=tmp_path / "missing.mp4",
                observation_id=observation_id,
                source=_source(),
                received_at=NOW,
            )
        )

    assert store.get_observation(observation_id) is None


def test_video_ingestion_rejects_invalid_hash_chunk_size(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(ValueError, match="chunk_size"):
        LocalVideoIngestor(store, chunk_size=0)
