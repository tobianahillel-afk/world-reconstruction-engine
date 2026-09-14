from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain import (
    ImageObservation,
    MediaAssetRef,
    Observation,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.ingestion import ImageIngestor, ImageIngestRequest
from wre.persistence import PersistenceConflictError, SQLiteLocalStore

RECEIVED_AT = datetime(2026, 9, 14, 20, 0, tzinfo=UTC)
CAPTURED_AT = datetime(2026, 9, 13, 10, 30, tzinfo=UTC)


def _asset(*, digest: str = "a", uri: str = "file:///input/photo.jpg") -> MediaAssetRef:
    return MediaAssetRef(
        uri=uri,
        sha256=Sha256Digest(digest * 64),
        byte_length=1234,
        mime_type="image/jpeg",
    )


def _request(
    *,
    observation_id: str = "obs:image-001",
    asset: MediaAssetRef | None = None,
    received_at: datetime = RECEIVED_AT,
    captured_at: datetime | None = CAPTURED_AT,
) -> ImageIngestRequest:
    return ImageIngestRequest(
        observation_id=ObservationId(observation_id),
        asset=asset or _asset(),
        source=SourceRef(
            source_id=SourceId("source:local-import"),
            locator="camera-roll/photo.jpg",
        ),
        received_at=received_at,
        captured_at=captured_at,
    )


class CapturingSink:
    def __init__(self) -> None:
        self.observations: list[Observation] = []

    def put_observation(self, observation: Observation) -> None:
        self.observations.append(observation)


def test_image_ingestor_builds_raw_observation_without_storage_coupling() -> None:
    sink = CapturingSink()
    ingestor = ImageIngestor(sink)
    request = _request()

    observation = ingestor.ingest(request)

    assert observation == ImageObservation(
        observation_id=request.observation_id,
        asset=request.asset,
        source=request.source,
        received_at=request.received_at,
        captured_at=request.captured_at,
    )
    assert sink.observations == [observation]


def test_image_ingestion_round_trips_through_local_store(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "state" / "wre.db")
    ingestor = ImageIngestor(store)
    request = _request()

    observation = ingestor.ingest(request)

    assert store.get_observation(request.observation_id) == observation
    assert store.get_observation_metadata(request.observation_id) is None


def test_identical_image_ingestion_is_idempotent(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "wre.db")
    ingestor = ImageIngestor(store)
    request = _request()

    first = ingestor.ingest(request)
    second = ingestor.ingest(request)

    assert first == second
    assert store.get_observation(request.observation_id) == first


def test_same_observation_identity_with_different_asset_is_rejected(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "wre.db")
    ingestor = ImageIngestor(store)
    original = _request()
    conflicting = _request(asset=_asset(digest="b", uri="file:///input/other.jpg"))

    stored = ingestor.ingest(original)

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        ingestor.ingest(conflicting)

    assert store.get_observation(original.observation_id) == stored


def test_ingestion_preserves_absent_capture_time_without_inference(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "wre.db")
    request = _request(captured_at=None)

    observation = ImageIngestor(store).ingest(request)

    assert observation.captured_at is None
    assert store.get_observation(request.observation_id) == observation


def test_ingestion_rejects_ambiguous_received_time_through_domain_invariant() -> None:
    sink = CapturingSink()
    naive_received_at = datetime(2026, 9, 14, 20, 0)

    with pytest.raises(ValueError, match="received_at must be timezone-aware"):
        ImageIngestor(sink).ingest(_request(received_at=naive_received_at))

    assert sink.observations == []
