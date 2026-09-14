from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceRef,
    VideoObservation,
)
from wre.ingestion.images import ImageIngestor, ImageIngestRequest, ObservationSink
from wre.ingestion.videos import VideoIngestor, VideoIngestRequest

DEFAULT_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FileContentHash:
    """SHA-256 identity and exact byte count for one read of a file."""

    sha256: Sha256Digest
    byte_length: int


class DuplicateObservationLookup(Protocol):
    """Lookup exact-content matches among already persisted observations."""

    def find_observation_ids_by_content(
        self,
        *,
        sha256: Sha256Digest,
        byte_length: int,
    ) -> tuple[ObservationId, ...]:
        """Return observations with the same content digest and byte length."""
        ...


class _HashingObservationStore(ObservationSink, DuplicateObservationLookup, Protocol):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalImageIngestRequest:
    """Explicit local-file input for deterministic hashing and image ingestion."""

    path: Path
    observation_id: ObservationId
    source: SourceRef
    received_at: datetime
    captured_at: datetime | None = None
    asset_uri: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class LocalImageIngestResult:
    observation: ImageObservation
    duplicate_observation_ids: tuple[ObservationId, ...]

    @property
    def has_duplicate_content(self) -> bool:
        return bool(self.duplicate_observation_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class LocalVideoIngestRequest:
    """Explicit local-file input for hashing one source video without decoding it."""

    path: Path
    observation_id: ObservationId
    source: SourceRef
    received_at: datetime
    captured_at: datetime | None = None
    asset_uri: str | None = None
    mime_type: str | None = None


@dataclass(frozen=True, slots=True)
class LocalVideoIngestResult:
    observation: VideoObservation
    duplicate_observation_ids: tuple[ObservationId, ...]

    @property
    def has_duplicate_content(self) -> bool:
        return bool(self.duplicate_observation_ids)


def _validate_chunk_size(chunk_size: int) -> None:
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")


def hash_file_content(
    path: str | Path,
    *,
    chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
) -> FileContentHash:
    """Hash a file deterministically without loading the whole asset into memory."""

    _validate_chunk_size(chunk_size)
    hasher = hashlib.sha256()
    byte_length = 0

    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            hasher.update(chunk)
            byte_length += len(chunk)

    return FileContentHash(
        sha256=Sha256Digest(hasher.hexdigest()),
        byte_length=byte_length,
    )


def _canonical_duplicate_ids(
    observation_ids: tuple[ObservationId, ...],
    *,
    exclude: ObservationId,
) -> tuple[ObservationId, ...]:
    by_value = {
        observation_id.value: observation_id
        for observation_id in observation_ids
        if observation_id != exclude
    }
    return tuple(sorted(by_value.values(), key=lambda item: item.value))


class LocalImageIngestor:
    """Hash a local image, detect exact-content duplicates, then use L2.1 ingestion."""

    def __init__(
        self,
        store: _HashingObservationStore,
        *,
        chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    ) -> None:
        _validate_chunk_size(chunk_size)
        self._store = store
        self._chunk_size = chunk_size
        self._image_ingestor = ImageIngestor(store)

    def ingest(self, request: LocalImageIngestRequest) -> LocalImageIngestResult:
        resolved_path = request.path.expanduser().resolve(strict=True)
        content_hash = hash_file_content(resolved_path, chunk_size=self._chunk_size)
        asset = MediaAssetRef(
            uri=request.asset_uri or resolved_path.as_uri(),
            sha256=content_hash.sha256,
            byte_length=content_hash.byte_length,
            mime_type=request.mime_type,
        )
        duplicate_ids = self._store.find_observation_ids_by_content(
            sha256=content_hash.sha256,
            byte_length=content_hash.byte_length,
        )
        duplicate_ids = _canonical_duplicate_ids(
            duplicate_ids,
            exclude=request.observation_id,
        )
        observation = self._image_ingestor.ingest(
            ImageIngestRequest(
                observation_id=request.observation_id,
                asset=asset,
                source=request.source,
                received_at=request.received_at,
                captured_at=request.captured_at,
            )
        )
        return LocalImageIngestResult(
            observation=observation,
            duplicate_observation_ids=duplicate_ids,
        )


class LocalVideoIngestor:
    """Hash and persist a source video without decoding frames or probing metadata."""

    def __init__(
        self,
        store: _HashingObservationStore,
        *,
        chunk_size: int = DEFAULT_HASH_CHUNK_SIZE,
    ) -> None:
        _validate_chunk_size(chunk_size)
        self._store = store
        self._chunk_size = chunk_size
        self._video_ingestor = VideoIngestor(store)

    def ingest(self, request: LocalVideoIngestRequest) -> LocalVideoIngestResult:
        resolved_path = request.path.expanduser().resolve(strict=True)
        content_hash = hash_file_content(resolved_path, chunk_size=self._chunk_size)
        asset = MediaAssetRef(
            uri=request.asset_uri or resolved_path.as_uri(),
            sha256=content_hash.sha256,
            byte_length=content_hash.byte_length,
            mime_type=request.mime_type,
        )
        duplicate_ids = self._store.find_observation_ids_by_content(
            sha256=content_hash.sha256,
            byte_length=content_hash.byte_length,
        )
        duplicate_ids = _canonical_duplicate_ids(
            duplicate_ids,
            exclude=request.observation_id,
        )
        observation = self._video_ingestor.ingest(
            VideoIngestRequest(
                observation_id=request.observation_id,
                asset=asset,
                source=request.source,
                received_at=request.received_at,
                captured_at=request.captured_at,
            )
        )
        return LocalVideoIngestResult(
            observation=observation,
            duplicate_observation_ids=duplicate_ids,
        )
