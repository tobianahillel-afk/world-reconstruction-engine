from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    Observation,
    ObservationId,
    SourceRef,
)


class ObservationSink(Protocol):
    """Minimal persistence boundary required by image ingestion."""

    def put_observation(self, observation: Observation) -> None:
        """Persist one raw observation or reject an identity conflict."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageIngestRequest:
    """Deterministic input needed to create one raw image observation.

    The asset reference must already contain its content identity. L2.1 does not
    read media bytes to calculate hashes; that workflow is owned by L2.2.
    """

    observation_id: ObservationId
    asset: MediaAssetRef
    source: SourceRef
    received_at: datetime
    captured_at: datetime | None = None

    def to_observation(self) -> ImageObservation:
        return ImageObservation(
            observation_id=self.observation_id,
            asset=self.asset,
            source=self.source,
            received_at=self.received_at,
            captured_at=self.captured_at,
        )


class ImageIngestor:
    """Create and persist raw image observations without metadata inference."""

    def __init__(self, sink: ObservationSink) -> None:
        self._sink = sink

    def ingest(self, request: ImageIngestRequest) -> ImageObservation:
        observation = request.to_observation()
        self._sink.put_observation(observation)
        return observation
