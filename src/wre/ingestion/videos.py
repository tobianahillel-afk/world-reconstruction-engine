from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from wre.domain.observations import (
    MediaAssetRef,
    ObservationId,
    SourceRef,
    VideoObservation,
)
from wre.ingestion.images import ObservationSink


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoIngestRequest:
    """Explicit input needed to persist one raw source-video observation.

    The video asset must already be content-addressed. Local-file hashing is
    composed separately so this boundary never decodes video or selects frames.
    """

    observation_id: ObservationId
    asset: MediaAssetRef
    source: SourceRef
    received_at: datetime
    captured_at: datetime | None = None

    def to_observation(self) -> VideoObservation:
        return VideoObservation(
            observation_id=self.observation_id,
            asset=self.asset,
            source=self.source,
            received_at=self.received_at,
            captured_at=self.captured_at,
        )


class VideoIngestor:
    """Persist a raw source video without decoding it or creating frame observations."""

    def __init__(self, sink: ObservationSink) -> None:
        self._sink = sink

    def ingest(self, request: VideoIngestRequest) -> VideoObservation:
        observation = request.to_observation()
        self._sink.put_observation(observation)
        return observation
