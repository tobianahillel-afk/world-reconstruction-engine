from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_opaque_id(value: str, context: str) -> None:
    if not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _require_non_empty_text(value: str, context: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{context} must be non-empty")


def _require_aware_datetime(value: datetime, context: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{context} must be timezone-aware")


class ObservationKind(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    VIDEO_FRAME = "video_frame"


@dataclass(frozen=True, slots=True, order=True)
class ObservationId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "observation_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class SourceId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "source_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class Sha256Digest:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.lower()
        if not _SHA256_RE.fullmatch(normalized):
            raise ValueError("sha256 must be exactly 64 hexadecimal characters")
        object.__setattr__(self, "value", normalized)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class MediaAssetRef:
    uri: str
    sha256: Sha256Digest
    byte_length: int
    mime_type: str | None = None

    def __post_init__(self) -> None:
        _require_non_empty_text(self.uri, "asset.uri")
        if isinstance(self.byte_length, bool) or self.byte_length < 0:
            raise ValueError("asset.byte_length must be a non-negative integer")
        if self.mime_type is not None:
            _require_non_empty_text(self.mime_type, "asset.mime_type")


@dataclass(frozen=True, slots=True)
class SourceRef:
    source_id: SourceId
    locator: str | None = None

    def __post_init__(self) -> None:
        if self.locator is not None:
            _require_non_empty_text(self.locator, "source.locator")


@dataclass(frozen=True, slots=True, kw_only=True)
class Observation(ABC):
    observation_id: ObservationId
    asset: MediaAssetRef
    source: SourceRef
    received_at: datetime
    captured_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_aware_datetime(self.received_at, "received_at")
        if self.captured_at is not None:
            _require_aware_datetime(self.captured_at, "captured_at")

    @property
    @abstractmethod
    def kind(self) -> ObservationKind:
        """Return the stable observation discriminator."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageObservation(Observation):
    @property
    def kind(self) -> ObservationKind:
        return ObservationKind.IMAGE


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoObservation(Observation):
    """Raw source-video observation before any frame/keyframe extraction."""

    @property
    def kind(self) -> ObservationKind:
        return ObservationKind.VIDEO


@dataclass(frozen=True, slots=True, kw_only=True)
class VideoFrameObservation(Observation):
    video_asset: MediaAssetRef
    frame_index: int
    frame_time_us: int

    def __post_init__(self) -> None:
        Observation.__post_init__(self)
        if isinstance(self.frame_index, bool) or self.frame_index < 0:
            raise ValueError("frame_index must be a non-negative integer")
        if isinstance(self.frame_time_us, bool) or self.frame_time_us < 0:
            raise ValueError("frame_time_us must be a non-negative integer")

    @property
    def kind(self) -> ObservationKind:
        return ObservationKind.VIDEO_FRAME
