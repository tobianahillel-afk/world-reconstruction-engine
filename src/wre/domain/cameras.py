from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.observations import ObservationId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _optional_non_blank_text(value: str | None, context: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise ValueError(f"{context} must be non-blank when present")


@dataclass(frozen=True, slots=True, order=True)
class CameraId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "camera_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Camera:
    """Explicit camera identity, independent of any solver camera model."""

    camera_id: CameraId
    manufacturer: str | None = None
    model: str | None = None
    serial_number: str | None = None

    def __post_init__(self) -> None:
        _optional_non_blank_text(self.manufacturer, "camera.manufacturer")
        _optional_non_blank_text(self.model, "camera.model")
        _optional_non_blank_text(self.serial_number, "camera.serial_number")


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width_px: int
    height_px: int

    def __post_init__(self) -> None:
        if isinstance(self.width_px, bool) or not isinstance(self.width_px, int) or self.width_px <= 0:
            raise ValueError("width_px must be a positive integer")
        if (
            isinstance(self.height_px, bool)
            or not isinstance(self.height_px, int)
            or self.height_px <= 0
        ):
            raise ValueError("height_px must be a positive integer")


@dataclass(frozen=True, slots=True)
class RawMetadataEntry:
    """One uninterpreted metadata value preserved exactly as supplied by an ingestor."""

    namespace: str
    key: str
    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("metadata namespace must be non-blank")
        if not isinstance(self.key, str) or not self.key.strip():
            raise ValueError("metadata key must be non-blank")
        if not isinstance(self.value, str):
            raise ValueError("metadata value must be a string")


@dataclass(frozen=True, slots=True)
class ObservationMetadata:
    """Solver-independent metadata associated with exactly one raw observation."""

    observation_id: ObservationId
    camera_id: CameraId | None = None
    dimensions: ImageDimensions | None = None
    raw_entries: tuple[RawMetadataEntry, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.raw_entries, tuple):
            raise ValueError("raw_entries must be an immutable tuple")
        if not all(isinstance(entry, RawMetadataEntry) for entry in self.raw_entries):
            raise ValueError("raw_entries must contain only RawMetadataEntry values")
