from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import exifread

from wre.domain.cameras import ObservationMetadata, RawMetadataEntry
from wre.domain.observations import ObservationId

EXIF_NAMESPACE = "exif"
_EXIFREAD_NON_METADATA_KEYS = frozenset({"Filename", "JPEGThumbnail", "TIFFThumbnail"})


class ObservationMetadataSink(Protocol):
    """Minimal persistence boundary required by EXIF metadata ingestion."""

    def put_observation_metadata(self, metadata: ObservationMetadata) -> None:
        """Persist metadata for one raw observation or reject an identity conflict."""
        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class ExifMetadataIngestRequest:
    """Explicit input required to extract raw EXIF metadata for one observation."""

    path: Path
    observation_id: ObservationId


def _render_raw_tag_value(tag: object) -> str:
    """Render parser values without converting them into domain semantics."""

    values = getattr(tag, "values", tag)
    return str(values)


def extract_exif_entries(path: str | Path) -> tuple[RawMetadataEntry, ...]:
    """Extract canonical raw EXIF entries without GPS/time/camera interpretation."""

    resolved_path = Path(path).expanduser().resolve(strict=True)
    with resolved_path.open("rb") as stream:
        tags = exifread.process_file(
            stream,
            details=False,
            extract_thumbnail=False,
            strict=True,
        )

    entries = [
        RawMetadataEntry(
            namespace=EXIF_NAMESPACE,
            key=tag_name,
            value=_render_raw_tag_value(tags[tag_name]),
        )
        for tag_name in sorted(tags)
        if tag_name not in _EXIFREAD_NON_METADATA_KEYS
    ]
    return tuple(entries)


class ExifMetadataIngestor:
    """Extract and persist raw EXIF metadata without semantic interpretation."""

    def __init__(self, sink: ObservationMetadataSink) -> None:
        self._sink = sink

    def ingest(self, request: ExifMetadataIngestRequest) -> ObservationMetadata:
        metadata = ObservationMetadata(
            observation_id=request.observation_id,
            raw_entries=extract_exif_entries(request.path),
        )
        self._sink.put_observation_metadata(metadata)
        return metadata
