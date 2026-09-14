"""Media-ingestion services built on the solver-independent domain contracts."""

from wre.ingestion.exif import (
    EXIF_NAMESPACE,
    ExifMetadataIngestor,
    ExifMetadataIngestRequest,
    ObservationMetadataSink,
    extract_exif_entries,
)
from wre.ingestion.hashing import (
    DEFAULT_HASH_CHUNK_SIZE,
    DuplicateObservationLookup,
    FileContentHash,
    LocalImageIngestor,
    LocalImageIngestRequest,
    LocalImageIngestResult,
    LocalVideoIngestor,
    LocalVideoIngestRequest,
    LocalVideoIngestResult,
    hash_file_content,
)
from wre.ingestion.images import ImageIngestor, ImageIngestRequest, ObservationSink
from wre.ingestion.metadata_interpretation import (
    MetadataInterpretationSink,
    MetadataInterpreter,
    interpret_observation_metadata,
)
from wre.ingestion.videos import VideoIngestor, VideoIngestRequest

__all__ = [
    "DEFAULT_HASH_CHUNK_SIZE",
    "EXIF_NAMESPACE",
    "DuplicateObservationLookup",
    "ExifMetadataIngestRequest",
    "ExifMetadataIngestor",
    "FileContentHash",
    "ImageIngestRequest",
    "ImageIngestor",
    "LocalImageIngestRequest",
    "LocalImageIngestResult",
    "LocalImageIngestor",
    "LocalVideoIngestRequest",
    "LocalVideoIngestResult",
    "LocalVideoIngestor",
    "MetadataInterpretationSink",
    "MetadataInterpreter",
    "ObservationMetadataSink",
    "ObservationSink",
    "VideoIngestRequest",
    "VideoIngestor",
    "extract_exif_entries",
    "hash_file_content",
    "interpret_observation_metadata",
]
