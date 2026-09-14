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
    hash_file_content,
)
from wre.ingestion.images import ImageIngestor, ImageIngestRequest, ObservationSink
from wre.ingestion.metadata_interpretation import (
    MetadataInterpretationSink,
    MetadataInterpreter,
    interpret_observation_metadata,
)

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
    "MetadataInterpretationSink",
    "MetadataInterpreter",
    "ObservationMetadataSink",
    "ObservationSink",
    "extract_exif_entries",
    "hash_file_content",
    "interpret_observation_metadata",
]
