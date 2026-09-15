"""External reconstruction-engine adapters and environment contracts."""

from wre.reconstruction.colmap_environment import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionError,
    ColmapFeatureExtractionRequest,
    ColmapFeatureExtractionResult,
    ColmapFeatureInput,
    ColmapImageFeatureSummary,
    extract_colmap_features,
)

__all__ = [
    "SUPPORTED_COLMAP_VERSION",
    "SUPPORTED_PYCOLMAP_VERSION",
    "ColmapEnvironmentError",
    "ColmapEnvironmentIdentity",
    "ColmapFeatureExtractionConfig",
    "ColmapFeatureExtractionError",
    "ColmapFeatureExtractionRequest",
    "ColmapFeatureExtractionResult",
    "ColmapFeatureInput",
    "ColmapImageFeatureSummary",
    "extract_colmap_features",
    "inspect_colmap_environment",
]
