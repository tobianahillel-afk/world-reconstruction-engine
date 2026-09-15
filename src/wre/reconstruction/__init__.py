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
from wre.reconstruction.colmap_matching import (
    ColmapPairMatchingConfig,
    ColmapPairMatchingError,
    ColmapPairMatchingRequest,
    ColmapPairMatchingResult,
    ColmapPairMatchSummary,
    match_colmap_pairs,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionError,
    ColmapIncrementalReconstructionRequest,
    ColmapIncrementalReconstructionResult,
    ColmapModelFileArtifact,
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
    reconstruct_colmap_incrementally,
)
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationError,
    ColmapGeometricVerificationRequest,
    ColmapGeometricVerificationResult,
    ColmapPairGeometryEvidence,
    verify_colmap_geometry,
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
    "ColmapGeometricVerificationConfig",
    "ColmapGeometricVerificationError",
    "ColmapGeometricVerificationRequest",
    "ColmapGeometricVerificationResult",
    "ColmapImageFeatureSummary",
    "ColmapIncrementalReconstructionConfig",
    "ColmapIncrementalReconstructionError",
    "ColmapIncrementalReconstructionRequest",
    "ColmapIncrementalReconstructionResult",
    "ColmapModelFileArtifact",
    "ColmapPairGeometryEvidence",
    "ColmapPairMatchSummary",
    "ColmapPairMatchingConfig",
    "ColmapPairMatchingError",
    "ColmapPairMatchingRequest",
    "ColmapPairMatchingResult",
    "ColmapReconstructionInput",
    "ColmapSparseModelArtifact",
    "extract_colmap_features",
    "inspect_colmap_environment",
    "match_colmap_pairs",
    "reconstruct_colmap_incrementally",
    "verify_colmap_geometry",
]
