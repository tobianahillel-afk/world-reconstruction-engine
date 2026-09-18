"""Deterministic candidate-retrieval policies."""

from wre.retrieval.gps import (
    COLMAP_SPATIAL_REFERENCE_VERSION,
    GPS_DISTANCE_MODEL,
    GPS_MAP_DATUM_POLICY,
    GPS_PAIRING_IMPLEMENTATION,
    GPS_PAIRING_VERSION,
    GpsPairCandidate,
    GpsPairingConfig,
    GpsPairingEligibility,
    GpsPairingEligibilityStatus,
    GpsPairingRequest,
    GpsPairingResult,
    generate_gps_candidates,
)
from wre.retrieval.sequential import (
    COLMAP_SEQUENTIAL_REFERENCE_VERSION,
    SEQUENTIAL_PAIRING_IMPLEMENTATION,
    SEQUENTIAL_PAIRING_VERSION,
    SequentialPairCandidate,
    SequentialPairingConfig,
    SequentialPairingRequest,
    SequentialPairingResult,
    generate_sequential_candidates,
)
from wre.retrieval.sequential_adapter import (
    SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
    SequentialPairCandidateAdapterInput,
    adapt_sequential_pairing_result,
)

__all__ = [
    "COLMAP_SEQUENTIAL_REFERENCE_VERSION",
    "COLMAP_SPATIAL_REFERENCE_VERSION",
    "GPS_DISTANCE_MODEL",
    "GPS_MAP_DATUM_POLICY",
    "GPS_PAIRING_IMPLEMENTATION",
    "GPS_PAIRING_VERSION",
    "SEQUENTIAL_PAIRING_IMPLEMENTATION",
    "SEQUENTIAL_PAIRING_VERSION",
    "SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID",
    "GpsPairCandidate",
    "GpsPairingConfig",
    "GpsPairingEligibility",
    "GpsPairingEligibilityStatus",
    "GpsPairingRequest",
    "GpsPairingResult",
    "SequentialPairCandidate",
    "SequentialPairCandidateAdapterInput",
    "SequentialPairingConfig",
    "SequentialPairingRequest",
    "SequentialPairingResult",
    "adapt_sequential_pairing_result",
    "generate_gps_candidates",
    "generate_sequential_candidates",
]
