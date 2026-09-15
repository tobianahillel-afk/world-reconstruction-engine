"""Deterministic candidate-retrieval policies."""

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

__all__ = [
    "COLMAP_SEQUENTIAL_REFERENCE_VERSION",
    "SEQUENTIAL_PAIRING_IMPLEMENTATION",
    "SEQUENTIAL_PAIRING_VERSION",
    "SequentialPairCandidate",
    "SequentialPairingConfig",
    "SequentialPairingRequest",
    "SequentialPairingResult",
    "generate_sequential_candidates",
]
