from __future__ import annotations

from enum import StrEnum


class ProvenanceClass(StrEnum):
    """Epistemic classification of an artifact's content."""

    OBSERVED_RECONSTRUCTED = "observed_reconstructed"
    INFERRED = "inferred"
    GENERATED = "generated"
