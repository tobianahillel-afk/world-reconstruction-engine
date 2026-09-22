from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wre.reconstruction.colmap_canonical_geometry import (
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    CanonicalColmapSparseModel,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_global_reconstruction import (
    COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
)
from wre.reconstruction.colmap_reconstruction import ColmapSparseModelArtifact

_ALLOWED_SOURCE_ADAPTER_IDS = frozenset(
    {
        COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
    }
)


class ColmapGeometryOutcomeState(StrEnum):
    """Exact cardinality-derived outcome states for current COLMAP precision geometry."""

    NO_MODEL = "no_model"
    SINGLE_MODEL = "single_model"
    DISCONNECTED_MODELS = "disconnected_models"


@dataclass(frozen=True, slots=True)
class ColmapGeometryOutcome:
    """Validated mapper outcome preserving every audited canonical local frame independently."""

    source_adapter_id: str
    native_models: tuple[ColmapSparseModelArtifact, ...]
    canonical_models: tuple[CanonicalColmapSparseModel, ...]

    def __post_init__(self) -> None:
        if self.source_adapter_id not in _ALLOWED_SOURCE_ADAPTER_IDS:
            raise ValueError(
                "source_adapter_id must identify the current incremental or global "
                "COLMAP precision adapter"
            )

        if not isinstance(self.native_models, tuple):
            raise TypeError("native_models must be an immutable tuple")
        if any(not isinstance(model, ColmapSparseModelArtifact) for model in self.native_models):
            raise TypeError("native_models members must be ColmapSparseModelArtifact")
        native_indices = tuple(model.model_index for model in self.native_models)
        if native_indices != tuple(sorted(native_indices)):
            raise ValueError("native_models must already be canonically ordered by model index")
        if len(native_indices) != len(set(native_indices)):
            raise ValueError("native_models must have unique model indices")

        if not isinstance(self.canonical_models, tuple):
            raise TypeError("canonical_models must be an immutable tuple")
        if any(
            not isinstance(model, CanonicalColmapSparseModel) for model in self.canonical_models
        ):
            raise TypeError("canonical_models members must be CanonicalColmapSparseModel")

        if len(self.canonical_models) != len(self.native_models):
            raise ValueError(
                "canonical_models must exactly match the audited native-model cardinality"
            )

        canonical_indices = tuple(model.source_model_index for model in self.canonical_models)
        if canonical_indices != native_indices:
            raise ValueError(
                "canonical_models must preserve exact audited native model-index order"
            )

        expected_identities = tuple(
            colmap_sparse_model_content_identity(model) for model in self.native_models
        )
        canonical_identities = tuple(
            model.source_model_identity_sha256 for model in self.canonical_models
        )
        if canonical_identities != expected_identities:
            raise ValueError(
                "canonical_models must preserve exact audited native model content identities"
            )

    @property
    def state(self) -> ColmapGeometryOutcomeState:
        model_count = len(self.native_models)
        if model_count == 0:
            return ColmapGeometryOutcomeState.NO_MODEL
        if model_count == 1:
            return ColmapGeometryOutcomeState.SINGLE_MODEL
        return ColmapGeometryOutcomeState.DISCONNECTED_MODELS

    @property
    def model_count(self) -> int:
        return len(self.native_models)

    @property
    def has_geometry(self) -> bool:
        return bool(self.native_models)
