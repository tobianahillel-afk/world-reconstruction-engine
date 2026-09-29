from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wre.domain.artifact_materialization import ArtifactMaterializationMetadata
from wre.domain.artifacts import ArtifactRef
from wre.domain.surfaces import (
    SurfaceIntendedUse,
    SurfaceModel,
    SurfaceRepresentationName,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _validate_supporting_artifacts(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("hybrid_surface_candidate.supporting_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("hybrid_surface_candidate.supporting_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("hybrid_surface_candidate.supporting_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "hybrid_surface_candidate.supporting_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )


def _validate_intended_uses(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("hybrid_surface_candidate.intended_uses must be an immutable tuple")
    if not value:
        raise ValueError("hybrid_surface_candidate.intended_uses must be non-empty")
    if any(not isinstance(item, SurfaceIntendedUse) for item in value):
        raise TypeError("hybrid_surface_candidate.intended_uses members must be SurfaceIntendedUse")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("hybrid_surface_candidate.intended_uses must be unique")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "hybrid_surface_candidate.intended_uses must use canonical "
            "SurfaceIntendedUse value order"
        )


def _required_source_artifact_identities(
    request: HybridSurfaceCandidateRequest,
) -> tuple[tuple[str, str], ...]:
    identities = {
        *(_artifact_key(item) for item in request.source_geometry.source_artifacts),
        *(_artifact_key(item) for item in request.supporting_artifacts),
    }
    return tuple(sorted(identities))


@dataclass(frozen=True, slots=True)
class HybridSurfaceCandidateRequest:
    """Pure solver-independent request for one optional physical-surface candidate."""

    source_geometry: GeometrySolutionCandidate
    supporting_artifacts: tuple[ArtifactRef, ...]
    output_representation: SurfaceRepresentationName
    intended_uses: tuple[SurfaceIntendedUse, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError(
                "hybrid_surface_candidate.source_geometry must be GeometrySolutionCandidate"
            )
        _validate_supporting_artifacts(self.supporting_artifacts)
        if not isinstance(self.output_representation, SurfaceRepresentationName):
            raise TypeError(
                "hybrid_surface_candidate.output_representation must be SurfaceRepresentationName"
            )
        _validate_intended_uses(self.intended_uses)


@dataclass(frozen=True, slots=True)
class HybridSurfaceCandidateResult:
    """Validated physical SurfaceModel candidate with exact immutable lineage."""

    request: HybridSurfaceCandidateRequest
    candidate: SurfaceModel
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.request, HybridSurfaceCandidateRequest):
            raise TypeError(
                "hybrid_surface_candidate_result.request must be HybridSurfaceCandidateRequest"
            )
        if not isinstance(self.candidate, SurfaceModel):
            raise TypeError("hybrid_surface_candidate_result.candidate must be SurfaceModel")
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "hybrid_surface_candidate_result.materialization must be "
                "ArtifactMaterializationMetadata"
            )

        source_geometry = self.request.source_geometry
        source_solution = source_geometry.geometry_solution

        if self.candidate.source_geometry != source_geometry:
            raise ValueError(
                "hybrid surface candidate must preserve the exact source GeometrySolutionCandidate"
            )
        if self.candidate.local_frame_id != source_solution.local_frame_id:
            raise ValueError(
                "hybrid surface candidate must preserve source geometry LocalFrameId exactly"
            )
        if self.candidate.scale_status != source_solution.scale_status:
            raise ValueError(
                "hybrid surface candidate must preserve source geometry scale status exactly"
            )
        if self.candidate.representation != self.request.output_representation:
            raise ValueError(
                "hybrid surface candidate must preserve the declared SurfaceRepresentationName"
            )
        if self.candidate.intended_uses != self.request.intended_uses:
            raise ValueError(
                "hybrid surface candidate must preserve the declared intended uses exactly"
            )

        output_key = _artifact_key(self.candidate.artifact_ref)
        source_keys = {
            *(_artifact_key(item) for item in source_geometry.source_artifacts),
            *(_artifact_key(item) for item in self.request.supporting_artifacts),
        }
        if output_key in source_keys:
            raise ValueError(
                "hybrid surface candidate output ArtifactRef must be distinct from all inputs"
            )

        actual_ancestry = tuple(_artifact_key(item) for item in self.candidate.source_artifacts)
        required_ancestry = _required_source_artifact_identities(self.request)
        if actual_ancestry != required_ancestry:
            raise ValueError(
                "hybrid surface candidate source_artifacts must equal the canonical union "
                "of source geometry ancestry and supporting artifacts"
            )

        if self.materialization.artifact_ref != self.candidate.artifact_ref:
            raise ValueError(
                "hybrid surface candidate materialization ArtifactRef must exactly match "
                "candidate.artifact_ref"
            )


class HybridSurfaceCandidateAdapter(Protocol):
    """Solver-independent boundary implemented by later optional hybrid specialists."""

    def derive_candidate(
        self,
        request: HybridSurfaceCandidateRequest,
    ) -> HybridSurfaceCandidateResult:
        """Derive one explicit physical SurfaceModel candidate."""
        ...


def build_hybrid_surface_candidate_result(
    request: HybridSurfaceCandidateRequest,
    candidate: SurfaceModel,
    materialization: ArtifactMaterializationMetadata,
) -> HybridSurfaceCandidateResult:
    """Construct and validate one hybrid-surface result without external work."""

    return HybridSurfaceCandidateResult(
        request=request,
        candidate=candidate,
        materialization=materialization,
    )


__all__ = [
    "HybridSurfaceCandidateAdapter",
    "HybridSurfaceCandidateRequest",
    "HybridSurfaceCandidateResult",
    "build_hybrid_surface_candidate_result",
]
