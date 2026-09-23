from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wre.domain.artifacts import ArtifactRef
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _validate_supporting_artifacts(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("geometry_refinement.supporting_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("geometry_refinement.supporting_artifacts members must be ArtifactRef")

    identities = tuple((item.artifact_id.value, item.artifact_kind.value) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("geometry_refinement.supporting_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "geometry_refinement.supporting_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )


@dataclass(frozen=True, slots=True)
class GeometryRefinementRequest:
    """Pure immutable input to one geometry-refinement adapter execution.

    The initialization remains a complete canonical hypothesis. Supporting artifacts
    are additional evidence only and do not replace the initialization's provenance.
    """

    initialization: GeometrySolutionCandidate
    supporting_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.initialization, GeometrySolutionCandidate):
            raise TypeError("geometry_refinement.initialization must be GeometrySolutionCandidate")
        _validate_supporting_artifacts(self.supporting_artifacts)


@dataclass(frozen=True, slots=True)
class GeometryRefinementResult:
    """One refined hypothesis with exact immutable initialization lineage.

    This value makes no claim that initialization and refined output share a coordinate
    gauge, scale, observation set, projection family, child responsibility or quality.
    """

    request: GeometryRefinementRequest
    refined_candidate: GeometrySolutionCandidate

    def __post_init__(self) -> None:
        if not isinstance(self.request, GeometryRefinementRequest):
            raise TypeError("geometry_refinement_result.request must be GeometryRefinementRequest")
        if not isinstance(self.refined_candidate, GeometrySolutionCandidate):
            raise TypeError(
                "geometry_refinement_result.refined_candidate must be GeometrySolutionCandidate"
            )

        initialization_id = self.request.initialization.geometry_solution_id
        refined_id = self.refined_candidate.geometry_solution_id
        if refined_id == initialization_id:
            raise ValueError("geometry refinement output must use a distinct GeometrySolutionId")


class GeometryRefinementAdapter(Protocol):
    """Solver-independent refinement boundary implemented by later concrete adapters."""

    def refine(self, request: GeometryRefinementRequest) -> GeometryRefinementResult:
        """Refine exactly one explicit initialization under adapter-owned semantics."""
        ...


def build_geometry_refinement_result(
    request: GeometryRefinementRequest,
    refined_candidate: GeometrySolutionCandidate,
) -> GeometryRefinementResult:
    """Construct and validate one immutable refinement result without external work."""

    return GeometryRefinementResult(
        request=request,
        refined_candidate=refined_candidate,
    )
