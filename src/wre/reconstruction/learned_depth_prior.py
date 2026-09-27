from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wre.domain.artifacts import ArtifactRef
from wre.domain.depth_fields import DepthValueConventionName
from wre.reconstruction.dense_depth import DenseDepthArtifact
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _validate_supporting_artifacts(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("learned_depth_prior.supporting_artifacts must be an immutable tuple")
    if not value:
        raise ValueError("learned_depth_prior.supporting_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("learned_depth_prior.supporting_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("learned_depth_prior.supporting_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "learned_depth_prior.supporting_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )


def _required_source_artifact_identities(
    request: LearnedDepthPriorRequest,
) -> tuple[tuple[str, str], ...]:
    identities = {
        *(_artifact_key(item) for item in request.source_geometry.source_artifacts),
        *(_artifact_key(item) for item in request.supporting_artifacts),
    }
    return tuple(sorted(identities))


@dataclass(frozen=True, slots=True)
class LearnedDepthPriorRequest:
    """Pure immutable evidence boundary for one learned depth-prior adapter.

    The request declares only canonical source geometry, explicit immutable input
    evidence, and the semantic depth-value convention the adapter claims to emit.
    It selects, loads, executes, converts, calibrates, scores, or fuses no model.
    """

    source_geometry: GeometrySolutionCandidate
    supporting_artifacts: tuple[ArtifactRef, ...]
    output_depth_value_convention: DepthValueConventionName

    def __post_init__(self) -> None:
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError("learned_depth_prior.source_geometry must be GeometrySolutionCandidate")
        _validate_supporting_artifacts(self.supporting_artifacts)
        if not isinstance(self.output_depth_value_convention, DepthValueConventionName):
            raise TypeError(
                "learned_depth_prior.output_depth_value_convention must be DepthValueConventionName"
            )


@dataclass(frozen=True, slots=True)
class LearnedDepthPriorResult:
    """Validated canonical learned-depth evidence without executing a learned model."""

    request: LearnedDepthPriorRequest
    depth_prior: DenseDepthArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.request, LearnedDepthPriorRequest):
            raise TypeError("learned_depth_prior_result.request must be LearnedDepthPriorRequest")
        if not isinstance(self.depth_prior, DenseDepthArtifact):
            raise TypeError("learned_depth_prior_result.depth_prior must be DenseDepthArtifact")

        if self.depth_prior.source_geometry != self.request.source_geometry:
            raise ValueError(
                "learned depth prior must preserve the exact source GeometrySolutionCandidate"
            )

        supporting_identities = {_artifact_key(item) for item in self.request.supporting_artifacts}
        if _artifact_key(self.depth_prior.artifact_ref) in supporting_identities:
            raise ValueError("learned depth prior output must not reuse a supporting ArtifactRef")

        for depth_field in self.depth_prior.depth_fields:
            if depth_field.depth_value_convention != self.request.output_depth_value_convention:
                raise ValueError(
                    "learned depth prior DepthFields must preserve the declared "
                    "DepthValueConventionName"
                )

        actual_ancestry = tuple(_artifact_key(item) for item in self.depth_prior.source_artifacts)
        required_ancestry = _required_source_artifact_identities(self.request)
        if actual_ancestry != required_ancestry:
            raise ValueError(
                "learned depth prior source_artifacts must equal the canonical union "
                "of source geometry ancestry and supporting artifacts"
            )


class LearnedDepthPriorAdapter(Protocol):
    """Model-independent learned depth-prior boundary."""

    def infer(self, request: LearnedDepthPriorRequest) -> LearnedDepthPriorResult:
        """Produce canonical learned-depth evidence under adapter-owned semantics."""
        ...


def build_learned_depth_prior_result(
    request: LearnedDepthPriorRequest,
    depth_prior: DenseDepthArtifact,
) -> LearnedDepthPriorResult:
    """Construct and validate one learned-depth result without external work."""

    return LearnedDepthPriorResult(
        request=request,
        depth_prior=depth_prior,
    )


__all__ = [
    "LearnedDepthPriorAdapter",
    "LearnedDepthPriorRequest",
    "LearnedDepthPriorResult",
    "build_learned_depth_prior_result",
]
