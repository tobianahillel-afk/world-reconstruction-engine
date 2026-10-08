from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wre.domain.appearance import AppearanceModel, AppearanceRepresentationName
from wre.domain.artifact_materialization import ArtifactMaterializationMetadata
from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId
from wre.domain.surfaces import SurfaceModel
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _validate_supporting_artifacts(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("static_appearance_candidate.supporting_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("static_appearance_candidate.supporting_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("static_appearance_candidate.supporting_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "static_appearance_candidate.supporting_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )


def _validate_observation_ids(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError(
            "static_appearance_candidate.source_observation_ids must be an immutable tuple"
        )
    if not value:
        raise ValueError("static_appearance_candidate.source_observation_ids must be non-empty")
    if any(not isinstance(item, ObservationId) for item in value):
        raise TypeError("static_appearance_candidate.source_observation_ids members must be ObservationId")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("static_appearance_candidate.source_observation_ids must be unique")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "static_appearance_candidate.source_observation_ids must use canonical "
            "ObservationId value order"
        )


def _required_ancestry(request: StaticAppearanceCandidateRequest) -> tuple[tuple[str, str], ...]:
    identities = {_artifact_key(item) for item in request.source_geometry.source_artifacts}
    if request.source_surface is not None:
        identities.add(_artifact_key(request.source_surface.artifact_ref))
        identities.update(_artifact_key(item) for item in request.source_surface.source_artifacts)
    identities.update(_artifact_key(item) for item in request.supporting_artifacts)
    return tuple(sorted(identities))


@dataclass(frozen=True, slots=True)
class StaticAppearanceCandidateRequest:
    """One solver-independent static appearance request, without rendering or training."""

    source_geometry: GeometrySolutionCandidate
    source_surface: SurfaceModel | None
    source_observation_ids: tuple[ObservationId, ...]
    output_representation: AppearanceRepresentationName
    supporting_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError(
                "static_appearance_candidate.source_geometry must be GeometrySolutionCandidate"
            )
        if self.source_surface is not None:
            if not isinstance(self.source_surface, SurfaceModel):
                raise TypeError(
                    "static_appearance_candidate.source_surface must be SurfaceModel or None"
                )
            if self.source_surface.source_geometry != self.source_geometry:
                raise ValueError(
                    "static appearance source surface must preserve the exact source "
                    "GeometrySolutionCandidate"
                )
            geometry = self.source_geometry.geometry_solution
            if self.source_surface.local_frame_id != geometry.local_frame_id:
                raise ValueError("static appearance source surface must preserve LocalFrameId")
            if self.source_surface.scale_status != geometry.scale_status:
                raise ValueError("static appearance source surface must preserve scale status")

        _validate_observation_ids(self.source_observation_ids)
        if not isinstance(self.output_representation, AppearanceRepresentationName):
            raise TypeError(
                "static_appearance_candidate.output_representation must be "
                "AppearanceRepresentationName"
            )
        _validate_supporting_artifacts(self.supporting_artifacts)


@dataclass(frozen=True, slots=True)
class StaticAppearanceCandidateResult:
    """One complete appearance envelope and an exact, uninspected output-file manifest."""

    request: StaticAppearanceCandidateRequest
    candidate: AppearanceModel
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.request, StaticAppearanceCandidateRequest):
            raise TypeError(
                "static_appearance_candidate_result.request must be StaticAppearanceCandidateRequest"
            )
        if not isinstance(self.candidate, AppearanceModel):
            raise TypeError(
                "static_appearance_candidate_result.candidate must be AppearanceModel"
            )
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "static_appearance_candidate_result.materialization must be "
                "ArtifactMaterializationMetadata"
            )

        request = self.request
        geometry = request.source_geometry.geometry_solution

        if self.candidate.source_geometry != request.source_geometry:
            raise ValueError(
                "static appearance candidate must preserve the exact source GeometrySolutionCandidate"
            )
        if self.candidate.source_surface != request.source_surface:
            raise ValueError("static appearance candidate must preserve the exact optional SurfaceModel")
        if self.candidate.source_observation_ids != request.source_observation_ids:
            raise ValueError("static appearance candidate must preserve source ObservationIds exactly")
        if self.candidate.representation != request.output_representation:
            raise ValueError(
                "static appearance candidate must preserve the declared AppearanceRepresentationName"
            )
        if self.candidate.local_frame_id != geometry.local_frame_id:
            raise ValueError("static appearance candidate must preserve source LocalFrameId")
        if self.candidate.scale_status != geometry.scale_status:
            raise ValueError("static appearance candidate must preserve source scale status")

        expected = _required_ancestry(request)
        actual = tuple(_artifact_key(item) for item in self.candidate.source_artifacts)
        if actual != expected:
            raise ValueError(
                "static appearance candidate source_artifacts must equal the canonical union "
                "of geometry, optional surface and supporting artifacts"
            )
        if _artifact_key(self.candidate.artifact_ref) in set(expected):
            raise ValueError(
                "static appearance candidate output ArtifactRef must be distinct from all inputs"
            )
        if self.materialization.artifact_ref != self.candidate.artifact_ref:
            raise ValueError(
                "static appearance materialization ArtifactRef must exactly match "
                "candidate.artifact_ref"
            )


class StaticAppearanceCandidateAdapter(Protocol):
    """A future specialist implements this boundary without changing WRE's appearance truth."""

    def derive_candidate(
        self, request: StaticAppearanceCandidateRequest
    ) -> StaticAppearanceCandidateResult:
        """Return one retained canonical appearance candidate."""
        ...


def build_static_appearance_candidate_result(
    request: StaticAppearanceCandidateRequest,
    candidate: AppearanceModel,
    materialization: ArtifactMaterializationMetadata,
) -> StaticAppearanceCandidateResult:
    """Validate a candidate and its exact file identity without inspecting payload bytes."""

    return StaticAppearanceCandidateResult(
        request=request, candidate=candidate, materialization=materialization
    )


__all__ = [
    "StaticAppearanceCandidateAdapter",
    "StaticAppearanceCandidateRequest",
    "StaticAppearanceCandidateResult",
    "build_static_appearance_candidate_result",
]
