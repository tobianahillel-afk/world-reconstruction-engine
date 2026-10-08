from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import GeometryScaleStatus
from wre.domain.observations import ObservationId
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.surfaces import SurfaceModel

if TYPE_CHECKING:
    from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_LOWER_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")

APPEARANCE_MODEL_ARTIFACT_KIND = ArtifactKind("appearance.static")


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


@dataclass(frozen=True, slots=True, order=True)
class AppearanceRepresentationName:
    """Open solver-independent token naming a retained static visual representation."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _LOWER_TOKEN_RE.fullmatch(self.value) is None:
            raise ValueError(
                "appearance_representation must be a 1-128 character lowercase token using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


def _validate_observations(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("appearance_model.source_observation_ids must be an immutable tuple")
    if not value:
        raise ValueError("appearance_model.source_observation_ids must be non-empty")
    if any(not isinstance(item, ObservationId) for item in value):
        raise TypeError("appearance_model.source_observation_ids members must be ObservationId")
    identities = tuple(item.value for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("appearance_model.source_observation_ids must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "appearance_model.source_observation_ids must use canonical ObservationId value order"
        )


def _validate_source_artifacts(
    artifact_ref: ArtifactRef,
    source_geometry: GeometrySolutionCandidate,
    source_surface: SurfaceModel | None,
    source_artifacts: object,
) -> None:
    if not isinstance(source_artifacts, tuple):
        raise TypeError("appearance_model.source_artifacts must be an immutable tuple")
    if not source_artifacts:
        raise ValueError("appearance_model.source_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in source_artifacts):
        raise TypeError("appearance_model.source_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in source_artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError("appearance_model.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "appearance_model.source_artifacts must use canonical ArtifactId/ArtifactKind order"
        )

    provided = set(identities)
    required = {_artifact_key(item) for item in source_geometry.source_artifacts}
    if source_surface is not None:
        required.add(_artifact_key(source_surface.artifact_ref))
        required.update(_artifact_key(item) for item in source_surface.source_artifacts)
    if not required.issubset(provided):
        raise ValueError(
            "appearance_model.source_artifacts must retain all geometry and optional surface "
            "artifact ancestry"
        )
    if _artifact_key(artifact_ref) in provided:
        raise ValueError("appearance_model.artifact_ref must not reuse a source artifact identity")


@dataclass(frozen=True, slots=True)
class AppearanceModel:
    """Immutable provenance envelope for one retained reconstructed static visual artifact.

    Appearance is visual evidence, not collision, measurement or navigation geometry,
    physical-surface truth, materials, environment or generated-completion truth.
    Solver-private payloads and rendering behavior remain owned by later adapters.
    """

    artifact_ref: ArtifactRef
    source_geometry: GeometrySolutionCandidate
    source_surface: SurfaceModel | None
    source_observation_ids: tuple[ObservationId, ...]
    representation: AppearanceRepresentationName
    local_frame_id: LocalFrameId
    scale_status: GeometryScaleStatus
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("appearance_model.artifact_ref must be ArtifactRef")
        if self.artifact_ref.artifact_kind != APPEARANCE_MODEL_ARTIFACT_KIND:
            raise ValueError("appearance_model.artifact_ref must use kind appearance.static")
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError("appearance_model.source_geometry must be GeometrySolutionCandidate")
        if self.source_surface is not None and not isinstance(self.source_surface, SurfaceModel):
            raise TypeError("appearance_model.source_surface must be SurfaceModel or None")
        if not isinstance(self.representation, AppearanceRepresentationName):
            raise TypeError("appearance_model.representation must be AppearanceRepresentationName")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("appearance_model.local_frame_id must be LocalFrameId")
        if not isinstance(self.scale_status, GeometryScaleStatus):
            raise TypeError("appearance_model.scale_status must be GeometryScaleStatus")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("appearance_model.producer must be ArtifactProducerIdentity")

        _validate_observations(self.source_observation_ids)

        geometry = self.source_geometry.geometry_solution
        if self.local_frame_id != geometry.local_frame_id:
            raise ValueError(
                "appearance_model.local_frame_id must exactly preserve source geometry LocalFrameId"
            )
        if self.scale_status != geometry.scale_status:
            raise ValueError(
                "appearance_model.scale_status must exactly preserve source geometry scale status"
            )

        if self.source_surface is not None:
            if self.source_surface.source_geometry != self.source_geometry:
                raise ValueError(
                    "appearance_model.source_surface must reference the exact source geometry"
                )
            if self.source_surface.local_frame_id != self.local_frame_id:
                raise ValueError(
                    "appearance_model.source_surface must preserve source geometry LocalFrameId"
                )
            if self.source_surface.scale_status != self.scale_status:
                raise ValueError(
                    "appearance_model.source_surface must preserve source geometry scale status"
                )

        _validate_source_artifacts(
            self.artifact_ref,
            self.source_geometry,
            self.source_surface,
            self.source_artifacts,
        )
