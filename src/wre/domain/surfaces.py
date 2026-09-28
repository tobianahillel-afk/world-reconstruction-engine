from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import GeometryScaleStatus
from wre.domain.producer_identity import ArtifactProducerIdentity

if TYPE_CHECKING:
    from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_LOWER_TOKEN_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")

SURFACE_MODEL_ARTIFACT_KIND = ArtifactKind("geometry.surface")


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


@dataclass(frozen=True, slots=True, order=True)
class SurfaceRepresentationName:
    """Open solver-independent token describing one retained physical-surface representation."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _LOWER_TOKEN_RE.fullmatch(self.value) is None:
            raise ValueError(
                "surface_representation must be a 1-128 character lowercase token using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


class SurfaceIntendedUse(StrEnum):
    """Declared downstream intent only; not proof of suitability."""

    COLLISION = "collision"
    MEASUREMENT = "measurement"
    NAVIGATION = "navigation"


def _validate_intended_uses(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("surface_model.intended_uses must be an immutable tuple")
    if not value:
        raise ValueError("surface_model.intended_uses must be non-empty")
    if any(not isinstance(item, SurfaceIntendedUse) for item in value):
        raise TypeError("surface_model.intended_uses members must be SurfaceIntendedUse")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("surface_model.intended_uses must be unique")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "surface_model.intended_uses must use canonical SurfaceIntendedUse value order"
        )


def _validate_source_artifacts(
    artifact_ref: ArtifactRef,
    source_geometry: GeometrySolutionCandidate,
    source_artifacts: object,
) -> None:
    if not isinstance(source_artifacts, tuple):
        raise TypeError("surface_model.source_artifacts must be an immutable tuple")
    if not source_artifacts:
        raise ValueError("surface_model.source_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in source_artifacts):
        raise TypeError("surface_model.source_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in source_artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError("surface_model.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "surface_model.source_artifacts must use canonical ArtifactId/ArtifactKind order"
        )

    source_geometry_identities = {_artifact_key(item) for item in source_geometry.source_artifacts}
    if not source_geometry_identities.issubset(set(identities)):
        raise ValueError(
            "surface_model.source_artifacts must retain every source GeometrySolutionCandidate "
            "artifact"
        )
    if _artifact_key(artifact_ref) in set(identities):
        raise ValueError("surface_model.artifact_ref must not reuse a source artifact identity")


@dataclass(frozen=True, slots=True)
class SurfaceModel:
    """Immutable metadata/provenance envelope for one explicit physical-surface artifact.

    The contract records what representation was retained and what downstream uses it was
    built for. It does not generate a surface, inspect payload bytes, prove suitability,
    fill unsupported regions, create runtime collision/navigation assets, or infer scale.
    """

    artifact_ref: ArtifactRef
    source_geometry: GeometrySolutionCandidate
    representation: SurfaceRepresentationName
    intended_uses: tuple[SurfaceIntendedUse, ...]
    local_frame_id: LocalFrameId
    scale_status: GeometryScaleStatus
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("surface_model.artifact_ref must be ArtifactRef")
        if self.artifact_ref.artifact_kind != SURFACE_MODEL_ARTIFACT_KIND:
            raise ValueError("surface_model.artifact_ref must use artifact kind geometry.surface")
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError("surface_model.source_geometry must be GeometrySolutionCandidate")
        if not isinstance(self.representation, SurfaceRepresentationName):
            raise TypeError("surface_model.representation must be SurfaceRepresentationName")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("surface_model.local_frame_id must be LocalFrameId")
        if not isinstance(self.scale_status, GeometryScaleStatus):
            raise TypeError("surface_model.scale_status must be GeometryScaleStatus")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("surface_model.producer must be ArtifactProducerIdentity")

        _validate_intended_uses(self.intended_uses)

        source_solution = self.source_geometry.geometry_solution
        if self.local_frame_id != source_solution.local_frame_id:
            raise ValueError(
                "surface_model.local_frame_id must exactly preserve source geometry LocalFrameId"
            )
        if self.scale_status != source_solution.scale_status:
            raise ValueError(
                "surface_model.scale_status must exactly preserve source geometry scale status"
            )

        _validate_source_artifacts(
            self.artifact_ref,
            self.source_geometry,
            self.source_artifacts,
        )
