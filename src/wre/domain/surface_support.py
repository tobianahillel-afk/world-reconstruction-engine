from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.fragments import LocalFrameId
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.surfaces import SurfaceModel

SURFACE_SUPPORT_MAP_ARTIFACT_KIND = ArtifactKind("geometry.surface_support")


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


class SurfaceRegionSupportStatus(StrEnum):
    """Explicit negative physical-surface support state for one caller-supplied region."""

    UNSUPPORTED = "unsupported"
    HOLE = "hole"


@dataclass(frozen=True, slots=True)
class SurfaceSupportRegion:
    """One explicit region selector and its negative physical-surface support state."""

    selector_artifact: ArtifactRef
    status: SurfaceRegionSupportStatus

    def __post_init__(self) -> None:
        if not isinstance(self.selector_artifact, ArtifactRef):
            raise TypeError("surface_support_region.selector_artifact must be ArtifactRef")
        if not isinstance(self.status, SurfaceRegionSupportStatus):
            raise TypeError("surface_support_region.status must be SurfaceRegionSupportStatus")


def _validate_regions(
    artifact_ref: ArtifactRef,
    surface_model: SurfaceModel,
    regions: object,
) -> None:
    if not isinstance(regions, tuple):
        raise TypeError("surface_support_map.regions must be an immutable tuple")
    if not regions:
        raise ValueError("surface_support_map.regions must be non-empty")
    if any(not isinstance(region, SurfaceSupportRegion) for region in regions):
        raise TypeError("surface_support_map.regions members must be SurfaceSupportRegion")

    selector_keys = tuple(_artifact_key(region.selector_artifact) for region in regions)
    if len(selector_keys) != len(set(selector_keys)):
        raise ValueError("surface_support_map.regions selector ArtifactRefs must be unique")
    if selector_keys != tuple(sorted(selector_keys)):
        raise ValueError(
            "surface_support_map.regions must use canonical selector ArtifactId/ArtifactKind order"
        )

    reserved_keys = {
        _artifact_key(artifact_ref),
        _artifact_key(surface_model.artifact_ref),
    }
    if any(key in reserved_keys for key in selector_keys):
        raise ValueError(
            "surface_support_map selector ArtifactRefs must be distinct from support-map "
            "and SurfaceModel ArtifactRefs"
        )


def _validate_source_artifacts(
    artifact_ref: ArtifactRef,
    surface_model: SurfaceModel,
    regions: tuple[SurfaceSupportRegion, ...],
    source_artifacts: object,
) -> None:
    if not isinstance(source_artifacts, tuple):
        raise TypeError("surface_support_map.source_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in source_artifacts):
        raise TypeError("surface_support_map.source_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in source_artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError("surface_support_map.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "surface_support_map.source_artifacts must use canonical ArtifactId/ArtifactKind order"
        )

    required = {
        _artifact_key(surface_model.artifact_ref),
        *(_artifact_key(region.selector_artifact) for region in regions),
    }
    expected = tuple(sorted(required))
    if identities != expected:
        raise ValueError(
            "surface_support_map.source_artifacts must equal the canonical union of "
            "SurfaceModel and selector ArtifactRefs"
        )

    if _artifact_key(artifact_ref) in required:
        raise ValueError(
            "surface_support_map.artifact_ref must be distinct from every source ArtifactRef"
        )


@dataclass(frozen=True, slots=True)
class SurfaceSupportMap:
    """Immutable negative support annotation over one exact physical SurfaceModel.

    Region entries record only known negative support evidence. Missing entries remain
    unknown and make no positive completeness, safety, suitability or reconstruction claim.
    """

    artifact_ref: ArtifactRef
    surface_model: SurfaceModel
    local_frame_id: LocalFrameId
    regions: tuple[SurfaceSupportRegion, ...]
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("surface_support_map.artifact_ref must be ArtifactRef")
        if self.artifact_ref.artifact_kind != SURFACE_SUPPORT_MAP_ARTIFACT_KIND:
            raise ValueError(
                "surface_support_map.artifact_ref must use artifact kind geometry.surface_support"
            )
        if not isinstance(self.surface_model, SurfaceModel):
            raise TypeError("surface_support_map.surface_model must be SurfaceModel")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("surface_support_map.local_frame_id must be LocalFrameId")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("surface_support_map.producer must be ArtifactProducerIdentity")

        if self.local_frame_id != self.surface_model.local_frame_id:
            raise ValueError(
                "surface_support_map.local_frame_id must exactly preserve SurfaceModel LocalFrameId"
            )

        _validate_regions(
            self.artifact_ref,
            self.surface_model,
            self.regions,
        )
        _validate_source_artifacts(
            self.artifact_ref,
            self.surface_model,
            self.regions,
            self.source_artifacts,
        )


__all__ = [
    "SURFACE_SUPPORT_MAP_ARTIFACT_KIND",
    "SurfaceRegionSupportStatus",
    "SurfaceSupportMap",
    "SurfaceSupportRegion",
]
