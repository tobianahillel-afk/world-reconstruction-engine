from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.depth_fields import DepthField
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

DENSE_DEPTH_ARTIFACT_KIND = ArtifactKind("geometry.dense_depth")


def _validate_depth_fields(
    source_geometry: GeometrySolutionCandidate,
    value: object,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError("dense_depth.depth_fields must be an immutable tuple")
    if not value:
        raise ValueError("dense_depth.depth_fields must be non-empty")
    if any(not isinstance(item, DepthField) for item in value):
        raise TypeError("dense_depth.depth_fields members must be DepthField")

    depth_ids = tuple(item.depth_field_id.value for item in value)
    if len(depth_ids) != len(set(depth_ids)):
        raise ValueError("dense_depth.depth_fields must have unique DepthFieldIds")
    if depth_ids != tuple(sorted(depth_ids)):
        raise ValueError("dense_depth.depth_fields must be in canonical DepthFieldId order")

    camera_ids = tuple(item.camera_solution_id.value for item in value)
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError(
            "dense_depth.depth_fields must have unique CameraSolutionId support"
        )

    observation_ids = tuple(item.observation_id.value for item in value)
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError(
            "dense_depth.depth_fields must have unique ObservationId support"
        )

    cameras_by_id = {
        camera.solution_id: camera for camera in source_geometry.camera_solutions
    }
    for depth in value:
        camera = cameras_by_id.get(depth.camera_solution_id)
        if camera is None:
            raise ValueError(
                "dense_depth DepthField must reference a CameraSolution supplied "
                "by source_geometry"
            )
        if depth.observation_id != camera.observation_id:
            raise ValueError(
                "dense_depth DepthField ObservationId must match its CameraSolution"
            )
        if depth.dimensions != camera.dimensions:
            raise ValueError(
                "dense_depth DepthField dimensions must match its CameraSolution"
            )


def _validate_source_artifacts(
    source_geometry: GeometrySolutionCandidate,
    value: object,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError("dense_depth.source_artifacts must be an immutable tuple")
    if not value:
        raise ValueError("dense_depth.source_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("dense_depth.source_artifacts members must be ArtifactRef")

    identities = tuple(
        (item.artifact_id.value, item.artifact_kind.value) for item in value
    )
    if len(identities) != len(set(identities)):
        raise ValueError("dense_depth.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "dense_depth.source_artifacts must use canonical "
            "ArtifactId/ArtifactKind order"
        )

    source_geometry_identities = {
        (item.artifact_id.value, item.artifact_kind.value)
        for item in source_geometry.source_artifacts
    }
    if not source_geometry_identities.issubset(set(identities)):
        raise ValueError(
            "dense_depth.source_artifacts must retain every source_geometry "
            "ArtifactRef"
        )


@dataclass(frozen=True, slots=True)
class DenseDepthArtifact:
    """Immutable dense-depth evidence derived from one canonical geometry hypothesis.

    This value validates evidence ownership and ancestry only. It does not execute MVS,
    convert depth conventions, infer scale, fuse fields, create points or surfaces, or
    make any quality/routing decision.
    """

    artifact_ref: ArtifactRef
    source_geometry: GeometrySolutionCandidate
    depth_fields: tuple[DepthField, ...]
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("dense_depth.artifact_ref must be ArtifactRef")
        if self.artifact_ref.artifact_kind != DENSE_DEPTH_ARTIFACT_KIND:
            raise ValueError(
                "dense_depth.artifact_ref kind must be geometry.dense_depth"
            )
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError(
                "dense_depth.source_geometry must be GeometrySolutionCandidate"
            )
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("dense_depth.producer must be ArtifactProducerIdentity")

        _validate_depth_fields(self.source_geometry, self.depth_fields)
        _validate_source_artifacts(self.source_geometry, self.source_artifacts)
