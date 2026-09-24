from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.depth_fields import DepthField
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

DENSE_DEPTH_ARTIFACT_KIND = ArtifactKind("geometry.dense_depth")


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _validate_source_artifacts(
    source_geometry: GeometrySolutionCandidate,
    source_artifacts: object,
) -> None:
    if not isinstance(source_artifacts, tuple):
        raise TypeError("dense_depth.source_artifacts must be an immutable tuple")
    if not source_artifacts:
        raise ValueError("dense_depth.source_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in source_artifacts):
        raise TypeError("dense_depth.source_artifacts members must be ArtifactRef")

    identities = tuple(_artifact_key(item) for item in source_artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError("dense_depth.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "dense_depth.source_artifacts must use canonical ArtifactId/ArtifactKind order"
        )

    source_geometry_identities = {
        _artifact_key(item) for item in source_geometry.source_artifacts
    }
    if not source_geometry_identities.issubset(set(identities)):
        raise ValueError(
            "dense_depth.source_artifacts must retain every source GeometrySolutionCandidate "
            "artifact"
        )


def _validate_depth_fields(
    source_geometry: GeometrySolutionCandidate,
    depth_fields: object,
) -> None:
    if not isinstance(depth_fields, tuple):
        raise TypeError("dense_depth.depth_fields must be an immutable tuple")
    if not depth_fields:
        raise ValueError("dense_depth.depth_fields must be non-empty")
    if any(not isinstance(item, DepthField) for item in depth_fields):
        raise TypeError("dense_depth.depth_fields members must be DepthField")

    depth_ids = tuple(item.depth_field_id.value for item in depth_fields)
    if len(depth_ids) != len(set(depth_ids)):
        raise ValueError("dense_depth.depth_fields must have unique DepthFieldIds")
    if depth_ids != tuple(sorted(depth_ids)):
        raise ValueError("dense_depth.depth_fields must use canonical DepthFieldId order")

    camera_ids = tuple(item.camera_solution_id.value for item in depth_fields)
    if len(camera_ids) != len(set(camera_ids)):
        raise ValueError(
            "dense_depth.depth_fields must not contain competing support for one CameraSolution"
        )

    observation_ids = tuple(item.observation_id.value for item in depth_fields)
    if len(observation_ids) != len(set(observation_ids)):
        raise ValueError(
            "dense_depth.depth_fields must not contain competing support for one Observation"
        )

    cameras_by_id = {
        camera.solution_id: camera for camera in source_geometry.camera_solutions
    }
    for depth in depth_fields:
        camera = cameras_by_id.get(depth.camera_solution_id)
        if camera is None:
            raise ValueError(
                "dense_depth DepthField must reference a CameraSolution supplied by "
                "source_geometry"
            )
        if depth.observation_id != camera.observation_id:
            raise ValueError(
                "dense_depth DepthField ObservationId must match its source CameraSolution"
            )
        if depth.dimensions != camera.dimensions:
            raise ValueError(
                "dense_depth DepthField ImageDimensions must match its source CameraSolution"
            )


@dataclass(frozen=True, slots=True)
class DenseDepthArtifact:
    """Immutable dense-depth evidence derived from one canonical geometry hypothesis.

    The artifact validates camera linkage and provenance only. It does not execute
    stereo/MVS, convert depth conventions or scale, fuse fields, fill holes, generate
    point/surface geometry, or mutate the source geometry hypothesis.
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
                "dense_depth.artifact_ref must use artifact kind geometry.dense_depth"
            )
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError(
                "dense_depth.source_geometry must be GeometrySolutionCandidate"
            )
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("dense_depth.producer must be ArtifactProducerIdentity")

        _validate_depth_fields(self.source_geometry, self.depth_fields)
        _validate_source_artifacts(self.source_geometry, self.source_artifacts)
