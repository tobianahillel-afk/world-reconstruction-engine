from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.camera_solutions import CameraSolution
from wre.domain.depth_fields import DepthField
from wre.domain.geometry_solutions import GeometrySolution, GeometrySolutionId
from wre.domain.point_maps import PointMap
from wre.domain.producer_identity import ArtifactProducerIdentity


def _validate_source_artifacts(source_artifacts: object) -> None:
    if not isinstance(source_artifacts, tuple):
        raise TypeError("geometry_candidate.source_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in source_artifacts):
        raise TypeError("geometry_candidate.source_artifacts members must be ArtifactRef")

    identities = tuple(
        (item.artifact_id.value, item.artifact_kind.value) for item in source_artifacts
    )
    if len(identities) != len(set(identities)):
        raise ValueError("geometry_candidate.source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "geometry_candidate.source_artifacts must use canonical ArtifactId/ArtifactKind order"
        )


def _validate_cameras(
    geometry: GeometrySolution,
    cameras: object,
) -> tuple[CameraSolution, ...]:
    if not isinstance(cameras, tuple):
        raise TypeError("geometry_candidate.camera_solutions must be an immutable tuple")
    if any(not isinstance(item, CameraSolution) for item in cameras):
        raise TypeError("geometry_candidate.camera_solutions members must be CameraSolution")

    identifiers = tuple(item.solution_id.value for item in cameras)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_candidate.camera_solutions must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "geometry_candidate.camera_solutions must be in canonical CameraSolutionId order"
        )
    if any(item.local_frame_id != geometry.local_frame_id for item in cameras):
        raise ValueError(
            "geometry_candidate cameras must preserve the GeometrySolution LocalFrameId"
        )
    if tuple(item.solution_id for item in cameras) != geometry.camera_solution_ids:
        raise ValueError(
            "geometry_candidate cameras must exactly match GeometrySolution camera references"
        )
    return cameras


def _validate_depths(
    geometry: GeometrySolution,
    cameras: tuple[CameraSolution, ...],
    depths: object,
) -> tuple[DepthField, ...]:
    if not isinstance(depths, tuple):
        raise TypeError("geometry_candidate.depth_fields must be an immutable tuple")
    if any(not isinstance(item, DepthField) for item in depths):
        raise TypeError("geometry_candidate.depth_fields members must be DepthField")

    identifiers = tuple(item.depth_field_id.value for item in depths)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_candidate.depth_fields must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("geometry_candidate.depth_fields must be in canonical DepthFieldId order")
    if tuple(item.depth_field_id for item in depths) != geometry.depth_field_ids:
        raise ValueError(
            "geometry_candidate depths must exactly match GeometrySolution depth references"
        )

    cameras_by_id = {item.solution_id: item for item in cameras}
    for depth in depths:
        camera = cameras_by_id.get(depth.camera_solution_id)
        if camera is None:
            raise ValueError(
                "geometry_candidate DepthField must reference a supplied CameraSolution"
            )
        if depth.observation_id != camera.observation_id:
            raise ValueError(
                "geometry_candidate DepthField observation must match its CameraSolution"
            )
        if depth.dimensions != camera.dimensions:
            raise ValueError(
                "geometry_candidate DepthField dimensions must match its CameraSolution"
            )
    return depths


def _validate_point_maps(
    geometry: GeometrySolution,
    point_maps: object,
) -> tuple[PointMap, ...]:
    if not isinstance(point_maps, tuple):
        raise TypeError("geometry_candidate.point_maps must be an immutable tuple")
    if any(not isinstance(item, PointMap) for item in point_maps):
        raise TypeError("geometry_candidate.point_maps members must be PointMap")

    identifiers = tuple(item.point_map_id.value for item in point_maps)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_candidate.point_maps must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("geometry_candidate.point_maps must be in canonical PointMapId order")
    if any(item.local_frame_id != geometry.local_frame_id for item in point_maps):
        raise ValueError(
            "geometry_candidate PointMaps must preserve the GeometrySolution LocalFrameId"
        )
    if tuple(item.point_map_id for item in point_maps) != geometry.point_map_ids:
        raise ValueError(
            "geometry_candidate PointMaps must exactly match GeometrySolution point-map references"
        )
    return point_maps


@dataclass(frozen=True, slots=True)
class GeometrySolutionCandidate:
    """One complete canonical geometry hypothesis retained as a comparison alternative.

    This wrapper validates ownership and provenance only. It does not align, rescale,
    compare, score, refine, fuse, select or otherwise mutate the supplied geometry.
    """

    geometry_solution: GeometrySolution
    camera_solutions: tuple[CameraSolution, ...]
    depth_fields: tuple[DepthField, ...]
    point_maps: tuple[PointMap, ...]
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_solution, GeometrySolution):
            raise TypeError("geometry_candidate.geometry_solution must be GeometrySolution")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("geometry_candidate.producer must be ArtifactProducerIdentity")

        cameras = _validate_cameras(self.geometry_solution, self.camera_solutions)
        _validate_depths(self.geometry_solution, cameras, self.depth_fields)
        _validate_point_maps(self.geometry_solution, self.point_maps)
        _validate_source_artifacts(self.source_artifacts)

    @property
    def geometry_solution_id(self) -> GeometrySolutionId:
        return self.geometry_solution.geometry_solution_id


@dataclass(frozen=True, slots=True)
class CompetingGeometrySolutions:
    """Canonical set of retained geometry alternatives with no winner semantics."""

    candidates: tuple[GeometrySolutionCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise TypeError("competing_geometry.candidates must be an immutable tuple")
        if len(self.candidates) < 2:
            raise ValueError("competing_geometry requires at least two candidates")
        if any(not isinstance(item, GeometrySolutionCandidate) for item in self.candidates):
            raise TypeError(
                "competing_geometry.candidates members must be GeometrySolutionCandidate"
            )

        identifiers = tuple(item.geometry_solution_id.value for item in self.candidates)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("competing_geometry candidates must have unique GeometrySolutionIds")
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError(
                "competing_geometry candidates must be in canonical GeometrySolutionId order"
            )


@dataclass(frozen=True, slots=True, order=True)
class GeometrySolutionPair:
    """Identity-only unordered comparison edge between two geometry hypotheses."""

    left_geometry_solution_id: GeometrySolutionId
    right_geometry_solution_id: GeometrySolutionId

    def __post_init__(self) -> None:
        if not isinstance(self.left_geometry_solution_id, GeometrySolutionId):
            raise TypeError("geometry_pair.left_geometry_solution_id must be GeometrySolutionId")
        if not isinstance(self.right_geometry_solution_id, GeometrySolutionId):
            raise TypeError("geometry_pair.right_geometry_solution_id must be GeometrySolutionId")
        if self.left_geometry_solution_id == self.right_geometry_solution_id:
            raise ValueError("geometry_pair must reference two distinct GeometrySolutionIds")
        if self.left_geometry_solution_id.value > self.right_geometry_solution_id.value:
            raise ValueError("geometry_pair IDs must use canonical lexical left/right order")


def derive_geometry_solution_pairs(
    competing: CompetingGeometrySolutions,
) -> tuple[GeometrySolutionPair, ...]:
    """Return the complete deterministic unordered candidate-pair topology."""

    if not isinstance(competing, CompetingGeometrySolutions):
        raise TypeError("competing must be CompetingGeometrySolutions")

    identifiers = tuple(item.geometry_solution_id for item in competing.candidates)
    return tuple(
        GeometrySolutionPair(
            left_geometry_solution_id=identifiers[left_index],
            right_geometry_solution_id=identifiers[right_index],
        )
        for left_index in range(len(identifiers))
        for right_index in range(left_index + 1, len(identifiers))
    )
