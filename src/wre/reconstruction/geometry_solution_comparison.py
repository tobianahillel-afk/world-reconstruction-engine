from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.camera_solutions import CameraSolution
from wre.domain.depth_fields import DepthField
from wre.domain.geometry_solutions import GeometrySolution, GeometrySolutionId
from wre.domain.point_maps import PointMap
from wre.domain.producer_identity import ArtifactProducerIdentity


def _validate_cameras(
    cameras: object,
    geometry_solution: GeometrySolution,
) -> tuple[CameraSolution, ...]:
    if not isinstance(cameras, tuple):
        raise TypeError("geometry candidate camera_solutions must be an immutable tuple")
    if any(not isinstance(item, CameraSolution) for item in cameras):
        raise TypeError("geometry candidate camera_solutions members must be CameraSolution")

    identifiers = tuple(item.solution_id.value for item in cameras)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry candidate camera_solutions must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "geometry candidate camera_solutions must be in canonical CameraSolutionId order"
        )

    local_frame = geometry_solution.local_frame_id
    if any(item.local_frame_id != local_frame for item in cameras):
        raise ValueError(
            "geometry candidate cameras and GeometrySolution must share one LocalFrameId"
        )

    expected = tuple(item.solution_id for item in cameras)
    if geometry_solution.camera_solution_ids != expected:
        raise ValueError(
            "geometry candidate GeometrySolution must reference exactly all supplied cameras"
        )
    return cameras


def _validate_depths(
    depths: object,
    geometry_solution: GeometrySolution,
    cameras: tuple[CameraSolution, ...],
) -> tuple[DepthField, ...]:
    if not isinstance(depths, tuple):
        raise TypeError("geometry candidate depth_fields must be an immutable tuple")
    if any(not isinstance(item, DepthField) for item in depths):
        raise TypeError("geometry candidate depth_fields members must be DepthField")

    identifiers = tuple(item.depth_field_id.value for item in depths)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry candidate depth_fields must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("geometry candidate depth_fields must be in canonical DepthFieldId order")

    expected = tuple(item.depth_field_id for item in depths)
    if geometry_solution.depth_field_ids != expected:
        raise ValueError(
            "geometry candidate GeometrySolution must reference exactly all supplied depths"
        )

    cameras_by_id = {item.solution_id: item for item in cameras}
    for depth in depths:
        camera = cameras_by_id.get(depth.camera_solution_id)
        if camera is None:
            raise ValueError(
                "geometry candidate DepthField must reference a supplied CameraSolution"
            )
        if depth.observation_id != camera.observation_id:
            raise ValueError(
                "geometry candidate DepthField observation must match its CameraSolution"
            )
        if depth.dimensions != camera.dimensions:
            raise ValueError(
                "geometry candidate DepthField dimensions must match its CameraSolution"
            )
    return depths


def _validate_point_maps(
    point_maps: object,
    geometry_solution: GeometrySolution,
    cameras: tuple[CameraSolution, ...],
) -> tuple[PointMap, ...]:
    if not isinstance(point_maps, tuple):
        raise TypeError("geometry candidate point_maps must be an immutable tuple")
    if any(not isinstance(item, PointMap) for item in point_maps):
        raise TypeError("geometry candidate point_maps members must be PointMap")

    identifiers = tuple(item.point_map_id.value for item in point_maps)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry candidate point_maps must have unique IDs")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("geometry candidate point_maps must be in canonical PointMapId order")

    local_frame = geometry_solution.local_frame_id
    if any(item.local_frame_id != local_frame for item in point_maps):
        raise ValueError(
            "geometry candidate PointMaps and GeometrySolution must share one LocalFrameId"
        )

    expected = tuple(item.point_map_id for item in point_maps)
    if geometry_solution.point_map_ids != expected:
        raise ValueError(
            "geometry candidate GeometrySolution must reference exactly all supplied point maps"
        )

    camera_observations = {item.observation_id for item in cameras}
    for point_map in point_maps:
        if not set(point_map.source_observation_ids).issubset(camera_observations):
            raise ValueError(
                "geometry candidate PointMap support must stay inside supplied camera membership"
            )
    return point_maps


def _validate_source_artifacts(value: object) -> tuple[ArtifactRef, ...]:
    if not isinstance(value, tuple):
        raise TypeError("geometry candidate source_artifacts must be an immutable tuple")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("geometry candidate source_artifacts members must be ArtifactRef")

    identities = tuple((item.artifact_id.value, item.artifact_kind.value) for item in value)
    if len(identities) != len(set(identities)):
        raise ValueError("geometry candidate source_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "geometry candidate source_artifacts must be in canonical ArtifactId/ArtifactKind order"
        )
    return value


@dataclass(frozen=True, slots=True)
class GeometrySolutionCandidate:
    """One complete canonical geometry hypothesis retained as a comparison alternative.

    This wrapper preserves canonical child values, local-frame/scale semantics and
    provenance exactly. It performs no alignment, coordinate conversion, scoring,
    ranking, refinement, quality decision or solver execution.
    """

    geometry_solution: GeometrySolution
    camera_solutions: tuple[CameraSolution, ...]
    depth_fields: tuple[DepthField, ...]
    point_maps: tuple[PointMap, ...]
    producer: ArtifactProducerIdentity
    source_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_solution, GeometrySolution):
            raise TypeError("geometry candidate geometry_solution must be GeometrySolution")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("geometry candidate producer must be ArtifactProducerIdentity")

        cameras = _validate_cameras(self.camera_solutions, self.geometry_solution)
        _validate_depths(self.depth_fields, self.geometry_solution, cameras)
        _validate_point_maps(self.point_maps, self.geometry_solution, cameras)
        _validate_source_artifacts(self.source_artifacts)


@dataclass(frozen=True, slots=True)
class CompetingGeometrySolutions:
    """Two or more complete geometry hypotheses retained without winner semantics."""

    candidates: tuple[GeometrySolutionCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise TypeError("competing geometry candidates must be an immutable tuple")
        if len(self.candidates) < 2:
            raise ValueError("competing geometry requires at least two candidates")
        if any(not isinstance(item, GeometrySolutionCandidate) for item in self.candidates):
            raise TypeError(
                "competing geometry candidates members must be GeometrySolutionCandidate"
            )

        identifiers = tuple(
            item.geometry_solution.geometry_solution_id.value for item in self.candidates
        )
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("competing geometry candidates must have unique GeometrySolutionIds")
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError(
                "competing geometry candidates must be in canonical GeometrySolutionId order"
            )


@dataclass(frozen=True, slots=True, order=True)
class GeometrySolutionPair:
    """Identity-only unordered pair of distinct canonical geometry hypotheses."""

    left_geometry_solution_id: GeometrySolutionId
    right_geometry_solution_id: GeometrySolutionId

    def __post_init__(self) -> None:
        if not isinstance(self.left_geometry_solution_id, GeometrySolutionId):
            raise TypeError("geometry pair left_geometry_solution_id must be GeometrySolutionId")
        if not isinstance(self.right_geometry_solution_id, GeometrySolutionId):
            raise TypeError("geometry pair right_geometry_solution_id must be GeometrySolutionId")
        if self.left_geometry_solution_id == self.right_geometry_solution_id:
            raise ValueError("geometry pair must contain two distinct GeometrySolutionIds")
        if self.left_geometry_solution_id.value > self.right_geometry_solution_id.value:
            raise ValueError("geometry pair IDs must be in canonical lexical left/right order")


def derive_geometry_solution_pairs(
    competing: CompetingGeometrySolutions,
) -> tuple[GeometrySolutionPair, ...]:
    """Derive the complete deterministic unordered pair topology."""

    if not isinstance(competing, CompetingGeometrySolutions):
        raise TypeError("competing must be CompetingGeometrySolutions")

    identifiers = tuple(
        item.geometry_solution.geometry_solution_id for item in competing.candidates
    )
    pairs: list[GeometrySolutionPair] = []
    for left_index, left_id in enumerate(identifiers):
        for right_id in identifiers[left_index + 1 :]:
            pairs.append(
                GeometrySolutionPair(
                    left_geometry_solution_id=left_id,
                    right_geometry_solution_id=right_id,
                )
            )
    return tuple(pairs)
