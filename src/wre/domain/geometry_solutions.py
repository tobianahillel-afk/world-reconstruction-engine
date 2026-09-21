from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.camera_solutions import CameraSolutionId
from wre.domain.depth_fields import DepthFieldId
from wre.domain.fragments import LocalFrameId
from wre.domain.metrics import MetricVector
from wre.domain.point_maps import PointMapId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: object, context: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _validate_camera_solution_ids(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("geometry_solution.camera_solution_ids must be an immutable tuple")
    if not value:
        raise ValueError("geometry_solution.camera_solution_ids must be non-empty")
    if any(not isinstance(item, CameraSolutionId) for item in value):
        raise TypeError("geometry_solution.camera_solution_ids members must be CameraSolutionId")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_solution.camera_solution_ids cannot contain duplicates")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "geometry_solution.camera_solution_ids must be in canonical CameraSolutionId order"
        )


def _validate_depth_field_ids(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("geometry_solution.depth_field_ids must be an immutable tuple")
    if any(not isinstance(item, DepthFieldId) for item in value):
        raise TypeError("geometry_solution.depth_field_ids members must be DepthFieldId")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_solution.depth_field_ids cannot contain duplicates")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError(
            "geometry_solution.depth_field_ids must be in canonical DepthFieldId order"
        )


def _validate_point_map_ids(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("geometry_solution.point_map_ids must be an immutable tuple")
    if any(not isinstance(item, PointMapId) for item in value):
        raise TypeError("geometry_solution.point_map_ids members must be PointMapId")

    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("geometry_solution.point_map_ids cannot contain duplicates")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("geometry_solution.point_map_ids must be in canonical PointMapId order")


@dataclass(frozen=True, slots=True, order=True)
class GeometrySolutionId:
    """Opaque identity for one canonical geometry hypothesis."""

    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "geometry_solution_id")

    def __str__(self) -> str:
        return self.value


class GeometryScaleStatus(StrEnum):
    """Truth-bearing local-scale semantics for one canonical geometry hypothesis."""

    UNRESOLVED = "unresolved"
    METRIC = "metric"


@dataclass(frozen=True, slots=True)
class GeometrySolution:
    """One coherent canonical reconstruction hypothesis in a single local frame.

    UNRESOLVED means local distances have arbitrary unresolved scale. METRIC means
    one local-frame distance unit is one meter, without asserting origin, orientation,
    Earth alignment, geographic placement, or compatibility with another local frame.
    """

    geometry_solution_id: GeometrySolutionId
    local_frame_id: LocalFrameId
    scale_status: GeometryScaleStatus
    camera_solution_ids: tuple[CameraSolutionId, ...]
    depth_field_ids: tuple[DepthFieldId, ...]
    point_map_ids: tuple[PointMapId, ...]
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_solution_id, GeometrySolutionId):
            raise TypeError("geometry_solution.geometry_solution_id must be GeometrySolutionId")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("geometry_solution.local_frame_id must be LocalFrameId")
        if not isinstance(self.scale_status, GeometryScaleStatus):
            raise TypeError("geometry_solution.scale_status must be GeometryScaleStatus")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("geometry_solution.metrics must be MetricVector")

        _validate_camera_solution_ids(self.camera_solution_ids)
        _validate_depth_field_ids(self.depth_field_ids)
        _validate_point_map_ids(self.point_map_ids)

        if not self.depth_field_ids and not self.point_map_ids:
            raise ValueError(
                "geometry_solution must reference at least one DepthFieldId or PointMapId"
            )
