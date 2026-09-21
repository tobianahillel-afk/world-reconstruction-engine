from __future__ import annotations

import math
import re
from dataclasses import dataclass

from wre.domain.fragments import LocalFrameId
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: object, context: str) -> None:
    if not isinstance(value, str) or _OPAQUE_ID_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _require_finite_float(value: object, context: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")


def _validate_source_observation_ids(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("point_map.source_observation_ids must be an immutable tuple")
    if not value:
        raise ValueError("point_map.source_observation_ids must be non-empty")
    if any(not isinstance(item, ObservationId) for item in value):
        raise TypeError("point_map.source_observation_ids members must be ObservationId")

    observation_values = tuple(item.value for item in value)
    if len(observation_values) != len(set(observation_values)):
        raise ValueError("point_map.source_observation_ids cannot contain duplicates")
    if observation_values != tuple(sorted(observation_values)):
        raise ValueError(
            "point_map.source_observation_ids must be in canonical ObservationId order"
        )


def _validate_positions_xyz(value: object) -> None:
    if not isinstance(value, tuple):
        raise TypeError("point_map.positions_xyz must be an immutable tuple")

    for position in value:
        if not isinstance(position, tuple):
            raise TypeError("point_map.positions_xyz members must be immutable tuples")
        if len(position) != 3:
            raise ValueError("point_map.positions_xyz members must be 3-vectors")
        for component in position:
            _require_finite_float(component, "point_map.positions_xyz component")


def _validate_confidence(value: object, point_count: int) -> None:
    if value is None:
        return
    if not isinstance(value, tuple):
        raise TypeError("point_map.confidence must be None or an immutable tuple")
    if len(value) != point_count:
        raise ValueError("point_map.confidence length must match point count")

    for member in value:
        _require_finite_float(member, "point_map.confidence member")
        if member < 0.0 or member > 1.0:
            raise ValueError(
                "point_map.confidence members must be in the inclusive range 0.0 to 1.0"
            )


@dataclass(frozen=True, slots=True, order=True)
class PointMapId:
    """Opaque identity for one canonical dense-or-sparse point prediction."""

    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "point_map_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class PointMap:
    """Canonical local-frame 3D point predictions with aggregate source support.

    The ordered point sequence is retained exactly as supplied. This value contract
    performs no point generation, sorting, deduplication, fusion, scale resolution,
    topology inference, quality decision, or coordinate-frame conversion.
    """

    point_map_id: PointMapId
    local_frame_id: LocalFrameId
    source_observation_ids: tuple[ObservationId, ...]
    positions_xyz: tuple[tuple[float, float, float], ...]
    confidence: tuple[float, ...] | None
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.point_map_id, PointMapId):
            raise TypeError("point_map.point_map_id must be PointMapId")
        if not isinstance(self.local_frame_id, LocalFrameId):
            raise TypeError("point_map.local_frame_id must be LocalFrameId")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("point_map.metrics must be MetricVector")

        _validate_source_observation_ids(self.source_observation_ids)
        _validate_positions_xyz(self.positions_xyz)
        _validate_confidence(self.confidence, len(self.positions_xyz))
