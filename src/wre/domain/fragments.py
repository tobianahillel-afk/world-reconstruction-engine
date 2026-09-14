from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.observations import ObservationId

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


@dataclass(frozen=True, slots=True, order=True)
class SpatialFragmentId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "fragment_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class LocalFrameId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "local_frame_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class SpatialFragment:
    """Solver-independent identity and membership for one local spatial component."""

    fragment_id: SpatialFragmentId
    local_frame_id: LocalFrameId
    observation_ids: tuple[ObservationId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_ids, tuple):
            raise ValueError("observation_ids must be an immutable tuple")
        if not self.observation_ids:
            raise ValueError("SpatialFragment must contain at least one observation")
        if not all(
            isinstance(observation_id, ObservationId) for observation_id in self.observation_ids
        ):
            raise ValueError("observation_ids must contain only ObservationId values")

        values = [observation_id.value for observation_id in self.observation_ids]
        if len(values) != len(set(values)):
            raise ValueError("SpatialFragment observation membership cannot contain duplicates")

        canonical = tuple(sorted(self.observation_ids, key=lambda item: item.value))
        object.__setattr__(self, "observation_ids", canonical)

    @property
    def observation_count(self) -> int:
        return len(self.observation_ids)

    def contains(self, observation_id: ObservationId) -> bool:
        return observation_id in self.observation_ids
