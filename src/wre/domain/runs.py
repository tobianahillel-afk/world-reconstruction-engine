from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from wre.domain.observations import ObservationId, Sha256Digest

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _require_opaque_id(value: str, context: str) -> None:
    if not isinstance(value, str) or not _OPAQUE_ID_RE.fullmatch(value):
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _require_non_blank_text(value: str, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-blank")


def _require_aware_datetime(value: datetime, context: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{context} must be timezone-aware")


def _canonical_observation_ids(
    observation_ids: tuple[ObservationId, ...], context: str
) -> tuple[ObservationId, ...]:
    if not isinstance(observation_ids, tuple):
        raise ValueError(f"{context} must be an immutable tuple")
    if not observation_ids:
        raise ValueError(f"{context} must contain at least one observation")
    if not all(isinstance(observation_id, ObservationId) for observation_id in observation_ids):
        raise ValueError(f"{context} must contain only ObservationId values")

    values = [observation_id.value for observation_id in observation_ids]
    if len(values) != len(set(values)):
        raise ValueError(f"{context} cannot contain duplicates")

    return tuple(sorted(observation_ids, key=lambda item: item.value))


@dataclass(frozen=True, slots=True, order=True)
class ReconstructionRunId:
    value: str

    def __post_init__(self) -> None:
        _require_opaque_id(self.value, "reconstruction_run_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ProducerRef:
    """Versioned implementation identity that produced a run."""

    implementation: str
    version: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank_text(self.implementation, "producer.implementation")
        _require_non_blank_text(self.version, "producer.version")
        if self.revision is not None:
            _require_non_blank_text(self.revision, "producer.revision")


@dataclass(frozen=True, slots=True)
class ReconstructionRun:
    """Immutable provenance envelope for one reconstruction computation."""

    run_id: ReconstructionRunId
    producer: ProducerRef
    input_observation_ids: tuple[ObservationId, ...]
    started_at: datetime
    completed_at: datetime | None = None
    configuration_sha256: Sha256Digest | None = None

    def __post_init__(self) -> None:
        canonical = _canonical_observation_ids(
            self.input_observation_ids, "input_observation_ids"
        )
        object.__setattr__(self, "input_observation_ids", canonical)

        _require_aware_datetime(self.started_at, "started_at")
        if self.completed_at is not None:
            _require_aware_datetime(self.completed_at, "completed_at")
            if self.completed_at < self.started_at:
                raise ValueError("completed_at cannot be earlier than started_at")

        if self.configuration_sha256 is not None and not isinstance(
            self.configuration_sha256, Sha256Digest
        ):
            raise ValueError("configuration_sha256 must be a Sha256Digest when present")


@dataclass(frozen=True, slots=True)
class DerivedArtifactProvenance:
    """Composable provenance reference for a derived evidence/geometry artifact."""

    producing_run_id: ReconstructionRunId
    source_observation_ids: tuple[ObservationId, ...]

    def __post_init__(self) -> None:
        canonical = _canonical_observation_ids(
            self.source_observation_ids, "source_observation_ids"
        )
        object.__setattr__(self, "source_observation_ids", canonical)
