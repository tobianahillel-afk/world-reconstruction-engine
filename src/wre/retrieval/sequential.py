from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun

COLMAP_SEQUENTIAL_REFERENCE_VERSION = "4.2.0"
SEQUENTIAL_PAIRING_IMPLEMENTATION = "wre.sequential_pairing"
SEQUENTIAL_PAIRING_VERSION = "1"


@dataclass(frozen=True, slots=True, kw_only=True)
class SequentialPairingConfig:
    """Deterministic L4.1 sequence-neighbour candidate policy.

    The overlap semantics intentionally match COLMAP 4.2.0's non-rig,
    non-loop-detection SequentialPairGenerator. WRE supplies sequence order
    explicitly rather than treating lexicographic file names as temporal truth.
    """

    schema_version: int = 1
    overlap: int = 10
    quadratic_overlap: bool = True
    expand_rig_images: bool = False
    loop_detection: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("sequential pairing config schema_version must be 1")
        if isinstance(self.overlap, bool) or not isinstance(self.overlap, int):
            raise ValueError("overlap must be an integer")
        if self.overlap <= 0:
            raise ValueError("overlap must be a positive integer")
        if not isinstance(self.quadratic_overlap, bool):
            raise ValueError("quadratic_overlap must be a boolean")
        if not isinstance(self.expand_rig_images, bool):
            raise ValueError("expand_rig_images must be a boolean")
        if not isinstance(self.loop_detection, bool):
            raise ValueError("loop_detection must be a boolean")
        if self.expand_rig_images:
            raise ValueError("rig expansion is outside L4.1 sequential-pairing scope")
        if self.loop_detection:
            raise ValueError("loop detection is outside L4.1 sequential-pairing scope")

    def canonical_document(self) -> dict[str, object]:
        return {
            "expand_rig_images": self.expand_rig_images,
            "loop_detection": self.loop_detection,
            "overlap": self.overlap,
            "quadratic_overlap": self.quadratic_overlap,
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> Sha256Digest:
        payload = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return Sha256Digest(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True)
class SequentialPairCandidate:
    """One unordered observation pair proposed from explicit sequence proximity."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    sequence_index1: int
    sequence_index2: int
    sequence_distance: int

    def __post_init__(self) -> None:
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError("candidate observation IDs must be in canonical ascending order")
        for name in ("sequence_index1", "sequence_index2"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.sequence_index1 == self.sequence_index2:
            raise ValueError("candidate observations must occupy distinct sequence indices")
        if (
            isinstance(self.sequence_distance, bool)
            or not isinstance(self.sequence_distance, int)
            or self.sequence_distance <= 0
        ):
            raise ValueError("sequence_distance must be a positive integer")
        if self.sequence_distance != abs(self.sequence_index2 - self.sequence_index1):
            raise ValueError("sequence_distance must match the candidate sequence indices")


@dataclass(frozen=True, slots=True, kw_only=True)
class SequentialPairingRequest:
    run: ReconstructionRun
    ordered_observation_ids: tuple[ObservationId, ...]
    config: SequentialPairingConfig = field(default_factory=SequentialPairingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_observation_ids, tuple):
            raise ValueError("ordered_observation_ids must be an immutable tuple")
        if not self.ordered_observation_ids:
            raise ValueError("sequential pairing requires at least one observation")
        if not all(
            isinstance(observation_id, ObservationId)
            for observation_id in self.ordered_observation_ids
        ):
            raise ValueError("ordered_observation_ids must contain only ObservationId values")
        values = tuple(item.value for item in self.ordered_observation_ids)
        if len(values) != len(set(values)):
            raise ValueError("ordered_observation_ids cannot contain duplicates")

        canonical_ids = tuple(sorted(self.ordered_observation_ids, key=lambda item: item.value))
        if self.run.input_observation_ids != canonical_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match the explicit sequence membership"
            )
        if self.run.producer.implementation != SEQUENTIAL_PAIRING_IMPLEMENTATION:
            raise ValueError(
                f"ReconstructionRun producer must be {SEQUENTIAL_PAIRING_IMPLEMENTATION!r}"
            )
        if self.run.producer.version != SEQUENTIAL_PAIRING_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SEQUENTIAL_PAIRING_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the sequential pairing config"
            )


@dataclass(frozen=True, slots=True)
class SequentialPairingResult:
    provenance: DerivedArtifactProvenance
    configuration_sha256: Sha256Digest
    ordered_observation_ids: tuple[ObservationId, ...]
    candidates: tuple[SequentialPairCandidate, ...]
    colmap_reference_version: str = COLMAP_SEQUENTIAL_REFERENCE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.ordered_observation_ids, tuple) or not self.ordered_observation_ids:
            raise ValueError("ordered_observation_ids must be a non-empty immutable tuple")
        if len(self.ordered_observation_ids) != len(set(self.ordered_observation_ids)):
            raise ValueError("ordered_observation_ids cannot contain duplicates")
        canonical_ids = tuple(sorted(self.ordered_observation_ids, key=lambda item: item.value))
        if self.provenance.source_observation_ids != canonical_ids:
            raise ValueError("result provenance must cover the exact sequence membership")
        if self.colmap_reference_version != COLMAP_SEQUENTIAL_REFERENCE_VERSION:
            raise ValueError(
                "colmap_reference_version must identify the reviewed COLMAP sequential semantics"
            )

        pair_keys: list[tuple[str, str]] = []
        ordering_keys: list[tuple[int, int]] = []
        for candidate in self.candidates:
            if not isinstance(candidate, SequentialPairCandidate):
                raise ValueError("candidates must contain only SequentialPairCandidate values")
            if candidate.sequence_index1 >= len(self.ordered_observation_ids):
                raise ValueError("candidate sequence_index1 is outside the supplied sequence")
            if candidate.sequence_index2 >= len(self.ordered_observation_ids):
                raise ValueError("candidate sequence_index2 is outside the supplied sequence")
            if (
                self.ordered_observation_ids[candidate.sequence_index1]
                != candidate.observation_id1
            ):
                raise ValueError("candidate observation_id1 does not match its sequence index")
            if (
                self.ordered_observation_ids[candidate.sequence_index2]
                != candidate.observation_id2
            ):
                raise ValueError("candidate observation_id2 does not match its sequence index")
            pair_keys.append(
                (candidate.observation_id1.value, candidate.observation_id2.value)
            )
            ordering_keys.append(
                (
                    min(candidate.sequence_index1, candidate.sequence_index2),
                    candidate.sequence_distance,
                )
            )

        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("sequential candidate pairs must be unique")
        if ordering_keys != sorted(ordering_keys):
            raise ValueError("sequential candidates must retain deterministic sequence order")

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)


def _offsets(config: SequentialPairingConfig) -> tuple[int, ...]:
    if config.quadratic_overlap:
        return tuple(1 << exponent for exponent in range(config.overlap))
    return tuple(range(1, config.overlap + 1))


def _candidate(
    observation_id1: ObservationId,
    sequence_index1: int,
    observation_id2: ObservationId,
    sequence_index2: int,
) -> SequentialPairCandidate:
    if observation_id2.value < observation_id1.value:
        observation_id1, observation_id2 = observation_id2, observation_id1
        sequence_index1, sequence_index2 = sequence_index2, sequence_index1
    return SequentialPairCandidate(
        observation_id1=observation_id1,
        observation_id2=observation_id2,
        sequence_index1=sequence_index1,
        sequence_index2=sequence_index2,
        sequence_distance=abs(sequence_index2 - sequence_index1),
    )


def generate_sequential_candidates(request: SequentialPairingRequest) -> SequentialPairingResult:
    """Propose sequence-neighbour pairs without matching or geometric acceptance."""

    candidates: list[SequentialPairCandidate] = []
    offsets = _offsets(request.config)
    for sequence_index1, observation_id1 in enumerate(request.ordered_observation_ids):
        for offset in offsets:
            sequence_index2 = sequence_index1 + offset
            if sequence_index2 >= len(request.ordered_observation_ids):
                break
            candidates.append(
                _candidate(
                    observation_id1,
                    sequence_index1,
                    request.ordered_observation_ids[sequence_index2],
                    sequence_index2,
                )
            )

    return SequentialPairingResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=request.run.run_id,
            source_observation_ids=request.run.input_observation_ids,
        ),
        configuration_sha256=request.config.sha256,
        ordered_observation_ids=request.ordered_observation_ids,
        candidates=tuple(candidates),
    )
