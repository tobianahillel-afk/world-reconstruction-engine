from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import StrEnum

from wre.domain.metadata import (
    GpsInterpretationStatus,
    ObservationMetadataInterpretation,
)
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun

COLMAP_SPATIAL_REFERENCE_VERSION = "4.2.0"
GPS_PAIRING_IMPLEMENTATION = "wre.gps_pairing"
GPS_PAIRING_VERSION = "1"
GPS_DISTANCE_MODEL = "wgs84_ecef_chord_altitude_zero"
GPS_MAP_DATUM_POLICY = "explicit_wgs84_only"

_WGS84_SEMI_MAJOR_AXIS_M = 6_378_137.0
_WGS84_FLATTENING = 1.0 / 298.257223563
_WGS84_ECCENTRICITY_SQUARED = _WGS84_FLATTENING * (2.0 - _WGS84_FLATTENING)


class GpsPairingEligibilityStatus(StrEnum):
    """Why one observation is or is not usable for L4.2 GPS pairing."""

    ELIGIBLE = "eligible"
    GPS_ABSENT = "gps_absent"
    GPS_INCOMPLETE = "gps_incomplete"
    GPS_INVALID = "gps_invalid"
    MAP_DATUM_MISSING = "map_datum_missing"
    MAP_DATUM_UNSUPPORTED = "map_datum_unsupported"


@dataclass(frozen=True, slots=True, kw_only=True)
class GpsPairingConfig:
    """Correctness-first spatial-neighbour policy for L4.2.

    The neighbor-count and radius semantics reuse COLMAP 4.2.0 spatial
    pairing. WRE fixes the upstream minimum-neighbor option to zero so a
    candidate is never forced outside the configured radius.
    """

    schema_version: int = 1
    max_num_neighbors: int = 50
    max_distance_m: float = 100.0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("GPS pairing config schema_version must be 1")
        if isinstance(self.max_num_neighbors, bool) or not isinstance(
            self.max_num_neighbors, int
        ):
            raise ValueError("max_num_neighbors must be an integer")
        if self.max_num_neighbors <= 0:
            raise ValueError("max_num_neighbors must be a positive integer")
        if isinstance(self.max_distance_m, bool) or not isinstance(
            self.max_distance_m, (int, float)
        ):
            raise ValueError("max_distance_m must be numeric")
        if not math.isfinite(float(self.max_distance_m)) or self.max_distance_m <= 0:
            raise ValueError("max_distance_m must be finite and positive")

    def canonical_document(self) -> dict[str, object]:
        return {
            "distance_model": GPS_DISTANCE_MODEL,
            "map_datum_policy": GPS_MAP_DATUM_POLICY,
            "max_distance_m": float(self.max_distance_m),
            "max_num_neighbors": self.max_num_neighbors,
            "min_num_neighbors": 0,
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
class GpsPairingEligibility:
    """Auditable eligibility decision for one source observation."""

    observation_id: ObservationId
    status: GpsPairingEligibilityStatus
    source_gps_status: GpsInterpretationStatus
    map_datum: str | None = None

    @property
    def eligible(self) -> bool:
        return self.status is GpsPairingEligibilityStatus.ELIGIBLE


@dataclass(frozen=True, slots=True)
class GpsPairCandidate:
    """One unordered pair proposed from explicit WGS84 proximity."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    distance_m: float

    def __post_init__(self) -> None:
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError("candidate observation IDs must be in canonical ascending order")
        if isinstance(self.distance_m, bool) or not isinstance(self.distance_m, (int, float)):
            raise ValueError("distance_m must be numeric")
        if not math.isfinite(float(self.distance_m)) or self.distance_m < 0:
            raise ValueError("distance_m must be finite and non-negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class GpsPairingRequest:
    run: ReconstructionRun
    interpretations: tuple[ObservationMetadataInterpretation, ...]
    config: GpsPairingConfig = field(default_factory=GpsPairingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.interpretations, tuple):
            raise ValueError("interpretations must be an immutable tuple")
        if not self.interpretations:
            raise ValueError("GPS pairing requires at least one metadata interpretation")
        if not all(
            isinstance(interpretation, ObservationMetadataInterpretation)
            for interpretation in self.interpretations
        ):
            raise ValueError(
                "interpretations must contain only ObservationMetadataInterpretation values"
            )
        observation_ids = tuple(item.observation_id for item in self.interpretations)
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("interpretations cannot contain duplicate observation IDs")
        canonical_ids = tuple(sorted(observation_ids, key=lambda item: item.value))
        if self.run.input_observation_ids != canonical_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match GPS interpretation membership"
            )
        if self.run.producer.implementation != GPS_PAIRING_IMPLEMENTATION:
            raise ValueError(
                f"ReconstructionRun producer must be {GPS_PAIRING_IMPLEMENTATION!r}"
            )
        if self.run.producer.version != GPS_PAIRING_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {GPS_PAIRING_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the GPS pairing config"
            )


@dataclass(frozen=True, slots=True)
class GpsPairingResult:
    provenance: DerivedArtifactProvenance
    configuration_sha256: Sha256Digest
    eligibility: tuple[GpsPairingEligibility, ...]
    candidates: tuple[GpsPairCandidate, ...]
    colmap_reference_version: str = COLMAP_SPATIAL_REFERENCE_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.eligibility, tuple) or not self.eligibility:
            raise ValueError("eligibility must be a non-empty immutable tuple")
        if not all(isinstance(item, GpsPairingEligibility) for item in self.eligibility):
            raise ValueError("eligibility must contain only GpsPairingEligibility values")

        eligibility_ids = tuple(item.observation_id for item in self.eligibility)
        canonical_ids = tuple(sorted(eligibility_ids, key=lambda item: item.value))
        if eligibility_ids != canonical_ids:
            raise ValueError("eligibility must use canonical observation order")
        if len(eligibility_ids) != len(set(eligibility_ids)):
            raise ValueError("eligibility cannot contain duplicate observation IDs")
        if self.provenance.source_observation_ids != canonical_ids:
            raise ValueError("result provenance must cover exact GPS pairing membership")
        if self.colmap_reference_version != COLMAP_SPATIAL_REFERENCE_VERSION:
            raise ValueError(
                "colmap_reference_version must identify reviewed COLMAP spatial semantics"
            )

        eligible_ids = {item.observation_id for item in self.eligibility if item.eligible}
        pair_keys: list[tuple[str, str]] = []
        for candidate in self.candidates:
            if not isinstance(candidate, GpsPairCandidate):
                raise ValueError("candidates must contain only GpsPairCandidate values")
            if (
                candidate.observation_id1 not in eligible_ids
                or candidate.observation_id2 not in eligible_ids
            ):
                raise ValueError("GPS candidates must reference only eligible observations")
            pair_keys.append((candidate.observation_id1.value, candidate.observation_id2.value))

        if len(pair_keys) != len(set(pair_keys)):
            raise ValueError("GPS candidate pairs must be unique")
        if pair_keys != sorted(pair_keys):
            raise ValueError("GPS candidates must retain deterministic canonical pair order")

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def eligible_observation_count(self) -> int:
        return sum(item.eligible for item in self.eligibility)


def _normalized_map_datum(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


def _eligibility(
    interpretation: ObservationMetadataInterpretation,
) -> GpsPairingEligibility:
    gps = interpretation.gps
    if gps.status is GpsInterpretationStatus.ABSENT:
        status = GpsPairingEligibilityStatus.GPS_ABSENT
    elif gps.status is GpsInterpretationStatus.INCOMPLETE:
        status = GpsPairingEligibilityStatus.GPS_INCOMPLETE
    elif gps.status is GpsInterpretationStatus.INVALID:
        status = GpsPairingEligibilityStatus.GPS_INVALID
    elif gps.map_datum is None:
        status = GpsPairingEligibilityStatus.MAP_DATUM_MISSING
    elif _normalized_map_datum(gps.map_datum) != "WGS84":
        status = GpsPairingEligibilityStatus.MAP_DATUM_UNSUPPORTED
    else:
        status = GpsPairingEligibilityStatus.ELIGIBLE

    return GpsPairingEligibility(
        observation_id=interpretation.observation_id,
        status=status,
        source_gps_status=gps.status,
        map_datum=gps.map_datum,
    )


def _wgs84_ecef_xyz_m(latitude_deg: float, longitude_deg: float) -> tuple[float, float, float]:
    """Convert WGS84 lat/lon to ECEF at zero altitude using COLMAP's formula."""

    latitude_rad = math.radians(latitude_deg)
    longitude_rad = math.radians(longitude_deg)
    sin_latitude = math.sin(latitude_rad)
    cos_latitude = math.cos(latitude_rad)
    sin_longitude = math.sin(longitude_rad)
    cos_longitude = math.cos(longitude_rad)
    prime_vertical_radius = _WGS84_SEMI_MAJOR_AXIS_M / math.sqrt(
        1.0 - _WGS84_ECCENTRICITY_SQUARED * sin_latitude * sin_latitude
    )
    return (
        prime_vertical_radius * cos_latitude * cos_longitude,
        prime_vertical_radius * cos_latitude * sin_longitude,
        prime_vertical_radius * (1.0 - _WGS84_ECCENTRICITY_SQUARED) * sin_latitude,
    )


def _canonical_pair(
    observation_id1: ObservationId,
    observation_id2: ObservationId,
    distance_m: float,
) -> GpsPairCandidate:
    if observation_id2.value < observation_id1.value:
        observation_id1, observation_id2 = observation_id2, observation_id1
    return GpsPairCandidate(
        observation_id1=observation_id1,
        observation_id2=observation_id2,
        distance_m=distance_m,
    )


def generate_gps_candidates(request: GpsPairingRequest) -> GpsPairingResult:
    """Propose WGS84-nearby pairs without making a geometric acceptance claim."""

    interpretations = tuple(
        sorted(request.interpretations, key=lambda item: item.observation_id.value)
    )
    eligibility = tuple(_eligibility(interpretation) for interpretation in interpretations)

    positions: dict[ObservationId, tuple[float, float, float]] = {}
    for interpretation, decision in zip(interpretations, eligibility, strict=True):
        if not decision.eligible:
            continue
        latitude_deg = interpretation.gps.latitude_deg
        longitude_deg = interpretation.gps.longitude_deg
        if latitude_deg is None or longitude_deg is None:
            raise ValueError("eligible GPS interpretation must expose latitude and longitude")
        positions[interpretation.observation_id] = _wgs84_ecef_xyz_m(
            latitude_deg,
            longitude_deg,
        )

    selected_pairs: dict[tuple[str, str], GpsPairCandidate] = {}
    eligible_ids = tuple(sorted(positions, key=lambda item: item.value))
    for query_id in eligible_ids:
        neighbors: list[tuple[float, str, ObservationId]] = []
        for neighbor_id in eligible_ids:
            if neighbor_id == query_id:
                continue
            distance_m = math.dist(positions[query_id], positions[neighbor_id])
            neighbors.append((distance_m, neighbor_id.value, neighbor_id))
        neighbors.sort(key=lambda item: (item[0], item[1]))

        for distance_m, _, neighbor_id in neighbors[: request.config.max_num_neighbors]:
            if distance_m > request.config.max_distance_m:
                break
            candidate = _canonical_pair(query_id, neighbor_id, distance_m)
            pair_key = (candidate.observation_id1.value, candidate.observation_id2.value)
            selected_pairs.setdefault(pair_key, candidate)

    candidates = tuple(selected_pairs[pair_key] for pair_key in sorted(selected_pairs))
    return GpsPairingResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=request.run.run_id,
            source_observation_ids=request.run.input_observation_ids,
        ),
        configuration_sha256=request.config.sha256,
        eligibility=eligibility,
        candidates=candidates,
    )
