from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.camera_solutions import CameraSolution, RotationMatrix3x3, TranslationVector3
from wre.domain.metrics import (
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
)
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.geometry_solution_comparison import (
    CompetingGeometrySolutions,
    GeometrySolutionCandidate,
    GeometrySolutionPair,
    derive_geometry_solution_pairs,
)

GEOMETRY_CONSENSUS_IMPLEMENTATION = "wre.reconstruction.geometry_consensus"
GEOMETRY_CONSENSUS_VERSION = "1"
GEOMETRY_CONSENSUS_REVISION = "v2l14.5"

_EVALUATOR_CONFIGURATION_DOCUMENT = {
    "schema_version": 1,
    "candidate_pairing": "canonical_geometry_solution_pair",
    "observation_overlap": "shared_over_union_empty_union_zero",
    "relative_rotation": "camera_from_local_pair_geodesic_degrees",
    "relative_translation": "first_camera_baseline_direction_degrees",
    "alignment": "none",
    "aggregation": "deterministic_exact_median",
}
_EVALUATOR_CONFIGURATION_BYTES = json.dumps(
    _EVALUATOR_CONFIGURATION_DOCUMENT,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
GEOMETRY_CONSENSUS_EVALUATOR = ArtifactProducerIdentity(
    producer=ProducerRef(
        implementation=GEOMETRY_CONSENSUS_IMPLEMENTATION,
        version=GEOMETRY_CONSENSUS_VERSION,
        revision=GEOMETRY_CONSENSUS_REVISION,
    ),
    configuration=ConfigurationIdentity(
        sha256=Sha256Digest(hashlib.sha256(_EVALUATOR_CONFIGURATION_BYTES).hexdigest())
    ),
)

_GEOMETRY_CONSENSUS_DIMENSION = MetricDimension("geometry.consensus")
_RATIO_UNIT = MetricUnit("ratio")
_DEGREE_UNIT = MetricUnit("degree")
_MEDIAN_SHARED_PAIR = MetricAggregation("median_shared_camera_pair")

SHARED_OBSERVATION_COVERAGE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.consensus.shared_observation_union_ratio"),
    dimension=_GEOMETRY_CONSENSUS_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("shared_over_union"),
)
RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.consensus.relative_rotation_disagreement_deg_median"),
    dimension=_GEOMETRY_CONSENSUS_DIMENSION,
    unit=_DEGREE_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=_MEDIAN_SHARED_PAIR,
)
CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.consensus.translation_pair_coverage_ratio"),
    dimension=_GEOMETRY_CONSENSUS_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("eligible_shared_pair_ratio"),
)
RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR = MetricDescriptor(
    name=MetricName(
        "geometry.consensus.relative_translation_direction_disagreement_deg_median"
    ),
    dimension=_GEOMETRY_CONSENSUS_DIMENSION,
    unit=_DEGREE_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=_MEDIAN_SHARED_PAIR,
)

Vector3 = tuple[float, float, float]


def _validate_input_artifacts(value: object) -> tuple[ArtifactRef, ...]:
    if not isinstance(value, tuple):
        raise TypeError("geometry_consensus.input_artifacts must be an immutable tuple")
    if not value:
        raise ValueError("geometry_consensus.input_artifacts must be non-empty")
    if any(not isinstance(item, ArtifactRef) for item in value):
        raise TypeError("geometry_consensus.input_artifacts members must be ArtifactRef")

    artifacts = value
    identities = tuple(
        (item.artifact_id.value, item.artifact_kind.value) for item in artifacts
    )
    if len(identities) != len(set(identities)):
        raise ValueError("geometry_consensus.input_artifacts must be unique")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "geometry_consensus.input_artifacts must use canonical ArtifactId/ArtifactKind order"
        )
    return artifacts


@dataclass(frozen=True, slots=True, kw_only=True)
class GeometryConsensusRequest:
    """Pure pairwise disagreement request over retained geometry alternatives."""

    competing: CompetingGeometrySolutions
    input_artifacts: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.competing, CompetingGeometrySolutions):
            raise TypeError("geometry_consensus.competing must be CompetingGeometrySolutions")
        _validate_input_artifacts(self.input_artifacts)


@dataclass(frozen=True, slots=True)
class GeometryPairDisagreement:
    """One canonical candidate pair and its solver-independent disagreement evidence."""

    pair: GeometrySolutionPair
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.pair, GeometrySolutionPair):
            raise TypeError("geometry_pair_disagreement.pair must be GeometrySolutionPair")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("geometry_pair_disagreement.metrics must be MetricVector")


@dataclass(frozen=True, slots=True)
class GeometryConsensusResult:
    """Complete deterministic disagreement evidence for every candidate pair."""

    request: GeometryConsensusRequest
    pair_disagreements: tuple[GeometryPairDisagreement, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.request, GeometryConsensusRequest):
            raise TypeError("geometry_consensus_result.request must be GeometryConsensusRequest")
        if not isinstance(self.pair_disagreements, tuple):
            raise TypeError(
                "geometry_consensus_result.pair_disagreements must be an immutable tuple"
            )
        if any(
            not isinstance(item, GeometryPairDisagreement)
            for item in self.pair_disagreements
        ):
            raise TypeError(
                "geometry_consensus_result.pair_disagreements members must be "
                "GeometryPairDisagreement"
            )

        expected_pairs = derive_geometry_solution_pairs(self.request.competing)
        actual_pairs = tuple(item.pair for item in self.pair_disagreements)
        if actual_pairs != expected_pairs:
            raise ValueError(
                "geometry_consensus_result must contain exactly one metric vector "
                "for every canonical GeometrySolutionPair"
            )


def _transpose(matrix: RotationMatrix3x3) -> RotationMatrix3x3:
    return (
        (matrix[0][0], matrix[1][0], matrix[2][0]),
        (matrix[0][1], matrix[1][1], matrix[2][1]),
        (matrix[0][2], matrix[1][2], matrix[2][2]),
    )


def _matmul(
    left: RotationMatrix3x3,
    right: RotationMatrix3x3,
) -> RotationMatrix3x3:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _matvec(
    matrix: RotationMatrix3x3,
    vector: TranslationVector3 | Vector3,
) -> Vector3:
    return (
        sum(matrix[0][axis] * vector[axis] for axis in range(3)),
        sum(matrix[1][axis] * vector[axis] for axis in range(3)),
        sum(matrix[2][axis] * vector[axis] for axis in range(3)),
    )


def _camera_center(camera: CameraSolution) -> Vector3:
    rotated = _matvec(_transpose(camera.rotation_matrix), camera.translation_xyz)
    return (-rotated[0], -rotated[1], -rotated[2])


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _unit(vector: Vector3) -> Vector3 | None:
    norm = math.sqrt(sum(member * member for member in vector))
    if norm == 0.0:
        return None
    return tuple(member / norm for member in vector)  # type: ignore[return-value]


def _relative_rotation(
    first: CameraSolution,
    second: CameraSolution,
) -> RotationMatrix3x3:
    return _matmul(second.rotation_matrix, _transpose(first.rotation_matrix))


def _rotation_angle_degrees(
    left: RotationMatrix3x3,
    right: RotationMatrix3x3,
) -> float:
    difference = _matmul(left, _transpose(right))
    trace = difference[0][0] + difference[1][1] + difference[2][2]
    cosine = max(-1.0, min(1.0, (trace - 1.0) * 0.5))
    return math.degrees(math.acos(cosine))


def _baseline_direction_in_first_camera(
    first: CameraSolution,
    second: CameraSolution,
) -> Vector3 | None:
    baseline_local = _subtract(_camera_center(second), _camera_center(first))
    return _unit(_matvec(first.rotation_matrix, baseline_local))


def _direction_angle_degrees(left: Vector3, right: Vector3) -> float:
    cosine = max(
        -1.0,
        min(1.0, sum(left[axis] * right[axis] for axis in range(3))),
    )
    return math.degrees(math.acos(cosine))


def _median(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("geometry_consensus median requires at least one value")
    ordered = tuple(sorted(values))
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) * 0.5


def _observation(
    descriptor: MetricDescriptor,
    value: float,
    provenance: MetricProvenance,
) -> MetricObservation:
    return MetricObservation(
        descriptor=descriptor,
        value=float(value),
        provenance=provenance,
    )


def _candidate_cameras_by_observation(
    candidate: GeometrySolutionCandidate,
) -> dict[ObservationId, CameraSolution]:
    return {camera.observation_id: camera for camera in candidate.camera_solutions}


def _pair_metrics(
    left: GeometrySolutionCandidate,
    right: GeometrySolutionCandidate,
    *,
    provenance: MetricProvenance,
) -> MetricVector:
    left_by_observation = _candidate_cameras_by_observation(left)
    right_by_observation = _candidate_cameras_by_observation(right)

    left_ids = set(left_by_observation)
    right_ids = set(right_by_observation)
    shared_ids = tuple(
        sorted(left_ids.intersection(right_ids), key=lambda item: item.value)
    )
    union_count = len(left_ids.union(right_ids))
    shared_ratio = len(shared_ids) / union_count if union_count else 0.0

    observations: list[MetricObservation] = [
        _observation(
            SHARED_OBSERVATION_COVERAGE_DESCRIPTOR,
            shared_ratio,
            provenance,
        )
    ]

    if len(shared_ids) >= 2:
        rotation_disagreements: list[float] = []
        translation_disagreements: list[float] = []
        pair_count = 0

        for first_index in range(len(shared_ids) - 1):
            for second_index in range(first_index + 1, len(shared_ids)):
                first_id = shared_ids[first_index]
                second_id = shared_ids[second_index]
                pair_count += 1

                left_first = left_by_observation[first_id]
                left_second = left_by_observation[second_id]
                right_first = right_by_observation[first_id]
                right_second = right_by_observation[second_id]

                rotation_disagreements.append(
                    _rotation_angle_degrees(
                        _relative_rotation(left_first, left_second),
                        _relative_rotation(right_first, right_second),
                    )
                )

                left_direction = _baseline_direction_in_first_camera(
                    left_first,
                    left_second,
                )
                right_direction = _baseline_direction_in_first_camera(
                    right_first,
                    right_second,
                )
                if left_direction is not None and right_direction is not None:
                    translation_disagreements.append(
                        _direction_angle_degrees(
                            left_direction,
                            right_direction,
                        )
                    )

        observations.extend(
            (
                _observation(
                    RELATIVE_ROTATION_DISAGREEMENT_DESCRIPTOR,
                    _median(tuple(rotation_disagreements)),
                    provenance,
                ),
                _observation(
                    CONSENSUS_TRANSLATION_PAIR_COVERAGE_DESCRIPTOR,
                    len(translation_disagreements) / pair_count,
                    provenance,
                ),
            )
        )
        if translation_disagreements:
            observations.append(
                _observation(
                    RELATIVE_TRANSLATION_DIRECTION_DISAGREEMENT_DESCRIPTOR,
                    _median(tuple(translation_disagreements)),
                    provenance,
                )
            )

    return MetricVector(
        observations=tuple(
            sorted(
                observations,
                key=lambda observation: observation.descriptor.name.value,
            )
        )
    )


def evaluate_geometry_consensus(
    request: GeometryConsensusRequest,
) -> GeometryConsensusResult:
    """Compute symmetric pairwise disagreement evidence without alignment or selection."""

    if not isinstance(request, GeometryConsensusRequest):
        raise TypeError("request must be GeometryConsensusRequest")

    provenance = MetricProvenance(
        evaluator=GEOMETRY_CONSENSUS_EVALUATOR,
        input_artifacts=request.input_artifacts,
    )
    candidates_by_id = {
        candidate.geometry_solution_id: candidate for candidate in request.competing.candidates
    }
    disagreements = tuple(
        GeometryPairDisagreement(
            pair=pair,
            metrics=_pair_metrics(
                candidates_by_id[pair.left_geometry_solution_id],
                candidates_by_id[pair.right_geometry_solution_id],
                provenance=provenance,
            ),
        )
        for pair in derive_geometry_solution_pairs(request.competing)
    )
    return GeometryConsensusResult(
        request=request,
        pair_disagreements=disagreements,
    )
