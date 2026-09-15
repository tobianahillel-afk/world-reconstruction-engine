from __future__ import annotations

from datetime import UTC, datetime

import pytest

from wre.domain.observations import ObservationId
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.retrieval import (
    COLMAP_SEQUENTIAL_REFERENCE_VERSION,
    SEQUENTIAL_PAIRING_IMPLEMENTATION,
    SEQUENTIAL_PAIRING_VERSION,
    SequentialPairCandidate,
    SequentialPairingConfig,
    SequentialPairingRequest,
    generate_sequential_candidates,
)


def _ids(*values: str) -> tuple[ObservationId, ...]:
    return tuple(ObservationId(value) for value in values)


def _request(
    ordered_ids: tuple[ObservationId, ...],
    *,
    config: SequentialPairingConfig | None = None,
) -> SequentialPairingRequest:
    actual_config = config or SequentialPairingConfig()
    return SequentialPairingRequest(
        run=ReconstructionRun(
            run_id=ReconstructionRunId("run:l4.1-test"),
            producer=ProducerRef(
                implementation=SEQUENTIAL_PAIRING_IMPLEMENTATION,
                version=SEQUENTIAL_PAIRING_VERSION,
            ),
            input_observation_ids=ordered_ids,
            started_at=datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
            configuration_sha256=actual_config.sha256,
        ),
        ordered_observation_ids=ordered_ids,
        config=actual_config,
    )


def _sequence_edges(
    candidates: tuple[SequentialPairCandidate, ...],
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (
            min(candidate.sequence_index1, candidate.sequence_index2),
            max(candidate.sequence_index1, candidate.sequence_index2),
        )
        for candidate in candidates
    )


def test_config_is_canonical_and_limits_l41_scope() -> None:
    config = SequentialPairingConfig()

    assert config.sha256 == SequentialPairingConfig().sha256
    assert config.overlap == 10
    assert config.quadratic_overlap is True
    assert config.expand_rig_images is False
    assert config.loop_detection is False

    with pytest.raises(ValueError, match="positive integer"):
        SequentialPairingConfig(overlap=0)
    with pytest.raises(ValueError, match="rig expansion"):
        SequentialPairingConfig(expand_rig_images=True)
    with pytest.raises(ValueError, match="loop detection"):
        SequentialPairingConfig(loop_detection=True)


def test_linear_overlap_matches_colmap_420_non_rig_semantics() -> None:
    ordered_ids = _ids("obs:z", "obs:a", "obs:m", "obs:b", "obs:y")
    config = SequentialPairingConfig(overlap=3, quadratic_overlap=False)

    result = generate_sequential_candidates(_request(ordered_ids, config=config))

    assert result.colmap_reference_version == COLMAP_SEQUENTIAL_REFERENCE_VERSION
    assert result.ordered_observation_ids == ordered_ids
    assert _sequence_edges(result.candidates) == (
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    )
    assert tuple(candidate.sequence_distance for candidate in result.candidates) == (
        1,
        2,
        3,
        1,
        2,
        3,
        1,
        2,
        1,
    )


def test_quadratic_overlap_matches_colmap_420_offsets() -> None:
    ordered_ids = _ids("obs:z", "obs:a", "obs:m", "obs:b", "obs:y")
    config = SequentialPairingConfig(overlap=3, quadratic_overlap=True)

    result = generate_sequential_candidates(_request(ordered_ids, config=config))

    assert _sequence_edges(result.candidates) == (
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 2),
        (1, 3),
        (2, 3),
        (2, 4),
        (3, 4),
    )
    assert tuple(candidate.sequence_distance for candidate in result.candidates) == (
        1,
        2,
        4,
        1,
        2,
        1,
        2,
        1,
    )


@pytest.mark.parametrize("quadratic_overlap", [False, True])
def test_overlap_work_is_bounded_by_actual_sequence_length(quadratic_overlap: bool) -> None:
    ordered_ids = _ids("obs:a", "obs:b", "obs:c")
    config = SequentialPairingConfig(
        overlap=1_000_000,
        quadratic_overlap=quadratic_overlap,
    )

    result = generate_sequential_candidates(_request(ordered_ids, config=config))

    assert result.candidate_count == 3
    assert _sequence_edges(result.candidates) == ((0, 1), (0, 2), (1, 2))


def test_explicit_sequence_order_is_not_replaced_by_observation_id_sorting() -> None:
    ordered_ids = _ids("obs:z", "obs:a", "obs:m")
    config = SequentialPairingConfig(overlap=1, quadratic_overlap=False)

    result = generate_sequential_candidates(_request(ordered_ids, config=config))

    assert _sequence_edges(result.candidates) == ((0, 1), (1, 2))
    first = result.candidates[0]
    assert (first.observation_id1, first.observation_id2) == (
        ObservationId("obs:a"),
        ObservationId("obs:z"),
    )
    assert (first.sequence_index1, first.sequence_index2) == (1, 0)
    assert result.provenance.source_observation_ids == tuple(
        sorted(ordered_ids, key=lambda item: item.value)
    )


def test_single_observation_produces_no_candidate_without_forcing_a_pair() -> None:
    ordered_ids = _ids("obs:only")

    result = generate_sequential_candidates(_request(ordered_ids))

    assert result.candidate_count == 0
    assert result.candidates == ()


def test_request_rejects_duplicate_or_mismatched_sequence_membership() -> None:
    config = SequentialPairingConfig(overlap=1, quadratic_overlap=False)
    observation_a = ObservationId("obs:a")
    observation_b = ObservationId("obs:b")
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.1-membership"),
        producer=ProducerRef(
            implementation=SEQUENTIAL_PAIRING_IMPLEMENTATION,
            version=SEQUENTIAL_PAIRING_VERSION,
        ),
        input_observation_ids=(observation_a, observation_b),
        started_at=datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )

    with pytest.raises(ValueError, match="cannot contain duplicates"):
        SequentialPairingRequest(
            run=run,
            ordered_observation_ids=(observation_a, observation_a),
            config=config,
        )
    with pytest.raises(ValueError, match="exactly match"):
        SequentialPairingRequest(
            run=run,
            ordered_observation_ids=(observation_a,),
            config=config,
        )


def test_request_rejects_wrong_producer_or_configuration_identity() -> None:
    ordered_ids = _ids("obs:a", "obs:b")
    config = SequentialPairingConfig(overlap=1, quadratic_overlap=False)
    wrong_config = SequentialPairingConfig(overlap=2, quadratic_overlap=False)

    wrong_producer_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.1-wrong-producer"),
        producer=ProducerRef(implementation="other", version=SEQUENTIAL_PAIRING_VERSION),
        input_observation_ids=ordered_ids,
        started_at=datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    with pytest.raises(ValueError, match="producer must be"):
        SequentialPairingRequest(
            run=wrong_producer_run,
            ordered_observation_ids=ordered_ids,
            config=config,
        )

    wrong_config_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.1-wrong-config"),
        producer=ProducerRef(
            implementation=SEQUENTIAL_PAIRING_IMPLEMENTATION,
            version=SEQUENTIAL_PAIRING_VERSION,
        ),
        input_observation_ids=ordered_ids,
        started_at=datetime(2026, 9, 15, 14, 0, tzinfo=UTC),
        configuration_sha256=wrong_config.sha256,
    )
    with pytest.raises(ValueError, match="configuration SHA-256"):
        SequentialPairingRequest(
            run=wrong_config_run,
            ordered_observation_ids=ordered_ids,
            config=config,
        )
