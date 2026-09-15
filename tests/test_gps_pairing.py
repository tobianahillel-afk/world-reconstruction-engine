from __future__ import annotations

import math
import os
from datetime import UTC, datetime

import pytest

from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
)
from wre.domain.observations import ObservationId
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.retrieval.gps import (
    COLMAP_SPATIAL_REFERENCE_VERSION,
    GPS_PAIRING_IMPLEMENTATION,
    GPS_PAIRING_VERSION,
    GpsPairingConfig,
    GpsPairingEligibilityStatus,
    GpsPairingRequest,
    generate_gps_candidates,
)


def _resolved(
    observation_id: str,
    *,
    latitude_deg: float,
    longitude_deg: float,
    map_datum: str | None = "WGS-84",
) -> ObservationMetadataInterpretation:
    return ObservationMetadataInterpretation(
        observation_id=ObservationId(observation_id),
        gps=GpsMetadataInterpretation(
            status=GpsInterpretationStatus.RESOLVED,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
            map_datum=map_datum,
            evidence_keys=("GPS GPSLatitude", "GPS GPSLongitude"),
        ),
        capture_time=CaptureTimeInterpretation(status=CaptureTimeInterpretationStatus.ABSENT),
    )


def _unresolved(
    observation_id: str,
    status: GpsInterpretationStatus,
) -> ObservationMetadataInterpretation:
    if status is GpsInterpretationStatus.ABSENT:
        gps = GpsMetadataInterpretation(status=status)
    else:
        gps = GpsMetadataInterpretation(
            status=status,
            map_datum="WGS-84" if status is GpsInterpretationStatus.INCOMPLETE else None,
            evidence_keys=("GPS GPSLatitude",),
            issue="test unresolved GPS evidence",
        )
    return ObservationMetadataInterpretation(
        observation_id=ObservationId(observation_id),
        gps=gps,
        capture_time=CaptureTimeInterpretation(status=CaptureTimeInterpretationStatus.ABSENT),
    )


def _request(
    interpretations: tuple[ObservationMetadataInterpretation, ...],
    *,
    config: GpsPairingConfig | None = None,
) -> GpsPairingRequest:
    actual_config = config or GpsPairingConfig()
    observation_ids = tuple(
        sorted((item.observation_id for item in interpretations), key=lambda item: item.value)
    )
    return GpsPairingRequest(
        run=ReconstructionRun(
            run_id=ReconstructionRunId("run:l4.2-test"),
            producer=ProducerRef(
                implementation=GPS_PAIRING_IMPLEMENTATION,
                version=GPS_PAIRING_VERSION,
            ),
            input_observation_ids=observation_ids,
            started_at=datetime(2026, 9, 15, 14, 45, tzinfo=UTC),
            configuration_sha256=actual_config.sha256,
        ),
        interpretations=interpretations,
        config=actual_config,
    )


def _pair_ids(result: object) -> tuple[tuple[str, str], ...]:
    candidates = getattr(result, "candidates")
    return tuple(
        (candidate.observation_id1.value, candidate.observation_id2.value)
        for candidate in candidates
    )


def test_config_is_canonical_and_rejects_unsafe_values() -> None:
    config = GpsPairingConfig()

    assert config.sha256 == GpsPairingConfig().sha256
    assert config.max_num_neighbors == 50
    assert config.max_distance_m == 100.0
    assert config.canonical_document()["min_num_neighbors"] == 0

    with pytest.raises(ValueError, match="positive integer"):
        GpsPairingConfig(max_num_neighbors=0)
    with pytest.raises(ValueError, match="finite and positive"):
        GpsPairingConfig(max_distance_m=math.inf)
    with pytest.raises(ValueError, match="finite and positive"):
        GpsPairingConfig(max_distance_m=0.0)


def test_only_resolved_explicit_wgs84_positions_are_eligible() -> None:
    interpretations = (
        _resolved("obs:wgs84", latitude_deg=48.8566, longitude_deg=2.3522),
        _resolved(
            "obs:wgs84-alias",
            latitude_deg=48.85661,
            longitude_deg=2.35221,
            map_datum="WGS 84",
        ),
        _resolved(
            "obs:missing-datum",
            latitude_deg=48.85662,
            longitude_deg=2.35222,
            map_datum=None,
        ),
        _resolved(
            "obs:unsupported-datum",
            latitude_deg=48.85663,
            longitude_deg=2.35223,
            map_datum="NAD83",
        ),
        _unresolved("obs:absent", GpsInterpretationStatus.ABSENT),
        _unresolved("obs:incomplete", GpsInterpretationStatus.INCOMPLETE),
        _unresolved("obs:invalid", GpsInterpretationStatus.INVALID),
    )

    result = generate_gps_candidates(_request(interpretations))

    statuses = {item.observation_id.value: item.status for item in result.eligibility}
    assert statuses == {
        "obs:absent": GpsPairingEligibilityStatus.GPS_ABSENT,
        "obs:incomplete": GpsPairingEligibilityStatus.GPS_INCOMPLETE,
        "obs:invalid": GpsPairingEligibilityStatus.GPS_INVALID,
        "obs:missing-datum": GpsPairingEligibilityStatus.MAP_DATUM_MISSING,
        "obs:unsupported-datum": GpsPairingEligibilityStatus.MAP_DATUM_UNSUPPORTED,
        "obs:wgs84": GpsPairingEligibilityStatus.ELIGIBLE,
        "obs:wgs84-alias": GpsPairingEligibilityStatus.ELIGIBLE,
    }
    assert result.eligible_observation_count == 2
    assert _pair_ids(result) == (("obs:wgs84", "obs:wgs84-alias"),)


def test_radius_filter_uses_wgs84_ecef_distance_without_forcing_work() -> None:
    interpretations = (
        _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
        _resolved("obs:b", latitude_deg=48.8566, longitude_deg=2.3527),
        _resolved("obs:far", latitude_deg=48.8666, longitude_deg=2.3522),
    )
    config = GpsPairingConfig(max_distance_m=100.0)

    result = generate_gps_candidates(_request(interpretations, config=config))

    assert _pair_ids(result) == (("obs:a", "obs:b"),)
    assert 0.0 < result.candidates[0].distance_m < 100.0


def test_antimeridian_neighbors_are_not_rejected_by_naive_longitude_difference() -> None:
    interpretations = (
        _resolved("obs:east", latitude_deg=0.0, longitude_deg=179.9995),
        _resolved("obs:west", latitude_deg=0.0, longitude_deg=-179.9995),
    )
    config = GpsPairingConfig(max_distance_m=150.0)

    result = generate_gps_candidates(_request(interpretations, config=config))

    assert _pair_ids(result) == (("obs:east", "obs:west"),)
    assert result.candidates[0].distance_m == pytest.approx(111.31949, rel=1e-5)


def test_neighbor_cap_is_deterministic_and_pair_identity_is_canonical() -> None:
    interpretations = (
        _resolved("obs:c", latitude_deg=0.0, longitude_deg=0.0002),
        _resolved("obs:a", latitude_deg=0.0, longitude_deg=0.0),
        _resolved("obs:b", latitude_deg=0.0, longitude_deg=0.0001),
    )
    config = GpsPairingConfig(max_num_neighbors=1, max_distance_m=100.0)

    first = generate_gps_candidates(_request(interpretations, config=config))
    second = generate_gps_candidates(_request(tuple(reversed(interpretations)), config=config))

    assert first == second
    assert _pair_ids(first) == (("obs:a", "obs:b"), ("obs:b", "obs:c"))


def test_identical_positions_can_propose_a_zero_distance_candidate() -> None:
    interpretations = (
        _resolved("obs:a", latitude_deg=35.0, longitude_deg=139.0),
        _resolved("obs:b", latitude_deg=35.0, longitude_deg=139.0),
    )

    result = generate_gps_candidates(_request(interpretations))

    assert result.candidate_count == 1
    assert result.candidates[0].distance_m == 0.0


def test_single_or_ineligible_observation_can_yield_no_candidate() -> None:
    single = generate_gps_candidates(
        _request((_resolved("obs:single", latitude_deg=1.0, longitude_deg=2.0),))
    )
    missing_datum = generate_gps_candidates(
        _request(
            (
                _resolved(
                    "obs:missing",
                    latitude_deg=1.0,
                    longitude_deg=2.0,
                    map_datum=None,
                ),
            )
        )
    )

    assert single.candidates == ()
    assert missing_datum.candidates == ()
    assert missing_datum.eligible_observation_count == 0


def test_request_rejects_duplicate_membership_and_run_identity_mismatch() -> None:
    interpretation = _resolved("obs:a", latitude_deg=0.0, longitude_deg=0.0)
    config = GpsPairingConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.2-membership"),
        producer=ProducerRef(
            implementation=GPS_PAIRING_IMPLEMENTATION,
            version=GPS_PAIRING_VERSION,
        ),
        input_observation_ids=(interpretation.observation_id,),
        started_at=datetime(2026, 9, 15, 14, 45, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )

    with pytest.raises(ValueError, match="duplicate observation IDs"):
        GpsPairingRequest(
            run=run,
            interpretations=(interpretation, interpretation),
            config=config,
        )

    other = _resolved("obs:b", latitude_deg=0.0, longitude_deg=0.0)
    with pytest.raises(ValueError, match="exactly match"):
        GpsPairingRequest(
            run=run,
            interpretations=(other,),
            config=config,
        )


def test_request_rejects_wrong_producer_or_configuration_identity() -> None:
    interpretation = _resolved("obs:a", latitude_deg=0.0, longitude_deg=0.0)
    config = GpsPairingConfig(max_distance_m=100.0)
    wrong_config = GpsPairingConfig(max_distance_m=200.0)

    wrong_producer_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.2-wrong-producer"),
        producer=ProducerRef(implementation="other", version=GPS_PAIRING_VERSION),
        input_observation_ids=(interpretation.observation_id,),
        started_at=datetime(2026, 9, 15, 14, 45, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    with pytest.raises(ValueError, match="producer must be"):
        GpsPairingRequest(
            run=wrong_producer_run,
            interpretations=(interpretation,),
            config=config,
        )

    wrong_config_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l4.2-wrong-config"),
        producer=ProducerRef(
            implementation=GPS_PAIRING_IMPLEMENTATION,
            version=GPS_PAIRING_VERSION,
        ),
        input_observation_ids=(interpretation.observation_id,),
        started_at=datetime(2026, 9, 15, 14, 45, tzinfo=UTC),
        configuration_sha256=wrong_config.sha256,
    )
    with pytest.raises(ValueError, match="configuration SHA-256"):
        GpsPairingRequest(
            run=wrong_config_run,
            interpretations=(interpretation,),
            config=config,
        )


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="real PyCOLMAP GPS equivalence runs only in the dedicated COLMAP lane",
)
def test_real_pycolmap_420_explicit_wgs84_distance_equivalence() -> None:
    pycolmap = pytest.importorskip("pycolmap")
    assert pycolmap.__version__ == COLMAP_SPATIAL_REFERENCE_VERSION

    first = (48.8566, 2.3522, 0.0)
    second = (48.8570, 2.3528, 0.0)
    transform = pycolmap.GPSTransform(pycolmap.GPSTransformEllipsoid.WGS84)
    upstream_ecef = transform.ellipsoid_to_ecef([first, second])
    upstream_distance_m = math.dist(upstream_ecef[0], upstream_ecef[1])

    interpretations = (
        _resolved("obs:a", latitude_deg=first[0], longitude_deg=first[1]),
        _resolved("obs:b", latitude_deg=second[0], longitude_deg=second[1]),
    )
    config = GpsPairingConfig(max_distance_m=1_000.0)
    result = generate_gps_candidates(_request(interpretations, config=config))

    assert result.candidate_count == 1
    assert result.candidates[0].distance_m == pytest.approx(upstream_distance_m, abs=1e-8)
