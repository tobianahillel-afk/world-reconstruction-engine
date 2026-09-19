from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from typing import Any, cast

import pytest

import wre.retrieval.gps as gps_donor_module
import wre.retrieval.gps_adapter as gps_adapter_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ObservationId,
    PairCandidate,
    PairCandidateSourceId,
)
from wre.domain.metadata import (
    CaptureTimeInterpretation,
    CaptureTimeInterpretationStatus,
    GpsInterpretationStatus,
    GpsMetadataInterpretation,
    ObservationMetadataInterpretation,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.retrieval import (
    GPS_DISTANCE_MODEL,
    GPS_MAP_DATUM_POLICY,
    GPS_PAIR_CANDIDATE_SOURCE_ID,
    GPS_PAIRING_IMPLEMENTATION,
    GPS_PAIRING_VERSION,
    GpsPairCandidateAdapterInput,
    GpsPairingConfig,
    GpsPairingEligibilityStatus,
    GpsPairingRequest,
    GpsPairingResult,
    adapt_gps_pairing_result,
    generate_gps_candidates,
)


def _evidence_ref(value: str = "artifact:gps-evidence") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(value),
        artifact_kind=ArtifactKind("pair.evidence.gps"),
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
        capture_time=CaptureTimeInterpretation(
            status=CaptureTimeInterpretationStatus.ABSENT
        ),
    )


def _gps_result(
    interpretations: tuple[ObservationMetadataInterpretation, ...],
    *,
    config: GpsPairingConfig | None = None,
) -> GpsPairingResult:
    actual_config = config or GpsPairingConfig(max_distance_m=1_000.0)
    observation_ids = tuple(
        sorted((item.observation_id for item in interpretations), key=lambda item: item.value)
    )
    request = GpsPairingRequest(
        run=ReconstructionRun(
            run_id=ReconstructionRunId("run:v2l7.3-gps-adapter"),
            producer=ProducerRef(
                implementation=GPS_PAIRING_IMPLEMENTATION,
                version=GPS_PAIRING_VERSION,
            ),
            input_observation_ids=observation_ids,
            started_at=datetime(2026, 9, 19, 20, 20, tzinfo=UTC),
            configuration_sha256=actual_config.sha256,
        ),
        interpretations=interpretations,
        config=actual_config,
    )
    return generate_gps_candidates(request)


def test_adapter_input_is_exact_typed_immutable_value() -> None:
    result = _gps_result(
        (
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
        )
    )
    evidence_ref = _evidence_ref()
    adapter_input = GpsPairCandidateAdapterInput(
        result=result,
        evidence_ref=evidence_ref,
    )

    assert tuple(field.name for field in fields(GpsPairCandidateAdapterInput)) == (
        "result",
        "evidence_ref",
    )
    assert adapter_input.result is result
    assert adapter_input.evidence_ref is evidence_ref
    with pytest.raises(FrozenInstanceError):
        adapter_input.evidence_ref = _evidence_ref("artifact:other")  # type: ignore[misc]

    with pytest.raises(TypeError, match="GpsPairingResult"):
        GpsPairCandidateAdapterInput(
            result=cast(Any, "result"),
            evidence_ref=evidence_ref,
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=cast(Any, "evidence"),
        )
    with pytest.raises(TypeError, match="GpsPairCandidateAdapterInput"):
        adapt_gps_pairing_result(cast(Any, result))


def test_gps_source_identity_is_stable_and_canonical() -> None:
    assert GPS_PAIR_CANDIDATE_SOURCE_ID == PairCandidateSourceId("gps")
    assert str(GPS_PAIR_CANDIDATE_SOURCE_ID) == "gps"


def test_one_donor_pair_maps_to_one_canonical_pair_with_exact_evidence() -> None:
    result = _gps_result(
        (
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
        )
    )
    evidence_ref = _evidence_ref()

    adapted = adapt_gps_pairing_result(
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=evidence_ref,
        )
    )

    assert len(result.candidates) == 1
    assert adapted == (
        PairCandidate(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            sources=(
                gps_adapter_module.PairCandidateSource(
                    source_id=GPS_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(evidence_ref,),
                ),
            ),
        ),
    )


def test_multi_pair_adaptation_preserves_donor_result_and_eligibility() -> None:
    result = _gps_result(
        (
            _resolved("obs:c", latitude_deg=48.8568, longitude_deg=2.3524),
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
        ),
        config=GpsPairingConfig(max_num_neighbors=2, max_distance_m=1_000.0),
    )
    before = result

    adapted = adapt_gps_pairing_result(
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=_evidence_ref(),
        )
    )

    assert result == before
    assert tuple(item.status for item in result.eligibility) == (
        GpsPairingEligibilityStatus.ELIGIBLE,
        GpsPairingEligibilityStatus.ELIGIBLE,
        GpsPairingEligibilityStatus.ELIGIBLE,
    )
    assert tuple(
        (candidate.observation_id1.value, candidate.observation_id2.value)
        for candidate in adapted
    ) == (("obs:a", "obs:b"), ("obs:a", "obs:c"), ("obs:b", "obs:c"))


def test_valid_no_candidate_result_maps_to_empty_tuple_without_losing_eligibility() -> None:
    result = _gps_result(
        (_resolved("obs:single", latitude_deg=48.8566, longitude_deg=2.3522),)
    )

    assert len(result.eligibility) == 1
    assert result.eligibility[0].status is GpsPairingEligibilityStatus.ELIGIBLE
    assert result.candidates == ()
    assert (
        adapt_gps_pairing_result(
            GpsPairCandidateAdapterInput(
                result=result,
                evidence_ref=_evidence_ref(),
            )
        )
        == ()
    )
    assert result.eligibility[0].status is GpsPairingEligibilityStatus.ELIGIBLE


def test_adaptation_is_repeatable_and_evidence_ref_is_the_only_changed_source_link() -> None:
    result = _gps_result(
        (
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
            _resolved("obs:c", latitude_deg=48.8568, longitude_deg=2.3524),
        )
    )
    first_ref = _evidence_ref("artifact:first")
    second_ref = _evidence_ref("artifact:second")
    first_input = GpsPairCandidateAdapterInput(
        result=result,
        evidence_ref=first_ref,
    )

    first = adapt_gps_pairing_result(first_input)
    repeated = adapt_gps_pairing_result(first_input)
    changed_evidence = adapt_gps_pairing_result(
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=second_ref,
        )
    )

    assert first == repeated
    assert tuple(
        (candidate.observation_id1, candidate.observation_id2)
        for candidate in changed_evidence
    ) == tuple(
        (candidate.observation_id1, candidate.observation_id2)
        for candidate in first
    )
    assert all(
        candidate.sources[0].evidence_refs == (first_ref,)
        for candidate in first
    )
    assert all(
        candidate.sources[0].evidence_refs == (second_ref,)
        for candidate in changed_evidence
    )


def test_gps_specific_values_remain_only_on_donor_result() -> None:
    result = _gps_result(
        (
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
        ),
        config=GpsPairingConfig(max_num_neighbors=1, max_distance_m=250.0),
    )
    adapted = adapt_gps_pairing_result(
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=_evidence_ref(),
        )
    )

    assert result.configuration_sha256 is not None
    assert result.colmap_reference_version == "4.2.0"
    assert result.provenance.producing_run_id == ReconstructionRunId(
        "run:v2l7.3-gps-adapter"
    )
    assert result.candidates[0].distance_m > 0.0
    assert result.eligibility[0].source_gps_status is GpsInterpretationStatus.RESOLVED
    assert result.eligibility[0].map_datum == "WGS-84"
    assert GPS_DISTANCE_MODEL == "wgs84_ecef_chord_altitude_zero"
    assert GPS_MAP_DATUM_POLICY == "explicit_wgs84_only"

    assert tuple(field.name for field in fields(PairCandidate)) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    assert all(len(candidate.sources) == 1 for candidate in adapted)


def test_adapter_never_invokes_gps_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _gps_result(
        (
            _resolved("obs:a", latitude_deg=48.8566, longitude_deg=2.3522),
            _resolved("obs:b", latitude_deg=48.8567, longitude_deg=2.3523),
        )
    )
    evidence_ref = _evidence_ref()

    def _unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"adapter invoked GPS generation: {args} {kwargs}")

    monkeypatch.setattr(
        gps_donor_module,
        "generate_gps_candidates",
        _unexpected,
    )

    adapted = adapt_gps_pairing_result(
        GpsPairCandidateAdapterInput(
            result=result,
            evidence_ref=evidence_ref,
        )
    )

    assert len(adapted) == 1


def test_adapter_module_has_no_visual_or_future_truth_surface() -> None:
    for name in (
        "generate_gps_candidates",
        "SequentialPairCandidate",
        "MetricVector",
        "QualityDecision",
        "SceneCluster",
        "CorrespondenceSet",
        "RouteGraph",
        "GeometrySolution",
        "JobSpec",
        "scheduler",
    ):
        assert name not in gps_adapter_module.__dict__
