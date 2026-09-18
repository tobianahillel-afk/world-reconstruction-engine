from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from typing import Any, cast

import pytest

import wre.retrieval.sequential as sequential_donor_module
import wre.retrieval.sequential_adapter as sequential_adapter_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ObservationId,
    PairCandidate,
    PairCandidateSourceId,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.retrieval import (
    SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
    SEQUENTIAL_PAIRING_IMPLEMENTATION,
    SEQUENTIAL_PAIRING_VERSION,
    SequentialPairCandidateAdapterInput,
    SequentialPairingConfig,
    SequentialPairingResult,
    adapt_sequential_pairing_result,
    generate_sequential_candidates,
)


def _evidence_ref(value: str = "artifact:sequential-evidence") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(value),
        artifact_kind=ArtifactKind("pair.evidence.sequential"),
    )


def _donor_result(
    ordered_values: tuple[str, ...],
    *,
    overlap: int = 1,
    quadratic_overlap: bool = False,
) -> SequentialPairingResult:
    ordered_ids = tuple(ObservationId(value) for value in ordered_values)
    config = SequentialPairingConfig(
        overlap=overlap,
        quadratic_overlap=quadratic_overlap,
    )
    request = sequential_donor_module.SequentialPairingRequest(
        run=ReconstructionRun(
            run_id=ReconstructionRunId("run:v2l7.2-sequential-adapter"),
            producer=ProducerRef(
                implementation=SEQUENTIAL_PAIRING_IMPLEMENTATION,
                version=SEQUENTIAL_PAIRING_VERSION,
            ),
            input_observation_ids=ordered_ids,
            started_at=datetime(2026, 9, 18, 18, 5, tzinfo=UTC),
            configuration_sha256=config.sha256,
        ),
        ordered_observation_ids=ordered_ids,
        config=config,
    )
    return generate_sequential_candidates(request)


def test_adapter_input_is_exact_typed_immutable_value() -> None:
    result = _donor_result(("obs:a", "obs:b"))
    evidence_ref = _evidence_ref()
    adapter_input = SequentialPairCandidateAdapterInput(
        result=result,
        evidence_ref=evidence_ref,
    )

    assert tuple(field.name for field in fields(SequentialPairCandidateAdapterInput)) == (
        "result",
        "evidence_ref",
    )
    assert adapter_input.result is result
    assert adapter_input.evidence_ref is evidence_ref
    assert hash(adapter_input) == hash(
        SequentialPairCandidateAdapterInput(result=result, evidence_ref=evidence_ref)
    )
    with pytest.raises(FrozenInstanceError):
        adapter_input.evidence_ref = _evidence_ref("artifact:other")  # type: ignore[misc]

    with pytest.raises(TypeError, match="SequentialPairingResult"):
        SequentialPairCandidateAdapterInput(
            result=cast(Any, "result"),
            evidence_ref=evidence_ref,
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        SequentialPairCandidateAdapterInput(
            result=result,
            evidence_ref=cast(Any, "evidence"),
        )
    with pytest.raises(TypeError, match="SequentialPairCandidateAdapterInput"):
        adapt_sequential_pairing_result(cast(Any, result))


def test_sequential_source_identity_is_stable_and_canonical() -> None:
    assert SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID == PairCandidateSourceId("sequential")
    assert str(SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID) == "sequential"


def test_one_donor_pair_maps_to_one_canonical_pair_with_exact_evidence() -> None:
    result = _donor_result(("obs:a", "obs:b"))
    evidence_ref = _evidence_ref()

    adapted = adapt_sequential_pairing_result(
        SequentialPairCandidateAdapterInput(
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
                sequential_adapter_module.PairCandidateSource(
                    source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(evidence_ref,),
                ),
            ),
        ),
    )


def test_multi_pair_output_is_pair_lexical_while_donor_result_is_unchanged() -> None:
    result = _donor_result(("obs:z", "obs:a", "obs:m"))
    before = result

    adapted = adapt_sequential_pairing_result(
        SequentialPairCandidateAdapterInput(
            result=result,
            evidence_ref=_evidence_ref(),
        )
    )

    assert result == before
    assert tuple(
        (candidate.observation_id1.value, candidate.observation_id2.value)
        for candidate in result.candidates
    ) == (("obs:a", "obs:z"), ("obs:a", "obs:m"))
    assert tuple(
        (candidate.observation_id1.value, candidate.observation_id2.value)
        for candidate in adapted
    ) == (("obs:a", "obs:m"), ("obs:a", "obs:z"))


def test_empty_donor_candidate_result_maps_to_empty_tuple() -> None:
    result = _donor_result(("obs:only",))

    assert result.candidates == ()
    assert (
        adapt_sequential_pairing_result(
            SequentialPairCandidateAdapterInput(
                result=result,
                evidence_ref=_evidence_ref(),
            )
        )
        == ()
    )


def test_adaptation_is_repeatable_and_evidence_ref_is_the_only_changed_source_link() -> None:
    result = _donor_result(("obs:a", "obs:b", "obs:c"), overlap=2)
    first_ref = _evidence_ref("artifact:first")
    second_ref = _evidence_ref("artifact:second")
    first_input = SequentialPairCandidateAdapterInput(
        result=result,
        evidence_ref=first_ref,
    )

    first = adapt_sequential_pairing_result(first_input)
    repeated = adapt_sequential_pairing_result(first_input)
    changed_evidence = adapt_sequential_pairing_result(
        SequentialPairCandidateAdapterInput(
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


def test_sequential_specific_values_remain_only_on_donor_result() -> None:
    result = _donor_result(("obs:a", "obs:b", "obs:c"), overlap=2)
    adapted = adapt_sequential_pairing_result(
        SequentialPairCandidateAdapterInput(
            result=result,
            evidence_ref=_evidence_ref(),
        )
    )

    assert result.configuration_sha256 is not None
    assert result.colmap_reference_version == "4.2.0"
    assert result.provenance.producing_run_id == ReconstructionRunId(
        "run:v2l7.2-sequential-adapter"
    )
    assert result.candidates[0].sequence_index1 == 0
    assert result.candidates[0].sequence_index2 == 1
    assert result.candidates[0].sequence_distance == 1

    assert tuple(field.name for field in fields(PairCandidate)) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    assert all(len(candidate.sources) == 1 for candidate in adapted)


def test_adapter_never_invokes_donor_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _donor_result(("obs:a", "obs:b"))
    evidence_ref = _evidence_ref()

    def _unexpected(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"adapter invoked donor generation: {args} {kwargs}")

    monkeypatch.setattr(
        sequential_donor_module,
        "generate_sequential_candidates",
        _unexpected,
    )

    adapted = adapt_sequential_pairing_result(
        SequentialPairCandidateAdapterInput(
            result=result,
            evidence_ref=evidence_ref,
        )
    )

    assert len(adapted) == 1


def test_adapter_module_has_no_gps_visual_or_future_truth_surface() -> None:
    for name in (
        "GpsPairCandidate",
        "GpsPairingResult",
        "generate_gps_candidates",
        "MetricVector",
        "QualityDecision",
        "SceneCluster",
        "CorrespondenceSet",
        "RouteGraph",
        "GeometrySolution",
        "JobSpec",
        "scheduler",
    ):
        assert name not in sequential_adapter_module.__dict__
