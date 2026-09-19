from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.retrieval.gps import GpsPairingResult

GPS_PAIR_CANDIDATE_SOURCE_ID = PairCandidateSourceId("gps")


@dataclass(frozen=True, slots=True)
class GpsPairCandidateAdapterInput:
    """Explicit retained GPS donor result plus exact source-evidence identity."""

    result: GpsPairingResult
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.result, GpsPairingResult):
            raise TypeError("gps_pair_adapter.result must be GpsPairingResult")
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("gps_pair_adapter.evidence_ref must be ArtifactRef")


def adapt_gps_pairing_result(
    adapter_input: GpsPairCandidateAdapterInput,
) -> tuple[PairCandidate, ...]:
    """Translate an already-produced GPS donor result into canonical pair proposals."""

    if not isinstance(adapter_input, GpsPairCandidateAdapterInput):
        raise TypeError("adapter_input must be GpsPairCandidateAdapterInput")

    candidates = tuple(
        PairCandidate(
            observation_id1=candidate.observation_id1,
            observation_id2=candidate.observation_id2,
            sources=(
                PairCandidateSource(
                    source_id=GPS_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(adapter_input.evidence_ref,),
                ),
            ),
        )
        for candidate in adapter_input.result.candidates
    )
    return merge_pair_candidates(candidates)
