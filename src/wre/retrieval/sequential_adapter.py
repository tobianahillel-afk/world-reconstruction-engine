from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.retrieval.sequential import SequentialPairingResult

SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID = PairCandidateSourceId("sequential")


@dataclass(frozen=True, slots=True)
class SequentialPairCandidateAdapterInput:
    """Explicit retained-donor result plus exact source-evidence identity."""

    result: SequentialPairingResult
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.result, SequentialPairingResult):
            raise TypeError("sequential_pair_adapter.result must be SequentialPairingResult")
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("sequential_pair_adapter.evidence_ref must be ArtifactRef")


def adapt_sequential_pairing_result(
    adapter_input: SequentialPairCandidateAdapterInput,
) -> tuple[PairCandidate, ...]:
    """Translate an already-produced sequential donor result into canonical pair proposals."""

    if not isinstance(adapter_input, SequentialPairCandidateAdapterInput):
        raise TypeError("adapter_input must be SequentialPairCandidateAdapterInput")

    candidates = tuple(
        PairCandidate(
            observation_id1=candidate.observation_id1,
            observation_id2=candidate.observation_id2,
            sources=(
                PairCandidateSource(
                    source_id=SEQUENTIAL_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(adapter_input.evidence_ref,),
                ),
            ),
        )
        for candidate in adapter_input.result.candidates
    )
    return merge_pair_candidates(candidates)
