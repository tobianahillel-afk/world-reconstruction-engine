from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId

_SOURCE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True, order=True)
class PairCandidateSourceId:
    """Stable solver-independent identity for one pair-proposal source family."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("pair_candidate_source_id.value must be str")
        if _SOURCE_ID_RE.fullmatch(self.value) is None:
            raise ValueError(
                "pair_candidate_source_id must be 1-128 lowercase characters using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


@dataclass(frozen=True, slots=True)
class PairCandidateSource:
    """One proposal source plus exact references to its source-specific evidence."""

    source_id: PairCandidateSourceId
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, PairCandidateSourceId):
            raise TypeError("pair_candidate_source.source_id must be PairCandidateSourceId")
        if not isinstance(self.evidence_refs, tuple):
            raise TypeError("pair_candidate_source.evidence_refs must be an immutable tuple")
        if not self.evidence_refs:
            raise ValueError("pair_candidate_source.evidence_refs must not be empty")
        if any(not isinstance(ref, ArtifactRef) for ref in self.evidence_refs):
            raise TypeError("pair_candidate_source.evidence_refs must contain only ArtifactRef")

        keys = tuple(_artifact_ref_key(ref) for ref in self.evidence_refs)
        if len(keys) != len(set(keys)):
            raise ValueError("pair_candidate_source.evidence_refs must not contain duplicates")
        if keys != tuple(sorted(keys)):
            raise ValueError(
                "pair_candidate_source.evidence_refs must use canonical artifact order"
            )


@dataclass(frozen=True, slots=True)
class PairCandidate:
    """One unordered observation pair proposed for later relationship verification."""

    observation_id1: ObservationId
    observation_id2: ObservationId
    sources: tuple[PairCandidateSource, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("pair_candidate.observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("pair_candidate.observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "pair_candidate observation IDs must be distinct and in canonical ascending order"
            )
        if not isinstance(self.sources, tuple):
            raise TypeError("pair_candidate.sources must be an immutable tuple")
        if not self.sources:
            raise ValueError("pair_candidate.sources must not be empty")
        if any(not isinstance(source, PairCandidateSource) for source in self.sources):
            raise TypeError("pair_candidate.sources must contain only PairCandidateSource")

        source_ids = tuple(source.source_id.value for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("pair_candidate.sources must not contain duplicate source IDs")
        if source_ids != tuple(sorted(source_ids)):
            raise ValueError("pair_candidate.sources must use canonical source-ID order")


def merge_pair_candidates(
    candidates: tuple[PairCandidate, ...],
) -> tuple[PairCandidate, ...]:
    """Deduplicate canonical pairs while preserving every source-evidence reference."""

    if not isinstance(candidates, tuple):
        raise TypeError("pair_candidates must be an immutable tuple")
    if any(not isinstance(candidate, PairCandidate) for candidate in candidates):
        raise TypeError("pair_candidates must contain only PairCandidate values")
    if not candidates:
        return ()

    pair_sources: dict[
        tuple[str, str],
        dict[str, dict[tuple[str, str], ArtifactRef]],
    ] = {}
    endpoints: dict[tuple[str, str], tuple[ObservationId, ObservationId]] = {}

    for candidate in candidates:
        pair_key = (
            candidate.observation_id1.value,
            candidate.observation_id2.value,
        )
        endpoints.setdefault(
            pair_key,
            (candidate.observation_id1, candidate.observation_id2),
        )
        source_map = pair_sources.setdefault(pair_key, {})
        for source in candidate.sources:
            evidence_map = source_map.setdefault(source.source_id.value, {})
            for ref in source.evidence_refs:
                evidence_map.setdefault(_artifact_ref_key(ref), ref)

    merged: list[PairCandidate] = []
    for pair_key in sorted(pair_sources):
        source_map = pair_sources[pair_key]
        sources = tuple(
            PairCandidateSource(
                source_id=PairCandidateSourceId(source_id),
                evidence_refs=tuple(
                    source_map[source_id][evidence_key]
                    for evidence_key in sorted(source_map[source_id])
                ),
            )
            for source_id in sorted(source_map)
        )
        observation_id1, observation_id2 = endpoints[pair_key]
        merged.append(
            PairCandidate(
                observation_id1=observation_id1,
                observation_id2=observation_id2,
                sources=sources,
            )
        )

    return tuple(merged)
