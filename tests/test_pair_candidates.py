from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.domain.pair_candidates as pair_candidates_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ObservationId,
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.retrieval import (
    GpsPairCandidate,
    SequentialPairCandidate,
    generate_gps_candidates,
    generate_sequential_candidates,
)


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _source(
    source_id: str,
    *refs: ArtifactRef,
) -> PairCandidateSource:
    return PairCandidateSource(
        source_id=PairCandidateSourceId(source_id),
        evidence_refs=tuple(refs),
    )


def _pair(
    first: str,
    second: str,
    *sources: PairCandidateSource,
) -> PairCandidate:
    return PairCandidate(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        sources=tuple(sources),
    )


def test_source_id_is_exact_immutable_hashable_lowercase_token() -> None:
    source_id = PairCandidateSourceId("visual.place-v1")

    assert str(source_id) == "visual.place-v1"
    assert hash(source_id) == hash(PairCandidateSourceId("visual.place-v1"))
    with pytest.raises(FrozenInstanceError):
        source_id.value = "other"  # type: ignore[misc]

    for value in (
        "",
        "Sequential",
        " visual",
        "visual ",
        "visual/source",
        "a" * 129,
    ):
        with pytest.raises(ValueError):
            PairCandidateSourceId(value)
    with pytest.raises(TypeError, match="must be str"):
        PairCandidateSourceId(cast(Any, 7))


def test_source_requires_canonical_nonempty_unique_artifact_refs() -> None:
    first = _ref("pair.evidence", "artifact:a")
    second = _ref("pair.evidence", "artifact:b")
    source = _source("gps", first, second)

    assert tuple(field.name for field in fields(PairCandidateSource)) == (
        "source_id",
        "evidence_refs",
    )
    assert source.evidence_refs == (first, second)
    assert hash(source) == hash(_source("gps", first, second))
    with pytest.raises(FrozenInstanceError):
        source.evidence_refs = (first,)  # type: ignore[misc]

    with pytest.raises(TypeError, match="source_id"):
        PairCandidateSource(
            source_id=cast(Any, "gps"),
            evidence_refs=(first,),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        PairCandidateSource(
            source_id=PairCandidateSourceId("gps"),
            evidence_refs=cast(Any, [first]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        PairCandidateSource(
            source_id=PairCandidateSourceId("gps"),
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        PairCandidateSource(
            source_id=PairCandidateSourceId("gps"),
            evidence_refs=cast(Any, ("artifact:a",)),
        )
    with pytest.raises(ValueError, match="duplicates"):
        _source("gps", first, first)
    with pytest.raises(ValueError, match="canonical artifact order"):
        _source("gps", second, first)


def test_pair_candidate_requires_canonical_distinct_endpoints_and_sources() -> None:
    gps = _source("gps", _ref("pair.gps", "artifact:gps"))
    visual = _source("visual", _ref("pair.visual", "artifact:visual"))
    candidate = _pair("obs:a", "obs:b", gps, visual)

    assert tuple(field.name for field in fields(PairCandidate)) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    assert hash(candidate) == hash(_pair("obs:a", "obs:b", gps, visual))
    with pytest.raises(FrozenInstanceError):
        candidate.sources = (gps,)  # type: ignore[misc]

    with pytest.raises(TypeError, match="observation_id1"):
        PairCandidate(
            observation_id1=cast(Any, "obs:a"),
            observation_id2=ObservationId("obs:b"),
            sources=(gps,),
        )
    with pytest.raises(TypeError, match="observation_id2"):
        PairCandidate(
            observation_id1=ObservationId("obs:a"),
            observation_id2=cast(Any, "obs:b"),
            sources=(gps,),
        )
    with pytest.raises(ValueError, match="canonical ascending order"):
        _pair("obs:b", "obs:a", gps)
    with pytest.raises(ValueError, match="canonical ascending order"):
        _pair("obs:a", "obs:a", gps)
    with pytest.raises(TypeError, match="immutable tuple"):
        PairCandidate(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            sources=cast(Any, [gps]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        _pair("obs:a", "obs:b")
    with pytest.raises(TypeError, match="PairCandidateSource"):
        PairCandidate(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
            sources=cast(Any, ("gps",)),
        )
    with pytest.raises(ValueError, match="duplicate source IDs"):
        _pair("obs:a", "obs:b", gps, gps)
    with pytest.raises(ValueError, match="canonical source-ID order"):
        _pair("obs:a", "obs:b", visual, gps)


def test_pair_candidate_contains_no_generic_score_or_truth_fields() -> None:
    assert tuple(field.name for field in fields(PairCandidate)) == (
        "observation_id1",
        "observation_id2",
        "sources",
    )
    assert tuple(field.name for field in fields(PairCandidateSource)) == (
        "source_id",
        "evidence_refs",
    )
    forbidden = {
        "score",
        "confidence",
        "probability",
        "rank",
        "distance",
        "sequence_index",
        "gps",
        "latitude",
        "longitude",
        "geolocation",
        "time",
        "overlap",
        "match_count",
        "inlier_count",
        "scene",
        "route",
        "quality_decision",
        "config",
        "metadata",
    }
    assert forbidden.isdisjoint(PairCandidate.__dataclass_fields__)
    assert forbidden.isdisjoint(PairCandidateSource.__dataclass_fields__)


def test_merge_unions_multiple_sources_without_losing_evidence() -> None:
    pair = (
        _pair(
            "obs:a",
            "obs:b",
            _source("sequential", _ref("pair.sequential", "artifact:seq")),
        ),
        _pair(
            "obs:a",
            "obs:b",
            _source("gps", _ref("pair.gps", "artifact:gps")),
        ),
        _pair(
            "obs:a",
            "obs:b",
            _source("visual", _ref("pair.visual", "artifact:visual")),
        ),
    )

    merged = merge_pair_candidates(pair)

    assert len(merged) == 1
    candidate = merged[0]
    assert tuple(source.source_id.value for source in candidate.sources) == (
        "gps",
        "sequential",
        "visual",
    )
    assert tuple(
        ref.artifact_id.value
        for source in candidate.sources
        for ref in source.evidence_refs
    ) == ("artifact:gps", "artifact:seq", "artifact:visual")


def test_merge_unions_same_source_evidence_deterministically() -> None:
    first = _pair(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:a")),
    )
    second = _pair(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:b")),
    )
    duplicate = _pair(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:a")),
    )

    merged = merge_pair_candidates((second, duplicate, first))

    assert merged == (
        _pair(
            "obs:a",
            "obs:b",
            _source(
                "visual",
                _ref("pair.visual", "artifact:a"),
                _ref("pair.visual", "artifact:b"),
            ),
        ),
    )


def test_merge_is_order_independent_idempotent_and_pair_sorted() -> None:
    ab = _pair(
        "obs:a",
        "obs:b",
        _source("sequential", _ref("pair.sequential", "artifact:ab")),
    )
    ac = _pair(
        "obs:a",
        "obs:c",
        _source("gps", _ref("pair.gps", "artifact:ac")),
    )
    bc = _pair(
        "obs:b",
        "obs:c",
        _source("visual", _ref("pair.visual", "artifact:bc")),
    )

    first = merge_pair_candidates((bc, ab, ac))
    second = merge_pair_candidates((ac, bc, ab))

    assert first == second == (ab, ac, bc)
    assert merge_pair_candidates(first) == first
    assert merge_pair_candidates(()) == ()

    with pytest.raises(TypeError, match="immutable tuple"):
        merge_pair_candidates(cast(Any, [ab]))
    with pytest.raises(TypeError, match="PairCandidate"):
        merge_pair_candidates(cast(Any, (ab, "not-a-candidate")))


def test_retained_pairing_donors_remain_separate_and_unchanged() -> None:
    sequential = SequentialPairCandidate(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        sequence_index1=0,
        sequence_index2=1,
        sequence_distance=1,
    )
    gps = GpsPairCandidate(
        observation_id1=ObservationId("obs:a"),
        observation_id2=ObservationId("obs:b"),
        distance_m=12.5,
    )

    assert sequential.sequence_distance == 1
    assert gps.distance_m == 12.5
    assert generate_sequential_candidates.__module__ == "wre.retrieval.sequential"
    assert generate_gps_candidates.__module__ == "wre.retrieval.gps"


def test_pair_candidate_module_has_no_retrieval_execution_or_future_truth_surface() -> None:
    for name in (
        "SequentialPairCandidate",
        "GpsPairCandidate",
        "generate_sequential_candidates",
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
        assert name not in pair_candidates_module.__dict__
