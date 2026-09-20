from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

import wre.reconstruction.colmap_matching as colmap_matching_module
import wre.reconstruction.colmap_verification as colmap_verification_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
)
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRunId
from wre.domain.scene_clusters import (
    SceneCluster,
    SceneRelationDisposition,
    SceneRelationHypothesis,
)
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationResult,
    ColmapPairGeometryEvidence,
)
from wre.regression import FixtureSpec, RegressionResult, evaluate_fixture, load_fixture
from wre.scene_identity import (
    ColmapSceneRelationAdapterInput,
    SceneClusteringConflictError,
    VerifiedSceneClusteringInput,
    adapt_colmap_scene_relations,
    cluster_verified_scene_relations,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "synthetic" / "repeated-symmetric-lookalike"
_FIXTURE_PATH = _FIXTURE_DIR / "fixture.json"


@dataclass(frozen=True)
class _BaselineRun:
    fixture: FixtureSpec
    scenario: dict[str, object]
    candidates: tuple[PairCandidate, ...]
    relations: tuple[SceneRelationHypothesis, ...]
    clusters: tuple[SceneCluster, ...]
    metrics: dict[str, int]
    regression: RegressionResult


def _object(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"{context} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise AssertionError(f"{context} keys must be strings")
    return cast(dict[str, object], value)


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise AssertionError(f"{context} must be a non-empty string")
    return value


def _int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AssertionError(f"{context} must be an integer")
    return value


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise AssertionError(f"{context} must be an array")
    return cast(list[object], value)


def _pair(value: object, context: str) -> tuple[str, str]:
    raw = _array(value, context)
    if len(raw) != 2:
        raise AssertionError(f"{context} must contain exactly two endpoints")
    first = _string(raw[0], f"{context}[0]")
    second = _string(raw[1], f"{context}[1]")
    if first >= second:
        raise AssertionError(f"{context} must use canonical distinct endpoint order")
    return (first, second)


def _artifact_ref(data: dict[str, object], context: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(_string(data.get("artifact_id"), f"{context}.artifact_id")),
        artifact_kind=ArtifactKind(_string(data.get("artifact_kind"), f"{context}.artifact_kind")),
    )


def _load_scenario(fixture: FixtureSpec) -> dict[str, object]:
    scenario_path = fixture.resolve_input("scenario.json")
    raw = json.loads(scenario_path.read_text(encoding="utf-8"))
    scenario = _object(raw, "scenario")
    expected_keys = {
        "schema_version",
        "profile",
        "observation_ids",
        "benchmark_scene_groups",
        "candidate_pairs",
        "false_bridge_control",
    }
    assert set(scenario) == expected_keys
    assert scenario["schema_version"] == 1
    assert scenario["profile"] == "repeated/symmetric architecture"
    return scenario


def _observation_values(scenario: dict[str, object]) -> tuple[str, ...]:
    raw = _array(scenario["observation_ids"], "scenario.observation_ids")
    values = tuple(
        _string(value, f"scenario.observation_ids[{index}]") for index, value in enumerate(raw)
    )
    assert len(values) == 4
    assert values == tuple(sorted(values))
    assert len(values) == len(set(values))
    return values


def _expected_scene_groups(scenario: dict[str, object]) -> tuple[tuple[str, ...], ...]:
    raw_groups = _array(
        scenario["benchmark_scene_groups"],
        "scenario.benchmark_scene_groups",
    )
    groups: list[tuple[str, ...]] = []
    for group_index, raw_group in enumerate(raw_groups):
        values = tuple(
            _string(value, f"benchmark_scene_groups[{group_index}]")
            for value in _array(raw_group, f"benchmark_scene_groups[{group_index}]")
        )
        assert len(values) == 2
        assert values == tuple(sorted(values))
        groups.append(values)
    result = tuple(groups)
    assert len(result) == 2
    assert result == tuple(sorted(result))
    assert sorted(value for group in result for value in group) == list(
        _observation_values(scenario)
    )
    return result


def _candidate_records(scenario: dict[str, object]) -> tuple[dict[str, object], ...]:
    records = tuple(
        _object(value, f"candidate_pairs[{index}]")
        for index, value in enumerate(
            _array(scenario["candidate_pairs"], "scenario.candidate_pairs")
        )
    )
    assert len(records) == 3
    for record in records:
        assert set(record) == {"endpoints", "role", "sources", "verification"}
    pair_keys = tuple(_pair(record["endpoints"], "candidate_pair.endpoints") for record in records)
    assert pair_keys == tuple(sorted(pair_keys))
    assert len(pair_keys) == len(set(pair_keys))
    return records


def _pair_candidate(record: dict[str, object]) -> PairCandidate:
    first, second = _pair(record["endpoints"], "candidate_pair.endpoints")
    raw_sources = _array(record["sources"], "candidate_pair.sources")
    sources: list[PairCandidateSource] = []
    for index, raw_source in enumerate(raw_sources):
        source = _object(raw_source, f"candidate_pair.sources[{index}]")
        assert set(source) == {"source_id", "artifact_kind", "artifact_id"}
        sources.append(
            PairCandidateSource(
                source_id=PairCandidateSourceId(
                    _string(source["source_id"], f"candidate_pair.sources[{index}].source_id")
                ),
                evidence_refs=(_artifact_ref(source, f"candidate_pair.sources[{index}]"),),
            )
        )
    return PairCandidate(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        sources=tuple(sources),
    )


def _verification_result(
    candidate: PairCandidate,
    verification_data: dict[str, object],
) -> tuple[ColmapGeometricVerificationResult, ArtifactRef]:
    assert set(verification_data) == {
        "artifact_kind",
        "artifact_id",
        "configuration",
        "raw_match_count",
        "inlier_count",
    }
    raw_match_count = _int(
        verification_data["raw_match_count"],
        "verification.raw_match_count",
    )
    inlier_count = _int(
        verification_data["inlier_count"],
        "verification.inlier_count",
    )
    geometry = ColmapPairGeometryEvidence(
        observation_id1=candidate.observation_id1,
        observation_id2=candidate.observation_id2,
        image_name1=f"{candidate.observation_id1.value}.png",
        image_name2=f"{candidate.observation_id2.value}.png",
        raw_match_count=raw_match_count,
        configuration=_string(
            verification_data["configuration"],
            "verification.configuration",
        ),
        inlier_matches=tuple((index, index) for index in range(inlier_count)),
        fundamental_matrix=None,
        essential_matrix=None,
        homography_matrix=None,
        relative_pose_matrix=None,
        triangulation_angle_rad=None,
        has_estimated_camera1=False,
        has_estimated_camera2=False,
    )
    verification_ref = _artifact_ref(verification_data, "verification")
    safe_name = f"{candidate.observation_id1.value}-{candidate.observation_id2.value}".replace(
        ":", "_"
    ).replace("/", "_")
    result = ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId(f"run:{safe_name}"),
            source_observation_ids=(
                candidate.observation_id1,
                candidate.observation_id2,
            ),
        ),
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build="fixture-only retained evidence",
            ceres_version="2.2.0",
            upstream_has_cuda=False,
        ),
        configuration_sha256=Sha256Digest("1" * 64),
        source_matching_database_sha256=Sha256Digest("2" * 64),
        database_path=Path("/fixture-never-read") / f"{safe_name}.db",
        database_sha256=Sha256Digest("3" * 64),
        database_byte_length=1,
        geometries=(geometry,),
    )
    return result, verification_ref


def _adapt_relations(
    records: tuple[dict[str, object], ...],
) -> tuple[tuple[PairCandidate, ...], tuple[SceneRelationHypothesis, ...]]:
    candidates: list[PairCandidate] = []
    relations: list[SceneRelationHypothesis] = []
    for record in records:
        candidate = _pair_candidate(record)
        verification, evidence_ref = _verification_result(
            candidate,
            _object(record["verification"], "candidate_pair.verification"),
        )
        relation = adapt_colmap_scene_relations(
            ColmapSceneRelationAdapterInput(
                candidates=(candidate,),
                verification=verification,
                evidence_ref=evidence_ref,
            )
        )[0]
        candidates.append(candidate)
        relations.append(relation)
    return tuple(candidates), tuple(relations)


def _false_merge_count(
    clusters: tuple[SceneCluster, ...],
    expected_groups: tuple[tuple[str, ...], ...],
) -> int:
    group_by_observation = {
        observation_id: group_index
        for group_index, group in enumerate(expected_groups)
        for observation_id in group
    }
    count = 0
    for cluster_object in clusters:
        observation_ids = cluster_object.observation_ids
        group_ids = {
            group_by_observation[observation_id.value] for observation_id in observation_ids
        }
        if len(group_ids) > 1:
            count += 1
    return count


def _conflict_detected(
    scenario: dict[str, object],
    baseline_relations: tuple[SceneRelationHypothesis, ...],
) -> bool:
    control = _object(
        scenario["false_bridge_control"],
        "scenario.false_bridge_control",
    )
    assert set(control) == {
        "bridge_pair",
        "contradicted_pair",
        "bridge_artifact",
        "contradiction_artifact",
    }
    bridge_pair = _pair(control["bridge_pair"], "false_bridge_control.bridge_pair")
    contradicted_pair = _pair(
        control["contradicted_pair"],
        "false_bridge_control.contradicted_pair",
    )

    baseline_by_pair = {
        (relation.observation_id1.value, relation.observation_id2.value): relation
        for relation in baseline_relations
    }
    assert bridge_pair in baseline_by_pair
    assert baseline_by_pair[bridge_pair].disposition is SceneRelationDisposition.UNRESOLVED

    conflict_relations = [
        relation for pair, relation in baseline_by_pair.items() if pair != bridge_pair
    ]
    conflict_relations.extend(
        (
            SceneRelationHypothesis(
                observation_id1=ObservationId(contradicted_pair[0]),
                observation_id2=ObservationId(contradicted_pair[1]),
                disposition=SceneRelationDisposition.CONTRADICTED,
                evidence_refs=(
                    _artifact_ref(
                        _object(
                            control["contradiction_artifact"],
                            "false_bridge_control.contradiction_artifact",
                        ),
                        "false_bridge_control.contradiction_artifact",
                    ),
                ),
            ),
            SceneRelationHypothesis(
                observation_id1=ObservationId(bridge_pair[0]),
                observation_id2=ObservationId(bridge_pair[1]),
                disposition=SceneRelationDisposition.SUPPORTED,
                evidence_refs=(
                    _artifact_ref(
                        _object(
                            control["bridge_artifact"],
                            "false_bridge_control.bridge_artifact",
                        ),
                        "false_bridge_control.bridge_artifact",
                    ),
                ),
            ),
        )
    )
    canonical_relations = tuple(
        sorted(
            conflict_relations,
            key=lambda relation: (
                relation.observation_id1.value,
                relation.observation_id2.value,
            ),
        )
    )

    with pytest.raises(SceneClusteringConflictError):
        cluster_verified_scene_relations(
            VerifiedSceneClusteringInput(
                observation_ids=tuple(
                    ObservationId(value) for value in _observation_values(scenario)
                ),
                relations=canonical_relations,
            )
        )
    return True


def _run_baseline() -> _BaselineRun:
    fixture = load_fixture(_FIXTURE_PATH)
    scenario = _load_scenario(fixture)
    observations = _observation_values(scenario)
    expected_groups = _expected_scene_groups(scenario)
    records = _candidate_records(scenario)
    candidates, relations = _adapt_relations(records)

    clusters = cluster_verified_scene_relations(
        VerifiedSceneClusteringInput(
            observation_ids=tuple(ObservationId(value) for value in observations),
            relations=relations,
        )
    )

    lookalike_indices = tuple(
        index for index, record in enumerate(records) if record["role"] == "cross_scene_lookalike"
    )
    assert len(lookalike_indices) == 1
    lookalike_index = lookalike_indices[0]
    unresolved_cross_scene_relation_count = int(
        relations[lookalike_index].disposition is SceneRelationDisposition.UNRESOLVED
    )
    conflict_detected = _conflict_detected(scenario, relations)

    metrics = {
        "observation_count": len(observations),
        "weak_similarity_candidate_count": len(lookalike_indices),
        "unresolved_cross_scene_relation_count": unresolved_cross_scene_relation_count,
        "baseline_cluster_count": len(clusters),
        "false_merge_count": _false_merge_count(clusters, expected_groups),
        "conflict_detected_count": int(conflict_detected),
    }
    regression = evaluate_fixture(
        fixture,
        metrics,
        runner="tests.test_repeated_symmetric_lookalike_fixture",
    )
    return _BaselineRun(
        fixture=fixture,
        scenario=scenario,
        candidates=candidates,
        relations=relations,
        clusters=clusters,
        metrics=metrics,
        regression=regression,
    )


def test_fixture_metadata_and_scenario_contract_are_explicit() -> None:
    run = _run_baseline()

    assert run.fixture.fixture_id == "synthetic.repeated-symmetric-lookalike.v1"
    assert run.fixture.kind == "synthetic"
    assert run.fixture.lane == "fast"
    assert run.fixture.deterministic is True
    assert run.fixture.seed == 0
    assert run.fixture.input_files == ("scenario.json",)
    assert run.fixture.provenance["source"] == "repository-authored"
    assert run.fixture.provenance["profile"] == "repeated/symmetric architecture"

    assert _observation_values(run.scenario) == (
        "obs:facade-a-1",
        "obs:facade-a-2",
        "obs:facade-b-1",
        "obs:facade-b-2",
    )
    assert _expected_scene_groups(run.scenario) == (
        ("obs:facade-a-1", "obs:facade-a-2"),
        ("obs:facade-b-1", "obs:facade-b-2"),
    )


def test_lookalike_candidate_keeps_multiple_retrieval_sources_without_scene_truth() -> None:
    run = _run_baseline()
    records = _candidate_records(run.scenario)
    lookalike_index = next(
        index for index, record in enumerate(records) if record["role"] == "cross_scene_lookalike"
    )
    candidate = run.candidates[lookalike_index]
    relation = run.relations[lookalike_index]

    assert tuple(source.source_id.value for source in candidate.sources) == (
        "colmap.vocab_tree",
        "selavpr_plus",
    )
    assert relation.disposition is SceneRelationDisposition.UNRESOLVED
    assert relation.disposition is not SceneRelationDisposition.SUPPORTED
    assert relation.disposition is not SceneRelationDisposition.CONTRADICTED


def test_ambiguous_geometry_keeps_distinct_repeated_scenes_separate() -> None:
    run = _run_baseline()
    records = _candidate_records(run.scenario)

    assert tuple(relation.disposition for relation in run.relations) == (
        SceneRelationDisposition.SUPPORTED,
        SceneRelationDisposition.UNRESOLVED,
        SceneRelationDisposition.SUPPORTED,
    )

    cluster_membership = tuple(
        tuple(observation_id.value for observation_id in cluster.observation_ids)
        for cluster in run.clusters
    )
    assert cluster_membership == _expected_scene_groups(run.scenario)
    assert run.metrics["false_merge_count"] == 0

    lookalike_index = next(
        index
        for index, record in enumerate(records)
        if record["role"] == "cross_scene_lookalike"
    )
    unresolved_evidence = set(run.relations[lookalike_index].evidence_refs)
    cluster_evidence = {ref for cluster in run.clusters for ref in cluster.evidence_refs}
    assert unresolved_evidence.isdisjoint(cluster_evidence)


def test_false_supported_bridge_with_distinct_contradiction_fails_closed() -> None:
    run = _run_baseline()

    assert run.metrics["conflict_detected_count"] == 1


def test_fixture_metrics_and_repeated_execution_are_exact() -> None:
    first = _run_baseline()
    second = _run_baseline()

    assert first.candidates == second.candidates
    assert first.relations == second.relations
    assert first.clusters == second.clusters
    assert first.metrics == second.metrics
    assert first.regression.to_dict() == second.regression.to_dict()
    assert first.regression.passed
    assert first.metrics == {
        "observation_count": 4,
        "weak_similarity_candidate_count": 1,
        "unresolved_cross_scene_relation_count": 1,
        "baseline_cluster_count": 2,
        "false_merge_count": 0,
        "conflict_detected_count": 1,
    }


def test_fixture_does_not_execute_external_colmap_or_model_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _unexpected_call(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected external execution: {args} {kwargs}")

    monkeypatch.setattr(
        colmap_matching_module,
        "match_colmap_pairs",
        _unexpected_call,
    )
    monkeypatch.setattr(
        colmap_verification_module,
        "verify_colmap_geometry",
        _unexpected_call,
    )
    monkeypatch.setattr(
        colmap_verification_module.importlib,
        "import_module",
        _unexpected_call,
    )

    run = _run_baseline()

    assert run.regression.passed
