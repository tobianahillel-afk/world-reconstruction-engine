from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from typing import Any, cast

import pytest

import wre.reconstruction.colmap_matching as colmap_matching_module
import wre.reconstruction.colmap_verification as colmap_verification_module
import wre.scene_identity.colmap_adapter as adapter_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
)
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRunId
from wre.domain.scene_clusters import SceneRelationDisposition, SceneRelationHypothesis
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationResult,
    ColmapPairGeometryEvidence,
)
from wre.scene_identity import ColmapSceneRelationAdapterInput, adapt_colmap_scene_relations


def _ref(kind: str, artifact_id: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(kind),
    )


def _source(source_id: str, *refs: ArtifactRef) -> PairCandidateSource:
    return PairCandidateSource(
        source_id=PairCandidateSourceId(source_id),
        evidence_refs=tuple(refs),
    )


def _candidate(
    first: str,
    second: str,
    *sources: PairCandidateSource,
) -> PairCandidate:
    return PairCandidate(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        sources=tuple(sources),
    )


def _geometry(
    first: str,
    second: str,
    *,
    configuration: str = "CALIBRATED",
    inlier_count: int = 2,
) -> ColmapPairGeometryEvidence:
    return ColmapPairGeometryEvidence(
        observation_id1=ObservationId(first),
        observation_id2=ObservationId(second),
        image_name1=f"{first}.jpg",
        image_name2=f"{second}.jpg",
        raw_match_count=max(inlier_count, 4),
        configuration=configuration,
        inlier_matches=tuple((index, index + 1) for index in range(inlier_count)),
        fundamental_matrix=None,
        essential_matrix=None,
        homography_matrix=None,
        relative_pose_matrix=None,
        triangulation_angle_rad=None,
        has_estimated_camera1=False,
        has_estimated_camera2=False,
    )


def _verification(
    geometries: tuple[ColmapPairGeometryEvidence, ...] = (),
    *,
    observation_ids: tuple[str, ...] = ("obs:a", "obs:b", "obs:c"),
) -> ColmapGeometricVerificationResult:
    return ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:verification"),
            source_observation_ids=tuple(ObservationId(value) for value in observation_ids),
        ),
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build="Commit fixture without GPU support",
            ceres_version="2.2.0",
            upstream_has_cuda=False,
        ),
        configuration_sha256=Sha256Digest("a" * 64),
        source_matching_database_sha256=Sha256Digest("b" * 64),
        database_path=Path("/tmp/verified.db"),
        database_sha256=Sha256Digest("c" * 64),
        database_byte_length=1,
        geometries=geometries,
    )


def _input(
    candidates: tuple[PairCandidate, ...],
    *,
    verification: ColmapGeometricVerificationResult | None = None,
    evidence_ref: ArtifactRef | None = None,
) -> ColmapSceneRelationAdapterInput:
    return ColmapSceneRelationAdapterInput(
        candidates=candidates,
        verification=verification or _verification(),
        evidence_ref=evidence_ref or _ref("scene.verification", "artifact:verification"),
    )


def test_adapter_input_is_exact_immutable_contract() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("sequential", _ref("pair.sequential", "artifact:sequence")),
    )
    adapter_input = _input(
        (candidate,),
        verification=_verification((_geometry("obs:a", "obs:b"),)),
    )

    assert tuple(field.name for field in fields(ColmapSceneRelationAdapterInput)) == (
        "candidates",
        "verification",
        "evidence_ref",
    )
    with pytest.raises(FrozenInstanceError):
        adapter_input.candidates = ()  # type: ignore[misc]


def test_adapter_input_requires_canonical_unique_candidates_and_typed_values() -> None:
    source = _source("sequential", _ref("pair.sequential", "artifact:sequence"))
    ab = _candidate("obs:a", "obs:b", source)
    ac = _candidate("obs:a", "obs:c", source)

    with pytest.raises(TypeError, match="immutable tuple"):
        ColmapSceneRelationAdapterInput(
            candidates=cast(Any, [ab]),
            verification=_verification(),
            evidence_ref=_ref("scene.verification", "artifact:verification"),
        )
    with pytest.raises(TypeError, match="PairCandidate"):
        ColmapSceneRelationAdapterInput(
            candidates=cast(Any, ("not-a-candidate",)),
            verification=_verification(),
            evidence_ref=_ref("scene.verification", "artifact:verification"),
        )
    with pytest.raises(ValueError, match="unique"):
        _input((ab, ab))
    with pytest.raises(ValueError, match="canonical pair order"):
        _input((ac, ab))
    with pytest.raises(TypeError, match="ColmapGeometricVerificationResult"):
        ColmapSceneRelationAdapterInput(
            candidates=(ab,),
            verification=cast(Any, "verification"),
            evidence_ref=_ref("scene.verification", "artifact:verification"),
        )
    with pytest.raises(TypeError, match="ArtifactRef"):
        ColmapSceneRelationAdapterInput(
            candidates=(ab,),
            verification=_verification(),
            evidence_ref=cast(Any, "artifact:verification"),
        )


def test_adapter_input_rejects_candidate_outside_verification_provenance() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:z",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )

    with pytest.raises(ValueError, match="verification provenance observations"):
        _input(
            (candidate,),
            verification=_verification(observation_ids=("obs:a", "obs:b")),
        )


@pytest.mark.parametrize(
    "configuration",
    (
        "CALIBRATED",
        "CALIBRATED_RIG",
        "UNCALIBRATED",
        "PLANAR",
        "PANORAMIC",
        "PLANAR_OR_PANORAMIC",
    ),
)
def test_supported_colmap_configurations_produce_supported_relation(configuration: str) -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )

    relations = adapt_colmap_scene_relations(
        _input(
            (candidate,),
            verification=_verification(
                (_geometry("obs:a", "obs:b", configuration=configuration, inlier_count=2),)
            ),
        )
    )

    assert len(relations) == 1
    assert relations[0].disposition is SceneRelationDisposition.SUPPORTED


@pytest.mark.parametrize(
    "configuration",
    ("UNDEFINED", "DEGENERATE", "WATERMARK", "MULTIPLE"),
)
def test_ambiguous_colmap_configurations_remain_unresolved(configuration: str) -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )

    relations = adapt_colmap_scene_relations(
        _input(
            (candidate,),
            verification=_verification(
                (_geometry("obs:a", "obs:b", configuration=configuration, inlier_count=2),)
            ),
        )
    )

    assert relations[0].disposition is SceneRelationDisposition.UNRESOLVED


def test_missing_or_zero_inlier_geometry_remains_unresolved() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )

    absent = adapt_colmap_scene_relations(_input((candidate,), verification=_verification()))
    zero_inlier = adapt_colmap_scene_relations(
        _input(
            (candidate,),
            verification=_verification(
                (_geometry("obs:a", "obs:b", configuration="CALIBRATED", inlier_count=0),)
            ),
        )
    )

    assert absent[0].disposition is SceneRelationDisposition.UNRESOLVED
    assert zero_inlier[0].disposition is SceneRelationDisposition.UNRESOLVED


def test_future_unknown_configuration_fails_closed_to_unresolved() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )
    geometry = _geometry("obs:a", "obs:b")
    object.__setattr__(geometry, "configuration", "FUTURE_CONFIGURATION")

    relations = adapt_colmap_scene_relations(
        _input((candidate,), verification=_verification((geometry,)))
    )

    assert relations[0].disposition is SceneRelationDisposition.UNRESOLVED


def test_adapter_preserves_all_candidate_evidence_and_verification_ref() -> None:
    gps_ref = _ref("pair.gps", "artifact:gps")
    sequential_ref = _ref("pair.sequential", "artifact:sequence")
    shared_ref = _ref("pair.shared", "artifact:shared")
    verification_ref = _ref("scene.verification", "artifact:verification")
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("gps", gps_ref, shared_ref),
        _source("sequential", sequential_ref),
    )

    relation = adapt_colmap_scene_relations(
        _input(
            (candidate,),
            verification=_verification((_geometry("obs:a", "obs:b"),)),
            evidence_ref=verification_ref,
        )
    )[0]

    assert relation.evidence_refs == tuple(
        sorted(
            (gps_ref, shared_ref, sequential_ref, verification_ref),
            key=lambda ref: (ref.artifact_kind.value, ref.artifact_id.value),
        )
    )


def test_duplicate_verification_ref_is_deduplicated_without_losing_source_evidence() -> None:
    shared = _ref("scene.verification", "artifact:verification")
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", shared),
    )

    relation = adapt_colmap_scene_relations(
        _input(
            (candidate,),
            verification=_verification((_geometry("obs:a", "obs:b"),)),
            evidence_ref=shared,
        )
    )[0]

    assert relation.evidence_refs == (shared,)


def test_extra_donor_geometry_does_not_create_unsupplied_relationships() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )
    verification = _verification(
        (
            _geometry("obs:a", "obs:b"),
            _geometry("obs:b", "obs:c"),
        )
    )

    relations = adapt_colmap_scene_relations(_input((candidate,), verification=verification))

    assert tuple(
        (relation.observation_id1.value, relation.observation_id2.value) for relation in relations
    ) == (("obs:a", "obs:b"),)


def test_output_order_is_canonical_and_repeated_adaptation_is_deterministic() -> None:
    source = _source("visual", _ref("pair.visual", "artifact:visual"))
    candidates = (
        _candidate("obs:a", "obs:b", source),
        _candidate("obs:a", "obs:c", source),
    )
    adapter_input = _input(
        candidates,
        verification=_verification(
            (
                _geometry("obs:a", "obs:b"),
                _geometry("obs:a", "obs:c"),
            )
        ),
    )

    first = adapt_colmap_scene_relations(adapter_input)
    second = adapt_colmap_scene_relations(adapter_input)

    assert first == second
    assert tuple(
        (relation.observation_id1.value, relation.observation_id2.value) for relation in first
    ) == (("obs:a", "obs:b"), ("obs:a", "obs:c"))


def test_empty_candidate_set_is_a_valid_no_relationship_result() -> None:
    adapter_input = _input((), verification=_verification((_geometry("obs:a", "obs:b"),)))

    assert adapt_colmap_scene_relations(adapter_input) == ()


def test_adapter_never_emits_contradicted_or_leaks_solver_private_payloads() -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )
    relations = (
        adapt_colmap_scene_relations(
            _input(
                (candidate,),
                verification=_verification((_geometry("obs:a", "obs:b"),)),
            )
        ),
        adapt_colmap_scene_relations(_input((candidate,), verification=_verification())),
    )

    assert all(
        relation.disposition is not SceneRelationDisposition.CONTRADICTED
        for result in relations
        for relation in result
    )
    assert tuple(field.name for field in fields(SceneRelationHypothesis)) == (
        "observation_id1",
        "observation_id2",
        "disposition",
        "evidence_refs",
    )
    forbidden = {
        "raw_match_count",
        "inlier_count",
        "inlier_matches",
        "fundamental_matrix",
        "essential_matrix",
        "homography_matrix",
        "relative_pose_matrix",
        "triangulation_angle_rad",
        "has_estimated_camera1",
        "has_estimated_camera2",
        "image_name1",
        "image_name2",
        "configuration",
        "score",
        "confidence",
        "rank",
        "route",
    }
    assert forbidden.isdisjoint(SceneRelationHypothesis.__dataclass_fields__)


def test_adaptation_is_pure_and_does_not_execute_colmap_donors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _candidate(
        "obs:a",
        "obs:b",
        _source("visual", _ref("pair.visual", "artifact:visual")),
    )
    adapter_input = _input(
        (candidate,),
        verification=_verification((_geometry("obs:a", "obs:b"),)),
    )

    def _unexpected_call(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"unexpected donor/external call: {args} {kwargs}")

    monkeypatch.setattr(colmap_matching_module, "match_colmap_pairs", _unexpected_call)
    monkeypatch.setattr(colmap_verification_module, "verify_colmap_geometry", _unexpected_call)
    monkeypatch.setattr(
        colmap_verification_module.importlib,
        "import_module",
        _unexpected_call,
    )

    relations = adapt_colmap_scene_relations(adapter_input)

    assert len(relations) == 1
    assert relations[0].disposition is SceneRelationDisposition.SUPPORTED
    assert "match_colmap_pairs" not in adapter_module.__dict__
    assert "verify_colmap_geometry" not in adapter_module.__dict__
    assert "pycolmap" not in adapter_module.__dict__
