from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.colmap_evidence_artifacts as evidence_artifacts
from wre.domain.artifact_keys import (
    ArtifactInputFingerprint,
    ArtifactKeyMaterial,
    derive_artifact_key,
)
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.projects import SceneProjectId
from wre.domain.provenance import ProvenanceClass
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRunId
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_evidence_artifacts import (
    COLMAP_EVIDENCE_ARTIFACT_KEY_HARDWARE_POLICY,
    COLMAP_EVIDENCE_CHECKPOINT,
    COLMAP_EVIDENCE_DEPENDENCY_REF,
    COLMAP_EVIDENCE_MODEL,
    COLMAP_EVIDENCE_PRODUCER_VERSION,
    COLMAP_EVIDENCE_SHIPPING_STATUS,
    COLMAP_GEOMETRIC_VERIFICATION_ADAPTER_ID,
    COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY,
    COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY_NAME,
    COLMAP_GEOMETRIC_VERIFICATION_PRODUCER_IMPLEMENTATION,
    COLMAP_GEOMETRIC_VERIFICATION_REPRODUCIBILITY_NOTES,
    COLMAP_LOCAL_FEATURES_ADAPTER_ID,
    COLMAP_LOCAL_FEATURES_CAPABILITY,
    COLMAP_LOCAL_FEATURES_CAPABILITY_NAME,
    COLMAP_LOCAL_FEATURES_PRODUCER_IMPLEMENTATION,
    COLMAP_LOCAL_FEATURES_REPRODUCIBILITY_NOTES,
    COLMAP_PAIR_MATCHING_ADAPTER_ID,
    COLMAP_PAIR_MATCHING_CAPABILITY,
    COLMAP_PAIR_MATCHING_CAPABILITY_NAME,
    COLMAP_PAIR_MATCHING_PRODUCER_IMPLEMENTATION,
    COLMAP_PAIR_MATCHING_REPRODUCIBILITY_NOTES,
    GEOMETRIC_VERIFICATION_KIND,
    IMAGE_OBSERVATION_KIND,
    LOCAL_FEATURES_KIND,
    PAIR_MATCHES_KIND,
    ColmapEvidenceArtifactPlan,
    plan_colmap_geometric_verification,
    plan_colmap_local_features,
    plan_colmap_pair_matches,
    publish_colmap_geometric_verification,
    publish_colmap_local_features,
    publish_colmap_pair_matches,
)
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_matching import (
    ColmapPairMatchingConfig,
    ColmapPairMatchingResult,
    ColmapPairMatchSummary,
)
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationResult,
)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _hardware(character: str = "9") -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(sha256=_digest(character))


def _fingerprint(kind: ArtifactKind, character: str) -> ArtifactInputFingerprint:
    return ArtifactInputFingerprint(artifact_kind=kind, sha256=_digest(character))


def _environment() -> ColmapEnvironmentIdentity:
    return ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build="Commit fixture without GPU support",
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )


def _feature_result(
    *,
    configuration_sha256: Sha256Digest,
    database_sha256: Sha256Digest | None = None,
    database_byte_length: int = 101,
) -> ColmapFeatureExtractionResult:
    actual_database_sha256 = database_sha256 or _digest("d")
    observation_id = ObservationId("obs:a")
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:features"),
            source_observation_ids=(observation_id,),
        ),
        environment=_environment(),
        configuration_sha256=configuration_sha256,
        database_path=Path("/does/not/need/to/exist/features.db"),
        database_sha256=actual_database_sha256,
        database_byte_length=database_byte_length,
        images=(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name="000000-a.pgm",
                keypoint_rows=8,
                keypoint_cols=4,
                descriptor_rows=8,
                descriptor_cols=128,
            ),
        ),
    )


def _matching_result(
    *,
    configuration_sha256: Sha256Digest,
    source_feature_database_sha256: Sha256Digest,
    database_sha256: Sha256Digest | None = None,
    database_byte_length: int = 202,
) -> ColmapPairMatchingResult:
    actual_database_sha256 = database_sha256 or _digest("e")
    first = ObservationId("obs:a")
    second = ObservationId("obs:b")
    return ColmapPairMatchingResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:matches"),
            source_observation_ids=(first, second),
        ),
        environment=_environment(),
        configuration_sha256=configuration_sha256,
        source_feature_database_sha256=source_feature_database_sha256,
        database_path=Path("/does/not/need/to/exist/matches.db"),
        database_sha256=actual_database_sha256,
        database_byte_length=database_byte_length,
        attempted_pair_count=1,
        unverified_two_view_placeholder_count=1,
        pairs=(
            ColmapPairMatchSummary(
                observation_id1=first,
                observation_id2=second,
                image_name1="a.pgm",
                image_name2="b.pgm",
                num_matches=7,
            ),
        ),
    )


def _verification_result(
    *,
    configuration_sha256: Sha256Digest,
    source_matching_database_sha256: Sha256Digest,
    database_sha256: Sha256Digest | None = None,
    database_byte_length: int = 303,
) -> ColmapGeometricVerificationResult:
    actual_database_sha256 = database_sha256 or _digest("f")
    return ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:verification"),
            source_observation_ids=(ObservationId("obs:a"), ObservationId("obs:b")),
        ),
        environment=_environment(),
        configuration_sha256=configuration_sha256,
        source_matching_database_sha256=source_matching_database_sha256,
        database_path=Path("/does/not/need/to/exist/verification.db"),
        database_sha256=actual_database_sha256,
        database_byte_length=database_byte_length,
        geometries=(),
    )


def _entries_by_id() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    entries = document["entries"]
    assert isinstance(entries, list)
    return {
        cast(str, entry["adapter_id"]): cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
    }


def test_colmap_evidence_stage_descriptors_have_exact_v2_semantics() -> None:
    assert IMAGE_OBSERVATION_KIND == ArtifactKind("image.observation")
    assert LOCAL_FEATURES_KIND == ArtifactKind("evidence.local_features")
    assert PAIR_MATCHES_KIND == ArtifactKind("evidence.pair_matches")
    assert GEOMETRIC_VERIFICATION_KIND == ArtifactKind("evidence.geometric_verification")

    assert COLMAP_LOCAL_FEATURES_CAPABILITY.capability is COLMAP_LOCAL_FEATURES_CAPABILITY_NAME
    assert COLMAP_LOCAL_FEATURES_CAPABILITY.input_kinds == frozenset({IMAGE_OBSERVATION_KIND})
    assert COLMAP_LOCAL_FEATURES_CAPABILITY.output_kinds == frozenset({LOCAL_FEATURES_KIND})

    assert COLMAP_PAIR_MATCHING_CAPABILITY.capability is COLMAP_PAIR_MATCHING_CAPABILITY_NAME
    assert COLMAP_PAIR_MATCHING_CAPABILITY.input_kinds == frozenset({LOCAL_FEATURES_KIND})
    assert COLMAP_PAIR_MATCHING_CAPABILITY.output_kinds == frozenset({PAIR_MATCHES_KIND})

    assert (
        COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY.capability
        is COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY_NAME
    )
    assert COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY.input_kinds == frozenset(
        {PAIR_MATCHES_KIND}
    )
    assert COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY.output_kinds == frozenset(
        {GEOMETRIC_VERIFICATION_KIND}
    )


def test_feature_plan_uses_existing_config_digest_exact_producer_and_hardware_key() -> None:
    config = ColmapFeatureExtractionConfig()
    inputs = (
        _fingerprint(IMAGE_OBSERVATION_KIND, "1"),
        _fingerprint(IMAGE_OBSERVATION_KIND, "2"),
    )
    hardware = _hardware("3")

    plan = plan_colmap_local_features(
        inputs,
        configuration_sha256=config.sha256,
        hardware_runtime=hardware,
    )

    assert isinstance(plan, ColmapEvidenceArtifactPlan)
    assert plan.output_kind == LOCAL_FEATURES_KIND
    assert plan.input_fingerprints is inputs
    assert plan.producer.producer.implementation == COLMAP_LOCAL_FEATURES_PRODUCER_IMPLEMENTATION
    assert plan.producer.producer.version == "4.2.0"
    assert plan.producer.producer.revision is None
    assert plan.producer.configuration.sha256 == config.sha256
    assert plan.producer.model is None
    assert plan.producer.checkpoint is None
    assert plan.hardware_runtime is hardware
    assert plan.artifact_key == derive_artifact_key(
        ArtifactKeyMaterial(
            output_kind=LOCAL_FEATURES_KIND,
            input_fingerprints=inputs,
            producer=plan.producer,
            hardware_runtime=hardware,
        )
    )


def test_colmap_evidence_key_changes_with_input_configuration_or_hardware() -> None:
    base_inputs = (_fingerprint(IMAGE_OBSERVATION_KIND, "1"),)
    config = ColmapFeatureExtractionConfig().sha256
    base = plan_colmap_local_features(
        base_inputs,
        configuration_sha256=config,
        hardware_runtime=_hardware("3"),
    )
    changed_input = plan_colmap_local_features(
        (_fingerprint(IMAGE_OBSERVATION_KIND, "2"),),
        configuration_sha256=config,
        hardware_runtime=_hardware("3"),
    )
    changed_config = plan_colmap_local_features(
        base_inputs,
        configuration_sha256=_digest("4"),
        hardware_runtime=_hardware("3"),
    )
    changed_hardware = plan_colmap_local_features(
        base_inputs,
        configuration_sha256=config,
        hardware_runtime=_hardware("5"),
    )

    assert len(
        {
            base.artifact_key,
            changed_input.artifact_key,
            changed_config.artifact_key,
            changed_hardware.artifact_key,
        }
    ) == 4


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (cast(Any, [_fingerprint(IMAGE_OBSERVATION_KIND, "1")]), TypeError, "immutable tuple"),
        ((), ValueError, "at least one"),
        (cast(Any, ("raw",)), TypeError, "ArtifactInputFingerprint"),
        ((_fingerprint(ArtifactKind("video.observation"), "1"),), ValueError, "image.observation"),
        (
            (
                _fingerprint(IMAGE_OBSERVATION_KIND, "1"),
                _fingerprint(IMAGE_OBSERVATION_KIND, "1"),
            ),
            ValueError,
            "duplicate",
        ),
        (
            (
                _fingerprint(IMAGE_OBSERVATION_KIND, "2"),
                _fingerprint(IMAGE_OBSERVATION_KIND, "1"),
            ),
            ValueError,
            "canonical fingerprint order",
        ),
    ],
)
def test_feature_plan_inputs_fail_closed(
    inputs: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        plan_colmap_local_features(
            inputs,
            configuration_sha256=ColmapFeatureExtractionConfig().sha256,
            hardware_runtime=_hardware(),
        )


@pytest.mark.parametrize(
    ("planner", "kind", "config_sha256"),
    [
        (plan_colmap_pair_matches, LOCAL_FEATURES_KIND, ColmapPairMatchingConfig().sha256),
        (
            plan_colmap_geometric_verification,
            PAIR_MATCHES_KIND,
            ColmapGeometricVerificationConfig().sha256,
        ),
    ],
)
def test_downstream_plans_require_exactly_one_parent_of_exact_kind(
    planner: Any,
    kind: ArtifactKind,
    config_sha256: Sha256Digest,
) -> None:
    parent = _fingerprint(kind, "6")

    plan = planner(
        (parent,),
        configuration_sha256=config_sha256,
        hardware_runtime=_hardware(),
    )

    assert plan.input_fingerprints == (parent,)

    with pytest.raises(ValueError, match="exactly 1"):
        planner(
            (parent, _fingerprint(kind, "7")),
            configuration_sha256=config_sha256,
            hardware_runtime=_hardware(),
        )
    with pytest.raises(ValueError, match=kind.value):
        planner(
            (_fingerprint(ArtifactKind("wrong.kind"), "6"),),
            configuration_sha256=config_sha256,
            hardware_runtime=_hardware(),
        )


def test_feature_publication_preserves_exact_result_bytes_without_reading_path() -> None:
    config_sha256 = ColmapFeatureExtractionConfig().sha256
    plan = plan_colmap_local_features(
        (_fingerprint(IMAGE_OBSERVATION_KIND, "1"),),
        configuration_sha256=config_sha256,
        hardware_runtime=_hardware(),
    )
    result = _feature_result(configuration_sha256=config_sha256)
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:features"),
        artifact_kind=LOCAL_FEATURES_KIND,
    )

    publication = publish_colmap_local_features(
        project_id=SceneProjectId("project:one"),
        artifact_ref=artifact_ref,
        plan=plan,
        result=result,
        relative_path="colmap/features.db",
    )

    assert publication.metadata.project_id == SceneProjectId("project:one")
    assert publication.metadata.artifact_ref == artifact_ref
    assert publication.metadata.artifact_key == plan.artifact_key
    assert publication.metadata.producer is plan.producer
    assert publication.metadata.provenance_class is ProvenanceClass.OBSERVED_RECONSTRUCTED
    assert publication.materialization.artifact_ref == artifact_ref
    assert publication.materialization.entries[0].relative_path == "colmap/features.db"
    assert publication.materialization.entries[0].sha256 == result.database_sha256
    assert publication.materialization.entries[0].byte_length == result.database_byte_length


def test_matching_and_verification_publication_require_exact_parent_database_digest() -> None:
    feature_database_sha256 = _digest("a")
    matching_config_sha256 = ColmapPairMatchingConfig().sha256
    matching_plan = plan_colmap_pair_matches(
        (_fingerprint(LOCAL_FEATURES_KIND, "a"),),
        configuration_sha256=matching_config_sha256,
        hardware_runtime=_hardware(),
    )
    matching_result = _matching_result(
        configuration_sha256=matching_config_sha256,
        source_feature_database_sha256=feature_database_sha256,
    )

    matching_publication = publish_colmap_pair_matches(
        project_id=SceneProjectId("project:one"),
        artifact_ref=ArtifactRef(ArtifactId("artifact:matches"), PAIR_MATCHES_KIND),
        plan=matching_plan,
        result=matching_result,
        relative_path="colmap/matches.db",
    )
    assert matching_publication.materialization.entries[0].sha256 == matching_result.database_sha256

    wrong_matching_plan = plan_colmap_pair_matches(
        (_fingerprint(LOCAL_FEATURES_KIND, "b"),),
        configuration_sha256=matching_config_sha256,
        hardware_runtime=_hardware(),
    )
    with pytest.raises(ValueError, match="source database"):
        publish_colmap_pair_matches(
            project_id=SceneProjectId("project:one"),
            artifact_ref=ArtifactRef(ArtifactId("artifact:bad-matches"), PAIR_MATCHES_KIND),
            plan=wrong_matching_plan,
            result=matching_result,
            relative_path="colmap/bad-matches.db",
        )

    verification_config_sha256 = ColmapGeometricVerificationConfig().sha256
    verification_plan = plan_colmap_geometric_verification(
        (
            ArtifactInputFingerprint(
                artifact_kind=PAIR_MATCHES_KIND,
                sha256=matching_result.database_sha256,
            ),
        ),
        configuration_sha256=verification_config_sha256,
        hardware_runtime=_hardware(),
    )
    verification_result = _verification_result(
        configuration_sha256=verification_config_sha256,
        source_matching_database_sha256=matching_result.database_sha256,
    )
    verification_publication = publish_colmap_geometric_verification(
        project_id=SceneProjectId("project:one"),
        artifact_ref=ArtifactRef(
            ArtifactId("artifact:verification"),
            GEOMETRIC_VERIFICATION_KIND,
        ),
        plan=verification_plan,
        result=verification_result,
        relative_path="colmap/verification.db",
    )

    assert (
        verification_publication.materialization.entries[0].sha256
        == verification_result.database_sha256
    )


def test_publication_identity_excludes_project_artifact_path_and_run_id() -> None:
    config_sha256 = ColmapFeatureExtractionConfig().sha256
    plan = plan_colmap_local_features(
        (_fingerprint(IMAGE_OBSERVATION_KIND, "1"),),
        configuration_sha256=config_sha256,
        hardware_runtime=_hardware(),
    )
    first_result = _feature_result(configuration_sha256=config_sha256)
    second_result = ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:other"),
            source_observation_ids=first_result.provenance.source_observation_ids,
        ),
        environment=first_result.environment,
        configuration_sha256=first_result.configuration_sha256,
        database_path=Path("/another/nonexistent/location.db"),
        database_sha256=first_result.database_sha256,
        database_byte_length=first_result.database_byte_length,
        images=first_result.images,
    )

    first = publish_colmap_local_features(
        project_id=SceneProjectId("project:first"),
        artifact_ref=ArtifactRef(ArtifactId("artifact:first"), LOCAL_FEATURES_KIND),
        plan=plan,
        result=first_result,
        relative_path="first/features.db",
    )
    second = publish_colmap_local_features(
        project_id=SceneProjectId("project:second"),
        artifact_ref=ArtifactRef(ArtifactId("artifact:second"), LOCAL_FEATURES_KIND),
        plan=plan,
        result=second_result,
        relative_path="second/features.db",
    )

    assert first.metadata.artifact_key == second.metadata.artifact_key == plan.artifact_key


def test_publication_delegates_canonical_relative_path_validation() -> None:
    config_sha256 = ColmapFeatureExtractionConfig().sha256
    plan = plan_colmap_local_features(
        (_fingerprint(IMAGE_OBSERVATION_KIND, "1"),),
        configuration_sha256=config_sha256,
        hardware_runtime=_hardware(),
    )

    with pytest.raises(ValueError, match="parent traversal"):
        publish_colmap_local_features(
            project_id=SceneProjectId("project:one"),
            artifact_ref=ArtifactRef(ArtifactId("artifact:features"), LOCAL_FEATURES_KIND),
            plan=plan,
            result=_feature_result(configuration_sha256=config_sha256),
            relative_path="../features.db",
        )


def test_publication_rejects_wrong_stage_kind_or_configuration() -> None:
    config_sha256 = ColmapFeatureExtractionConfig().sha256
    plan = plan_colmap_local_features(
        (_fingerprint(IMAGE_OBSERVATION_KIND, "1"),),
        configuration_sha256=config_sha256,
        hardware_runtime=_hardware(),
    )

    with pytest.raises(ValueError, match="ArtifactRef kind"):
        publish_colmap_local_features(
            project_id=SceneProjectId("project:one"),
            artifact_ref=ArtifactRef(ArtifactId("artifact:wrong"), PAIR_MATCHES_KIND),
            plan=plan,
            result=_feature_result(configuration_sha256=config_sha256),
            relative_path="features.db",
        )
    with pytest.raises(ValueError, match="configuration"):
        publish_colmap_local_features(
            project_id=SceneProjectId("project:one"),
            artifact_ref=ArtifactRef(ArtifactId("artifact:features"), LOCAL_FEATURES_KIND),
            plan=plan,
            result=_feature_result(configuration_sha256=_digest("8")),
            relative_path="features.db",
        )


def test_registry_entries_match_colmap_evidence_module_contract() -> None:
    entries = _entries_by_id()
    expected = {
        COLMAP_GEOMETRIC_VERIFICATION_ADAPTER_ID: {
            "name": COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY_NAME.value,
            "inputs": ["evidence.pair_matches"],
            "outputs": ["evidence.geometric_verification"],
            "producer": COLMAP_GEOMETRIC_VERIFICATION_PRODUCER_IMPLEMENTATION,
            "notes": COLMAP_GEOMETRIC_VERIFICATION_REPRODUCIBILITY_NOTES,
        },
        COLMAP_LOCAL_FEATURES_ADAPTER_ID: {
            "name": COLMAP_LOCAL_FEATURES_CAPABILITY_NAME.value,
            "inputs": ["image.observation"],
            "outputs": ["evidence.local_features"],
            "producer": COLMAP_LOCAL_FEATURES_PRODUCER_IMPLEMENTATION,
            "notes": COLMAP_LOCAL_FEATURES_REPRODUCIBILITY_NOTES,
        },
        COLMAP_PAIR_MATCHING_ADAPTER_ID: {
            "name": COLMAP_PAIR_MATCHING_CAPABILITY_NAME.value,
            "inputs": ["evidence.local_features"],
            "outputs": ["evidence.pair_matches"],
            "producer": COLMAP_PAIR_MATCHING_PRODUCER_IMPLEMENTATION,
            "notes": COLMAP_PAIR_MATCHING_REPRODUCIBILITY_NOTES,
        },
    }

    for adapter_id, expected_entry in expected.items():
        entry = entries[adapter_id]
        assert entry["capability"] == {
            "name": expected_entry["name"],
            "input_kinds": expected_entry["inputs"],
            "output_kinds": expected_entry["outputs"],
        }
        assert entry["producer"] == {
            "implementation": expected_entry["producer"],
            "version": COLMAP_EVIDENCE_PRODUCER_VERSION,
            "revision": None,
        }
        assert entry["dependency_refs"] == [COLMAP_EVIDENCE_DEPENDENCY_REF]
        assert entry["model"] is COLMAP_EVIDENCE_MODEL
        assert entry["checkpoint"] is COLMAP_EVIDENCE_CHECKPOINT
        assert (
            entry["artifact_key_hardware_policy"]
            == COLMAP_EVIDENCE_ARTIFACT_KEY_HARDWARE_POLICY
        )
        assert entry["shipping_status"] == COLMAP_EVIDENCE_SHIPPING_STATUS
        assert entry["reproducibility_notes"] == expected_entry["notes"]
        assert entry["failure_signals"] == []
        assert entry["metric_names"] == []
        assert entry["resume_mode"] == "unsupported"


def test_registry_preserves_legacy_colmap_sparse_sfm_donor_semantics() -> None:
    donor = _entries_by_id()["colmap.sparse_sfm"]

    assert donor["capability"] == {
        "name": "geometry.sparse_sfm",
        "input_kinds": ["image.observation"],
        "output_kinds": ["geometry.sparse_reconstruction_estimate"],
    }
    assert donor["producer"] == {
        "implementation": "wre.colmap_reconstruction_importer",
        "version": "2",
        "revision": None,
    }
    assert donor["dependency_refs"] == ["colmap"]
    assert donor["artifact_key_hardware_policy"] == "required"
    assert donor["shipping_status"] == "approved"


def test_existing_donor_configuration_digests_remain_authoritative() -> None:
    assert ColmapFeatureExtractionConfig().sha256 == ColmapFeatureExtractionConfig().sha256
    assert ColmapPairMatchingConfig().sha256 == ColmapPairMatchingConfig().sha256
    assert (
        ColmapGeometricVerificationConfig().sha256
        == ColmapGeometricVerificationConfig().sha256
    )
    assert COLMAP_EVIDENCE_PRODUCER_VERSION == "4.2.0"


def test_colmap_evidence_module_has_no_solver_io_or_persistence_surface() -> None:
    forbidden_names = {
        "Path",
        "pycolmap",
        "sqlite3",
        "subprocess",
        "socket",
        "requests",
        "SQLiteLocalStore",
        "verify_local_artifact_materialization",
        "extract_colmap_features",
        "match_colmap_pairs",
        "verify_colmap_geometry",
        "reconstruct_colmap_incrementally",
        "CameraSolution",
        "PointMap",
        "GeometrySolution",
    }

    assert forbidden_names.isdisjoint(vars(evidence_artifacts))
