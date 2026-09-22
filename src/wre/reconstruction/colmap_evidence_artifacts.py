from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.artifact_keys import (
    ArtifactInputFingerprint,
    ArtifactKey,
    ArtifactKeyMaterial,
    derive_artifact_key,
)
from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifact_metadata import ArtifactMetadata
from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.projects import SceneProjectId
from wre.domain.provenance import ProvenanceClass
from wre.domain.runs import ProducerRef
from wre.reconstruction.colmap_environment import SUPPORTED_PYCOLMAP_VERSION

if TYPE_CHECKING:
    from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
    from wre.reconstruction.colmap_matching import ColmapPairMatchingResult
    from wre.reconstruction.colmap_verification import ColmapGeometricVerificationResult


COLMAP_LOCAL_FEATURES_ADAPTER_ID = "colmap.local_features"
COLMAP_PAIR_MATCHING_ADAPTER_ID = "colmap.pair_matching"
COLMAP_GEOMETRIC_VERIFICATION_ADAPTER_ID = "colmap.geometric_verification"

COLMAP_LOCAL_FEATURES_CAPABILITY_NAME = AdapterCapabilityName("evidence.local_features")
COLMAP_PAIR_MATCHING_CAPABILITY_NAME = AdapterCapabilityName("evidence.pair_matching")
COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY_NAME = AdapterCapabilityName(
    "evidence.geometric_verification"
)

IMAGE_OBSERVATION_KIND = ArtifactKind("image.observation")
LOCAL_FEATURES_KIND = ArtifactKind("evidence.local_features")
PAIR_MATCHES_KIND = ArtifactKind("evidence.pair_matches")
GEOMETRIC_VERIFICATION_KIND = ArtifactKind("evidence.geometric_verification")

COLMAP_LOCAL_FEATURES_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_LOCAL_FEATURES_CAPABILITY_NAME,
    input_kinds=frozenset({IMAGE_OBSERVATION_KIND}),
    output_kinds=frozenset({LOCAL_FEATURES_KIND}),
)
COLMAP_PAIR_MATCHING_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_PAIR_MATCHING_CAPABILITY_NAME,
    input_kinds=frozenset({LOCAL_FEATURES_KIND}),
    output_kinds=frozenset({PAIR_MATCHES_KIND}),
)
COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_GEOMETRIC_VERIFICATION_CAPABILITY_NAME,
    input_kinds=frozenset({PAIR_MATCHES_KIND}),
    output_kinds=frozenset({GEOMETRIC_VERIFICATION_KIND}),
)

COLMAP_LOCAL_FEATURES_PRODUCER_IMPLEMENTATION = "pycolmap.extract_features"
COLMAP_PAIR_MATCHING_PRODUCER_IMPLEMENTATION = "pycolmap.match_exhaustive"
COLMAP_GEOMETRIC_VERIFICATION_PRODUCER_IMPLEMENTATION = "pycolmap.geometric_verification"

COLMAP_EVIDENCE_DEPENDENCY_REF = "colmap"
COLMAP_EVIDENCE_PRODUCER_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_EVIDENCE_MODEL = None
COLMAP_EVIDENCE_CHECKPOINT = None
COLMAP_EVIDENCE_ARTIFACT_KEY_HARDWARE_POLICY = "required"
COLMAP_EVIDENCE_SHIPPING_STATUS = "experimental"

COLMAP_LOCAL_FEATURES_REPRODUCIBILITY_NOTES = (
    "V2L12.2 identifies the deterministic PyCOLMAP 4.2.0 CPU SIFT feature database as "
    "evidence.local_features through canonical V2 artifact-key and materialization contracts. "
    "The adapter remains experimental and this registry entry does not execute COLMAP."
)
COLMAP_PAIR_MATCHING_REPRODUCIBILITY_NOTES = (
    "V2L12.2 identifies the deterministic PyCOLMAP 4.2.0 CPU exhaustive raw-match database as "
    "evidence.pair_matches through canonical V2 artifact-key and materialization contracts. "
    "Geometric verification remains a distinct downstream evidence stage."
)
COLMAP_GEOMETRIC_VERIFICATION_REPRODUCIBILITY_NOTES = (
    "V2L12.2 identifies the deterministic PyCOLMAP 4.2.0 two-view verification database as "
    "evidence.geometric_verification through canonical V2 artifact-key and materialization "
    "contracts without promoting pair evidence to accepted scene geometry."
)

_EVIDENCE_OUTPUT_KINDS = frozenset(
    {
        LOCAL_FEATURES_KIND,
        PAIR_MATCHES_KIND,
        GEOMETRIC_VERIFICATION_KIND,
    }
)


@dataclass(frozen=True, slots=True)
class ColmapEvidenceArtifactPlan:
    """Pure content-addressed plan for one current-COLMAP evidence artifact."""

    output_kind: ArtifactKind
    input_fingerprints: tuple[ArtifactInputFingerprint, ...]
    producer: ArtifactProducerIdentity
    hardware_runtime: HardwareRuntimeIdentity
    artifact_key: ArtifactKey

    def __post_init__(self) -> None:
        if self.output_kind not in _EVIDENCE_OUTPUT_KINDS:
            raise ValueError("COLMAP evidence plan output_kind is unsupported")
        if not isinstance(self.input_fingerprints, tuple):
            raise TypeError("COLMAP evidence plan inputs must be an immutable tuple")
        if not all(isinstance(item, ArtifactInputFingerprint) for item in self.input_fingerprints):
            raise TypeError(
                "COLMAP evidence plan inputs must contain ArtifactInputFingerprint values"
            )
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("COLMAP evidence plan producer must be ArtifactProducerIdentity")
        if not isinstance(self.hardware_runtime, HardwareRuntimeIdentity):
            raise TypeError("COLMAP evidence plan hardware_runtime must be HardwareRuntimeIdentity")
        if not isinstance(self.artifact_key, ArtifactKey):
            raise TypeError("COLMAP evidence plan artifact_key must be ArtifactKey")

        expected = derive_artifact_key(
            ArtifactKeyMaterial(
                output_kind=self.output_kind,
                input_fingerprints=self.input_fingerprints,
                producer=self.producer,
                hardware_runtime=self.hardware_runtime,
            )
        )
        if self.artifact_key != expected:
            raise ValueError("COLMAP evidence plan artifact_key does not match its key material")


@dataclass(frozen=True, slots=True)
class ColmapEvidenceArtifactPublication:
    """Pure metadata/materialization publication for one already-produced evidence artifact."""

    metadata: ArtifactMetadata
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, ArtifactMetadata):
            raise TypeError("COLMAP evidence publication metadata must be ArtifactMetadata")
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "COLMAP evidence publication materialization must be "
                "ArtifactMaterializationMetadata"
            )
        if self.metadata.artifact_ref != self.materialization.artifact_ref:
            raise ValueError("COLMAP evidence publication artifact references must match")


def _validate_inputs(
    inputs: tuple[ArtifactInputFingerprint, ...],
    *,
    expected_kind: ArtifactKind,
    exact_count: int | None,
) -> tuple[ArtifactInputFingerprint, ...]:
    if not isinstance(inputs, tuple):
        raise TypeError("COLMAP evidence inputs must be an immutable tuple")
    if not inputs:
        raise ValueError("COLMAP evidence inputs must contain at least one fingerprint")
    if not all(isinstance(item, ArtifactInputFingerprint) for item in inputs):
        raise TypeError("COLMAP evidence inputs must contain ArtifactInputFingerprint values")
    if exact_count is not None and len(inputs) != exact_count:
        raise ValueError(f"COLMAP evidence stage requires exactly {exact_count} input fingerprint")
    if any(item.artifact_kind != expected_kind for item in inputs):
        raise ValueError(f"COLMAP evidence inputs must all have kind {expected_kind.value}")
    if len(inputs) != len(set(inputs)):
        raise ValueError("COLMAP evidence inputs cannot contain duplicate fingerprints")

    canonical = tuple(
        sorted(
            inputs,
            key=lambda item: (item.artifact_kind.value, item.sha256.value),
        )
    )
    if inputs != canonical:
        raise ValueError("COLMAP evidence inputs must be in canonical fingerprint order")
    return inputs


def _build_plan(
    *,
    output_kind: ArtifactKind,
    inputs: tuple[ArtifactInputFingerprint, ...],
    producer_implementation: str,
    configuration_sha256: Sha256Digest,
    hardware_runtime: HardwareRuntimeIdentity,
) -> ColmapEvidenceArtifactPlan:
    if not isinstance(configuration_sha256, Sha256Digest):
        raise TypeError("configuration_sha256 must be Sha256Digest")
    if not isinstance(hardware_runtime, HardwareRuntimeIdentity):
        raise TypeError("hardware_runtime must be HardwareRuntimeIdentity")

    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=producer_implementation,
            version=COLMAP_EVIDENCE_PRODUCER_VERSION,
            revision=None,
        ),
        configuration=ConfigurationIdentity(sha256=configuration_sha256),
    )
    artifact_key = derive_artifact_key(
        ArtifactKeyMaterial(
            output_kind=output_kind,
            input_fingerprints=inputs,
            producer=producer,
            hardware_runtime=hardware_runtime,
        )
    )
    return ColmapEvidenceArtifactPlan(
        output_kind=output_kind,
        input_fingerprints=inputs,
        producer=producer,
        hardware_runtime=hardware_runtime,
        artifact_key=artifact_key,
    )


def plan_colmap_local_features(
    inputs: tuple[ArtifactInputFingerprint, ...],
    *,
    configuration_sha256: Sha256Digest,
    hardware_runtime: HardwareRuntimeIdentity,
) -> ColmapEvidenceArtifactPlan:
    canonical = _validate_inputs(
        inputs,
        expected_kind=IMAGE_OBSERVATION_KIND,
        exact_count=None,
    )
    return _build_plan(
        output_kind=LOCAL_FEATURES_KIND,
        inputs=canonical,
        producer_implementation=COLMAP_LOCAL_FEATURES_PRODUCER_IMPLEMENTATION,
        configuration_sha256=configuration_sha256,
        hardware_runtime=hardware_runtime,
    )


def plan_colmap_pair_matches(
    inputs: tuple[ArtifactInputFingerprint, ...],
    *,
    configuration_sha256: Sha256Digest,
    hardware_runtime: HardwareRuntimeIdentity,
) -> ColmapEvidenceArtifactPlan:
    canonical = _validate_inputs(
        inputs,
        expected_kind=LOCAL_FEATURES_KIND,
        exact_count=1,
    )
    return _build_plan(
        output_kind=PAIR_MATCHES_KIND,
        inputs=canonical,
        producer_implementation=COLMAP_PAIR_MATCHING_PRODUCER_IMPLEMENTATION,
        configuration_sha256=configuration_sha256,
        hardware_runtime=hardware_runtime,
    )


def plan_colmap_geometric_verification(
    inputs: tuple[ArtifactInputFingerprint, ...],
    *,
    configuration_sha256: Sha256Digest,
    hardware_runtime: HardwareRuntimeIdentity,
) -> ColmapEvidenceArtifactPlan:
    canonical = _validate_inputs(
        inputs,
        expected_kind=PAIR_MATCHES_KIND,
        exact_count=1,
    )
    return _build_plan(
        output_kind=GEOMETRIC_VERIFICATION_KIND,
        inputs=canonical,
        producer_implementation=COLMAP_GEOMETRIC_VERIFICATION_PRODUCER_IMPLEMENTATION,
        configuration_sha256=configuration_sha256,
        hardware_runtime=hardware_runtime,
    )


def _result_database_identity(
    result: object,
    *,
    plan: ColmapEvidenceArtifactPlan,
    source_digest_attribute: str | None,
) -> tuple[Sha256Digest, int]:
    configuration_sha256 = getattr(result, "configuration_sha256", None)
    if not isinstance(configuration_sha256, Sha256Digest):
        raise TypeError("COLMAP donor result configuration_sha256 must be Sha256Digest")
    if configuration_sha256 != plan.producer.configuration.sha256:
        raise ValueError("COLMAP donor result configuration does not match artifact plan")

    environment = getattr(result, "environment", None)
    if getattr(environment, "pycolmap_version", None) != COLMAP_EVIDENCE_PRODUCER_VERSION:
        raise ValueError("COLMAP donor result PyCOLMAP version does not match artifact plan")

    if source_digest_attribute is not None:
        source_sha256 = getattr(result, source_digest_attribute, None)
        if not isinstance(source_sha256, Sha256Digest):
            raise TypeError(f"COLMAP donor result {source_digest_attribute} must be Sha256Digest")
        if len(plan.input_fingerprints) != 1:
            raise ValueError("COLMAP downstream evidence plan must have exactly one parent")
        if source_sha256 != plan.input_fingerprints[0].sha256:
            raise ValueError("COLMAP donor result source database does not match artifact plan")

    database_sha256 = getattr(result, "database_sha256", None)
    if not isinstance(database_sha256, Sha256Digest):
        raise TypeError("COLMAP donor result database_sha256 must be Sha256Digest")
    database_byte_length = getattr(result, "database_byte_length", None)
    if (
        isinstance(database_byte_length, bool)
        or not isinstance(database_byte_length, int)
        or database_byte_length <= 0
    ):
        raise ValueError("COLMAP donor result database_byte_length must be a positive integer")
    return database_sha256, database_byte_length


def _publish(
    *,
    project_id: SceneProjectId,
    artifact_ref: ArtifactRef,
    plan: ColmapEvidenceArtifactPlan,
    result: object,
    relative_path: str,
    expected_output_kind: ArtifactKind,
    source_digest_attribute: str | None,
) -> ColmapEvidenceArtifactPublication:
    if not isinstance(project_id, SceneProjectId):
        raise TypeError("project_id must be SceneProjectId")
    if not isinstance(artifact_ref, ArtifactRef):
        raise TypeError("artifact_ref must be ArtifactRef")
    if not isinstance(plan, ColmapEvidenceArtifactPlan):
        raise TypeError("plan must be ColmapEvidenceArtifactPlan")
    if plan.output_kind != expected_output_kind:
        raise ValueError("COLMAP evidence plan output kind does not match publication stage")
    if artifact_ref.artifact_kind != expected_output_kind:
        raise ValueError("COLMAP evidence ArtifactRef kind does not match publication stage")

    database_sha256, database_byte_length = _result_database_identity(
        result,
        plan=plan,
        source_digest_attribute=source_digest_attribute,
    )
    metadata = ArtifactMetadata(
        project_id=project_id,
        artifact_ref=artifact_ref,
        artifact_key=plan.artifact_key,
        producer=plan.producer,
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )
    materialization = ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path=relative_path,
                sha256=database_sha256,
                byte_length=database_byte_length,
            ),
        ),
    )
    return ColmapEvidenceArtifactPublication(
        metadata=metadata,
        materialization=materialization,
    )


def publish_colmap_local_features(
    *,
    project_id: SceneProjectId,
    artifact_ref: ArtifactRef,
    plan: ColmapEvidenceArtifactPlan,
    result: ColmapFeatureExtractionResult,
    relative_path: str,
) -> ColmapEvidenceArtifactPublication:
    return _publish(
        project_id=project_id,
        artifact_ref=artifact_ref,
        plan=plan,
        result=cast(object, result),
        relative_path=relative_path,
        expected_output_kind=LOCAL_FEATURES_KIND,
        source_digest_attribute=None,
    )


def publish_colmap_pair_matches(
    *,
    project_id: SceneProjectId,
    artifact_ref: ArtifactRef,
    plan: ColmapEvidenceArtifactPlan,
    result: ColmapPairMatchingResult,
    relative_path: str,
) -> ColmapEvidenceArtifactPublication:
    return _publish(
        project_id=project_id,
        artifact_ref=artifact_ref,
        plan=plan,
        result=cast(object, result),
        relative_path=relative_path,
        expected_output_kind=PAIR_MATCHES_KIND,
        source_digest_attribute="source_feature_database_sha256",
    )


def publish_colmap_geometric_verification(
    *,
    project_id: SceneProjectId,
    artifact_ref: ArtifactRef,
    plan: ColmapEvidenceArtifactPlan,
    result: ColmapGeometricVerificationResult,
    relative_path: str,
) -> ColmapEvidenceArtifactPublication:
    return _publish(
        project_id=project_id,
        artifact_ref=artifact_ref,
        plan=plan,
        result=cast(object, result),
        relative_path=relative_path,
        expected_output_kind=GEOMETRIC_VERIFICATION_KIND,
        source_digest_attribute="source_matching_database_sha256",
    )
