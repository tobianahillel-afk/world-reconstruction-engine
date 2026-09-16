from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final

from wre.domain.artifacts import ArtifactKind
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import (
    ArtifactProducerIdentity,
    CheckpointIdentity,
    ModelIdentity,
)

ARTIFACT_KEY_SCHEMA_VERSION: Final = 1
_ARTIFACT_KEY_DOMAIN: Final = "wre.artifact-key"


@dataclass(frozen=True, slots=True)
class ArtifactInputFingerprint:
    """Exact typed content identity for one computation input."""

    artifact_kind: ArtifactKind
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_kind, ArtifactKind):
            raise TypeError("artifact_input.artifact_kind must be ArtifactKind")
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("artifact_input.sha256 must be Sha256Digest")


@dataclass(frozen=True, slots=True)
class ArtifactKeyMaterial:
    """Immutable ordered computation identity used to derive an ArtifactKey."""

    output_kind: ArtifactKind
    input_fingerprints: tuple[ArtifactInputFingerprint, ...]
    producer: ArtifactProducerIdentity
    hardware_runtime: HardwareRuntimeIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output_kind, ArtifactKind):
            raise TypeError("artifact_key_material.output_kind must be ArtifactKind")
        if not isinstance(self.input_fingerprints, tuple):
            raise TypeError("artifact_key_material.input_fingerprints must be an immutable tuple")
        if not all(
            isinstance(fingerprint, ArtifactInputFingerprint)
            for fingerprint in self.input_fingerprints
        ):
            raise TypeError(
                "artifact_key_material.input_fingerprints must contain only "
                "ArtifactInputFingerprint values"
            )
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("artifact_key_material.producer must be ArtifactProducerIdentity")
        if self.hardware_runtime is not None and not isinstance(
            self.hardware_runtime, HardwareRuntimeIdentity
        ):
            raise TypeError(
                "artifact_key_material.hardware_runtime must be "
                "HardwareRuntimeIdentity when present"
            )


@dataclass(frozen=True, slots=True)
class ArtifactKey:
    """Version-1 content-derived computation identity for an artifact."""

    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("artifact_key.sha256 must be Sha256Digest")

    def __str__(self) -> str:
        return f"artifact-key:v{ARTIFACT_KEY_SCHEMA_VERSION}:{self.sha256}"


def _model_payload(model: ModelIdentity | None) -> dict[str, str | None] | None:
    if model is None:
        return None
    return {
        "name": model.name,
        "version": model.version,
        "revision": model.revision,
    }


def _checkpoint_payload(
    checkpoint: CheckpointIdentity | None,
) -> dict[str, str] | None:
    if checkpoint is None:
        return None
    return {
        "identifier": checkpoint.identifier,
        "sha256": str(checkpoint.sha256),
    }


def canonical_artifact_key_bytes(material: ArtifactKeyMaterial) -> bytes:
    """Serialize version-1 key material into stable canonical UTF-8 JSON bytes."""

    if not isinstance(material, ArtifactKeyMaterial):
        raise TypeError("material must be ArtifactKeyMaterial")

    producer = material.producer
    payload: dict[str, object] = {
        "domain": _ARTIFACT_KEY_DOMAIN,
        "schema_version": ARTIFACT_KEY_SCHEMA_VERSION,
        "output_kind": str(material.output_kind),
        "inputs": [
            {
                "artifact_kind": str(fingerprint.artifact_kind),
                "sha256": str(fingerprint.sha256),
            }
            for fingerprint in material.input_fingerprints
        ],
        "producer": {
            "implementation": producer.producer.implementation,
            "version": producer.producer.version,
            "revision": producer.producer.revision,
        },
        "configuration_sha256": str(producer.configuration.sha256),
        "model": _model_payload(producer.model),
        "checkpoint": _checkpoint_payload(producer.checkpoint),
    }
    if material.hardware_runtime is not None:
        payload["hardware_runtime_sha256"] = str(material.hardware_runtime.sha256)

    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical.encode("utf-8")


def derive_artifact_key(material: ArtifactKeyMaterial) -> ArtifactKey:
    """Derive a deterministic version-1 key without consulting external state."""

    digest = hashlib.sha256(canonical_artifact_key_bytes(material)).hexdigest()
    return ArtifactKey(sha256=Sha256Digest(digest))
