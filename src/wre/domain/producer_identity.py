from __future__ import annotations

from dataclasses import dataclass

from wre.domain.observations import Sha256Digest
from wre.domain.runs import ProducerRef


def _require_non_blank_text(value: object, context: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be non-blank")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    name: str
    version: str
    revision: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank_text(self.name, "model.name")
        _require_non_blank_text(self.version, "model.version")
        if self.revision is not None:
            _require_non_blank_text(self.revision, "model.revision")


@dataclass(frozen=True, slots=True)
class CheckpointIdentity:
    identifier: str
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        _require_non_blank_text(self.identifier, "checkpoint.identifier")
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("checkpoint.sha256 must be Sha256Digest")


@dataclass(frozen=True, slots=True)
class ConfigurationIdentity:
    sha256: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.sha256, Sha256Digest):
            raise TypeError("configuration.sha256 must be Sha256Digest")


@dataclass(frozen=True, slots=True)
class ArtifactProducerIdentity:
    producer: ProducerRef
    configuration: ConfigurationIdentity
    model: ModelIdentity | None = None
    checkpoint: CheckpointIdentity | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.producer, ProducerRef):
            raise TypeError("artifact_producer.producer must be ProducerRef")
        if not isinstance(self.configuration, ConfigurationIdentity):
            raise TypeError(
                "artifact_producer.configuration must be ConfigurationIdentity"
            )
        if self.model is not None and not isinstance(self.model, ModelIdentity):
            raise TypeError("artifact_producer.model must be ModelIdentity when present")
        if self.checkpoint is not None and not isinstance(
            self.checkpoint, CheckpointIdentity
        ):
            raise TypeError(
                "artifact_producer.checkpoint must be CheckpointIdentity when present"
            )
        if self.checkpoint is not None and self.model is None:
            raise ValueError("artifact_producer.checkpoint requires model identity")
