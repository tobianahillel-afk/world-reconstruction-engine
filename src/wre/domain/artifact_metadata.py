from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifact_keys import ArtifactKey
from wre.domain.artifacts import ArtifactRef
from wre.domain.producer_identity import ArtifactProducerIdentity
from wre.domain.projects import SceneProjectId
from wre.domain.provenance import ProvenanceClass


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Minimal immutable metadata persisted for one V2 artifact identity."""

    project_id: SceneProjectId
    artifact_ref: ArtifactRef
    artifact_key: ArtifactKey
    producer: ArtifactProducerIdentity
    provenance_class: ProvenanceClass

    def __post_init__(self) -> None:
        if not isinstance(self.project_id, SceneProjectId):
            raise TypeError("artifact_metadata.project_id must be SceneProjectId")
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("artifact_metadata.artifact_ref must be ArtifactRef")
        if not isinstance(self.artifact_key, ArtifactKey):
            raise TypeError("artifact_metadata.artifact_key must be ArtifactKey")
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("artifact_metadata.producer must be ArtifactProducerIdentity")
        if not isinstance(self.provenance_class, ProvenanceClass):
            raise TypeError("artifact_metadata.provenance_class must be ProvenanceClass")
