from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId


@dataclass(frozen=True, slots=True)
class MediaProfile:
    """Immutable profile boundary for one observation and exact derived evidence."""

    observation_id: ObservationId
    evidence_artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("media_profile.observation_id must be ObservationId")
        if not isinstance(self.evidence_artifacts, tuple):
            raise TypeError("media_profile.evidence_artifacts must be an immutable tuple")
        if any(not isinstance(item, ArtifactRef) for item in self.evidence_artifacts):
            raise TypeError("media_profile.evidence_artifacts members must be ArtifactRef")

        evidence_keys = [
            (item.artifact_id.value, item.artifact_kind.value) for item in self.evidence_artifacts
        ]
        if len(set(self.evidence_artifacts)) != len(self.evidence_artifacts):
            raise ValueError("media_profile.evidence_artifacts members must be unique")
        if evidence_keys != sorted(evidence_keys):
            raise ValueError(
                "media_profile.evidence_artifacts must use canonical ArtifactId/ArtifactKind order"
            )

        kinds_by_artifact_id: dict[str, str] = {}
        for item in self.evidence_artifacts:
            artifact_id = item.artifact_id.value
            artifact_kind = item.artifact_kind.value
            previous_kind = kinds_by_artifact_id.setdefault(artifact_id, artifact_kind)
            if previous_kind != artifact_kind:
                raise ValueError(
                    "media_profile.evidence_artifacts must not declare conflicting "
                    "ArtifactKind values for one ArtifactId"
                )
