from __future__ import annotations

from dataclasses import dataclass

from wre.domain.artifacts import ArtifactRef
from wre.domain.pair_candidates import PairCandidate
from wre.domain.scene_clusters import SceneRelationDisposition, SceneRelationHypothesis
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationResult,
    ColmapPairGeometryEvidence,
)

_SUPPORTED_CONFIGURATIONS = frozenset(
    {
        "CALIBRATED",
        "CALIBRATED_RIG",
        "UNCALIBRATED",
        "PLANAR",
        "PANORAMIC",
        "PLANAR_OR_PANORAMIC",
    }
)


def _pair_key(candidate: PairCandidate) -> tuple[str, str]:
    return (candidate.observation_id1.value, candidate.observation_id2.value)


def _geometry_key(geometry: ColmapPairGeometryEvidence) -> tuple[str, str]:
    return (geometry.observation_id1.value, geometry.observation_id2.value)


def _artifact_ref_key(ref: ArtifactRef) -> tuple[str, str]:
    return (ref.artifact_kind.value, ref.artifact_id.value)


@dataclass(frozen=True, slots=True)
class ColmapSceneRelationAdapterInput:
    """Explicit pair proposals plus one already-produced COLMAP verification artifact."""

    candidates: tuple[PairCandidate, ...]
    verification: ColmapGeometricVerificationResult
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise TypeError("colmap_scene_relation_adapter.candidates must be an immutable tuple")
        if any(not isinstance(candidate, PairCandidate) for candidate in self.candidates):
            raise TypeError(
                "colmap_scene_relation_adapter.candidates must contain only PairCandidate values"
            )
        candidate_keys = tuple(_pair_key(candidate) for candidate in self.candidates)
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("colmap_scene_relation_adapter.candidates must be unique")
        if candidate_keys != tuple(sorted(candidate_keys)):
            raise ValueError(
                "colmap_scene_relation_adapter.candidates must use canonical pair order"
            )
        if not isinstance(self.verification, ColmapGeometricVerificationResult):
            raise TypeError(
                "colmap_scene_relation_adapter.verification must be "
                "ColmapGeometricVerificationResult"
            )
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("colmap_scene_relation_adapter.evidence_ref must be ArtifactRef")

        membership = {
            observation_id.value
            for observation_id in self.verification.provenance.source_observation_ids
        }
        if any(
            candidate.observation_id1.value not in membership
            or candidate.observation_id2.value not in membership
            for candidate in self.candidates
        ):
            raise ValueError(
                "colmap_scene_relation_adapter candidate endpoints must belong to "
                "verification provenance observations"
            )


def _candidate_evidence_refs(
    candidate: PairCandidate,
    verification_ref: ArtifactRef,
) -> tuple[ArtifactRef, ...]:
    by_key: dict[tuple[str, str], ArtifactRef] = {}
    for source in candidate.sources:
        for evidence_ref in source.evidence_refs:
            by_key.setdefault(_artifact_ref_key(evidence_ref), evidence_ref)
    by_key.setdefault(_artifact_ref_key(verification_ref), verification_ref)
    return tuple(by_key[key] for key in sorted(by_key))


def _disposition(
    geometry: ColmapPairGeometryEvidence | None,
) -> SceneRelationDisposition:
    if geometry is None:
        return SceneRelationDisposition.UNRESOLVED
    if geometry.inlier_count <= 0:
        return SceneRelationDisposition.UNRESOLVED
    if geometry.configuration in _SUPPORTED_CONFIGURATIONS:
        return SceneRelationDisposition.SUPPORTED
    return SceneRelationDisposition.UNRESOLVED


def adapt_colmap_scene_relations(
    adapter_input: ColmapSceneRelationAdapterInput,
) -> tuple[SceneRelationHypothesis, ...]:
    """Adapt retained COLMAP two-view evidence without executing matching or verification."""

    if not isinstance(adapter_input, ColmapSceneRelationAdapterInput):
        raise TypeError("adapter_input must be ColmapSceneRelationAdapterInput")

    geometry_by_pair = {
        _geometry_key(geometry): geometry for geometry in adapter_input.verification.geometries
    }
    return tuple(
        SceneRelationHypothesis(
            observation_id1=candidate.observation_id1,
            observation_id2=candidate.observation_id2,
            disposition=_disposition(geometry_by_pair.get(_pair_key(candidate))),
            evidence_refs=_candidate_evidence_refs(candidate, adapter_input.evidence_ref),
        )
        for candidate in adapter_input.candidates
    )
