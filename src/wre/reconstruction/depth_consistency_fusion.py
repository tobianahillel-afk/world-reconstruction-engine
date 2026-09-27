from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from wre.domain.artifacts import ArtifactRef
from wre.domain.depth_fields import DepthField
from wre.reconstruction.dense_depth import DenseDepthArtifact


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


def _validate_inputs(inputs: object) -> None:
    if not isinstance(inputs, tuple):
        raise TypeError("depth_consistency_fusion.inputs must be an immutable tuple")
    if len(inputs) < 2:
        raise ValueError("depth_consistency_fusion.inputs must contain at least two artifacts")
    if any(not isinstance(item, DenseDepthArtifact) for item in inputs):
        raise TypeError("depth_consistency_fusion.inputs members must be DenseDepthArtifact")

    artifacts = tuple(inputs)
    identities = tuple(_artifact_key(item.artifact_ref) for item in artifacts)
    if len(identities) != len(set(identities)):
        raise ValueError("depth_consistency_fusion.inputs must have unique artifact identities")
    if identities != tuple(sorted(identities)):
        raise ValueError(
            "depth_consistency_fusion.inputs must use canonical ArtifactId/ArtifactKind order"
        )

    source_geometry = artifacts[0].source_geometry
    if any(item.source_geometry != source_geometry for item in artifacts[1:]):
        raise ValueError(
            "depth_consistency_fusion.inputs must share the exact source GeometrySolutionCandidate"
        )

    fields_by_camera: dict[object, list[DepthField]] = {}
    for artifact in artifacts:
        for depth_field in artifact.depth_fields:
            fields_by_camera.setdefault(depth_field.camera_solution_id, []).append(depth_field)

    for fields in fields_by_camera.values():
        if len(fields) < 2:
            continue
        first = fields[0]
        for field in fields[1:]:
            if field.observation_id != first.observation_id:
                raise ValueError("overlapping depth inputs must use the same ObservationId")
            if field.dimensions != first.dimensions:
                raise ValueError("overlapping depth inputs must use the same ImageDimensions")
            if field.depth_value_convention != first.depth_value_convention:
                raise ValueError(
                    "overlapping depth inputs must use the same DepthValueConventionName"
                )


@dataclass(frozen=True, slots=True)
class DepthConsistencyFusionRequest:
    """Immutable evidence boundary for depth-consistency fusion.

    The request groups already-canonical dense-depth hypotheses that share one
    complete geometry source. It validates compatibility only and performs no
    numeric fusion, depth conversion, alignment, scoring, or hole filling.
    """

    inputs: tuple[DenseDepthArtifact, ...]

    def __post_init__(self) -> None:
        _validate_inputs(self.inputs)


def _required_ancestry(
    request: DepthConsistencyFusionRequest,
) -> set[tuple[str, str]]:
    required: set[tuple[str, str]] = set()
    for artifact in request.inputs:
        required.add(_artifact_key(artifact.artifact_ref))
        required.update(_artifact_key(item) for item in artifact.source_artifacts)
    return required


def _fields_by_camera(
    request: DepthConsistencyFusionRequest,
) -> dict[object, tuple[DepthField, ...]]:
    grouped: dict[object, list[DepthField]] = {}
    for artifact in request.inputs:
        for field in artifact.depth_fields:
            grouped.setdefault(field.camera_solution_id, []).append(field)
    return {key: tuple(value) for key, value in grouped.items()}


@dataclass(frozen=True, slots=True)
class DepthConsistencyFusionResult:
    """Validated fused dense-depth evidence without a built-in fusion algorithm."""

    request: DepthConsistencyFusionRequest
    fused_depth: DenseDepthArtifact

    def __post_init__(self) -> None:
        if not isinstance(self.request, DepthConsistencyFusionRequest):
            raise TypeError(
                "depth_consistency_fusion_result.request must be DepthConsistencyFusionRequest"
            )
        if not isinstance(self.fused_depth, DenseDepthArtifact):
            raise TypeError(
                "depth_consistency_fusion_result.fused_depth must be DenseDepthArtifact"
            )

        source_geometry = self.request.inputs[0].source_geometry
        if self.fused_depth.source_geometry != source_geometry:
            raise ValueError("fused depth must preserve the exact source GeometrySolutionCandidate")

        input_artifact_identities = {
            _artifact_key(artifact.artifact_ref) for artifact in self.request.inputs
        }
        if _artifact_key(self.fused_depth.artifact_ref) in input_artifact_identities:
            raise ValueError("fused depth must use a new DenseDepthArtifact identity")

        input_depth_ids = {
            field.depth_field_id
            for artifact in self.request.inputs
            for field in artifact.depth_fields
        }
        support_by_camera = _fields_by_camera(self.request)

        for fused_field in self.fused_depth.depth_fields:
            if fused_field.depth_field_id in input_depth_ids:
                raise ValueError("fused DepthFieldIds must be new evidence")

            supporting_fields = support_by_camera.get(fused_field.camera_solution_id)
            if not supporting_fields:
                raise ValueError("fused DepthField support must already exist in an input artifact")

            first = supporting_fields[0]
            if fused_field.observation_id != first.observation_id:
                raise ValueError("fused DepthField ObservationId must match input support")
            if fused_field.dimensions != first.dimensions:
                raise ValueError("fused DepthField ImageDimensions must match input support")

            conventions = {field.depth_value_convention for field in supporting_fields}
            if len(conventions) != 1:
                raise ValueError("input support for a fused camera must use one depth convention")
            if fused_field.depth_value_convention not in conventions:
                raise ValueError(
                    "fused DepthField must preserve the input DepthValueConventionName"
                )

        fused_ancestry = {_artifact_key(item) for item in self.fused_depth.source_artifacts}
        required_ancestry = _required_ancestry(self.request)
        if fused_ancestry != required_ancestry:
            raise ValueError(
                "fused depth source_artifacts must equal the canonical union of every "
                "input artifact and ancestry"
            )


class DepthConsistencyFusionAdapter(Protocol):
    def fuse(
        self,
        request: DepthConsistencyFusionRequest,
    ) -> DepthConsistencyFusionResult: ...


def build_depth_consistency_fusion_result(
    request: DepthConsistencyFusionRequest,
    fused_depth: DenseDepthArtifact,
) -> DepthConsistencyFusionResult:
    """Build one validated fusion result without executing a fusion solver."""

    return DepthConsistencyFusionResult(
        request=request,
        fused_depth=fused_depth,
    )


__all__ = [
    "DepthConsistencyFusionAdapter",
    "DepthConsistencyFusionRequest",
    "DepthConsistencyFusionResult",
    "build_depth_consistency_fusion_result",
]
