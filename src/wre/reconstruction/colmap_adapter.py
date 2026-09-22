from __future__ import annotations

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.artifacts import ArtifactKind, ArtifactRef
from wre.reconstruction.colmap_environment import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
)

COLMAP_PRECISION_ADAPTER_ID = "colmap.precision_geometry"
COLMAP_PRECISION_CAPABILITY_NAME = AdapterCapabilityName("geometry.precision_sfm")
COLMAP_PRECISION_INPUT_KIND = ArtifactKind("image.observation")
COLMAP_PRECISION_OUTPUT_KINDS = frozenset(
    {
        ArtifactKind("geometry.camera_solution"),
        ArtifactKind("geometry.point_map"),
        ArtifactKind("geometry.solution"),
    }
)
COLMAP_PRECISION_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_PRECISION_CAPABILITY_NAME,
    input_kinds=frozenset({COLMAP_PRECISION_INPUT_KIND}),
    output_kinds=COLMAP_PRECISION_OUTPUT_KINDS,
)

COLMAP_PRECISION_DEPENDENCY_REF = "colmap"
COLMAP_PRECISION_PYCOLMAP_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_PRECISION_COLMAP_VERSION = SUPPORTED_COLMAP_VERSION
COLMAP_PRECISION_PRODUCER_IMPLEMENTATION = "wre.reconstruction.colmap_adapter"
COLMAP_PRECISION_PRODUCER_VERSION = "1"
COLMAP_PRECISION_MODEL = None
COLMAP_PRECISION_CHECKPOINT = None
COLMAP_PRECISION_ARTIFACT_KEY_HARDWARE_POLICY = "required"
COLMAP_PRECISION_SHIPPING_STATUS = "experimental"
COLMAP_PRECISION_REPRODUCIBILITY_NOTES = (
    "V2L12.1 declares the current PyCOLMAP 4.2.0 / COLMAP 4.2.0 precision-geometry "
    "capability through canonical V2 artifact kinds only. It does not execute COLMAP or "
    "materialize CameraSolution, PointMap, or GeometrySolution values."
)


def normalize_colmap_precision_inputs(
    inputs: tuple[ArtifactRef, ...],
) -> tuple[ArtifactRef, ...]:
    """Validate canonical ordered image-observation artifact inputs without rewriting them."""

    if not isinstance(inputs, tuple):
        raise TypeError("COLMAP precision inputs must be an immutable tuple")
    if not inputs:
        raise ValueError("COLMAP precision inputs must contain at least one artifact")
    if any(not isinstance(item, ArtifactRef) for item in inputs):
        raise TypeError("COLMAP precision inputs must contain only ArtifactRef values")
    if any(item.artifact_kind != COLMAP_PRECISION_INPUT_KIND for item in inputs):
        raise ValueError("COLMAP precision inputs must all have kind image.observation")
    if len(inputs) != len(set(inputs)):
        raise ValueError("COLMAP precision inputs cannot contain duplicate ArtifactRef values")

    canonical = tuple(
        sorted(
            inputs,
            key=lambda item: (item.artifact_kind.value, item.artifact_id.value),
        )
    )
    if inputs != canonical:
        raise ValueError("COLMAP precision inputs must be in canonical ArtifactRef order")

    return inputs
