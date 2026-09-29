from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from wre.domain.artifact_materialization import ArtifactMaterializationMetadata
from wre.domain.metrics import (
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
)
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.domain.surfaces import SurfaceModel, SurfaceRepresentationName

_SURFACE_SUITABILITY_IMPLEMENTATION = "wre.reconstruction.surface_suitability"
_SURFACE_SUITABILITY_VERSION = "1"
_SURFACE_SUITABILITY_REVISION = "v2l16.3"
_MESH_REPRESENTATION = SurfaceRepresentationName("mesh")

_EVALUATOR_CONFIGURATION_DOCUMENT = {
    "schema_version": 1,
    "representation": "mesh",
    "finite_vertex_denominator": "vertex_count",
    "degenerate_triangle_denominator": "triangle_count",
    "edge_ratio_denominator": "unique_edge_count",
    "zero_unique_edges": "omit_edge_ratio_observations",
    "largest_component_denominator": "triangle_count",
    "connected_components": "descriptive_count",
    "intended_uses_affect_metrics": False,
    "payload_bytes_opened": False,
    "quality_thresholds": "none",
    "quality_decision": "none",
    "watertightness_claim": "none",
    "measurement_accuracy_claim": "none",
}
_EVALUATOR_CONFIGURATION_BYTES = json.dumps(
    _EVALUATOR_CONFIGURATION_DOCUMENT,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")

SURFACE_SUITABILITY_EVALUATOR = ArtifactProducerIdentity(
    producer=ProducerRef(
        implementation=_SURFACE_SUITABILITY_IMPLEMENTATION,
        version=_SURFACE_SUITABILITY_VERSION,
        revision=_SURFACE_SUITABILITY_REVISION,
    ),
    configuration=ConfigurationIdentity(
        sha256=Sha256Digest(hashlib.sha256(_EVALUATOR_CONFIGURATION_BYTES).hexdigest())
    ),
)

_SURFACE_DIMENSION = MetricDimension("geometry.surface")
_RATIO_UNIT = MetricUnit("ratio")
_COUNT_UNIT = MetricUnit("count")

BOUNDARY_EDGE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.boundary_edge_ratio"),
    dimension=_SURFACE_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("unique_edge_ratio"),
)
CONNECTED_COMPONENT_COUNT_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.connected_component_count"),
    dimension=_SURFACE_DIMENSION,
    unit=_COUNT_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("count"),
)
DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.degenerate_triangle_ratio"),
    dimension=_SURFACE_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("triangle_ratio"),
)
FINITE_VERTEX_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.finite_vertex_ratio"),
    dimension=_SURFACE_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("vertex_ratio"),
)
LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.largest_component_triangle_ratio"),
    dimension=_SURFACE_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("triangle_ratio"),
)
NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.surface.non_manifold_edge_ratio"),
    dimension=_SURFACE_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("unique_edge_ratio"),
)


def _require_int(value: object, context: str, *, positive: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be int")
    if positive:
        if value <= 0:
            raise ValueError(f"{context} must be positive")
    elif value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class SurfaceMeshInspection:
    """Caller-supplied descriptive topology evidence for one retained mesh surface.

    V2L16.3 deliberately does not open or parse the retained mesh payload. The producer
    identifies how the caller obtained these counts; the evaluator only validates and
    summarizes the supplied evidence.
    """

    surface_model: SurfaceModel
    materialization: ArtifactMaterializationMetadata
    producer: ArtifactProducerIdentity
    vertex_count: int
    triangle_count: int
    finite_vertex_count: int
    degenerate_triangle_count: int
    unique_edge_count: int
    boundary_edge_count: int
    non_manifold_edge_count: int
    connected_component_count: int
    largest_component_triangle_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.surface_model, SurfaceModel):
            raise TypeError("surface_mesh_inspection.surface_model must be SurfaceModel")
        if self.surface_model.representation != _MESH_REPRESENTATION:
            raise ValueError(
                "surface_mesh_inspection.surface_model must use representation mesh"
            )
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "surface_mesh_inspection.materialization must be ArtifactMaterializationMetadata"
            )
        if self.materialization.artifact_ref != self.surface_model.artifact_ref:
            raise ValueError(
                "surface_mesh_inspection.materialization artifact_ref must exactly match "
                "surface_model.artifact_ref"
            )
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("surface_mesh_inspection.producer must be ArtifactProducerIdentity")

        vertex_count = _require_int(
            self.vertex_count,
            "surface_mesh_inspection.vertex_count",
            positive=True,
        )
        triangle_count = _require_int(
            self.triangle_count,
            "surface_mesh_inspection.triangle_count",
            positive=True,
        )
        finite_vertex_count = _require_int(
            self.finite_vertex_count,
            "surface_mesh_inspection.finite_vertex_count",
        )
        degenerate_triangle_count = _require_int(
            self.degenerate_triangle_count,
            "surface_mesh_inspection.degenerate_triangle_count",
        )
        unique_edge_count = _require_int(
            self.unique_edge_count,
            "surface_mesh_inspection.unique_edge_count",
        )
        boundary_edge_count = _require_int(
            self.boundary_edge_count,
            "surface_mesh_inspection.boundary_edge_count",
        )
        non_manifold_edge_count = _require_int(
            self.non_manifold_edge_count,
            "surface_mesh_inspection.non_manifold_edge_count",
        )
        connected_component_count = _require_int(
            self.connected_component_count,
            "surface_mesh_inspection.connected_component_count",
            positive=True,
        )
        largest_component_triangle_count = _require_int(
            self.largest_component_triangle_count,
            "surface_mesh_inspection.largest_component_triangle_count",
            positive=True,
        )

        if finite_vertex_count > vertex_count:
            raise ValueError(
                "surface_mesh_inspection.finite_vertex_count cannot exceed vertex_count"
            )
        if degenerate_triangle_count > triangle_count:
            raise ValueError(
                "surface_mesh_inspection.degenerate_triangle_count cannot exceed triangle_count"
            )
        if boundary_edge_count > unique_edge_count:
            raise ValueError(
                "surface_mesh_inspection.boundary_edge_count cannot exceed unique_edge_count"
            )
        if non_manifold_edge_count > unique_edge_count:
            raise ValueError(
                "surface_mesh_inspection.non_manifold_edge_count cannot exceed unique_edge_count"
            )
        if connected_component_count > triangle_count:
            raise ValueError(
                "surface_mesh_inspection.connected_component_count cannot exceed triangle_count"
            )
        if largest_component_triangle_count > triangle_count:
            raise ValueError(
                "surface_mesh_inspection.largest_component_triangle_count cannot exceed "
                "triangle_count"
            )


def _observation(
    descriptor: MetricDescriptor,
    value: float,
    provenance: MetricProvenance,
) -> MetricObservation:
    return MetricObservation(
        descriptor=descriptor,
        value=float(value),
        provenance=provenance,
    )


def _surface_metrics(inspection: SurfaceMeshInspection) -> MetricVector:
    provenance = MetricProvenance(
        evaluator=SURFACE_SUITABILITY_EVALUATOR,
        input_artifacts=(inspection.surface_model.artifact_ref,),
    )

    observations = [
        _observation(
            CONNECTED_COMPONENT_COUNT_DESCRIPTOR,
            float(inspection.connected_component_count),
            provenance,
        ),
        _observation(
            DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR,
            inspection.degenerate_triangle_count / inspection.triangle_count,
            provenance,
        ),
        _observation(
            FINITE_VERTEX_RATIO_DESCRIPTOR,
            inspection.finite_vertex_count / inspection.vertex_count,
            provenance,
        ),
        _observation(
            LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR,
            inspection.largest_component_triangle_count / inspection.triangle_count,
            provenance,
        ),
    ]
    if inspection.unique_edge_count:
        observations.extend(
            (
                _observation(
                    BOUNDARY_EDGE_RATIO_DESCRIPTOR,
                    inspection.boundary_edge_count / inspection.unique_edge_count,
                    provenance,
                ),
                _observation(
                    NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR,
                    inspection.non_manifold_edge_count / inspection.unique_edge_count,
                    provenance,
                ),
            )
        )

    return MetricVector(
        observations=tuple(
            sorted(
                observations,
                key=lambda observation: observation.descriptor.name.value,
            )
        )
    )


@dataclass(frozen=True, slots=True)
class SurfaceSuitabilityReport:
    """Canonical descriptive topology metrics over one SurfaceMeshInspection."""

    inspection: SurfaceMeshInspection
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.inspection, SurfaceMeshInspection):
            raise TypeError("surface_suitability_report.inspection must be SurfaceMeshInspection")
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("surface_suitability_report.metrics must be MetricVector")

        expected = _surface_metrics(self.inspection)
        if self.metrics != expected:
            raise ValueError(
                "surface_suitability_report.metrics must exactly match canonical "
                "descriptive surface metrics"
            )


def evaluate_surface_suitability(
    inspection: SurfaceMeshInspection,
) -> SurfaceSuitabilityReport:
    """Summarize supplied mesh topology evidence without making a suitability decision."""

    if not isinstance(inspection, SurfaceMeshInspection):
        raise TypeError("inspection must be SurfaceMeshInspection")

    return SurfaceSuitabilityReport(
        inspection=inspection,
        metrics=_surface_metrics(inspection),
    )


__all__ = [
    "BOUNDARY_EDGE_RATIO_DESCRIPTOR",
    "CONNECTED_COMPONENT_COUNT_DESCRIPTOR",
    "DEGENERATE_TRIANGLE_RATIO_DESCRIPTOR",
    "FINITE_VERTEX_RATIO_DESCRIPTOR",
    "LARGEST_COMPONENT_TRIANGLE_RATIO_DESCRIPTOR",
    "NON_MANIFOLD_EDGE_RATIO_DESCRIPTOR",
    "SURFACE_SUITABILITY_EVALUATOR",
    "SurfaceMeshInspection",
    "SurfaceSuitabilityReport",
    "evaluate_surface_suitability",
]
