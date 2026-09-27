from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from wre.domain.camera_solutions import CameraSolutionId
from wre.domain.depth_fields import DepthField, DepthFieldId
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
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.dense_depth import DenseDepthArtifact

DENSE_DEPTH_COVERAGE_IMPLEMENTATION = "wre.reconstruction.dense_depth_coverage"
DENSE_DEPTH_COVERAGE_VERSION = "1"
DENSE_DEPTH_COVERAGE_REVISION = "v2l15.5"

_EVALUATOR_CONFIGURATION_DOCUMENT = {
    "schema_version": 1,
    "camera_support_denominator": "source_geometry_camera_solution_count",
    "pixel_support_denominator": "emitted_depth_field_pixel_count",
    "holes": "depth_field_validity_false",
    "confidence_availability": "depth_field_confidence_present",
    "confidence_aggregation": "arithmetic_mean_valid_pixels_with_explicit_confidence",
    "missing_confidence": "omit_valid_confidence_mean",
    "quality_thresholds": "none",
}
_EVALUATOR_CONFIGURATION_BYTES = json.dumps(
    _EVALUATOR_CONFIGURATION_DOCUMENT,
    ensure_ascii=True,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
).encode("utf-8")
DENSE_DEPTH_COVERAGE_EVALUATOR = ArtifactProducerIdentity(
    producer=ProducerRef(
        implementation=DENSE_DEPTH_COVERAGE_IMPLEMENTATION,
        version=DENSE_DEPTH_COVERAGE_VERSION,
        revision=DENSE_DEPTH_COVERAGE_REVISION,
    ),
    configuration=ConfigurationIdentity(
        sha256=Sha256Digest(hashlib.sha256(_EVALUATOR_CONFIGURATION_BYTES).hexdigest())
    ),
)

_DENSE_DEPTH_DIMENSION = MetricDimension("geometry.dense_depth")
_RATIO_UNIT = MetricUnit("ratio")

CAMERA_SUPPORT_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.camera_support_ratio"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("source_camera_ratio"),
)
CONFIDENCE_FIELD_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.confidence_field_ratio"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("emitted_field_ratio"),
)
HOLE_PIXEL_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.hole_pixel_ratio"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("emitted_pixel_ratio"),
)
VALID_CONFIDENCE_MEAN_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.valid_confidence_mean"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("mean_valid_confidence_sample"),
)
VALID_PIXEL_RATIO_DESCRIPTOR = MetricDescriptor(
    name=MetricName("geometry.dense_depth.valid_pixel_ratio"),
    dimension=_DENSE_DEPTH_DIMENSION,
    unit=_RATIO_UNIT,
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("emitted_pixel_ratio"),
)


def _require_non_negative_int(value: object, context: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be int")
    if value < 0:
        raise ValueError(f"{context} must be non-negative")
    return value


def _require_finite_ratio(value: object, context: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value):
        raise ValueError(f"{context} must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{context} must be in the inclusive range 0.0 to 1.0")
    return value


@dataclass(frozen=True, slots=True)
class DenseDepthFieldCoverage:
    """Deterministic support summary for one existing canonical DepthField."""

    depth_field_id: DepthFieldId
    camera_solution_id: CameraSolutionId
    observation_id: ObservationId
    total_pixel_count: int
    valid_pixel_count: int
    hole_pixel_count: int
    valid_pixel_ratio: float
    confidence_available: bool
    valid_confidence_mean: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.depth_field_id, DepthFieldId):
            raise TypeError("dense_depth_field_coverage.depth_field_id must be DepthFieldId")
        if not isinstance(self.camera_solution_id, CameraSolutionId):
            raise TypeError(
                "dense_depth_field_coverage.camera_solution_id must be CameraSolutionId"
            )
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("dense_depth_field_coverage.observation_id must be ObservationId")

        total = _require_non_negative_int(
            self.total_pixel_count,
            "dense_depth_field_coverage.total_pixel_count",
        )
        valid = _require_non_negative_int(
            self.valid_pixel_count,
            "dense_depth_field_coverage.valid_pixel_count",
        )
        holes = _require_non_negative_int(
            self.hole_pixel_count,
            "dense_depth_field_coverage.hole_pixel_count",
        )
        if total == 0:
            raise ValueError("dense_depth_field_coverage.total_pixel_count must be positive")
        if valid > total:
            raise ValueError(
                "dense_depth_field_coverage.valid_pixel_count cannot exceed total_pixel_count"
            )
        if holes != total - valid:
            raise ValueError(
                "dense_depth_field_coverage.hole_pixel_count must equal total minus valid"
            )

        ratio = _require_finite_ratio(
            self.valid_pixel_ratio,
            "dense_depth_field_coverage.valid_pixel_ratio",
        )
        if ratio != valid / total:
            raise ValueError(
                "dense_depth_field_coverage.valid_pixel_ratio must equal valid divided by total"
            )

        if type(self.confidence_available) is not bool:
            raise TypeError("dense_depth_field_coverage.confidence_available must be bool")
        if self.valid_confidence_mean is None:
            if self.confidence_available and valid > 0:
                raise ValueError(
                    "dense_depth_field_coverage.valid_confidence_mean is required when "
                    "explicit confidence has valid pixels"
                )
            return

        mean = _require_finite_ratio(
            self.valid_confidence_mean,
            "dense_depth_field_coverage.valid_confidence_mean",
        )
        if not self.confidence_available:
            raise ValueError(
                "dense_depth_field_coverage.valid_confidence_mean requires explicit confidence"
            )
        if valid == 0:
            raise ValueError(
                "dense_depth_field_coverage.valid_confidence_mean must be None with zero valid pixels"
            )
        if mean != self.valid_confidence_mean:
            raise AssertionError("unreachable confidence validation state")


def _summarize_depth_field(depth_field: DepthField) -> DenseDepthFieldCoverage:
    total = len(depth_field.validity)
    valid = sum(1 for supported in depth_field.validity if supported)
    holes = total - valid

    confidence_mean: float | None = None
    if depth_field.confidence is not None and valid:
        confidence_mean = sum(
            confidence
            for confidence, supported in zip(
                depth_field.confidence,
                depth_field.validity,
                strict=True,
            )
            if supported
        ) / valid

    return DenseDepthFieldCoverage(
        depth_field_id=depth_field.depth_field_id,
        camera_solution_id=depth_field.camera_solution_id,
        observation_id=depth_field.observation_id,
        total_pixel_count=total,
        valid_pixel_count=valid,
        hole_pixel_count=holes,
        valid_pixel_ratio=valid / total,
        confidence_available=depth_field.confidence is not None,
        valid_confidence_mean=confidence_mean,
    )


def _metric_observation(
    descriptor: MetricDescriptor,
    value: float,
    provenance: MetricProvenance,
) -> MetricObservation:
    return MetricObservation(
        descriptor=descriptor,
        value=float(value),
        provenance=provenance,
    )


def _aggregate_metrics(
    source_depth: DenseDepthArtifact,
    field_coverages: tuple[DenseDepthFieldCoverage, ...],
) -> MetricVector:
    provenance = MetricProvenance(
        evaluator=DENSE_DEPTH_COVERAGE_EVALUATOR,
        input_artifacts=(source_depth.artifact_ref,),
    )

    source_camera_count = len(source_depth.source_geometry.camera_solutions)
    field_count = len(field_coverages)
    total_pixels = sum(field.total_pixel_count for field in field_coverages)
    valid_pixels = sum(field.valid_pixel_count for field in field_coverages)
    hole_pixels = sum(field.hole_pixel_count for field in field_coverages)
    confidence_fields = sum(1 for field in field_coverages if field.confidence_available)

    confidence_values: list[float] = []
    for depth_field in source_depth.depth_fields:
        if depth_field.confidence is None:
            continue
        confidence_values.extend(
            confidence
            for confidence, supported in zip(
                depth_field.confidence,
                depth_field.validity,
                strict=True,
            )
            if supported
        )

    observations = [
        _metric_observation(
            CAMERA_SUPPORT_RATIO_DESCRIPTOR,
            field_count / source_camera_count,
            provenance,
        ),
        _metric_observation(
            CONFIDENCE_FIELD_RATIO_DESCRIPTOR,
            confidence_fields / field_count,
            provenance,
        ),
        _metric_observation(
            HOLE_PIXEL_RATIO_DESCRIPTOR,
            hole_pixels / total_pixels,
            provenance,
        ),
    ]
    if confidence_values:
        observations.append(
            _metric_observation(
                VALID_CONFIDENCE_MEAN_DESCRIPTOR,
                sum(confidence_values) / len(confidence_values),
                provenance,
            )
        )
    observations.append(
        _metric_observation(
            VALID_PIXEL_RATIO_DESCRIPTOR,
            valid_pixels / total_pixels,
            provenance,
        )
    )
    return MetricVector(observations=tuple(observations))


@dataclass(frozen=True, slots=True)
class DenseDepthCoverageReport:
    """Immutable deterministic support report over one DenseDepthArtifact."""

    source_depth: DenseDepthArtifact
    field_coverages: tuple[DenseDepthFieldCoverage, ...]
    metrics: MetricVector

    def __post_init__(self) -> None:
        if not isinstance(self.source_depth, DenseDepthArtifact):
            raise TypeError("dense_depth_coverage_report.source_depth must be DenseDepthArtifact")
        if not isinstance(self.field_coverages, tuple):
            raise TypeError(
                "dense_depth_coverage_report.field_coverages must be an immutable tuple"
            )
        if any(
            not isinstance(field, DenseDepthFieldCoverage)
            for field in self.field_coverages
        ):
            raise TypeError(
                "dense_depth_coverage_report.field_coverages members must be "
                "DenseDepthFieldCoverage"
            )
        if not isinstance(self.metrics, MetricVector):
            raise TypeError("dense_depth_coverage_report.metrics must be MetricVector")

        expected_fields = tuple(
            _summarize_depth_field(depth_field)
            for depth_field in self.source_depth.depth_fields
        )
        if self.field_coverages != expected_fields:
            raise ValueError(
                "dense_depth_coverage_report.field_coverages must exactly summarize "
                "source DepthFields in canonical order"
            )

        expected_metrics = _aggregate_metrics(self.source_depth, expected_fields)
        if self.metrics != expected_metrics:
            raise ValueError(
                "dense_depth_coverage_report.metrics must exactly match canonical "
                "aggregate support metrics"
            )


def evaluate_dense_depth_coverage(
    source_depth: DenseDepthArtifact,
) -> DenseDepthCoverageReport:
    """Return deterministic support diagnostics without changing dense-depth evidence."""

    if not isinstance(source_depth, DenseDepthArtifact):
        raise TypeError("source_depth must be DenseDepthArtifact")

    fields = tuple(
        _summarize_depth_field(depth_field)
        for depth_field in source_depth.depth_fields
    )
    return DenseDepthCoverageReport(
        source_depth=source_depth,
        field_coverages=fields,
        metrics=_aggregate_metrics(source_depth, fields),
    )


__all__ = [
    "CAMERA_SUPPORT_RATIO_DESCRIPTOR",
    "CONFIDENCE_FIELD_RATIO_DESCRIPTOR",
    "DENSE_DEPTH_COVERAGE_EVALUATOR",
    "DENSE_DEPTH_COVERAGE_IMPLEMENTATION",
    "DENSE_DEPTH_COVERAGE_REVISION",
    "DENSE_DEPTH_COVERAGE_VERSION",
    "DenseDepthCoverageReport",
    "DenseDepthFieldCoverage",
    "HOLE_PIXEL_RATIO_DESCRIPTOR",
    "VALID_CONFIDENCE_MEAN_DESCRIPTOR",
    "VALID_PIXEL_RATIO_DESCRIPTOR",
    "evaluate_dense_depth_coverage",
]
