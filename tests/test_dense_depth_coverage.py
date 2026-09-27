from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast

import pytest

import wre.reconstruction.dense_depth_coverage as coverage
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricDirection, MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact
from wre.reconstruction.dense_depth_coverage import (
    DenseDepthCoverageReport,
    DenseDepthFieldCoverage,
    evaluate_dense_depth_coverage,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.dense_depth_coverage.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "geometry.solution") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _camera(
    token: str,
    *,
    frame: LocalFrameId,
    width_px: int = 2,
    height_px: int = 2,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=width_px, height_px=height_px),
        intrinsic_parameters=(2.0, 2.0, 1.0, 1.0),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _candidate(token: str, camera_count: int = 2) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    cameras = tuple(
        _camera(chr(ord("a") + index), frame=frame)
        for index in range(camera_count)
    )
    point_map = PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=tuple(camera.observation_id for camera in cameras),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(camera.solution_id for camera in cameras),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=cameras,
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=(_artifact(f"artifact:source:{token}"),),
    )


def _depth(
    camera: CameraSolution,
    token: str,
    *,
    validity: tuple[bool, ...],
    confidence: tuple[float, ...] | None = None,
) -> DepthField:
    if len(validity) != camera.dimensions.width_px * camera.dimensions.height_px:
        raise AssertionError("test validity must match camera dimensions")
    values = tuple(
        float(index + 1) if supported else 0.0
        for index, supported in enumerate(validity)
    )
    return DepthField(
        depth_field_id=DepthFieldId(f"depth:{token}"),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName("camera-z"),
        depth_values=values,
        validity=validity,
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense(
    token: str,
    *,
    source: GeometrySolutionCandidate,
    depth_fields: tuple[DepthField, ...],
) -> DenseDepthArtifact:
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(f"dense:{token}"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer(token),
        source_artifacts=source.source_artifacts,
    )


def _metric_values(report: DenseDepthCoverageReport) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value
        for observation in report.metrics.observations
    }


def test_field_and_report_have_exact_frozen_shapes() -> None:
    source = _candidate("shape", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "shape",
        validity=(True, True, False, False),
    )
    dense = _dense("shape", source=source, depth_fields=(depth,))
    report = evaluate_dense_depth_coverage(dense)
    field = report.field_coverages[0]

    assert tuple(item.name for item in fields(DenseDepthFieldCoverage)) == (
        "depth_field_id",
        "camera_solution_id",
        "observation_id",
        "total_pixel_count",
        "valid_pixel_count",
        "hole_pixel_count",
        "valid_pixel_ratio",
        "confidence_available",
        "valid_confidence_mean",
    )
    assert tuple(item.name for item in fields(DenseDepthCoverageReport)) == (
        "source_depth",
        "field_coverages",
        "metrics",
    )

    with pytest.raises(FrozenInstanceError):
        field.valid_pixel_count = 0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.field_coverages = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "error_type", "message"),
    [
        ("depth_field_id", cast(Any, "depth"), TypeError, "depth_field_id"),
        ("camera_solution_id", cast(Any, "camera"), TypeError, "camera_solution_id"),
        ("observation_id", cast(Any, "obs"), TypeError, "observation_id"),
        ("total_pixel_count", cast(Any, 4.0), TypeError, "total_pixel_count"),
        ("valid_pixel_count", -1, ValueError, "valid_pixel_count"),
        ("hole_pixel_count", -1, ValueError, "hole_pixel_count"),
        ("valid_pixel_ratio", cast(Any, 1), TypeError, "valid_pixel_ratio"),
        ("confidence_available", cast(Any, 1), TypeError, "confidence_available"),
    ],
)
def test_field_summary_rejects_wrong_types_and_ranges(
    field_name: str,
    replacement: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    source = _candidate("field-validation", camera_count=1)
    camera = source.camera_solutions[0]
    kwargs: dict[str, Any] = {
        "depth_field_id": DepthFieldId("depth:field-validation"),
        "camera_solution_id": camera.solution_id,
        "observation_id": camera.observation_id,
        "total_pixel_count": 4,
        "valid_pixel_count": 2,
        "hole_pixel_count": 2,
        "valid_pixel_ratio": 0.5,
        "confidence_available": False,
        "valid_confidence_mean": None,
    }
    kwargs[field_name] = replacement

    with pytest.raises(error_type, match=message):
        DenseDepthFieldCoverage(**kwargs)


def test_field_summary_enforces_exact_count_ratio_and_confidence_relationships() -> None:
    source = _candidate("relationships", camera_count=1)
    camera = source.camera_solutions[0]
    base = DenseDepthFieldCoverage(
        depth_field_id=DepthFieldId("depth:relationships"),
        camera_solution_id=camera.solution_id,
        observation_id=camera.observation_id,
        total_pixel_count=4,
        valid_pixel_count=2,
        hole_pixel_count=2,
        valid_pixel_ratio=0.5,
        confidence_available=False,
        valid_confidence_mean=None,
    )

    with pytest.raises(ValueError, match="total minus valid"):
        replace(base, hole_pixel_count=1)
    with pytest.raises(ValueError, match="valid divided by total"):
        replace(base, valid_pixel_ratio=0.75)
    with pytest.raises(ValueError, match="requires explicit confidence"):
        replace(base, valid_confidence_mean=0.5)
    with pytest.raises(ValueError, match="is required"):
        replace(base, confidence_available=True)


@pytest.mark.parametrize(
    ("validity", "expected_valid", "expected_holes", "expected_ratio"),
    [
        ((True, True, True, True), 4, 0, 1.0),
        ((True, False, True, False), 2, 2, 0.5),
        ((False, False, False, False), 0, 4, 0.0),
    ],
)
def test_validity_alone_defines_exact_pixel_support(
    validity: tuple[bool, ...],
    expected_valid: int,
    expected_holes: int,
    expected_ratio: float,
) -> None:
    source = _candidate("validity", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "validity",
        validity=validity,
    )
    report = evaluate_dense_depth_coverage(
        _dense("validity", source=source, depth_fields=(depth,))
    )
    field = report.field_coverages[0]

    assert field.total_pixel_count == 4
    assert field.valid_pixel_count == expected_valid
    assert field.hole_pixel_count == expected_holes
    assert field.valid_pixel_ratio == expected_ratio

    metrics = _metric_values(report)
    assert metrics["geometry.dense_depth.valid_pixel_ratio"] == expected_ratio
    assert metrics["geometry.dense_depth.hole_pixel_ratio"] == 1.0 - expected_ratio


def test_camera_support_and_pixel_holes_use_separate_denominators() -> None:
    source = _candidate("partial-camera", camera_count=3)
    depth = _depth(
        source.camera_solutions[0],
        "partial-camera",
        validity=(True, True, True, False),
    )
    report = evaluate_dense_depth_coverage(
        _dense("partial-camera", source=source, depth_fields=(depth,))
    )
    metrics = _metric_values(report)

    assert metrics["geometry.dense_depth.camera_support_ratio"] == pytest.approx(1.0 / 3.0)
    assert metrics["geometry.dense_depth.valid_pixel_ratio"] == 0.75
    assert metrics["geometry.dense_depth.hole_pixel_ratio"] == 0.25
    assert report.field_coverages[0].hole_pixel_count == 1


def test_zero_confidence_on_valid_pixel_does_not_create_a_hole() -> None:
    source = _candidate("zero-confidence", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "zero-confidence",
        validity=(True, False, True, False),
        confidence=(0.0, 0.0, 1.0, 0.0),
    )
    report = evaluate_dense_depth_coverage(
        _dense("zero-confidence", source=source, depth_fields=(depth,))
    )
    field = report.field_coverages[0]

    assert field.valid_pixel_count == 2
    assert field.hole_pixel_count == 2
    assert field.confidence_available is True
    assert field.valid_confidence_mean == 0.5
    assert _metric_values(report)["geometry.dense_depth.valid_confidence_mean"] == 0.5


def test_absent_confidence_stays_absent_and_emits_no_confidence_mean_metric() -> None:
    source = _candidate("no-confidence", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "no-confidence",
        validity=(True, True, False, False),
        confidence=None,
    )
    report = evaluate_dense_depth_coverage(
        _dense("no-confidence", source=source, depth_fields=(depth,))
    )
    field = report.field_coverages[0]
    metrics = _metric_values(report)

    assert field.confidence_available is False
    assert field.valid_confidence_mean is None
    assert metrics["geometry.dense_depth.confidence_field_ratio"] == 0.0
    assert "geometry.dense_depth.valid_confidence_mean" not in metrics


def test_mixed_confidence_uses_only_valid_pixels_from_confidence_bearing_fields() -> None:
    source = _candidate("mixed", camera_count=2)
    first, second = source.camera_solutions
    depth_a = _depth(
        first,
        "a",
        validity=(True, False, True, False),
        confidence=(0.2, 0.0, 0.8, 0.0),
    )
    depth_b = _depth(
        second,
        "b",
        validity=(True, True, True, True),
        confidence=None,
    )
    report = evaluate_dense_depth_coverage(
        _dense("mixed", source=source, depth_fields=(depth_a, depth_b))
    )
    metrics = _metric_values(report)

    assert metrics["geometry.dense_depth.confidence_field_ratio"] == 0.5
    assert metrics["geometry.dense_depth.valid_confidence_mean"] == 0.5
    assert report.field_coverages[0].valid_confidence_mean == 0.5
    assert report.field_coverages[1].valid_confidence_mean is None


def test_confidence_with_zero_valid_pixels_does_not_fabricate_a_mean() -> None:
    source = _candidate("zero-valid", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "zero-valid",
        validity=(False, False, False, False),
        confidence=(0.0, 0.0, 0.0, 0.0),
    )
    report = evaluate_dense_depth_coverage(
        _dense("zero-valid", source=source, depth_fields=(depth,))
    )
    field = report.field_coverages[0]
    metrics = _metric_values(report)

    assert field.confidence_available is True
    assert field.valid_confidence_mean is None
    assert metrics["geometry.dense_depth.confidence_field_ratio"] == 1.0
    assert "geometry.dense_depth.valid_confidence_mean" not in metrics


def test_field_order_metric_order_descriptors_and_provenance_are_canonical() -> None:
    source = _candidate("canonical", camera_count=2)
    first, second = source.camera_solutions
    depth_a = _depth(
        first,
        "a",
        validity=(True, True, True, True),
        confidence=(0.25, 0.5, 0.75, 1.0),
    )
    depth_b = _depth(
        second,
        "b",
        validity=(True, False, False, False),
        confidence=None,
    )
    dense = _dense(
        "canonical",
        source=source,
        depth_fields=(depth_a, depth_b),
    )
    report = evaluate_dense_depth_coverage(dense)

    assert tuple(field.depth_field_id for field in report.field_coverages) == (
        depth_a.depth_field_id,
        depth_b.depth_field_id,
    )
    names = tuple(item.descriptor.name.value for item in report.metrics.observations)
    assert names == tuple(sorted(names))
    assert names == (
        "geometry.dense_depth.camera_support_ratio",
        "geometry.dense_depth.confidence_field_ratio",
        "geometry.dense_depth.hole_pixel_ratio",
        "geometry.dense_depth.valid_confidence_mean",
        "geometry.dense_depth.valid_pixel_ratio",
    )
    assert all(
        item.provenance.evaluator == coverage.DENSE_DEPTH_COVERAGE_EVALUATOR
        for item in report.metrics.observations
    )
    assert all(
        item.provenance.input_artifacts == (dense.artifact_ref,)
        for item in report.metrics.observations
    )

    descriptors = {
        item.descriptor.name.value: item.descriptor
        for item in report.metrics.observations
    }
    assert (
        descriptors["geometry.dense_depth.camera_support_ratio"].direction
        is MetricDirection.HIGHER_IS_BETTER
    )
    assert (
        descriptors["geometry.dense_depth.hole_pixel_ratio"].direction
        is MetricDirection.LOWER_IS_BETTER
    )
    assert (
        descriptors["geometry.dense_depth.valid_confidence_mean"].direction
        is MetricDirection.INFORMATIONAL
    )


def test_report_rejects_noncanonical_field_or_metric_content() -> None:
    source = _candidate("report-validation", camera_count=1)
    depth = _depth(
        source.camera_solutions[0],
        "report-validation",
        validity=(True, True, False, False),
    )
    dense = _dense(
        "report-validation",
        source=source,
        depth_fields=(depth,),
    )
    valid = evaluate_dense_depth_coverage(dense)
    field = valid.field_coverages[0]

    with pytest.raises(ValueError, match="exactly summarize"):
        DenseDepthCoverageReport(
            source_depth=dense,
            field_coverages=(
                replace(
                    field,
                    hole_pixel_count=1,
                    valid_pixel_count=3,
                    valid_pixel_ratio=0.75,
                ),
            ),
            metrics=valid.metrics,
        )

    with pytest.raises(ValueError, match="canonical aggregate"):
        DenseDepthCoverageReport(
            source_depth=dense,
            field_coverages=valid.field_coverages,
            metrics=MetricVector(observations=()),
        )


def test_evaluation_preserves_source_depth_identity_and_payloads() -> None:
    source = _candidate("immutable", camera_count=1)
    validity = (True, False, True, False)
    confidence = (0.3, 0.0, 0.9, 0.0)
    depth = _depth(
        source.camera_solutions[0],
        "immutable",
        validity=validity,
        confidence=confidence,
    )
    dense = _dense("immutable", source=source, depth_fields=(depth,))
    depth_values = depth.depth_values
    source_artifacts = dense.source_artifacts

    report = evaluate_dense_depth_coverage(dense)

    assert report.source_depth is dense
    assert dense.depth_fields[0] is depth
    assert depth.validity is validity
    assert depth.confidence is confidence
    assert depth.depth_values is depth_values
    assert dense.source_artifacts is source_artifacts
    assert dense.source_geometry is source


def test_evaluator_rejects_wrong_source_type() -> None:
    with pytest.raises(TypeError, match="DenseDepthArtifact"):
        evaluate_dense_depth_coverage(cast(Any, "dense-depth"))


def test_module_surface_has_no_solver_filling_quality_or_later_item_dependencies() -> None:
    forbidden = {
        "numpy",
        "torch",
        "pycolmap",
        "PatchMatch",
        "StereoFusion",
        "DepthConsistencyFusionAdapter",
        "LearnedDepthPriorAdapter",
        "PointMap",
        "SurfaceModel",
        "TSDF",
        "SDF",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Scheduler",
        "Benchmark",
        "MasterScene",
        "RuntimeScene",
    }

    assert forbidden.isdisjoint(vars(coverage))
    for attribute in (
        "fill",
        "inpaint",
        "interpolate",
        "fuse",
        "select",
        "benchmark",
        "decide",
    ):
        assert not hasattr(DenseDepthCoverageReport, attribute)
