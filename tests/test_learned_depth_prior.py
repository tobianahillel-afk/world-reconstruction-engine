from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast, get_type_hints

import pytest

import wre.reconstruction.learned_depth_prior as learned_prior_module
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
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate
from wre.reconstruction.learned_depth_prior import (
    LearnedDepthPriorAdapter,
    LearnedDepthPriorRequest,
    LearnedDepthPriorResult,
    build_learned_depth_prior_result,
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.learned_depth_prior.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "evidence.learned_depth") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _canonical_artifacts(*items: ArtifactRef) -> tuple[ArtifactRef, ...]:
    unique = {
        (item.artifact_id.value, item.artifact_kind.value): item
        for item in items
    }
    return tuple(unique[key] for key in sorted(unique))


def _camera(
    token: str,
    *,
    frame: LocalFrameId,
    observation: str | None = None,
    width_px: int = 2,
    height_px: int = 1,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(observation or f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=width_px, height_px=height_px),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _candidate(
    token: str,
    *,
    camera_count: int = 2,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
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
        scale_status=scale,
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
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (_artifact(f"artifact:source:{token}", "geometry.solution"),)
        ),
    )


def _depth(
    camera: CameraSolution,
    depth_id: str,
    *,
    convention: str = "learned.relative-depth",
    observation: ObservationId | None = None,
    dimensions: ImageDimensions | None = None,
    confidence: tuple[float, ...] | None = None,
) -> DepthField:
    dims = dimensions or camera.dimensions
    pixel_count = dims.width_px * dims.height_px
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=observation or camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=dims,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=tuple(float(index + 1) for index in range(pixel_count)),
        validity=tuple(True for _ in range(pixel_count)),
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense(
    token: str,
    *,
    source: GeometrySolutionCandidate,
    depth_fields: tuple[DepthField, ...],
    source_artifacts: tuple[ArtifactRef, ...],
    artifact_ref: ArtifactRef | None = None,
) -> DenseDepthArtifact:
    return DenseDepthArtifact(
        artifact_ref=artifact_ref
        or ArtifactRef(
            artifact_id=ArtifactId(f"dense:{token}"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer(token),
        source_artifacts=source_artifacts,
    )


def _request(
    source: GeometrySolutionCandidate,
    *supporting: ArtifactRef,
    convention: str = "learned.relative-depth",
) -> LearnedDepthPriorRequest:
    return LearnedDepthPriorRequest(
        source_geometry=source,
        supporting_artifacts=_canonical_artifacts(*supporting),
        output_depth_value_convention=DepthValueConventionName(convention),
    )


def _valid_result_parts(
    token: str = "valid",
    *,
    confidence: tuple[float, ...] | None = None,
) -> tuple[LearnedDepthPriorRequest, DenseDepthArtifact]:
    source = _candidate(token)
    support_a = _artifact(f"artifact:{token}:a", "evidence.image")
    support_b = _artifact(f"artifact:{token}:b", "model.checkpoint")
    request = _request(source, support_a, support_b)
    ancestry = _canonical_artifacts(
        *source.source_artifacts,
        support_a,
        support_b,
    )
    depth = _depth(
        source.camera_solutions[0],
        f"depth:{token}",
        confidence=confidence,
    )
    dense = _dense(
        token,
        source=source,
        depth_fields=(depth,),
        source_artifacts=ancestry,
    )
    return request, dense


def test_contract_has_exact_frozen_shapes_and_protocol_surface() -> None:
    request, dense = _valid_result_parts("shape")
    result = LearnedDepthPriorResult(request=request, depth_prior=dense)

    assert tuple(field.name for field in fields(LearnedDepthPriorRequest)) == (
        "source_geometry",
        "supporting_artifacts",
        "output_depth_value_convention",
    )
    assert tuple(field.name for field in fields(LearnedDepthPriorResult)) == (
        "request",
        "depth_prior",
    )

    signature = inspect.signature(LearnedDepthPriorAdapter.infer)
    assert tuple(signature.parameters) == ("self", "request")
    hints = get_type_hints(LearnedDepthPriorAdapter.infer)
    assert hints["request"] is LearnedDepthPriorRequest
    assert hints["return"] is LearnedDepthPriorResult

    with pytest.raises(FrozenInstanceError):
        request.supporting_artifacts = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.depth_prior = dense  # type: ignore[misc]


@pytest.mark.parametrize(
    ("source_geometry", "supporting_artifacts", "convention", "error_type", "message"),
    [
        (cast(Any, "geometry"), (_artifact("artifact:a"),), DepthValueConventionName("camera-z"), TypeError, "source_geometry"),
        (_candidate("a"), cast(Any, []), DepthValueConventionName("camera-z"), TypeError, "immutable tuple"),
        (_candidate("a"), (), DepthValueConventionName("camera-z"), ValueError, "non-empty"),
        (_candidate("a"), cast(Any, ("artifact",)), DepthValueConventionName("camera-z"), TypeError, "ArtifactRef"),
        (_candidate("a"), (_artifact("artifact:a"),), cast(Any, "camera-z"), TypeError, "DepthValueConventionName"),
    ],
)
def test_request_rejects_wrong_member_shapes(
    source_geometry: Any,
    supporting_artifacts: Any,
    convention: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        LearnedDepthPriorRequest(
            source_geometry=source_geometry,
            supporting_artifacts=supporting_artifacts,
            output_depth_value_convention=convention,
        )


def test_supporting_artifacts_must_be_unique_and_canonical_without_reordering() -> None:
    source = _candidate("support")
    first = _artifact("artifact:a", "evidence.alpha")
    second = _artifact("artifact:b", "evidence.beta")
    supporting = (first, second)

    request = LearnedDepthPriorRequest(
        source_geometry=source,
        supporting_artifacts=supporting,
        output_depth_value_convention=DepthValueConventionName("camera-z"),
    )
    assert request.supporting_artifacts is supporting

    with pytest.raises(ValueError, match="unique"):
        LearnedDepthPriorRequest(
            source_geometry=source,
            supporting_artifacts=(first, first),
            output_depth_value_convention=DepthValueConventionName("camera-z"),
        )
    with pytest.raises(ValueError, match="canonical"):
        LearnedDepthPriorRequest(
            source_geometry=source,
            supporting_artifacts=(second, first),
            output_depth_value_convention=DepthValueConventionName("camera-z"),
        )


@pytest.mark.parametrize(
    ("request_value", "depth_value", "message"),
    [
        (cast(Any, "request"), None, "request"),
        (None, cast(Any, "depth"), "depth_prior"),
    ],
)
def test_result_rejects_wrong_member_types(
    request_value: Any,
    depth_value: Any,
    message: str,
) -> None:
    request, dense = _valid_result_parts("wrong-types")

    with pytest.raises(TypeError, match=message):
        LearnedDepthPriorResult(
            request=request if request_value is None else request_value,
            depth_prior=dense if depth_value is None else depth_value,
        )


def test_result_requires_exact_source_geometry_value() -> None:
    source = _candidate("source")
    support = _artifact("artifact:support")
    request = _request(source, support)

    altered_source = replace(source, producer=_producer("different-producer"))
    assert (
        altered_source.geometry_solution.geometry_solution_id
        == source.geometry_solution.geometry_solution_id
    )
    dense = _dense(
        "foreign",
        source=altered_source,
        depth_fields=(_depth(altered_source.camera_solutions[0], "depth:foreign"),),
        source_artifacts=_canonical_artifacts(
            *altered_source.source_artifacts,
            support,
        ),
    )

    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        LearnedDepthPriorResult(request=request, depth_prior=dense)


def test_result_rejects_reused_supporting_artifact_identity() -> None:
    source = _candidate("reused")
    reused = _artifact("dense:reused", "geometry.dense_depth")
    request = _request(source, reused)
    dense = _dense(
        "ignored",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:reused"),),
        source_artifacts=_canonical_artifacts(*source.source_artifacts, reused),
        artifact_ref=reused,
    )

    with pytest.raises(ValueError, match="must not reuse a supporting ArtifactRef"):
        LearnedDepthPriorResult(request=request, depth_prior=dense)


def test_output_depth_convention_must_match_declared_convention_without_conversion() -> None:
    source = _candidate("convention")
    support = _artifact("artifact:support")
    request = _request(source, support, convention="inverse-depth")
    ancestry = _canonical_artifacts(*source.source_artifacts, support)

    matching = _dense(
        "matching",
        source=source,
        depth_fields=(
            _depth(
                source.camera_solutions[0],
                "depth:matching",
                convention="inverse-depth",
            ),
        ),
        source_artifacts=ancestry,
    )
    result = LearnedDepthPriorResult(request=request, depth_prior=matching)
    assert (
        result.depth_prior.depth_fields[0].depth_value_convention
        == request.output_depth_value_convention
    )

    mismatched = _dense(
        "mismatched",
        source=source,
        depth_fields=(
            _depth(
                source.camera_solutions[0],
                "depth:mismatched",
                convention="camera-z",
            ),
        ),
        source_artifacts=ancestry,
    )
    with pytest.raises(ValueError, match="declared DepthValueConventionName"):
        LearnedDepthPriorResult(request=request, depth_prior=mismatched)


def test_partial_camera_coverage_is_valid_and_dense_depth_keeps_linkage_fail_closed() -> None:
    source = _candidate("partial", camera_count=2)
    support = _artifact("artifact:support")
    request = _request(source, support)
    ancestry = _canonical_artifacts(*source.source_artifacts, support)
    first_camera, second_camera = source.camera_solutions

    dense = _dense(
        "partial",
        source=source,
        depth_fields=(_depth(first_camera, "depth:partial"),),
        source_artifacts=ancestry,
    )
    result = LearnedDepthPriorResult(request=request, depth_prior=dense)
    assert len(result.depth_prior.depth_fields) == 1
    assert result.depth_prior.depth_fields[0].camera_solution_id == first_camera.solution_id
    assert second_camera.solution_id not in {
        field.camera_solution_id for field in result.depth_prior.depth_fields
    }

    foreign_frame = LocalFrameId("frame:foreign")
    foreign_camera = _camera("foreign", frame=foreign_frame)
    with pytest.raises(ValueError, match="supplied by source_geometry"):
        _dense(
            "foreign",
            source=source,
            depth_fields=(_depth(foreign_camera, "depth:foreign"),),
            source_artifacts=ancestry,
        )

    with pytest.raises(ValueError, match="ObservationId"):
        _dense(
            "wrong-observation",
            source=source,
            depth_fields=(
                _depth(
                    first_camera,
                    "depth:wrong-observation",
                    observation=ObservationId("obs:wrong"),
                ),
            ),
            source_artifacts=ancestry,
        )

    with pytest.raises(ValueError, match="ImageDimensions"):
        _dense(
            "wrong-dimensions",
            source=source,
            depth_fields=(
                _depth(
                    first_camera,
                    "depth:wrong-dimensions",
                    dimensions=ImageDimensions(width_px=3, height_px=1),
                ),
            ),
            source_artifacts=ancestry,
        )


def test_result_requires_exact_canonical_union_of_source_and_supporting_ancestry() -> None:
    source_a = _artifact("artifact:a-source", "geometry.solution")
    source_b = _artifact("artifact:b-source", "evidence.camera")
    source = _candidate(
        "ancestry",
        source_artifacts=(source_a, source_b),
    )
    support_a = _artifact("artifact:c-support", "evidence.image")
    support_b = _artifact("artifact:d-support", "model.checkpoint")
    request = _request(source, support_a, support_b)
    depth = _depth(source.camera_solutions[0], "depth:ancestry")
    expected = _canonical_artifacts(source_a, source_b, support_a, support_b)

    valid = _dense(
        "valid-ancestry",
        source=source,
        depth_fields=(depth,),
        source_artifacts=expected,
    )
    result = build_learned_depth_prior_result(request, valid)
    assert result.depth_prior.source_artifacts is expected

    missing = _dense(
        "missing-ancestry",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:missing"),),
        source_artifacts=_canonical_artifacts(source_a, source_b, support_a),
    )
    with pytest.raises(ValueError, match="canonical union"):
        LearnedDepthPriorResult(request=request, depth_prior=missing)

    extra = _dense(
        "extra-ancestry",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:extra"),),
        source_artifacts=_canonical_artifacts(
            *expected,
            _artifact("artifact:e-unexpected", "evidence.unexpected"),
        ),
    )
    with pytest.raises(ValueError, match="canonical union"):
        LearnedDepthPriorResult(request=request, depth_prior=extra)


@pytest.mark.parametrize(
    "confidence",
    [
        None,
        (0.7, 0.4),
    ],
)
def test_optional_confidence_is_preserved_without_synthesis_or_calibration(
    confidence: tuple[float, ...] | None,
) -> None:
    request, dense = _valid_result_parts("confidence", confidence=confidence)
    original_field = dense.depth_fields[0]

    result = build_learned_depth_prior_result(request, dense)
    retained = result.depth_prior.depth_fields[0]

    assert retained is original_field
    assert retained.confidence is confidence
    if confidence is None:
        assert retained.confidence is None
    else:
        assert retained.confidence == (0.7, 0.4)


def test_unresolved_scale_and_local_frame_remain_source_context_only() -> None:
    source = _candidate("scale", scale=GeometryScaleStatus.UNRESOLVED)
    support = _artifact("artifact:support")
    request = _request(source, support)
    dense = _dense(
        "scale",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:scale"),),
        source_artifacts=_canonical_artifacts(*source.source_artifacts, support),
    )

    result = LearnedDepthPriorResult(request=request, depth_prior=dense)

    assert (
        result.depth_prior.source_geometry.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert (
        result.depth_prior.source_geometry.geometry_solution.local_frame_id
        == source.geometry_solution.local_frame_id
    )
    for attribute in (
        "scale_factor",
        "world_transform",
        "alignment",
        "crs",
        "metric_scale",
    ):
        assert not hasattr(result, attribute)


def test_construction_and_helper_do_not_mutate_or_reorder_evidence() -> None:
    source = _candidate("immutable")
    support_a = _artifact("artifact:a-support")
    support_b = _artifact("artifact:b-support")
    supporting = (support_a, support_b)
    convention = DepthValueConventionName("learned.relative-depth")
    request = LearnedDepthPriorRequest(
        source_geometry=source,
        supporting_artifacts=supporting,
        output_depth_value_convention=convention,
    )
    ancestry = _canonical_artifacts(*source.source_artifacts, *supporting)
    depth_field = _depth(source.camera_solutions[0], "depth:immutable")
    depth_fields = (depth_field,)
    dense = _dense(
        "immutable",
        source=source,
        depth_fields=depth_fields,
        source_artifacts=ancestry,
    )

    result = build_learned_depth_prior_result(request, dense)

    assert request.source_geometry is source
    assert request.supporting_artifacts is supporting
    assert request.output_depth_value_convention is convention
    assert result.request is request
    assert result.depth_prior is dense
    assert result.depth_prior.depth_fields is depth_fields
    assert result.depth_prior.depth_fields[0] is depth_field
    assert result.depth_prior.source_artifacts is ancestry


def test_module_exposes_no_model_runtime_fusion_quality_or_next_item_surface() -> None:
    forbidden = {
        "torch",
        "numpy",
        "pycolmap",
        "DA3",
        "VGGT",
        "Pi3",
        "MapAnything",
        "Marigold",
        "Metric3D",
        "UniDepth",
        "Checkpoint",
        "Network",
        "DepthConsistencyFusionAdapter",
        "PointMap",
        "SurfaceModel",
        "CoverageMetric",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Scheduler",
        "MasterScene",
        "RuntimeScene",
    }
    assert forbidden.isdisjoint(vars(learned_prior_module))
    assert not hasattr(LearnedDepthPriorRequest, "run")
    assert not hasattr(LearnedDepthPriorResult, "run")
    assert not hasattr(LearnedDepthPriorResult, "fuse")
    assert not hasattr(LearnedDepthPriorResult, "score")
