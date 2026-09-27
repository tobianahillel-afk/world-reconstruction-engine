from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast, get_type_hints

import pytest

import wre.reconstruction.depth_consistency_fusion as fusion_module
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
from wre.reconstruction.dense_depth import (
    DENSE_DEPTH_ARTIFACT_KIND,
    DenseDepthArtifact,
)
from wre.reconstruction.depth_consistency_fusion import (
    DepthConsistencyFusionAdapter,
    DepthConsistencyFusionRequest,
    DepthConsistencyFusionResult,
    build_depth_consistency_fusion_result,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.depth_fusion.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "evidence.input") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _canonical_artifacts(*items: ArtifactRef) -> tuple[ArtifactRef, ...]:
    return tuple(
        sorted(
            items,
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )


def _camera(token: str, *, frame: LocalFrameId) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=2, height_px=1),
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
    cameras = tuple(_camera(chr(ord("a") + index), frame=frame) for index in range(camera_count))
    observations = tuple(camera.observation_id for camera in cameras)
    point_map = PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=tuple(sorted(observations, key=lambda item: item.value)),
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
        source_artifacts=source_artifacts or (_artifact(f"artifact:source:{token}"),),
    )


def _depth(
    camera: CameraSolution,
    depth_id: str,
    *,
    convention: str = "camera-z",
    values: tuple[float, ...] = (1.0, 2.0),
) -> DepthField:
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=values,
        validity=(True, True),
        confidence=None,
        metrics=_metrics(),
    )


def _dense(
    token: str,
    *,
    source: GeometrySolutionCandidate,
    depth_fields: tuple[DepthField, ...],
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> DenseDepthArtifact:
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(f"dense:{token}"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer(token),
        source_artifacts=source_artifacts or source.source_artifacts,
    )


def _request_pair(
    source: GeometrySolutionCandidate,
    *,
    convention: str = "camera-z",
) -> tuple[DenseDepthArtifact, DenseDepthArtifact]:
    camera = source.camera_solutions[0]
    return (
        _dense(
            "a",
            source=source,
            depth_fields=(_depth(camera, "depth:a", convention=convention),),
        ),
        _dense(
            "b",
            source=source,
            depth_fields=(_depth(camera, "depth:b", convention=convention),),
        ),
    )


def test_contract_has_exact_frozen_shapes_and_protocol_surface() -> None:
    source = _candidate("shape")
    first, second = _request_pair(source)
    request = DepthConsistencyFusionRequest(inputs=(first, second))
    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:fused"),),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )
    result = DepthConsistencyFusionResult(request=request, fused_depth=fused)

    assert tuple(field.name for field in fields(DepthConsistencyFusionRequest)) == ("inputs",)
    assert tuple(field.name for field in fields(DepthConsistencyFusionResult)) == (
        "request",
        "fused_depth",
    )

    signature = inspect.signature(DepthConsistencyFusionAdapter.fuse)
    assert tuple(signature.parameters) == ("self", "request")
    hints = get_type_hints(DepthConsistencyFusionAdapter.fuse)
    assert hints["request"] is DepthConsistencyFusionRequest
    assert hints["return"] is DepthConsistencyFusionResult

    with pytest.raises(FrozenInstanceError):
        request.inputs = ()  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.fused_depth = first  # type: ignore[misc]


@pytest.mark.parametrize(
    ("inputs", "error_type", "message"),
    [
        (cast(Any, []), TypeError, "immutable tuple"),
        (cast(Any, ()), ValueError, "at least two"),
        (cast(Any, ("depth", "depth")), TypeError, "DenseDepthArtifact"),
    ],
)
def test_request_rejects_wrong_container_shape(
    inputs: Any,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        DepthConsistencyFusionRequest(inputs=inputs)


def test_request_requires_unique_canonical_artifact_order() -> None:
    source = _candidate("order")
    first, second = _request_pair(source)

    with pytest.raises(ValueError, match="canonical"):
        DepthConsistencyFusionRequest(inputs=(second, first))

    duplicate = replace(
        second,
        artifact_ref=first.artifact_ref,
    )
    with pytest.raises(ValueError, match="unique"):
        DepthConsistencyFusionRequest(inputs=(first, duplicate))


def test_request_requires_exact_complete_source_geometry_value() -> None:
    source = _candidate("same-id")
    first, _ = _request_pair(source)

    altered_source = replace(source, producer=_producer("different-producer"))
    assert altered_source.geometry_solution.geometry_solution_id == (
        source.geometry_solution.geometry_solution_id
    )
    second = _dense(
        "b",
        source=altered_source,
        depth_fields=(_depth(altered_source.camera_solutions[0], "depth:b"),),
    )

    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        DepthConsistencyFusionRequest(inputs=(first, second))


def test_partial_and_disjoint_camera_coverage_is_valid() -> None:
    source = _candidate("partial")
    camera_a, camera_b = source.camera_solutions
    first = _dense(
        "a",
        source=source,
        depth_fields=(_depth(camera_a, "depth:a"),),
    )
    second = _dense(
        "b",
        source=source,
        depth_fields=(_depth(camera_b, "depth:b"),),
    )

    request = DepthConsistencyFusionRequest(inputs=(first, second))
    assert request.inputs is not None
    assert request.inputs[0] is first
    assert request.inputs[1] is second


def test_overlapping_camera_requires_one_exact_depth_convention() -> None:
    source = _candidate("convention")
    camera = source.camera_solutions[0]
    first = _dense(
        "a",
        source=source,
        depth_fields=(_depth(camera, "depth:a", convention="camera-z"),),
    )
    second = _dense(
        "b",
        source=source,
        depth_fields=(_depth(camera, "depth:b", convention="inverse-depth"),),
    )

    with pytest.raises(ValueError, match="DepthValueConventionName"):
        DepthConsistencyFusionRequest(inputs=(first, second))


@pytest.mark.parametrize(
    ("request_value", "fused_value", "message"),
    [
        (cast(Any, "request"), None, "request"),
        (None, cast(Any, "fused"), "fused_depth"),
    ],
)
def test_result_rejects_wrong_member_types(
    request_value: Any,
    fused_value: Any,
    message: str,
) -> None:
    source = _candidate("wrong-result")
    first, second = _request_pair(source)
    request = DepthConsistencyFusionRequest(inputs=(first, second))
    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:fused"),),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )

    with pytest.raises(TypeError, match=message):
        DepthConsistencyFusionResult(
            request=request if request_value is None else request_value,
            fused_depth=fused if fused_value is None else fused_value,
        )


def test_result_requires_exact_source_and_new_artifact_identity() -> None:
    source = _candidate("result-source")
    first, second = _request_pair(source)
    request = DepthConsistencyFusionRequest(inputs=(first, second))

    foreign = _candidate("foreign")
    foreign_fused = _dense(
        "foreign",
        source=foreign,
        depth_fields=(_depth(foreign.camera_solutions[0], "depth:foreign"),),
    )
    with pytest.raises(ValueError, match="exact source GeometrySolutionCandidate"):
        DepthConsistencyFusionResult(request=request, fused_depth=foreign_fused)

    reused = replace(
        first,
        producer=_producer("reused-output"),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )
    with pytest.raises(ValueError, match="new DenseDepthArtifact identity"):
        DepthConsistencyFusionResult(request=request, fused_depth=reused)


def test_fused_support_must_exist_in_an_input() -> None:
    source = _candidate("support")
    camera_a, camera_b = source.camera_solutions
    first = _dense("a", source=source, depth_fields=(_depth(camera_a, "depth:a"),))
    second = _dense("b", source=source, depth_fields=(_depth(camera_a, "depth:b"),))
    request = DepthConsistencyFusionRequest(inputs=(first, second))

    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(camera_b, "depth:fused"),),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )
    with pytest.raises(ValueError, match="support must already exist"):
        DepthConsistencyFusionResult(request=request, fused_depth=fused)


def test_fused_depth_fields_must_use_new_ids_and_preserve_convention() -> None:
    source = _candidate("field-evidence")
    first, second = _request_pair(source, convention="camera-z")
    request = DepthConsistencyFusionRequest(inputs=(first, second))
    ancestry = _canonical_artifacts(
        *source.source_artifacts,
        first.artifact_ref,
        second.artifact_ref,
    )

    reused_id = _dense(
        "fused-reused",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:a"),),
        source_artifacts=ancestry,
    )
    with pytest.raises(ValueError, match="DepthFieldIds must be new"):
        DepthConsistencyFusionResult(request=request, fused_depth=reused_id)

    converted = _dense(
        "fused-converted",
        source=source,
        depth_fields=(
            _depth(
                source.camera_solutions[0],
                "depth:fused",
                convention="inverse-depth",
            ),
        ),
        source_artifacts=ancestry,
    )
    with pytest.raises(ValueError, match="preserve.*DepthValueConventionName"):
        DepthConsistencyFusionResult(request=request, fused_depth=converted)


def test_result_requires_union_of_input_and_transitive_ancestry() -> None:
    source_ancestor = _artifact("artifact:a-source")
    source = _candidate("ancestry", source_artifacts=(source_ancestor,))
    camera = source.camera_solutions[0]
    extra_a = _artifact("artifact:b-extra")
    extra_b = _artifact("artifact:c-extra")
    first = _dense(
        "a",
        source=source,
        depth_fields=(_depth(camera, "depth:a"),),
        source_artifacts=_canonical_artifacts(source_ancestor, extra_a),
    )
    second = _dense(
        "b",
        source=source,
        depth_fields=(_depth(camera, "depth:b"),),
        source_artifacts=_canonical_artifacts(source_ancestor, extra_b),
    )
    request = DepthConsistencyFusionRequest(inputs=(first, second))

    missing = _dense(
        "fused-missing",
        source=source,
        depth_fields=(_depth(camera, "depth:fused"),),
        source_artifacts=_canonical_artifacts(
            source_ancestor,
            extra_a,
            extra_b,
            first.artifact_ref,
        ),
    )
    with pytest.raises(ValueError, match="retain every input artifact and ancestry"):
        DepthConsistencyFusionResult(request=request, fused_depth=missing)

    full_ancestry = _canonical_artifacts(
        source_ancestor,
        extra_a,
        extra_b,
        first.artifact_ref,
        second.artifact_ref,
    )
    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(camera, "depth:fused"),),
        source_artifacts=full_ancestry,
    )
    result = build_depth_consistency_fusion_result(request, fused)

    assert result.request is request
    assert result.fused_depth is fused
    assert result.fused_depth.source_artifacts is full_ancestry


def test_fusion_may_conservatively_reduce_support_without_fabricating_cameras() -> None:
    source = _candidate("reduced")
    camera_a, camera_b = source.camera_solutions
    first = _dense(
        "a",
        source=source,
        depth_fields=(
            _depth(camera_a, "depth:a-a"),
            _depth(camera_b, "depth:a-b"),
        ),
    )
    second = _dense(
        "b",
        source=source,
        depth_fields=(
            _depth(camera_a, "depth:b-a"),
            _depth(camera_b, "depth:b-b"),
        ),
    )
    request = DepthConsistencyFusionRequest(inputs=(first, second))
    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(camera_a, "depth:fused-a"),),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )

    result = DepthConsistencyFusionResult(request=request, fused_depth=fused)
    assert tuple(field.camera_solution_id for field in result.fused_depth.depth_fields) == (
        camera_a.solution_id,
    )
    assert camera_b.solution_id not in {
        field.camera_solution_id for field in result.fused_depth.depth_fields
    }


def test_unresolved_scale_and_local_frame_are_preserved_only_through_source() -> None:
    source = _candidate("scale", scale=GeometryScaleStatus.UNRESOLVED)
    first, second = _request_pair(source)
    request = DepthConsistencyFusionRequest(inputs=(first, second))
    fused = _dense(
        "fused",
        source=source,
        depth_fields=(_depth(source.camera_solutions[0], "depth:fused"),),
        source_artifacts=_canonical_artifacts(
            *source.source_artifacts,
            first.artifact_ref,
            second.artifact_ref,
        ),
    )

    result = DepthConsistencyFusionResult(request=request, fused_depth=fused)
    assert (
        result.fused_depth.source_geometry.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert (
        result.fused_depth.source_geometry.geometry_solution.local_frame_id
        == source.geometry_solution.local_frame_id
    )
    for attribute in ("scale_factor", "world_transform", "alignment", "crs"):
        assert not hasattr(result, attribute)


def test_construction_and_helper_do_not_mutate_or_reorder_evidence() -> None:
    source = _candidate("immutable")
    first, second = _request_pair(source)
    inputs = (first, second)
    request = DepthConsistencyFusionRequest(inputs=inputs)
    ancestry = _canonical_artifacts(
        *source.source_artifacts,
        first.artifact_ref,
        second.artifact_ref,
    )
    fused_field = _depth(source.camera_solutions[0], "depth:fused")
    fused_fields = (fused_field,)
    fused = _dense(
        "fused",
        source=source,
        depth_fields=fused_fields,
        source_artifacts=ancestry,
    )

    result = build_depth_consistency_fusion_result(request, fused)

    assert request.inputs is inputs
    assert request.inputs[0] is first
    assert request.inputs[1] is second
    assert result.request is request
    assert result.fused_depth is fused
    assert result.fused_depth.depth_fields is fused_fields
    assert result.fused_depth.depth_fields[0] is fused_field
    assert result.fused_depth.source_artifacts is ancestry


def test_module_exposes_no_solver_quality_surface_or_next_item() -> None:
    forbidden = {
        "pycolmap",
        "numpy",
        "torch",
        "stereo_fusion",
        "PatchMatch",
        "OpenMVS",
        "PointMap",
        "SurfaceModel",
        "TSDF",
        "SDF",
        "LearnedPrior",
        "CoverageMetric",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Scheduler",
        "MasterScene",
        "RuntimeScene",
    }
    assert forbidden.isdisjoint(vars(fusion_module))
    assert not hasattr(DepthConsistencyFusionRequest, "run")
    assert not hasattr(DepthConsistencyFusionResult, "run")
    assert not hasattr(DepthConsistencyFusionResult, "score")
