from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.reconstruction.dense_depth as dense_depth_module
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
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.dense_depth.{token}",
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


def _point_map(
    token: str,
    *,
    frame: LocalFrameId,
    observations: tuple[ObservationId, ...],
) -> PointMap:
    return PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=tuple(sorted(observations, key=lambda item: item.value)),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )


def _candidate(
    token: str,
    *,
    cameras: tuple[CameraSolution, ...] | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    candidate_cameras = cameras or (_camera(token, frame=frame),)
    candidate_cameras = tuple(sorted(candidate_cameras, key=lambda item: item.solution_id.value))
    point_map = _point_map(
        token,
        frame=candidate_cameras[0].local_frame_id,
        observations=tuple({camera.observation_id for camera in candidate_cameras}),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=candidate_cameras[0].local_frame_id,
        scale_status=scale,
        camera_solution_ids=tuple(camera.solution_id for camera in candidate_cameras),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=candidate_cameras,
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (_artifact(f"source:{token}"),)
        ),
    )


def _depth(
    camera: CameraSolution,
    depth_id: str,
    *,
    observation: ObservationId | None = None,
    dimensions: ImageDimensions | None = None,
    convention: str = "camera-z",
    values: tuple[float, ...] | None = None,
    validity: tuple[bool, ...] | None = None,
    confidence: tuple[float, ...] | None = None,
) -> DepthField:
    dims = dimensions or camera.dimensions
    pixel_count = dims.width_px * dims.height_px
    valid = validity or tuple(True for _ in range(pixel_count))
    depth_values = values or tuple(float(index + 1) for index in range(pixel_count))
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=observation or camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=dims,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=depth_values,
        validity=valid,
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense(
    token: str,
    *,
    source_geometry: GeometrySolutionCandidate,
    depth_fields: tuple[DepthField, ...],
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> DenseDepthArtifact:
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(f"dense:{token}"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source_geometry,
        depth_fields=depth_fields,
        producer=_producer(token),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else source_geometry.source_artifacts
        ),
    )


def test_dense_depth_has_exact_frozen_shape_and_kind() -> None:
    source = _candidate("a")
    depth = _depth(source.camera_solutions[0], "dense-depth:a")
    artifact = _dense("a", source_geometry=source, depth_fields=(depth,))

    assert DENSE_DEPTH_ARTIFACT_KIND == ArtifactKind("geometry.dense_depth")
    assert tuple(field.name for field in fields(DenseDepthArtifact)) == (
        "artifact_ref",
        "source_geometry",
        "depth_fields",
        "producer",
        "source_artifacts",
    )
    assert artifact.artifact_ref.artifact_kind is DENSE_DEPTH_ARTIFACT_KIND
    assert artifact.source_geometry is source
    assert artifact.depth_fields[0] is depth

    with pytest.raises(FrozenInstanceError):
        artifact.depth_fields = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("artifact_ref", cast(Any, "artifact"), "artifact_ref"),
        ("source_geometry", cast(Any, "geometry"), "source_geometry"),
        ("depth_fields", cast(Any, []), "immutable tuple"),
        ("depth_fields", cast(Any, ("depth",)), "DepthField"),
        ("producer", cast(Any, "producer"), "producer"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
        ("source_artifacts", cast(Any, ("source",)), "ArtifactRef"),
    ],
)
def test_dense_depth_rejects_wrong_member_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    source = _candidate("a")
    depth = _depth(source.camera_solutions[0], "dense-depth:a")
    kwargs: dict[str, Any] = {
        "artifact_ref": ArtifactRef(
            ArtifactId("dense:a"),
            DENSE_DEPTH_ARTIFACT_KIND,
        ),
        "source_geometry": source,
        "depth_fields": (depth,),
        "producer": _producer("a"),
        "source_artifacts": source.source_artifacts,
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        DenseDepthArtifact(**kwargs)


def test_dense_depth_rejects_wrong_artifact_kind() -> None:
    source = _candidate("a")
    depth = _depth(source.camera_solutions[0], "dense-depth:a")

    with pytest.raises(ValueError, match=r"geometry\.dense_depth"):
        DenseDepthArtifact(
            artifact_ref=_artifact("dense:a", "geometry.solution"),
            source_geometry=source,
            depth_fields=(depth,),
            producer=_producer("a"),
            source_artifacts=source.source_artifacts,
        )


def test_depth_fields_must_be_non_empty_unique_and_canonical() -> None:
    frame = LocalFrameId("frame:multi")
    camera_a = _camera("a", frame=frame)
    camera_b = _camera("b", frame=frame)
    source = _candidate("multi", cameras=(camera_a, camera_b))
    depth_a = _depth(camera_a, "dense-depth:a")
    depth_b = _depth(camera_b, "dense-depth:b")

    artifact = _dense(
        "multi",
        source_geometry=source,
        depth_fields=(depth_a, depth_b),
    )
    assert artifact.depth_fields == (depth_a, depth_b)

    with pytest.raises(ValueError, match="non-empty"):
        _dense("empty", source_geometry=source, depth_fields=())
    with pytest.raises(ValueError, match="unique DepthFieldIds"):
        _dense(
            "duplicate",
            source_geometry=source,
            depth_fields=(depth_a, _depth(camera_b, "dense-depth:a")),
        )
    with pytest.raises(ValueError, match="canonical DepthFieldId"):
        _dense(
            "unordered",
            source_geometry=source,
            depth_fields=(depth_b, depth_a),
        )


def test_dense_depth_rejects_duplicate_camera_and_observation_support() -> None:
    frame = LocalFrameId("frame:duplicates")
    camera_a = _camera("a", frame=frame)
    camera_b = _camera("b", frame=frame)
    source = _candidate("duplicates", cameras=(camera_a, camera_b))

    first = _depth(camera_a, "dense-depth:a")
    same_camera = _depth(camera_a, "dense-depth:b")
    with pytest.raises(ValueError, match="CameraSolution"):
        _dense(
            "same-camera",
            source_geometry=source,
            depth_fields=(first, same_camera),
        )

    shared_observation = ObservationId("obs:shared")
    shared_a = _camera("shared-a", frame=frame, observation=shared_observation.value)
    shared_b = _camera("shared-b", frame=frame, observation=shared_observation.value)
    shared_source = _candidate("shared", cameras=(shared_a, shared_b))
    depth_a = _depth(shared_a, "dense-depth:a")
    depth_b = _depth(shared_b, "dense-depth:b")
    with pytest.raises(ValueError, match="Observation"):
        _dense(
            "same-observation",
            source_geometry=shared_source,
            depth_fields=(depth_a, depth_b),
        )


def test_dense_depth_requires_exact_source_camera_linkage() -> None:
    source = _candidate("source")
    source_camera = source.camera_solutions[0]

    foreign_frame = LocalFrameId("frame:foreign")
    foreign_camera = _camera("foreign", frame=foreign_frame)
    unknown = _depth(foreign_camera, "dense-depth:unknown")
    with pytest.raises(ValueError, match="supplied by source_geometry"):
        _dense("unknown", source_geometry=source, depth_fields=(unknown,))

    wrong_observation = _depth(
        source_camera,
        "dense-depth:observation",
        observation=ObservationId("obs:wrong"),
    )
    with pytest.raises(ValueError, match="ObservationId"):
        _dense(
            "wrong-observation",
            source_geometry=source,
            depth_fields=(wrong_observation,),
        )

    wrong_dimensions = _depth(
        source_camera,
        "dense-depth:dimensions",
        dimensions=ImageDimensions(width_px=3, height_px=1),
    )
    with pytest.raises(ValueError, match="ImageDimensions"):
        _dense(
            "wrong-dimensions",
            source_geometry=source,
            depth_fields=(wrong_dimensions,),
        )


def test_partial_camera_coverage_is_valid_and_fabricates_nothing() -> None:
    frame = LocalFrameId("frame:partial")
    camera_a = _camera("a", frame=frame)
    camera_b = _camera("b", frame=frame)
    source = _candidate("partial", cameras=(camera_a, camera_b))
    depth_a = _depth(camera_a, "dense-depth:a")

    artifact = _dense(
        "partial",
        source_geometry=source,
        depth_fields=(depth_a,),
    )

    assert len(source.camera_solutions) == 2
    assert artifact.depth_fields == (depth_a,)
    assert {item.camera_solution_id for item in artifact.depth_fields} == {
        camera_a.solution_id
    }
    assert camera_b.solution_id not in {
        item.camera_solution_id for item in artifact.depth_fields
    }


def test_new_dense_depth_ids_need_not_exist_on_source_geometry() -> None:
    source = _candidate("new-depth")
    assert source.geometry_solution.depth_field_ids == ()

    depth = _depth(source.camera_solutions[0], "dense-depth:new")
    artifact = _dense(
        "new-depth",
        source_geometry=source,
        depth_fields=(depth,),
    )

    assert artifact.depth_fields[0].depth_field_id == DepthFieldId("dense-depth:new")
    assert artifact.source_geometry.geometry_solution.depth_field_ids == ()
    assert artifact.source_geometry is source


def test_depth_payload_convention_validity_and_confidence_are_preserved_exactly() -> None:
    source = _candidate("payload")
    camera = source.camera_solutions[0]
    values = (4.5, 0.0)
    validity = (True, False)
    confidence = (0.7, 0.0)
    depth = _depth(
        camera,
        "dense-depth:payload",
        convention="inverse-depth",
        values=values,
        validity=validity,
        confidence=confidence,
    )

    artifact = _dense(
        "payload",
        source_geometry=source,
        depth_fields=(depth,),
    )
    retained = artifact.depth_fields[0]

    assert retained is depth
    assert retained.depth_value_convention == DepthValueConventionName("inverse-depth")
    assert retained.depth_values is values
    assert retained.validity is validity
    assert retained.confidence is confidence
    assert retained.depth_values == (4.5, 0.0)
    assert retained.validity == (True, False)
    assert retained.confidence == (0.7, 0.0)


def test_unresolved_scale_and_local_frame_remain_source_context_only() -> None:
    source = _candidate("scale", scale=GeometryScaleStatus.UNRESOLVED)
    depth = _depth(source.camera_solutions[0], "dense-depth:scale")
    artifact = _dense("scale", source_geometry=source, depth_fields=(depth,))

    assert (
        artifact.source_geometry.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )
    assert artifact.source_geometry.geometry_solution.local_frame_id == LocalFrameId(
        "frame:scale"
    )
    for attribute in (
        "scale_status",
        "scale_factor",
        "world_transform",
        "ecef",
        "enu",
        "crs",
        "alignment",
    ):
        assert not hasattr(artifact, attribute)


def test_source_artifacts_retain_ancestry_and_allow_only_canonical_additions() -> None:
    ancestor_a = _artifact("artifact:a", "evidence.alpha")
    ancestor_b = _artifact("artifact:b", "geometry.solution")
    additional = _artifact("artifact:c", "evidence.matches")
    source = _candidate(
        "ancestry",
        source_artifacts=(ancestor_a, ancestor_b),
    )
    depth = _depth(source.camera_solutions[0], "dense-depth:ancestry")

    artifact = _dense(
        "ancestry",
        source_geometry=source,
        depth_fields=(depth,),
        source_artifacts=(ancestor_a, ancestor_b, additional),
    )
    assert artifact.source_artifacts == (ancestor_a, ancestor_b, additional)

    with pytest.raises(ValueError, match="retain every source"):
        _dense(
            "erased",
            source_geometry=source,
            depth_fields=(depth,),
            source_artifacts=(ancestor_a, additional),
        )
    with pytest.raises(ValueError, match="unique"):
        _dense(
            "duplicate",
            source_geometry=source,
            depth_fields=(depth,),
            source_artifacts=(ancestor_a, ancestor_b, ancestor_b),
        )
    with pytest.raises(ValueError, match="canonical"):
        _dense(
            "unordered",
            source_geometry=source,
            depth_fields=(depth,),
            source_artifacts=(ancestor_b, ancestor_a, additional),
        )


def test_dense_artifact_requires_non_empty_source_artifact_evidence() -> None:
    source = _candidate("memory", source_artifacts=())
    depth = _depth(source.camera_solutions[0], "dense-depth:memory")

    with pytest.raises(ValueError, match="non-empty"):
        _dense(
            "memory",
            source_geometry=source,
            depth_fields=(depth,),
            source_artifacts=(),
        )

    evidence = (_artifact("artifact:dense-input", "evidence.images"),)
    artifact = _dense(
        "memory-evidence",
        source_geometry=source,
        depth_fields=(depth,),
        source_artifacts=evidence,
    )
    assert artifact.source_artifacts is evidence


def test_construction_does_not_mutate_or_reorder_caller_evidence() -> None:
    frame = LocalFrameId("frame:immutable")
    camera_a = _camera("a", frame=frame)
    camera_b = _camera("b", frame=frame)
    source = _candidate("immutable", cameras=(camera_a, camera_b))
    depth_a = _depth(camera_a, "dense-depth:a")
    depth_b = _depth(camera_b, "dense-depth:b")
    depths = (depth_a, depth_b)
    sources = source.source_artifacts

    artifact = _dense(
        "immutable",
        source_geometry=source,
        depth_fields=depths,
        source_artifacts=sources,
    )

    assert artifact.source_geometry is source
    assert artifact.depth_fields is depths
    assert artifact.source_artifacts is sources
    assert artifact.depth_fields[0] is depth_a
    assert artifact.depth_fields[1] is depth_b
    assert source.geometry_solution.depth_field_ids == ()
    assert source.camera_solutions == (camera_a, camera_b)


def test_dense_depth_module_exposes_no_solver_fusion_surface_or_policy() -> None:
    forbidden = {
        "torch",
        "numpy",
        "pycolmap",
        "PatchMatch",
        "MVS",
        "StereoFusion",
        "OpenMVS",
        "PointMap",
        "SurfaceModel",
        "TSDF",
        "SDF",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "Scheduler",
        "MasterScene",
        "RuntimeScene",
        "LearnedPrior",
    }

    assert forbidden.isdisjoint(vars(dense_depth_module))
    assert not hasattr(DenseDepthArtifact, "run")
    assert not hasattr(DenseDepthArtifact, "fuse")
    assert not hasattr(DenseDepthArtifact, "select")
