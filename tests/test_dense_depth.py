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
    *,
    frame: LocalFrameId,
    observation: str,
    camera_id: str,
    width_px: int = 2,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(camera_id),
        observation_id=ObservationId(observation),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=width_px, height_px=1),
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
    *,
    frame: LocalFrameId,
    point_map_id: str,
    observation: ObservationId,
) -> PointMap:
    return PointMap(
        point_map_id=PointMapId(point_map_id),
        local_frame_id=frame,
        source_observation_ids=(observation,),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=_metrics(),
    )


def _candidate(
    token: str = "source",
    *,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    same_observation: bool = False,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    first_observation = f"obs:{token}:a"
    second_observation = first_observation if same_observation else f"obs:{token}:b"
    camera_a = _camera(
        frame=frame,
        observation=first_observation,
        camera_id=f"camera:{token}:a",
    )
    camera_b = _camera(
        frame=frame,
        observation=second_observation,
        camera_id=f"camera:{token}:b",
    )
    point_map = _point_map(
        frame=frame,
        point_map_id=f"points:{token}",
        observation=camera_a.observation_id,
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=(camera_a.solution_id, camera_b.solution_id),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera_a, camera_b),
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"source:{token}"),
        source_artifacts=(
            source_artifacts
            if source_artifacts is not None
            else (
                _artifact(f"artifact:{token}:a"),
                _artifact(f"artifact:{token}:c"),
            )
        ),
    )


def _depth(
    *,
    camera: CameraSolution,
    depth_id: str,
    observation: ObservationId | None = None,
    camera_solution_id: CameraSolutionId | None = None,
    dimensions: ImageDimensions | None = None,
    convention: str = "camera-z",
    depth_values: tuple[float, ...] | None = None,
    validity: tuple[bool, ...] | None = None,
    confidence: tuple[float, ...] | None = None,
) -> DepthField:
    dimensions_value = dimensions or camera.dimensions
    pixel_count = dimensions_value.width_px * dimensions_value.height_px
    validity_value = validity or tuple(True for _ in range(pixel_count))
    depth_values_value = depth_values or tuple(1.0 for _ in range(pixel_count))
    return DepthField(
        depth_field_id=DepthFieldId(depth_id),
        observation_id=observation or camera.observation_id,
        camera_solution_id=camera_solution_id or camera.solution_id,
        dimensions=dimensions_value,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=depth_values_value,
        validity=validity_value,
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense_artifact(
    *,
    source_geometry: GeometrySolutionCandidate | None = None,
    depth_fields: tuple[DepthField, ...] | None = None,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
    artifact_ref: ArtifactRef | None = None,
) -> DenseDepthArtifact:
    source = source_geometry or _candidate()
    fields_value = depth_fields or (
        _depth(
            camera=source.camera_solutions[0],
            depth_id="dense:source:a",
        ),
    )
    return DenseDepthArtifact(
        artifact_ref=artifact_ref
        or _artifact("artifact:dense", DENSE_DEPTH_ARTIFACT_KIND.value),
        source_geometry=source,
        depth_fields=fields_value,
        producer=_producer("dense"),
        source_artifacts=(
            source_artifacts if source_artifacts is not None else source.source_artifacts
        ),
    )


def test_dense_depth_has_exact_frozen_slots_shape_and_artifact_kind() -> None:
    artifact = _dense_artifact()

    assert DENSE_DEPTH_ARTIFACT_KIND == ArtifactKind("geometry.dense_depth")
    assert tuple(field.name for field in fields(DenseDepthArtifact)) == (
        "artifact_ref",
        "source_geometry",
        "depth_fields",
        "producer",
        "source_artifacts",
    )
    assert artifact.artifact_ref.artifact_kind is DENSE_DEPTH_ARTIFACT_KIND
    with pytest.raises(FrozenInstanceError):
        artifact.depth_fields = ()  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "replacement", "message"),
    [
        ("artifact_ref", cast(Any, "artifact"), "ArtifactRef"),
        ("source_geometry", cast(Any, "geometry"), "GeometrySolutionCandidate"),
        ("depth_fields", cast(Any, []), "immutable tuple"),
        ("depth_fields", cast(Any, ("depth",)), "DepthField"),
        ("producer", cast(Any, "producer"), "ArtifactProducerIdentity"),
        ("source_artifacts", cast(Any, []), "immutable tuple"),
        ("source_artifacts", cast(Any, ("artifact",)), "ArtifactRef"),
    ],
)
def test_dense_depth_rejects_wrong_field_types(
    field_name: str,
    replacement: Any,
    message: str,
) -> None:
    artifact = _dense_artifact()
    kwargs: dict[str, Any] = {
        field.name: getattr(artifact, field.name) for field in fields(DenseDepthArtifact)
    }
    kwargs[field_name] = replacement

    with pytest.raises((TypeError, ValueError), match=message):
        DenseDepthArtifact(**kwargs)


def test_dense_depth_rejects_wrong_artifact_kind() -> None:
    with pytest.raises(ValueError, match="geometry.dense_depth"):
        _dense_artifact(artifact_ref=_artifact("artifact:dense", "geometry.solution"))


def test_dense_depth_requires_non_empty_canonical_unique_depth_ids() -> None:
    source = _candidate()
    camera_a, camera_b = source.camera_solutions
    depth_a = _depth(camera=camera_a, depth_id="dense:a")
    depth_b = _depth(camera=camera_b, depth_id="dense:b")

    with pytest.raises(ValueError, match="non-empty"):
        _dense_artifact(source_geometry=source, depth_fields=())
    with pytest.raises(ValueError, match="canonical DepthFieldId"):
        _dense_artifact(
            source_geometry=source,
            depth_fields=(depth_b, depth_a),
        )

    duplicate_id = _depth(camera=camera_b, depth_id="dense:a")
    with pytest.raises(ValueError, match="unique DepthFieldIds"):
        _dense_artifact(
            source_geometry=source,
            depth_fields=(depth_a, duplicate_id),
        )


def test_dense_depth_rejects_duplicate_camera_support() -> None:
    source = _candidate()
    camera = source.camera_solutions[0]
    first = _depth(camera=camera, depth_id="dense:a")
    second = _depth(camera=camera, depth_id="dense:b")

    with pytest.raises(ValueError, match="unique CameraSolutionId"):
        _dense_artifact(
            source_geometry=source,
            depth_fields=(first, second),
        )


def test_dense_depth_rejects_duplicate_observation_support() -> None:
    source = _candidate("shared-observation", same_observation=True)
    camera_a, camera_b = source.camera_solutions
    first = _depth(camera=camera_a, depth_id="dense:a")
    second = _depth(camera=camera_b, depth_id="dense:b")

    with pytest.raises(ValueError, match="unique ObservationId"):
        _dense_artifact(
            source_geometry=source,
            depth_fields=(first, second),
        )


def test_dense_depth_requires_exact_source_camera_linkage() -> None:
    source = _candidate()
    camera = source.camera_solutions[0]

    foreign_camera = _depth(
        camera=camera,
        depth_id="dense:a",
        camera_solution_id=CameraSolutionId("camera:foreign"),
    )
    with pytest.raises(ValueError, match="CameraSolution supplied"):
        _dense_artifact(source_geometry=source, depth_fields=(foreign_camera,))

    wrong_observation = _depth(
        camera=camera,
        depth_id="dense:a",
        observation=ObservationId("obs:foreign"),
    )
    with pytest.raises(ValueError, match="ObservationId"):
        _dense_artifact(source_geometry=source, depth_fields=(wrong_observation,))

    wrong_dimensions = _depth(
        camera=camera,
        depth_id="dense:a",
        dimensions=ImageDimensions(width_px=3, height_px=1),
    )
    with pytest.raises(ValueError, match="dimensions"):
        _dense_artifact(source_geometry=source, depth_fields=(wrong_dimensions,))


def test_dense_depth_accepts_partial_camera_coverage_without_fabrication() -> None:
    source = _candidate()
    depth = _depth(
        camera=source.camera_solutions[0],
        depth_id="dense:a",
    )

    artifact = _dense_artifact(
        source_geometry=source,
        depth_fields=(depth,),
    )

    assert artifact.depth_fields == (depth,)
    assert len(source.camera_solutions) == 2
    assert artifact.depth_fields[0].camera_solution_id == source.camera_solutions[0].solution_id
    assert source.camera_solutions[1].solution_id not in {
        item.camera_solution_id for item in artifact.depth_fields
    }


def test_dense_depth_accepts_new_depth_ids_not_in_source_geometry() -> None:
    source = _candidate()
    assert source.geometry_solution.depth_field_ids == ()

    depth = _depth(
        camera=source.camera_solutions[0],
        depth_id="dense:new",
    )
    artifact = _dense_artifact(
        source_geometry=source,
        depth_fields=(depth,),
    )

    assert artifact.depth_fields[0].depth_field_id == DepthFieldId("dense:new")
    assert source.geometry_solution.depth_field_ids == ()
    assert artifact.source_geometry is source


def test_dense_depth_preserves_depth_semantics_and_payload_exactly() -> None:
    source = _candidate()
    camera = source.camera_solutions[0]
    depth = _depth(
        camera=camera,
        depth_id="dense:payload",
        convention="ray-distance",
        depth_values=(2.5, 0.0),
        validity=(True, False),
        confidence=(0.75, 0.0),
    )
    depth_tuple = (depth,)

    artifact = _dense_artifact(
        source_geometry=source,
        depth_fields=depth_tuple,
    )

    assert artifact.depth_fields is depth_tuple
    assert artifact.depth_fields[0] is depth
    assert depth.depth_value_convention == DepthValueConventionName("ray-distance")
    assert depth.depth_values == (2.5, 0.0)
    assert depth.validity == (True, False)
    assert depth.confidence == (0.75, 0.0)


def test_dense_depth_preserves_unresolved_scale_and_local_frame_context() -> None:
    source = _candidate(scale=GeometryScaleStatus.UNRESOLVED)
    artifact = _dense_artifact(source_geometry=source)

    assert artifact.source_geometry is source
    assert artifact.source_geometry.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert artifact.source_geometry.geometry_solution.local_frame_id == LocalFrameId(
        "frame:source"
    )
    assert not hasattr(artifact, "scale_status")
    assert not hasattr(artifact, "world_transform")
    assert not hasattr(artifact, "local_frame_id")


def test_dense_depth_requires_canonical_non_empty_source_ancestry_and_retains_all_parents() -> None:
    parent_a = _artifact("artifact:a", "evidence.alpha")
    parent_c = _artifact("artifact:c", "evidence.gamma")
    source = _candidate(source_artifacts=(parent_a, parent_c))
    additional = _artifact("artifact:b", "evidence.beta")
    ancestry = (parent_a, additional, parent_c)

    artifact = _dense_artifact(
        source_geometry=source,
        source_artifacts=ancestry,
    )

    assert artifact.source_artifacts is ancestry
    assert set(source.source_artifacts).issubset(set(artifact.source_artifacts))

    with pytest.raises(ValueError, match="non-empty"):
        _dense_artifact(source_geometry=_candidate(source_artifacts=()), source_artifacts=())
    with pytest.raises(ValueError, match="unique"):
        _dense_artifact(
            source_geometry=source,
            source_artifacts=(parent_a, parent_a, parent_c),
        )
    with pytest.raises(ValueError, match="canonical"):
        _dense_artifact(
            source_geometry=source,
            source_artifacts=(parent_c, parent_a),
        )
    with pytest.raises(ValueError, match="retain every"):
        _dense_artifact(
            source_geometry=source,
            source_artifacts=(parent_a, additional),
        )


def test_dense_depth_allows_new_stage_evidence_when_source_candidate_has_no_artifacts() -> None:
    source = _candidate(source_artifacts=())
    stage_evidence = (_artifact("artifact:dense-input", "evidence.mvs_input"),)

    artifact = _dense_artifact(
        source_geometry=source,
        source_artifacts=stage_evidence,
    )

    assert source.source_artifacts == ()
    assert artifact.source_artifacts is stage_evidence


def test_dense_depth_construction_does_not_mutate_or_reorder_inputs() -> None:
    parent_a = _artifact("artifact:a", "evidence.alpha")
    parent_c = _artifact("artifact:c", "evidence.gamma")
    source = _candidate(source_artifacts=(parent_a, parent_c))
    camera_a, camera_b = source.camera_solutions
    depth_a = _depth(camera=camera_a, depth_id="dense:a")
    depth_b = _depth(camera=camera_b, depth_id="dense:b")
    depths = (depth_a, depth_b)
    ancestry = (parent_a, parent_c)

    source_before = source
    source_cameras_before = source.camera_solutions
    source_geometry_before = source.geometry_solution

    artifact = _dense_artifact(
        source_geometry=source,
        depth_fields=depths,
        source_artifacts=ancestry,
    )

    assert artifact.source_geometry is source
    assert artifact.depth_fields is depths
    assert artifact.source_artifacts is ancestry
    assert source is source_before
    assert source.camera_solutions is source_cameras_before
    assert source.geometry_solution is source_geometry_before
    assert source.geometry_solution.depth_field_ids == ()


def test_dense_depth_module_exposes_no_solver_fusion_quality_or_later_surface() -> None:
    forbidden = {
        "MVS",
        "PatchMatch",
        "StereoFusion",
        "OpenMVS",
        "PointMap",
        "SurfaceModel",
        "QualityDecision",
        "QualityPolicy",
        "Router",
        "torch",
        "numpy",
        "pycolmap",
        "MasterScene",
        "RuntimeScene",
    }

    assert forbidden.isdisjoint(vars(dense_depth_module))
