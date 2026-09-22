from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
    DepthField,
    DepthFieldId,
    DepthValueConventionName,
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
    ImageDimensions,
    LocalFrameId,
    MetricVector,
    ObservationId,
    Sha256Digest,
)
from wre.domain.decoded_images import DECODED_IMAGE_PYRAMID_KIND, DecodedImagePyramidSpec
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import ImageObservation, MediaAssetRef, SourceId, SourceRef
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.decoded_images import (
    DecodedImagePyramidMaterializationRequest,
    FFmpegDecodedImagePyramidMaterializer,
)
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.keyframes import FFmpegToolchain
from wre.reconstruction import (
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationRequest,
    ColmapGeometryOutcome,
    ColmapGeometryOutcomeState,
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionRequest,
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapReconstructionInput,
    Da3ExecutionRequest,
    Da3ImageInput,
    canonicalize_colmap_sparse_model,
    execute_da3_base_preview,
    extract_colmap_features,
    match_colmap_pairs,
    reconstruct_colmap_incrementally,
    verify_colmap_geometry,
)
from wre.reconstruction.feed_forward_camera_quality import (
    CameraPoseQualityRequest,
    FeedForwardCameraQualityRequest,
    evaluate_camera_pose_quality,
    evaluate_feed_forward_camera_quality,
)
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult
from wre.regression import load_fixture

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "synthetic" / "colmap-l3-end-to-end" / "fixture.json"
)
_REFERENCE_ARTIFACT = ArtifactRef(
    artifact_id=ArtifactId("fixture:colmap-l3-end-to-end"),
    artifact_kind=ArtifactKind("benchmark.camera_reference"),
)

_POSE_METRIC_NAMES = (
    "geometry.camera.observation_coverage_ratio",
    "geometry.camera.relative_rotation_error_deg_median",
    "geometry.camera.relative_translation_direction_error_deg_median",
    "geometry.camera.translation_pair_coverage_ratio",
)


def _camera(
    observation: str,
    *,
    frame: str,
    center_x: float,
    projection: str = "pinhole",
    intrinsics: tuple[float, ...] = (100.0, 100.0, 50.0, 40.0),
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{frame}:{observation}"),
        observation_id=ObservationId(observation),
        local_frame_id=LocalFrameId(f"frame:{frame}"),
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=100, height_px=80),
        intrinsic_parameters=intrinsics,
        rotation_matrix=_IDENTITY,
        translation_xyz=(-center_x, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )


def _geometry(*cameras: CameraSolution) -> FeedForwardGeometryResult:
    ordered = tuple(sorted(cameras, key=lambda camera: camera.observation_id.value))
    depths = tuple(
        DepthField(
            depth_field_id=DepthFieldId(f"depth:{camera.observation_id.value}"),
            observation_id=camera.observation_id,
            camera_solution_id=camera.solution_id,
            dimensions=camera.dimensions,
            depth_value_convention=DepthValueConventionName("relative-depth"),
            depth_values=(1.0,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            validity=(True,) * (camera.dimensions.width_px * camera.dimensions.height_px),
            confidence=None,
            metrics=MetricVector(observations=()),
        )
        for camera in ordered
    )
    solution = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:comparison"),
        local_frame_id=ordered[0].local_frame_id,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=tuple(camera.solution_id for camera in ordered),
        depth_field_ids=tuple(depth.depth_field_id for depth in depths),
        point_map_ids=(),
        metrics=MetricVector(observations=()),
    )
    return FeedForwardGeometryResult(
        camera_solutions=ordered,
        depth_fields=depths,
        point_maps=(),
        geometry_solution=solution,
    )


def _artifact() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact:camera-reference"),
        artifact_kind=ArtifactKind("geometry.camera_reference"),
    )


def _values(vector: MetricVector) -> dict[str, float]:
    return {item.descriptor.name.value: item.value for item in vector.observations}


def test_pose_request_is_frozen_and_allows_projection_independent_candidate_cameras() -> None:
    references = (
        _camera("obs:a", frame="reference", center_x=0.0),
        _camera("obs:b", frame="reference", center_x=1.0),
    )
    simple_radial = (
        _camera(
            "obs:a",
            frame="classical",
            center_x=0.0,
            projection="simple_radial",
            intrinsics=(100.0, 50.0, 40.0, 0.01),
        ),
        _camera(
            "obs:b",
            frame="classical",
            center_x=1.0,
            projection="simple_radial",
            intrinsics=(100.0, 50.0, 40.0, 0.01),
        ),
    )
    request = CameraPoseQualityRequest(
        candidate_cameras=simple_radial,
        reference_cameras=references,
        input_artifacts=(_artifact(),),
    )

    result = evaluate_camera_pose_quality(request)

    assert tuple(item.descriptor.name.value for item in result.observations) == _POSE_METRIC_NAMES
    assert all(value == 0.0 or value == 1.0 for value in _values(result).values())
    with pytest.raises(FrozenInstanceError):
        request.input_artifacts = ()  # type: ignore[misc]


def test_shared_pose_evaluator_exactly_matches_feed_forward_pose_subset() -> None:
    references = (
        _camera("obs:a", frame="reference", center_x=0.0),
        _camera("obs:b", frame="reference", center_x=1.0),
        _camera("obs:c", frame="reference", center_x=2.0),
    )
    candidate_cameras = (
        _camera("obs:a", frame="preview", center_x=0.0),
        _camera("obs:b", frame="preview", center_x=1.0),
        _camera("obs:c", frame="preview", center_x=2.0),
    )
    artifact = (_artifact(),)

    pose = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=candidate_cameras,
            reference_cameras=references,
            input_artifacts=artifact,
        )
    )
    feed_forward = evaluate_feed_forward_camera_quality(
        FeedForwardCameraQualityRequest(
            candidate=_geometry(*candidate_cameras),
            reference_cameras=references,
            input_artifacts=artifact,
        )
    )
    feed_forward_pose = {
        item.descriptor.name.value: item
        for item in feed_forward.observations
        if item.descriptor.name.value in _POSE_METRIC_NAMES
    }

    assert tuple(item.descriptor.name.value for item in pose.observations) == _POSE_METRIC_NAMES
    assert {item.descriptor.name.value: item for item in pose.observations} == feed_forward_pose


def test_pose_evaluator_keeps_zero_candidate_as_explicit_zero_coverage() -> None:
    reference = (_camera("obs:a", frame="reference", center_x=0.0),)

    result = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=(),
            reference_cameras=reference,
            input_artifacts=(),
        )
    )

    assert _values(result) == {"geometry.camera.observation_coverage_ratio": 0.0}


def test_pose_request_fails_closed_on_noncanonical_or_cross_frame_camera_sets() -> None:
    reference_a = _camera("obs:a", frame="reference", center_x=0.0)
    reference_b = _camera("obs:b", frame="reference", center_x=1.0)
    candidate_a = _camera("obs:a", frame="candidate", center_x=0.0)
    candidate_b = _camera("obs:b", frame="candidate", center_x=1.0)

    with pytest.raises(ValueError, match="canonical ObservationId"):
        CameraPoseQualityRequest(
            candidate_cameras=(candidate_b, candidate_a),
            reference_cameras=(reference_a, reference_b),
            input_artifacts=(),
        )
    with pytest.raises(ValueError, match="share one LocalFrameId"):
        CameraPoseQualityRequest(
            candidate_cameras=(
                candidate_a,
                _camera("obs:b", frame="other", center_x=1.0),
            ),
            reference_cameras=(reference_a, reference_b),
            input_artifacts=(),
        )
    with pytest.raises(ValueError, match="non-empty"):
        CameraPoseQualityRequest(
            candidate_cameras=(candidate_a, candidate_b),
            reference_cameras=(),
            input_artifacts=(),
        )


def _load_scene() -> dict[str, Any]:
    fixture = load_fixture(_FIXTURE_PATH)
    raw = json.loads(fixture.resolve_input("scene.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V2L13.5 scene specification must be an object")
    return raw


def _number_list(scene: dict[str, Any], key: str) -> tuple[float, ...]:
    value = scene[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"scene.{key} must be a non-empty list")
    return tuple(float(item) for item in value)


def _integer_list(scene: dict[str, Any], key: str) -> tuple[int, ...]:
    value = scene[key]
    if not isinstance(value, list) or not value:
        raise ValueError(f"scene.{key} must be a non-empty list")
    return tuple(int(item) for item in value)


def _patch(seed: int, size: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    values: list[int] = []
    for y in range(size):
        for x in range(size):
            if x in (0, size - 1) or y in (0, size - 1):
                values.append(224 if (x + y + seed) % 2 else 24)
            else:
                values.append(240 if rng.getrandbits(1) else 16)
    return tuple(values)


def _render_scene(scene: dict[str, Any], directory: Path) -> tuple[Path, ...]:
    width = int(scene["width"])
    height = int(scene["height"])
    focal = float(scene["focal_length_px"])
    patch_size = int(scene["patch_size"])
    background = int(scene["background_value"])
    scene_seed = int(scene["seed"])
    grid_u = _integer_list(scene, "grid_u")
    grid_v = _integer_list(scene, "grid_v")
    depths = _number_list(scene, "depths")
    camera_x = _number_list(scene, "camera_centers_x")
    camera_y = _number_list(scene, "camera_centers_y")
    if len(camera_x) != len(camera_y):
        raise ValueError("camera center coordinate lists must have equal length")

    points: list[tuple[int, int, float, tuple[int, ...]]] = []
    point_index = 0
    for v in grid_v:
        for u in grid_u:
            depth = depths[point_index % len(depths)]
            points.append(
                (
                    u,
                    v,
                    depth,
                    _patch(scene_seed + point_index * 104_729, patch_size),
                )
            )
            point_index += 1

    directory.mkdir(parents=True, exist_ok=True)
    half = patch_size // 2
    paths: list[Path] = []
    for camera_index, (center_x, center_y) in enumerate(zip(camera_x, camera_y, strict=True)):
        pixels = bytearray([background]) * (width * height)
        for base_u, base_v, depth, patch in points:
            center_u = round(base_u - focal * center_x / depth)
            center_v = round(base_v - focal * center_y / depth)
            for patch_y in range(patch_size):
                image_y = center_v + patch_y - half
                if image_y < 0 or image_y >= height:
                    continue
                for patch_x in range(patch_size):
                    image_x = center_u + patch_x - half
                    if image_x < 0 or image_x >= width:
                        continue
                    pixels[image_y * width + image_x] = patch[patch_y * patch_size + patch_x]

        output = directory / f"view-{camera_index:03d}.pgm"
        output.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))
        paths.append(output)
    return tuple(paths)


def _image_observation(path: Path, index: int) -> ImageObservation:
    content = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(f"obs:v2l13-5:{index:03d}"),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("fixture:v2l13-5"), locator=path.name),
        received_at=datetime(2026, 9, 22, 18, 0, tzinfo=UTC),
    )


def _run(
    *,
    value: str,
    implementation: str,
    version: str,
    revision: str | None,
    observation_ids: tuple[ObservationId, ...],
    configuration_sha256: Sha256Digest,
    minute: int,
) -> ReconstructionRun:
    return ReconstructionRun(
        run_id=ReconstructionRunId(value),
        producer=ProducerRef(
            implementation=implementation,
            version=version,
            revision=revision,
        ),
        input_observation_ids=observation_ids,
        started_at=datetime(2026, 9, 22, 18, 0, tzinfo=UTC) + timedelta(minutes=minute),
        configuration_sha256=configuration_sha256,
    )


def _reference_cameras(
    scene: dict[str, Any],
    observations: tuple[ImageObservation, ...],
) -> tuple[CameraSolution, ...]:
    width = int(scene["width"])
    height = int(scene["height"])
    focal = float(scene["focal_length_px"])
    center_x = _number_list(scene, "camera_centers_x")
    center_y = _number_list(scene, "camera_centers_y")
    if len(observations) != len(center_x) or len(center_x) != len(center_y):
        raise ValueError("reference camera counts must match observations")

    frame = LocalFrameId("frame:v2l13-5:reference")
    return tuple(
        CameraSolution(
            solution_id=CameraSolutionId(f"camera:v2l13-5:reference:{index:03d}"),
            observation_id=observation.observation_id,
            local_frame_id=frame,
            projection_model=CameraProjectionModelName("pinhole"),
            dimensions=ImageDimensions(width_px=width, height_px=height),
            intrinsic_parameters=(
                focal,
                focal,
                float(width) * 0.5,
                float(height) * 0.5,
            ),
            rotation_matrix=_IDENTITY,
            translation_xyz=(-center_x[index], -center_y[index], 0.0),
            uncertainty_artifacts=(),
            metrics=MetricVector(observations=()),
        )
        for index, observation in enumerate(observations)
    )


def _decoded_da3_inputs(
    *,
    scene: dict[str, Any],
    observations: tuple[ImageObservation, ...],
    paths: tuple[Path, ...],
    root: Path,
) -> tuple[Da3ImageInput, ...]:
    materializer = FFmpegDecodedImagePyramidMaterializer(FFmpegToolchain())
    minimum_edge = max(int(scene["width"]), int(scene["height"]))
    inputs: list[Da3ImageInput] = []
    for index, (observation, source_path) in enumerate(zip(observations, paths, strict=True)):
        output_root = root / f"observation-{index:03d}"
        result = materializer.materialize(
            DecodedImagePyramidMaterializationRequest(
                source=observation,
                source_path=source_path,
                output_root=output_root,
                artifact_ref=ArtifactRef(
                    artifact_id=ArtifactId(f"decoded:v2l13-5:{index:03d}"),
                    artifact_kind=ArtifactKind(DECODED_IMAGE_PYRAMID_KIND),
                ),
                spec=DecodedImagePyramidSpec(minimum_max_edge_px=minimum_edge),
            )
        )
        inputs.append(
            Da3ImageInput(
                decoded_result=result,
                materialization_root=output_root,
            )
        )
    return tuple(inputs)


def _metric_record(vector: MetricVector) -> dict[str, float]:
    return {item.descriptor.name.value: item.value for item in vector.observations}


@pytest.mark.skipif(
    os.environ.get("WRE_FEED_FORWARD_CLASSICAL_COMPARISON") != "1",
    reason="exact DA3 plus PyCOLMAP comparison environment is not provisioned",
)
def test_real_da3_preview_and_classical_incremental_share_one_controlled_reference(
    tmp_path: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    scene = _load_scene()
    paths = _render_scene(scene, tmp_path / "rendered")
    observations = tuple(_image_observation(path, index) for index, path in enumerate(paths))
    observation_ids = tuple(item.observation_id for item in observations)
    references = _reference_cameras(scene, observations)
    assert len(observations) == len(references) == 12

    feature_config = ColmapFeatureExtractionConfig()
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=_run(
                value="run:v2l13-5:features",
                implementation="pycolmap.extract_features",
                version="4.2.0",
                revision=str(pycolmap.COLMAP_build),
                observation_ids=observation_ids,
                configuration_sha256=feature_config.sha256,
                minute=0,
            ),
            inputs=tuple(
                ColmapFeatureInput(observation=observation, source_path=path)
                for observation, path in zip(observations, paths, strict=True)
            ),
            database_path=tmp_path / "features.db",
            config=feature_config,
        ),
        module=pycolmap,
    )
    matching_config = ColmapPairMatchingConfig()
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=_run(
                value="run:v2l13-5:matching",
                implementation="pycolmap.match_exhaustive",
                version="4.2.0",
                revision=str(pycolmap.COLMAP_build),
                observation_ids=observation_ids,
                configuration_sha256=matching_config.sha256,
                minute=1,
            ),
            features=features,
            database_path=tmp_path / "matches.db",
            config=matching_config,
        ),
        module=pycolmap,
    )
    verification_config = ColmapGeometricVerificationConfig()
    verification = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=_run(
                value="run:v2l13-5:verification",
                implementation="pycolmap.geometric_verification",
                version="4.2.0",
                revision=str(pycolmap.COLMAP_build),
                observation_ids=observation_ids,
                configuration_sha256=verification_config.sha256,
                minute=2,
            ),
            matching=matching,
            database_path=tmp_path / "verified.db",
            config=verification_config,
        ),
        module=pycolmap,
    )
    verification_before = hash_file_content(verification.database_path)

    image_name_by_observation = {item.observation_id: item.image_name for item in features.images}
    reconstruction_inputs = tuple(
        ColmapReconstructionInput(
            observation=observation,
            source_path=path,
            image_name=image_name_by_observation[observation.observation_id],
        )
        for observation, path in zip(observations, paths, strict=True)
    )
    incremental_config = ColmapIncrementalReconstructionConfig()
    incremental = reconstruct_colmap_incrementally(
        ColmapIncrementalReconstructionRequest(
            run=_run(
                value="run:v2l13-5:incremental",
                implementation="pycolmap.incremental_mapping",
                version="4.2.0",
                revision=str(pycolmap.COLMAP_build),
                observation_ids=observation_ids,
                configuration_sha256=incremental_config.sha256,
                minute=3,
            ),
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=tmp_path / "incremental-sparse",
            config=incremental_config,
        ),
        module=pycolmap,
    )
    assert hash_file_content(verification.database_path) == verification_before

    classical_models = tuple(
        canonicalize_colmap_sparse_model(
            output_path=incremental.output_path,
            model_artifact=model,
            features=features,
            expected_environment=incremental.environment,
            module=pycolmap,
        )
        for model in incremental.models
    )
    classical_outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        native_models=incremental.models,
        canonical_models=classical_models,
    )
    assert classical_outcome.state is ColmapGeometryOutcomeState.SINGLE_MODEL
    assert classical_outcome.model_count == 1
    native_classical_cameras = classical_outcome.canonical_models[0].camera_solutions
    classical_cameras = tuple(
        sorted(
            native_classical_cameras,
            key=lambda camera: camera.observation_id.value,
        )
    )
    assert len(classical_cameras) == len(native_classical_cameras)
    assert {camera.solution_id for camera in classical_cameras} == {
        camera.solution_id for camera in native_classical_cameras
    }
    assert all(camera.projection_model.value == "simple_radial" for camera in classical_cameras)

    source_root = Path(os.environ["WRE_DA3_SOURCE_ROOT"])
    checkpoint_path = Path(os.environ["WRE_DA3_CHECKPOINT_PATH"])
    checkpoint_before = hash_file_content(checkpoint_path)
    da3_inputs = _decoded_da3_inputs(
        scene=scene,
        observations=observations,
        paths=paths,
        root=tmp_path / "decoded",
    )
    preview = execute_da3_base_preview(
        Da3ExecutionRequest(
            inputs=da3_inputs,
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            hardware_runtime=HardwareRuntimeIdentity(
                sha256=Sha256Digest(os.environ["WRE_DA3_HARDWARE_RUNTIME_SHA256"])
            ),
        )
    )
    assert hash_file_content(checkpoint_path) == checkpoint_before
    git_status = subprocess.run(
        ["git", "-C", str(source_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert git_status.stdout.strip() == ""

    preview_metrics = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=preview.geometry.camera_solutions,
            reference_cameras=references,
            input_artifacts=(_REFERENCE_ARTIFACT,),
        )
    )
    classical_metrics = evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=classical_cameras,
            reference_cameras=references,
            input_artifacts=(_REFERENCE_ARTIFACT,),
        )
    )
    preview_names = tuple(item.descriptor.name.value for item in preview_metrics.observations)
    classical_names = tuple(item.descriptor.name.value for item in classical_metrics.observations)
    assert preview_names == classical_names == _POSE_METRIC_NAMES
    assert all(
        item.provenance == preview_metrics.observations[0].provenance
        for item in preview_metrics.observations
    )
    assert all(
        item.provenance == classical_metrics.observations[0].provenance
        for item in classical_metrics.observations
    )

    record = {
        "schema_version": 1,
        "fixture_id": "colmap-l3-end-to-end",
        "reference_observation_count": len(references),
        "directly_comparable_metric_names": list(_POSE_METRIC_NAMES),
        "preview": {
            "adapter_id": "da3.base_preview",
            "source_revision": preview.model.revision,
            "checkpoint_identifier": preview.checkpoint.identifier,
            "checkpoint_sha256": preview.checkpoint.sha256.value,
            "metrics": _metric_record(preview_metrics),
        },
        "classical": {
            "adapter_id": COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            "pycolmap_version": str(pycolmap.__version__),
            "projection_model": "simple_radial",
            "metrics": _metric_record(classical_metrics),
        },
        "intrinsic_comparison": "not_directly_comparable_projection_families",
        "overall_score": None,
        "winner": None,
        "default_route": None,
    }
    assert record["overall_score"] is None
    assert record["winner"] is None
    assert record["default_route"] is None

    output_path_value = os.environ.get("WRE_COMPARISON_OUTPUT")
    if output_path_value:
        Path(output_path_value).write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
