from __future__ import annotations

import json
import os
import random
import subprocess
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
    GeometryScaleStatus,
    ImageDimensions,
    LocalFrameId,
    MetricVector,
    ObservationId,
    Sha256Digest,
)
from wre.domain.decoded_images import DECODED_IMAGE_PYRAMID_KIND, DecodedImagePyramidSpec
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import ImageObservation, MediaAssetRef, SourceId, SourceRef
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.decoded_images import (
    DecodedImagePyramidMaterializationRequest,
    FFmpegDecodedImagePyramidMaterializer,
)
from wre.ingestion.hashing import hash_file_content
from wre.ingestion.keyframes import FFmpegToolchain
from wre.reconstruction import (
    COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID,
    COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    ColmapBundleAdjustmentRefinementAdapter,
    ColmapBundleAdjustmentRefinementConfig,
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationRequest,
    ColmapGeometryOutcome,
    ColmapGeometryOutcomeState,
    ColmapGeometryRefinementSource,
    ColmapGlobalReconstructionConfig,
    ColmapGlobalReconstructionRequest,
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionRequest,
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapReconstructionInput,
    CompetingGeometrySolutions,
    Da3ExecutionRequest,
    Da3ImageInput,
    GeometryConsensusRequest,
    GeometryRefinementRequest,
    GeometrySolutionCandidate,
    canonicalize_colmap_sparse_model,
    colmap_native_sparse_model_artifact_ref,
    evaluate_geometry_consensus,
    execute_da3_base_preview,
    extract_colmap_features,
    match_colmap_pairs,
    reconstruct_colmap_globally,
    reconstruct_colmap_incrementally,
    verify_colmap_geometry,
)
from wre.reconstruction.feed_forward_camera_quality import (
    CameraPoseQualityRequest,
    evaluate_camera_pose_quality,
)
from wre.regression import load_fixture

_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "synthetic" / "colmap-l3-end-to-end" / "fixture.json"
)
_WORKFLOW_PATH = (
    Path(__file__).parents[1] / ".github" / "workflows" / "geometry-controlled-benchmark.yml"
)
_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_POSE_METRIC_NAMES = (
    "geometry.camera.observation_coverage_ratio",
    "geometry.camera.relative_rotation_error_deg_median",
    "geometry.camera.relative_translation_direction_error_deg_median",
    "geometry.camera.translation_pair_coverage_ratio",
)
_FIXTURE_ARTIFACT = ArtifactRef(
    artifact_id=ArtifactId("artifact:fixture:colmap-l3-end-to-end"),
    artifact_kind=ArtifactKind("benchmark.fixture"),
)
_REFERENCE_ARTIFACT = ArtifactRef(
    artifact_id=ArtifactId("artifact:reference:colmap-l3-end-to-end"),
    artifact_kind=ArtifactKind("benchmark.camera_reference"),
)
_FORBIDDEN_RECORD_KEYS = {
    "default_route",
    "overall_score",
    "preferred_route",
    "qualitydecision",
    "rank",
    "retry",
    "fallback",
    "shipping_promotion",
    "threshold",
    "winner",
}


class _ControlledBenchmarkUnresolved(RuntimeError):
    """Raised when the mandatory controlled benchmark cannot retain its three hypotheses."""


def _canonical_artifacts(*refs: ArtifactRef) -> tuple[ArtifactRef, ...]:
    by_identity = {
        (ref.artifact_id.value, ref.artifact_kind.value): ref
        for ref in refs
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _artifact_record(ref: ArtifactRef) -> dict[str, str]:
    return {
        "artifact_id": ref.artifact_id.value,
        "artifact_kind": ref.artifact_kind.value,
    }


def _producer_record(producer: ArtifactProducerIdentity) -> dict[str, Any]:
    return {
        "producer": {
            "implementation": producer.producer.implementation,
            "version": producer.producer.version,
            "revision": producer.producer.revision,
        },
        "configuration_sha256": producer.configuration.sha256.value,
        "model": (
            None
            if producer.model is None
            else {
                "name": producer.model.name,
                "version": producer.model.version,
                "revision": producer.model.revision,
            }
        ),
        "checkpoint": (
            None
            if producer.checkpoint is None
            else {
                "identifier": producer.checkpoint.identifier,
                "sha256": producer.checkpoint.sha256.value,
            }
        ),
    }


def _metric_vector_record(vector: MetricVector) -> list[dict[str, Any]]:
    return [
        {
            "name": observation.descriptor.name.value,
            "dimension": observation.descriptor.dimension.value,
            "unit": observation.descriptor.unit.value,
            "direction": observation.descriptor.direction.value,
            "aggregation": observation.descriptor.aggregation.value,
            "value": observation.value,
            "provenance": {
                "evaluator": _producer_record(observation.provenance.evaluator),
                "input_artifacts": [
                    _artifact_record(ref) for ref in observation.provenance.input_artifacts
                ],
            },
        }
        for observation in vector.observations
    ]


def _assert_no_selection_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = key.lower().replace("-", "_")
            assert normalized not in _FORBIDDEN_RECORD_KEYS
            _assert_no_selection_fields(child)
    elif isinstance(value, list):
        for child in value:
            _assert_no_selection_fields(child)


def _candidate_record(
    *,
    route_id: str,
    candidate: GeometrySolutionCandidate,
    reference_metrics: MetricVector,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "geometry_solution_id": candidate.geometry_solution_id.value,
        "local_frame_id": candidate.geometry_solution.local_frame_id.value,
        "scale_status": candidate.geometry_solution.scale_status.value,
        "observation_ids": [
            camera.observation_id.value for camera in candidate.camera_solutions
        ],
        "projection_models": sorted(
            {camera.projection_model.value for camera in candidate.camera_solutions}
        ),
        "camera_solution_ids": [
            camera.solution_id.value for camera in candidate.camera_solutions
        ],
        "depth_field_ids": [
            depth.depth_field_id.value for depth in candidate.depth_fields
        ],
        "point_map_ids": [
            point.point_map_id.value for point in candidate.point_maps
        ],
        "producer": _producer_record(candidate.producer),
        "source_artifacts": [
            _artifact_record(ref) for ref in candidate.source_artifacts
        ],
        "trusted_reference_metrics": _metric_vector_record(reference_metrics),
    }


def _load_scene() -> dict[str, Any]:
    fixture = load_fixture(_FIXTURE_PATH)
    raw = json.loads(fixture.resolve_input("scene.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V2L14.6 scene specification must be an object")
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
        observation_id=ObservationId(f"obs:v2l14-6:{index:03d}"),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("fixture:v2l14-6"), locator=path.name),
        received_at=datetime(2026, 9, 24, 8, 0, tzinfo=UTC),
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
        started_at=datetime(2026, 9, 24, 8, 0, tzinfo=UTC) + timedelta(minutes=minute),
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

    frame = LocalFrameId("frame:v2l14-6:reference")
    return tuple(
        CameraSolution(
            solution_id=CameraSolutionId(f"camera:v2l14-6:reference:{index:03d}"),
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
                    artifact_id=ArtifactId(f"decoded:v2l14-6:{index:03d}"),
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


def _native_model_bytes(root: Path, model: Any) -> dict[str, bytes]:
    model_path = root / model.relative_path
    return {
        item.relative_path: (model_path / item.relative_path).read_bytes()
        for item in model.files
    }


def _colmap_candidate(
    *,
    canonical: Any,
    native: Any,
    producer: ProducerRef,
    configuration_sha256: Sha256Digest,
    verification_artifact: ArtifactRef,
) -> GeometrySolutionCandidate:
    return GeometrySolutionCandidate(
        geometry_solution=canonical.geometry_solution,
        camera_solutions=canonical.camera_solutions,
        depth_fields=(),
        point_maps=(canonical.point_map,),
        producer=ArtifactProducerIdentity(
            producer=producer,
            configuration=ConfigurationIdentity(sha256=configuration_sha256),
        ),
        source_artifacts=_canonical_artifacts(
            _FIXTURE_ARTIFACT,
            verification_artifact,
            colmap_native_sparse_model_artifact_ref(native),
        ),
    )


def _pose_metrics(
    candidate: GeometrySolutionCandidate,
    references: tuple[CameraSolution, ...],
) -> MetricVector:
    return evaluate_camera_pose_quality(
        CameraPoseQualityRequest(
            candidate_cameras=candidate.camera_solutions,
            reference_cameras=references,
            input_artifacts=(_REFERENCE_ARTIFACT,),
        )
    )


def _route_record(
    *,
    route_id: str,
    state: str,
    model_count: int | None,
    candidate_id: str | None,
    configuration_sha256: str,
) -> dict[str, Any]:
    return {
        "route_id": route_id,
        "state": state,
        "model_count": model_count,
        "candidate_geometry_solution_id": candidate_id,
        "configuration_sha256": configuration_sha256,
    }


def test_record_guard_rejects_selection_semantics_recursively() -> None:
    clean = {
        "schema_version": 1,
        "candidates": [{"route_id": "a"}, {"route_id": "b"}],
        "unavailable_dimensions": ["raw_point_coordinates"],
    }
    _assert_no_selection_fields(clean)

    for forbidden in sorted(_FORBIDDEN_RECORD_KEYS):
        with pytest.raises(AssertionError):
            _assert_no_selection_fields({"nested": [{forbidden: True}]})


def test_workflow_surface_is_cpu_only_and_excludes_deferred_accelerator() -> None:
    text_value = _WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "ubuntu-24.04" in text_value
    assert 'python-version: "3.12.12"' in text_value
    assert "torch==2.4.1+cpu" in text_value
    assert "pycolmap==4.2.0" in text_value
    assert "WRE_GEOMETRY_CONTROLLED_BENCHMARK" in text_value
    lowered = text_value.lower()
    assert "gluemap" not in lowered
    assert "vggt" not in lowered
    assert "cuda" not in lowered


@pytest.mark.skipif(
    os.environ.get("WRE_GEOMETRY_CONTROLLED_BENCHMARK") != "1",
    reason="exact DA3 plus PyCOLMAP benchmark environment is not provisioned",
)
def test_real_controlled_geometry_benchmark(tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    scene = _load_scene()
    paths = _render_scene(scene, tmp_path / "rendered")
    source_hashes_before = tuple(hash_file_content(path) for path in paths)
    observations = tuple(_image_observation(path, index) for index, path in enumerate(paths))
    observation_ids = tuple(item.observation_id for item in observations)
    references = _reference_cameras(scene, observations)
    reference_ids = tuple(camera.observation_id.value for camera in references)
    assert len(observations) == len(references) == 12

    feature_config = ColmapFeatureExtractionConfig()
    feature_run = _run(
        value="run:v2l14-6:features",
        implementation="pycolmap.extract_features",
        version="4.2.0",
        revision=str(pycolmap.COLMAP_build),
        observation_ids=observation_ids,
        configuration_sha256=feature_config.sha256,
        minute=0,
    )
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=feature_run,
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
    matching_run = _run(
        value="run:v2l14-6:matching",
        implementation="pycolmap.match_exhaustive",
        version="4.2.0",
        revision=str(pycolmap.COLMAP_build),
        observation_ids=observation_ids,
        configuration_sha256=matching_config.sha256,
        minute=1,
    )
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=matching_run,
            features=features,
            database_path=tmp_path / "matches.db",
            config=matching_config,
        ),
        module=pycolmap,
    )

    verification_config = ColmapGeometricVerificationConfig()
    verification_run = _run(
        value="run:v2l14-6:verification",
        implementation="pycolmap.geometric_verification",
        version="4.2.0",
        revision=str(pycolmap.COLMAP_build),
        observation_ids=observation_ids,
        configuration_sha256=verification_config.sha256,
        minute=2,
    )
    verification = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=verification_run,
            matching=matching,
            database_path=tmp_path / "verified.db",
            config=verification_config,
        ),
        module=pycolmap,
    )
    verification_before = hash_file_content(verification.database_path)
    verification_artifact = ArtifactRef(
        artifact_id=ArtifactId(f"artifact:verification:{verification_before.sha256.value}"),
        artifact_kind=ArtifactKind("geometry.colmap_verification"),
    )

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
    incremental_run = _run(
        value="run:v2l14-6:incremental",
        implementation="pycolmap.incremental_mapping",
        version="4.2.0",
        revision=str(pycolmap.COLMAP_build),
        observation_ids=observation_ids,
        configuration_sha256=incremental_config.sha256,
        minute=3,
    )
    incremental = reconstruct_colmap_incrementally(
        ColmapIncrementalReconstructionRequest(
            run=incremental_run,
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=tmp_path / "incremental-sparse",
            config=incremental_config,
        ),
        module=pycolmap,
    )
    assert hash_file_content(verification.database_path) == verification_before

    incremental_models = tuple(
        canonicalize_colmap_sparse_model(
            output_path=incremental.output_path,
            model_artifact=model,
            features=features,
            expected_environment=incremental.environment,
            module=pycolmap,
        )
        for model in incremental.models
    )
    incremental_outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        native_models=incremental.models,
        canonical_models=incremental_models,
    )
    if (
        incremental_outcome.state is not ColmapGeometryOutcomeState.SINGLE_MODEL
        or incremental_outcome.model_count != 1
    ):
        raise _ControlledBenchmarkUnresolved(
            "controlled benchmark requires exactly one incremental COLMAP model"
        )
    incremental_native = incremental_outcome.native_models[0]
    incremental_canonical = incremental_outcome.canonical_models[0]
    incremental_candidate = _colmap_candidate(
        canonical=incremental_canonical,
        native=incremental_native,
        producer=incremental_run.producer,
        configuration_sha256=incremental_config.sha256,
        verification_artifact=verification_artifact,
    )
    incremental_native_before = _native_model_bytes(incremental.output_path, incremental_native)

    refinement_source = ColmapGeometryRefinementSource(
        model_root=incremental.output_path,
        model_artifact=incremental_native,
        features=features,
        expected_environment=incremental.environment,
        artifact_ref=colmap_native_sparse_model_artifact_ref(incremental_native),
    )
    refinement_config = ColmapBundleAdjustmentRefinementConfig()
    refinement_result = ColmapBundleAdjustmentRefinementAdapter(
        source=refinement_source,
        output_root=tmp_path / "ba-refined",
        config=refinement_config,
        module=pycolmap,
    ).refine(
        GeometryRefinementRequest(
            initialization=incremental_candidate,
            supporting_artifacts=(refinement_source.artifact_ref,),
        )
    )
    refined_candidate = refinement_result.refined_candidate
    assert refined_candidate.geometry_solution_id != incremental_candidate.geometry_solution_id
    assert refined_candidate.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert _native_model_bytes(incremental.output_path, incremental_native) == incremental_native_before

    global_config = ColmapGlobalReconstructionConfig()
    global_run = _run(
        value="run:v2l14-6:global",
        implementation="pycolmap.global_mapping",
        version="4.2.0",
        revision=str(pycolmap.COLMAP_build),
        observation_ids=observation_ids,
        configuration_sha256=global_config.sha256,
        minute=4,
    )
    global_result = reconstruct_colmap_globally(
        ColmapGlobalReconstructionRequest(
            run=global_run,
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=tmp_path / "global-sparse",
            config=global_config,
        ),
        module=pycolmap,
    )
    assert hash_file_content(verification.database_path) == verification_before

    global_models = tuple(
        canonicalize_colmap_sparse_model(
            output_path=global_result.output_path,
            model_artifact=model,
            features=features,
            expected_environment=global_result.environment,
            module=pycolmap,
        )
        for model in global_result.models
    )
    global_outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
        native_models=global_result.models,
        canonical_models=global_models,
    )

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

    preview_candidate = GeometrySolutionCandidate(
        geometry_solution=preview.geometry.geometry_solution,
        camera_solutions=preview.geometry.camera_solutions,
        depth_fields=preview.geometry.depth_fields,
        point_maps=preview.geometry.point_maps,
        producer=preview.producer,
        source_artifacts=_canonical_artifacts(
            _FIXTURE_ARTIFACT,
            ArtifactRef(
                artifact_id=ArtifactId(
                    f"artifact:da3-checkpoint:{preview.checkpoint.sha256.value}"
                ),
                artifact_kind=ArtifactKind("model.checkpoint"),
            ),
            ArtifactRef(
                artifact_id=ArtifactId(
                    f"artifact:da3-source:{preview.environment.source_revision}"
                ),
                artifact_kind=ArtifactKind("source.checkout"),
            ),
        ),
    )

    candidates: list[GeometrySolutionCandidate] = [
        preview_candidate,
        incremental_candidate,
        refined_candidate,
    ]
    route_by_geometry_id = {
        preview_candidate.geometry_solution_id: "da3.base_preview",
        incremental_candidate.geometry_solution_id: COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        refined_candidate.geometry_solution_id: COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID,
    }

    global_candidate: GeometrySolutionCandidate | None = None
    if (
        global_outcome.state is ColmapGeometryOutcomeState.SINGLE_MODEL
        and global_outcome.model_count == 1
    ):
        global_candidate = _colmap_candidate(
            canonical=global_outcome.canonical_models[0],
            native=global_outcome.native_models[0],
            producer=global_run.producer,
            configuration_sha256=global_config.sha256,
            verification_artifact=verification_artifact,
        )
        candidates.append(global_candidate)
        route_by_geometry_id[
            global_candidate.geometry_solution_id
        ] = COLMAP_GLOBAL_CANONICAL_ADAPTER_ID

    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.geometry_solution_id.value))
    assert len(ordered_candidates) >= 3
    assert {
        route_by_geometry_id[candidate.geometry_solution_id]
        for candidate in ordered_candidates
    }.issuperset(
        {
            "da3.base_preview",
            COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID,
        }
    )
    competing = CompetingGeometrySolutions(candidates=ordered_candidates)
    consensus_artifacts = _canonical_artifacts(
        _FIXTURE_ARTIFACT,
        _REFERENCE_ARTIFACT,
        *(
            source
            for candidate in ordered_candidates
            for source in candidate.source_artifacts
        ),
    )
    consensus = evaluate_geometry_consensus(
        GeometryConsensusRequest(
            competing=competing,
            input_artifacts=consensus_artifacts,
        )
    )
    expected_pair_count = len(ordered_candidates) * (len(ordered_candidates) - 1) // 2
    assert len(consensus.pair_disagreements) == expected_pair_count

    reference_metrics = {
        candidate.geometry_solution_id: _pose_metrics(candidate, references)
        for candidate in ordered_candidates
    }
    for metrics in reference_metrics.values():
        assert tuple(
            observation.descriptor.name.value for observation in metrics.observations
        ) == _POSE_METRIC_NAMES
        assert all(
            observation.provenance.input_artifacts == (_REFERENCE_ARTIFACT,)
            for observation in metrics.observations
        )

    source_hashes_after = tuple(hash_file_content(path) for path in paths)
    assert source_hashes_after == source_hashes_before
    assert hash_file_content(verification.database_path) == verification_before

    route_outcomes = [
        _route_record(
            route_id="da3.base_preview",
            state="complete",
            model_count=None,
            candidate_id=preview_candidate.geometry_solution_id.value,
            configuration_sha256=preview.producer.configuration.sha256.value,
        ),
        _route_record(
            route_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
            state=incremental_outcome.state.value,
            model_count=incremental_outcome.model_count,
            candidate_id=incremental_candidate.geometry_solution_id.value,
            configuration_sha256=incremental_config.sha256.value,
        ),
        _route_record(
            route_id=COLMAP_BUNDLE_ADJUSTMENT_REFINEMENT_ADAPTER_ID,
            state="complete",
            model_count=1,
            candidate_id=refined_candidate.geometry_solution_id.value,
            configuration_sha256=refinement_config.sha256.value,
        ),
        _route_record(
            route_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
            state=global_outcome.state.value,
            model_count=global_outcome.model_count,
            candidate_id=(
                None
                if global_candidate is None
                else global_candidate.geometry_solution_id.value
            ),
            configuration_sha256=global_config.sha256.value,
        ),
    ]

    record = {
        "schema_version": 1,
        "benchmark_id": "v2l14.6-controlled-geometry-comparison",
        "execution_profile": {
            "device_class": "cpu",
            "python_version": "3.12.12",
            "pycolmap_version": str(pycolmap.__version__),
            "colmap_build": str(pycolmap.COLMAP_build),
            "da3_device": preview.environment.device,
            "da3_precision": preview.environment.precision,
            "hardware_runtime_sha256": preview.hardware_runtime.sha256.value,
        },
        "fixture": {
            "fixture_id": "colmap-l3-end-to-end",
            "source_observation_ids": [
                observation.observation_id.value for observation in observations
            ],
            "source_asset_sha256": [
                observation.asset.sha256.value for observation in observations
            ],
            "trusted_reference_camera_observation_ids": list(reference_ids),
            "trusted_reference_artifact": _artifact_record(_REFERENCE_ARTIFACT),
        },
        "shared_classical_evidence": {
            "feature_configuration_sha256": feature_config.sha256.value,
            "matching_configuration_sha256": matching_config.sha256.value,
            "verification_configuration_sha256": verification_config.sha256.value,
            "verification_database_sha256": verification_before.sha256.value,
            "verification_database_byte_length": verification_before.byte_length,
        },
        "route_outcomes": route_outcomes,
        "candidates": [
            _candidate_record(
                route_id=route_by_geometry_id[candidate.geometry_solution_id],
                candidate=candidate,
                reference_metrics=reference_metrics[candidate.geometry_solution_id],
            )
            for candidate in ordered_candidates
        ],
        "consensus": {
            "evaluator": _producer_record(
                consensus.pair_disagreements[0].metrics.observations[0].provenance.evaluator
            ),
            "input_artifacts": [
                _artifact_record(ref) for ref in consensus_artifacts
            ],
            "pair_count": len(consensus.pair_disagreements),
            "pairs": [
                {
                    "left_geometry_solution_id": item.pair.left_geometry_solution_id.value,
                    "right_geometry_solution_id": item.pair.right_geometry_solution_id.value,
                    "metrics": _metric_vector_record(item.metrics),
                }
                for item in consensus.pair_disagreements
            ],
        },
        "unavailable_dimensions": [
            "absolute_translation_across_unrelated_local_frames",
            "intrinsic_parameter_cross_projection_comparison",
            "raw_depth_value_cross_candidate_comparison",
            "raw_point_coordinate_cross_candidate_comparison",
            "scale_cross_candidate_comparison",
        ],
        "limitations": [
            "synthetic_fixture_not_representative_natural_image_benchmark",
            "descriptive_evidence_only",
            "deferred_accelerator_specialist_not_executed",
        ],
    }

    _assert_no_selection_fields(record)
    assert len(record["candidates"]) >= 3
    assert record["consensus"]["pair_count"] == expected_pair_count
    assert reference_ids == tuple(
        observation.observation_id.value for observation in observations
    )
    assert global_outcome.model_count != 1 or global_candidate is not None

    output_path_value = os.environ.get("WRE_GEOMETRY_CONTROLLED_BENCHMARK_OUTPUT")
    if output_path_value:
        Path(output_path_value).write_text(
            json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
