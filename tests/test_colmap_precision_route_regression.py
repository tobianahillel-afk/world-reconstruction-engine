from __future__ import annotations

import json
import os
import random
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

from wre.domain.geometry_solutions import GeometryScaleStatus
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.hashing import FileContentHash, hash_file_content
from wre.reconstruction import (
    COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
    COLMAP_IMPORTER_VERSION,
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationRequest,
    ColmapGeometryOutcome,
    ColmapGeometryOutcomeState,
    ColmapGlobalReconstructionConfig,
    ColmapGlobalReconstructionRequest,
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionRequest,
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapReconstructionImportConfig,
    ColmapReconstructionImportRequest,
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
    canonicalize_colmap_sparse_model,
    colmap_sparse_model_content_identity,
    convert_sparse_reconstruction_estimate,
    extract_colmap_features,
    import_colmap_reconstruction,
    match_colmap_pairs,
    reconstruct_colmap_globally,
    reconstruct_colmap_incrementally,
    verify_colmap_geometry,
)
from wre.regression import load_fixture

_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "synthetic" / "colmap-l3-end-to-end" / "fixture.json"
)
_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "registry" / "adapter-models.yaml"
_V2L12_ADAPTER_IDS = (
    "colmap.geometric_verification",
    "colmap.global_precision_geometry",
    "colmap.incremental_precision_geometry",
    "colmap.local_features",
    "colmap.pair_matching",
    "colmap.precision_geometry",
)


def _load_scene() -> dict[str, Any]:
    fixture = load_fixture(_FIXTURE_PATH)
    raw = json.loads(fixture.resolve_input("scene.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("V2L12.5 scene specification must be an object")
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
    image_paths: list[Path] = []
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

        path = directory / f"view-{camera_index:03d}.pgm"
        path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))
        image_paths.append(path)
    return tuple(image_paths)


def _observation(path: Path, index: int) -> ImageObservation:
    content = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(f"obs:v2l12-5:{index:03d}"),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("fixture:v2l12-5"), locator=path.name),
        received_at=datetime(2026, 9, 22, 10, 0, tzinfo=UTC),
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
        started_at=datetime(2026, 9, 22, 10, 0, tzinfo=UTC) + timedelta(minutes=minute),
        configuration_sha256=configuration_sha256,
    )


def _native_model_hashes(
    output_path: Path,
    models: tuple[ColmapSparseModelArtifact, ...],
) -> dict[tuple[int, str], FileContentHash]:
    hashes: dict[tuple[int, str], FileContentHash] = {}
    for model in models:
        expected_paths = tuple(item.relative_path for item in model.files)
        actual_paths = tuple(
            path.relative_to(output_path / model.relative_path).as_posix()
            for path in sorted(
                (output_path / model.relative_path).rglob("*"),
                key=lambda item: item.as_posix(),
            )
            if path.is_file()
        )
        assert actual_paths == expected_paths
        for model_file in model.files:
            digest = hash_file_content(output_path / model.relative_path / model_file.relative_path)
            assert digest.sha256 == model_file.sha256
            assert digest.byte_length == model_file.byte_length
            hashes[(model.model_index, model_file.relative_path)] = digest
    return hashes


def _assert_bridge_matches_direct(
    direct: Any,
    bridged: Any,
) -> None:
    direct_cameras = {camera.observation_id: camera for camera in direct.camera_solutions}
    bridged_cameras = {camera.observation_id: camera for camera in bridged.camera_solutions}
    assert set(direct_cameras) == set(bridged_cameras)
    for observation_id in direct_cameras:
        direct_camera = direct_cameras[observation_id]
        bridged_camera = bridged_cameras[observation_id]
        assert direct_camera.rotation_matrix == bridged_camera.rotation_matrix
        assert direct_camera.translation_xyz == bridged_camera.translation_xyz
        assert direct_camera.projection_model == bridged_camera.projection_model
        assert direct_camera.dimensions == bridged_camera.dimensions
        assert direct_camera.intrinsic_parameters == bridged_camera.intrinsic_parameters

    assert direct.point_map.source_observation_ids == bridged.point_map.source_observation_ids
    assert len(direct.point_map.positions_xyz) == len(bridged.point_map.positions_xyz)
    assert Counter(direct.point_map.positions_xyz) == Counter(bridged.point_map.positions_xyz)
    assert (
        direct.geometry_solution.scale_status
        is bridged.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )


def test_v2l12_5_reuses_versioned_image_based_fixture_without_new_benchmark_truth() -> None:
    fixture = load_fixture(_FIXTURE_PATH)
    scene = _load_scene()

    assert fixture.fixture_id == "colmap-l3-end-to-end"
    assert fixture.kind == "synthetic"
    assert fixture.deterministic is True
    assert fixture.seed == scene["seed"] == 7301
    assert len(_number_list(scene, "camera_centers_x")) == 12


def test_v2l12_shipping_decision_remains_explicitly_experimental() -> None:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    entries = {
        entry["adapter_id"]: entry
        for entry in document["entries"]
        if entry["adapter_id"] in _V2L12_ADAPTER_IDS
    }

    assert tuple(sorted(entries)) == _V2L12_ADAPTER_IDS
    assert all(entry["shipping_status"] == "experimental" for entry in entries.values())
    assert all(entry["dependency_refs"] == ["colmap"] for entry in entries.values())
    assert all(entry["model"] is None for entry in entries.values())
    assert all(entry["checkpoint"] is None for entry in entries.values())
    assert all(entry["artifact_key_hardware_policy"] == "required" for entry in entries.values())
    assert all(entry["license"]["review"] == "approved" for entry in entries.values())


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="V2L12.5 route regression runs only in the dedicated COLMAP integration lane",
)
def test_real_colmap_current_routes_share_verified_evidence_and_preserve_v2_semantics(
    tmp_path: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    scene = _load_scene()
    image_paths = _render_scene(scene, tmp_path / "rendered")
    observations = tuple(_observation(path, index) for index, path in enumerate(image_paths))
    observation_ids = tuple(item.observation_id for item in observations)
    revision = str(pycolmap.COLMAP_build)

    feature_config = ColmapFeatureExtractionConfig()
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=_run(
                value="run:v2l12-5:features",
                implementation="pycolmap.extract_features",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=feature_config.sha256,
                minute=0,
            ),
            inputs=tuple(
                ColmapFeatureInput(observation=observation, source_path=path)
                for observation, path in zip(observations, image_paths, strict=True)
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
                value="run:v2l12-5:matching",
                implementation="pycolmap.match_exhaustive",
                version="4.2.0",
                revision=revision,
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
                value="run:v2l12-5:verification",
                implementation="pycolmap.geometric_verification",
                version="4.2.0",
                revision=revision,
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
        for observation, path in zip(observations, image_paths, strict=True)
    )

    incremental_config = ColmapIncrementalReconstructionConfig()
    incremental = reconstruct_colmap_incrementally(
        ColmapIncrementalReconstructionRequest(
            run=_run(
                value="run:v2l12-5:incremental",
                implementation="pycolmap.incremental_mapping",
                version="4.2.0",
                revision=revision,
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
    assert incremental.model_count >= 1
    incremental_native_hashes = _native_model_hashes(
        incremental.output_path,
        incremental.models,
    )

    direct_incremental = tuple(
        canonicalize_colmap_sparse_model(
            output_path=incremental.output_path,
            model_artifact=model,
            features=features,
            expected_environment=incremental.environment,
            module=pycolmap,
        )
        for model in incremental.models
    )
    assert _native_model_hashes(incremental.output_path, incremental.models) == (
        incremental_native_hashes
    )
    assert tuple(model.source_model_identity_sha256 for model in direct_incremental) == tuple(
        colmap_sparse_model_content_identity(model) for model in incremental.models
    )
    incremental_outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
        native_models=incremental.models,
        canonical_models=direct_incremental,
    )
    assert incremental_outcome.state in {
        ColmapGeometryOutcomeState.SINGLE_MODEL,
        ColmapGeometryOutcomeState.DISCONNECTED_MODELS,
    }

    import_config = ColmapReconstructionImportConfig()
    imported = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=_run(
                value="run:v2l12-5:legacy-import",
                implementation="wre.colmap_reconstruction_importer",
                version=COLMAP_IMPORTER_VERSION,
                revision=None,
                observation_ids=observation_ids,
                configuration_sha256=import_config.sha256,
                minute=4,
            ),
            reconstruction=incremental,
            features=features,
            config=import_config,
        ),
        module=pycolmap,
    )
    assert imported.model_count == incremental.model_count
    bridged_first = convert_sparse_reconstruction_estimate(imported.models[0].estimate)
    _assert_bridge_matches_direct(direct_incremental[0], bridged_first)

    global_config = ColmapGlobalReconstructionConfig()
    global_result = reconstruct_colmap_globally(
        ColmapGlobalReconstructionRequest(
            run=_run(
                value="run:v2l12-5:global",
                implementation="pycolmap.global_mapping",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=global_config.sha256,
                minute=5,
            ),
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=tmp_path / "global-sparse",
            config=global_config,
        ),
        module=pycolmap,
    )
    assert hash_file_content(verification.database_path) == verification_before
    assert global_result.model_count >= 1
    global_native_hashes = _native_model_hashes(
        global_result.output_path,
        global_result.models,
    )

    direct_global = tuple(
        canonicalize_colmap_sparse_model(
            output_path=global_result.output_path,
            model_artifact=model,
            features=features,
            expected_environment=global_result.environment,
            module=pycolmap,
        )
        for model in global_result.models
    )
    assert _native_model_hashes(global_result.output_path, global_result.models) == (
        global_native_hashes
    )
    assert tuple(model.source_model_identity_sha256 for model in direct_global) == tuple(
        colmap_sparse_model_content_identity(model) for model in global_result.models
    )
    global_outcome = ColmapGeometryOutcome(
        source_adapter_id=COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
        native_models=global_result.models,
        canonical_models=direct_global,
    )
    assert global_outcome.state in {
        ColmapGeometryOutcomeState.SINGLE_MODEL,
        ColmapGeometryOutcomeState.DISCONNECTED_MODELS,
    }

    for outcome in (incremental_outcome, global_outcome):
        local_frames = tuple(
            model.geometry_solution.local_frame_id.value for model in outcome.canonical_models
        )
        assert len(local_frames) == len(set(local_frames))
        assert all(
            model.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
            for model in outcome.canonical_models
        )

    comparison_facts = {
        "global_model_count": global_outcome.model_count,
        "global_registered_observation_count": sum(
            len(model.point_map.source_observation_ids) for model in global_outcome.canonical_models
        ),
        "incremental_model_count": incremental_outcome.model_count,
        "incremental_registered_observation_count": sum(
            len(model.point_map.source_observation_ids)
            for model in incremental_outcome.canonical_models
        ),
    }
    assert set(comparison_facts) == {
        "global_model_count",
        "global_registered_observation_count",
        "incremental_model_count",
        "incremental_registered_observation_count",
    }
    assert all(value > 0 for value in comparison_facts.values())
