from __future__ import annotations

import json
import os
import random
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from wre.domain.estimated_geometry import LocalScaleStatus
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    SourceId,
    SourceRef,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.hashing import FileContentHash, hash_file_content
from wre.reconstruction import (
    COLMAP_IMPORTER_VERSION,
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationRequest,
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionRequest,
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapReconstructionImportConfig,
    ColmapReconstructionImportRequest,
    ColmapReconstructionInput,
    extract_colmap_features,
    import_colmap_reconstruction,
    match_colmap_pairs,
    reconstruct_colmap_incrementally,
    verify_colmap_geometry,
)
from wre.regression import evaluate_fixture, load_fixture

_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "synthetic"
    / "colmap-l3-end-to-end"
    / "fixture.json"
)


def _load_scene() -> dict[str, Any]:
    fixture = load_fixture(_FIXTURE_PATH)
    raw = json.loads(fixture.resolve_input("scene.json").read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("L3.7 scene specification must be an object")
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
    if patch_size < 9 or patch_size % 2 == 0:
        raise ValueError("patch_size must be an odd integer >= 9")

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
    for camera_index, (center_x, center_y) in enumerate(
        zip(camera_x, camera_y, strict=True)
    ):
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
        observation_id=ObservationId(f"obs:l37:{index:03d}"),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("fixture:l37"), locator=path.name),
        received_at=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
    )


def _run(
    *,
    value: str,
    implementation: str,
    version: str,
    revision: str | None,
    observation_ids: tuple[ObservationId, ...],
    configuration_sha256: object,
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
        started_at=datetime(2026, 9, 15, 13, 0, tzinfo=UTC) + timedelta(minutes=minute),
        configuration_sha256=configuration_sha256,  # type: ignore[arg-type]
    )


def _native_model_hashes(
    output_path: Path,
    models: tuple[object, ...],
) -> dict[tuple[int, str], FileContentHash]:
    hashes: dict[tuple[int, str], FileContentHash] = {}
    for raw_model in models:
        model = raw_model
        model_index = int(getattr(model, "model_index"))
        relative_path = str(getattr(model, "relative_path"))
        for raw_file in getattr(model, "files"):
            file_path = str(getattr(raw_file, "relative_path"))
            hashes[(model_index, file_path)] = hash_file_content(
                output_path / relative_path / file_path
            )
    return hashes


def test_l37_fixture_metadata_is_valid_and_matches_camera_prior() -> None:
    fixture = load_fixture(_FIXTURE_PATH)
    scene = _load_scene()

    assert fixture.fixture_id == "colmap-l3-end-to-end"
    assert fixture.kind == "synthetic"
    assert fixture.lane == "full"
    assert fixture.deterministic is True
    assert fixture.seed == scene["seed"] == 7301
    assert len(_number_list(scene, "camera_centers_x")) == 12
    assert float(scene["focal_length_px"]) == pytest.approx(
        1.2 * int(scene["width"])
    )


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="L3.7 full fixture runs only in the dedicated COLMAP integration lane",
)
def test_real_colmap_l3_end_to_end_fixture(tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    fixture = load_fixture(_FIXTURE_PATH)
    scene = _load_scene()
    image_paths = _render_scene(scene, tmp_path / "rendered")
    observations = tuple(_observation(path, index) for index, path in enumerate(image_paths))
    observation_ids = tuple(item.observation_id for item in observations)
    revision = str(pycolmap.COLMAP_build)

    feature_config = ColmapFeatureExtractionConfig()
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=_run(
                value="run:l37:features",
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
    feature_hash = hash_file_content(features.database_path)

    matching_config = ColmapPairMatchingConfig()
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=_run(
                value="run:l37:matching",
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
    assert hash_file_content(features.database_path) == feature_hash
    matching_hash = hash_file_content(matching.database_path)

    verification_config = ColmapGeometricVerificationConfig()
    verification = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=_run(
                value="run:l37:verification",
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
    assert hash_file_content(matching.database_path) == matching_hash
    verification_hash = hash_file_content(verification.database_path)

    image_name_by_observation = {
        item.observation_id: item.image_name for item in features.images
    }
    reconstruction_inputs = tuple(
        ColmapReconstructionInput(
            observation=observation,
            source_path=path,
            image_name=image_name_by_observation[observation.observation_id],
        )
        for observation, path in zip(observations, image_paths, strict=True)
    )
    reconstruction_config = ColmapIncrementalReconstructionConfig()
    reconstruction = reconstruct_colmap_incrementally(
        ColmapIncrementalReconstructionRequest(
            run=_run(
                value="run:l37:reconstruction",
                implementation="pycolmap.incremental_mapping",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=reconstruction_config.sha256,
                minute=3,
            ),
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=tmp_path / "sparse",
            config=reconstruction_config,
        ),
        module=pycolmap,
    )
    assert hash_file_content(verification.database_path) == verification_hash
    native_hashes_before = _native_model_hashes(
        reconstruction.output_path,
        reconstruction.models,
    )

    import_config = ColmapReconstructionImportConfig()
    imported = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=_run(
                value="run:l37:import",
                implementation="wre.colmap_reconstruction_importer",
                version=COLMAP_IMPORTER_VERSION,
                revision=None,
                observation_ids=observation_ids,
                configuration_sha256=import_config.sha256,
                minute=4,
            ),
            reconstruction=reconstruction,
            features=features,
            config=import_config,
        ),
        module=pycolmap,
    )
    assert _native_model_hashes(reconstruction.output_path, reconstruction.models) == (
        native_hashes_before
    )

    metrics = {
        "observation_count": len(observations),
        "attempted_pair_count": matching.attempted_pair_count,
        "raw_matched_pair_count": len(matching.pairs),
        "verified_pair_count": len(verification.geometries),
        "reconstructed_model_count": reconstruction.model_count,
        "registered_image_count": sum(
            model.num_registered_images for model in reconstruction.models
        ),
        "imported_model_count": imported.model_count,
        "imported_observation_count": sum(
            model.estimate.observation_count for model in imported.models
        ),
        "imported_point_count": sum(
            model.estimate.point_count for model in imported.models
        ),
        "unresolved_scale_model_count": sum(
            model.estimate.scale_status is LocalScaleStatus.UNRESOLVED
            for model in imported.models
        ),
    }
    regression = evaluate_fixture(
        fixture,
        metrics,
        runner="wre.l3.7.colmap-end-to-end",
    )

    assert regression.passed, regression.to_dict()
    assert all(
        model.estimate.scale_status is LocalScaleStatus.UNRESOLVED
        for model in imported.models
    )
