from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

from wre.domain.estimated_geometry import LocalScaleStatus
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    SourceId,
    SourceRef,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.hashing import hash_file_content
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


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
INPUT_DIR = HERE / "input_images"
FIXTURE_PATH = (
    REPO_ROOT
    / "tests"
    / "fixtures"
    / "synthetic"
    / "colmap-l3-end-to-end"
    / "fixture.json"
)


def _observation(path: Path, index: int) -> ImageObservation:
    content = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(f"obs:demo:v1:{index:03d}"),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(
            source_id=SourceId("demo:v1-preview"),
            locator=path.name,
        ),
        received_at=datetime(2026, 9, 15, 13, 0, tzinfo=UTC),
    )


def _run(
    *,
    value: str,
    implementation: str,
    version: str,
    revision: str | None,
    observation_ids: tuple[ObservationId, ...],
    configuration_sha256,
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
        started_at=datetime(2026, 9, 15, 13, 0, tzinfo=UTC)
        + timedelta(minutes=minute),
        configuration_sha256=configuration_sha256,
    )


def _camera_center(rotation_matrix, translation_xyz):
    return tuple(
        -sum(rotation_matrix[row][col] * translation_xyz[row] for row in range(3))
        for col in range(3)
    )


def _write_wre_ascii_ply(path: Path, imported) -> None:
    if imported.model_count != 1:
        raise ValueError("demo ASCII PLY export expects exactly one imported model")
    estimate = imported.models[0].estimate
    vertices: list[tuple[float, float, float, int, int, int]] = []
    for point in estimate.points3d:
        x, y, z = point.position_xyz
        vertices.append((x, y, z, 235, 235, 235))
    for pose in estimate.camera_poses:
        x, y, z = _camera_center(pose.rotation_matrix, pose.translation_xyz)
        vertices.append((x, y, z, 255, 64, 64))

    lines = [
        "ply",
        "format ascii 1.0",
        "comment WRE L3 imported sparse reconstruction",
        "comment white points = reconstructed 3D points; red points = reconstructed camera centers",
        "comment local scale is UNRESOLVED; coordinates are not world/geographic coordinates",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}"
        for x, y, z, r, g, b in vertices
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the WRE L3 sparse reconstruction demo and export a PLY."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "generated",
        help="Output directory (default: demo/v1_preview/generated).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete an existing output directory before running.",
    )
    args = parser.parse_args()

    output_dir = args.output.expanduser().resolve()
    if output_dir.exists():
        if not args.force:
            raise SystemExit(
                f"{output_dir} already exists; rerun with --force to replace it"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    image_paths = tuple(sorted(INPUT_DIR.glob("view-*.pgm")))
    if len(image_paths) != 12:
        raise SystemExit(
            f"expected 12 committed demo images in {INPUT_DIR}, found {len(image_paths)}"
        )

    try:
        import pycolmap
    except ImportError as exc:
        raise SystemExit(
            "pycolmap is missing. Install the exact external solver environment "
            "documented in README.md (pycolmap==4.2.0, numpy==2.5.3)."
        ) from exc

    if str(pycolmap.__version__) != "4.2.0":
        raise SystemExit(
            f"expected pycolmap==4.2.0, got {pycolmap.__version__!s}"
        )

    observations = tuple(
        _observation(path, index) for index, path in enumerate(image_paths)
    )
    observation_ids = tuple(item.observation_id for item in observations)
    revision = str(pycolmap.COLMAP_build)
    work = output_dir / "work"
    work.mkdir()

    feature_config = ColmapFeatureExtractionConfig()
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=_run(
                value="run:demo:v1:features",
                implementation="pycolmap.extract_features",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=feature_config.sha256,
                minute=0,
            ),
            inputs=tuple(
                ColmapFeatureInput(observation=observation, source_path=path)
                for observation, path in zip(
                    observations, image_paths, strict=True
                )
            ),
            database_path=work / "features.db",
            config=feature_config,
        ),
        module=pycolmap,
    )

    matching_config = ColmapPairMatchingConfig()
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=_run(
                value="run:demo:v1:matching",
                implementation="pycolmap.match_exhaustive",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=matching_config.sha256,
                minute=1,
            ),
            features=features,
            database_path=work / "matches.db",
            config=matching_config,
        ),
        module=pycolmap,
    )

    verification_config = ColmapGeometricVerificationConfig()
    verification = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=_run(
                value="run:demo:v1:verification",
                implementation="pycolmap.geometric_verification",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=verification_config.sha256,
                minute=2,
            ),
            matching=matching,
            database_path=work / "verified.db",
            config=verification_config,
        ),
        module=pycolmap,
    )

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
                value="run:demo:v1:reconstruction",
                implementation="pycolmap.incremental_mapping",
                version="4.2.0",
                revision=revision,
                observation_ids=observation_ids,
                configuration_sha256=reconstruction_config.sha256,
                minute=3,
            ),
            verification=verification,
            inputs=reconstruction_inputs,
            output_path=work / "sparse",
            config=reconstruction_config,
        ),
        module=pycolmap,
    )

    import_config = ColmapReconstructionImportConfig()
    imported = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=_run(
                value="run:demo:v1:import",
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

    if reconstruction.model_count != 1:
        raise SystemExit(
            f"expected exactly one reconstruction model, got {reconstruction.model_count}"
        )

    model = reconstruction.models[0]
    model_path = reconstruction.output_path / model.relative_path
    native = pycolmap.Reconstruction(str(model_path))
    native.export_PLY(str(output_dir / "reconstruction_colmap.ply"))
    text_dir = output_dir / "colmap_text_model"
    native.write_text(str(text_dir))
    _write_wre_ascii_ply(output_dir / "reconstruction_wre_ascii.ply", imported)

    metrics = {
        "observation_count": len(observations),
        "attempted_pair_count": matching.attempted_pair_count,
        "raw_matched_pair_count": len(matching.pairs),
        "verified_pair_count": len(verification.geometries),
        "reconstructed_model_count": reconstruction.model_count,
        "registered_image_count": sum(
            item.num_registered_images for item in reconstruction.models
        ),
        "imported_model_count": imported.model_count,
        "imported_observation_count": sum(
            item.estimate.observation_count for item in imported.models
        ),
        "imported_point_count": sum(
            item.estimate.point_count for item in imported.models
        ),
        "unresolved_scale_model_count": sum(
            item.estimate.scale_status is LocalScaleStatus.UNRESOLVED
            for item in imported.models
        ),
    }
    fixture = load_fixture(FIXTURE_PATH)
    regression = evaluate_fixture(
        fixture,
        metrics,
        runner="demo.v1_preview.run_demo",
    )
    if not regression.passed:
        raise SystemExit(
            "reconstruction finished but the L3.7 regression contract failed:\n"
            + json.dumps(regression.to_dict(), indent=2, sort_keys=True)
        )

    summary = {
        "pycolmap_version": str(pycolmap.__version__),
        "colmap_version": str(pycolmap.COLMAP_version),
        "colmap_build": str(pycolmap.COLMAP_build),
        "fixture_id": fixture.fixture_id,
        "fixture_passed": regression.passed,
        "scale_status": "unresolved",
        "metrics": metrics,
        "files": {
            "reconstruction_colmap_ply": "reconstruction_colmap.ply",
            "reconstruction_wre_ascii_ply": "reconstruction_wre_ascii.ply",
            "colmap_text_model": "colmap_text_model/",
            "native_workdir": "work/",
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"\nASCII WRE PLY exported to: "
        f"{output_dir / 'reconstruction_wre_ascii.ply'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
