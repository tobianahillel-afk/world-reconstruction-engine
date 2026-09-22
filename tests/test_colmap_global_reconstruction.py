from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.colmap_global_reconstruction as global_module
from wre.domain.artifacts import ArtifactKind
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_canonical_geometry import (
    canonicalize_colmap_sparse_model,
)
from wre.reconstruction.colmap_environment import (
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_global_reconstruction import (
    COLMAP_GLOBAL_CANONICAL_ADAPTER_ID,
    COLMAP_GLOBAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY,
    COLMAP_GLOBAL_CANONICAL_CAPABILITY,
    COLMAP_GLOBAL_CANONICAL_CAPABILITY_NAME,
    COLMAP_GLOBAL_CANONICAL_CHECKPOINT,
    COLMAP_GLOBAL_CANONICAL_DEPENDENCY_REF,
    COLMAP_GLOBAL_CANONICAL_MODEL,
    COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION,
    COLMAP_GLOBAL_CANONICAL_PRODUCER_VERSION,
    COLMAP_GLOBAL_CANONICAL_REPRODUCIBILITY_NOTES,
    COLMAP_GLOBAL_CANONICAL_SHIPPING_STATUS,
    ColmapGlobalReconstructionConfig,
    ColmapGlobalReconstructionError,
    ColmapGlobalReconstructionRequest,
    reconstruct_colmap_globally,
)
from wre.reconstruction.colmap_reconstruction import ColmapReconstructionInput
from wre.reconstruction.colmap_verification import ColmapGeometricVerificationResult

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"


class _FakeGlobalPositionerOptions:
    def __init__(self) -> None:
        self.use_gpu = True


class _FakeCeresOptions:
    def __init__(self) -> None:
        self.use_gpu = True


class _FakeBundleAdjustmentOptions:
    def __init__(self) -> None:
        self.backend: object = "unset"
        self.ceres = _FakeCeresOptions()


class _FakeGlobalMapperOptions:
    def __init__(self) -> None:
        self.num_threads = -1
        self.random_seed = -1
        self.global_positioning = _FakeGlobalPositionerOptions()
        self.bundle_adjustment = _FakeBundleAdjustmentOptions()


class _FakeGlobalPipelineOptions:
    def __init__(self) -> None:
        self.min_num_matches = -1
        self.ignore_watermarks = True
        self.num_threads = -1
        self.random_seed = -1
        self.decompose_relative_pose = False
        self.multiple_models = False
        self.min_model_size = -1
        self.mapper = _FakeGlobalMapperOptions()


class _FakeViewGraphCalibrationOptions:
    def __init__(self) -> None:
        self.random_seed = -1


class _FakeDatabase:
    def __init__(self, path: Path) -> None:
        self._path = path

    def read_all_images(self) -> list[SimpleNamespace]:
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                "SELECT image_id, name FROM images ORDER BY image_id"
            ).fetchall()
        finally:
            connection.close()
        return [SimpleNamespace(image_id=int(image_id), name=str(name)) for image_id, name in rows]

    def close(self) -> None:
        return None


class _FakeReconstruction:
    def __init__(
        self,
        *,
        num_registered_images: int = 2,
        num_points3d: int = 5,
        valid: bool = True,
    ) -> None:
        self._num_registered_images = num_registered_images
        self._num_points3d = num_points3d
        self._valid = valid

    def is_valid(self) -> bool:
        return self._valid

    def num_reg_images(self) -> int:
        return self._num_registered_images

    def num_points3D(self) -> int:
        return self._num_points3d


class _FakeBundleAdjustmentBackend:
    CERES = "CERES"
    CASPAR = "CASPAR"


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-v2l12-3-global without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    GlobalPipelineOptions = _FakeGlobalPipelineOptions
    ViewGraphCalibrationOptions = _FakeViewGraphCalibrationOptions
    BundleAdjustmentBackend = _FakeBundleAdjustmentBackend

    def __init__(
        self,
        *,
        models: dict[int, _FakeReconstruction] | None = None,
        calibration_success: bool = True,
        fail_mapping: bool = False,
        mutate_during_mapping: bool = False,
    ) -> None:
        self.models = {0: _FakeReconstruction()} if models is None else models
        self.calibration_success = calibration_success
        self.fail_mapping = fail_mapping
        self.mutate_during_mapping = mutate_during_mapping
        self.random_seed: int | None = None
        self.calibration_calls: list[dict[str, object]] = []
        self.mapping_calls: list[dict[str, object]] = []
        self.Database = SimpleNamespace(open=self._open_database)

    @staticmethod
    def _open_database(path: Path) -> _FakeDatabase:
        return _FakeDatabase(Path(path))

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def calibrate_view_graph(
        self,
        database_path: Path,
        *,
        options: _FakeViewGraphCalibrationOptions,
    ) -> bool:
        path = Path(database_path)
        before = path.read_bytes()
        connection = sqlite3.connect(path)
        try:
            connection.execute("CREATE TABLE wre_private_calibration(marker INTEGER)")
            connection.execute("INSERT INTO wre_private_calibration VALUES (1)")
            connection.commit()
        finally:
            connection.close()
        self.calibration_calls.append(
            {
                "database_path": path,
                "before": before,
                "after": path.read_bytes(),
                "options": options,
            }
        )
        return self.calibration_success

    def global_mapping(
        self,
        database_path: Path,
        image_path: Path,
        output_path: Path,
        *,
        options: _FakeGlobalPipelineOptions,
    ) -> dict[int, _FakeReconstruction]:
        database = Path(database_path)
        if self.fail_mapping:
            raise RuntimeError("synthetic global mapper failure")
        if self.mutate_during_mapping:
            connection = sqlite3.connect(database)
            try:
                connection.execute("CREATE TABLE wre_private_mapping(marker INTEGER)")
                connection.commit()
            finally:
                connection.close()

        staged = {
            path.name: path.read_bytes()
            for path in sorted(Path(image_path).iterdir(), key=lambda item: item.name)
        }
        self.mapping_calls.append(
            {
                "database_path": database,
                "database_bytes": database.read_bytes(),
                "image_path": Path(image_path),
                "output_path": Path(output_path),
                "options": options,
                "staged": staged,
            }
        )
        for index in self.models:
            model_dir = Path(output_path) / str(index)
            model_dir.mkdir()
            (model_dir / "cameras.bin").write_bytes(f"camera-{index}".encode())
            (model_dir / "images.bin").write_bytes(f"images-{index}".encode())
            (model_dir / "points3D.bin").write_bytes(f"points-{index}".encode())
        return self.models


def _write_database(path: Path, image_names: tuple[str, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL)")
        connection.executemany(
            "INSERT INTO images(image_id, name) VALUES(?, ?)",
            [(index + 1, name) for index, name in enumerate(image_names)],
        )
        connection.commit()
    finally:
        connection.close()


def _observation(path: Path, observation_id: str) -> ImageObservation:
    digest = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(observation_id),
        asset=MediaAssetRef(
            uri=path.resolve().as_uri(),
            sha256=digest.sha256,
            byte_length=digest.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("source:test")),
        received_at=datetime(2026, 9, 22, tzinfo=UTC),
    )


def _environment() -> ColmapEnvironmentIdentity:
    return ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build=_FakePycolmap.COLMAP_build,
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )


def _request(
    tmp_path: Path,
    *,
    database_names: tuple[str, ...] = ("000000-a.pgm", "000001-b.pgm"),
) -> tuple[ColmapGlobalReconstructionRequest, Path, tuple[Path, ...]]:
    source_paths = (tmp_path / "source-a.pgm", tmp_path / "source-b.pgm")
    source_paths[0].write_bytes(b"P5\n1 1\n255\n\x40")
    source_paths[1].write_bytes(b"P5\n1 1\n255\n\x80")
    observations = (
        _observation(source_paths[0], "obs:a"),
        _observation(source_paths[1], "obs:b"),
    )

    database_path = tmp_path / "verified.db"
    _write_database(database_path, database_names)
    database_hash = hash_file_content(database_path)
    verification = ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:verification"),
            source_observation_ids=tuple(
                observation.observation_id for observation in observations
            ),
        ),
        environment=_environment(),
        configuration_sha256=Sha256Digest("1" * 64),
        source_matching_database_sha256=Sha256Digest("2" * 64),
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        geometries=(),
    )
    config = ColmapGlobalReconstructionConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:global"),
        producer=ProducerRef(
            implementation=COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION,
            version="4.2.0",
            revision=_FakePycolmap.COLMAP_build,
        ),
        input_observation_ids=tuple(observation.observation_id for observation in observations),
        started_at=datetime(2026, 9, 22, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    inputs = tuple(
        ColmapReconstructionInput(
            observation=observation,
            source_path=source_path,
            image_name=image_name,
        )
        for observation, source_path, image_name in zip(
            observations,
            source_paths,
            ("000000-a.pgm", "000001-b.pgm"),
            strict=True,
        )
    )
    return (
        ColmapGlobalReconstructionRequest(
            run=run,
            verification=verification,
            inputs=inputs,
            output_path=tmp_path / "global-sparse",
            config=config,
        ),
        database_path,
        source_paths,
    )


def _entries_by_id() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    entries = document["entries"]
    assert isinstance(entries, list)
    return {
        cast(str, entry["adapter_id"]): cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
    }


def test_global_config_is_canonical_deterministic_and_cpu_only() -> None:
    config = ColmapGlobalReconstructionConfig()

    assert config.sha256 == ColmapGlobalReconstructionConfig().sha256
    assert config.num_threads == 1
    assert config.random_seed == 0
    assert config.multiple_models is True
    assert config.calibrate_view_graph is True
    assert config.global_positioning_use_gpu is False
    assert config.bundle_adjustment_backend == "CERES"
    assert config.bundle_adjustment_use_gpu is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_threads": 2}, "one mapper thread"),
        ({"random_seed": -1}, "non-negative random seed"),
        ({"multiple_models": False}, "disconnected mapper models"),
        ({"calibrate_view_graph": False}, "view-graph calibration"),
        ({"global_positioning_use_gpu": True}, "CPU global positioning"),
        ({"bundle_adjustment_backend": "CASPAR"}, "CERES"),
        ({"bundle_adjustment_use_gpu": True}, "CPU Ceres"),
    ],
)
def test_global_config_rejects_nondeterministic_or_accelerated_reference_modes(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ColmapGlobalReconstructionConfig(**kwargs)


def test_global_route_uses_private_calibrated_database_and_explicit_cpu_controls(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(tmp_path)
    parent_before = database_path.read_bytes()
    module = _FakePycolmap(mutate_during_mapping=True)

    result = reconstruct_colmap_globally(request, module=module)

    assert database_path.read_bytes() == parent_before
    assert result.model_count == 1
    assert result.source_verification_database_sha256 == (request.verification.database_sha256)
    assert result.provenance.source_observation_ids == request.run.input_observation_ids
    assert module.random_seed == 0
    assert len(module.calibration_calls) == 1
    assert len(module.mapping_calls) == 1

    calibration = module.calibration_calls[0]
    mapping = module.mapping_calls[0]
    private_database = cast(Path, calibration["database_path"])
    assert private_database != database_path
    assert calibration["before"] == parent_before
    assert calibration["after"] != parent_before
    assert mapping["database_path"] == private_database
    assert not private_database.exists()

    calibration_options = cast(
        _FakeViewGraphCalibrationOptions,
        calibration["options"],
    )
    options = cast(_FakeGlobalPipelineOptions, mapping["options"])
    assert calibration_options.random_seed == 0
    assert options.num_threads == 1
    assert options.random_seed == 0
    assert options.multiple_models is True
    assert options.mapper.num_threads == 1
    assert options.mapper.random_seed == 0
    assert options.mapper.global_positioning.use_gpu is False
    assert options.mapper.bundle_adjustment.backend == (_FakeBundleAdjustmentBackend.CERES)
    assert options.mapper.bundle_adjustment.ceres.use_gpu is False
    assert mapping["staged"] == {
        "000000-a.pgm": b"P5\n1 1\n255\n\x40",
        "000001-b.pgm": b"P5\n1 1\n255\n\x80",
    }


def test_global_route_rejects_parent_database_tampering_before_private_copy(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(tmp_path)
    with database_path.open("ab") as stream:
        stream.write(b"tamper")
    module = _FakePycolmap()

    with pytest.raises(ValueError, match="recorded artifact identity"):
        reconstruct_colmap_globally(request, module=module)

    assert module.calibration_calls == []
    assert module.mapping_calls == []
    assert not request.output_path.exists()


def test_global_route_rejects_database_membership_mismatch_and_cleans_output(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(
        tmp_path,
        database_names=("000000-a.pgm", "wrong.pgm"),
    )
    before = database_path.read_bytes()
    module = _FakePycolmap()

    with pytest.raises(ValueError, match="exactly match COLMAP image membership"):
        reconstruct_colmap_globally(request, module=module)

    assert database_path.read_bytes() == before
    assert module.calibration_calls == []
    assert module.mapping_calls == []
    assert not request.output_path.exists()


def test_global_route_rejects_failed_private_view_graph_calibration(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(tmp_path)
    before = database_path.read_bytes()
    module = _FakePycolmap(calibration_success=False)

    with pytest.raises(
        ColmapGlobalReconstructionError,
        match="calibration did not succeed",
    ):
        reconstruct_colmap_globally(request, module=module)

    assert database_path.read_bytes() == before
    assert module.mapping_calls == []
    assert not request.output_path.exists()


def test_global_solver_failure_preserves_parent_and_cleans_only_owned_output(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(tmp_path)
    before = database_path.read_bytes()
    module = _FakePycolmap(fail_mapping=True)

    with pytest.raises(RuntimeError, match="synthetic global mapper failure"):
        reconstruct_colmap_globally(request, module=module)

    assert database_path.read_bytes() == before
    assert not request.output_path.exists()


def test_global_existing_output_is_never_reused_or_deleted(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path)
    request.output_path.mkdir()
    sentinel = request.output_path / "sentinel.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")

    with pytest.raises(ValueError, match="must not already exist"):
        reconstruct_colmap_globally(request, module=_FakePycolmap())

    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"


def test_global_adapter_registry_contract_is_distinct_and_canonical() -> None:
    assert COLMAP_GLOBAL_CANONICAL_CAPABILITY.capability is (
        COLMAP_GLOBAL_CANONICAL_CAPABILITY_NAME
    )
    assert COLMAP_GLOBAL_CANONICAL_CAPABILITY.input_kinds == frozenset(
        {
            ArtifactKind("image.observation"),
            ArtifactKind("evidence.geometric_verification"),
        }
    )
    assert COLMAP_GLOBAL_CANONICAL_CAPABILITY.output_kinds == frozenset(
        {
            ArtifactKind("geometry.camera_solution"),
            ArtifactKind("geometry.point_map"),
            ArtifactKind("geometry.solution"),
        }
    )

    entry = _entries_by_id()[COLMAP_GLOBAL_CANONICAL_ADAPTER_ID]
    assert entry["capability"] == {
        "name": COLMAP_GLOBAL_CANONICAL_CAPABILITY_NAME.value,
        "input_kinds": ["evidence.geometric_verification", "image.observation"],
        "output_kinds": [
            "geometry.camera_solution",
            "geometry.point_map",
            "geometry.solution",
        ],
    }
    assert entry["producer"] == {
        "implementation": COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION,
        "version": COLMAP_GLOBAL_CANONICAL_PRODUCER_VERSION,
        "revision": None,
    }
    assert entry["dependency_refs"] == [COLMAP_GLOBAL_CANONICAL_DEPENDENCY_REF]
    assert entry["model"] is COLMAP_GLOBAL_CANONICAL_MODEL
    assert entry["checkpoint"] is COLMAP_GLOBAL_CANONICAL_CHECKPOINT
    assert (
        entry["artifact_key_hardware_policy"]
        == COLMAP_GLOBAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY
    )
    assert entry["shipping_status"] == COLMAP_GLOBAL_CANONICAL_SHIPPING_STATUS
    assert entry["reproducibility_notes"] == COLMAP_GLOBAL_CANONICAL_REPRODUCIBILITY_NOTES


def test_global_module_has_no_routing_quality_scale_or_later_product_surface() -> None:
    forbidden_names = {
        "SparseReconstructionEstimate",
        "convert_sparse_reconstruction_estimate",
        "SurfaceModel",
        "SpatialFragment",
        "CameraId",
        "QualityDecision",
        "Router",
        "MasterScene",
        "RuntimeScene",
        "SQLiteLocalStore",
    }

    assert forbidden_names.isdisjoint(vars(global_module))


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="real PyCOLMAP integration runs only in the dedicated COLMAP lane",
)
def test_real_pycolmap_global_mapping_is_cpu_bounded_audited_and_canonical(
    tmp_path: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    environment = inspect_colmap_environment(pycolmap)
    database_path = tmp_path / "synthetic.db"

    with pycolmap.Database.open(database_path) as database:
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 2
        options.num_cameras_per_rig = 1
        options.num_frames_per_rig = 7
        options.num_points3D = 50
        options.camera_width = 8
        options.camera_height = 8
        options.camera_params = [10.0, 4.0, 4.0, 0.05]
        options.camera_has_prior_focal_length = False
        ground_truth = pycolmap.synthesize_dataset(options, database)

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute("SELECT image_id FROM images ORDER BY image_id").fetchall()
        image_names = tuple(f"{index:06d}-synthetic.pgm" for index in range(len(rows)))
        for name, (image_id,) in zip(image_names, rows, strict=True):
            connection.execute(
                "UPDATE images SET name = ? WHERE image_id = ?",
                (name, int(image_id)),
            )
        connection.commit()
    finally:
        connection.close()

    source_paths: list[Path] = []
    observations: list[ImageObservation] = []
    for index, name in enumerate(image_names):
        source_path = tmp_path / name
        pixel = bytes([(index * 17) % 255])
        source_path.write_bytes(b"P5\n8 8\n255\n" + pixel * 64)
        source_paths.append(source_path)
        observations.append(_observation(source_path, f"obs:{index:03d}"))

    observation_ids = tuple(item.observation_id for item in observations)
    database_hash = hash_file_content(database_path)
    verification = ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:real-verification"),
            source_observation_ids=observation_ids,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("5" * 64),
        source_matching_database_sha256=Sha256Digest("6" * 64),
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        geometries=(),
    )
    config = ColmapGlobalReconstructionConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:real-global"),
        producer=ProducerRef(
            implementation=COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION,
            version="4.2.0",
            revision=environment.colmap_build,
        ),
        input_observation_ids=observation_ids,
        started_at=datetime(2026, 9, 22, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    inputs = tuple(
        ColmapReconstructionInput(
            observation=observation,
            source_path=source_path,
            image_name=image_name,
        )
        for observation, source_path, image_name in zip(
            observations,
            source_paths,
            image_names,
            strict=True,
        )
    )
    request = ColmapGlobalReconstructionRequest(
        run=run,
        verification=verification,
        inputs=inputs,
        output_path=tmp_path / "global-sparse",
        config=config,
    )
    parent_before = database_path.read_bytes()

    result = reconstruct_colmap_globally(request, module=pycolmap)

    assert database_path.read_bytes() == parent_before
    assert result.model_count >= 1
    assert sum(model.num_registered_images for model in result.models) >= (
        ground_truth.num_reg_images()
    )
    assert all(model.files for model in result.models)

    features = ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:real-features"),
            source_observation_ids=observation_ids,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("7" * 64),
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=image_name,
                keypoint_rows=1,
                keypoint_cols=2,
                descriptor_rows=1,
                descriptor_cols=128,
            )
            for observation_id, image_name in zip(
                observation_ids,
                image_names,
                strict=True,
            )
        ),
    )
    canonical = canonicalize_colmap_sparse_model(
        output_path=result.output_path,
        model_artifact=result.models[0],
        features=features,
        expected_environment=result.environment,
        module=pycolmap,
    )

    assert canonical.geometry_solution.scale_status.value == "unresolved"
    assert canonical.camera_solutions
    assert canonical.point_map.source_observation_ids
    assert canonical.geometry_solution.depth_field_ids == ()
