from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

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
from wre.reconstruction.colmap_environment import (
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionConfig,
    ColmapIncrementalReconstructionError,
    ColmapIncrementalReconstructionRequest,
    ColmapReconstructionInput,
    reconstruct_colmap_incrementally,
)
from wre.reconstruction.colmap_verification import ColmapGeometricVerificationResult


class _FakeIncrementalPipelineOptions:
    def __init__(self) -> None:
        self.min_num_matches = -1
        self.ignore_watermarks = True
        self.multiple_models = False
        self.max_num_models = -1
        self.max_model_overlap = -1
        self.min_model_size = -1
        self.init_num_trials = -1
        self.structure_less_registration_fallback = False
        self.structure_less_registration_only = True
        self.extract_colors = True
        self.num_threads = -1
        self.random_seed = -1
        self.ba_refine_focal_length = False
        self.ba_refine_principal_point = True
        self.ba_refine_extra_params = False
        self.ba_refine_sensor_from_rig = False
        self.ba_use_gpu = True
        self.use_prior_position = True
        self.load_all_images = True
        self.max_runtime_seconds = 0


class _FakeDatabase:
    def __init__(self, path: Path) -> None:
        self._path = path
        self.closed = False

    def read_all_images(self) -> list[SimpleNamespace]:
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                "SELECT image_id, name FROM images ORDER BY image_id"
            ).fetchall()
        finally:
            connection.close()
        return [SimpleNamespace(image_id=image_id, name=name) for image_id, name in rows]

    def close(self) -> None:
        self.closed = True


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


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-l35 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    IncrementalPipelineOptions = _FakeIncrementalPipelineOptions

    def __init__(
        self,
        *,
        models: dict[int, _FakeReconstruction] | None = None,
        fail: bool = False,
        mutate_database: bool = False,
        extra_output: bool = False,
    ) -> None:
        self.models = {0: _FakeReconstruction()} if models is None else models
        self.fail = fail
        self.mutate_database = mutate_database
        self.extra_output = extra_output
        self.random_seed: int | None = None
        self.calls: list[dict[str, object]] = []
        self.Database = SimpleNamespace(open=self._open_database)

    @staticmethod
    def _open_database(path: Path) -> _FakeDatabase:
        return _FakeDatabase(Path(path))

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def incremental_mapping(
        self,
        database_path: Path,
        image_path: Path,
        output_path: Path,
        *,
        options: object,
    ) -> dict[int, _FakeReconstruction]:
        self.calls.append(
            {
                "database_path": Path(database_path),
                "image_path": Path(image_path),
                "output_path": Path(output_path),
                "options": options,
                "staged": {
                    path.name: path.read_bytes()
                    for path in sorted(Path(image_path).iterdir(), key=lambda item: item.name)
                },
            }
        )
        if self.fail:
            raise RuntimeError("synthetic mapper failure")
        if self.mutate_database:
            with Path(database_path).open("ab") as stream:
                stream.write(b"mutation")
        for index in self.models:
            model_dir = Path(output_path) / str(index)
            model_dir.mkdir()
            (model_dir / "cameras.bin").write_bytes(f"camera-{index}".encode())
            (model_dir / "images.bin").write_bytes(f"images-{index}".encode())
            (model_dir / "points3D.bin").write_bytes(f"points-{index}".encode())
        if self.extra_output:
            (Path(output_path) / "unexpected").mkdir()
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
            mime_type="image/png",
        ),
        source=SourceRef(source_id=SourceId("source:test")),
        received_at=datetime(2026, 9, 15, tzinfo=UTC),
    )


def _request(
    tmp_path: Path,
    *,
    database_names: tuple[str, ...] = ("000000-a.png", "000001-b.png"),
) -> tuple[
    ColmapIncrementalReconstructionRequest,
    Path,
    tuple[Path, ...],
]:
    source_paths = (tmp_path / "source-a.png", tmp_path / "source-b.png")
    source_paths[0].write_bytes(b"image-a")
    source_paths[1].write_bytes(b"image-b")
    observations = (
        _observation(source_paths[0], "obs:a"),
        _observation(source_paths[1], "obs:b"),
    )
    database_path = tmp_path / "verified.db"
    _write_database(database_path, database_names)
    database_hash = hash_file_content(database_path)
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:l34"),
        source_observation_ids=tuple(item.observation_id for item in observations),
    )
    environment = ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build=_FakePycolmap.COLMAP_build,
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )
    verification = ColmapGeometricVerificationResult(
        provenance=provenance,
        environment=environment,
        configuration_sha256=Sha256Digest("1" * 64),
        source_matching_database_sha256=Sha256Digest("2" * 64),
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        geometries=(),
    )
    config = ColmapIncrementalReconstructionConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l35"),
        producer=ProducerRef(
            implementation="pycolmap.incremental_mapping",
            version="4.2.0",
            revision=_FakePycolmap.COLMAP_build,
        ),
        input_observation_ids=tuple(item.observation_id for item in observations),
        started_at=datetime(2026, 9, 15, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    inputs = tuple(
        ColmapReconstructionInput(
            observation=observation,
            source_path=source_path,
            image_name=image_name,
        )
        for observation, source_path, image_name in zip(
            observations, source_paths, ("000000-a.png", "000001-b.png"), strict=True
        )
    )
    return (
        ColmapIncrementalReconstructionRequest(
            run=run,
            verification=verification,
            inputs=inputs,
            output_path=tmp_path / "sparse",
            config=config,
        ),
        database_path,
        source_paths,
    )


def test_config_is_canonical_and_deterministic() -> None:
    left = ColmapIncrementalReconstructionConfig()
    right = ColmapIncrementalReconstructionConfig()
    assert left.sha256 == right.sha256
    assert left.canonical_document()["multiple_models"] is True
    assert left.canonical_document()["extract_colors"] is False
    assert left.canonical_document()["num_threads"] == 1
    assert left.canonical_document()["random_seed"] == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_threads": 2}, "one reconstruction thread"),
        ({"ba_use_gpu": True}, "CPU bundle adjustment"),
        ({"extract_colors": True}, "point colors"),
        ({"use_prior_position": True}, "position priors"),
        ({"multiple_models": False}, "disconnected COLMAP sub-models"),
        ({"structure_less_registration_only": True}, "structure-based registration"),
    ],
)
def test_config_rejects_out_of_scope_or_nondeterministic_modes(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ColmapIncrementalReconstructionConfig(**kwargs)


def test_request_rejects_unsafe_image_name(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"image")
    observation = _observation(source, "obs:a")
    with pytest.raises(ValueError, match="safe relative filename"):
        ColmapReconstructionInput(
            observation=observation,
            source_path=source,
            image_name="../escape.png",
        )


def test_incremental_reconstruction_preserves_parent_and_audits_output(tmp_path: Path) -> None:
    request, database_path, _ = _request(tmp_path)
    before = database_path.read_bytes()
    module = _FakePycolmap()

    result = reconstruct_colmap_incrementally(request, module=module)

    assert database_path.read_bytes() == before
    assert result.model_count == 1
    assert result.has_reconstruction is True
    assert result.source_verification_database_sha256 == request.verification.database_sha256
    assert result.provenance.source_observation_ids == request.run.input_observation_ids
    assert result.models[0].model_index == 0
    assert result.models[0].num_registered_images == 2
    assert result.models[0].num_points3d == 5
    assert tuple(item.relative_path for item in result.models[0].files) == (
        "cameras.bin",
        "images.bin",
        "points3D.bin",
    )
    assert all(item.byte_length > 0 for item in result.models[0].files)
    assert module.random_seed == 0
    assert len(module.calls) == 1
    call = module.calls[0]
    options = call["options"]
    assert isinstance(options, _FakeIncrementalPipelineOptions)
    assert options.min_num_matches == 15
    assert options.multiple_models is True
    assert options.extract_colors is False
    assert options.num_threads == 1
    assert options.random_seed == 0
    assert options.ba_use_gpu is False
    assert options.use_prior_position is False
    assert call["staged"] == {
        "000000-a.png": b"image-a",
        "000001-b.png": b"image-b",
    }


def test_zero_models_is_explicit_unresolved_outcome(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path)
    result = reconstruct_colmap_incrementally(request, module=_FakePycolmap(models={}))
    assert result.models == ()
    assert result.model_count == 0
    assert result.has_reconstruction is False
    assert request.output_path.is_dir()
    assert list(request.output_path.iterdir()) == []


def test_database_image_membership_mismatch_fails_and_cleans_owned_output(
    tmp_path: Path,
) -> None:
    request, database_path, _ = _request(
        tmp_path,
        database_names=("000000-a.png", "wrong-name.png"),
    )
    before = database_path.read_bytes()
    with pytest.raises(ValueError, match="exactly match COLMAP image membership"):
        reconstruct_colmap_incrementally(request, module=_FakePycolmap())
    assert database_path.read_bytes() == before
    assert not request.output_path.exists()


def test_source_byte_mismatch_fails_before_solver_and_cleans_output(tmp_path: Path) -> None:
    request, database_path, source_paths = _request(tmp_path)
    before = database_path.read_bytes()
    source_paths[0].write_bytes(b"changed-after-ingestion")
    module = _FakePycolmap()
    with pytest.raises(ValueError, match="do not match the persisted observation asset"):
        reconstruct_colmap_incrementally(request, module=module)
    assert module.calls == []
    assert database_path.read_bytes() == before
    assert not request.output_path.exists()


def test_existing_output_is_never_reused_or_deleted(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path)
    request.output_path.mkdir()
    sentinel = request.output_path / "sentinel.txt"
    sentinel.write_text("owned by another invocation", encoding="utf-8")
    with pytest.raises(ValueError, match="must not already exist"):
        reconstruct_colmap_incrementally(request, module=_FakePycolmap())
    assert sentinel.read_text(encoding="utf-8") == "owned by another invocation"


def test_solver_failure_cleans_only_owned_output_and_preserves_parent(tmp_path: Path) -> None:
    request, database_path, _ = _request(tmp_path)
    before = database_path.read_bytes()
    with pytest.raises(RuntimeError, match="synthetic mapper failure"):
        reconstruct_colmap_incrementally(request, module=_FakePycolmap(fail=True))
    assert database_path.read_bytes() == before
    assert not request.output_path.exists()


def test_database_mutation_is_detected_and_output_is_removed(tmp_path: Path) -> None:
    request, database_path, _ = _request(tmp_path)
    before = database_path.read_bytes()
    with pytest.raises(ColmapIncrementalReconstructionError, match="mutated the immutable"):
        reconstruct_colmap_incrementally(
            request,
            module=_FakePycolmap(mutate_database=True),
        )
    assert database_path.read_bytes() != before
    assert not request.output_path.exists()


def test_unexpected_solver_output_membership_is_rejected(tmp_path: Path) -> None:
    request, _, _ = _request(tmp_path)
    with pytest.raises(
        ColmapIncrementalReconstructionError,
        match="output directory membership",
    ):
        reconstruct_colmap_incrementally(request, module=_FakePycolmap(extra_output=True))
    assert not request.output_path.exists()


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="real PyCOLMAP integration runs only in the dedicated COLMAP lane",
)
def test_real_pycolmap_incremental_mapping_preserves_parent_database(tmp_path: Path) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    environment = inspect_colmap_environment(pycolmap)
    database_path = tmp_path / "synthetic.db"

    with pycolmap.Database.open(database_path) as database:
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 2
        options.num_cameras_per_rig = 1
        options.num_frames_per_rig = 7
        options.num_points3D = 50
        options.camera_has_prior_focal_length = False
        ground_truth = pycolmap.synthesize_dataset(options, database)

    with pycolmap.Database.open(database_path) as database:
        database_images = sorted(database.read_all_images(), key=lambda image: image.name)

    observations: list[ImageObservation] = []
    reconstruction_inputs: list[ColmapReconstructionInput] = []
    for index, image in enumerate(database_images):
        source_path = tmp_path / f"source-{index:03d}.bin"
        source_path.write_bytes(f"synthetic-source-{index}".encode())
        observation = _observation(source_path, f"obs:{index:03d}")
        observations.append(observation)
        reconstruction_inputs.append(
            ColmapReconstructionInput(
                observation=observation,
                source_path=source_path,
                image_name=str(image.name),
            )
        )

    canonical_inputs = tuple(
        sorted(reconstruction_inputs, key=lambda item: item.observation.observation_id.value)
    )
    observation_ids = tuple(item.observation.observation_id for item in canonical_inputs)
    database_hash = hash_file_content(database_path)
    verification = ColmapGeometricVerificationResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:real-l34-parent"),
            source_observation_ids=observation_ids,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("3" * 64),
        source_matching_database_sha256=Sha256Digest("4" * 64),
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        geometries=(),
    )
    config = ColmapIncrementalReconstructionConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:real-l35"),
        producer=ProducerRef(
            implementation="pycolmap.incremental_mapping",
            version="4.2.0",
            revision=environment.colmap_build,
        ),
        input_observation_ids=observation_ids,
        started_at=datetime(2026, 9, 15, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    request = ColmapIncrementalReconstructionRequest(
        run=run,
        verification=verification,
        inputs=canonical_inputs,
        output_path=tmp_path / "sparse",
        config=config,
    )
    before = database_path.read_bytes()

    result = reconstruct_colmap_incrementally(request, module=pycolmap)

    assert database_path.read_bytes() == before
    assert result.model_count == 1
    assert result.models[0].num_registered_images == ground_truth.num_reg_images()
    assert result.models[0].num_points3d > 0
    assert result.models[0].files
    assert (result.output_path / "0").is_dir()
