from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    SourceId,
    SourceRef,
)
from wre.domain.runs import ProducerRef, ReconstructionRun, ReconstructionRunId
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionError,
    ColmapFeatureExtractionRequest,
    ColmapFeatureInput,
    extract_colmap_features,
)


class _FakeFeatureExtractionOptions:
    def __init__(self) -> None:
        self.type: object | None = None
        self.max_image_size = -1
        self.num_threads = -1
        self.sift = SimpleNamespace(
            max_num_features=0,
            first_octave=0,
            num_octaves=0,
            octave_resolution=0,
            peak_threshold=0.0,
            edge_threshold=0.0,
            estimate_affine_shape=False,
            max_num_orientations=0,
            upright=False,
            darkness_adaptivity=False,
            domain_size_pooling=False,
            dsp_min_scale=0.0,
            dsp_max_scale=0.0,
            dsp_num_scales=0,
            normalization=None,
        )


class _FakeImageReaderOptions:
    def __init__(self) -> None:
        self.camera_model = ""
        self.camera_params = ""
        self.default_focal_length_factor = 0.0


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-l32 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    FeatureExtractionOptions = _FakeFeatureExtractionOptions
    ImageReaderOptions = _FakeImageReaderOptions
    FeatureExtractorType = SimpleNamespace(SIFT="SIFT")
    Normalization = SimpleNamespace(L1_ROOT="L1_ROOT")
    CameraMode = SimpleNamespace(PER_IMAGE="PER_IMAGE")
    Device = SimpleNamespace(cpu="cpu")

    def __init__(self, *, inject_match: bool = False) -> None:
        self.inject_match = inject_match
        self.random_seed: int | None = None
        self.calls: list[dict[str, object]] = []

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def extract_features(
        self,
        database_path: Path,
        image_path: Path,
        *,
        image_names: list[str],
        camera_mode: object,
        reader_options: object,
        extraction_options: object,
        device: object,
    ) -> None:
        self.calls.append(
            {
                "database_path": database_path,
                "image_path": image_path,
                "image_names": tuple(image_names),
                "camera_mode": camera_mode,
                "reader_options": reader_options,
                "extraction_options": extraction_options,
                "device": device,
            }
        )
        connection = sqlite3.connect(database_path)
        try:
            connection.executescript(
                """
                CREATE TABLE images (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    name TEXT NOT NULL UNIQUE
                );
                CREATE TABLE keypoints (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                );
                CREATE TABLE descriptors (
                    image_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                );
                CREATE TABLE matches (
                    pair_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                );
                CREATE TABLE two_view_geometries (
                    pair_id INTEGER PRIMARY KEY NOT NULL,
                    rows INTEGER NOT NULL,
                    cols INTEGER NOT NULL,
                    data BLOB
                );
                """
            )
            for image_id, image_name in enumerate(image_names, start=1):
                connection.execute(
                    "INSERT INTO images(image_id, name) VALUES(?, ?)",
                    (image_id, image_name),
                )
                connection.execute(
                    "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, 12, 4, X'00')",
                    (image_id,),
                )
                connection.execute(
                    "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, 12, 128, X'00')",
                    (image_id,),
                )
            if self.inject_match:
                connection.execute(
                    "INSERT INTO matches(pair_id, rows, cols, data) VALUES(1, 1, 2, X'00')"
                )
            connection.commit()
        finally:
            connection.close()


def _write_pgm(path: Path, *, shift: int = 0) -> None:
    width = 256
    height = 256
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            checker = ((x // 16) + (y // 16) + shift) % 2
            gradient = (x * 5 + y * 3 + shift * 17) % 64
            pixels.append(min(255, checker * 176 + gradient))
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode() + bytes(pixels))


def _observation(path: Path, value: str) -> ImageObservation:
    content = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(value),
        asset=MediaAssetRef(
            uri=path.as_uri(),
            sha256=content.sha256,
            byte_length=content.byte_length,
            mime_type="image/x-portable-graymap",
        ),
        source=SourceRef(source_id=SourceId("fixture"), locator=path.name),
        received_at=datetime(2026, 9, 15, tzinfo=UTC),
    )


def _request(
    tmp_path: Path,
    observations: tuple[tuple[ImageObservation, Path], ...],
    *,
    config: ColmapFeatureExtractionConfig | None = None,
    revision: str | None = _FakePycolmap.COLMAP_build,
) -> ColmapFeatureExtractionRequest:
    actual_config = config or ColmapFeatureExtractionConfig()
    canonical_ids = tuple(sorted((item[0].observation_id for item in observations), key=str))
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.2-test"),
        producer=ProducerRef(
            implementation="pycolmap.extract_features",
            version="4.2.0",
            revision=revision,
        ),
        input_observation_ids=canonical_ids,
        started_at=datetime(2026, 9, 15, 1, 0, tzinfo=UTC),
        configuration_sha256=actual_config.sha256,
    )
    inputs = tuple(
        ColmapFeatureInput(observation=observation, source_path=path)
        for observation, path in observations
    )
    return ColmapFeatureExtractionRequest(
        run=run,
        inputs=inputs,
        database_path=tmp_path / "features.db",
        config=actual_config,
    )


def test_feature_config_has_stable_canonical_digest() -> None:
    first = ColmapFeatureExtractionConfig()
    second = ColmapFeatureExtractionConfig()

    assert first.canonical_document() == second.canonical_document()
    assert first.sha256 == second.sha256
    assert first.canonical_document()["device"] == "cpu"
    assert first.canonical_document()["camera_mode"] == "PER_IMAGE"
    assert first.canonical_document()["num_threads"] == 1


def test_request_canonicalizes_inputs_and_requires_matching_run(tmp_path: Path) -> None:
    path_b = tmp_path / "b.pgm"
    path_a = tmp_path / "a.pgm"
    _write_pgm(path_a)
    _write_pgm(path_b, shift=1)
    observation_b = _observation(path_b, "obs:b")
    observation_a = _observation(path_a, "obs:a")

    request = _request(
        tmp_path,
        ((observation_b, path_b), (observation_a, path_a)),
    )

    assert tuple(item.observation.observation_id.value for item in request.inputs) == (
        "obs:a",
        "obs:b",
    )


def test_fake_feature_extraction_is_auditable_and_feature_only(tmp_path: Path) -> None:
    path_b = tmp_path / "b.pgm"
    path_a = tmp_path / "a.pgm"
    _write_pgm(path_a)
    _write_pgm(path_b, shift=1)
    observation_b = _observation(path_b, "obs:b")
    observation_a = _observation(path_a, "obs:a")
    request = _request(tmp_path, ((observation_b, path_b), (observation_a, path_a)))
    fake = _FakePycolmap()

    result = extract_colmap_features(request, module=fake)

    assert result.environment.pycolmap_version == "4.2.0"
    assert result.configuration_sha256 == request.config.sha256
    assert result.database_path == request.database_path.resolve()
    assert result.database_path.is_file()
    assert result.database_byte_length > 0
    assert result.database_sha256 == hash_file_content(result.database_path).sha256
    assert result.provenance.producing_run_id == request.run.run_id
    assert tuple(item.observation_id.value for item in result.images) == ("obs:a", "obs:b")
    assert tuple(item.image_name for item in result.images) == (
        f"000000-{observation_a.asset.sha256.value[:16]}.pgm",
        f"000001-{observation_b.asset.sha256.value[:16]}.pgm",
    )
    assert all(item.keypoint_rows == 12 for item in result.images)
    assert all(item.keypoint_cols == 4 for item in result.images)
    assert all(item.descriptor_rows == 12 for item in result.images)
    assert all(item.descriptor_cols == 128 for item in result.images)

    assert fake.random_seed == 0
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["camera_mode"] == "PER_IMAGE"
    assert call["device"] == "cpu"
    options = call["extraction_options"]
    assert isinstance(options, _FakeFeatureExtractionOptions)
    assert options.type == "SIFT"
    assert options.num_threads == 1
    assert options.sift.max_num_features == 8192
    reader = call["reader_options"]
    assert isinstance(reader, _FakeImageReaderOptions)
    assert reader.camera_model == "SIMPLE_RADIAL"

    connection = sqlite3.connect(result.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 0
    finally:
        connection.close()


def test_changed_source_bytes_fail_before_solver_and_leave_no_database(tmp_path: Path) -> None:
    source = tmp_path / "image.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:image")
    request = _request(tmp_path, ((observation, source),))
    source.write_bytes(source.read_bytes() + b"changed")
    fake = _FakePycolmap()

    with pytest.raises(ValueError, match="do not match the persisted observation asset"):
        extract_colmap_features(request, module=fake)

    assert fake.calls == []
    assert not request.database_path.exists()


def test_existing_database_is_never_reused(tmp_path: Path) -> None:
    source = tmp_path / "image.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:image")
    request = _request(tmp_path, ((observation, source),))
    request.database_path.write_bytes(b"existing")

    with pytest.raises(ValueError, match="must not already exist"):
        extract_colmap_features(request, module=_FakePycolmap())

    assert request.database_path.read_bytes() == b"existing"


def test_run_configuration_must_match_request(tmp_path: Path) -> None:
    source = tmp_path / "image.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:image")
    config = ColmapFeatureExtractionConfig(max_num_features=4096)
    wrong_config = ColmapFeatureExtractionConfig(max_num_features=2048)
    base = _request(tmp_path, ((observation, source),), config=config)
    wrong_run = ReconstructionRun(
        run_id=base.run.run_id,
        producer=base.run.producer,
        input_observation_ids=base.run.input_observation_ids,
        started_at=base.run.started_at,
        configuration_sha256=wrong_config.sha256,
    )

    with pytest.raises(ValueError, match="configuration SHA-256"):
        ColmapFeatureExtractionRequest(
            run=wrong_run,
            inputs=base.inputs,
            database_path=base.database_path,
            config=config,
        )


def test_run_revision_must_match_real_environment_when_supplied(tmp_path: Path) -> None:
    source = tmp_path / "image.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:image")
    request = _request(tmp_path, ((observation, source),), revision="different build")

    with pytest.raises(ValueError, match="producer revision must match COLMAP_build"):
        extract_colmap_features(request, module=_FakePycolmap())

    assert not request.database_path.exists()


def test_match_contamination_is_rejected_and_partial_database_removed(tmp_path: Path) -> None:
    source = tmp_path / "image.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:image")
    request = _request(tmp_path, ((observation, source),))

    with pytest.raises(ColmapFeatureExtractionError, match="must not populate matches"):
        extract_colmap_features(request, module=_FakePycolmap(inject_match=True))

    assert not request.database_path.exists()


def test_real_pycolmap_feature_extraction_when_integration_lane_enabled(tmp_path: Path) -> None:
    if os.environ.get("WRE_COLMAP_INTEGRATION") != "1":
        pytest.skip("real PyCOLMAP extraction is exercised only in the COLMAP integration lane")

    pycolmap = __import__("pycolmap")

    source = tmp_path / "checkerboard.pgm"
    _write_pgm(source)
    observation = _observation(source, "obs:real")
    config = ColmapFeatureExtractionConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.2-real"),
        producer=ProducerRef(
            implementation="pycolmap.extract_features",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=(observation.observation_id,),
        started_at=datetime(2026, 9, 15, 1, 0, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    request = ColmapFeatureExtractionRequest(
        run=run,
        inputs=(ColmapFeatureInput(observation=observation, source_path=source),),
        database_path=tmp_path / "real-features.db",
        config=config,
    )

    result = extract_colmap_features(request)

    assert result.environment.pycolmap_version == "4.2.0"
    assert len(result.images) == 1
    summary = result.images[0]
    assert summary.observation_id == observation.observation_id
    assert summary.keypoint_rows == summary.descriptor_rows
    assert summary.keypoint_rows > 0
    assert summary.keypoint_cols in (4, 6)
    assert summary.descriptor_cols == 128
    assert result.database_sha256 == hash_file_content(result.database_path).sha256

    connection = sqlite3.connect(result.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 0
    finally:
        connection.close()
