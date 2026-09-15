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
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureExtractionResult,
    ColmapFeatureInput,
    ColmapImageFeatureSummary,
    extract_colmap_features,
)
from wre.reconstruction.colmap_matching import (
    ColmapPairMatchingConfig,
    ColmapPairMatchingError,
    ColmapPairMatchingRequest,
    match_colmap_pairs,
)


class _FakeSiftMatchingOptions:
    def __init__(self) -> None:
        self.max_ratio = 0.0
        self.max_distance = 0.0
        self.cross_check = False
        self.cpu_brute_force_matcher = False


class _FakeFeatureMatchingOptions:
    def __init__(self) -> None:
        self.type: object | None = None
        self.num_threads = -1
        self.use_gpu = True
        self.max_num_matches = 0
        self.guided_matching = True
        self.skip_geometric_verification = False
        self.rig_verification = True
        self.sift = _FakeSiftMatchingOptions()


class _FakeExhaustivePairingOptions:
    def __init__(self) -> None:
        self.block_size = 0


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-l33 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    FeatureMatchingOptions = _FakeFeatureMatchingOptions
    ExhaustivePairingOptions = _FakeExhaustivePairingOptions
    FeatureMatcherType = SimpleNamespace(SIFT_BRUTEFORCE="SIFT_BRUTEFORCE")
    Device = SimpleNamespace(cpu="cpu")

    def __init__(self, *, inject_geometry: bool = False) -> None:
        self.inject_geometry = inject_geometry
        self.random_seed: int | None = None
        self.calls: list[dict[str, object]] = []

    @staticmethod
    def _pair_id(image_id1: int, image_id2: int) -> int:
        low, high = sorted((image_id1, image_id2))
        return low * 1_000_000 + high

    @staticmethod
    def pair_id_to_image_pair(pair_id: int) -> tuple[int, int]:
        return pair_id // 1_000_000, pair_id % 1_000_000

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def match_exhaustive(
        self,
        database_path: Path,
        *,
        matching_options: object,
        pairing_options: object,
        device: object,
    ) -> None:
        self.calls.append(
            {
                "database_path": database_path,
                "matching_options": matching_options,
                "pairing_options": pairing_options,
                "device": device,
            }
        )
        connection = sqlite3.connect(database_path)
        try:
            image_ids = [
                int(row[0])
                for row in connection.execute("SELECT image_id FROM images ORDER BY image_id")
            ]
            for offset, image_id1 in enumerate(image_ids):
                for image_id2 in image_ids[offset + 1 :]:
                    pair_id = self._pair_id(image_id1, image_id2)
                    connection.execute(
                        "INSERT INTO matches(pair_id, rows, cols, data) "
                        "VALUES(?, 7, 2, X'00000000')",
                        (pair_id,),
                    )
                    connection.execute(
                        "INSERT INTO two_view_geometries("
                        "pair_id, rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2"
                        ") VALUES(?, 0, 2, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL)",
                        (pair_id,),
                    )
            if self.inject_geometry:
                connection.execute(
                    "UPDATE two_view_geometries SET rows = 1, data = X'00000000', "
                    "config = 2, F = X'00' WHERE pair_id = ?",
                    (self._pair_id(image_ids[0], image_ids[1]),),
                )
            connection.commit()
        finally:
            connection.close()


def _write_pgm(path: Path, *, shift: int = 0) -> None:
    width = 192
    height = 192
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            checker = ((x // 12) + (y // 12) + shift) % 2
            gradient = (x * 7 + y * 5 + shift * 11) % 64
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


def _create_feature_database(path: Path, image_names: tuple[str, ...]) -> None:
    connection = sqlite3.connect(path)
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
                data BLOB,
                config INTEGER NOT NULL,
                F BLOB,
                E BLOB,
                H BLOB,
                qvec BLOB,
                tvec BLOB,
                camera1 BLOB,
                camera2 BLOB
            );
            """
        )
        for image_id, name in enumerate(image_names, start=1):
            connection.execute(
                "INSERT INTO images(image_id, name) VALUES(?, ?)",
                (image_id, name),
            )
            connection.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, 12, 4, X'00')",
                (image_id,),
            )
            connection.execute(
                "INSERT INTO descriptors(image_id, rows, cols, data) "
                "VALUES(?, 12, 128, X'00')",
                (image_id,),
            )
        connection.commit()
    finally:
        connection.close()


def _fake_features(tmp_path: Path) -> ColmapFeatureExtractionResult:
    image_names = ("000000-a.pgm", "000001-b.pgm", "000002-c.pgm")
    observation_ids = tuple(ObservationId(value) for value in ("obs:a", "obs:b", "obs:c"))
    database_path = tmp_path / "features.db"
    _create_feature_database(database_path, image_names)
    database_hash = hash_file_content(database_path)
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:l3.2-parent"),
        source_observation_ids=observation_ids,
    )
    return ColmapFeatureExtractionResult(
        provenance=provenance,
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build="Commit fake-l32 without GPU support",
            ceres_version="2.2.0",
            upstream_has_cuda=False,
        ),
        configuration_sha256=ColmapFeatureExtractionConfig().sha256,
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=image_name,
                keypoint_rows=12,
                keypoint_cols=4,
                descriptor_rows=12,
                descriptor_cols=128,
            )
            for observation_id, image_name in zip(observation_ids, image_names, strict=True)
        ),
    )


def _request(
    tmp_path: Path,
    features: ColmapFeatureExtractionResult,
    *,
    config: ColmapPairMatchingConfig | None = None,
    revision: str | None = _FakePycolmap.COLMAP_build,
) -> ColmapPairMatchingRequest:
    actual_config = config or ColmapPairMatchingConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.3-test"),
        producer=ProducerRef(
            implementation="pycolmap.match_exhaustive",
            version="4.2.0",
            revision=revision,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        configuration_sha256=actual_config.sha256,
    )
    return ColmapPairMatchingRequest(
        run=run,
        features=features,
        database_path=tmp_path / "matches.db",
        config=actual_config,
    )


def test_matching_config_is_canonical_and_geometry_free() -> None:
    config = ColmapPairMatchingConfig()

    assert config.sha256 == ColmapPairMatchingConfig().sha256
    assert config.canonical_document()["device"] == "cpu"
    assert config.canonical_document()["num_threads"] == 1
    assert config.canonical_document()["skip_geometric_verification"] is True
    assert config.canonical_document()["guided_matching"] is False

    with pytest.raises(ValueError, match="skip geometric verification"):
        ColmapPairMatchingConfig(skip_geometric_verification=False)
    with pytest.raises(ValueError, match="guided matching"):
        ColmapPairMatchingConfig(guided_matching=True)
    with pytest.raises(ValueError, match="exactly one matching thread"):
        ColmapPairMatchingConfig(num_threads=2)


def test_fake_exhaustive_matching_preserves_parent_and_skips_geometry(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    source_before = features.database_path.read_bytes()
    request = _request(tmp_path, features)
    fake = _FakePycolmap()

    result = match_colmap_pairs(request, module=fake)

    assert features.database_path.read_bytes() == source_before
    assert hash_file_content(features.database_path).sha256 == features.database_sha256
    assert result.source_feature_database_sha256 == features.database_sha256
    assert result.database_path != features.database_path
    assert result.database_sha256 == hash_file_content(result.database_path).sha256
    assert result.attempted_pair_count == 3
    assert result.unverified_two_view_placeholder_count == 3
    assert tuple(
        (pair.observation_id1.value, pair.observation_id2.value, pair.num_matches)
        for pair in result.pairs
    ) == (
        ("obs:a", "obs:b", 7),
        ("obs:a", "obs:c", 7),
        ("obs:b", "obs:c", 7),
    )
    assert result.provenance.source_observation_ids == features.provenance.source_observation_ids

    assert fake.random_seed == 0
    assert len(fake.calls) == 1
    call = fake.calls[0]
    options = call["matching_options"]
    assert isinstance(options, _FakeFeatureMatchingOptions)
    assert options.type == "SIFT_BRUTEFORCE"
    assert options.num_threads == 1
    assert options.use_gpu is False
    assert options.guided_matching is False
    assert options.skip_geometric_verification is True
    assert options.rig_verification is False
    assert options.sift.max_ratio == 0.8
    assert options.sift.max_distance == 0.7
    assert options.sift.cross_check is True
    assert options.sift.cpu_brute_force_matcher is True
    pairing = call["pairing_options"]
    assert isinstance(pairing, _FakeExhaustivePairingOptions)
    assert pairing.block_size == 50
    assert call["device"] == "cpu"

    connection = sqlite3.connect(result.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 3
        placeholders = connection.execute(
            "SELECT rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2 "
            "FROM two_view_geometries ORDER BY pair_id"
        ).fetchall()
    finally:
        connection.close()
    assert placeholders == [(0, 2, None, 0, None, None, None, None, None, None, None)] * 3


def test_changed_feature_database_fails_before_solver(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    request = _request(tmp_path, features)
    features.database_path.write_bytes(features.database_path.read_bytes() + b"changed")
    fake = _FakePycolmap()

    with pytest.raises(ValueError, match="recorded artifact identity"):
        match_colmap_pairs(request, module=fake)

    assert fake.calls == []
    assert not request.database_path.exists()


def test_existing_output_database_is_never_reused(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    request = _request(tmp_path, features)
    request.database_path.write_bytes(b"existing")

    with pytest.raises(ValueError, match="must not already exist"):
        match_colmap_pairs(request, module=_FakePycolmap())

    assert request.database_path.read_bytes() == b"existing"


def test_geometric_contamination_is_rejected_and_partial_output_removed(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    request = _request(tmp_path, features)

    with pytest.raises(ColmapPairMatchingError, match="geometric"):
        match_colmap_pairs(request, module=_FakePycolmap(inject_geometry=True))

    assert not request.database_path.exists()
    assert features.database_path.exists()


def test_run_revision_must_match_current_matching_environment(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    request = _request(tmp_path, features, revision="different build")

    with pytest.raises(ValueError, match="producer revision must match COLMAP_build"):
        match_colmap_pairs(request, module=_FakePycolmap())

    assert not request.database_path.exists()


def test_request_rejects_wrong_membership_and_configuration(tmp_path: Path) -> None:
    features = _fake_features(tmp_path)
    config = ColmapPairMatchingConfig()
    wrong_config = ColmapPairMatchingConfig(max_num_matches=2048)
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.3-wrong"),
        producer=ProducerRef(
            implementation="pycolmap.match_exhaustive",
            version="4.2.0",
            revision=None,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        configuration_sha256=wrong_config.sha256,
    )

    with pytest.raises(ValueError, match="configuration SHA-256"):
        ColmapPairMatchingRequest(
            run=run,
            features=features,
            database_path=tmp_path / "matches.db",
            config=config,
        )


def test_real_pycolmap_raw_matching_when_integration_lane_enabled(tmp_path: Path) -> None:
    if os.environ.get("WRE_COLMAP_INTEGRATION") != "1":
        pytest.skip("real PyCOLMAP matching is exercised only in the COLMAP integration lane")

    pycolmap = __import__("pycolmap")
    path_a = tmp_path / "a.pgm"
    path_b = tmp_path / "b.pgm"
    _write_pgm(path_a)
    path_b.write_bytes(path_a.read_bytes())
    observation_a = _observation(path_a, "obs:real-a")
    observation_b = _observation(path_b, "obs:real-b")

    feature_config = ColmapFeatureExtractionConfig()
    feature_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.2-for-l3.3"),
        producer=ProducerRef(
            implementation="pycolmap.extract_features",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=(observation_a.observation_id, observation_b.observation_id),
        started_at=datetime(2026, 9, 15, 10, 0, tzinfo=UTC),
        configuration_sha256=feature_config.sha256,
    )
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=feature_run,
            inputs=(
                ColmapFeatureInput(observation=observation_a, source_path=path_a),
                ColmapFeatureInput(observation=observation_b, source_path=path_b),
            ),
            database_path=tmp_path / "real-features.db",
            config=feature_config,
        )
    )
    feature_hash_before = hash_file_content(features.database_path)

    matching_config = ColmapPairMatchingConfig()
    matching_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.3-real"),
        producer=ProducerRef(
            implementation="pycolmap.match_exhaustive",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 1, tzinfo=UTC),
        configuration_sha256=matching_config.sha256,
    )
    result = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=matching_run,
            features=features,
            database_path=tmp_path / "real-matches.db",
            config=matching_config,
        )
    )

    assert hash_file_content(features.database_path) == feature_hash_before
    assert result.environment.pycolmap_version == "4.2.0"
    assert result.attempted_pair_count == 1
    assert result.unverified_two_view_placeholder_count == 1
    assert len(result.pairs) == 1
    assert result.pairs[0].observation_id1 == observation_a.observation_id
    assert result.pairs[0].observation_id2 == observation_b.observation_id
    assert result.pairs[0].num_matches > 0
    assert result.database_sha256 == hash_file_content(result.database_path).sha256

    connection = sqlite3.connect(result.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 1
        placeholder = connection.execute(
            "SELECT rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2 "
            "FROM two_view_geometries"
        ).fetchone()
    finally:
        connection.close()
    assert placeholder == (0, 2, None, 0, None, None, None, None, None, None, None)
