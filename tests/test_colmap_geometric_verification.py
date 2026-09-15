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
    ColmapFeatureInput,
    extract_colmap_features,
)
from wre.reconstruction.colmap_matching import (
    ColmapPairMatchingConfig,
    ColmapPairMatchingRequest,
    ColmapPairMatchingResult,
    ColmapPairMatchSummary,
    match_colmap_pairs,
)
from wre.reconstruction.colmap_verification import (
    ColmapGeometricVerificationConfig,
    ColmapGeometricVerificationError,
    ColmapGeometricVerificationRequest,
    verify_colmap_geometry,
)


class _FakeRansacOptions:
    def __init__(self) -> None:
        self.max_error = -1.0
        self.confidence = -1.0
        self.min_num_trials = -1
        self.max_num_trials = -1
        self.min_inlier_ratio = -1.0
        self.dyn_num_trials_multiplier = -1.0
        self.random_seed = -1
        self.num_threads = -1


class _FakeTwoViewGeometryOptions:
    def __init__(self) -> None:
        self.min_num_inliers = -1
        self.min_inlier_ratio = -1.0
        self.min_E_F_inlier_ratio = -1.0
        self.max_H_inlier_ratio = -1.0
        self.watermark_min_inlier_ratio = -1.0
        self.watermark_border_size = -1.0
        self.detect_watermark = False
        self.multiple_ignore_watermark = False
        self.watermark_detection_max_error = -1.0
        self.filter_stationary_matches = True
        self.stationary_matches_max_error = -1.0
        self.force_H_use = True
        self.use_degensac = True
        self.use_sampson_refinement = False
        self.compute_relative_pose = False
        self.multiple_models = True
        self.ransac = _FakeRansacOptions()


class _FakeGeometricVerifierOptions:
    def __init__(self) -> None:
        self.num_threads = -1
        self.rig_verification = True
        self.use_existing_relative_pose = True


class _FakeExistingMatchedPairingOptions:
    def __init__(self) -> None:
        self.batch_size = -1


class _FakePose:
    def __init__(self) -> None:
        self._matrix = [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
        ]

    def matrix(self) -> list[list[float]]:
        return self._matrix


class _FakeGeometry:
    def __init__(self, config: int, num_inliers: int) -> None:
        self.config = config
        self.E = [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
        self.F = [
            [0.0, -0.1, 0.2],
            [0.1, 0.0, -0.3],
            [-0.2, 0.3, 0.0],
        ]
        self.H = None
        self.cam2_from_cam1 = _FakePose()
        self.camera1 = object()
        self.camera2 = None
        self.inlier_matches = [[index, index + 1] for index in range(num_inliers)]
        self.tri_angle = 0.25
        self.inverted = False

    def invert(self) -> None:
        self.inverted = True
        self.inlier_matches = [[index2, index1] for index1, index2 in self.inlier_matches]
        self.camera1, self.camera2 = self.camera2, self.camera1


class _FakeDatabase:
    def __init__(self, path: Path, module: _FakePycolmap) -> None:
        self._path = path
        self._module = module
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

    def read_num_matches(self) -> tuple[list[int], list[int]]:
        connection = sqlite3.connect(self._path)
        try:
            rows = connection.execute(
                "SELECT pair_id, rows FROM matches ORDER BY pair_id"
            ).fetchall()
        finally:
            connection.close()
        return [int(row[0]) for row in rows], [int(row[1]) for row in rows]

    def read_two_view_geometries(self) -> tuple[list[int], list[_FakeGeometry]]:
        pair_ids, _ = self.read_num_matches()
        geometries = [self._module.geometries[pair_id] for pair_id in pair_ids]
        if self._module.drop_last_geometry and pair_ids:
            pair_ids = pair_ids[:-1]
            geometries = geometries[:-1]
        return pair_ids, geometries

    def close(self) -> None:
        self.closed = True


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-l34 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    GeometricVerifierOptions = _FakeGeometricVerifierOptions
    ExistingMatchedPairingOptions = _FakeExistingMatchedPairingOptions
    TwoViewGeometryOptions = _FakeTwoViewGeometryOptions
    TwoViewGeometryConfiguration = SimpleNamespace(
        UNDEFINED=0,
        DEGENERATE=1,
        CALIBRATED=2,
        UNCALIBRATED=3,
        PLANAR=4,
        PANORAMIC=5,
        PLANAR_OR_PANORAMIC=6,
        WATERMARK=7,
        MULTIPLE=8,
        CALIBRATED_RIG=9,
    )

    def __init__(self, *, drop_last_geometry: bool = False) -> None:
        self.random_seed: int | None = None
        self.calls: list[dict[str, object]] = []
        self.geometries: dict[int, _FakeGeometry] = {}
        self.drop_last_geometry = drop_last_geometry
        self.Database = SimpleNamespace(open=self._open_database)

    @staticmethod
    def _pair_id(image_id1: int, image_id2: int) -> int:
        low, high = sorted((image_id1, image_id2))
        return low * 1_000_000 + high

    @staticmethod
    def pair_id_to_image_pair(pair_id: int) -> tuple[int, int]:
        return pair_id // 1_000_000, pair_id % 1_000_000

    def _open_database(self, path: Path) -> _FakeDatabase:
        return _FakeDatabase(Path(path), self)

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def geometric_verification(
        self,
        database_path: Path,
        *,
        verifier_options: object,
        pairing_options: object,
        two_view_geometry_options: object,
    ) -> None:
        self.calls.append(
            {
                "database_path": database_path,
                "verifier_options": verifier_options,
                "pairing_options": pairing_options,
                "two_view_geometry_options": two_view_geometry_options,
            }
        )
        connection = sqlite3.connect(database_path)
        try:
            pair_rows = connection.execute(
                "SELECT pair_id, rows FROM matches ORDER BY pair_id"
            ).fetchall()
            for pair_id, raw_count in pair_rows:
                inliers = min(int(raw_count), 20)
                self.geometries[int(pair_id)] = _FakeGeometry(
                    self.TwoViewGeometryConfiguration.CALIBRATED,
                    inliers,
                )
                connection.execute(
                    "UPDATE two_view_geometries SET rows = ?, config = 2 WHERE pair_id = ?",
                    (inliers, pair_id),
                )
            connection.commit()
        finally:
            connection.close()


def _create_matching_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE images (
                image_id INTEGER PRIMARY KEY NOT NULL,
                name TEXT NOT NULL UNIQUE
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
        connection.execute("INSERT INTO images(image_id, name) VALUES(1, 'z.pgm')")
        connection.execute("INSERT INTO images(image_id, name) VALUES(2, 'a.pgm')")
        connection.execute(
            "INSERT INTO matches(pair_id, rows, cols, data) VALUES(1000002, 25, 2, X'00')"
        )
        connection.execute(
            "INSERT INTO two_view_geometries("
            "pair_id, rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2"
            ") VALUES(1000002, 0, 2, NULL, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL)"
        )
        connection.commit()
    finally:
        connection.close()


def _matching_result(tmp_path: Path) -> ColmapPairMatchingResult:
    database_path = tmp_path / "raw-matches.db"
    _create_matching_database(database_path)
    content = hash_file_content(database_path)
    observation_a = ObservationId("obs:a")
    observation_z = ObservationId("obs:z")
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:l3.3-parent"),
        source_observation_ids=(observation_a, observation_z),
    )
    return ColmapPairMatchingResult(
        provenance=provenance,
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build="Commit fake-l33 without GPU support",
            ceres_version="2.2.0",
            upstream_has_cuda=False,
        ),
        configuration_sha256=ColmapPairMatchingConfig().sha256,
        source_feature_database_sha256=hash_file_content(database_path).sha256,
        database_path=database_path,
        database_sha256=content.sha256,
        database_byte_length=content.byte_length,
        attempted_pair_count=1,
        unverified_two_view_placeholder_count=1,
        pairs=(
            ColmapPairMatchSummary(
                observation_id1=observation_a,
                observation_id2=observation_z,
                image_name1="a.pgm",
                image_name2="z.pgm",
                num_matches=25,
            ),
        ),
    )


def _request(
    tmp_path: Path,
    matching: ColmapPairMatchingResult,
    *,
    config: ColmapGeometricVerificationConfig | None = None,
    revision: str | None = _FakePycolmap.COLMAP_build,
) -> ColmapGeometricVerificationRequest:
    actual_config = config or ColmapGeometricVerificationConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.4-test"),
        producer=ProducerRef(
            implementation="pycolmap.geometric_verification",
            version="4.2.0",
            revision=revision,
        ),
        input_observation_ids=matching.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 30, tzinfo=UTC),
        configuration_sha256=actual_config.sha256,
    )
    return ColmapGeometricVerificationRequest(
        run=run,
        matching=matching,
        database_path=tmp_path / "verified.db",
        config=actual_config,
    )


def test_verification_config_is_canonical_and_keeps_later_geometry_out() -> None:
    config = ColmapGeometricVerificationConfig()

    assert config.sha256 == ColmapGeometricVerificationConfig().sha256
    assert config.verifier_num_threads == 1
    assert config.ransac_num_threads == 1
    assert config.ransac_random_seed == 0
    assert config.compute_relative_pose is True
    assert config.use_degensac is False
    assert config.multiple_models is False

    with pytest.raises(ValueError, match="one verifier thread"):
        ColmapGeometricVerificationConfig(verifier_num_threads=2)
    with pytest.raises(ValueError, match="DEGENSAC"):
        ColmapGeometricVerificationConfig(use_degensac=True)
    with pytest.raises(ValueError, match="multiple-model"):
        ColmapGeometricVerificationConfig(multiple_models=True)
    with pytest.raises(ValueError, match="deterministic non-negative RANSAC seed"):
        ColmapGeometricVerificationConfig(ransac_random_seed=-1)


def test_fake_geometric_verification_preserves_parent_and_canonicalizes_orientation(
    tmp_path: Path,
) -> None:
    matching = _matching_result(tmp_path)
    parent_before = matching.database_path.read_bytes()
    request = _request(tmp_path, matching)
    fake = _FakePycolmap()

    result = verify_colmap_geometry(request, module=fake)

    assert matching.database_path.read_bytes() == parent_before
    assert hash_file_content(matching.database_path).sha256 == matching.database_sha256
    assert result.source_matching_database_sha256 == matching.database_sha256
    assert result.database_path != matching.database_path
    assert result.database_sha256 == hash_file_content(result.database_path).sha256
    assert result.provenance.source_observation_ids == matching.provenance.source_observation_ids
    assert len(result.geometries) == 1

    evidence = result.geometries[0]
    assert evidence.observation_id1 == ObservationId("obs:a")
    assert evidence.observation_id2 == ObservationId("obs:z")
    assert evidence.image_name1 == "a.pgm"
    assert evidence.image_name2 == "z.pgm"
    assert evidence.raw_match_count == 25
    assert evidence.configuration == "CALIBRATED"
    assert evidence.inlier_count == 20
    assert evidence.inlier_matches[0] == (1, 0)
    assert evidence.fundamental_matrix is not None
    assert evidence.essential_matrix is not None
    assert evidence.homography_matrix is None
    assert evidence.relative_pose_matrix == (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
    )
    assert evidence.triangulation_angle_rad == 0.25
    assert evidence.has_estimated_camera1 is False
    assert evidence.has_estimated_camera2 is True
    assert fake.geometries[1_000_002].inverted is True

    assert fake.random_seed == 0
    assert len(fake.calls) == 1
    call = fake.calls[0]
    verifier = call["verifier_options"]
    pairing = call["pairing_options"]
    geometry = call["two_view_geometry_options"]
    assert isinstance(verifier, _FakeGeometricVerifierOptions)
    assert verifier.num_threads == 1
    assert verifier.rig_verification is False
    assert verifier.use_existing_relative_pose is False
    assert isinstance(pairing, _FakeExistingMatchedPairingOptions)
    assert pairing.batch_size == 1000
    assert isinstance(geometry, _FakeTwoViewGeometryOptions)
    assert geometry.min_num_inliers == 15
    assert geometry.compute_relative_pose is True
    assert geometry.use_degensac is False
    assert geometry.multiple_models is False
    assert geometry.ransac.random_seed == 0
    assert geometry.ransac.num_threads == 1


def test_changed_matching_database_fails_before_solver(tmp_path: Path) -> None:
    matching = _matching_result(tmp_path)
    request = _request(tmp_path, matching)
    matching.database_path.write_bytes(matching.database_path.read_bytes() + b"changed")
    fake = _FakePycolmap()

    with pytest.raises(ValueError, match="recorded artifact identity"):
        verify_colmap_geometry(request, module=fake)

    assert fake.calls == []
    assert not request.database_path.exists()


def test_existing_output_database_is_never_reused(tmp_path: Path) -> None:
    matching = _matching_result(tmp_path)
    request = _request(tmp_path, matching)
    request.database_path.write_bytes(b"existing")

    with pytest.raises(ValueError, match="must not already exist"):
        verify_colmap_geometry(request, module=_FakePycolmap())

    assert request.database_path.read_bytes() == b"existing"


def test_missing_geometry_result_is_rejected_and_partial_output_removed(tmp_path: Path) -> None:
    matching = _matching_result(tmp_path)
    request = _request(tmp_path, matching)

    with pytest.raises(ColmapGeometricVerificationError, match="exactly one geometry result"):
        verify_colmap_geometry(request, module=_FakePycolmap(drop_last_geometry=True))

    assert matching.database_path.exists()
    assert not request.database_path.exists()


def test_run_revision_must_match_current_verification_environment(tmp_path: Path) -> None:
    matching = _matching_result(tmp_path)
    request = _request(tmp_path, matching, revision="different build")

    with pytest.raises(ValueError, match="producer revision must match COLMAP_build"):
        verify_colmap_geometry(request, module=_FakePycolmap())

    assert not request.database_path.exists()


def test_request_rejects_wrong_configuration(tmp_path: Path) -> None:
    matching = _matching_result(tmp_path)
    config = ColmapGeometricVerificationConfig()
    wrong_config = ColmapGeometricVerificationConfig(ransac_max_num_trials=9000)
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.4-wrong"),
        producer=ProducerRef(
            implementation="pycolmap.geometric_verification",
            version="4.2.0",
            revision=None,
        ),
        input_observation_ids=matching.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 30, tzinfo=UTC),
        configuration_sha256=wrong_config.sha256,
    )

    with pytest.raises(ValueError, match="configuration SHA-256"):
        ColmapGeometricVerificationRequest(
            run=run,
            matching=matching,
            database_path=tmp_path / "verified.db",
            config=config,
        )


def _write_pgm(path: Path) -> None:
    width = 192
    height = 192
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            checker = ((x // 12) + (y // 12)) % 2
            gradient = (x * 7 + y * 5) % 64
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


def test_real_pycolmap_geometric_verification_when_integration_lane_enabled(
    tmp_path: Path,
) -> None:
    if os.environ.get("WRE_COLMAP_INTEGRATION") != "1":
        pytest.skip("real PyCOLMAP verification is exercised only in the COLMAP integration lane")

    pycolmap = __import__("pycolmap")
    path_a = tmp_path / "a.pgm"
    path_b = tmp_path / "b.pgm"
    _write_pgm(path_a)
    path_b.write_bytes(path_a.read_bytes())
    observation_a = _observation(path_a, "obs:real-a")
    observation_b = _observation(path_b, "obs:real-b")

    feature_config = ColmapFeatureExtractionConfig()
    feature_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.2-for-l3.4"),
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

    matching_config = ColmapPairMatchingConfig()
    matching_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.3-for-l3.4"),
        producer=ProducerRef(
            implementation="pycolmap.match_exhaustive",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 1, tzinfo=UTC),
        configuration_sha256=matching_config.sha256,
    )
    matching = match_colmap_pairs(
        ColmapPairMatchingRequest(
            run=matching_run,
            features=features,
            database_path=tmp_path / "real-matches.db",
            config=matching_config,
        )
    )
    matching_hash_before = hash_file_content(matching.database_path)

    verification_config = ColmapGeometricVerificationConfig()
    verification_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l3.4-real"),
        producer=ProducerRef(
            implementation="pycolmap.geometric_verification",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=matching.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, 10, 2, tzinfo=UTC),
        configuration_sha256=verification_config.sha256,
    )
    result = verify_colmap_geometry(
        ColmapGeometricVerificationRequest(
            run=verification_run,
            matching=matching,
            database_path=tmp_path / "real-verified.db",
            config=verification_config,
        )
    )

    assert hash_file_content(matching.database_path) == matching_hash_before
    assert result.environment.pycolmap_version == "4.2.0"
    assert result.source_matching_database_sha256 == matching.database_sha256
    assert result.database_sha256 == hash_file_content(result.database_path).sha256
    assert len(result.geometries) == 1
    evidence = result.geometries[0]
    assert evidence.observation_id1 == observation_a.observation_id
    assert evidence.observation_id2 == observation_b.observation_id
    assert evidence.raw_match_count > 0
    assert evidence.configuration != "UNDEFINED"
    assert evidence.inlier_count > 0
    assert (
        evidence.fundamental_matrix is not None
        or evidence.essential_matrix is not None
        or evidence.homography_matrix is not None
    )
