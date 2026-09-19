from __future__ import annotations

import os
import sqlite3
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import wre.retrieval.vocab_tree as vocab_tree_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    ObservationId,
    PairCandidate,
    PairCandidateSourceId,
)
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    SourceId,
    SourceRef,
)
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)
from wre.ingestion.hashing import FileContentHash, hash_file_content
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionConfig,
    ColmapFeatureExtractionRequest,
    ColmapFeatureExtractionResult,
    ColmapFeatureInput,
    ColmapImageFeatureSummary,
    extract_colmap_features,
)
from wre.retrieval import (
    COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION,
    COLMAP_VOCAB_TREE_RETRIEVAL_VERSION,
    VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID,
    ColmapVocabTreePair,
    ColmapVocabTreePairCandidateAdapterInput,
    ColmapVocabTreeRetrievalError,
    ColmapVocabTreeRetrievalRequest,
    ColmapVocabTreeRetrievalResult,
    VocabTreeRetrievalConfig,
    adapt_colmap_vocab_tree_retrieval_result,
    retrieve_colmap_vocab_tree_pairs,
)


class _FakeVocabTreePairingOptions:
    def __init__(self) -> None:
        self.num_images = -1
        self.num_nearest_neighbors = -1
        self.num_checks = -1
        self.num_images_after_verification = -1
        self.max_num_features = -1
        self.vocab_tree_path = ""
        self.match_list_path = "unexpected"
        self.num_threads = -1

    def check(self) -> bool:
        return bool(self.vocab_tree_path)


class _FakeDatabase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeDatabaseApi:
    def __init__(self, owner: _FakePycolmap) -> None:
        self.owner = owner

    def open(self, path: Path) -> _FakeDatabase:
        database = _FakeDatabase(path)
        self.owner.opened_databases.append(database)
        return database


class _FakeGenerator:
    def __init__(self, pairs: tuple[tuple[int, int], ...]) -> None:
        self.pairs = pairs

    def all_pairs(self) -> list[tuple[int, int]]:
        return list(self.pairs)


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-v2l7.4 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    def __init__(self, *, pairs: tuple[tuple[int, int], ...] = ()) -> None:
        self.pairs = pairs
        self.random_seed: int | None = None
        self.options: _FakeVocabTreePairingOptions | None = None
        self.generator_query_ids: tuple[int, ...] | None = None
        self.opened_databases: list[_FakeDatabase] = []
        self.Database = _FakeDatabaseApi(self)

    def VocabTreePairingOptions(self) -> _FakeVocabTreePairingOptions:
        self.options = _FakeVocabTreePairingOptions()
        return self.options

    def VocabTreePairGenerator(
        self,
        options: _FakeVocabTreePairingOptions,
        database: _FakeDatabase,
        *,
        query_image_ids: tuple[int, ...],
    ) -> _FakeGenerator:
        assert database in self.opened_databases
        assert options is self.options
        self.generator_query_ids = tuple(query_image_ids)
        return _FakeGenerator(self.pairs)

    def set_random_seed(self, value: int) -> None:
        self.random_seed = value

    def match_vocabtree(self, *args: object, **kwargs: object) -> None:
        raise AssertionError(f"retrieval must not call match_vocabtree: {args} {kwargs}")


def _write_feature_database(path: Path, image_names: tuple[str, ...]) -> None:
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
                data BLOB
            );
            """
        )
        for image_id, name in enumerate(image_names, start=1):
            connection.execute(
                "INSERT INTO images(image_id, name) VALUES(?, ?)",
                (image_id, name),
            )
            connection.execute(
                "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, 8, 4, X'00')",
                (image_id,),
            )
            connection.execute(
                "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, 8, 128, X'00')",
                (image_id,),
            )
        connection.commit()
    finally:
        connection.close()


def _feature_result(tmp_path: Path) -> ColmapFeatureExtractionResult:
    tmp_path.mkdir(parents=True, exist_ok=True)
    database_path = tmp_path / "features.db"
    image_names = ("image-a.pgm", "image-b.pgm", "image-c.pgm")
    _write_feature_database(database_path, image_names)
    database_hash = hash_file_content(database_path)
    observation_ids = tuple(ObservationId(value) for value in ("obs:a", "obs:b", "obs:c"))
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:feature-fixture"),
            source_observation_ids=observation_ids,
        ),
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build=_FakePycolmap.COLMAP_build,
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
                keypoint_rows=8,
                keypoint_cols=4,
                descriptor_rows=8,
                descriptor_cols=128,
            )
            for observation_id, image_name in zip(observation_ids, image_names, strict=True)
        ),
    )


def _vocabulary(tmp_path: Path) -> tuple[Path, FileContentHash]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "vocab-tree.bin"
    path.write_bytes(b"local-vocabulary-tree-fixture")
    return path, hash_file_content(path)


def _request(
    tmp_path: Path,
    *,
    features: ColmapFeatureExtractionResult | None = None,
    config: VocabTreeRetrievalConfig | None = None,
    revision: str | None = _FakePycolmap.COLMAP_build,
) -> ColmapVocabTreeRetrievalRequest:
    actual_features = features or _feature_result(tmp_path)
    vocab_path, vocab_hash = _vocabulary(tmp_path)
    actual_config = config or VocabTreeRetrievalConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:v2l7.4-vocab-tree"),
        producer=ProducerRef(
            implementation=COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION,
            version=COLMAP_VOCAB_TREE_RETRIEVAL_VERSION,
            revision=revision,
        ),
        input_observation_ids=actual_features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 19, 20, 30, tzinfo=UTC),
        configuration_sha256=actual_config.sha256,
    )
    return ColmapVocabTreeRetrievalRequest(
        run=run,
        features=actual_features,
        vocab_tree_path=vocab_path,
        vocab_tree_sha256=vocab_hash.sha256,
        vocab_tree_byte_length=vocab_hash.byte_length,
        config=actual_config,
    )


def _evidence_ref(value: str = "artifact:vocab-evidence") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(value),
        artifact_kind=ArtifactKind("pair.evidence.vocab_tree"),
    )


def test_config_is_exact_deterministic_and_fail_closed() -> None:
    config = VocabTreeRetrievalConfig()

    assert tuple(field.name for field in fields(VocabTreeRetrievalConfig)) == (
        "schema_version",
        "num_images",
        "num_nearest_neighbors",
        "num_checks",
        "max_num_features",
        "num_threads",
        "random_seed",
    )
    assert config.canonical_document() == VocabTreeRetrievalConfig().canonical_document()
    assert config.sha256 == VocabTreeRetrievalConfig().sha256
    assert config.num_threads == 1
    assert config.random_seed == 0
    with pytest.raises(FrozenInstanceError):
        config.num_images = 1  # type: ignore[misc]

    for kwargs in (
        {"num_images": 0},
        {"num_nearest_neighbors": -1},
        {"num_checks": True},
        {"max_num_features": 0},
        {"num_threads": 2},
        {"random_seed": 1},
    ):
        with pytest.raises(ValueError):
            VocabTreeRetrievalConfig(**kwargs)


def test_request_is_exact_typed_and_matches_run(tmp_path: Path) -> None:
    request = _request(tmp_path)

    assert tuple(field.name for field in fields(ColmapVocabTreeRetrievalRequest)) == (
        "run",
        "features",
        "vocab_tree_path",
        "vocab_tree_sha256",
        "vocab_tree_byte_length",
        "config",
    )
    assert request.run.configuration_sha256 == request.config.sha256
    assert request.run.input_observation_ids == request.features.provenance.source_observation_ids

    with pytest.raises(TypeError, match=r"pathlib\.Path"):
        ColmapVocabTreeRetrievalRequest(
            run=request.run,
            features=request.features,
            vocab_tree_path=cast(Any, str(request.vocab_tree_path)),
            vocab_tree_sha256=request.vocab_tree_sha256,
            vocab_tree_byte_length=request.vocab_tree_byte_length,
            config=request.config,
        )


def test_fake_retrieval_uses_only_local_pair_generator_and_preserves_source_db(
    tmp_path: Path,
) -> None:
    request = _request(
        tmp_path,
        config=VocabTreeRetrievalConfig(
            num_images=7,
            num_nearest_neighbors=2,
            num_checks=17,
            max_num_features=123,
        ),
    )
    before = hash_file_content(request.features.database_path)
    fake = _FakePycolmap(pairs=((1, 3), (1, 2)))

    result = retrieve_colmap_vocab_tree_pairs(request, module=fake)

    assert result.environment.pycolmap_version == "4.2.0"
    assert result.configuration_sha256 == request.config.sha256
    assert result.source_feature_database_sha256 == request.features.database_sha256
    assert result.vocab_tree_sha256 == request.vocab_tree_sha256
    assert tuple(
        (pair.observation_id1.value, pair.observation_id2.value) for pair in result.pairs
    ) == (("obs:a", "obs:b"), ("obs:a", "obs:c"))
    assert hash_file_content(request.features.database_path) == before
    assert fake.random_seed == 0
    assert fake.generator_query_ids == (1, 2, 3)
    assert fake.options is not None
    assert fake.options.num_images == 7
    assert fake.options.num_nearest_neighbors == 2
    assert fake.options.num_checks == 17
    assert fake.options.num_images_after_verification == 0
    assert fake.options.max_num_features == 123
    assert fake.options.vocab_tree_path == str(request.vocab_tree_path.resolve())
    assert fake.options.match_list_path == ""
    assert fake.options.num_threads == 1
    assert len(fake.opened_databases) == 1
    assert fake.opened_databases[0].closed


def test_source_and_vocabulary_mismatch_fail_before_native_execution(tmp_path: Path) -> None:
    request = _request(tmp_path)
    fake = _FakePycolmap()

    request.features.database_path.write_bytes(
        request.features.database_path.read_bytes() + b"changed"
    )
    with pytest.raises(ValueError, match="feature database bytes"):
        retrieve_colmap_vocab_tree_pairs(request, module=fake)
    assert fake.opened_databases == []

    fresh = _request(tmp_path / "fresh")
    fresh.vocab_tree_path.write_bytes(fresh.vocab_tree_path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="vocabulary tree bytes"):
        retrieve_colmap_vocab_tree_pairs(fresh, module=fake)
    assert fake.opened_databases == []


def test_network_like_vocabulary_path_is_rejected_before_native_execution(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    invalid = ColmapVocabTreeRetrievalRequest(
        run=request.run,
        features=request.features,
        vocab_tree_path=Path("https://example.invalid/vocab.bin"),
        vocab_tree_sha256=request.vocab_tree_sha256,
        vocab_tree_byte_length=request.vocab_tree_byte_length,
        config=request.config,
    )
    fake = _FakePycolmap()

    with pytest.raises(ValueError, match="local filesystem path"):
        retrieve_colmap_vocab_tree_pairs(invalid, module=fake)

    assert fake.opened_databases == []
    assert "urllib" not in vocab_tree_module.__dict__
    assert "requests" not in vocab_tree_module.__dict__


def test_invalid_generator_pairs_fail_closed(tmp_path: Path) -> None:
    request = _request(tmp_path)

    with pytest.raises(ColmapVocabTreeRetrievalError, match="unknown database image"):
        retrieve_colmap_vocab_tree_pairs(
            request,
            module=_FakePycolmap(pairs=((1, 99),)),
        )
    with pytest.raises(ColmapVocabTreeRetrievalError, match="image IDs must be integers"):
        retrieve_colmap_vocab_tree_pairs(
            request,
            module=_FakePycolmap(pairs=cast(Any, ((1, "2"),))),
        )


def test_native_self_and_symmetric_pairs_normalize_to_unique_relationships(
    tmp_path: Path,
) -> None:
    result = retrieve_colmap_vocab_tree_pairs(
        _request(tmp_path),
        module=_FakePycolmap(pairs=((1, 1), (1, 2), (2, 1), (2, 2))),
    )

    assert result.pairs == (
        ColmapVocabTreePair(
            observation_id1=ObservationId("obs:a"),
            observation_id2=ObservationId("obs:b"),
        ),
    )


def test_result_and_pair_contracts_are_source_specific_without_scores(tmp_path: Path) -> None:
    request = _request(tmp_path)
    result = retrieve_colmap_vocab_tree_pairs(
        request,
        module=_FakePycolmap(pairs=((2, 3),)),
    )

    assert tuple(field.name for field in fields(ColmapVocabTreePair)) == (
        "observation_id1",
        "observation_id2",
    )
    assert tuple(field.name for field in fields(ColmapVocabTreeRetrievalResult)) == (
        "provenance",
        "environment",
        "configuration_sha256",
        "source_feature_database_sha256",
        "vocab_tree_sha256",
        "vocab_tree_byte_length",
        "pairs",
    )
    assert not any(
        name in ColmapVocabTreePair.__dataclass_fields__
        for name in ("score", "rank", "confidence", "scene", "overlap", "image_id")
    )
    assert result.pairs == (
        ColmapVocabTreePair(
            observation_id1=ObservationId("obs:b"),
            observation_id2=ObservationId("obs:c"),
        ),
    )


def test_empty_retrieval_is_valid_and_does_not_fabricate_pairs(tmp_path: Path) -> None:
    result = retrieve_colmap_vocab_tree_pairs(
        _request(tmp_path),
        module=_FakePycolmap(),
    )

    assert result.pairs == ()


def test_pair_candidate_adaptation_is_pure_and_evidence_preserving(tmp_path: Path) -> None:
    result = retrieve_colmap_vocab_tree_pairs(
        _request(tmp_path),
        module=_FakePycolmap(pairs=((1, 2), (2, 3))),
    )
    first_ref = _evidence_ref("artifact:first")
    second_ref = _evidence_ref("artifact:second")
    first_input = ColmapVocabTreePairCandidateAdapterInput(
        result=result,
        evidence_ref=first_ref,
    )

    first = adapt_colmap_vocab_tree_retrieval_result(first_input)
    repeated = adapt_colmap_vocab_tree_retrieval_result(first_input)
    changed = adapt_colmap_vocab_tree_retrieval_result(
        ColmapVocabTreePairCandidateAdapterInput(
            result=result,
            evidence_ref=second_ref,
        )
    )

    assert VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID == PairCandidateSourceId("colmap.vocab_tree")
    assert first == repeated
    assert tuple(
        (candidate.observation_id1, candidate.observation_id2) for candidate in changed
    ) == tuple((candidate.observation_id1, candidate.observation_id2) for candidate in first)
    assert all(candidate.sources[0].evidence_refs == (first_ref,) for candidate in first)
    assert all(candidate.sources[0].evidence_refs == (second_ref,) for candidate in changed)
    assert all(
        tuple(field.name for field in fields(PairCandidate))
        == ("observation_id1", "observation_id2", "sources")
        for candidate in first
    )


def test_native_failure_cannot_mutate_retained_source_database(tmp_path: Path) -> None:
    request = _request(tmp_path)
    before = hash_file_content(request.features.database_path)

    class _FailingFake(_FakePycolmap):
        def VocabTreePairGenerator(
            self,
            options: _FakeVocabTreePairingOptions,
            database: _FakeDatabase,
            *,
            query_image_ids: tuple[int, ...],
        ) -> _FakeGenerator:
            raise RuntimeError("native failure")

    with pytest.raises(RuntimeError, match="native failure"):
        retrieve_colmap_vocab_tree_pairs(request, module=_FailingFake())

    assert hash_file_content(request.features.database_path) == before


def _write_pgm(path: Path, *, shift: int) -> None:
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
        source=SourceRef(source_id=SourceId("vocab-fixture"), locator=path.name),
        received_at=datetime(2026, 9, 19, 20, 30, tzinfo=UTC),
    )


def test_real_pycolmap_vocab_tree_retrieval_when_integration_lane_enabled(
    tmp_path: Path,
) -> None:
    if os.environ.get("WRE_COLMAP_INTEGRATION") != "1":
        pytest.skip("real vocabulary retrieval is exercised only in the COLMAP integration lane")

    pycolmap = cast(Any, __import__("pycolmap"))
    numpy = cast(Any, __import__("numpy"))

    observations: list[tuple[ImageObservation, Path]] = []
    for index in range(3):
        path = tmp_path / f"image-{index}.pgm"
        _write_pgm(path, shift=index)
        observations.append((_observation(path, f"obs:{index}"), path))

    feature_config = ColmapFeatureExtractionConfig(max_num_features=512)
    feature_ids = tuple(item[0].observation_id for item in observations)
    feature_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:v2l7.4-real-features"),
        producer=ProducerRef(
            implementation="pycolmap.extract_features",
            version="4.2.0",
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=feature_ids,
        started_at=datetime(2026, 9, 19, 20, 30, tzinfo=UTC),
        configuration_sha256=feature_config.sha256,
    )
    features = extract_colmap_features(
        ColmapFeatureExtractionRequest(
            run=feature_run,
            inputs=tuple(
                ColmapFeatureInput(observation=observation, source_path=path)
                for observation, path in observations
            ),
            database_path=tmp_path / "real-features.db",
            config=feature_config,
        )
    )
    source_before = hash_file_content(features.database_path)

    pycolmap.set_random_seed(0)
    descriptors = pycolmap.FeatureDescriptorsFloat(
        type=pycolmap.FeatureExtractorType.SIFT,
        data=numpy.random.default_rng(0).random((256, 128), dtype=numpy.float32),
    )
    visual_index = pycolmap.VisualIndex.create(128, 64)
    build_options = pycolmap.VisualIndex.BuildOptions()
    build_options.num_visual_words = 8
    build_options.num_iterations = 2
    build_options.num_rounds = 1
    build_options.num_checks = 16
    build_options.num_threads = 1
    visual_index.build(build_options, descriptors)
    vocab_path = tmp_path / "real-vocab.bin"
    visual_index.write(vocab_path)
    vocab_hash = hash_file_content(vocab_path)

    config = VocabTreeRetrievalConfig(
        num_images=2,
        num_nearest_neighbors=1,
        num_checks=16,
        max_num_features=256,
    )
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:v2l7.4-real"),
        producer=ProducerRef(
            implementation=COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION,
            version=COLMAP_VOCAB_TREE_RETRIEVAL_VERSION,
            revision=pycolmap.COLMAP_build,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 19, 20, 30, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    result = retrieve_colmap_vocab_tree_pairs(
        ColmapVocabTreeRetrievalRequest(
            run=run,
            features=features,
            vocab_tree_path=vocab_path,
            vocab_tree_sha256=vocab_hash.sha256,
            vocab_tree_byte_length=vocab_hash.byte_length,
            config=config,
        )
    )

    assert result.environment.pycolmap_version == "4.2.0"
    assert result.source_feature_database_sha256 == features.database_sha256
    assert result.vocab_tree_sha256 == vocab_hash.sha256
    assert hash_file_content(features.database_path) == source_before
    assert tuple(
        (pair.observation_id1.value, pair.observation_id2.value) for pair in result.pairs
    ) == tuple(
        sorted(
            (pair.observation_id1.value, pair.observation_id2.value)
            for pair in result.pairs
        )
    )

    connection = sqlite3.connect(features.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0] == 0
    finally:
        connection.close()
