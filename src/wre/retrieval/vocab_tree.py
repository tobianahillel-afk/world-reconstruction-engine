from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.artifacts import ArtifactRef
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.pair_candidates import (
    PairCandidate,
    PairCandidateSource,
    PairCandidateSourceId,
    merge_pair_candidates,
)
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult

COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION = "pycolmap.vocab_tree_retrieval"
COLMAP_VOCAB_TREE_RETRIEVAL_VERSION = SUPPORTED_PYCOLMAP_VERSION
VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID = PairCandidateSourceId("colmap.vocab_tree")

_NETWORK_PREFIXES = ("http:", "https:", "ftp:", "s3:", "gs:")


class ColmapVocabTreeRetrievalError(RuntimeError):
    """Raised when classical vocabulary-tree retrieval cannot be audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class VocabTreeRetrievalConfig:
    """Canonical deterministic configuration for the classical retrieval baseline."""

    schema_version: int = 1
    num_images: int = 50
    num_nearest_neighbors: int = 5
    num_checks: int = 256
    max_num_features: int = 8192
    num_threads: int = 1
    random_seed: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("vocab tree retrieval config schema_version must be 1")
        for name in (
            "num_images",
            "num_nearest_neighbors",
            "num_checks",
            "max_num_features",
            "num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.num_threads != 1:
            raise ValueError("deterministic vocabulary retrieval requires exactly one thread")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.random_seed != 0:
            raise ValueError("deterministic vocabulary retrieval requires random_seed=0")

    def canonical_document(self) -> dict[str, object]:
        return {
            "max_num_features": self.max_num_features,
            "num_checks": self.num_checks,
            "num_images": self.num_images,
            "num_nearest_neighbors": self.num_nearest_neighbors,
            "num_threads": self.num_threads,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
        }

    @property
    def sha256(self) -> Sha256Digest:
        payload = json.dumps(
            self.canonical_document(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return Sha256Digest(hashlib.sha256(payload).hexdigest())


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapVocabTreeRetrievalRequest:
    """Exact retained SIFT input plus explicit local vocabulary identity."""

    run: ReconstructionRun
    features: ColmapFeatureExtractionResult
    vocab_tree_path: Path
    vocab_tree_sha256: Sha256Digest
    vocab_tree_byte_length: int
    config: VocabTreeRetrievalConfig = field(default_factory=VocabTreeRetrievalConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.run, ReconstructionRun):
            raise TypeError("vocab_tree_retrieval.run must be ReconstructionRun")
        if not isinstance(self.features, ColmapFeatureExtractionResult):
            raise TypeError(
                "vocab_tree_retrieval.features must be ColmapFeatureExtractionResult"
            )
        if not isinstance(self.vocab_tree_path, Path):
            raise TypeError("vocab_tree_retrieval.vocab_tree_path must be pathlib.Path")
        if not isinstance(self.vocab_tree_sha256, Sha256Digest):
            raise TypeError("vocab_tree_retrieval.vocab_tree_sha256 must be Sha256Digest")
        if (
            isinstance(self.vocab_tree_byte_length, bool)
            or not isinstance(self.vocab_tree_byte_length, int)
            or self.vocab_tree_byte_length <= 0
        ):
            raise ValueError("vocab_tree_retrieval.vocab_tree_byte_length must be positive")
        if not isinstance(self.config, VocabTreeRetrievalConfig):
            raise TypeError("vocab_tree_retrieval.config must be VocabTreeRetrievalConfig")
        if self.features.environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError("feature artifact was produced by an unsupported PyCOLMAP version")
        if self.features.environment.colmap_version != SUPPORTED_COLMAP_VERSION:
            raise ValueError("feature artifact was produced by an unsupported COLMAP version")
        if self.run.input_observation_ids != self.features.provenance.source_observation_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match the feature artifact observations"
            )
        if self.run.producer.implementation != COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION:
            raise ValueError(
                "ReconstructionRun producer must be "
                f"{COLMAP_VOCAB_TREE_RETRIEVAL_IMPLEMENTATION!r}"
            )
        if self.run.producer.version != COLMAP_VOCAB_TREE_RETRIEVAL_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical config"
            )


@dataclass(frozen=True, slots=True)
class ColmapVocabTreePair:
    observation_id1: ObservationId
    observation_id2: ObservationId

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id1, ObservationId):
            raise TypeError("vocab tree pair observation_id1 must be ObservationId")
        if not isinstance(self.observation_id2, ObservationId):
            raise TypeError("vocab tree pair observation_id2 must be ObservationId")
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError(
                "vocab tree pair observation IDs must be distinct and canonically ordered"
            )


@dataclass(frozen=True, slots=True)
class ColmapVocabTreeRetrievalResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    source_feature_database_sha256: Sha256Digest
    vocab_tree_sha256: Sha256Digest
    vocab_tree_byte_length: int
    pairs: tuple[ColmapVocabTreePair, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, DerivedArtifactProvenance):
            raise TypeError("vocab tree result provenance must be DerivedArtifactProvenance")
        if not isinstance(self.environment, ColmapEnvironmentIdentity):
            raise TypeError("vocab tree result environment must be ColmapEnvironmentIdentity")
        if not isinstance(self.configuration_sha256, Sha256Digest):
            raise TypeError("vocab tree result configuration_sha256 must be Sha256Digest")
        if not isinstance(self.source_feature_database_sha256, Sha256Digest):
            raise TypeError(
                "vocab tree result source_feature_database_sha256 must be Sha256Digest"
            )
        if not isinstance(self.vocab_tree_sha256, Sha256Digest):
            raise TypeError("vocab tree result vocab_tree_sha256 must be Sha256Digest")
        if (
            isinstance(self.vocab_tree_byte_length, bool)
            or not isinstance(self.vocab_tree_byte_length, int)
            or self.vocab_tree_byte_length <= 0
        ):
            raise ValueError("vocab tree result byte length must be positive")
        if not isinstance(self.pairs, tuple):
            raise TypeError("vocab tree result pairs must be an immutable tuple")
        if any(not isinstance(pair, ColmapVocabTreePair) for pair in self.pairs):
            raise TypeError("vocab tree result pairs must contain ColmapVocabTreePair")
        pair_keys = tuple(
            (pair.observation_id1.value, pair.observation_id2.value) for pair in self.pairs
        )
        if pair_keys != tuple(sorted(pair_keys)) or len(pair_keys) != len(set(pair_keys)):
            raise ValueError("vocab tree result pairs must be unique and canonically ordered")


@dataclass(frozen=True, slots=True)
class ColmapVocabTreePairCandidateAdapterInput:
    result: ColmapVocabTreeRetrievalResult
    evidence_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.result, ColmapVocabTreeRetrievalResult):
            raise TypeError(
                "vocab_tree_pair_adapter.result must be ColmapVocabTreeRetrievalResult"
            )
        if not isinstance(self.evidence_ref, ArtifactRef):
            raise TypeError("vocab_tree_pair_adapter.evidence_ref must be ArtifactRef")


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _verify_feature_database(features: ColmapFeatureExtractionResult) -> Path:
    database_path = features.database_path.expanduser().resolve(strict=True)
    if not database_path.is_file():
        raise ValueError("feature database path must be a regular file")
    actual = hash_file_content(database_path)
    if (
        actual.sha256 != features.database_sha256
        or actual.byte_length != features.database_byte_length
    ):
        raise ValueError("feature database bytes do not match the recorded artifact identity")
    return database_path


def _verify_local_vocab_tree(request: ColmapVocabTreeRetrievalRequest) -> Path:
    raw = str(request.vocab_tree_path).strip().lower()
    if raw.startswith(_NETWORK_PREFIXES):
        raise ValueError("vocabulary tree must be an explicit local filesystem path")
    vocab_tree_path = request.vocab_tree_path.expanduser().resolve(strict=True)
    if not vocab_tree_path.is_file():
        raise ValueError("vocabulary tree path must be a regular file")
    actual = hash_file_content(vocab_tree_path)
    if (
        actual.sha256 != request.vocab_tree_sha256
        or actual.byte_length != request.vocab_tree_byte_length
    ):
        raise ValueError("vocabulary tree bytes do not match the supplied identity")
    return vocab_tree_path


def _read_feature_membership(
    database_path: Path,
    features: ColmapFeatureExtractionResult,
) -> tuple[dict[int, ObservationId], tuple[int, ...]]:
    expected_by_name = {item.image_name: item.observation_id for item in features.images}
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ColmapVocabTreeRetrievalError("cannot open COLMAP feature database") from exc

    try:
        image_rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY image_id"
        ).fetchall()
        matches_count = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        geometry_count = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise ColmapVocabTreeRetrievalError(
            "COLMAP feature database has an invalid retrieval schema"
        ) from exc
    finally:
        connection.close()

    if matches_count != 0 or geometry_count != 0:
        raise ColmapVocabTreeRetrievalError(
            "vocabulary retrieval requires the retained feature-only database"
        )

    actual_names = tuple(str(row[1]) for row in image_rows)
    if len(actual_names) != len(set(actual_names)) or set(actual_names) != set(expected_by_name):
        raise ColmapVocabTreeRetrievalError(
            "feature database image membership differs from the feature artifact"
        )

    observation_by_image_id: dict[int, ObservationId] = {}
    for image_id_raw, image_name_raw in image_rows:
        image_id = int(image_id_raw)
        image_name = str(image_name_raw)
        if image_id <= 0 or image_id in observation_by_image_id:
            raise ColmapVocabTreeRetrievalError(
                "feature database contains invalid or duplicate image IDs"
            )
        observation_by_image_id[image_id] = expected_by_name[image_name]

    query_image_ids = tuple(
        image_id
        for image_id, _ in sorted(
            observation_by_image_id.items(),
            key=lambda item: item[1].value,
        )
    )
    return observation_by_image_id, query_image_ids


def _configure_pycolmap(
    pycolmap: Any,
    config: VocabTreeRetrievalConfig,
    vocab_tree_path: Path,
) -> Any:
    options = pycolmap.VocabTreePairingOptions()
    options.num_images = config.num_images
    options.num_nearest_neighbors = config.num_nearest_neighbors
    options.num_checks = config.num_checks
    options.num_images_after_verification = 0
    options.max_num_features = config.max_num_features
    options.vocab_tree_path = str(vocab_tree_path)
    options.match_list_path = ""
    options.num_threads = config.num_threads
    check = getattr(options, "check", None)
    if callable(check) and not bool(check()):
        raise ColmapVocabTreeRetrievalError("PyCOLMAP rejected vocabulary pairing options")
    return options


def _canonical_pairs(
    raw_pairs: object,
    observation_by_image_id: dict[int, ObservationId],
) -> tuple[ColmapVocabTreePair, ...]:
    if not isinstance(raw_pairs, (list, tuple)):
        raise ColmapVocabTreeRetrievalError(
            "PyCOLMAP vocabulary pair generator returned an unsupported container"
        )

    pairs: list[ColmapVocabTreePair] = []
    keys: set[tuple[str, str]] = set()
    for raw_pair in raw_pairs:
        if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
            raise ColmapVocabTreeRetrievalError(
                "PyCOLMAP vocabulary pair generator returned an invalid pair"
            )
        image_id1 = int(raw_pair[0])
        image_id2 = int(raw_pair[1])
        observation_id1 = observation_by_image_id.get(image_id1)
        observation_id2 = observation_by_image_id.get(image_id2)
        if observation_id1 is None or observation_id2 is None:
            raise ColmapVocabTreeRetrievalError(
                "PyCOLMAP vocabulary pair references an unknown database image"
            )
        if observation_id1 == observation_id2:
            raise ColmapVocabTreeRetrievalError(
                "PyCOLMAP vocabulary retrieval returned a self pair"
            )
        if observation_id2.value < observation_id1.value:
            observation_id1, observation_id2 = observation_id2, observation_id1
        key = (observation_id1.value, observation_id2.value)
        if key in keys:
            raise ColmapVocabTreeRetrievalError(
                "PyCOLMAP vocabulary retrieval returned a duplicate observation pair"
            )
        keys.add(key)
        pairs.append(
            ColmapVocabTreePair(
                observation_id1=observation_id1,
                observation_id2=observation_id2,
            )
        )

    pairs.sort(key=lambda pair: (pair.observation_id1.value, pair.observation_id2.value))
    return tuple(pairs)


def retrieve_colmap_vocab_tree_pairs(
    request: ColmapVocabTreeRetrievalRequest,
    *,
    module: object | None = None,
) -> ColmapVocabTreeRetrievalResult:
    """Propose image relationships with exact local COLMAP vocabulary retrieval only."""

    if not isinstance(request, ColmapVocabTreeRetrievalRequest):
        raise TypeError("request must be ColmapVocabTreeRetrievalRequest")

    source_database_path = _verify_feature_database(request.features)
    vocab_tree_path = _verify_local_vocab_tree(request)
    observation_by_image_id, query_image_ids = _read_feature_membership(
        source_database_path,
        request.features,
    )
    source_hash_before = hash_file_content(source_database_path)

    pycolmap = cast(Any, module) if module is not None else _load_pycolmap()
    environment = inspect_colmap_environment(pycolmap)
    if (
        request.run.producer.revision is not None
        and request.run.producer.revision != environment.colmap_build
    ):
        raise ValueError(
            "ReconstructionRun producer revision must match COLMAP_build when supplied"
        )

    provenance = DerivedArtifactProvenance(
        producing_run_id=request.run.run_id,
        source_observation_ids=request.features.provenance.source_observation_ids,
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="wre-colmap-vocab-tree-",
            dir=source_database_path.parent,
        ) as temporary_name:
            temporary_database_path = Path(temporary_name) / "features.db"
            shutil.copyfile(source_database_path, temporary_database_path)
            copied_hash = hash_file_content(temporary_database_path)
            if copied_hash != source_hash_before:
                raise ColmapVocabTreeRetrievalError(
                    "temporary feature database copy does not match the source artifact"
                )

            pairing_options = _configure_pycolmap(
                pycolmap,
                request.config,
                vocab_tree_path,
            )
            pycolmap.set_random_seed(request.config.random_seed)
            database = pycolmap.Database.open(temporary_database_path)
            try:
                generator = pycolmap.VocabTreePairGenerator(
                    pairing_options,
                    database,
                    query_image_ids=query_image_ids,
                )
                raw_pairs = generator.all_pairs()
            finally:
                database.close()

            temporary_mapping, temporary_query_ids = _read_feature_membership(
                temporary_database_path,
                request.features,
            )
            if temporary_mapping != observation_by_image_id or temporary_query_ids != query_image_ids:
                raise ColmapVocabTreeRetrievalError(
                    "vocabulary retrieval changed feature database membership"
                )
            pairs = _canonical_pairs(raw_pairs, observation_by_image_id)
    finally:
        source_hash_after = hash_file_content(source_database_path)
        if source_hash_after != source_hash_before:
            raise ColmapVocabTreeRetrievalError(
                "source feature database changed during vocabulary retrieval"
            )

    return ColmapVocabTreeRetrievalResult(
        provenance=provenance,
        environment=environment,
        configuration_sha256=request.config.sha256,
        source_feature_database_sha256=request.features.database_sha256,
        vocab_tree_sha256=request.vocab_tree_sha256,
        vocab_tree_byte_length=request.vocab_tree_byte_length,
        pairs=pairs,
    )


def adapt_colmap_vocab_tree_retrieval_result(
    adapter_input: ColmapVocabTreePairCandidateAdapterInput,
) -> tuple[PairCandidate, ...]:
    """Translate classical retrieval evidence into canonical pair proposals."""

    if not isinstance(adapter_input, ColmapVocabTreePairCandidateAdapterInput):
        raise TypeError(
            "adapter_input must be ColmapVocabTreePairCandidateAdapterInput"
        )

    candidates = tuple(
        PairCandidate(
            observation_id1=pair.observation_id1,
            observation_id2=pair.observation_id2,
            sources=(
                PairCandidateSource(
                    source_id=VOCAB_TREE_PAIR_CANDIDATE_SOURCE_ID,
                    evidence_refs=(adapter_input.evidence_ref,),
                ),
            ),
        )
        for pair in adapter_input.result.pairs
    )
    return merge_pair_candidates(candidates)
