from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import (
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult

_MATCH_PRODUCER = "pycolmap.match_exhaustive"
_UNDEFINED_TWO_VIEW_CONFIG = 0


class ColmapPairMatchingError(RuntimeError):
    """Raised when raw COLMAP pair matching cannot be completed or audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapPairMatchingConfig:
    """Canonical configuration for the deterministic L3.3 exhaustive SIFT baseline."""

    schema_version: int = 1
    max_num_matches: int = 32768
    max_ratio: float = 0.8
    max_distance: float = 0.7
    cross_check: bool = True
    cpu_brute_force_matcher: bool = True
    random_seed: int = 0
    num_threads: int = 1
    exhaustive_block_size: int = 50
    device: str = "cpu"
    matcher_type: str = "SIFT_BRUTEFORCE"
    pairing: str = "exhaustive"
    guided_matching: bool = False
    skip_geometric_verification: bool = True
    rig_verification: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("pair matching config schema_version must be 1")
        for name in ("max_num_matches", "num_threads", "exhaustive_block_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        for name in ("max_ratio", "max_distance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            numeric = float(value)
            if not math.isfinite(numeric) or numeric <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.max_ratio > 1:
            raise ValueError("max_ratio must be at most 1")
        if self.max_distance > 1:
            raise ValueError("max_distance must be at most 1")
        if self.device != "cpu":
            raise ValueError("L3.3 deterministic baseline requires CPU matching")
        if self.matcher_type != "SIFT_BRUTEFORCE":
            raise ValueError("L3.3 baseline supports only SIFT_BRUTEFORCE")
        if self.pairing != "exhaustive":
            raise ValueError("L3.3 baseline supports only exhaustive pairing")
        if self.num_threads != 1:
            raise ValueError("L3.3 deterministic baseline requires exactly one matching thread")
        if self.guided_matching:
            raise ValueError("guided matching belongs after geometric verification")
        if not self.skip_geometric_verification:
            raise ValueError("L3.3 must skip geometric verification")
        if self.rig_verification:
            raise ValueError("rig verification is geometric verification and is out of L3.3 scope")

    def canonical_document(self) -> dict[str, object]:
        return {
            "cpu_brute_force_matcher": self.cpu_brute_force_matcher,
            "cross_check": self.cross_check,
            "device": self.device,
            "exhaustive_block_size": self.exhaustive_block_size,
            "guided_matching": self.guided_matching,
            "matcher_type": self.matcher_type,
            "max_distance": self.max_distance,
            "max_num_matches": self.max_num_matches,
            "max_ratio": self.max_ratio,
            "num_threads": self.num_threads,
            "pairing": self.pairing,
            "random_seed": self.random_seed,
            "rig_verification": self.rig_verification,
            "schema_version": self.schema_version,
            "skip_geometric_verification": self.skip_geometric_verification,
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
class ColmapPairMatchingRequest:
    run: ReconstructionRun
    features: ColmapFeatureExtractionResult
    database_path: Path
    config: ColmapPairMatchingConfig = field(default_factory=ColmapPairMatchingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.database_path, Path):
            raise ValueError("database_path must be a pathlib.Path")
        if len(self.features.images) < 2:
            raise ValueError("pair matching requires at least two feature-bearing images")
        if self.run.input_observation_ids != self.features.provenance.source_observation_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match the L3.2 feature artifact"
            )
        if self.run.producer.implementation != _MATCH_PRODUCER:
            raise ValueError(f"ReconstructionRun producer must be {_MATCH_PRODUCER!r}")
        if self.run.producer.version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical matching config"
            )
        if self.features.environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                "L3.2 feature artifact was produced by an unsupported PyCOLMAP version"
            )


@dataclass(frozen=True, slots=True)
class ColmapPairMatchSummary:
    observation_id1: ObservationId
    observation_id2: ObservationId
    image_name1: str
    image_name2: str
    num_matches: int

    def __post_init__(self) -> None:
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError("pair observation IDs must be in canonical ascending order")
        if not self.image_name1.strip() or not self.image_name2.strip():
            raise ValueError("pair image names must be non-empty")
        if isinstance(self.num_matches, bool) or not isinstance(self.num_matches, int):
            raise ValueError("num_matches must be an integer")
        if self.num_matches < 0:
            raise ValueError("num_matches must be non-negative")


@dataclass(frozen=True, slots=True)
class ColmapPairMatchingResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    source_feature_database_sha256: Sha256Digest
    database_path: Path
    database_sha256: Sha256Digest
    database_byte_length: int
    attempted_pair_count: int
    unverified_two_view_placeholder_count: int
    pairs: tuple[ColmapPairMatchSummary, ...]

    def __post_init__(self) -> None:
        if isinstance(self.database_byte_length, bool) or self.database_byte_length <= 0:
            raise ValueError("database_byte_length must be a positive integer")
        if isinstance(self.attempted_pair_count, bool) or self.attempted_pair_count <= 0:
            raise ValueError("attempted_pair_count must be a positive integer")
        if (
            isinstance(self.unverified_two_view_placeholder_count, bool)
            or not isinstance(self.unverified_two_view_placeholder_count, int)
            or self.unverified_two_view_placeholder_count < 0
        ):
            raise ValueError("unverified_two_view_placeholder_count must be non-negative")
        if len(self.pairs) > self.attempted_pair_count:
            raise ValueError("stored match pairs cannot exceed attempted exhaustive pairs")
        if self.unverified_two_view_placeholder_count > self.attempted_pair_count:
            raise ValueError("unverified two-view placeholders cannot exceed attempted pairs")
        pair_ids = tuple(
            (item.observation_id1.value, item.observation_id2.value) for item in self.pairs
        )
        if pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("pair summaries must be unique and canonically ordered")


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _configure_pycolmap(pycolmap: Any, config: ColmapPairMatchingConfig) -> tuple[Any, Any]:
    matching_options = pycolmap.FeatureMatchingOptions()
    matching_options.type = pycolmap.FeatureMatcherType.SIFT_BRUTEFORCE
    matching_options.num_threads = config.num_threads
    matching_options.use_gpu = False
    matching_options.max_num_matches = config.max_num_matches
    matching_options.guided_matching = config.guided_matching
    matching_options.skip_geometric_verification = config.skip_geometric_verification
    matching_options.rig_verification = config.rig_verification
    matching_options.sift.max_ratio = config.max_ratio
    matching_options.sift.max_distance = config.max_distance
    matching_options.sift.cross_check = config.cross_check
    matching_options.sift.cpu_brute_force_matcher = config.cpu_brute_force_matcher

    pairing_options = pycolmap.ExhaustivePairingOptions()
    pairing_options.block_size = config.exhaustive_block_size
    return matching_options, pairing_options


def _read_match_database(
    database_path: Path,
    features: ColmapFeatureExtractionResult,
    pycolmap: Any,
) -> tuple[tuple[ColmapPairMatchSummary, ...], int]:
    expected_by_name = {item.image_name: item.observation_id for item in features.images}
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ColmapPairMatchingError("cannot open COLMAP matching database") from exc

    try:
        image_rows = connection.execute(
            "SELECT image_id, name FROM images ORDER BY image_id"
        ).fetchall()
        match_rows = connection.execute(
            "SELECT pair_id, rows, cols FROM matches ORDER BY pair_id"
        ).fetchall()
        geometry_rows = connection.execute(
            "SELECT pair_id, rows, cols, data, config, F, E, H, qvec, tvec, camera1, camera2 "
            "FROM two_view_geometries ORDER BY pair_id"
        ).fetchall()
    except sqlite3.Error as exc:
        raise ColmapPairMatchingError("COLMAP matching database has an invalid schema") from exc
    finally:
        connection.close()

    name_by_id = {int(image_id): str(name) for image_id, name in image_rows}
    if set(name_by_id.values()) != set(expected_by_name):
        raise ColmapPairMatchingError(
            "COLMAP matching database image membership differs from the L3.2 feature artifact"
        )

    match_pair_ids = {int(pair_id) for pair_id, _, _ in match_rows}
    for geometry_row in geometry_rows:
        (
            pair_id,
            rows,
            cols,
            data,
            config,
            fundamental,
            essential,
            homography,
            qvec,
            tvec,
            camera1,
            camera2,
        ) = geometry_row
        if int(pair_id) not in match_pair_ids:
            raise ColmapPairMatchingError(
                "COLMAP wrote an unverified two-view placeholder without a raw match row"
            )
        if int(config) != _UNDEFINED_TWO_VIEW_CONFIG:
            raise ColmapPairMatchingError(
                "L3.3 produced geometrically classified two-view content; verification belongs "
                "to L3.4"
            )
        if int(rows) != 0 or int(cols) != 2 or data not in (None, b""):
            raise ColmapPairMatchingError(
                "L3.3 produced two-view inlier content; geometric verification belongs to L3.4"
            )
        geometric_payloads = (
            fundamental,
            essential,
            homography,
            qvec,
            tvec,
            camera1,
            camera2,
        )
        if any(payload is not None for payload in geometric_payloads):
            raise ColmapPairMatchingError(
                "L3.3 produced geometric two-view payloads; verification belongs to L3.4"
            )

    summaries: list[ColmapPairMatchSummary] = []
    for pair_id, rows, cols in match_rows:
        num_matches = int(rows)
        if int(cols) != 2:
            raise ColmapPairMatchingError("COLMAP raw match rows must contain two feature indices")
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(int(pair_id))
        name1 = name_by_id.get(int(image_id1))
        name2 = name_by_id.get(int(image_id2))
        if name1 is None or name2 is None:
            raise ColmapPairMatchingError("COLMAP raw match references an unknown image")
        observation_id1 = expected_by_name[name1]
        observation_id2 = expected_by_name[name2]
        if observation_id1 == observation_id2:
            raise ColmapPairMatchingError("COLMAP raw match cannot pair an observation with itself")
        if observation_id2.value < observation_id1.value:
            observation_id1, observation_id2 = observation_id2, observation_id1
            name1, name2 = name2, name1
        summaries.append(
            ColmapPairMatchSummary(
                observation_id1=observation_id1,
                observation_id2=observation_id2,
                image_name1=name1,
                image_name2=name2,
                num_matches=num_matches,
            )
        )

    summaries.sort(key=lambda item: (item.observation_id1.value, item.observation_id2.value))
    pair_keys = [
        (item.observation_id1.value, item.observation_id2.value) for item in summaries
    ]
    if len(pair_keys) != len(set(pair_keys)):
        raise ColmapPairMatchingError(
            "COLMAP matching database contains duplicate observation pairs"
        )
    return tuple(summaries), len(geometry_rows)


def match_colmap_pairs(
    request: ColmapPairMatchingRequest,
    *,
    module: object | None = None,
) -> ColmapPairMatchingResult:
    """Exhaustively match L3.2 SIFT descriptors without geometric verification."""

    source_database_path = request.features.database_path.expanduser().resolve(strict=True)
    if not source_database_path.is_file():
        raise ValueError("L3.2 feature database path must be a regular file")
    source_hash = hash_file_content(source_database_path)
    if (
        source_hash.sha256 != request.features.database_sha256
        or source_hash.byte_length != request.features.database_byte_length
    ):
        raise ValueError("L3.2 feature database bytes do not match the recorded artifact identity")

    database_path = request.database_path.expanduser().resolve()
    if database_path.exists():
        raise ValueError("database_path must not already exist for a fresh L3.3 matching run")
    database_path.parent.mkdir(parents=True, exist_ok=True)

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
    attempted_pair_count = len(request.features.images) * (len(request.features.images) - 1) // 2

    try:
        shutil.copyfile(source_database_path, database_path)
        copied_hash = hash_file_content(database_path)
        if copied_hash != source_hash:
            raise ColmapPairMatchingError(
                "copied feature database does not match its parent artifact"
            )

        matching_options, pairing_options = _configure_pycolmap(pycolmap, request.config)
        pycolmap.set_random_seed(request.config.random_seed)
        pycolmap.match_exhaustive(
            database_path,
            matching_options=matching_options,
            pairing_options=pairing_options,
            device=pycolmap.Device.cpu,
        )
        pairs, placeholder_count = _read_match_database(
            database_path, request.features, pycolmap
        )
        database_hash = hash_file_content(database_path)
    except Exception:
        database_path.unlink(missing_ok=True)
        raise

    return ColmapPairMatchingResult(
        provenance=provenance,
        environment=environment,
        configuration_sha256=request.config.sha256,
        source_feature_database_sha256=request.features.database_sha256,
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        attempted_pair_count=attempted_pair_count,
        unverified_two_view_placeholder_count=placeholder_count,
        pairs=pairs,
    )