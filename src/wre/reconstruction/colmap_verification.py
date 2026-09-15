from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
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
from wre.reconstruction.colmap_matching import ColmapPairMatchingResult

_VERIFICATION_PRODUCER = "pycolmap.geometric_verification"
_CONFIGURATION_NAMES = (
    "UNDEFINED",
    "DEGENERATE",
    "CALIBRATED",
    "CALIBRATED_RIG",
    "UNCALIBRATED",
    "PLANAR",
    "PANORAMIC",
    "PLANAR_OR_PANORAMIC",
    "WATERMARK",
    "MULTIPLE",
)

Matrix3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]
Matrix3x4 = tuple[
    tuple[float, float, float, float],
    tuple[float, float, float, float],
    tuple[float, float, float, float],
]
FeatureMatch = tuple[int, int]


class ColmapGeometricVerificationError(RuntimeError):
    """Raised when COLMAP geometric verification cannot be completed or audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapGeometricVerificationConfig:
    """Canonical L3.4 baseline for COLMAP two-view geometric verification."""

    schema_version: int = 1
    verifier_num_threads: int = 1
    pairing_batch_size: int = 1000
    rig_verification: bool = False
    use_existing_relative_pose: bool = False
    min_num_inliers: int = 15
    min_inlier_ratio: float = 0.0
    min_E_F_inlier_ratio: float = 0.95
    max_H_inlier_ratio: float = 0.8
    watermark_min_inlier_ratio: float = 0.7
    watermark_border_size: float = 0.1
    detect_watermark: bool = True
    multiple_ignore_watermark: bool = True
    watermark_detection_max_error: float = 4.0
    filter_stationary_matches: bool = False
    stationary_matches_max_error: float = 4.0
    force_H_use: bool = False
    use_degensac: bool = False
    use_sampson_refinement: bool = True
    compute_relative_pose: bool = True
    multiple_models: bool = False
    ransac_max_error: float = 4.0
    ransac_confidence: float = 0.999
    ransac_min_num_trials: int = 100
    ransac_max_num_trials: int = 10000
    ransac_min_inlier_ratio: float = 0.25
    ransac_dyn_num_trials_multiplier: float = 3.0
    ransac_random_seed: int = 0
    ransac_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("geometric verification config schema_version must be 1")
        for name in (
            "verifier_num_threads",
            "pairing_batch_size",
            "min_num_inliers",
            "ransac_min_num_trials",
            "ransac_max_num_trials",
            "ransac_num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.pairing_batch_size <= 1:
            raise ValueError("pairing_batch_size must be greater than one")
        if self.verifier_num_threads != 1:
            raise ValueError("L3.4 deterministic baseline requires one verifier thread")
        if self.ransac_num_threads != 1:
            raise ValueError("L3.4 deterministic baseline requires one RANSAC thread")
        if isinstance(self.ransac_random_seed, bool) or not isinstance(
            self.ransac_random_seed, int
        ):
            raise ValueError("ransac_random_seed must be an integer")
        if self.ransac_random_seed < 0:
            raise ValueError("L3.4 requires a deterministic non-negative RANSAC seed")
        if self.ransac_min_num_trials > self.ransac_max_num_trials:
            raise ValueError("ransac_min_num_trials cannot exceed ransac_max_num_trials")

        positive_names = (
            "min_E_F_inlier_ratio",
            "max_H_inlier_ratio",
            "watermark_min_inlier_ratio",
            "watermark_border_size",
            "watermark_detection_max_error",
            "stationary_matches_max_error",
            "ransac_max_error",
            "ransac_confidence",
            "ransac_min_inlier_ratio",
            "ransac_dyn_num_trials_multiplier",
        )
        for name in positive_names:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if isinstance(self.min_inlier_ratio, bool) or not isinstance(
            self.min_inlier_ratio, (int, float)
        ):
            raise ValueError("min_inlier_ratio must be a finite number in [0, 1]")
        if not math.isfinite(float(self.min_inlier_ratio)) or not 0 <= self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be a finite number in [0, 1]")
        for name in (
            "min_E_F_inlier_ratio",
            "max_H_inlier_ratio",
            "watermark_min_inlier_ratio",
            "watermark_border_size",
            "ransac_confidence",
            "ransac_min_inlier_ratio",
        ):
            if float(getattr(self, name)) > 1:
                raise ValueError(f"{name} must be at most 1")
        if self.rig_verification:
            raise ValueError("rig verification is outside the L3.4 baseline")
        if self.use_existing_relative_pose:
            raise ValueError("L3.4 must estimate from L3.3 raw matches, not trust existing pose")
        if self.force_H_use:
            raise ValueError("L3.4 baseline must not force a homography-only model")
        if self.use_degensac:
            raise ValueError("DEGENSAC belongs to the later evidence-geometry lot")
        if self.multiple_models:
            raise ValueError("multiple-model verification belongs to later evidence geometry")
        if not self.compute_relative_pose:
            raise ValueError("L3.4 baseline retains COLMAP relative-pose evidence when available")

    def canonical_document(self) -> dict[str, object]:
        return {
            "compute_relative_pose": self.compute_relative_pose,
            "detect_watermark": self.detect_watermark,
            "filter_stationary_matches": self.filter_stationary_matches,
            "force_H_use": self.force_H_use,
            "max_H_inlier_ratio": self.max_H_inlier_ratio,
            "min_E_F_inlier_ratio": self.min_E_F_inlier_ratio,
            "min_inlier_ratio": self.min_inlier_ratio,
            "min_num_inliers": self.min_num_inliers,
            "multiple_ignore_watermark": self.multiple_ignore_watermark,
            "multiple_models": self.multiple_models,
            "pairing_batch_size": self.pairing_batch_size,
            "ransac_confidence": self.ransac_confidence,
            "ransac_dyn_num_trials_multiplier": self.ransac_dyn_num_trials_multiplier,
            "ransac_max_error": self.ransac_max_error,
            "ransac_max_num_trials": self.ransac_max_num_trials,
            "ransac_min_inlier_ratio": self.ransac_min_inlier_ratio,
            "ransac_min_num_trials": self.ransac_min_num_trials,
            "ransac_num_threads": self.ransac_num_threads,
            "ransac_random_seed": self.ransac_random_seed,
            "rig_verification": self.rig_verification,
            "schema_version": self.schema_version,
            "stationary_matches_max_error": self.stationary_matches_max_error,
            "use_degensac": self.use_degensac,
            "use_existing_relative_pose": self.use_existing_relative_pose,
            "use_sampson_refinement": self.use_sampson_refinement,
            "verifier_num_threads": self.verifier_num_threads,
            "watermark_border_size": self.watermark_border_size,
            "watermark_detection_max_error": self.watermark_detection_max_error,
            "watermark_min_inlier_ratio": self.watermark_min_inlier_ratio,
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
class ColmapGeometricVerificationRequest:
    run: ReconstructionRun
    matching: ColmapPairMatchingResult
    database_path: Path
    config: ColmapGeometricVerificationConfig = field(
        default_factory=ColmapGeometricVerificationConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.database_path, Path):
            raise ValueError("database_path must be a pathlib.Path")
        if self.run.input_observation_ids != self.matching.provenance.source_observation_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match the L3.3 matching artifact"
            )
        if self.run.producer.implementation != _VERIFICATION_PRODUCER:
            raise ValueError(
                f"ReconstructionRun producer must be {_VERIFICATION_PRODUCER!r}"
            )
        if self.run.producer.version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical verification config"
            )
        if self.matching.environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError("L3.3 match artifact was produced by an unsupported PyCOLMAP version")


@dataclass(frozen=True, slots=True)
class ColmapPairGeometryEvidence:
    observation_id1: ObservationId
    observation_id2: ObservationId
    image_name1: str
    image_name2: str
    raw_match_count: int
    configuration: str
    inlier_matches: tuple[FeatureMatch, ...]
    fundamental_matrix: Matrix3x3 | None
    essential_matrix: Matrix3x3 | None
    homography_matrix: Matrix3x3 | None
    relative_pose_matrix: Matrix3x4 | None
    triangulation_angle_rad: float | None
    has_estimated_camera1: bool
    has_estimated_camera2: bool

    def __post_init__(self) -> None:
        if self.observation_id1.value >= self.observation_id2.value:
            raise ValueError("pair observation IDs must be in canonical ascending order")
        if not self.image_name1.strip() or not self.image_name2.strip():
            raise ValueError("pair image names must be non-empty")
        if isinstance(self.raw_match_count, bool) or self.raw_match_count < 0:
            raise ValueError("raw_match_count must be a non-negative integer")
        if self.configuration not in _CONFIGURATION_NAMES:
            raise ValueError("unknown COLMAP two-view geometry configuration")
        if len(self.inlier_matches) > self.raw_match_count:
            raise ValueError("inlier matches cannot exceed raw match count")
        if len(self.inlier_matches) != len(set(self.inlier_matches)):
            raise ValueError("inlier matches must be unique")
        for index1, index2 in self.inlier_matches:
            if index1 < 0 or index2 < 0:
                raise ValueError("inlier feature indices must be non-negative")
        if self.triangulation_angle_rad is not None:
            if not math.isfinite(self.triangulation_angle_rad):
                raise ValueError("triangulation_angle_rad must be finite when present")
            if self.triangulation_angle_rad < 0:
                raise ValueError("triangulation_angle_rad cannot be negative")

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_matches)


@dataclass(frozen=True, slots=True)
class ColmapGeometricVerificationResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    source_matching_database_sha256: Sha256Digest
    database_path: Path
    database_sha256: Sha256Digest
    database_byte_length: int
    geometries: tuple[ColmapPairGeometryEvidence, ...]

    def __post_init__(self) -> None:
        if isinstance(self.database_byte_length, bool) or self.database_byte_length <= 0:
            raise ValueError("database_byte_length must be a positive integer")
        pair_ids = tuple(
            (item.observation_id1.value, item.observation_id2.value)
            for item in self.geometries
        )
        if pair_ids != tuple(sorted(pair_ids)) or len(pair_ids) != len(set(pair_ids)):
            raise ValueError("geometry evidence must be unique and canonically ordered")


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _configure_pycolmap(
    pycolmap: Any,
    config: ColmapGeometricVerificationConfig,
) -> tuple[Any, Any, Any]:
    verifier_options = pycolmap.GeometricVerifierOptions()
    verifier_options.num_threads = config.verifier_num_threads
    verifier_options.rig_verification = config.rig_verification
    verifier_options.use_existing_relative_pose = config.use_existing_relative_pose

    pairing_options = pycolmap.ExistingMatchedPairingOptions()
    pairing_options.batch_size = config.pairing_batch_size

    geometry_options = pycolmap.TwoViewGeometryOptions()
    geometry_options.min_num_inliers = config.min_num_inliers
    geometry_options.min_inlier_ratio = config.min_inlier_ratio
    geometry_options.min_E_F_inlier_ratio = config.min_E_F_inlier_ratio
    geometry_options.max_H_inlier_ratio = config.max_H_inlier_ratio
    geometry_options.watermark_min_inlier_ratio = config.watermark_min_inlier_ratio
    geometry_options.watermark_border_size = config.watermark_border_size
    geometry_options.detect_watermark = config.detect_watermark
    geometry_options.multiple_ignore_watermark = config.multiple_ignore_watermark
    geometry_options.watermark_detection_max_error = config.watermark_detection_max_error
    geometry_options.filter_stationary_matches = config.filter_stationary_matches
    geometry_options.stationary_matches_max_error = config.stationary_matches_max_error
    geometry_options.force_H_use = config.force_H_use
    geometry_options.use_degensac = config.use_degensac
    geometry_options.use_sampson_refinement = config.use_sampson_refinement
    geometry_options.compute_relative_pose = config.compute_relative_pose
    geometry_options.multiple_models = config.multiple_models
    geometry_options.ransac.max_error = config.ransac_max_error
    geometry_options.ransac.confidence = config.ransac_confidence
    geometry_options.ransac.min_num_trials = config.ransac_min_num_trials
    geometry_options.ransac.max_num_trials = config.ransac_max_num_trials
    geometry_options.ransac.min_inlier_ratio = config.ransac_min_inlier_ratio
    geometry_options.ransac.dyn_num_trials_multiplier = config.ransac_dyn_num_trials_multiplier
    geometry_options.ransac.random_seed = config.ransac_random_seed
    geometry_options.ransac.num_threads = config.ransac_num_threads
    return verifier_options, pairing_options, geometry_options


def _finite_matrix(value: object | None, rows: int, cols: int, label: str) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)) or len(raw) != rows:
        raise ColmapGeometricVerificationError(f"{label} must be a {rows}x{cols} matrix")
    converted: list[tuple[float, ...]] = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != cols:
            raise ColmapGeometricVerificationError(f"{label} must be a {rows}x{cols} matrix")
        values: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                raise ColmapGeometricVerificationError(f"{label} values must be numeric")
            numeric = float(item)
            if not math.isfinite(numeric):
                raise ColmapGeometricVerificationError(f"{label} values must be finite")
            values.append(numeric)
        converted.append(tuple(values))
    return tuple(converted)


def _matrix3(value: object | None, label: str) -> Matrix3x3 | None:
    matrix = _finite_matrix(value, 3, 3, label)
    return cast(Matrix3x3 | None, matrix)


def _pose_matrix(value: object | None) -> Matrix3x4 | None:
    if value is None:
        return None
    if not hasattr(value, "matrix"):
        raise ColmapGeometricVerificationError("relative pose does not expose a matrix")
    matrix = _finite_matrix(value.matrix(), 3, 4, "relative pose")
    return cast(Matrix3x4 | None, matrix)


def _inlier_matches(value: object) -> tuple[FeatureMatch, ...]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    if not isinstance(raw, (list, tuple)):
        raise ColmapGeometricVerificationError("inlier matches must be an Nx2 matrix")
    matches: list[FeatureMatch] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ColmapGeometricVerificationError("inlier matches must be an Nx2 matrix")
        index1, index2 = item
        if (
            isinstance(index1, bool)
            or isinstance(index2, bool)
            or not isinstance(index1, int)
            or not isinstance(index2, int)
            or index1 < 0
            or index2 < 0
        ):
            raise ColmapGeometricVerificationError(
                "inlier match indices must be non-negative integers"
            )
        matches.append((index1, index2))
    if len(matches) != len(set(matches)):
        raise ColmapGeometricVerificationError("COLMAP returned duplicate inlier matches")
    return tuple(matches)


def _configuration_name(pycolmap: Any, value: object) -> str:
    enum = pycolmap.TwoViewGeometryConfiguration
    for name in _CONFIGURATION_NAMES:
        if value == getattr(enum, name):
            return name
    raise ColmapGeometricVerificationError("COLMAP returned an unknown two-view configuration")


def _matching_name_map(matching: ColmapPairMatchingResult) -> dict[str, ObservationId]:
    name_map: dict[str, ObservationId] = {}
    for pair in matching.pairs:
        for name, observation_id in (
            (pair.image_name1, pair.observation_id1),
            (pair.image_name2, pair.observation_id2),
        ):
            existing = name_map.get(name)
            if existing is not None and existing != observation_id:
                raise ValueError("L3.3 image name maps to conflicting observation IDs")
            name_map[name] = observation_id
    return name_map


def _read_geometry_evidence(
    database_path: Path,
    matching: ColmapPairMatchingResult,
    pycolmap: Any,
) -> tuple[ColmapPairGeometryEvidence, ...]:
    expected_by_name = _matching_name_map(matching)
    expected_raw_counts = {
        (pair.observation_id1.value, pair.observation_id2.value): pair.num_matches
        for pair in matching.pairs
    }

    database = pycolmap.Database.open(database_path)
    try:
        images = database.read_all_images()
        pair_ids, raw_counts = database.read_num_matches()
        geometry_pair_ids, geometries = database.read_two_view_geometries()
    finally:
        database.close()

    name_by_id = {int(image.image_id): str(image.name) for image in images}
    raw_count_by_pair_id = {
        int(pair_id): int(count) for pair_id, count in zip(pair_ids, raw_counts, strict=True)
    }
    geometry_by_pair_id = {
        int(pair_id): geometry
        for pair_id, geometry in zip(geometry_pair_ids, geometries, strict=True)
    }
    if set(raw_count_by_pair_id) != set(geometry_by_pair_id):
        raise ColmapGeometricVerificationError(
            "COLMAP geometric verification did not produce exactly one geometry result per raw match pair"
        )

    evidence: list[ColmapPairGeometryEvidence] = []
    seen_wre_pairs: set[tuple[str, str]] = set()
    for pair_id in sorted(raw_count_by_pair_id):
        image_id1, image_id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1 = name_by_id.get(int(image_id1))
        name2 = name_by_id.get(int(image_id2))
        if name1 is None or name2 is None:
            raise ColmapGeometricVerificationError(
                "COLMAP verified geometry references an unknown image"
            )
        observation_id1 = expected_by_name.get(name1)
        observation_id2 = expected_by_name.get(name2)
        if observation_id1 is None or observation_id2 is None:
            raise ColmapGeometricVerificationError(
                "COLMAP verified geometry references an image outside the L3.3 raw-match evidence"
            )
        if observation_id1 == observation_id2:
            raise ColmapGeometricVerificationError(
                "COLMAP verified geometry cannot pair an observation with itself"
            )

        geometry = geometry_by_pair_id[pair_id]
        if observation_id2.value < observation_id1.value:
            geometry.invert()
            observation_id1, observation_id2 = observation_id2, observation_id1
            name1, name2 = name2, name1

        pair_key = (observation_id1.value, observation_id2.value)
        if pair_key in seen_wre_pairs:
            raise ColmapGeometricVerificationError(
                "COLMAP verification produced duplicate WRE observation pairs"
            )
        seen_wre_pairs.add(pair_key)
        expected_count = expected_raw_counts.get(pair_key)
        if expected_count is None:
            raise ColmapGeometricVerificationError(
                "COLMAP verification produced a pair not present in the L3.3 result"
            )
        raw_match_count = raw_count_by_pair_id[pair_id]
        if raw_match_count != expected_count:
            raise ColmapGeometricVerificationError(
                "raw match count changed between L3.3 and L3.4"
            )

        tri_angle = float(geometry.tri_angle)
        tri_angle_value = None if tri_angle < 0 else tri_angle
        evidence.append(
            ColmapPairGeometryEvidence(
                observation_id1=observation_id1,
                observation_id2=observation_id2,
                image_name1=name1,
                image_name2=name2,
                raw_match_count=raw_match_count,
                configuration=_configuration_name(pycolmap, geometry.config),
                inlier_matches=_inlier_matches(geometry.inlier_matches),
                fundamental_matrix=_matrix3(geometry.F, "fundamental matrix"),
                essential_matrix=_matrix3(geometry.E, "essential matrix"),
                homography_matrix=_matrix3(geometry.H, "homography matrix"),
                relative_pose_matrix=_pose_matrix(geometry.cam2_from_cam1),
                triangulation_angle_rad=tri_angle_value,
                has_estimated_camera1=geometry.camera1 is not None,
                has_estimated_camera2=geometry.camera2 is not None,
            )
        )

    evidence.sort(key=lambda item: (item.observation_id1.value, item.observation_id2.value))
    if set(expected_raw_counts) != {
        (item.observation_id1.value, item.observation_id2.value) for item in evidence
    }:
        raise ColmapGeometricVerificationError(
            "L3.4 geometry evidence membership differs from L3.3 raw-match membership"
        )
    return tuple(evidence)


def verify_colmap_geometry(
    request: ColmapGeometricVerificationRequest,
    *,
    module: object | None = None,
) -> ColmapGeometricVerificationResult:
    """Geometrically verify existing L3.3 raw matches without accepting world geometry."""

    source_database_path = request.matching.database_path.expanduser().resolve(strict=True)
    if not source_database_path.is_file():
        raise ValueError("L3.3 matching database path must be a regular file")
    source_hash = hash_file_content(source_database_path)
    if (
        source_hash.sha256 != request.matching.database_sha256
        or source_hash.byte_length != request.matching.database_byte_length
    ):
        raise ValueError("L3.3 matching database bytes do not match the recorded artifact identity")

    database_path = request.database_path.expanduser().resolve()
    if database_path.exists():
        raise ValueError("database_path must not already exist for a fresh L3.4 verification run")
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
        source_observation_ids=request.matching.provenance.source_observation_ids,
    )

    try:
        shutil.copyfile(source_database_path, database_path)
        copied_hash = hash_file_content(database_path)
        if copied_hash != source_hash:
            raise ColmapGeometricVerificationError(
                "copied matching database does not match its parent artifact"
            )

        verifier_options, pairing_options, geometry_options = _configure_pycolmap(
            pycolmap, request.config
        )
        pycolmap.set_random_seed(request.config.ransac_random_seed)
        pycolmap.geometric_verification(
            database_path,
            verifier_options=verifier_options,
            pairing_options=pairing_options,
            two_view_geometry_options=geometry_options,
        )
        geometries = _read_geometry_evidence(database_path, request.matching, pycolmap)
        database_hash = hash_file_content(database_path)
    except Exception:
        database_path.unlink(missing_ok=True)
        raise

    return ColmapGeometricVerificationResult(
        provenance=provenance,
        environment=environment,
        configuration_sha256=request.config.sha256,
        source_matching_database_sha256=request.matching.database_sha256,
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        geometries=geometries,
    )
