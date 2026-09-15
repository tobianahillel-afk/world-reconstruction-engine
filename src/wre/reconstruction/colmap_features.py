from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.observations import (
    ImageObservation,
    ObservationId,
    Sha256Digest,
    VideoFrameObservation,
)
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import (
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)

_FEATURE_PRODUCER = "pycolmap.extract_features"
_SAFE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,10}$")


class ColmapFeatureExtractionError(RuntimeError):
    """Raised when deterministic COLMAP feature extraction cannot be completed safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapFeatureExtractionConfig:
    """WRE-owned, canonical configuration for the L3.2 SIFT baseline."""

    schema_version: int = 1
    max_image_size: int = -1
    max_num_features: int = 8192
    first_octave: int = -1
    num_octaves: int = 4
    octave_resolution: int = 3
    peak_threshold: float = 0.02 / 3.0
    edge_threshold: float = 10.0
    estimate_affine_shape: bool = False
    max_num_orientations: int = 2
    upright: bool = False
    darkness_adaptivity: bool = False
    domain_size_pooling: bool = False
    dsp_min_scale: float = 1.0 / 6.0
    dsp_max_scale: float = 3.0
    dsp_num_scales: int = 10
    normalization: str = "L1_ROOT"
    camera_model: str = "SIMPLE_RADIAL"
    default_focal_length_factor: float = 1.2
    random_seed: int = 0
    num_threads: int = 1
    device: str = "cpu"
    camera_mode: str = "PER_IMAGE"
    feature_type: str = "SIFT"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("feature extraction config schema_version must be 1")
        if isinstance(self.max_image_size, bool) or not isinstance(self.max_image_size, int):
            raise ValueError("max_image_size must be an integer")
        if self.max_image_size == 0 or self.max_image_size < -1:
            raise ValueError("max_image_size must be -1 or a positive integer")
        for name in (
            "max_num_features",
            "num_octaves",
            "octave_resolution",
            "max_num_orientations",
            "dsp_num_scales",
            "num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.first_octave, bool) or not isinstance(self.first_octave, int):
            raise ValueError("first_octave must be an integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        for name in (
            "peak_threshold",
            "edge_threshold",
            "dsp_min_scale",
            "dsp_max_scale",
            "default_focal_length_factor",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite positive number")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.dsp_min_scale > self.dsp_max_scale:
            raise ValueError("dsp_min_scale cannot exceed dsp_max_scale")
        if self.normalization != "L1_ROOT":
            raise ValueError("L3.2 baseline supports only L1_ROOT SIFT normalization")
        if self.camera_model != "SIMPLE_RADIAL":
            raise ValueError("L3.2 baseline supports only the SIMPLE_RADIAL solver camera model")
        if self.device != "cpu":
            raise ValueError("L3.2 deterministic baseline requires CPU extraction")
        if self.camera_mode != "PER_IMAGE":
            raise ValueError("L3.2 baseline requires CameraMode.PER_IMAGE")
        if self.feature_type != "SIFT":
            raise ValueError("L3.2 baseline supports only SIFT")
        if self.num_threads != 1:
            raise ValueError("L3.2 deterministic baseline requires exactly one extraction thread")
        if self.darkness_adaptivity:
            raise ValueError("darkness_adaptivity is GPU-only and unsupported by the CPU baseline")

    def canonical_document(self) -> dict[str, object]:
        return {
            "camera_mode": self.camera_mode,
            "camera_model": self.camera_model,
            "darkness_adaptivity": self.darkness_adaptivity,
            "default_focal_length_factor": self.default_focal_length_factor,
            "device": self.device,
            "domain_size_pooling": self.domain_size_pooling,
            "dsp_max_scale": self.dsp_max_scale,
            "dsp_min_scale": self.dsp_min_scale,
            "dsp_num_scales": self.dsp_num_scales,
            "edge_threshold": self.edge_threshold,
            "estimate_affine_shape": self.estimate_affine_shape,
            "feature_type": self.feature_type,
            "first_octave": self.first_octave,
            "max_image_size": self.max_image_size,
            "max_num_features": self.max_num_features,
            "max_num_orientations": self.max_num_orientations,
            "normalization": self.normalization,
            "num_octaves": self.num_octaves,
            "num_threads": self.num_threads,
            "octave_resolution": self.octave_resolution,
            "peak_threshold": self.peak_threshold,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
            "upright": self.upright,
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


ImageLikeObservation = ImageObservation | VideoFrameObservation


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapFeatureInput:
    observation: ImageLikeObservation
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.observation, (ImageObservation, VideoFrameObservation)):
            raise ValueError("feature input observation must be image-like")
        if not isinstance(self.source_path, Path):
            raise ValueError("feature input source_path must be a pathlib.Path")


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapFeatureExtractionRequest:
    run: ReconstructionRun
    inputs: tuple[ColmapFeatureInput, ...]
    database_path: Path
    config: ColmapFeatureExtractionConfig = field(default_factory=ColmapFeatureExtractionConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("feature extraction inputs must be a non-empty immutable tuple")
        if not all(isinstance(item, ColmapFeatureInput) for item in self.inputs):
            raise ValueError("feature extraction inputs must contain only ColmapFeatureInput values")
        if not isinstance(self.database_path, Path):
            raise ValueError("database_path must be a pathlib.Path")

        values = [item.observation.observation_id.value for item in self.inputs]
        if len(values) != len(set(values)):
            raise ValueError("feature extraction inputs cannot repeat an observation ID")
        canonical = tuple(sorted(self.inputs, key=lambda item: item.observation.observation_id.value))
        object.__setattr__(self, "inputs", canonical)

        observation_ids = tuple(item.observation.observation_id for item in canonical)
        if self.run.input_observation_ids != observation_ids:
            raise ValueError("ReconstructionRun inputs must exactly match feature extraction inputs")
        if self.run.producer.implementation != _FEATURE_PRODUCER:
            raise ValueError(f"ReconstructionRun producer must be {_FEATURE_PRODUCER!r}")
        if self.run.producer.version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError("ReconstructionRun configuration SHA-256 must match the canonical config")


@dataclass(frozen=True, slots=True)
class ColmapImageFeatureSummary:
    observation_id: ObservationId
    image_name: str
    keypoint_rows: int
    keypoint_cols: int
    descriptor_rows: int
    descriptor_cols: int

    def __post_init__(self) -> None:
        if not self.image_name or not self.image_name.strip():
            raise ValueError("image_name must be non-empty")
        for name in ("keypoint_rows", "keypoint_cols", "descriptor_rows", "descriptor_cols"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.keypoint_rows != self.descriptor_rows:
            raise ValueError("keypoint and descriptor row counts must agree")


@dataclass(frozen=True, slots=True)
class ColmapFeatureExtractionResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    database_path: Path
    database_sha256: Sha256Digest
    database_byte_length: int
    images: tuple[ColmapImageFeatureSummary, ...]

    def __post_init__(self) -> None:
        if isinstance(self.database_byte_length, bool) or self.database_byte_length <= 0:
            raise ValueError("database_byte_length must be a positive integer")
        if not self.images:
            raise ValueError("feature extraction result must contain at least one image summary")
        ids = tuple(item.observation_id for item in self.images)
        if ids != self.provenance.source_observation_ids:
            raise ValueError("feature summaries must follow canonical provenance observation order")


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _safe_snapshot_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if not _SAFE_SUFFIX_RE.fullmatch(suffix):
        raise ValueError(
            "feature source files must have a simple alphanumeric extension for snapshot decoding"
        )
    return suffix


def _snapshot_inputs(
    inputs: tuple[ColmapFeatureInput, ...],
    snapshot_dir: Path,
) -> tuple[tuple[str, ColmapFeatureInput], ...]:
    snapshots: list[tuple[str, ColmapFeatureInput]] = []
    for index, item in enumerate(inputs):
        source_path = item.source_path.expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"feature source is not a regular file: {source_path}")
        suffix = _safe_snapshot_suffix(source_path)
        image_name = f"{index:06d}-{item.observation.asset.sha256.value[:16]}{suffix}"
        snapshot_path = snapshot_dir / image_name
        shutil.copyfile(source_path, snapshot_path)
        snapshot_hash = hash_file_content(snapshot_path)
        if (
            snapshot_hash.sha256 != item.observation.asset.sha256
            or snapshot_hash.byte_length != item.observation.asset.byte_length
        ):
            raise ValueError(
                "feature source bytes do not match the persisted observation asset: "
                f"{item.observation.observation_id.value}"
            )
        snapshots.append((image_name, item))
    return tuple(snapshots)


def _configure_pycolmap(pycolmap: Any, config: ColmapFeatureExtractionConfig) -> tuple[Any, Any]:
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.type = pycolmap.FeatureExtractorType.SIFT
    extraction_options.max_image_size = config.max_image_size
    extraction_options.num_threads = config.num_threads
    extraction_options.sift.max_num_features = config.max_num_features
    extraction_options.sift.first_octave = config.first_octave
    extraction_options.sift.num_octaves = config.num_octaves
    extraction_options.sift.octave_resolution = config.octave_resolution
    extraction_options.sift.peak_threshold = config.peak_threshold
    extraction_options.sift.edge_threshold = config.edge_threshold
    extraction_options.sift.estimate_affine_shape = config.estimate_affine_shape
    extraction_options.sift.max_num_orientations = config.max_num_orientations
    extraction_options.sift.upright = config.upright
    extraction_options.sift.darkness_adaptivity = config.darkness_adaptivity
    extraction_options.sift.domain_size_pooling = config.domain_size_pooling
    extraction_options.sift.dsp_min_scale = config.dsp_min_scale
    extraction_options.sift.dsp_max_scale = config.dsp_max_scale
    extraction_options.sift.dsp_num_scales = config.dsp_num_scales
    extraction_options.sift.normalization = pycolmap.Normalization.L1_ROOT

    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = config.camera_model
    reader_options.camera_params = ""
    reader_options.default_focal_length_factor = config.default_focal_length_factor
    return reader_options, extraction_options


def _read_feature_database(
    database_path: Path,
    snapshots: tuple[tuple[str, ColmapFeatureInput], ...],
) -> tuple[ColmapImageFeatureSummary, ...]:
    expected_names = tuple(image_name for image_name, _ in snapshots)
    by_name = {image_name: item for image_name, item in snapshots}
    try:
        connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise ColmapFeatureExtractionError("cannot open COLMAP feature database") from exc

    try:
        rows = connection.execute(
            """
            SELECT images.name,
                   keypoints.rows, keypoints.cols,
                   descriptors.rows, descriptors.cols
            FROM images
            JOIN keypoints USING(image_id)
            JOIN descriptors USING(image_id)
            ORDER BY images.name
            """
        ).fetchall()
        matches_count = int(connection.execute("SELECT COUNT(*) FROM matches").fetchone()[0])
        geometry_count = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries").fetchone()[0]
        )
    except sqlite3.Error as exc:
        raise ColmapFeatureExtractionError("COLMAP feature database has an invalid schema") from exc
    finally:
        connection.close()

    actual_names = tuple(str(row[0]) for row in rows)
    if actual_names != expected_names:
        raise ColmapFeatureExtractionError(
            "COLMAP feature database image membership does not match the canonical input snapshot"
        )
    if matches_count != 0 or geometry_count != 0:
        raise ColmapFeatureExtractionError(
            "L3.2 feature extraction must not populate matches or two-view geometries"
        )

    summaries: list[ColmapImageFeatureSummary] = []
    for row in rows:
        image_name = str(row[0])
        item = by_name[image_name]
        summaries.append(
            ColmapImageFeatureSummary(
                observation_id=item.observation.observation_id,
                image_name=image_name,
                keypoint_rows=int(row[1]),
                keypoint_cols=int(row[2]),
                descriptor_rows=int(row[3]),
                descriptor_cols=int(row[4]),
            )
        )
    return tuple(summaries)


def extract_colmap_features(
    request: ColmapFeatureExtractionRequest,
    *,
    module: object | None = None,
) -> ColmapFeatureExtractionResult:
    """Extract SIFT features into a fresh COLMAP database without matching or geometry."""

    database_path = request.database_path.expanduser().resolve()
    if database_path.exists():
        raise ValueError("database_path must not already exist for a fresh L3.2 extraction")
    database_path.parent.mkdir(parents=True, exist_ok=True)

    pycolmap = cast(Any, module) if module is not None else _load_pycolmap()
    environment = inspect_colmap_environment(pycolmap)
    if (
        request.run.producer.revision is not None
        and request.run.producer.revision != environment.colmap_build
    ):
        raise ValueError("ReconstructionRun producer revision must match COLMAP_build when supplied")

    provenance = DerivedArtifactProvenance(
        producing_run_id=request.run.run_id,
        source_observation_ids=tuple(item.observation.observation_id for item in request.inputs),
    )

    try:
        with tempfile.TemporaryDirectory(
            prefix="wre-colmap-features-", dir=database_path.parent
        ) as snapshot_name:
            snapshot_dir = Path(snapshot_name)
            snapshots = _snapshot_inputs(request.inputs, snapshot_dir)
            reader_options, extraction_options = _configure_pycolmap(pycolmap, request.config)
            pycolmap.set_random_seed(request.config.random_seed)
            pycolmap.extract_features(
                database_path,
                snapshot_dir,
                image_names=[image_name for image_name, _ in snapshots],
                camera_mode=pycolmap.CameraMode.PER_IMAGE,
                reader_options=reader_options,
                extraction_options=extraction_options,
                device=pycolmap.Device.cpu,
            )
            summaries = _read_feature_database(database_path, snapshots)

        database_hash = hash_file_content(database_path)
    except Exception:
        database_path.unlink(missing_ok=True)
        raise

    return ColmapFeatureExtractionResult(
        provenance=provenance,
        environment=environment,
        configuration_sha256=request.config.sha256,
        database_path=database_path,
        database_sha256=database_hash.sha256,
        database_byte_length=database_hash.byte_length,
        images=summaries,
    )
