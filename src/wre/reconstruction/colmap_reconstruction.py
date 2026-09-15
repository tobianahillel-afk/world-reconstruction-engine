from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

from wre.domain.observations import (
    ImageObservation,
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
from wre.reconstruction.colmap_verification import ColmapGeometricVerificationResult

_RECONSTRUCTION_PRODUCER = "pycolmap.incremental_mapping"


class ColmapIncrementalReconstructionError(RuntimeError):
    """Raised when COLMAP incremental reconstruction cannot be audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapIncrementalReconstructionConfig:
    """Canonical L3.5 baseline for COLMAP incremental sparse reconstruction."""

    schema_version: int = 1
    min_num_matches: int = 15
    ignore_watermarks: bool = False
    multiple_models: bool = True
    max_num_models: int = 50
    max_model_overlap: int = 20
    min_model_size: int = 10
    init_num_trials: int = 200
    structure_less_registration_fallback: bool = True
    structure_less_registration_only: bool = False
    extract_colors: bool = False
    num_threads: int = 1
    random_seed: int = 0
    ba_refine_focal_length: bool = True
    ba_refine_principal_point: bool = False
    ba_refine_extra_params: bool = True
    ba_refine_sensor_from_rig: bool = True
    ba_use_gpu: bool = False
    use_prior_position: bool = False
    load_all_images: bool = False
    max_runtime_seconds: int = -1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("incremental reconstruction config schema_version must be 1")
        for name in (
            "min_num_matches",
            "max_num_models",
            "init_num_trials",
            "num_threads",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_model_overlap", "min_model_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("L3.5 requires a deterministic non-negative random seed")
        if isinstance(self.max_runtime_seconds, bool) or not isinstance(
            self.max_runtime_seconds, int
        ):
            raise ValueError("max_runtime_seconds must be an integer")
        if self.num_threads != 1:
            raise ValueError("L3.5 deterministic baseline requires one reconstruction thread")
        if self.ba_use_gpu:
            raise ValueError("L3.5 deterministic baseline requires CPU bundle adjustment")
        if self.extract_colors:
            raise ValueError("L3.5 keeps point colors outside the sparse-geometry baseline")
        if self.use_prior_position:
            raise ValueError("absolute/position priors are outside the L3.5 local baseline")
        if self.load_all_images:
            raise ValueError("load_all_images is for triangulation of pre-registered images")
        if self.structure_less_registration_only:
            raise ValueError("L3.5 baseline must retain normal structure-based registration")
        if not self.multiple_models:
            raise ValueError("L3.5 must allow disconnected COLMAP sub-models")

    def canonical_document(self) -> dict[str, object]:
        return {
            "ba_refine_extra_params": self.ba_refine_extra_params,
            "ba_refine_focal_length": self.ba_refine_focal_length,
            "ba_refine_principal_point": self.ba_refine_principal_point,
            "ba_refine_sensor_from_rig": self.ba_refine_sensor_from_rig,
            "ba_use_gpu": self.ba_use_gpu,
            "extract_colors": self.extract_colors,
            "ignore_watermarks": self.ignore_watermarks,
            "init_num_trials": self.init_num_trials,
            "load_all_images": self.load_all_images,
            "max_model_overlap": self.max_model_overlap,
            "max_num_models": self.max_num_models,
            "max_runtime_seconds": self.max_runtime_seconds,
            "min_model_size": self.min_model_size,
            "min_num_matches": self.min_num_matches,
            "multiple_models": self.multiple_models,
            "num_threads": self.num_threads,
            "random_seed": self.random_seed,
            "schema_version": self.schema_version,
            "structure_less_registration_fallback": self.structure_less_registration_fallback,
            "structure_less_registration_only": self.structure_less_registration_only,
            "use_prior_position": self.use_prior_position,
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
class ColmapReconstructionInput:
    observation: ImageLikeObservation
    source_path: Path
    image_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.observation, (ImageObservation, VideoFrameObservation)):
            raise ValueError("reconstruction input observation must be image-like")
        if not isinstance(self.source_path, Path):
            raise ValueError("reconstruction input source_path must be a pathlib.Path")
        if not isinstance(self.image_name, str) or not self.image_name.strip():
            raise ValueError("reconstruction input image_name must be non-empty")
        path = PurePosixPath(self.image_name)
        if path.is_absolute() or len(path.parts) != 1 or "\\" in self.image_name:
            raise ValueError("reconstruction input image_name must be one safe relative filename")
        if path.name in {".", ".."}:
            raise ValueError("reconstruction input image_name must be one safe relative filename")


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapIncrementalReconstructionRequest:
    run: ReconstructionRun
    verification: ColmapGeometricVerificationResult
    inputs: tuple[ColmapReconstructionInput, ...]
    output_path: Path
    config: ColmapIncrementalReconstructionConfig = field(
        default_factory=ColmapIncrementalReconstructionConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("reconstruction inputs must be a non-empty immutable tuple")
        if not all(isinstance(item, ColmapReconstructionInput) for item in self.inputs):
            raise ValueError(
                "reconstruction inputs must contain only ColmapReconstructionInput values"
            )
        if not isinstance(self.output_path, Path):
            raise ValueError("output_path must be a pathlib.Path")

        values = [item.observation.observation_id.value for item in self.inputs]
        if len(values) != len(set(values)):
            raise ValueError("reconstruction inputs cannot repeat an observation ID")
        image_names = [item.image_name for item in self.inputs]
        if len(image_names) != len(set(image_names)):
            raise ValueError("reconstruction inputs cannot repeat a COLMAP image name")
        canonical = tuple(
            sorted(self.inputs, key=lambda item: item.observation.observation_id.value)
        )
        object.__setattr__(self, "inputs", canonical)

        observation_ids = tuple(item.observation.observation_id for item in canonical)
        if observation_ids != self.verification.provenance.source_observation_ids:
            raise ValueError("reconstruction inputs must exactly match L3.4 provenance membership")
        if self.run.input_observation_ids != observation_ids:
            raise ValueError("ReconstructionRun inputs must exactly match reconstruction inputs")
        if self.run.producer.implementation != _RECONSTRUCTION_PRODUCER:
            raise ValueError(f"ReconstructionRun producer must be {_RECONSTRUCTION_PRODUCER!r}")
        if self.run.producer.version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical "
                "reconstruction config"
            )
        if self.verification.environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError("L3.4 verification artifact uses an unsupported PyCOLMAP version")


@dataclass(frozen=True, slots=True)
class ColmapModelFileArtifact:
    relative_path: str
    sha256: Sha256Digest
    byte_length: int

    def __post_init__(self) -> None:
        if not self.relative_path or PurePosixPath(self.relative_path).is_absolute():
            raise ValueError("model file relative_path must be non-empty and relative")
        if ".." in PurePosixPath(self.relative_path).parts or "\\" in self.relative_path:
            raise ValueError("model file relative_path must stay inside the model directory")
        if (
            isinstance(self.byte_length, bool)
            or not isinstance(self.byte_length, int)
            or self.byte_length < 0
        ):
            raise ValueError("model file byte_length must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ColmapSparseModelArtifact:
    model_index: int
    relative_path: str
    num_registered_images: int
    num_points3d: int
    files: tuple[ColmapModelFileArtifact, ...]

    def __post_init__(self) -> None:
        if isinstance(self.model_index, bool) or not isinstance(self.model_index, int):
            raise ValueError("model_index must be an integer")
        if self.model_index < 0:
            raise ValueError("model_index must be non-negative")
        if self.relative_path != str(self.model_index):
            raise ValueError("model relative_path must equal its COLMAP model index")
        for name in ("num_registered_images", "num_points3d"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        paths = tuple(item.relative_path for item in self.files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("model files must be unique and canonically ordered")
        if not self.files:
            raise ValueError("a reconstructed COLMAP model must contain solver-native files")


@dataclass(frozen=True, slots=True)
class ColmapIncrementalReconstructionResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    source_verification_database_sha256: Sha256Digest
    output_path: Path
    models: tuple[ColmapSparseModelArtifact, ...]

    def __post_init__(self) -> None:
        indices = tuple(item.model_index for item in self.models)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("reconstruction models must be unique and canonically ordered")

    @property
    def model_count(self) -> int:
        return len(self.models)

    @property
    def has_reconstruction(self) -> bool:
        return bool(self.models)


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapEnvironmentError(
            "PyCOLMAP is unavailable; install the approved external pycolmap==4.2.0 environment"
        ) from exc


def _configure_pycolmap(pycolmap: Any, config: ColmapIncrementalReconstructionConfig) -> Any:
    options = pycolmap.IncrementalPipelineOptions()
    options.min_num_matches = config.min_num_matches
    options.ignore_watermarks = config.ignore_watermarks
    options.multiple_models = config.multiple_models
    options.max_num_models = config.max_num_models
    options.max_model_overlap = config.max_model_overlap
    options.min_model_size = config.min_model_size
    options.init_num_trials = config.init_num_trials
    options.structure_less_registration_fallback = config.structure_less_registration_fallback
    options.structure_less_registration_only = config.structure_less_registration_only
    options.extract_colors = config.extract_colors
    options.num_threads = config.num_threads
    options.random_seed = config.random_seed
    options.ba_refine_focal_length = config.ba_refine_focal_length
    options.ba_refine_principal_point = config.ba_refine_principal_point
    options.ba_refine_extra_params = config.ba_refine_extra_params
    options.ba_refine_sensor_from_rig = config.ba_refine_sensor_from_rig
    options.ba_use_gpu = config.ba_use_gpu
    options.use_prior_position = config.use_prior_position
    options.load_all_images = config.load_all_images
    options.max_runtime_seconds = config.max_runtime_seconds
    return options


def _verified_database_images(database_path: Path, pycolmap: Any) -> tuple[str, ...]:
    database = pycolmap.Database.open(database_path)
    try:
        images = database.read_all_images()
    finally:
        database.close()
    names = tuple(sorted(str(image.name) for image in images))
    if len(names) != len(set(names)):
        raise ColmapIncrementalReconstructionError(
            "L3.4 database contains duplicate COLMAP image names"
        )
    return names


def _stage_inputs(
    inputs: tuple[ColmapReconstructionInput, ...],
    image_dir: Path,
) -> None:
    for item in inputs:
        source_path = item.source_path.expanduser().resolve(strict=True)
        if not source_path.is_file():
            raise ValueError(f"reconstruction source is not a regular file: {source_path}")
        source_hash = hash_file_content(source_path)
        if (
            source_hash.sha256 != item.observation.asset.sha256
            or source_hash.byte_length != item.observation.asset.byte_length
        ):
            raise ValueError(
                "reconstruction source bytes do not match the persisted observation asset: "
                f"{item.observation.observation_id.value}"
            )
        staged_path = image_dir / item.image_name
        shutil.copyfile(source_path, staged_path)
        if hash_file_content(staged_path) != source_hash:
            raise ColmapIncrementalReconstructionError(
                "staged reconstruction image does not match its verified source bytes"
            )


def _audit_model_files(model_path: Path) -> tuple[ColmapModelFileArtifact, ...]:
    files: list[ColmapModelFileArtifact] = []
    for path in sorted(model_path.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ColmapIncrementalReconstructionError(
                "COLMAP model output must not contain symbolic links"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ColmapIncrementalReconstructionError(
                "COLMAP model output contains a non-regular filesystem entry"
            )
        relative_path = path.relative_to(model_path).as_posix()
        digest = hash_file_content(path)
        files.append(
            ColmapModelFileArtifact(
                relative_path=relative_path,
                sha256=digest.sha256,
                byte_length=digest.byte_length,
            )
        )
    return tuple(files)


def _audit_reconstructions(
    output_path: Path,
    reconstructions: object,
) -> tuple[ColmapSparseModelArtifact, ...]:
    if not hasattr(reconstructions, "items"):
        raise ColmapIncrementalReconstructionError(
            "PyCOLMAP incremental_mapping must return a model-index mapping"
        )
    items = list(cast(Any, reconstructions).items())
    normalized: list[tuple[int, Any]] = []
    for raw_index, reconstruction in items:
        if isinstance(raw_index, bool) or not isinstance(raw_index, int) or raw_index < 0:
            raise ColmapIncrementalReconstructionError(
                "PyCOLMAP reconstruction model indices must be non-negative integers"
            )
        normalized.append((raw_index, reconstruction))
    normalized.sort(key=lambda item: item[0])
    indices = [index for index, _ in normalized]
    if len(indices) != len(set(indices)):
        raise ColmapIncrementalReconstructionError("PyCOLMAP returned duplicate model indices")

    expected_entries = {str(index) for index in indices}
    actual_entries = {entry.name for entry in output_path.iterdir()}
    if actual_entries != expected_entries:
        raise ColmapIncrementalReconstructionError(
            "COLMAP output directory membership does not match returned reconstruction models"
        )

    models: list[ColmapSparseModelArtifact] = []
    for index, reconstruction in normalized:
        model_path = output_path / str(index)
        if not model_path.is_dir() or model_path.is_symlink():
            raise ColmapIncrementalReconstructionError(
                "COLMAP reconstruction model path must be a real directory"
            )
        if not bool(reconstruction.is_valid()):
            raise ColmapIncrementalReconstructionError(
                "COLMAP returned an internally invalid reconstruction"
            )
        num_registered_images = reconstruction.num_reg_images()
        num_points3d = reconstruction.num_points3D()
        if (
            isinstance(num_registered_images, bool)
            or not isinstance(num_registered_images, int)
            or num_registered_images < 0
        ):
            raise ColmapIncrementalReconstructionError(
                "COLMAP num_reg_images must be a non-negative integer"
            )
        if (
            isinstance(num_points3d, bool)
            or not isinstance(num_points3d, int)
            or num_points3d < 0
        ):
            raise ColmapIncrementalReconstructionError(
                "COLMAP num_points3D must be a non-negative integer"
            )
        models.append(
            ColmapSparseModelArtifact(
                model_index=index,
                relative_path=str(index),
                num_registered_images=num_registered_images,
                num_points3d=num_points3d,
                files=_audit_model_files(model_path),
            )
        )
    return tuple(models)


def reconstruct_colmap_incrementally(
    request: ColmapIncrementalReconstructionRequest,
    *,
    module: object | None = None,
) -> ColmapIncrementalReconstructionResult:
    """Run COLMAP incremental SfM without importing solver geometry into WRE."""

    database_path = request.verification.database_path.expanduser().resolve(strict=True)
    if not database_path.is_file():
        raise ValueError("L3.4 verification database path must be a regular file")
    database_hash = hash_file_content(database_path)
    if (
        database_hash.sha256 != request.verification.database_sha256
        or database_hash.byte_length != request.verification.database_byte_length
    ):
        raise ValueError("L3.4 verification database bytes do not match recorded artifact identity")

    output_path = request.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            "output_path must not already exist for a fresh L3.5 reconstruction run"
        ) from exc

    owns_output = True
    try:
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
            source_observation_ids=request.verification.provenance.source_observation_ids,
        )

        with tempfile.TemporaryDirectory(
            prefix="wre-colmap-reconstruction-",
            dir=output_path.parent,
        ) as working_dir_name:
            working_dir = Path(working_dir_name)
            working_database_path = working_dir / "verified.db"
            image_dir = working_dir / "images"
            image_dir.mkdir()

            shutil.copyfile(database_path, working_database_path)
            copied_database_hash = hash_file_content(working_database_path)
            if copied_database_hash != database_hash:
                raise ColmapIncrementalReconstructionError(
                    "private L3.4 database copy does not match its parent artifact"
                )

            expected_names = tuple(sorted(item.image_name for item in request.inputs))
            database_names = _verified_database_images(working_database_path, pycolmap)
            if database_names != expected_names:
                raise ValueError(
                    "reconstruction inputs must exactly match COLMAP image membership "
                    "in L3.4 database"
                )

            _stage_inputs(request.inputs, image_dir)
            options = _configure_pycolmap(pycolmap, request.config)
            pycolmap.set_random_seed(request.config.random_seed)
            reconstructions = pycolmap.incremental_mapping(
                working_database_path,
                image_dir,
                output_path,
                options=options,
            )
            models = _audit_reconstructions(output_path, reconstructions)

            working_database_hash_after = hash_file_content(working_database_path)
            if working_database_hash_after != database_hash:
                raise ColmapIncrementalReconstructionError(
                    "COLMAP mutated the private L3.4 database copy during L3.5"
                )

        database_hash_after = hash_file_content(database_path)
        if database_hash_after != database_hash:
            raise ColmapIncrementalReconstructionError(
                "immutable L3.4 verification database changed during L3.5"
            )

        owns_output = False
        return ColmapIncrementalReconstructionResult(
            provenance=provenance,
            environment=environment,
            configuration_sha256=request.config.sha256,
            source_verification_database_sha256=request.verification.database_sha256,
            output_path=output_path,
            models=models,
        )
    except Exception:
        if owns_output:
            shutil.rmtree(output_path, ignore_errors=True)
        raise
