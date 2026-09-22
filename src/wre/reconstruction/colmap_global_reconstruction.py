from __future__ import annotations

import hashlib
import importlib
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
from wre.domain.observations import Sha256Digest
from wre.domain.runs import DerivedArtifactProvenance, ReconstructionRun
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_adapter import COLMAP_PRECISION_OUTPUT_KINDS
from wre.reconstruction.colmap_environment import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_evidence_artifacts import (
    GEOMETRIC_VERIFICATION_KIND,
    IMAGE_OBSERVATION_KIND,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionError,
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
    _audit_reconstructions,
    _stage_inputs,
    _verified_database_images,
)
from wre.reconstruction.colmap_verification import ColmapGeometricVerificationResult

COLMAP_GLOBAL_CANONICAL_ADAPTER_ID = "colmap.global_precision_geometry"
COLMAP_GLOBAL_CANONICAL_CAPABILITY_NAME = AdapterCapabilityName(
    "geometry.precision_sfm.global"
)
COLMAP_GLOBAL_CANONICAL_CAPABILITY = AdapterCapabilityDescriptor(
    capability=COLMAP_GLOBAL_CANONICAL_CAPABILITY_NAME,
    input_kinds=frozenset({IMAGE_OBSERVATION_KIND, GEOMETRIC_VERIFICATION_KIND}),
    output_kinds=COLMAP_PRECISION_OUTPUT_KINDS,
)
COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION = "pycolmap.global_mapping"
COLMAP_GLOBAL_CANONICAL_PRODUCER_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_GLOBAL_CANONICAL_DEPENDENCY_REF = "colmap"
COLMAP_GLOBAL_CANONICAL_MODEL = None
COLMAP_GLOBAL_CANONICAL_CHECKPOINT = None
COLMAP_GLOBAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY = "required"
COLMAP_GLOBAL_CANONICAL_SHIPPING_STATUS = "experimental"
COLMAP_GLOBAL_CANONICAL_REPRODUCIBILITY_NOTES = (
    "V2L12.3 runs PyCOLMAP 4.2.0 global_mapping only on a fresh verified private "
    "copy of the geometric-verification database, with an explicit non-negative seed, "
    "one mapper thread, global-positioning GPU disabled, Ceres bundle adjustment selected "
    "and Ceres GPU disabled. View-graph calibration is also restricted to the private copy; "
    "PyCOLMAP 4.2.0 exposes its random seed but not its internal Ceres thread setting."
)

_GLOBAL_PRODUCER = COLMAP_GLOBAL_CANONICAL_PRODUCER_IMPLEMENTATION


class ColmapGlobalReconstructionError(RuntimeError):
    """Raised when the deterministic current COLMAP global route cannot be audited safely."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapGlobalReconstructionConfig:
    """Deterministic CPU reference profile for PyCOLMAP 4.2.0 global SfM."""

    schema_version: int = 1
    min_num_matches: int = 15
    ignore_watermarks: bool = False
    num_threads: int = 1
    random_seed: int = 0
    decompose_relative_pose: bool = True
    multiple_models: bool = True
    min_model_size: int = 3
    calibrate_view_graph: bool = True
    global_positioning_use_gpu: bool = False
    bundle_adjustment_backend: str = "CERES"
    bundle_adjustment_use_gpu: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("global reconstruction config schema_version must be 1")
        if (
            isinstance(self.min_num_matches, bool)
            or not isinstance(self.min_num_matches, int)
            or self.min_num_matches <= 0
        ):
            raise ValueError("min_num_matches must be a positive integer")
        if isinstance(self.num_threads, bool) or not isinstance(self.num_threads, int):
            raise ValueError("num_threads must be an integer")
        if self.num_threads != 1:
            raise ValueError("global deterministic reference requires one mapper thread")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if self.random_seed < 0:
            raise ValueError("global deterministic reference requires a non-negative random seed")
        if (
            isinstance(self.min_model_size, bool)
            or not isinstance(self.min_model_size, int)
            or self.min_model_size < 0
        ):
            raise ValueError("min_model_size must be a non-negative integer")
        if not self.multiple_models:
            raise ValueError("global reference must preserve disconnected mapper models")
        if not self.calibrate_view_graph:
            raise ValueError(
                "global reference requires explicit private view-graph calibration"
            )
        if self.global_positioning_use_gpu:
            raise ValueError("global reference requires CPU global positioning")
        if self.bundle_adjustment_backend != "CERES":
            raise ValueError("global reference requires the CERES bundle-adjustment backend")
        if self.bundle_adjustment_use_gpu:
            raise ValueError("global reference requires CPU Ceres bundle adjustment")

    def canonical_document(self) -> dict[str, object]:
        return {
            "bundle_adjustment_backend": self.bundle_adjustment_backend,
            "bundle_adjustment_use_gpu": self.bundle_adjustment_use_gpu,
            "calibrate_view_graph": self.calibrate_view_graph,
            "decompose_relative_pose": self.decompose_relative_pose,
            "global_positioning_use_gpu": self.global_positioning_use_gpu,
            "ignore_watermarks": self.ignore_watermarks,
            "min_model_size": self.min_model_size,
            "min_num_matches": self.min_num_matches,
            "multiple_models": self.multiple_models,
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
class ColmapGlobalReconstructionRequest:
    run: ReconstructionRun
    verification: ColmapGeometricVerificationResult
    inputs: tuple[ColmapReconstructionInput, ...]
    output_path: Path
    config: ColmapGlobalReconstructionConfig = field(
        default_factory=ColmapGlobalReconstructionConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("global reconstruction inputs must be a non-empty immutable tuple")
        if not all(isinstance(item, ColmapReconstructionInput) for item in self.inputs):
            raise ValueError(
                "global reconstruction inputs must contain only ColmapReconstructionInput values"
            )
        if not isinstance(self.output_path, Path):
            raise ValueError("output_path must be a pathlib.Path")
        if not isinstance(self.config, ColmapGlobalReconstructionConfig):
            raise TypeError("config must be ColmapGlobalReconstructionConfig")

        values = tuple(item.observation.observation_id.value for item in self.inputs)
        if len(values) != len(set(values)):
            raise ValueError("global reconstruction inputs cannot repeat an observation ID")
        image_names = tuple(item.image_name for item in self.inputs)
        if len(image_names) != len(set(image_names)):
            raise ValueError("global reconstruction inputs cannot repeat a COLMAP image name")

        canonical = tuple(
            sorted(self.inputs, key=lambda item: item.observation.observation_id.value)
        )
        object.__setattr__(self, "inputs", canonical)

        observation_ids = tuple(item.observation.observation_id for item in canonical)
        if observation_ids != self.verification.provenance.source_observation_ids:
            raise ValueError(
                "global reconstruction inputs must exactly match verification provenance membership"
            )
        if self.run.input_observation_ids != observation_ids:
            raise ValueError(
                "ReconstructionRun inputs must exactly match global reconstruction inputs"
            )
        if self.run.producer.implementation != _GLOBAL_PRODUCER:
            raise ValueError(f"ReconstructionRun producer must be {_GLOBAL_PRODUCER!r}")
        if self.run.producer.version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError(
                f"ReconstructionRun producer version must be {SUPPORTED_PYCOLMAP_VERSION!r}"
            )
        if self.run.configuration_sha256 != self.config.sha256:
            raise ValueError(
                "ReconstructionRun configuration SHA-256 must match the canonical global config"
            )
        if self.verification.environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
            raise ValueError("verification artifact uses an unsupported PyCOLMAP version")
        if self.verification.environment.colmap_version != SUPPORTED_COLMAP_VERSION:
            raise ValueError("verification artifact uses an unsupported COLMAP version")


@dataclass(frozen=True, slots=True)
class ColmapGlobalReconstructionResult:
    provenance: DerivedArtifactProvenance
    environment: ColmapEnvironmentIdentity
    configuration_sha256: Sha256Digest
    source_verification_database_sha256: Sha256Digest
    output_path: Path
    models: tuple[ColmapSparseModelArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.models, tuple):
            raise TypeError("models must be an immutable tuple")
        indices = tuple(item.model_index for item in self.models)
        if indices != tuple(sorted(indices)) or len(indices) != len(set(indices)):
            raise ValueError("global reconstruction models must be unique and canonically ordered")

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


def _configure_pycolmap(
    pycolmap: Any,
    config: ColmapGlobalReconstructionConfig,
) -> tuple[Any, Any]:
    try:
        calibration_options = pycolmap.ViewGraphCalibrationOptions()
        calibration_options.random_seed = config.random_seed

        options = pycolmap.GlobalPipelineOptions()
        options.min_num_matches = config.min_num_matches
        options.ignore_watermarks = config.ignore_watermarks
        options.num_threads = config.num_threads
        options.random_seed = config.random_seed
        options.decompose_relative_pose = config.decompose_relative_pose
        options.multiple_models = config.multiple_models
        options.min_model_size = config.min_model_size

        options.mapper.num_threads = config.num_threads
        options.mapper.random_seed = config.random_seed
        options.mapper.global_positioning.use_gpu = config.global_positioning_use_gpu
        options.mapper.bundle_adjustment.backend = pycolmap.BundleAdjustmentBackend.CERES
        if options.mapper.bundle_adjustment.ceres is None:
            raise ColmapGlobalReconstructionError(
                "PyCOLMAP global Ceres bundle-adjustment options are unavailable"
            )
        options.mapper.bundle_adjustment.ceres.use_gpu = config.bundle_adjustment_use_gpu
    except ColmapGlobalReconstructionError:
        raise
    except Exception as exc:
        raise ColmapGlobalReconstructionError(
            "PyCOLMAP 4.2.0 global options do not expose the required reference controls"
        ) from exc
    return calibration_options, options


def reconstruct_colmap_globally(
    request: ColmapGlobalReconstructionRequest,
    *,
    module: object | None = None,
) -> ColmapGlobalReconstructionResult:
    """Run pinned COLMAP global SfM only on verified private evidence copies."""

    database_path = request.verification.database_path.expanduser().resolve(strict=True)
    if not database_path.is_file():
        raise ValueError("verification database path must be a regular file")
    database_hash = hash_file_content(database_path)
    if (
        database_hash.sha256 != request.verification.database_sha256
        or database_hash.byte_length != request.verification.database_byte_length
    ):
        raise ValueError("verification database bytes do not match recorded artifact identity")

    output_path = request.output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            "output_path must not already exist for a fresh global reconstruction run"
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
        if (
            environment.pycolmap_version
            != request.verification.environment.pycolmap_version
            or environment.colmap_version
            != request.verification.environment.colmap_version
        ):
            raise ValueError(
                "global mapper and verification artifact must use the same COLMAP format version"
            )

        provenance = DerivedArtifactProvenance(
            producing_run_id=request.run.run_id,
            source_observation_ids=request.verification.provenance.source_observation_ids,
        )

        with tempfile.TemporaryDirectory(
            prefix="wre-colmap-global-",
            dir=output_path.parent,
        ) as working_dir_name:
            working_dir = Path(working_dir_name)
            working_database_path = working_dir / "verified.db"
            image_dir = working_dir / "images"
            image_dir.mkdir()

            shutil.copyfile(database_path, working_database_path)
            copied_database_hash = hash_file_content(working_database_path)
            if copied_database_hash != database_hash:
                raise ColmapGlobalReconstructionError(
                    "private verification database copy does not match its parent artifact"
                )

            expected_names = tuple(sorted(item.image_name for item in request.inputs))
            database_names = _verified_database_images(working_database_path, pycolmap)
            if database_names != expected_names:
                raise ValueError(
                    "global reconstruction inputs must exactly match COLMAP image membership "
                    "in the verification database"
                )

            try:
                _stage_inputs(request.inputs, image_dir)
            except ColmapIncrementalReconstructionError as exc:
                raise ColmapGlobalReconstructionError(str(exc)) from exc

            calibration_options, options = _configure_pycolmap(pycolmap, request.config)
            pycolmap.set_random_seed(request.config.random_seed)
            if request.config.calibrate_view_graph:
                calibrated = pycolmap.calibrate_view_graph(
                    working_database_path,
                    options=calibration_options,
                )
                if calibrated is not True:
                    raise ColmapGlobalReconstructionError(
                        "PyCOLMAP view-graph calibration did not succeed"
                    )

            reconstructions = pycolmap.global_mapping(
                working_database_path,
                image_dir,
                output_path,
                options=options,
            )
            try:
                models = _audit_reconstructions(output_path, reconstructions)
            except ColmapIncrementalReconstructionError as exc:
                raise ColmapGlobalReconstructionError(str(exc)) from exc

        database_hash_after = hash_file_content(database_path)
        if database_hash_after != database_hash:
            raise ColmapGlobalReconstructionError(
                "immutable verification database changed during global reconstruction"
            )

        owns_output = False
        return ColmapGlobalReconstructionResult(
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
