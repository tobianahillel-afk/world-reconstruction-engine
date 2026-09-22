from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from wre.domain.artifact_materialization import ArtifactMaterializationVerificationStatus
from wre.domain.cameras import ImageDimensions
from wre.domain.decoded_images import (
    DECODED_IMAGE_PYRAMID_KIND,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
)
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import (
    ArtifactProducerIdentity,
    CheckpointIdentity,
    ConfigurationIdentity,
    ModelIdentity,
)
from wre.domain.runs import ProducerRef
from wre.ingestion.decoded_images import DecodedImagePyramidMaterializationResult
from wre.ingestion.hashing import hash_file_content
from wre.materialization import verify_local_artifact_materialization
from wre.reconstruction.da3_preview import (
    Da3BaseObservationPrediction,
    normalize_da3_base_preview,
)
from wre.reconstruction.feed_forward_geometry import FeedForwardGeometryResult

DA3_SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DA3_RUNTIME_IMPLEMENTATION = "wre.reconstruction.da3_runtime"
DA3_RUNTIME_VERSION = "1"
DA3_MODEL = ModelIdentity(
    name="depth_anything_3.da3_base",
    version="1",
    revision=DA3_SOURCE_REVISION,
)
DA3_CHECKPOINT = CheckpointIdentity(
    identifier="depth-anything/DA3-BASE/model.safetensors",
    sha256=Sha256Digest(
        "e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5"
    ),
)

# Reference-v1 is intentionally a conservative CPU/float32 path. Accelerated GPU
# profiles are a later V2L13.6 responsibility and must receive distinct identities.
DA3_REFERENCE_PACKAGE_VERSIONS = (
    ("addict", "2.4.0"),
    ("einops", "0.8.0"),
    ("imageio", "2.35.1"),
    ("numpy", "1.26.4"),
    ("omegaconf", "2.3.0"),
    ("opencv-python", "4.10.0.84"),
    ("Pillow", "10.4.0"),
    ("safetensors", "0.4.5"),
    ("torch", "2.4.1"),
    ("torchvision", "0.19.1"),
    ("tqdm", "4.66.5"),
)

_NETWORK_PREFIXES = ("http:", "https:", "ftp:", "s3:", "gs:", "hf:")
_EXECUTABLE_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib")


class Da3RuntimeError(RuntimeError):
    """Raised when the exact DA3-BASE reference boundary cannot be verified."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Da3ReferenceConfig:
    schema_version: int = 1
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    ref_view_strategy: str = "saddle_balanced"
    infer_gs: bool = False
    use_ray_pose: bool = False
    num_workers: int = 1
    random_seed: int = 0
    device: str = "cpu"
    precision: str = "float32"

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("da3 config schema_version must be integer 1")
        if type(self.process_res) is not int or self.process_res != 504:
            raise ValueError("DA3-BASE reference process_res must be integer 504")
        if self.process_res_method != "upper_bound_resize":
            raise ValueError(
                "DA3-BASE reference process_res_method must be 'upper_bound_resize'"
            )
        if self.ref_view_strategy != "saddle_balanced":
            raise ValueError(
                "DA3-BASE reference ref_view_strategy must be 'saddle_balanced'"
            )
        if type(self.infer_gs) is not bool or self.infer_gs:
            raise ValueError("DA3-BASE reference infer_gs must be false")
        if type(self.use_ray_pose) is not bool or self.use_ray_pose:
            raise ValueError("DA3-BASE reference use_ray_pose must be false")
        if type(self.num_workers) is not int or self.num_workers != 1:
            raise ValueError("DA3-BASE reference num_workers must be integer 1")
        if type(self.random_seed) is not int or self.random_seed != 0:
            raise ValueError("DA3-BASE reference random_seed must be integer 0")
        if self.device != "cpu":
            raise ValueError("DA3-BASE reference device must be 'cpu'")
        if self.precision != "float32":
            raise ValueError("DA3-BASE reference precision must be 'float32'")

    def canonical_document(self) -> dict[str, object]:
        return {
            "device": self.device,
            "infer_gs": self.infer_gs,
            "num_workers": self.num_workers,
            "precision": self.precision,
            "process_res": self.process_res,
            "process_res_method": self.process_res_method,
            "random_seed": self.random_seed,
            "ref_view_strategy": self.ref_view_strategy,
            "schema_version": self.schema_version,
            "use_ray_pose": self.use_ray_pose,
        }

    @property
    def configuration(self) -> ConfigurationIdentity:
        encoded = json.dumps(
            self.canonical_document(),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(encoded).hexdigest())
        )


@dataclass(frozen=True, slots=True)
class Da3EnvironmentIdentity:
    source_revision: str
    python_version: str
    package_versions: tuple[tuple[str, str], ...]
    device: str
    precision: str

    def __post_init__(self) -> None:
        if self.source_revision != DA3_SOURCE_REVISION:
            raise ValueError("DA3 environment source revision is unsupported")
        if not isinstance(self.python_version, str) or not self.python_version.strip():
            raise ValueError("DA3 environment python_version must be non-blank")
        if not isinstance(self.package_versions, tuple):
            raise TypeError("DA3 environment package_versions must be an immutable tuple")
        if self.package_versions != DA3_REFERENCE_PACKAGE_VERSIONS:
            raise ValueError("DA3 environment package versions do not match reference-v1")
        if self.device != "cpu":
            raise ValueError("DA3 environment reference device must be cpu")
        if self.precision != "float32":
            raise ValueError("DA3 environment reference precision must be float32")


@dataclass(frozen=True, slots=True, kw_only=True)
class Da3ImageInput:
    decoded_result: DecodedImagePyramidMaterializationResult
    materialization_root: Path

    def __post_init__(self) -> None:
        if not isinstance(
            self.decoded_result,
            DecodedImagePyramidMaterializationResult,
        ):
            raise TypeError("DA3 image decoded_result has the wrong type")
        if not isinstance(self.materialization_root, Path):
            raise TypeError("DA3 image materialization_root must be pathlib.Path")


@dataclass(frozen=True, slots=True, kw_only=True)
class Da3ExecutionRequest:
    inputs: tuple[Da3ImageInput, ...]
    source_root: Path
    checkpoint_path: Path
    hardware_runtime: HardwareRuntimeIdentity
    config: Da3ReferenceConfig = Da3ReferenceConfig()
    model: ModelIdentity = DA3_MODEL
    checkpoint: CheckpointIdentity = DA3_CHECKPOINT

    def __post_init__(self) -> None:
        if not isinstance(self.inputs, tuple) or not self.inputs:
            raise ValueError("DA3 request inputs must be a non-empty immutable tuple")
        if any(not isinstance(item, Da3ImageInput) for item in self.inputs):
            raise TypeError("DA3 request inputs contain an invalid member")
        observation_values = tuple(
            item.decoded_result.manifest.source_observation_id.value for item in self.inputs
        )
        if observation_values != tuple(sorted(observation_values)):
            raise ValueError("DA3 request inputs must use canonical ObservationId order")
        if len(set(observation_values)) != len(observation_values):
            raise ValueError("DA3 request inputs must not repeat observations")
        if not isinstance(self.source_root, Path):
            raise TypeError("DA3 request source_root must be pathlib.Path")
        if not isinstance(self.checkpoint_path, Path):
            raise TypeError("DA3 request checkpoint_path must be pathlib.Path")
        if not isinstance(self.hardware_runtime, HardwareRuntimeIdentity):
            raise TypeError("DA3 request hardware_runtime must be HardwareRuntimeIdentity")
        if not isinstance(self.config, Da3ReferenceConfig):
            raise TypeError("DA3 request config must be Da3ReferenceConfig")
        if self.model != DA3_MODEL:
            raise ValueError("DA3 request model must match exact DA3-BASE identity")
        if self.checkpoint != DA3_CHECKPOINT:
            raise ValueError("DA3 request checkpoint must match exact DA3-BASE identity")


@dataclass(frozen=True, slots=True)
class _VerifiedRgbImage:
    observation_id: ObservationId
    width_px: int
    height_px: int
    rgb8: bytes
    artifact_key_sha256: Sha256Digest


@dataclass(frozen=True, slots=True)
class Da3ExecutionResult:
    producer: ArtifactProducerIdentity
    model: ModelIdentity
    checkpoint: CheckpointIdentity
    environment: Da3EnvironmentIdentity
    hardware_runtime: HardwareRuntimeIdentity
    normalization_identity: Sha256Digest
    geometry: FeedForwardGeometryResult

    def __post_init__(self) -> None:
        if not isinstance(self.producer, ArtifactProducerIdentity):
            raise TypeError("DA3 result producer must be ArtifactProducerIdentity")
        if self.model != DA3_MODEL:
            raise ValueError("DA3 result model must match exact DA3-BASE identity")
        if self.checkpoint != DA3_CHECKPOINT:
            raise ValueError("DA3 result checkpoint must match exact DA3-BASE identity")
        if not isinstance(self.environment, Da3EnvironmentIdentity):
            raise TypeError("DA3 result environment has the wrong type")
        if not isinstance(self.hardware_runtime, HardwareRuntimeIdentity):
            raise TypeError("DA3 result hardware_runtime has the wrong type")
        if not isinstance(self.normalization_identity, Sha256Digest):
            raise TypeError("DA3 result normalization_identity must be Sha256Digest")
        if not isinstance(self.geometry, FeedForwardGeometryResult):
            raise TypeError("DA3 result geometry must be FeedForwardGeometryResult")


class Da3Runtime(Protocol):
    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[_VerifiedRgbImage, ...],
        config: Da3ReferenceConfig,
    ) -> tuple[Da3EnvironmentIdentity, tuple[Da3BaseObservationPrediction, ...]]: ...


def _uri_like(path: Path) -> bool:
    value = str(path).strip().lower()
    return value.startswith(_NETWORK_PREFIXES)


def _run_git(source_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Da3RuntimeError("cannot inspect local DA3 git source") from exc
    if completed.returncode != 0:
        raise Da3RuntimeError("local DA3 source is not an auditable git checkout")
    return completed.stdout.strip()


def _verify_source_root(source_root: Path) -> Path:
    if _uri_like(source_root):
        raise Da3RuntimeError("DA3 source must be an explicit local path")
    try:
        resolved = source_root.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Da3RuntimeError("DA3 source root is unavailable") from exc
    if not resolved.is_dir():
        raise Da3RuntimeError("DA3 source root must be a directory")

    revision = _run_git(resolved, "rev-parse", "HEAD")
    if revision != DA3_SOURCE_REVISION:
        raise Da3RuntimeError("DA3 source revision does not match reviewed code")
    if _run_git(resolved, "status", "--porcelain", "--untracked-files=all"):
        raise Da3RuntimeError("DA3 source checkout must be clean with no untracked files")

    package_root = resolved / "src" / "depth_anything_3"
    config_path = package_root / "configs" / "da3-base.yaml"
    if not package_root.is_dir() or not config_path.is_file():
        raise Da3RuntimeError("DA3 source checkout is missing the reviewed BASE package/config")

    try:
        for candidate in package_root.rglob("*"):
            if candidate.is_symlink():
                raise Da3RuntimeError("DA3 reviewed source package must not contain symlinks")
            if candidate.is_file() and candidate.suffix.lower() in _EXECUTABLE_SUFFIXES:
                raise Da3RuntimeError(
                    "DA3 source package contains executable or bytecode shadow files"
                )
    except OSError as exc:
        raise Da3RuntimeError("cannot audit DA3 source package") from exc
    return resolved


def _verify_checkpoint(path: Path) -> Path:
    if _uri_like(path):
        raise Da3RuntimeError("DA3 checkpoint must be an explicit local path")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise Da3RuntimeError("DA3 checkpoint is unavailable") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise Da3RuntimeError("DA3 checkpoint must be a regular non-symlink file")
    content = hash_file_content(resolved)
    if content.byte_length <= 0:
        raise Da3RuntimeError("DA3 checkpoint must not be empty")
    if content.sha256 != DA3_CHECKPOINT.sha256:
        raise Da3RuntimeError("DA3 checkpoint SHA-256 does not match DA3-BASE")
    return resolved


def _verified_rgb_image(item: Da3ImageInput) -> _VerifiedRgbImage:
    result = item.decoded_result
    if result.materialization.artifact_ref.artifact_kind.value != DECODED_IMAGE_PYRAMID_KIND:
        raise Da3RuntimeError("DA3 requires media.decoded_image_pyramid artifacts")
    verification = verify_local_artifact_materialization(
        result.materialization,
        item.materialization_root,
    )
    if verification.status is not ArtifactMaterializationVerificationStatus.VERIFIED:
        raise Da3RuntimeError(
            f"DA3 decoded-image materialization is {verification.status.value}"
        )
    manifest = result.manifest
    if manifest.pixel_layout is not DecodedImagePixelLayout.RGB8_PACKED:
        raise Da3RuntimeError("DA3 requires packed RGB8 decoded pixels")
    if manifest.orientation_policy is not DecodedImageOrientationPolicy.SOURCE_PIXELS:
        raise Da3RuntimeError("DA3 requires source-pixel orientation")

    level = manifest.levels[0]
    entries = {entry.relative_path: entry for entry in result.materialization.entries}
    entry = entries.get(level.relative_path)
    if entry is None:
        raise Da3RuntimeError("DA3 decoded-image level zero is not materialized")
    expected_length = level.width_px * level.height_px * 3
    if entry.byte_length != expected_length:
        raise Da3RuntimeError("DA3 decoded-image level zero has invalid RGB8 byte length")

    path = item.materialization_root.joinpath(*level.relative_path.split("/"))
    try:
        rgb8 = path.read_bytes()
    except OSError as exc:
        raise Da3RuntimeError("cannot read verified DA3 decoded-image level zero") from exc
    if len(rgb8) != expected_length:
        raise Da3RuntimeError("DA3 decoded-image bytes changed after materialization verify")
    if Sha256Digest(hashlib.sha256(rgb8).hexdigest()) != entry.sha256:
        raise Da3RuntimeError("DA3 decoded-image SHA-256 changed after materialization verify")

    return _VerifiedRgbImage(
        observation_id=manifest.source_observation_id,
        width_px=level.width_px,
        height_px=level.height_px,
        rgb8=rgb8,
        artifact_key_sha256=result.artifact_key.sha256,
    )


def _producer(config: Da3ReferenceConfig) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=DA3_RUNTIME_IMPLEMENTATION,
            version=DA3_RUNTIME_VERSION,
            revision=DA3_SOURCE_REVISION,
        ),
        configuration=config.configuration,
        model=DA3_MODEL,
        checkpoint=DA3_CHECKPOINT,
    )


def _normalization_identity(
    *,
    producer: ArtifactProducerIdentity,
    images: tuple[_VerifiedRgbImage, ...],
    hardware_runtime: HardwareRuntimeIdentity,
) -> Sha256Digest:
    payload = {
        "domain": "wre.da3-base.normalization",
        "schema_version": 1,
        "producer": {
            "implementation": producer.producer.implementation,
            "version": producer.producer.version,
            "revision": producer.producer.revision,
        },
        "configuration_sha256": producer.configuration.sha256.value,
        "model": {
            "name": DA3_MODEL.name,
            "version": DA3_MODEL.version,
            "revision": DA3_MODEL.revision,
        },
        "checkpoint": {
            "identifier": DA3_CHECKPOINT.identifier,
            "sha256": DA3_CHECKPOINT.sha256.value,
        },
        "hardware_runtime_sha256": hardware_runtime.sha256.value,
        "inputs": [
            {
                "observation_id": image.observation_id.value,
                "artifact_key_sha256": image.artifact_key_sha256.value,
            }
            for image in images
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def inspect_da3_reference_environment() -> Da3EnvironmentIdentity:
    versions: list[tuple[str, str]] = []
    for distribution, expected in DA3_REFERENCE_PACKAGE_VERSIONS:
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise Da3RuntimeError(
                f"DA3 reference package {distribution!r} is unavailable"
            ) from exc
        if actual != expected:
            raise Da3RuntimeError(
                f"DA3 reference package {distribution!r} must be exactly {expected!r}"
            )
        versions.append((distribution, actual))
    return Da3EnvironmentIdentity(
        source_revision=DA3_SOURCE_REVISION,
        python_version=sys.version.split()[0],
        package_versions=tuple(versions),
        device="cpu",
        precision="float32",
    )


@contextmanager
def _isolated_da3_import(source_root: Path):
    package_parent = source_root / "src"
    preexisting = tuple(
        name for name in sys.modules if name == "depth_anything_3" or name.startswith("depth_anything_3.")
    )
    if preexisting:
        raise Da3RuntimeError("DA3 modules were already imported before source audit")
    sys.path.insert(0, str(package_parent))
    try:
        yield
    finally:
        if sys.path and sys.path[0] == str(package_parent):
            sys.path.pop(0)
        for name in tuple(sys.modules):
            if name == "depth_anything_3" or name.startswith("depth_anything_3."):
                sys.modules.pop(name, None)


def _tuple_matrix(value: Any, rows: int, cols: int, context: str) -> tuple[tuple[float, ...], ...]:
    try:
        shape = tuple(value.shape)
    except Exception as exc:
        raise Da3RuntimeError(f"{context} must expose an array shape") from exc
    if shape != (rows, cols):
        raise Da3RuntimeError(f"{context} has invalid shape {shape!r}")
    result: list[tuple[float, ...]] = []
    for row in range(rows):
        values: list[float] = []
        for column in range(cols):
            numeric = float(value[row, column])
            if not math.isfinite(numeric):
                raise Da3RuntimeError(f"{context} must contain finite values")
            values.append(numeric)
        result.append(tuple(values))
    return tuple(result)


class LocalDa3ReferenceRuntime:
    """Exact local CPU/float32 DA3-BASE reference execution.

    All learned packages and DA3 source are imported lazily only after local source,
    checkpoint and decoded input verification has completed.
    """

    def infer(
        self,
        *,
        source_root: Path,
        checkpoint_path: Path,
        images: tuple[_VerifiedRgbImage, ...],
        config: Da3ReferenceConfig,
    ) -> tuple[Da3EnvironmentIdentity, tuple[Da3BaseObservationPrediction, ...]]:
        environment = inspect_da3_reference_environment()

        with _isolated_da3_import(source_root):
            try:
                np = importlib.import_module("numpy")
                torch = importlib.import_module("torch")
                safetensors_torch = importlib.import_module("safetensors.torch")
                cfg = importlib.import_module("depth_anything_3.cfg")
                input_module = importlib.import_module(
                    "depth_anything_3.utils.io.input_processor"
                )
                output_module = importlib.import_module(
                    "depth_anything_3.utils.io.output_processor"
                )
            except Exception as exc:
                raise Da3RuntimeError("failed to import exact local DA3 reference runtime") from exc

            if torch.cuda.is_available():
                # The reference identity is deliberately CPU-only; a machine having CUDA does
                # not change execution because tensors/model stay on CPU.
                pass
            torch.manual_seed(config.random_seed)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)

            config_path = source_root / "src" / "depth_anything_3" / "configs" / "da3-base.yaml"
            try:
                model_config = cfg.load_config(str(config_path))
                model = cfg.create_object(model_config)
            except Exception as exc:
                raise Da3RuntimeError("failed to construct reviewed DA3-BASE model") from exc

            try:
                state = safetensors_torch.load_file(str(checkpoint_path), device="cpu")
            except Exception as exc:
                raise Da3RuntimeError("failed to load DA3-BASE safetensors checkpoint") from exc
            if not isinstance(state, dict) or not state:
                raise Da3RuntimeError("DA3-BASE safetensors checkpoint must be a non-empty mapping")
            keys = tuple(state)
            if any(not isinstance(key, str) or not key for key in keys):
                raise Da3RuntimeError("DA3-BASE checkpoint keys must be non-empty strings")
            prefixed = tuple(key.startswith("model.") for key in keys)
            if any(prefixed) and not all(prefixed):
                raise Da3RuntimeError("DA3-BASE checkpoint has mixed model. prefixes")
            if all(prefixed):
                state = {key[len("model.") :]: value for key, value in state.items()}
            try:
                model.load_state_dict(state, strict=True)
                model.to(device="cpu")
                model.eval()
            except Exception as exc:
                raise Da3RuntimeError("DA3-BASE checkpoint is incompatible with reviewed model") from exc

            arrays = []
            for image in images:
                array = np.frombuffer(image.rgb8, dtype=np.uint8).reshape(
                    image.height_px,
                    image.width_px,
                    3,
                )
                arrays.append(array.copy())

            processor = input_module.InputProcessor()
            try:
                batch, _, _ = processor(
                    image=arrays,
                    extrinsics=None,
                    intrinsics=None,
                    process_res=config.process_res,
                    process_res_method=config.process_res_method,
                    num_workers=config.num_workers,
                    print_progress=False,
                    sequential=True,
                )
            except Exception as exc:
                raise Da3RuntimeError("DA3 input preprocessing failed") from exc

            if batch.ndim != 5 or batch.shape[0] != 1 or batch.shape[1] != len(images):
                raise Da3RuntimeError("DA3 processed batch shape does not match inputs")
            batch = batch.to(device="cpu", dtype=torch.float32)

            try:
                with torch.inference_mode():
                    raw = model(
                        batch,
                        None,
                        None,
                        export_feat_layers=[],
                        infer_gs=config.infer_gs,
                        use_ray_pose=config.use_ray_pose,
                        ref_view_strategy=config.ref_view_strategy,
                    )
                prediction = output_module.OutputProcessor()(raw)
            except Exception as exc:
                raise Da3RuntimeError("DA3-BASE reference inference failed") from exc

            if getattr(prediction, "is_metric", 0) not in (0, False):
                raise Da3RuntimeError("DA3-BASE unexpectedly reported metric output")
            if getattr(prediction, "scale_factor", None) is not None:
                raise Da3RuntimeError("DA3-BASE unexpectedly reported a metric scale factor")
            if prediction.extrinsics is None or prediction.intrinsics is None:
                raise Da3RuntimeError("DA3-BASE prediction is missing camera parameters")
            if len(prediction.depth) != len(images):
                raise Da3RuntimeError("DA3-BASE depth count does not match inputs")
            if len(prediction.extrinsics) != len(images) or len(prediction.intrinsics) != len(images):
                raise Da3RuntimeError("DA3-BASE camera count does not match inputs")

            predictions: list[Da3BaseObservationPrediction] = []
            for index, image in enumerate(images):
                depth = prediction.depth[index]
                if getattr(depth, "ndim", None) != 2:
                    raise Da3RuntimeError("DA3-BASE depth must be one HxW raster per observation")
                height_px, width_px = int(depth.shape[0]), int(depth.shape[1])
                flat_depth: list[float] = []
                for raw_value in depth.reshape(-1):
                    numeric = float(raw_value)
                    if not math.isfinite(numeric) or numeric <= 0.0:
                        raise Da3RuntimeError(
                            "DA3-BASE relative depth must be finite and strictly positive"
                        )
                    flat_depth.append(numeric)

                extrinsic = _tuple_matrix(
                    prediction.extrinsics[index],
                    int(prediction.extrinsics[index].shape[0]),
                    4,
                    "DA3 extrinsic",
                )
                if len(extrinsic) not in (3, 4):
                    raise Da3RuntimeError("DA3 extrinsic must be 3x4 or 4x4")
                if len(extrinsic) == 4 and extrinsic[3] != (0.0, 0.0, 0.0, 1.0):
                    raise Da3RuntimeError("DA3 homogeneous extrinsic bottom row is invalid")
                rotation = (
                    (extrinsic[0][0], extrinsic[0][1], extrinsic[0][2]),
                    (extrinsic[1][0], extrinsic[1][1], extrinsic[1][2]),
                    (extrinsic[2][0], extrinsic[2][1], extrinsic[2][2]),
                )
                translation = (
                    extrinsic[0][3],
                    extrinsic[1][3],
                    extrinsic[2][3],
                )
                intrinsic_matrix = _tuple_matrix(
                    prediction.intrinsics[index],
                    3,
                    3,
                    "DA3 intrinsic",
                )
                intrinsics = (
                    (intrinsic_matrix[0][0], intrinsic_matrix[0][1], intrinsic_matrix[0][2]),
                    (intrinsic_matrix[1][0], intrinsic_matrix[1][1], intrinsic_matrix[1][2]),
                    (intrinsic_matrix[2][0], intrinsic_matrix[2][1], intrinsic_matrix[2][2]),
                )
                predictions.append(
                    Da3BaseObservationPrediction(
                        observation_id=image.observation_id,
                        dimensions=ImageDimensions(
                            width_px=width_px,
                            height_px=height_px,
                        ),
                        world_to_camera_rotation=rotation,
                        world_to_camera_translation=translation,
                        intrinsics=intrinsics,
                        depth_values=tuple(flat_depth),
                        validity=(True,) * (width_px * height_px),
                        # DA3 uses exp(x)+1 confidence, not a calibrated [0,1]
                        # probability. Do not invent a normalization here.
                        confidence=None,
                    )
                )

        return environment, tuple(predictions)


def execute_da3_base_preview(
    request: Da3ExecutionRequest,
    *,
    runtime: Da3Runtime | None = None,
) -> Da3ExecutionResult:
    if not isinstance(request, Da3ExecutionRequest):
        raise TypeError("request must be Da3ExecutionRequest")
    source_root = _verify_source_root(request.source_root)
    checkpoint_path = _verify_checkpoint(request.checkpoint_path)
    images = tuple(_verified_rgb_image(item) for item in request.inputs)

    producer = _producer(request.config)
    normalization_identity = _normalization_identity(
        producer=producer,
        images=images,
        hardware_runtime=request.hardware_runtime,
    )
    selected_runtime = runtime if runtime is not None else LocalDa3ReferenceRuntime()
    environment, predictions = selected_runtime.infer(
        source_root=source_root,
        checkpoint_path=checkpoint_path,
        images=images,
        config=request.config,
    )
    if not isinstance(environment, Da3EnvironmentIdentity):
        raise Da3RuntimeError("DA3 runtime returned an invalid environment identity")
    if not isinstance(predictions, tuple) or len(predictions) != len(images):
        raise Da3RuntimeError("DA3 runtime prediction count does not match inputs")
    if any(not isinstance(item, Da3BaseObservationPrediction) for item in predictions):
        raise Da3RuntimeError("DA3 runtime returned invalid candidate-native predictions")
    expected_ids = tuple(image.observation_id for image in images)
    actual_ids = tuple(item.observation_id for item in predictions)
    if actual_ids != expected_ids:
        raise Da3RuntimeError("DA3 runtime prediction observations do not match inputs")

    geometry = normalize_da3_base_preview(
        predictions,
        normalization_identity=normalization_identity,
    )
    return Da3ExecutionResult(
        producer=producer,
        model=DA3_MODEL,
        checkpoint=DA3_CHECKPOINT,
        environment=environment,
        hardware_runtime=request.hardware_runtime,
        normalization_identity=normalization_identity,
        geometry=geometry,
    )
