from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, cast

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import CameraSolution
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_canonical_geometry import (
    CanonicalColmapSparseModel,
    _verified_native_model_path,
    canonicalize_colmap_sparse_model,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_environment import (
    SUPPORTED_COLMAP_VERSION,
    SUPPORTED_PYCOLMAP_VERSION,
    ColmapEnvironmentError,
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import ColmapFeatureExtractionResult
from wre.reconstruction.colmap_geometry_refinement import (
    COLMAP_NATIVE_SPARSE_MODEL_KIND,
    colmap_native_sparse_model_artifact_ref,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
)
from wre.reconstruction.dense_depth import (
    DENSE_DEPTH_ARTIFACT_KIND,
    DenseDepthArtifact,
)
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

COLMAP_PATCH_MATCH_DENSE_DEPTH_ADAPTER_ID = "colmap.patch_match_dense_depth"
COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_IMPLEMENTATION = "pycolmap.patch_match_stereo"
COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_VERSION = SUPPORTED_PYCOLMAP_VERSION
COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION = (
    "be5e29168d4aff238409d60424812df66aac919f"
)
COLMAP_PATCH_MATCH_DENSE_DEPTH_DEPENDENCY_REF = "colmap_mvs_cuda12_4_2_0"
COLMAP_PATCH_MATCH_DENSE_DEPTH_SHIPPING_STATUS = "experimental"

COLMAP_MVS_IMAGE_SET_KIND = ArtifactKind("evidence.colmap_mvs_image_set")
COLMAP_MVS_ENVIRONMENT_KIND = ArtifactKind("environment.colmap_mvs_cuda")
COLMAP_CAMERA_Z_CONVENTION = DepthValueConventionName("camera-z")


class ColmapDenseDepthError(RuntimeError):
    """Raised when the exact COLMAP PatchMatch dense-depth donor fails closed."""


class ColmapDenseDepthEnvironmentError(ColmapDenseDepthError):
    """Raised when the active PyCOLMAP environment cannot run the reviewed donor."""


@dataclass(frozen=True, slots=True)
class ColmapPatchMatchDenseDepthConfig:
    """Exact V2L15.2 COLMAP PatchMatch donor configuration."""

    schema_version: int = 1
    gpu_index: str = "0"
    max_image_size: int = -1
    num_threads: int = 1
    depth_min: float = -1.0
    depth_max: float = -1.0
    window_radius: int = 5
    window_step: int = 1
    sigma_spatial: float = -1.0
    sigma_color: float = 0.2
    num_samples: int = 15
    ncc_sigma: float = 0.6
    min_triangulation_angle: float = 1.0
    incident_angle_sigma: float = 0.9
    num_iterations: int = 5
    geom_consistency: bool = True
    geom_consistency_regularizer: float = 0.3
    geom_consistency_max_cost: float = 3.0
    filter: bool = True
    filter_min_ncc: float = 0.1
    filter_min_triangulation_angle: float = 3.0
    filter_min_num_consistent: int = 2
    filter_geom_consistency_max_cost: float = 1.0
    cache_size: float = 32.0
    allow_missing_files: bool = False
    write_consistency_graph: bool = False
    undistort_blank_pixels: float = 0.0
    undistort_min_scale: float = 0.2
    undistort_max_scale: float = 2.0
    undistort_max_image_size: int = -1
    undistort_num_threads: int = 1
    num_patch_match_src_images: int = -1
    jpeg_quality: int = -1

    def __post_init__(self) -> None:
        exact = ColmapPatchMatchDenseDepthConfig.__dataclass_fields__
        if self.schema_version != 1:
            raise ValueError("COLMAP dense-depth config schema_version must be 1")
        if self.gpu_index != "0":
            raise ValueError("V2L15.2 requires the single explicit accelerator index '0'")
        if self.max_image_size != -1 or self.undistort_max_image_size != -1:
            raise ValueError("V2L15.2 requires the no-resize max_image_size=-1 path")
        if self.num_threads != 1 or self.undistort_num_threads != 1:
            raise ValueError("V2L15.2 requires one host thread for reviewed donor execution")
        if self.num_patch_match_src_images != -1:
            raise ValueError("V2L15.2 requires COLMAP's complete source-image selection")
        if self.jpeg_quality != -1:
            raise ValueError("V2L15.2 must not recompress JPEG inputs")
        if not self.geom_consistency:
            raise ValueError("V2L15.2 baseline requires geometric-consistency PatchMatch")
        if not self.filter:
            raise ValueError("V2L15.2 baseline requires PatchMatch filtering")
        if self.allow_missing_files:
            raise ValueError("V2L15.2 must fail closed on missing PatchMatch dependencies")
        if self.write_consistency_graph:
            raise ValueError("V2L15.2 does not publish consistency-graph evidence")

        expected_values: dict[str, object] = {
            "depth_min": -1.0,
            "depth_max": -1.0,
            "window_radius": 5,
            "window_step": 1,
            "sigma_spatial": -1.0,
            "sigma_color": 0.2,
            "num_samples": 15,
            "ncc_sigma": 0.6,
            "min_triangulation_angle": 1.0,
            "incident_angle_sigma": 0.9,
            "num_iterations": 5,
            "geom_consistency_regularizer": 0.3,
            "geom_consistency_max_cost": 3.0,
            "filter_min_ncc": 0.1,
            "filter_min_triangulation_angle": 3.0,
            "filter_min_num_consistent": 2,
            "filter_geom_consistency_max_cost": 1.0,
            "cache_size": 32.0,
            "undistort_blank_pixels": 0.0,
            "undistort_min_scale": 0.2,
            "undistort_max_scale": 2.0,
        }
        for name, expected_value in expected_values.items():
            if getattr(self, name) != expected_value:
                raise ValueError(
                    f"V2L15.2 reviewed baseline requires {name}={expected_value!r}"
                )

        if set(exact) != set(self.canonical_document()):
            raise RuntimeError("COLMAP dense-depth config canonical document is incomplete")

    def canonical_document(self) -> dict[str, object]:
        return {
            "allow_missing_files": self.allow_missing_files,
            "cache_size": self.cache_size,
            "depth_max": self.depth_max,
            "depth_min": self.depth_min,
            "filter": self.filter,
            "filter_geom_consistency_max_cost": self.filter_geom_consistency_max_cost,
            "filter_min_ncc": self.filter_min_ncc,
            "filter_min_num_consistent": self.filter_min_num_consistent,
            "filter_min_triangulation_angle": self.filter_min_triangulation_angle,
            "geom_consistency": self.geom_consistency,
            "geom_consistency_max_cost": self.geom_consistency_max_cost,
            "geom_consistency_regularizer": self.geom_consistency_regularizer,
            "gpu_index": self.gpu_index,
            "incident_angle_sigma": self.incident_angle_sigma,
            "jpeg_quality": self.jpeg_quality,
            "max_image_size": self.max_image_size,
            "min_triangulation_angle": self.min_triangulation_angle,
            "ncc_sigma": self.ncc_sigma,
            "num_iterations": self.num_iterations,
            "num_patch_match_src_images": self.num_patch_match_src_images,
            "num_samples": self.num_samples,
            "num_threads": self.num_threads,
            "schema_version": self.schema_version,
            "sigma_color": self.sigma_color,
            "sigma_spatial": self.sigma_spatial,
            "undistort_blank_pixels": self.undistort_blank_pixels,
            "undistort_max_image_size": self.undistort_max_image_size,
            "undistort_max_scale": self.undistort_max_scale,
            "undistort_min_scale": self.undistort_min_scale,
            "undistort_num_threads": self.undistort_num_threads,
            "window_radius": self.window_radius,
            "window_step": self.window_step,
            "write_consistency_graph": self.write_consistency_graph,
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
class ColmapDenseDepthSource:
    """Exact sparse geometry and immutable source-image evidence for one MVS donor run."""

    model_root: Path
    model_artifact: ColmapSparseModelArtifact
    source_geometry: GeometrySolutionCandidate
    features: ColmapFeatureExtractionResult
    image_root: Path
    images: tuple[ColmapReconstructionInput, ...]
    expected_environment: ColmapEnvironmentIdentity
    artifact_ref: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.model_root, Path):
            raise TypeError("colmap_dense_depth_source.model_root must be pathlib.Path")
        if not isinstance(self.model_artifact, ColmapSparseModelArtifact):
            raise TypeError(
                "colmap_dense_depth_source.model_artifact must be ColmapSparseModelArtifact"
            )
        if not isinstance(self.source_geometry, GeometrySolutionCandidate):
            raise TypeError(
                "colmap_dense_depth_source.source_geometry must be GeometrySolutionCandidate"
            )
        if not isinstance(self.features, ColmapFeatureExtractionResult):
            raise TypeError(
                "colmap_dense_depth_source.features must be ColmapFeatureExtractionResult"
            )
        if not isinstance(self.image_root, Path):
            raise TypeError("colmap_dense_depth_source.image_root must be pathlib.Path")
        if not isinstance(self.images, tuple) or not self.images:
            raise ValueError(
                "colmap_dense_depth_source.images must be a non-empty immutable tuple"
            )
        if any(not isinstance(item, ColmapReconstructionInput) for item in self.images):
            raise TypeError(
                "colmap_dense_depth_source.images members must be ColmapReconstructionInput"
            )
        if not isinstance(self.expected_environment, ColmapEnvironmentIdentity):
            raise TypeError(
                "colmap_dense_depth_source.expected_environment must be "
                "ColmapEnvironmentIdentity"
            )
        if not isinstance(self.artifact_ref, ArtifactRef):
            raise TypeError("colmap_dense_depth_source.artifact_ref must be ArtifactRef")

        expected_ref = colmap_native_sparse_model_artifact_ref(self.model_artifact)
        if self.artifact_ref != expected_ref:
            raise ValueError(
                "COLMAP dense-depth source ArtifactRef must identify the exact audited "
                "native sparse model"
            )
        _validate_expected_donor_environment(self.expected_environment)

        image_ids = tuple(item.observation.observation_id.value for item in self.images)
        if image_ids != tuple(sorted(image_ids)) or len(image_ids) != len(set(image_ids)):
            raise ValueError(
                "COLMAP dense-depth images must be unique and canonically ordered by "
                "ObservationId"
            )
        if tuple(item.observation.observation_id for item in self.images) != (
            self.features.provenance.source_observation_ids
        ):
            raise ValueError(
                "COLMAP dense-depth images must exactly match feature-extraction provenance"
            )
        feature_by_id = {item.observation_id: item for item in self.features.images}
        for image in self.images:
            feature = feature_by_id.get(image.observation.observation_id)
            if feature is None or feature.image_name != image.image_name:
                raise ValueError(
                    "COLMAP dense-depth image names must exactly match feature mapping"
                )
        if self.source_geometry.depth_fields:
            raise ValueError(
                "COLMAP dense-depth donor requires sparse COLMAP source geometry without "
                "pre-existing depth fields"
            )


@dataclass(frozen=True, slots=True)
class _VerifiedSourceImage:
    observation_id: ObservationId
    image_name: str
    source_path: Path
    sha256: Sha256Digest
    byte_length: int


def _validate_expected_donor_environment(environment: ColmapEnvironmentIdentity) -> None:
    if environment.pycolmap_version != SUPPORTED_PYCOLMAP_VERSION:
        raise ValueError("COLMAP dense-depth donor requires PyCOLMAP 4.2.0")
    if environment.colmap_version != SUPPORTED_COLMAP_VERSION:
        raise ValueError("COLMAP dense-depth donor requires COLMAP 4.2.0")
    if not environment.upstream_has_cuda:
        raise ValueError("COLMAP dense-depth donor environment must advertise CUDA support")
    if COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION not in environment.colmap_build:
        raise ValueError(
            "COLMAP dense-depth donor build must identify the exact COLMAP 4.2.0 source commit"
        )


def _load_pycolmap() -> Any:
    try:
        return cast(Any, importlib.import_module("pycolmap"))
    except (ImportError, OSError, RuntimeError) as exc:
        raise ColmapDenseDepthEnvironmentError(
            "PyCOLMAP is unavailable; install the exact external "
            "pycolmap-cuda12==4.2.0 donor environment"
        ) from exc


def inspect_colmap_dense_depth_environment(module: object | None = None) -> ColmapEnvironmentIdentity:
    pycolmap = cast(Any, module) if module is not None else _load_pycolmap()
    try:
        environment = inspect_colmap_environment(pycolmap)
    except ColmapEnvironmentError as exc:
        raise ColmapDenseDepthEnvironmentError(str(exc)) from exc
    if not environment.upstream_has_cuda:
        raise ColmapDenseDepthEnvironmentError(
            "COLMAP PatchMatch donor requires pycolmap.has_cuda == true"
        )
    if COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION not in environment.colmap_build:
        raise ColmapDenseDepthEnvironmentError(
            "COLMAP PatchMatch donor build does not identify the exact 4.2.0 source commit"
        )
    if not callable(getattr(pycolmap, "undistort_images", None)):
        raise ColmapDenseDepthEnvironmentError("pycolmap.undistort_images must be callable")
    if not callable(getattr(pycolmap, "patch_match_stereo", None)):
        raise ColmapDenseDepthEnvironmentError("pycolmap.patch_match_stereo must be callable")
    depth_map_type = getattr(pycolmap, "DepthMap", None)
    if not callable(depth_map_type):
        raise ColmapDenseDepthEnvironmentError("pycolmap.DepthMap must be available")
    if not callable(getattr(depth_map_type, "read", None)):
        raise ColmapDenseDepthEnvironmentError("pycolmap.DepthMap.read must be callable")
    if not callable(getattr(depth_map_type, "to_array", None)):
        raise ColmapDenseDepthEnvironmentError("pycolmap.DepthMap.to_array must be callable")
    return environment


def _safe_image_path(root: Path, image_name: str) -> Path:
    relative = PurePosixPath(image_name)
    if relative.is_absolute() or len(relative.parts) != 1 or "\\" in image_name:
        raise ColmapDenseDepthError(
            "COLMAP dense-depth image_name must be one safe relative filename"
        )
    if relative.name in {".", ".."}:
        raise ColmapDenseDepthError(
            "COLMAP dense-depth image_name must be one safe relative filename"
        )
    candidate = root / image_name
    if candidate.is_symlink():
        raise ColmapDenseDepthError("COLMAP dense-depth source image must not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ColmapDenseDepthError("COLMAP dense-depth source image is missing") from exc
    if resolved.parent != root or not resolved.is_file():
        raise ColmapDenseDepthError(
            "COLMAP dense-depth source image must be one regular file inside image_root"
        )
    return resolved


def _verified_source_images(source: ColmapDenseDepthSource) -> tuple[_VerifiedSourceImage, ...]:
    image_root_candidate = source.image_root.expanduser()
    if image_root_candidate.is_symlink():
        raise ColmapDenseDepthError("COLMAP dense-depth image_root must not be a symlink")
    try:
        image_root = image_root_candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ColmapDenseDepthError("COLMAP dense-depth image_root is missing") from exc
    if not image_root.is_dir():
        raise ColmapDenseDepthError("COLMAP dense-depth image_root must be a directory")

    verified: list[_VerifiedSourceImage] = []
    for item in source.images:
        expected_path = _safe_image_path(image_root, item.image_name)
        try:
            supplied_path = item.source_path.expanduser().resolve(strict=True)
        except FileNotFoundError as exc:
            raise ColmapDenseDepthError("COLMAP dense-depth bound source image is missing") from exc
        if supplied_path != expected_path:
            raise ColmapDenseDepthError(
                "COLMAP dense-depth source_path must resolve to image_root/image_name"
            )
        digest = hash_file_content(expected_path)
        if (
            digest.sha256 != item.observation.asset.sha256
            or digest.byte_length != item.observation.asset.byte_length
        ):
            raise ColmapDenseDepthError(
                "COLMAP dense-depth source image bytes do not match persisted observation"
            )
        verified.append(
            _VerifiedSourceImage(
                observation_id=item.observation.observation_id,
                image_name=item.image_name,
                source_path=expected_path,
                sha256=digest.sha256,
                byte_length=digest.byte_length,
            )
        )
    return tuple(verified)


def _image_manifest_sha256(images: tuple[_VerifiedSourceImage, ...]) -> Sha256Digest:
    payload = {
        "schema_version": 1,
        "images": [
            {
                "observation_id": item.observation_id.value,
                "image_name": item.image_name,
                "sha256": item.sha256.value,
                "byte_length": item.byte_length,
            }
            for item in images
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _image_evidence_ref(images: tuple[_VerifiedSourceImage, ...]) -> ArtifactRef:
    identity = _image_manifest_sha256(images)
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:colmap-mvs-images:{identity.value}"),
        artifact_kind=COLMAP_MVS_IMAGE_SET_KIND,
    )


def _environment_sha256(environment: ColmapEnvironmentIdentity) -> Sha256Digest:
    payload = {
        "schema_version": 1,
        "source_revision": COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION,
        "pycolmap_version": environment.pycolmap_version,
        "colmap_version": environment.colmap_version,
        "colmap_build": environment.colmap_build,
        "ceres_version": environment.ceres_version,
        "upstream_has_cuda": environment.upstream_has_cuda,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _environment_evidence_ref(environment: ColmapEnvironmentIdentity) -> ArtifactRef:
    identity = _environment_sha256(environment)
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:colmap-mvs-env:{identity.value}"),
        artifact_kind=COLMAP_MVS_ENVIRONMENT_KIND,
    )


def _validate_source_geometry_matches(
    source_geometry: GeometrySolutionCandidate,
    canonical_source: CanonicalColmapSparseModel,
) -> None:
    if source_geometry.geometry_solution != canonical_source.geometry_solution:
        raise ColmapDenseDepthError(
            "dense-depth source GeometrySolution does not match audited native COLMAP model"
        )
    if source_geometry.camera_solutions != canonical_source.camera_solutions:
        raise ColmapDenseDepthError(
            "dense-depth source cameras do not match audited native COLMAP model"
        )
    if source_geometry.depth_fields:
        raise ColmapDenseDepthError(
            "dense-depth source must not contain pre-existing DepthField values"
        )
    if source_geometry.point_maps != (canonical_source.point_map,):
        raise ColmapDenseDepthError(
            "dense-depth source PointMap does not match audited native COLMAP model"
        )


def _registered_image_names(source: ColmapDenseDepthSource) -> tuple[str, ...]:
    feature_by_observation = {
        item.observation_id: item.image_name for item in source.features.images
    }
    names: list[str] = []
    for camera in source.source_geometry.camera_solutions:
        try:
            names.append(feature_by_observation[camera.observation_id])
        except KeyError as exc:
            raise ColmapDenseDepthError(
                "source CameraSolution has no exact COLMAP feature image mapping"
            ) from exc
    if len(names) != len(set(names)):
        raise ColmapDenseDepthError("source CameraSolutions map to duplicate COLMAP image names")
    return tuple(sorted(names))


def _validate_source_cameras(source_geometry: GeometrySolutionCandidate) -> None:
    for camera in source_geometry.camera_solutions:
        if camera.projection_model.value != "pinhole":
            raise ColmapDenseDepthError(
                "V2L15.2 supports only already-undistorted COLMAP PINHOLE cameras"
            )
        if len(camera.intrinsic_parameters) != 4:
            raise ColmapDenseDepthError(
                "COLMAP PINHOLE CameraSolution must contain fx fy cx cy intrinsics"
            )


def _finite_float(value: object, context: str) -> float:
    if isinstance(value, bool):
        raise ColmapDenseDepthError(f"{context} must be finite")
    try:
        normalized = float(cast(Any, value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ColmapDenseDepthError(f"{context} must be finite") from exc
    if not math.isfinite(normalized):
        raise ColmapDenseDepthError(f"{context} must be finite")
    return normalized


def _vector3(value: object, context: str) -> tuple[float, float, float]:
    try:
        items = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapDenseDepthError(f"{context} must contain three finite values") from exc
    if len(items) != 3:
        raise ColmapDenseDepthError(f"{context} must contain three finite values")
    return (
        _finite_float(items[0], context),
        _finite_float(items[1], context),
        _finite_float(items[2], context),
    )


def _rotation3(value: object) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    try:
        rows = tuple(cast(Any, value))
    except TypeError as exc:
        raise ColmapDenseDepthError("workspace rotation must be a 3x3 matrix") from exc
    if len(rows) != 3:
        raise ColmapDenseDepthError("workspace rotation must be a 3x3 matrix")
    return (
        _vector3(rows[0], "workspace rotation"),
        _vector3(rows[1], "workspace rotation"),
        _vector3(rows[2], "workspace rotation"),
    )


def _validate_workspace_linkage(
    pycolmap: Any,
    workspace_root: Path,
    source: ColmapDenseDepthSource,
) -> None:
    sparse_path = workspace_root / "sparse"
    try:
        reconstruction = pycolmap.Reconstruction(sparse_path)
    except Exception as exc:
        raise ColmapDenseDepthError("cannot read private COLMAP MVS sparse workspace") from exc
    if not bool(reconstruction.is_valid()):
        raise ColmapDenseDepthError("private COLMAP MVS sparse workspace is invalid")

    source_by_observation = {
        camera.observation_id: camera for camera in source.source_geometry.camera_solutions
    }
    observation_by_name = {
        item.image_name: item.observation_id for item in source.features.images
    }
    expected_names = {
        next(
            feature.image_name
            for feature in source.features.images
            if feature.observation_id == camera.observation_id
        )
        for camera in source.source_geometry.camera_solutions
    }

    actual_names: set[str] = set()
    try:
        image_ids = sorted(int(value) for value in reconstruction.reg_image_ids())
    except Exception as exc:
        raise ColmapDenseDepthError("workspace registered image IDs are not readable") from exc

    if len(image_ids) != len(source.source_geometry.camera_solutions):
        raise ColmapDenseDepthError(
            "workspace registered camera count changed after no-resize undistortion"
        )

    for image_id in image_ids:
        try:
            image = reconstruction.image(image_id)
            image_name = str(image.name)
            observation_id = observation_by_name[image_name]
            expected_camera = source_by_observation[observation_id]
            camera = reconstruction.camera(int(image.camera_id))
            model_name = str(camera.model_name).lower()
            width = int(camera.width)
            height = int(camera.height)
            params = tuple(
                _finite_float(value, "workspace camera parameter") for value in camera.params
            )
            transform = image.cam_from_world()
            rotation = _rotation3(transform.rotation.matrix())
            translation = _vector3(transform.translation, "workspace translation")
        except ColmapDenseDepthError:
            raise
        except Exception as exc:
            raise ColmapDenseDepthError(
                "private COLMAP MVS workspace camera is not exactly linkable"
            ) from exc

        actual_names.add(image_name)
        if model_name != expected_camera.projection_model.value:
            raise ColmapDenseDepthError(
                "workspace projection model changed after no-resize undistortion"
            )
        if (
            width != expected_camera.dimensions.width_px
            or height != expected_camera.dimensions.height_px
        ):
            raise ColmapDenseDepthError(
                "workspace camera dimensions changed after no-resize undistortion"
            )
        if params != expected_camera.intrinsic_parameters:
            raise ColmapDenseDepthError(
                "workspace camera intrinsics changed after no-resize undistortion"
            )
        if (
            rotation != expected_camera.rotation_matrix
            or translation != expected_camera.translation_xyz
        ):
            raise ColmapDenseDepthError(
                "workspace camera pose changed after no-resize undistortion"
            )

    if actual_names != expected_names:
        raise ColmapDenseDepthError(
            "workspace image-name membership changed after no-resize undistortion"
        )


def _configure_undistortion(pycolmap: Any, config: ColmapPatchMatchDenseDepthConfig) -> Any:
    options = pycolmap.UndistortCameraOptions()
    options.blank_pixels = config.undistort_blank_pixels
    options.min_scale = config.undistort_min_scale
    options.max_scale = config.undistort_max_scale
    options.max_image_size = config.undistort_max_image_size
    options.roi_min_x = 0.0
    options.roi_min_y = 0.0
    options.roi_max_x = 1.0
    options.roi_max_y = 1.0
    return options


def _configure_patch_match(pycolmap: Any, config: ColmapPatchMatchDenseDepthConfig) -> Any:
    options = pycolmap.PatchMatchOptions()
    options.max_image_size = config.max_image_size
    options.gpu_index = config.gpu_index
    options.depth_min = config.depth_min
    options.depth_max = config.depth_max
    options.window_radius = config.window_radius
    options.window_step = config.window_step
    options.sigma_spatial = config.sigma_spatial
    options.sigma_color = config.sigma_color
    options.num_samples = config.num_samples
    options.ncc_sigma = config.ncc_sigma
    options.min_triangulation_angle = config.min_triangulation_angle
    options.incident_angle_sigma = config.incident_angle_sigma
    options.num_iterations = config.num_iterations
    options.geom_consistency = config.geom_consistency
    options.geom_consistency_regularizer = config.geom_consistency_regularizer
    options.geom_consistency_max_cost = config.geom_consistency_max_cost
    options.filter = config.filter
    options.filter_min_ncc = config.filter_min_ncc
    options.filter_min_triangulation_angle = config.filter_min_triangulation_angle
    options.filter_min_num_consistent = config.filter_min_num_consistent
    options.filter_geom_consistency_max_cost = config.filter_geom_consistency_max_cost
    options.cache_size = config.cache_size
    options.allow_missing_files = config.allow_missing_files
    options.write_consistency_graph = config.write_consistency_graph
    options.num_threads = config.num_threads
    return options


def _copy_private_inputs(
    *,
    source: ColmapDenseDepthSource,
    source_model_path: Path,
    verified_images: tuple[_VerifiedSourceImage, ...],
    working_root: Path,
) -> tuple[Path, Path]:
    copied_model_root = working_root / "models"
    copied_model_root.mkdir()
    copied_model_path = copied_model_root / source.model_artifact.relative_path
    shutil.copytree(source_model_path, copied_model_path)
    _verified_native_model_path(copied_model_root, source.model_artifact)

    copied_image_root = working_root / "images"
    copied_image_root.mkdir()
    for image in verified_images:
        destination = copied_image_root / image.image_name
        shutil.copyfile(image.source_path, destination)
        digest = hash_file_content(destination)
        if digest.sha256 != image.sha256 or digest.byte_length != image.byte_length:
            raise ColmapDenseDepthError(
                "private MVS image copy does not match verified source bytes"
            )
    return copied_model_path, copied_image_root


def _validate_no_overlap(output_root: Path, source: ColmapDenseDepthSource) -> None:
    source_model_root = source.model_root.expanduser().resolve(strict=True)
    source_image_root = source.image_root.expanduser().resolve(strict=True)
    resolved_output = output_root.expanduser().resolve()
    for retained_root in (source_model_root, source_image_root):
        if (
            resolved_output == retained_root
            or resolved_output.is_relative_to(retained_root)
            or retained_root.is_relative_to(resolved_output)
        ):
            raise ValueError(
                "COLMAP dense-depth output_root must not overlap retained source roots"
            )


def _verify_retained_sources(
    source: ColmapDenseDepthSource,
    expected_images: tuple[_VerifiedSourceImage, ...],
) -> None:
    _verified_native_model_path(source.model_root, source.model_artifact)
    actual_images = _verified_source_images(source)
    if actual_images != expected_images:
        raise ColmapDenseDepthError(
            "retained COLMAP dense-depth source images changed during donor execution"
        )


def _depth_array_values(
    depth_map: Any,
    *,
    expected_width: int,
    expected_height: int,
) -> tuple[tuple[float, ...], tuple[bool, ...]]:
    try:
        array = depth_map.to_array()
        shape = tuple(int(value) for value in array.shape)
    except Exception as exc:
        raise ColmapDenseDepthError("PyCOLMAP DepthMap.to_array returned unreadable data") from exc
    if shape != (expected_height, expected_width):
        raise ColmapDenseDepthError(
            "COLMAP PatchMatch depth-map dimensions do not match source CameraSolution"
        )
    try:
        flattened = tuple(cast(Any, array).reshape(-1).tolist())
    except Exception as exc:
        raise ColmapDenseDepthError("COLMAP PatchMatch depth map cannot be flattened") from exc
    if len(flattened) != expected_width * expected_height:
        raise ColmapDenseDepthError("COLMAP PatchMatch depth-map pixel count is invalid")

    values: list[float] = []
    validity: list[bool] = []
    for raw in flattened:
        value = _finite_float(raw, "COLMAP PatchMatch depth")
        if value <= 0.0:
            values.append(0.0)
            validity.append(False)
        else:
            values.append(value)
            validity.append(True)
    return tuple(values), tuple(validity)


def _identity_sha256(document: dict[str, object]) -> Sha256Digest:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _producer_document(producer: ArtifactProducerIdentity) -> dict[str, object]:
    return {
        "implementation": producer.producer.implementation,
        "version": producer.producer.version,
        "revision": producer.producer.revision,
        "configuration_sha256": producer.configuration.sha256.value,
        "model": (
            None
            if producer.model is None
            else {
                "name": producer.model.name,
                "version": producer.model.version,
                "revision": producer.model.revision,
            }
        ),
        "checkpoint": (
            None
            if producer.checkpoint is None
            else {
                "identifier": producer.checkpoint.identifier,
                "sha256": producer.checkpoint.sha256.value,
            }
        ),
    }


def _normalization_identity(
    *,
    source: ColmapDenseDepthSource,
    verified_images: tuple[_VerifiedSourceImage, ...],
    environment: ColmapEnvironmentIdentity,
    config: ColmapPatchMatchDenseDepthConfig,
    hardware_runtime: HardwareRuntimeIdentity,
) -> Sha256Digest:
    return _identity_sha256(
        {
            "domain": "wre.colmap-patch-match-dense-depth",
            "schema_version": 1,
            "source_model_sha256": colmap_sparse_model_content_identity(
                source.model_artifact
            ).value,
            "source_geometry_id": source.source_geometry.geometry_solution_id.value,
            "source_producer": _producer_document(source.source_geometry.producer),
            "source_artifacts": [
                [item.artifact_id.value, item.artifact_kind.value]
                for item in source.source_geometry.source_artifacts
            ],
            "image_manifest_sha256": _image_manifest_sha256(verified_images).value,
            "environment_sha256": _environment_sha256(environment).value,
            "configuration_sha256": config.sha256.value,
            "hardware_runtime_sha256": hardware_runtime.sha256.value,
        }
    )


def _canonical_source_artifacts(
    source: ColmapDenseDepthSource,
    verified_images: tuple[_VerifiedSourceImage, ...],
    environment: ColmapEnvironmentIdentity,
) -> tuple[ArtifactRef, ...]:
    references = (
        *source.source_geometry.source_artifacts,
        source.artifact_ref,
        _image_evidence_ref(verified_images),
        _environment_evidence_ref(environment),
    )
    by_identity = {
        (item.artifact_id.value, item.artifact_kind.value): item for item in references
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _read_depth_fields(
    *,
    pycolmap: Any,
    workspace_root: Path,
    source: ColmapDenseDepthSource,
    normalization_identity: Sha256Digest,
) -> tuple[tuple[DepthField, ...], tuple[dict[str, object], ...]]:
    suffix = "geometric" if True else "photometric"
    depth_root = workspace_root / "stereo" / "depth_maps"
    if not depth_root.is_dir() or depth_root.is_symlink():
        raise ColmapDenseDepthError("COLMAP PatchMatch depth-map directory is missing")

    feature_by_observation = {
        item.observation_id: item.image_name for item in source.features.images
    }
    expected_names = {
        feature_by_observation[camera.observation_id]
        for camera in source.source_geometry.camera_solutions
    }
    for path in depth_root.glob(f"*.{suffix}.bin"):
        name = path.name[: -len(f".{suffix}.bin")]
        if name not in expected_names:
            raise ColmapDenseDepthError(
                "COLMAP PatchMatch emitted a depth map for an unbound source image"
            )

    fields: list[DepthField] = []
    evidence: list[dict[str, object]] = []
    for camera in source.source_geometry.camera_solutions:
        image_name = feature_by_observation[camera.observation_id]
        path = depth_root / f"{image_name}.{suffix}.bin"
        if not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise ColmapDenseDepthError("COLMAP PatchMatch depth output must be a regular file")

        try:
            depth_map = pycolmap.DepthMap()
            depth_map.read(path)
        except Exception as exc:
            raise ColmapDenseDepthError("cannot read COLMAP PatchMatch depth map") from exc

        values, validity = _depth_array_values(
            depth_map,
            expected_width=camera.dimensions.width_px,
            expected_height=camera.dimensions.height_px,
        )
        digest = hash_file_content(path)
        field_identity = _identity_sha256(
            {
                "domain": "wre.colmap-patch-match-depth-field",
                "schema_version": 1,
                "normalization_sha256": normalization_identity.value,
                "camera_solution_id": camera.solution_id.value,
                "observation_id": camera.observation_id.value,
                "depth_map_sha256": digest.sha256.value,
                "depth_map_byte_length": digest.byte_length,
            }
        )
        fields.append(
            DepthField(
                depth_field_id=DepthFieldId(f"depth:colmap-pm:{field_identity.value}"),
                observation_id=camera.observation_id,
                camera_solution_id=camera.solution_id,
                dimensions=camera.dimensions,
                depth_value_convention=COLMAP_CAMERA_Z_CONVENTION,
                depth_values=values,
                validity=validity,
                confidence=None,
                metrics=MetricVector(observations=()),
            )
        )
        evidence.append(
            {
                "camera_solution_id": camera.solution_id.value,
                "observation_id": camera.observation_id.value,
                "image_name": image_name,
                "sha256": digest.sha256.value,
                "byte_length": digest.byte_length,
            }
        )

    fields.sort(key=lambda item: item.depth_field_id.value)
    evidence.sort(key=lambda item: cast(str, item["camera_solution_id"]))
    if not fields:
        raise ColmapDenseDepthError(
            "COLMAP PatchMatch produced no source-linked geometric depth maps"
        )
    return tuple(fields), tuple(evidence)


@dataclass(frozen=True, slots=True, kw_only=True)
class ColmapPatchMatchDenseDepthAdapter:
    """Exact accelerator-only COLMAP 4.2.0 PatchMatch dense-depth donor."""

    source: ColmapDenseDepthSource
    output_root: Path
    hardware_runtime: HardwareRuntimeIdentity
    config: ColmapPatchMatchDenseDepthConfig = field(
        default_factory=ColmapPatchMatchDenseDepthConfig
    )
    module: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, ColmapDenseDepthSource):
            raise TypeError("colmap_dense_depth.source must be ColmapDenseDepthSource")
        if not isinstance(self.output_root, Path):
            raise TypeError("colmap_dense_depth.output_root must be pathlib.Path")
        if not isinstance(self.hardware_runtime, HardwareRuntimeIdentity):
            raise TypeError(
                "colmap_dense_depth.hardware_runtime must be HardwareRuntimeIdentity"
            )
        if not isinstance(self.config, ColmapPatchMatchDenseDepthConfig):
            raise TypeError(
                "colmap_dense_depth.config must be ColmapPatchMatchDenseDepthConfig"
            )

    def derive(self) -> DenseDepthArtifact:
        source_model_path = _verified_native_model_path(
            self.source.model_root,
            self.source.model_artifact,
        )
        verified_images = _verified_source_images(self.source)
        _validate_source_cameras(self.source.source_geometry)

        pycolmap = cast(Any, self.module) if self.module is not None else _load_pycolmap()
        environment = inspect_colmap_dense_depth_environment(pycolmap)
        if environment != self.source.expected_environment:
            raise ColmapDenseDepthEnvironmentError(
                "active PyCOLMAP donor environment does not match bound exact environment"
            )

        canonical_source = canonicalize_colmap_sparse_model(
            output_path=self.source.model_root,
            model_artifact=self.source.model_artifact,
            features=self.source.features,
            expected_environment=environment,
            module=pycolmap,
        )
        _validate_source_geometry_matches(self.source.source_geometry, canonical_source)

        output_root = self.output_root.expanduser().resolve()
        _validate_no_overlap(output_root, self.source)
        if output_root.exists():
            raise ValueError("COLMAP dense-depth output_root must not already exist")
        output_root.parent.mkdir(parents=True, exist_ok=True)

        output_created = False
        try:
            output_root.mkdir()
            output_created = True
            with tempfile.TemporaryDirectory(
                prefix="wre-colmap-mvs-input-",
                dir=output_root.parent,
            ) as working_dir_name:
                working_root = Path(working_dir_name)
                copied_model_path, copied_image_root = _copy_private_inputs(
                    source=self.source,
                    source_model_path=source_model_path,
                    verified_images=verified_images,
                    working_root=working_root,
                )

                undistort_options = _configure_undistortion(pycolmap, self.config)
                copy_policy = pycolmap.FileCopyType.COPY
                try:
                    pycolmap.undistort_images(
                        output_root,
                        copied_model_path,
                        copied_image_root,
                        image_names=list(_registered_image_names(self.source)),
                        output_type="COLMAP",
                        copy_policy=copy_policy,
                        num_patch_match_src_images=self.config.num_patch_match_src_images,
                        undistort_options=undistort_options,
                        jpeg_quality=self.config.jpeg_quality,
                        num_threads=self.config.undistort_num_threads,
                    )
                except Exception as exc:
                    raise ColmapDenseDepthError(
                        "COLMAP no-resize MVS workspace preparation failed"
                    ) from exc

                _validate_workspace_linkage(pycolmap, output_root, self.source)

                patch_options = _configure_patch_match(pycolmap, self.config)
                try:
                    pycolmap.patch_match_stereo(
                        output_root,
                        workspace_format="COLMAP",
                        pmvs_option_name="option-all",
                        options=patch_options,
                        config_path="",
                    )
                except Exception as exc:
                    raise ColmapDenseDepthError("COLMAP PatchMatch stereo failed") from exc

            normalization_identity = _normalization_identity(
                source=self.source,
                verified_images=verified_images,
                environment=environment,
                config=self.config,
                hardware_runtime=self.hardware_runtime,
            )
            depth_fields, depth_evidence = _read_depth_fields(
                pycolmap=pycolmap,
                workspace_root=output_root,
                source=self.source,
                normalization_identity=normalization_identity,
            )
            producer = ArtifactProducerIdentity(
                producer=ProducerRef(
                    implementation=COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_IMPLEMENTATION,
                    version=COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_VERSION,
                    revision=environment.colmap_build,
                ),
                configuration=ConfigurationIdentity(sha256=self.config.sha256),
            )
            source_artifacts = _canonical_source_artifacts(
                self.source,
                verified_images,
                environment,
            )
            artifact_identity = _identity_sha256(
                {
                    "domain": "wre.colmap-patch-match-dense-depth-artifact",
                    "schema_version": 1,
                    "normalization_sha256": normalization_identity.value,
                    "producer": _producer_document(producer),
                    "source_artifacts": [
                        [item.artifact_id.value, item.artifact_kind.value]
                        for item in source_artifacts
                    ],
                    "depth_fields": [
                        item.depth_field_id.value for item in depth_fields
                    ],
                    "depth_outputs": list(depth_evidence),
                }
            )
            result = DenseDepthArtifact(
                artifact_ref=ArtifactRef(
                    artifact_id=ArtifactId(
                        f"artifact:colmap-dense:{artifact_identity.value}"
                    ),
                    artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
                ),
                source_geometry=self.source.source_geometry,
                depth_fields=depth_fields,
                producer=producer,
                source_artifacts=source_artifacts,
            )
            _verify_retained_sources(self.source, verified_images)
            return result
        except Exception:
            if output_created:
                shutil.rmtree(output_root, ignore_errors=True)
            try:
                _verify_retained_sources(self.source, verified_images)
            except Exception as source_exc:
                raise ColmapDenseDepthError(
                    "retained COLMAP dense-depth source changed during failed donor execution"
                ) from source_exc
            raise
