from __future__ import annotations

import hashlib
import importlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.artifact_materialization import (
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
)
from wre.domain.artifacts import ArtifactId, ArtifactRef
from wre.domain.camera_solutions import CameraProjectionModelName, CameraSolution
from wre.domain.depth_fields import DepthField, DepthValueConventionName
from wre.domain.observations import Sha256Digest
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.domain.surfaces import (
    SURFACE_MODEL_ARTIFACT_KIND,
    SurfaceIntendedUse,
    SurfaceModel,
    SurfaceRepresentationName,
)
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.dense_depth import DenseDepthArtifact

OPEN3D_SURFACE_ADAPTER_ID = "open3d.depth_tsdf_surface"
OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION = "open3d.t.geometry.VoxelBlockGrid.depth_only"
OPEN3D_SURFACE_PRODUCER_VERSION = "0.20.0"
OPEN3D_SURFACE_SOURCE_REVISION = "b6c5e196384ad71e75b6e6f9c5da22d046221f1d"
OPEN3D_SURFACE_DEPENDENCY_REF = "open3d_surface_cpu"
OPEN3D_SURFACE_SHIPPING_STATUS = "experimental"
OPEN3D_SURFACE_WHEEL_FILENAME = "open3d-0.20.0-cp312-cp312-manylinux_2_35_x86_64.whl"
OPEN3D_SURFACE_WHEEL_SHA256 = Sha256Digest(
    "f5cc6106d9c0c41beb8160aa99c2b663e7f08588a598ec7d464391a4cfe684d6"
)
OPEN3D_SURFACE_WHEEL_BYTE_LENGTH = 400_788_891
OPEN3D_SURFACE_DEVICE = "CPU:0"
OPEN3D_SURFACE_DEPTH_SCALE = 1.0
OPEN3D_SURFACE_REPRESENTATION = SurfaceRepresentationName("mesh")
OPEN3D_PINHOLE_PROJECTION = CameraProjectionModelName("pinhole")
OPEN3D_CAMERA_Z_CONVENTION = DepthValueConventionName("camera-z")
OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH = "surface.ply"


class Open3dSurfaceError(RuntimeError):
    """Raised when the bounded Open3D surface baseline fails closed."""


class Open3dSurfaceEnvironmentError(Open3dSurfaceError):
    """Raised when the active Open3D module does not match the reviewed baseline."""


def _positive_finite_float(value: object, context: str) -> None:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")


def _positive_int(value: object, context: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be int")
    if value <= 0:
        raise ValueError(f"{context} must be positive")


def _identity_sha256(document: object) -> Sha256Digest:
    payload = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(payload).hexdigest())


def _artifact_key(value: ArtifactRef) -> tuple[str, str]:
    return (value.artifact_id.value, value.artifact_kind.value)


@dataclass(frozen=True, slots=True)
class Open3dSurfaceEnvironmentIdentity:
    """Exact external CPU environment bound to the reviewed Open3D wheel closure."""

    manifest_sha256: Sha256Digest
    python_version: str = "3.12.12"
    platform: str = "linux_x86_64"
    architecture: str = "x86_64"
    open3d_version: str = OPEN3D_SURFACE_PRODUCER_VERSION
    source_revision: str = OPEN3D_SURFACE_SOURCE_REVISION
    wheel_filename: str = OPEN3D_SURFACE_WHEEL_FILENAME
    wheel_sha256: Sha256Digest = OPEN3D_SURFACE_WHEEL_SHA256
    wheel_byte_length: int = OPEN3D_SURFACE_WHEEL_BYTE_LENGTH
    device: str = OPEN3D_SURFACE_DEVICE

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_sha256, Sha256Digest):
            raise TypeError("open3d_surface_environment.manifest_sha256 must be Sha256Digest")
        if self.python_version != "3.12.12":
            raise ValueError("Open3D surface environment requires Python 3.12.12")
        if self.platform != "linux_x86_64" or self.architecture != "x86_64":
            raise ValueError("Open3D surface environment requires Linux x86_64")
        if self.open3d_version != OPEN3D_SURFACE_PRODUCER_VERSION:
            raise ValueError("Open3D surface environment requires open3d 0.20.0")
        if self.source_revision != OPEN3D_SURFACE_SOURCE_REVISION:
            raise ValueError(
                "Open3D surface environment source revision must match reviewed source"
            )
        if self.wheel_filename != OPEN3D_SURFACE_WHEEL_FILENAME:
            raise ValueError("Open3D surface environment wheel filename must match reviewed wheel")
        if self.wheel_sha256 != OPEN3D_SURFACE_WHEEL_SHA256:
            raise ValueError("Open3D surface environment wheel SHA-256 must match reviewed wheel")
        if self.wheel_byte_length != OPEN3D_SURFACE_WHEEL_BYTE_LENGTH:
            raise ValueError(
                "Open3D surface environment wheel byte length must match reviewed wheel"
            )
        if self.device != OPEN3D_SURFACE_DEVICE:
            raise ValueError("Open3D surface environment requires CPU:0")

    def canonical_document(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "device": self.device,
            "manifest_sha256": self.manifest_sha256.value,
            "open3d_version": self.open3d_version,
            "platform": self.platform,
            "python_version": self.python_version,
            "source_revision": self.source_revision,
            "wheel_byte_length": self.wheel_byte_length,
            "wheel_filename": self.wheel_filename,
            "wheel_sha256": self.wheel_sha256.value,
        }


@dataclass(frozen=True, slots=True)
class Open3dTsdfSurfaceConfig:
    """Explicit local-unit Open3D TSDF parameters for the bounded CPU baseline."""

    voxel_size: float
    depth_max: float
    trunc_voxel_multiplier: float
    block_resolution: int
    block_count: int
    mesh_weight_threshold: float
    schema_version: int = 1
    device: str = OPEN3D_SURFACE_DEVICE
    depth_scale: float = OPEN3D_SURFACE_DEPTH_SCALE

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Open3D surface config schema_version must be 1")
        _positive_finite_float(self.voxel_size, "open3d_surface.voxel_size")
        _positive_finite_float(self.depth_max, "open3d_surface.depth_max")
        _positive_finite_float(
            self.trunc_voxel_multiplier,
            "open3d_surface.trunc_voxel_multiplier",
        )
        _positive_int(self.block_resolution, "open3d_surface.block_resolution")
        _positive_int(self.block_count, "open3d_surface.block_count")
        _positive_finite_float(
            self.mesh_weight_threshold,
            "open3d_surface.mesh_weight_threshold",
        )
        if self.device != OPEN3D_SURFACE_DEVICE:
            raise ValueError("V2L16.2 requires Open3D device CPU:0")
        if type(self.depth_scale) is not float:
            raise TypeError("open3d_surface.depth_scale must be float")
        if self.depth_scale != OPEN3D_SURFACE_DEPTH_SCALE:
            raise ValueError("V2L16.2 requires depth_scale exactly 1.0")

    def canonical_document(self) -> dict[str, object]:
        return {
            "block_count": self.block_count,
            "block_resolution": self.block_resolution,
            "depth_max": self.depth_max,
            "depth_scale": self.depth_scale,
            "device": self.device,
            "mesh_weight_threshold": self.mesh_weight_threshold,
            "schema_version": self.schema_version,
            "trunc_voxel_multiplier": self.trunc_voxel_multiplier,
            "voxel_size": self.voxel_size,
        }

    @property
    def sha256(self) -> Sha256Digest:
        return _identity_sha256(self.canonical_document())


@dataclass(frozen=True, slots=True)
class Open3dSurfaceResult:
    """Published physical surface plus exact local materialization identity."""

    surface_model: SurfaceModel
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.surface_model, SurfaceModel):
            raise TypeError("open3d_surface_result.surface_model must be SurfaceModel")
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "open3d_surface_result.materialization must be ArtifactMaterializationMetadata"
            )
        if self.materialization.artifact_ref != self.surface_model.artifact_ref:
            raise ValueError(
                "Open3D surface materialization must identify the produced SurfaceModel artifact"
            )


def _load_open3d() -> Any:
    try:
        return importlib.import_module("open3d")
    except (ImportError, OSError, RuntimeError) as exc:
        raise Open3dSurfaceEnvironmentError(
            "Open3D is unavailable; use the exact optional open3d 0.20.0 CPU environment"
        ) from exc


def inspect_open3d_surface_environment(
    expected: Open3dSurfaceEnvironmentIdentity,
    module: object | None = None,
) -> Open3dSurfaceEnvironmentIdentity:
    """Validate the active Open3D API boundary without resolving or downloading anything."""

    if not isinstance(expected, Open3dSurfaceEnvironmentIdentity):
        raise TypeError("expected must be Open3dSurfaceEnvironmentIdentity")
    open3d = cast(Any, module) if module is not None else _load_open3d()

    if getattr(open3d, "__version__", None) != OPEN3D_SURFACE_PRODUCER_VERSION:
        raise Open3dSurfaceEnvironmentError("Open3D runtime version must be exactly 0.20.0")

    try:
        device = open3d.core.Device(OPEN3D_SURFACE_DEVICE)
        voxel_block_grid = open3d.t.geometry.VoxelBlockGrid
        image = open3d.t.geometry.Image
        write_triangle_mesh = open3d.t.io.write_triangle_mesh
        float32 = open3d.core.float32
        float64 = open3d.core.float64
    except AttributeError as exc:
        raise Open3dSurfaceEnvironmentError(
            "Open3D runtime is missing the reviewed tensor TSDF API"
        ) from exc

    if str(device) != OPEN3D_SURFACE_DEVICE:
        raise Open3dSurfaceEnvironmentError("Open3D CPU:0 device identity is unavailable")
    if not callable(voxel_block_grid):
        raise Open3dSurfaceEnvironmentError("Open3D VoxelBlockGrid must be callable")
    if not callable(image):
        raise Open3dSurfaceEnvironmentError("Open3D tensor Image must be callable")
    if not callable(write_triangle_mesh):
        raise Open3dSurfaceEnvironmentError("Open3D tensor write_triangle_mesh must be callable")
    if float32 is None or float64 is None:
        raise Open3dSurfaceEnvironmentError("Open3D Float32/Float64 dtypes must be available")
    return expected


def _camera_by_id(depth: DenseDepthArtifact) -> dict[object, CameraSolution]:
    return {
        camera.solution_id: camera
        for camera in depth.source_geometry.camera_solutions
    }


def _validate_depth_field(camera: CameraSolution, depth_field: DepthField) -> None:
    if camera.projection_model != OPEN3D_PINHOLE_PROJECTION:
        raise Open3dSurfaceError(
            "Open3D surface baseline accepts only canonical pinhole cameras"
        )
    if len(camera.intrinsic_parameters) != 4:
        raise Open3dSurfaceError(
            "Open3D surface baseline requires pinhole intrinsics exactly fx fy cx cy"
        )
    if depth_field.depth_value_convention != OPEN3D_CAMERA_Z_CONVENTION:
        raise Open3dSurfaceError(
            "Open3D surface baseline accepts only canonical camera-z depth"
        )
    if depth_field.observation_id != camera.observation_id:
        raise Open3dSurfaceError("DepthField observation must match its CameraSolution")
    if depth_field.dimensions != camera.dimensions:
        raise Open3dSurfaceError("DepthField dimensions must match its CameraSolution")


def _canonical_source_artifacts(depth: DenseDepthArtifact) -> tuple[ArtifactRef, ...]:
    values = {
        _artifact_key(item): item
        for item in (
            *depth.source_geometry.source_artifacts,
            depth.artifact_ref,
            *depth.source_artifacts,
        )
    }
    return tuple(values[key] for key in sorted(values))


def _producer_configuration_sha256(
    config: Open3dTsdfSurfaceConfig,
    environment: Open3dSurfaceEnvironmentIdentity,
) -> Sha256Digest:
    return _identity_sha256(
        {
            "adapter_id": OPEN3D_SURFACE_ADAPTER_ID,
            "config": config.canonical_document(),
            "environment": environment.canonical_document(),
        }
    )


def _build_intrinsic(open3d: Any, camera: CameraSolution, device: Any) -> Any:
    fx, fy, cx, cy = camera.intrinsic_parameters
    return open3d.core.Tensor(
        (
            (fx, 0.0, cx),
            (0.0, fy, cy),
            (0.0, 0.0, 1.0),
        ),
        dtype=open3d.core.float64,
        device=device,
    )


def _build_extrinsic(open3d: Any, camera: CameraSolution, device: Any) -> Any:
    rotation = camera.rotation_matrix
    tx, ty, tz = camera.translation_xyz
    # Open3D VoxelBlockGrid expects world-to-camera. WRE's canonical camera pose
    # is already local-frame-to-camera, so it must be forwarded directly.
    return open3d.core.Tensor(
        (
            (rotation[0][0], rotation[0][1], rotation[0][2], tx),
            (rotation[1][0], rotation[1][1], rotation[1][2], ty),
            (rotation[2][0], rotation[2][1], rotation[2][2], tz),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=open3d.core.float64,
        device=device,
    )


def _build_depth_image(open3d: Any, depth_field: DepthField, device: Any) -> Any:
    width = depth_field.dimensions.width_px
    height = depth_field.dimensions.height_px
    samples = tuple(
        value if valid else 0.0
        for value, valid in zip(
            depth_field.depth_values,
            depth_field.validity,
            strict=True,
        )
    )
    tensor = open3d.core.Tensor(
        samples,
        dtype=open3d.core.float32,
        device=device,
    ).reshape((height, width))
    return open3d.t.geometry.Image(tensor)


def _mesh_has_finite_geometry(mesh: Any) -> bool:
    try:
        positions = mesh.vertex.positions
        triangles = mesh.triangle.indices
        if int(positions.shape[0]) <= 0 or int(triangles.shape[0]) <= 0:
            return False
        flat_positions = positions.numpy().reshape(-1).tolist()
    except (AttributeError, IndexError, TypeError, ValueError, RuntimeError):
        return False
    return bool(flat_positions) and all(math.isfinite(float(value)) for value in flat_positions)


def _validate_output_path(path: Path) -> None:
    probe = path
    while True:
        if probe.is_symlink():
            raise Open3dSurfaceError("Open3D surface output path must not traverse symlinks")
        if probe.parent == probe:
            break
        probe = probe.parent


@dataclass(frozen=True, slots=True, kw_only=True)
class Open3dTsdfSurfaceAdapter:
    """Bounded Open3D 0.20.0 CPU depth-only TSDF baseline."""

    output_root: Path
    expected_environment: Open3dSurfaceEnvironmentIdentity
    config: Open3dTsdfSurfaceConfig
    module: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.output_root, Path):
            raise TypeError("open3d_surface.output_root must be pathlib.Path")
        if not isinstance(self.expected_environment, Open3dSurfaceEnvironmentIdentity):
            raise TypeError(
                "open3d_surface.expected_environment must be "
                "Open3dSurfaceEnvironmentIdentity"
            )
        if not isinstance(self.config, Open3dTsdfSurfaceConfig):
            raise TypeError("open3d_surface.config must be Open3dTsdfSurfaceConfig")

    def derive(
        self,
        depth: DenseDepthArtifact,
        *,
        intended_uses: tuple[SurfaceIntendedUse, ...],
    ) -> Open3dSurfaceResult:
        if not isinstance(depth, DenseDepthArtifact):
            raise TypeError("depth must be DenseDepthArtifact")
        if not isinstance(intended_uses, tuple):
            raise TypeError("intended_uses must be an immutable tuple")
        if not intended_uses:
            raise ValueError("intended_uses must be non-empty")
        # SurfaceModel owns canonical intended-use validation. Constructing it only
        # after solver success keeps the adapter from inventing suitability semantics.

        cameras = _camera_by_id(depth)
        ordered_inputs: list[tuple[DepthField, CameraSolution]] = []
        for depth_field in depth.depth_fields:
            camera = cameras.get(depth_field.camera_solution_id)
            if camera is None:
                raise Open3dSurfaceError("DepthField must reference one source CameraSolution")
            _validate_depth_field(camera, depth_field)
            ordered_inputs.append((depth_field, camera))

        open3d = cast(Any, self.module) if self.module is not None else _load_open3d()
        environment = inspect_open3d_surface_environment(
            self.expected_environment,
            open3d,
        )
        device = open3d.core.Device(self.config.device)

        output_candidate = self.output_root.expanduser()
        _validate_output_path(output_candidate)
        if output_candidate.exists():
            raise ValueError("Open3D surface output_root must not already exist")
        output_candidate.parent.mkdir(parents=True, exist_ok=True)
        _validate_output_path(output_candidate.parent)
        output_root = output_candidate.parent.resolve(strict=True) / output_candidate.name

        output_created = False
        try:
            output_root.mkdir()
            output_created = True

            volume = open3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight"),
                attr_dtypes=(open3d.core.float32, open3d.core.float32),
                attr_channels=((1,), (1,)),
                voxel_size=self.config.voxel_size,
                block_resolution=self.config.block_resolution,
                block_count=self.config.block_count,
                device=device,
            )

            for depth_field, camera in ordered_inputs:
                depth_image = _build_depth_image(open3d, depth_field, device)
                intrinsic = _build_intrinsic(open3d, camera, device)
                extrinsic = _build_extrinsic(open3d, camera, device)
                block_coords = volume.compute_unique_block_coordinates(
                    depth_image,
                    intrinsic,
                    extrinsic,
                    self.config.depth_scale,
                    self.config.depth_max,
                    self.config.trunc_voxel_multiplier,
                )
                volume.integrate(
                    block_coords,
                    depth_image,
                    intrinsic,
                    extrinsic,
                    self.config.depth_scale,
                    self.config.depth_max,
                    self.config.trunc_voxel_multiplier,
                )

            mesh = volume.extract_triangle_mesh(
                weight_threshold=self.config.mesh_weight_threshold
            )
            if not _mesh_has_finite_geometry(mesh):
                raise Open3dSurfaceError(
                    "Open3D TSDF surface must contain finite vertices and at least one triangle"
                )

            surface_path = output_root / OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH
            if surface_path.exists() or surface_path.is_symlink():
                raise Open3dSurfaceError("Open3D surface output file must be fresh")
            try:
                written = open3d.t.io.write_triangle_mesh(
                    str(surface_path),
                    mesh,
                    write_ascii=False,
                    compressed=False,
                    write_vertex_normals=False,
                    write_vertex_colors=False,
                    write_triangle_uvs=False,
                    print_progress=False,
                )
            except Exception as exc:
                raise Open3dSurfaceError("Open3D failed to write surface.ply") from exc
            if written is not True:
                raise Open3dSurfaceError("Open3D failed to publish surface.ply")
            if surface_path.is_symlink() or not surface_path.is_file():
                raise Open3dSurfaceError("Open3D surface.ply must be one regular file")
            published = tuple(sorted(path.name for path in output_root.iterdir()))
            if published != (OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH,):
                raise Open3dSurfaceError(
                    "Open3D surface baseline must publish exactly one geometry-only surface.ply"
                )

            file_hash = hash_file_content(surface_path)
            source_artifacts = _canonical_source_artifacts(depth)
            producer_configuration = _producer_configuration_sha256(
                self.config,
                environment,
            )
            producer = ArtifactProducerIdentity(
                producer=ProducerRef(
                    implementation=OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
                    version=OPEN3D_SURFACE_PRODUCER_VERSION,
                    revision=OPEN3D_SURFACE_SOURCE_REVISION,
                ),
                configuration=ConfigurationIdentity(sha256=producer_configuration),
            )
            artifact_identity = _identity_sha256(
                {
                    "domain": "wre.open3d-depth-tsdf-surface",
                    "schema_version": 1,
                    "mesh": {
                        "byte_length": file_hash.byte_length,
                        "sha256": file_hash.sha256.value,
                    },
                    "producer": {
                        "configuration_sha256": producer_configuration.value,
                        "implementation": producer.producer.implementation,
                        "revision": producer.producer.revision,
                        "version": producer.producer.version,
                    },
                    "source_artifacts": [
                        [item.artifact_id.value, item.artifact_kind.value]
                        for item in source_artifacts
                    ],
                    "source_geometry_solution_id": (
                        depth.source_geometry.geometry_solution_id.value
                    ),
                    "intended_uses": [item.value for item in intended_uses],
                }
            )
            artifact_ref = ArtifactRef(
                artifact_id=ArtifactId(f"artifact:open3d-surface:{artifact_identity.value}"),
                artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
            )
            if _artifact_key(artifact_ref) in {_artifact_key(item) for item in source_artifacts}:
                raise Open3dSurfaceError(
                    "Open3D surface artifact identity must differ from every source artifact"
                )

            source_solution = depth.source_geometry.geometry_solution
            surface = SurfaceModel(
                artifact_ref=artifact_ref,
                source_geometry=depth.source_geometry,
                representation=OPEN3D_SURFACE_REPRESENTATION,
                intended_uses=intended_uses,
                local_frame_id=source_solution.local_frame_id,
                scale_status=source_solution.scale_status,
                producer=producer,
                source_artifacts=source_artifacts,
            )
            materialization = ArtifactMaterializationMetadata(
                artifact_ref=artifact_ref,
                entries=(
                    ArtifactMaterializationEntry(
                        relative_path=OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH,
                        sha256=file_hash.sha256,
                        byte_length=file_hash.byte_length,
                    ),
                ),
            )
            return Open3dSurfaceResult(
                surface_model=surface,
                materialization=materialization,
            )
        except Exception:
            if output_created:
                shutil.rmtree(output_root, ignore_errors=True)
            raise
