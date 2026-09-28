from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from wre.domain.adapter_capabilities import AdapterCapabilityDescriptor, AdapterCapabilityName
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
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact

OPEN3D_SURFACE_ADAPTER_ID = "open3d.tsdf_surface"
OPEN3D_SURFACE_CAPABILITY_NAME = AdapterCapabilityName("geometry.surface.depth_fusion")
OPEN3D_SURFACE_CAPABILITY = AdapterCapabilityDescriptor(
    capability=OPEN3D_SURFACE_CAPABILITY_NAME,
    input_kinds=frozenset({DENSE_DEPTH_ARTIFACT_KIND}),
    output_kinds=frozenset({SURFACE_MODEL_ARTIFACT_KIND}),
)
OPEN3D_SURFACE_DEPENDENCY_REF = "open3d"
OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION = "open3d.t.geometry.VoxelBlockGrid"
OPEN3D_SURFACE_PRODUCER_VERSION = "0.20.0"
OPEN3D_SURFACE_SOURCE_REVISION = "b6c5e196384ad71e75b6e6f9c5da22d046221f1d"
OPEN3D_SURFACE_ROOT_WHEEL = "open3d-0.20.0-cp312-cp312-manylinux_2_35_x86_64.whl"
OPEN3D_SURFACE_ROOT_WHEEL_SHA256 = Sha256Digest(
    "f5cc6106d9c0c41beb8160aa99c2b663e7f08588a598ec7d464391a4cfe684d6"
)
OPEN3D_SURFACE_ROOT_WHEEL_BYTE_LENGTH = 400788891
OPEN3D_SURFACE_ENVIRONMENT_LOCK_SHA256 = Sha256Digest(
    "ac9b2e1346e7b0a0bbed92eee22887e0e662cc303ea9f97df0e5fde835b07214"
)
OPEN3D_SURFACE_ENVIRONMENT_ARTIFACT_COUNT = 59
OPEN3D_SURFACE_ENVIRONMENT_BYTE_LENGTH = 453384619
OPEN3D_SURFACE_TARGET_PYTHON = "3.12.12"
OPEN3D_SURFACE_TARGET_PLATFORM = "linux_x86_64"
OPEN3D_SURFACE_DEVICE = "CPU:0"
OPEN3D_SURFACE_DEPTH_SCALE = 1.0
OPEN3D_SURFACE_SHIPPING_STATUS = "experimental"
OPEN3D_SURFACE_ARTIFACT_KEY_HARDWARE_POLICY = "required"
OPEN3D_SURFACE_MESH_PATH = "surface.ply"

OPEN3D_CAMERA_Z_CONVENTION = DepthValueConventionName("camera-z")
OPEN3D_PINHOLE_PROJECTION = CameraProjectionModelName("pinhole")
OPEN3D_SURFACE_REPRESENTATION = SurfaceRepresentationName("mesh")


class Open3dSurfaceError(RuntimeError):
    """Raised when the bounded Open3D surface baseline fails closed."""


class Open3dSurfaceEnvironmentError(Open3dSurfaceError):
    """Raised when the active Open3D module is not the reviewed CPU baseline."""


def _identity_sha256(document: dict[str, object]) -> Sha256Digest:
    encoded = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Sha256Digest(hashlib.sha256(encoded).hexdigest())


def _positive_finite_float(value: object, context: str) -> float:
    if type(value) is not float:
        raise TypeError(f"{context} must be float")
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{context} must be positive and finite")
    return value


def _positive_int(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{context} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class Open3dTsdfSurfaceConfig:
    """Explicit local-unit configuration for the bounded CPU TSDF baseline."""

    voxel_size_local_units: float
    max_depth_local_units: float
    trunc_voxel_multiplier: float
    block_resolution: int
    block_count: int
    mesh_weight_threshold: float
    schema_version: int = 1
    device: str = OPEN3D_SURFACE_DEVICE
    depth_scale: float = OPEN3D_SURFACE_DEPTH_SCALE

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("open3d_surface.config.schema_version must be 1")
        _positive_finite_float(
            self.voxel_size_local_units,
            "open3d_surface.config.voxel_size_local_units",
        )
        _positive_finite_float(
            self.max_depth_local_units,
            "open3d_surface.config.max_depth_local_units",
        )
        _positive_finite_float(
            self.trunc_voxel_multiplier,
            "open3d_surface.config.trunc_voxel_multiplier",
        )
        _positive_int(
            self.block_resolution,
            "open3d_surface.config.block_resolution",
        )
        _positive_int(
            self.block_count,
            "open3d_surface.config.block_count",
        )
        _positive_finite_float(
            self.mesh_weight_threshold,
            "open3d_surface.config.mesh_weight_threshold",
        )
        if self.device != OPEN3D_SURFACE_DEVICE:
            raise ValueError("open3d_surface.config.device must be exactly CPU:0")
        if type(self.depth_scale) is not float or self.depth_scale != 1.0:
            raise ValueError("open3d_surface.config.depth_scale must be exactly 1.0")

    @property
    def sha256(self) -> Sha256Digest:
        return _identity_sha256(
            {
                "domain": "wre.open3d-tsdf-surface-config",
                "schema_version": self.schema_version,
                "voxel_size_local_units": self.voxel_size_local_units,
                "max_depth_local_units": self.max_depth_local_units,
                "trunc_voxel_multiplier": self.trunc_voxel_multiplier,
                "block_resolution": self.block_resolution,
                "block_count": self.block_count,
                "mesh_weight_threshold": self.mesh_weight_threshold,
                "device": self.device,
                "depth_scale": self.depth_scale,
            }
        )


@dataclass(frozen=True, slots=True)
class Open3dSurfaceEnvironmentIdentity:
    """Exact optional external environment identity for V2L16.2."""

    version: str = OPEN3D_SURFACE_PRODUCER_VERSION
    source_revision: str = OPEN3D_SURFACE_SOURCE_REVISION
    root_wheel: str = OPEN3D_SURFACE_ROOT_WHEEL
    root_wheel_sha256: Sha256Digest = OPEN3D_SURFACE_ROOT_WHEEL_SHA256
    root_wheel_byte_length: int = OPEN3D_SURFACE_ROOT_WHEEL_BYTE_LENGTH
    lock_sha256: Sha256Digest = OPEN3D_SURFACE_ENVIRONMENT_LOCK_SHA256
    artifact_count: int = OPEN3D_SURFACE_ENVIRONMENT_ARTIFACT_COUNT
    artifact_byte_length: int = OPEN3D_SURFACE_ENVIRONMENT_BYTE_LENGTH
    python: str = OPEN3D_SURFACE_TARGET_PYTHON
    platform: str = OPEN3D_SURFACE_TARGET_PLATFORM
    device: str = OPEN3D_SURFACE_DEVICE

    def __post_init__(self) -> None:
        if self.version != OPEN3D_SURFACE_PRODUCER_VERSION:
            raise ValueError("Open3D surface environment version must be exactly 0.20.0")
        if self.source_revision != OPEN3D_SURFACE_SOURCE_REVISION:
            raise ValueError("Open3D surface environment source revision is not reviewed")
        if self.root_wheel != OPEN3D_SURFACE_ROOT_WHEEL:
            raise ValueError("Open3D surface environment root wheel is not reviewed")
        if self.root_wheel_sha256 != OPEN3D_SURFACE_ROOT_WHEEL_SHA256:
            raise ValueError("Open3D surface environment root wheel SHA-256 is not reviewed")
        if self.root_wheel_byte_length != OPEN3D_SURFACE_ROOT_WHEEL_BYTE_LENGTH:
            raise ValueError("Open3D surface environment root wheel byte length is not reviewed")
        if self.lock_sha256 != OPEN3D_SURFACE_ENVIRONMENT_LOCK_SHA256:
            raise ValueError("Open3D surface environment lock SHA-256 is not reviewed")
        if self.artifact_count != OPEN3D_SURFACE_ENVIRONMENT_ARTIFACT_COUNT:
            raise ValueError("Open3D surface environment artifact count is not reviewed")
        if self.artifact_byte_length != OPEN3D_SURFACE_ENVIRONMENT_BYTE_LENGTH:
            raise ValueError("Open3D surface environment byte length is not reviewed")
        if self.python != OPEN3D_SURFACE_TARGET_PYTHON:
            raise ValueError("Open3D surface environment Python version is not reviewed")
        if self.platform != OPEN3D_SURFACE_TARGET_PLATFORM:
            raise ValueError("Open3D surface environment platform is not reviewed")
        if self.device != OPEN3D_SURFACE_DEVICE:
            raise ValueError("Open3D surface environment device must be CPU:0")

    @property
    def sha256(self) -> Sha256Digest:
        return _identity_sha256(
            {
                "domain": "wre.open3d-surface-environment",
                "schema_version": 1,
                "version": self.version,
                "source_revision": self.source_revision,
                "root_wheel": self.root_wheel,
                "root_wheel_sha256": self.root_wheel_sha256.value,
                "root_wheel_byte_length": self.root_wheel_byte_length,
                "lock_sha256": self.lock_sha256.value,
                "artifact_count": self.artifact_count,
                "artifact_byte_length": self.artifact_byte_length,
                "python": self.python,
                "platform": self.platform,
                "device": self.device,
            }
        )


OPEN3D_SURFACE_ENVIRONMENT = Open3dSurfaceEnvironmentIdentity()


@dataclass(frozen=True, slots=True)
class Open3dSurfaceResult:
    surface_model: SurfaceModel
    materialization: ArtifactMaterializationMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.surface_model, SurfaceModel):
            raise TypeError("open3d_surface_result.surface_model must be SurfaceModel")
        if not isinstance(self.materialization, ArtifactMaterializationMetadata):
            raise TypeError(
                "open3d_surface_result.materialization must be ArtifactMaterializationMetadata"
            )
        if self.surface_model.artifact_ref != self.materialization.artifact_ref:
            raise ValueError(
                "Open3D surface result model and materialization artifact references must match"
            )


def _load_open3d() -> Any:
    try:
        import open3d  # type: ignore[import-not-found]
    except ImportError as exc:
        raise Open3dSurfaceEnvironmentError(
            "Open3D 0.20.0 is required in the optional surface environment"
        ) from exc
    return open3d


def _require_attr(value: object, name: str, context: str) -> object:
    try:
        return getattr(value, name)
    except AttributeError as exc:
        raise Open3dSurfaceEnvironmentError(
            f"reviewed Open3D API is missing {context}.{name}"
        ) from exc


def _validate_open3d_module(module: object) -> Any:
    if getattr(module, "__version__", None) != OPEN3D_SURFACE_PRODUCER_VERSION:
        raise Open3dSurfaceEnvironmentError("Open3D module version must be exactly 0.20.0")

    core = _require_attr(module, "core", "open3d")
    _require_attr(core, "Device", "open3d.core")
    _require_attr(core, "Tensor", "open3d.core")
    _require_attr(core, "float32", "open3d.core")
    _require_attr(core, "float64", "open3d.core")

    t_module = _require_attr(module, "t", "open3d")
    geometry = _require_attr(t_module, "geometry", "open3d.t")
    image_type = _require_attr(geometry, "Image", "open3d.t.geometry")
    voxel_grid_type = _require_attr(geometry, "VoxelBlockGrid", "open3d.t.geometry")
    if not callable(image_type) or not callable(voxel_grid_type):
        raise Open3dSurfaceEnvironmentError("reviewed Open3D geometry API is not callable")

    io_module = _require_attr(t_module, "io", "open3d.t")
    writer = _require_attr(io_module, "write_triangle_mesh", "open3d.t.io")
    if not callable(writer):
        raise Open3dSurfaceEnvironmentError("reviewed Open3D mesh writer API is not callable")
    return cast(Any, module)


def _validate_intended_uses(value: object) -> tuple[SurfaceIntendedUse, ...]:
    if not isinstance(value, tuple):
        raise TypeError("open3d_surface.intended_uses must be an immutable tuple")
    if not value:
        raise ValueError("open3d_surface.intended_uses must be non-empty")
    if any(not isinstance(item, SurfaceIntendedUse) for item in value):
        raise TypeError("open3d_surface.intended_uses members must be SurfaceIntendedUse")
    identifiers = tuple(item.value for item in value)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("open3d_surface.intended_uses must be unique")
    if identifiers != tuple(sorted(identifiers)):
        raise ValueError("open3d_surface.intended_uses must use canonical value order")
    return value


def _validate_source(source: DenseDepthArtifact) -> dict[str, CameraSolution]:
    cameras = {camera.solution_id.value: camera for camera in source.source_geometry.camera_solutions}
    for depth in source.depth_fields:
        if depth.depth_value_convention != OPEN3D_CAMERA_Z_CONVENTION:
            raise Open3dSurfaceError(
                "Open3D surface baseline requires canonical camera-z depth without conversion"
            )
        camera = cameras.get(depth.camera_solution_id.value)
        if camera is None:
            raise Open3dSurfaceError("dense depth references an unavailable source camera")
        if camera.projection_model != OPEN3D_PINHOLE_PROJECTION:
            raise Open3dSurfaceError(
                "Open3D surface baseline accepts only canonical pinhole cameras"
            )
        if len(camera.intrinsic_parameters) != 4:
            raise Open3dSurfaceError(
                "Open3D surface pinhole camera requires exactly fx fy cx cy intrinsics"
            )
        fx, fy, _cx, _cy = camera.intrinsic_parameters
        if fx <= 0.0 or fy <= 0.0:
            raise Open3dSurfaceError("Open3D surface pinhole focal lengths must be positive")
        if depth.observation_id != camera.observation_id:
            raise Open3dSurfaceError("dense depth ObservationId changed from its source camera")
        if depth.dimensions != camera.dimensions:
            raise Open3dSurfaceError("dense depth dimensions changed from its source camera")
    return cameras


def _depth_rows(depth: DepthField) -> list[list[float]]:
    width = depth.dimensions.width_px
    height = depth.dimensions.height_px
    values = [float(value) for value in depth.depth_values]
    if len(values) != width * height:
        raise Open3dSurfaceError("dense depth pixel count changed before Open3D materialization")
    return [values[row * width : (row + 1) * width] for row in range(height)]


def _intrinsic_matrix(camera: CameraSolution) -> tuple[tuple[float, float, float], ...]:
    fx, fy, cx, cy = camera.intrinsic_parameters
    return (
        (fx, 0.0, cx),
        (0.0, fy, cy),
        (0.0, 0.0, 1.0),
    )


def _extrinsic_matrix(camera: CameraSolution) -> tuple[tuple[float, float, float, float], ...]:
    rotation = camera.rotation_matrix
    tx, ty, tz = camera.translation_xyz
    return (
        (rotation[0][0], rotation[0][1], rotation[0][2], tx),
        (rotation[1][0], rotation[1][1], rotation[1][2], ty),
        (rotation[2][0], rotation[2][1], rotation[2][2], tz),
        (0.0, 0.0, 0.0, 1.0),
    )


def _tensor_shape(tensor: object, context: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in cast(Any, tensor).shape)
    except Exception as exc:
        raise Open3dSurfaceError(f"{context} has no readable tensor shape") from exc


def _tensor_flat_values(tensor: object, context: str) -> tuple[float, ...]:
    try:
        array = cast(Any, tensor).cpu().numpy()
        return tuple(float(value) for value in array.reshape(-1).tolist())
    except Exception as exc:
        raise Open3dSurfaceError(f"{context} cannot be inspected on CPU") from exc


def _validate_mesh(mesh: object) -> None:
    try:
        positions = cast(Any, mesh).vertex.positions
        triangles = cast(Any, mesh).triangle.indices
    except Exception as exc:
        raise Open3dSurfaceError("Open3D triangle mesh lacks required geometry attributes") from exc

    position_shape = _tensor_shape(positions, "Open3D mesh vertex positions")
    triangle_shape = _tensor_shape(triangles, "Open3D mesh triangle indices")
    if len(position_shape) != 2 or position_shape[0] <= 0 or position_shape[1] != 3:
        raise Open3dSurfaceError("Open3D surface mesh must contain vertex xyz positions")
    if len(triangle_shape) != 2 or triangle_shape[0] <= 0 or triangle_shape[1] != 3:
        raise Open3dSurfaceError("Open3D surface mesh must contain triangles")

    values = _tensor_flat_values(positions, "Open3D mesh vertex positions")
    if not values or not all(math.isfinite(value) for value in values):
        raise Open3dSurfaceError("Open3D surface mesh vertices must all be finite")


def _validate_geometry_only_ply(path: Path) -> None:
    with path.open("rb") as handle:
        header_bytes = handle.read(65536)
    marker = b"end_header"
    index = header_bytes.find(marker)
    if index < 0:
        raise Open3dSurfaceError("Open3D surface output is not a readable PLY")
    header = header_bytes[: index + len(marker)].decode("ascii", errors="strict").lower()
    if not header.startswith("ply\n") and not header.startswith("ply\r\n"):
        raise Open3dSurfaceError("Open3D surface output does not start with a PLY header")
    for forbidden in (
        "property float nx",
        "property double nx",
        "property float ny",
        "property double ny",
        "property float nz",
        "property double nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property uchar alpha",
        "texture",
        "material",
    ):
        if forbidden in header:
            raise Open3dSurfaceError(
                "Open3D surface PLY must contain geometry only without appearance attributes"
            )
    for required in ("property float x", "property float y", "property float z"):
        if required not in header and required.replace("float", "double") not in header:
            raise Open3dSurfaceError("Open3D surface PLY is missing vertex xyz properties")
    if "vertex_indices" not in header:
        raise Open3dSurfaceError("Open3D surface PLY is missing triangle vertex indices")


def _canonical_source_artifacts(source: DenseDepthArtifact) -> tuple[ArtifactRef, ...]:
    references = (
        *source.source_geometry.source_artifacts,
        source.artifact_ref,
        *source.source_artifacts,
    )
    by_identity = {
        (item.artifact_id.value, item.artifact_kind.value): item for item in references
    }
    return tuple(by_identity[key] for key in sorted(by_identity))


def _producer_configuration_sha256(config: Open3dTsdfSurfaceConfig) -> Sha256Digest:
    return _identity_sha256(
        {
            "domain": "wre.open3d-tsdf-surface-producer-configuration",
            "schema_version": 1,
            "tsdf_configuration_sha256": config.sha256.value,
            "environment_sha256": OPEN3D_SURFACE_ENVIRONMENT.sha256.value,
        }
    )


def _surface_artifact_identity(
    *,
    source: DenseDepthArtifact,
    intended_uses: tuple[SurfaceIntendedUse, ...],
    config: Open3dTsdfSurfaceConfig,
    mesh_sha256: Sha256Digest,
    mesh_byte_length: int,
) -> Sha256Digest:
    return _identity_sha256(
        {
            "domain": "wre.open3d-tsdf-surface-artifact",
            "schema_version": 1,
            "source_dense_depth": [
                source.artifact_ref.artifact_id.value,
                source.artifact_ref.artifact_kind.value,
            ],
            "source_geometry_id": source.source_geometry.geometry_solution_id.value,
            "source_local_frame_id": source.source_geometry.geometry_solution.local_frame_id.value,
            "source_scale_status": source.source_geometry.geometry_solution.scale_status.value,
            "intended_uses": [item.value for item in intended_uses],
            "configuration_sha256": config.sha256.value,
            "environment_sha256": OPEN3D_SURFACE_ENVIRONMENT.sha256.value,
            "mesh_sha256": mesh_sha256.value,
            "mesh_byte_length": mesh_byte_length,
        }
    )


def _output_has_symlink_ancestor(path: Path) -> bool:
    expanded = path.expanduser()
    return any(
        candidate.is_symlink()
        for candidate in (expanded, *expanded.parents)
        if candidate.exists() or candidate.is_symlink()
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class Open3dTsdfSurfaceAdapter:
    """Bounded Open3D 0.20.0 CPU depth-only TSDF physical-surface baseline."""

    source: DenseDepthArtifact
    output_root: Path
    intended_uses: tuple[SurfaceIntendedUse, ...]
    config: Open3dTsdfSurfaceConfig
    module: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source, DenseDepthArtifact):
            raise TypeError("open3d_surface.source must be DenseDepthArtifact")
        if not isinstance(self.output_root, Path):
            raise TypeError("open3d_surface.output_root must be pathlib.Path")
        _validate_intended_uses(self.intended_uses)
        if not isinstance(self.config, Open3dTsdfSurfaceConfig):
            raise TypeError("open3d_surface.config must be Open3dTsdfSurfaceConfig")

    def derive(self) -> Open3dSurfaceResult:
        cameras = _validate_source(self.source)
        open3d = _validate_open3d_module(self.module if self.module is not None else _load_open3d())

        output_candidate = self.output_root.expanduser()
        if output_candidate.exists() or output_candidate.is_symlink():
            raise ValueError("Open3D surface output_root must not already exist")
        if _output_has_symlink_ancestor(output_candidate.parent):
            raise ValueError("Open3D surface output_root must not traverse symlink ancestors")
        output_root = output_candidate.resolve()
        output_root.parent.mkdir(parents=True, exist_ok=True)

        output_created = False
        try:
            output_root.mkdir()
            output_created = True

            device = open3d.core.Device(self.config.device)
            voxel_grid = open3d.t.geometry.VoxelBlockGrid(
                attr_names=("tsdf", "weight"),
                attr_dtypes=(open3d.core.float32, open3d.core.float32),
                attr_channels=((1,), (1,)),
                voxel_size=self.config.voxel_size_local_units,
                block_resolution=self.config.block_resolution,
                block_count=self.config.block_count,
                device=device,
            )

            for depth in self.source.depth_fields:
                camera = cameras[depth.camera_solution_id.value]
                depth_tensor = open3d.core.Tensor(
                    _depth_rows(depth),
                    dtype=open3d.core.float32,
                    device=device,
                )
                depth_image = open3d.t.geometry.Image(depth_tensor)
                intrinsic = open3d.core.Tensor(
                    _intrinsic_matrix(camera),
                    dtype=open3d.core.float64,
                    device=device,
                )
                extrinsic = open3d.core.Tensor(
                    _extrinsic_matrix(camera),
                    dtype=open3d.core.float64,
                    device=device,
                )

                block_coords = voxel_grid.compute_unique_block_coordinates(
                    depth_image,
                    intrinsic,
                    extrinsic,
                    depth_scale=self.config.depth_scale,
                    depth_max=self.config.max_depth_local_units,
                    trunc_voxel_multiplier=self.config.trunc_voxel_multiplier,
                )
                voxel_grid.integrate(
                    block_coords,
                    depth_image,
                    intrinsic,
                    extrinsic,
                    depth_scale=self.config.depth_scale,
                    depth_max=self.config.max_depth_local_units,
                    trunc_voxel_multiplier=self.config.trunc_voxel_multiplier,
                )

            mesh = voxel_grid.extract_triangle_mesh(
                weight_threshold=self.config.mesh_weight_threshold
            )
            _validate_mesh(mesh)

            output_path = output_root / OPEN3D_SURFACE_MESH_PATH
            wrote = bool(
                open3d.t.io.write_triangle_mesh(
                    str(output_path),
                    mesh,
                    write_ascii=False,
                    compressed=False,
                    write_vertex_normals=False,
                    write_vertex_colors=False,
                    write_triangle_uvs=False,
                    print_progress=False,
                )
            )
            if not wrote:
                raise Open3dSurfaceError("Open3D failed to write the surface PLY")
            if output_path.is_symlink() or not output_path.is_file():
                raise Open3dSurfaceError("Open3D surface output must be one regular PLY file")

            entries = tuple(output_root.iterdir())
            if entries != (output_path,):
                raise Open3dSurfaceError(
                    "Open3D surface output_root must contain exactly surface.ply"
                )
            _validate_geometry_only_ply(output_path)

            digest = hash_file_content(output_path)
            identity = _surface_artifact_identity(
                source=self.source,
                intended_uses=self.intended_uses,
                config=self.config,
                mesh_sha256=digest.sha256,
                mesh_byte_length=digest.byte_length,
            )
            artifact_ref = ArtifactRef(
                artifact_id=ArtifactId(f"surface:open3d-tsdf:{identity.value}"),
                artifact_kind=SURFACE_MODEL_ARTIFACT_KIND,
            )

            input_artifact_ids = {
                item.artifact_id.value
                for item in (
                    self.source.artifact_ref,
                    *self.source.source_artifacts,
                    *self.source.source_geometry.source_artifacts,
                )
            }
            if artifact_ref.artifact_id.value in input_artifact_ids:
                raise Open3dSurfaceError(
                    "Open3D surface output ArtifactId must be distinct from every input"
                )

            source_artifacts = _canonical_source_artifacts(self.source)
            producer = ArtifactProducerIdentity(
                producer=ProducerRef(
                    implementation=OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
                    version=OPEN3D_SURFACE_PRODUCER_VERSION,
                    revision=OPEN3D_SURFACE_SOURCE_REVISION,
                ),
                configuration=ConfigurationIdentity(
                    sha256=_producer_configuration_sha256(self.config)
                ),
            )
            model = SurfaceModel(
                artifact_ref=artifact_ref,
                source_geometry=self.source.source_geometry,
                representation=OPEN3D_SURFACE_REPRESENTATION,
                intended_uses=self.intended_uses,
                local_frame_id=self.source.source_geometry.geometry_solution.local_frame_id,
                scale_status=self.source.source_geometry.geometry_solution.scale_status,
                producer=producer,
                source_artifacts=source_artifacts,
            )
            materialization = ArtifactMaterializationMetadata(
                artifact_ref=artifact_ref,
                entries=(
                    ArtifactMaterializationEntry(
                        relative_path=OPEN3D_SURFACE_MESH_PATH,
                        sha256=digest.sha256,
                        byte_length=digest.byte_length,
                    ),
                ),
            )
            result = Open3dSurfaceResult(
                surface_model=model,
                materialization=materialization,
            )
        except Exception:
            if output_created:
                shutil.rmtree(output_root, ignore_errors=True)
            raise

        output_created = False
        return result
