from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.depth_fields import DepthField, DepthFieldId, DepthValueConventionName
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricVector
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.domain.surfaces import SurfaceIntendedUse
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate
from wre.reconstruction.open3d_surface import (
    OPEN3D_CAMERA_Z_CONVENTION,
    OPEN3D_PINHOLE_PROJECTION,
    OPEN3D_SURFACE_ADAPTER_ID,
    OPEN3D_SURFACE_DEPTH_SCALE,
    OPEN3D_SURFACE_DEVICE,
    OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH,
    OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
    OPEN3D_SURFACE_PRODUCER_VERSION,
    OPEN3D_SURFACE_REPRESENTATION,
    OPEN3D_SURFACE_SOURCE_REVISION,
    OPEN3D_SURFACE_WHEEL_BYTE_LENGTH,
    OPEN3D_SURFACE_WHEEL_FILENAME,
    OPEN3D_SURFACE_WHEEL_SHA256,
    Open3dSurfaceEnvironmentError,
    Open3dSurfaceEnvironmentIdentity,
    Open3dSurfaceError,
    Open3dSurfaceResult,
    Open3dTsdfSurfaceAdapter,
    Open3dTsdfSurfaceConfig,
    inspect_open3d_surface_environment,
)


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _sha(token: str) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.open3d_surface.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(sha256=_sha(token)),
    )


def _artifact(identifier: str, kind: str = "evidence.source") -> ArtifactRef:
    return ArtifactRef(ArtifactId(identifier), ArtifactKind(kind))


def _camera(
    token: str,
    *,
    frame: LocalFrameId,
    projection: str = "pinhole",
    intrinsics: tuple[float, ...] = (4.0, 5.0, 0.75, 0.5),
    rotation: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    width: int = 2,
    height: int = 2,
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=width, height_px=height),
        intrinsic_parameters=intrinsics,
        rotation_matrix=cast(Any, rotation),
        translation_xyz=translation,
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _candidate(
    token: str = "scene",
    *,
    cameras: tuple[CameraSolution, ...] | None = None,
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    source_artifacts: tuple[ArtifactRef, ...] | None = None,
) -> GeometrySolutionCandidate:
    frame = LocalFrameId(f"frame:{token}")
    candidate_cameras = cameras or (
        _camera("a", frame=frame),
        _camera(
            "b",
            frame=frame,
            intrinsics=(6.0, 7.0, 0.25, 0.75),
            translation=(0.5, -0.25, 0.125),
        ),
    )
    candidate_cameras = tuple(sorted(candidate_cameras, key=lambda item: item.solution_id.value))
    point_map = PointMap(
        point_map_id=PointMapId(f"points:{token}"),
        local_frame_id=frame,
        source_observation_ids=tuple(camera.observation_id for camera in candidate_cameras),
        positions_xyz=((0.0, 0.0, 1.0),),
        confidence=None,
        metrics=_metrics(),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=tuple(camera.solution_id for camera in candidate_cameras),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=candidate_cameras,
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=source_artifacts
        or (
            _artifact(f"source:{token}:a"),
            _artifact(f"source:{token}:b"),
        ),
    )


def _depth_field(
    camera: CameraSolution,
    token: str,
    *,
    convention: str = "camera-z",
    values: tuple[float, ...] = (1.0, 2.0, 0.0, 4.0),
    validity: tuple[bool, ...] = (True, True, False, True),
    confidence: tuple[float, ...] | None = (0.01, 0.25, 0.0, 1.0),
) -> DepthField:
    return DepthField(
        depth_field_id=DepthFieldId(f"depth:{token}"),
        observation_id=camera.observation_id,
        camera_solution_id=camera.solution_id,
        dimensions=camera.dimensions,
        depth_value_convention=DepthValueConventionName(convention),
        depth_values=values,
        validity=validity,
        confidence=confidence,
        metrics=_metrics(),
    )


def _dense(
    source: GeometrySolutionCandidate,
    *,
    fields_value: tuple[DepthField, ...] | None = None,
    extra_source: ArtifactRef | None = None,
) -> DenseDepthArtifact:
    depth_fields = fields_value or tuple(
        _depth_field(camera, camera.solution_id.value) for camera in source.camera_solutions
    )
    source_artifacts = (
        *source.source_artifacts,
        extra_source or _artifact("source:dense-extra", "evidence.depth_support"),
    )
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            ArtifactId("dense:surface-input"),
            DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=depth_fields,
        producer=_producer("dense"),
        source_artifacts=source_artifacts,
    )


def _environment(token: str = "environment") -> Open3dSurfaceEnvironmentIdentity:
    return Open3dSurfaceEnvironmentIdentity(manifest_sha256=_sha(token))


def _config() -> Open3dTsdfSurfaceConfig:
    return Open3dTsdfSurfaceConfig(
        voxel_size=0.05,
        depth_max=8.0,
        trunc_voxel_multiplier=6.0,
        block_resolution=8,
        block_count=512,
        mesh_weight_threshold=1.5,
    )


class _FakeArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def reshape(self, _shape: int) -> _FakeArray:
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


def _flatten(value: object) -> list[float]:
    if isinstance(value, (tuple, list)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    return [float(cast(Any, value))]


def _shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (tuple, list)):
        if not value:
            return (0,)
        return (len(value), *_shape(value[0]))
    return ()


class _FakeTensor:
    def __init__(
        self,
        data: object,
        *,
        dtype: object | None = None,
        device: object | None = None,
    ) -> None:
        self.data = data
        self.dtype = dtype
        self.device = device
        self.shape = _shape(data)

    def reshape(self, shape: tuple[int, ...]) -> _FakeTensor:
        self.shape = shape
        return self

    def numpy(self) -> _FakeArray:
        return _FakeArray(_flatten(self.data))


class _FakeDevice:
    def __init__(self, value: str) -> None:
        self.value = value

    def __str__(self) -> str:
        return self.value


class _FakeImage:
    def __init__(self, tensor: _FakeTensor) -> None:
        self.tensor = tensor


class _FakeMeshAttribute:
    def __init__(self, data: object, shape: tuple[int, ...]) -> None:
        self._data = data
        self.shape = shape

    def numpy(self) -> _FakeArray:
        return _FakeArray(_flatten(self._data))


class _FakeMesh:
    def __init__(
        self,
        *,
        vertices: object = ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
        triangles: object = ((0, 1, 2),),
    ) -> None:
        vertex_count = len(cast(Any, vertices))
        triangle_count = len(cast(Any, triangles))
        self.vertex = SimpleNamespace(positions=_FakeMeshAttribute(vertices, (vertex_count, 3)))
        self.triangle = SimpleNamespace(indices=_FakeMeshAttribute(triangles, (triangle_count, 3)))


class _FakeVoxelBlockGrid:
    def __init__(self, owner: _FakeOpen3d, **kwargs: object) -> None:
        self.owner = owner
        self.kwargs = kwargs
        owner.calls.append(("VoxelBlockGrid", kwargs))

    def compute_unique_block_coordinates(self, *args: object) -> object:
        self.owner.calls.append(("compute_unique_block_coordinates", args))
        return ("blocks", len(self.owner.calls))

    def integrate(self, *args: object) -> None:
        self.owner.calls.append(("integrate", args))

    def extract_triangle_mesh(self, *, weight_threshold: float) -> _FakeMesh:
        self.owner.calls.append(("extract_triangle_mesh", weight_threshold))
        return self.owner.mesh


class _FakeOpen3d:
    __version__ = "0.20.0"

    def __init__(
        self,
        *,
        mesh: _FakeMesh | None = None,
        write_success: bool = True,
        write_bytes: bytes = b"ply\nformat binary_little_endian 1.0\nend_header\nmesh",
    ) -> None:
        self.calls: list[tuple[str, object]] = []
        self.mesh = mesh or _FakeMesh()
        self.write_success = write_success
        self.write_bytes = write_bytes

        owner = self

        class VoxelBlockGrid:
            def __new__(cls, **kwargs: object) -> _FakeVoxelBlockGrid:
                return _FakeVoxelBlockGrid(owner, **kwargs)

        def write_triangle_mesh(path: str, mesh_value: object, **kwargs: object) -> bool:
            owner.calls.append(
                (
                    "write_triangle_mesh",
                    {
                        "path": path,
                        "mesh": mesh_value,
                        **kwargs,
                    },
                )
            )
            if owner.write_success:
                Path(path).write_bytes(owner.write_bytes)
            return owner.write_success

        self.core = SimpleNamespace(
            Device=_FakeDevice,
            Tensor=_FakeTensor,
            float32="float32",
            float64="float64",
        )
        self.t = SimpleNamespace(
            geometry=SimpleNamespace(
                VoxelBlockGrid=VoxelBlockGrid,
                Image=_FakeImage,
            ),
            io=SimpleNamespace(write_triangle_mesh=write_triangle_mesh),
        )


def _adapter(tmp_path: Path, module: _FakeOpen3d) -> Open3dTsdfSurfaceAdapter:
    return Open3dTsdfSurfaceAdapter(
        output_root=tmp_path / "surface-output",
        expected_environment=_environment(),
        config=_config(),
        module=module,
    )


def test_constants_and_frozen_shapes_are_exact() -> None:
    config = _config()
    environment = _environment()
    source = _candidate()
    result = _adapter(Path("/tmp"), _FakeOpen3d())

    assert OPEN3D_SURFACE_ADAPTER_ID == "open3d.depth_tsdf_surface"
    assert OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION == "open3d.t.geometry.VoxelBlockGrid.depth_only"
    assert OPEN3D_SURFACE_PRODUCER_VERSION == "0.20.0"
    assert OPEN3D_SURFACE_SOURCE_REVISION == "b6c5e196384ad71e75b6e6f9c5da22d046221f1d"
    assert OPEN3D_SURFACE_WHEEL_FILENAME == ("open3d-0.20.0-cp312-cp312-manylinux_2_35_x86_64.whl")
    assert OPEN3D_SURFACE_WHEEL_SHA256.value == (
        "f5cc6106d9c0c41beb8160aa99c2b663e7f08588a598ec7d464391a4cfe684d6"
    )
    assert OPEN3D_SURFACE_WHEEL_BYTE_LENGTH == 400_788_891
    assert OPEN3D_SURFACE_DEVICE == "CPU:0"
    assert OPEN3D_SURFACE_DEPTH_SCALE == 1.0
    assert OPEN3D_PINHOLE_PROJECTION.value == "pinhole"
    assert OPEN3D_CAMERA_Z_CONVENTION.value == "camera-z"
    assert OPEN3D_SURFACE_REPRESENTATION.value == "mesh"
    assert tuple(field.name for field in fields(Open3dTsdfSurfaceConfig)) == (
        "voxel_size",
        "depth_max",
        "trunc_voxel_multiplier",
        "block_resolution",
        "block_count",
        "mesh_weight_threshold",
        "schema_version",
        "device",
        "depth_scale",
    )
    assert next(field.name for field in fields(Open3dSurfaceEnvironmentIdentity)) == (
        "manifest_sha256"
    )
    assert source.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.config is not config

    with pytest.raises(FrozenInstanceError):
        config.depth_max = 3.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        environment.device = "CUDA:0"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("voxel_size", 0.0, "voxel_size"),
        ("voxel_size", float("inf"), "voxel_size"),
        ("depth_max", -1.0, "depth_max"),
        ("trunc_voxel_multiplier", 0.0, "trunc_voxel_multiplier"),
        ("block_resolution", 0, "block_resolution"),
        ("block_count", 0, "block_count"),
        ("mesh_weight_threshold", 0.0, "mesh_weight_threshold"),
        ("device", "CUDA:0", "CPU:0"),
        ("depth_scale", 1000.0, "1.0"),
    ],
)
def test_config_fails_closed_on_unreviewed_values(
    field_name: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "voxel_size": 0.05,
        "depth_max": 8.0,
        "trunc_voxel_multiplier": 6.0,
        "block_resolution": 8,
        "block_count": 512,
        "mesh_weight_threshold": 1.5,
    }
    kwargs[field_name] = value
    with pytest.raises((TypeError, ValueError), match=message):
        Open3dTsdfSurfaceConfig(**cast(Any, kwargs))


def test_config_hash_is_canonical_and_sensitive_to_local_unit_parameters() -> None:
    first = _config()
    same = _config()
    changed = Open3dTsdfSurfaceConfig(
        voxel_size=0.1,
        depth_max=8.0,
        trunc_voxel_multiplier=6.0,
        block_resolution=8,
        block_count=512,
        mesh_weight_threshold=1.5,
    )

    assert first.sha256 == same.sha256
    assert first.sha256 != changed.sha256
    assert first.canonical_document()["device"] == "CPU:0"
    assert first.canonical_document()["depth_scale"] == 1.0


def test_environment_requires_exact_open3d_wheel_identity() -> None:
    environment = _environment()
    assert environment.open3d_version == "0.20.0"
    assert environment.source_revision == OPEN3D_SURFACE_SOURCE_REVISION
    assert environment.wheel_sha256 == OPEN3D_SURFACE_WHEEL_SHA256
    assert environment.wheel_byte_length == OPEN3D_SURFACE_WHEEL_BYTE_LENGTH

    with pytest.raises(ValueError, match="wheel byte length"):
        Open3dSurfaceEnvironmentIdentity(
            manifest_sha256=_sha("env"),
            wheel_byte_length=1,
        )


def test_environment_inspection_requires_exact_tensor_tsdf_api() -> None:
    environment = _environment()
    module = _FakeOpen3d()
    assert inspect_open3d_surface_environment(environment, module) is environment

    wrong_version = _FakeOpen3d()
    wrong_version.__version__ = "0.20.1"
    with pytest.raises(Open3dSurfaceEnvironmentError, match=r"0\.20\.0"):
        inspect_open3d_surface_environment(environment, wrong_version)

    broken = _FakeOpen3d()
    del broken.t.io.write_triangle_mesh
    with pytest.raises(Open3dSurfaceEnvironmentError, match="tensor TSDF API"):
        inspect_open3d_surface_environment(environment, broken)


def test_real_call_boundary_forwards_depth_intrinsics_and_camera_from_local_pose(
    tmp_path: Path,
) -> None:
    frame = LocalFrameId("frame:forward")
    camera_a = _camera(
        "a",
        frame=frame,
        intrinsics=(4.0, 5.0, 0.75, 0.5),
        translation=(0.25, -0.5, 1.25),
    )
    camera_b = _camera(
        "b",
        frame=frame,
        intrinsics=(6.0, 7.0, 0.25, 0.75),
        translation=(-1.0, 0.5, 0.25),
    )
    source = _candidate("forward", cameras=(camera_a, camera_b))
    first = _depth_field(
        camera_a,
        "a",
        values=(1.0, 2.0, 0.0, 4.0),
        validity=(True, True, False, True),
        confidence=(0.0, 0.01, 0.0, 1.0),
    )
    second = _depth_field(
        camera_b,
        "b",
        values=(2.0, 0.0, 3.0, 5.0),
        validity=(True, False, True, True),
        confidence=None,
    )
    dense = _dense(source, fields_value=(first, second))
    module = _FakeOpen3d()

    original_fields = dense.depth_fields
    original_geometry = dense.source_geometry
    result = _adapter(tmp_path, module).derive(
        dense,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )

    vbg_call = module.calls[0]
    assert vbg_call[0] == "VoxelBlockGrid"
    assert cast(dict[str, object], vbg_call[1]) == {
        "attr_names": ("tsdf", "weight"),
        "attr_dtypes": ("float32", "float32"),
        "attr_channels": ((1,), (1,)),
        "voxel_size": 0.05,
        "block_resolution": 8,
        "block_count": 512,
        "device": cast(Any, cast(dict[str, object], vbg_call[1])["device"]),
    }
    assert str(cast(dict[str, object], vbg_call[1])["device"]) == "CPU:0"

    compute_calls = [
        value for name, value in module.calls if name == "compute_unique_block_coordinates"
    ]
    integrate_calls = [value for name, value in module.calls if name == "integrate"]
    assert len(compute_calls) == 2
    assert len(integrate_calls) == 2

    first_compute = cast(tuple[object, ...], compute_calls[0])
    first_depth = cast(_FakeImage, first_compute[0])
    first_intrinsic = cast(_FakeTensor, first_compute[1])
    first_extrinsic = cast(_FakeTensor, first_compute[2])
    assert first_depth.tensor.shape == (2, 2)
    assert first_depth.tensor.data == (1.0, 2.0, 0.0, 4.0)
    assert first_intrinsic.data == (
        (4.0, 0.0, 0.75),
        (0.0, 5.0, 0.5),
        (0.0, 0.0, 1.0),
    )
    assert first_extrinsic.data == (
        (1.0, 0.0, 0.0, 0.25),
        (0.0, 1.0, 0.0, -0.5),
        (0.0, 0.0, 1.0, 1.25),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert first_compute[3:] == (1.0, 8.0, 6.0)
    assert cast(tuple[object, ...], integrate_calls[0])[4:] == (1.0, 8.0, 6.0)

    # Confidence 0.0 on a valid pixel does not remove depth support.
    assert first_depth.tensor.data[0] == 1.0
    # Invalid support stays the canonical exact zero hole.
    assert first_depth.tensor.data[2] == 0.0

    write_call = next(value for name, value in module.calls if name == "write_triangle_mesh")
    write = cast(dict[str, object], write_call)
    assert Path(cast(str, write["path"])).name == "surface.ply"
    assert write["write_vertex_normals"] is False
    assert write["write_vertex_colors"] is False
    assert write["write_triangle_uvs"] is False
    assert write["compressed"] is False

    surface = result.surface_model
    assert surface.source_geometry is source
    assert surface.local_frame_id == source.geometry_solution.local_frame_id
    assert surface.scale_status is GeometryScaleStatus.UNRESOLVED
    assert surface.representation == OPEN3D_SURFACE_REPRESENTATION
    assert surface.intended_uses == (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.NAVIGATION,
    )
    expected_sources = tuple(
        sorted(
            {
                *(source.source_artifacts),
                dense.artifact_ref,
                *(dense.source_artifacts),
            },
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    assert surface.source_artifacts == expected_sources
    assert dense.depth_fields is original_fields
    assert dense.source_geometry is original_geometry

    output_path = tmp_path / "surface-output" / OPEN3D_SURFACE_OUTPUT_RELATIVE_PATH
    materialized_hash = hash_file_content(output_path)
    assert result.materialization.artifact_ref == surface.artifact_ref
    assert result.materialization.entries[0].relative_path == "surface.ply"
    assert result.materialization.entries[0].sha256 == materialized_hash.sha256
    assert result.materialization.entries[0].byte_length == materialized_hash.byte_length
    assert tuple(path.name for path in (tmp_path / "surface-output").iterdir()) == ("surface.ply",)

    forbidden_call_names = {
        "color",
        "rgbd",
        "registration",
        "odometry",
        "icp",
        "slam",
        "poisson",
    }
    assert not any(name.lower() in forbidden_call_names for name, _ in module.calls)


def test_producer_identity_binds_config_and_exact_external_environment(tmp_path: Path) -> None:
    source = _candidate("producer")
    dense = _dense(source)
    first = Open3dTsdfSurfaceAdapter(
        output_root=tmp_path / "first",
        expected_environment=_environment("env-a"),
        config=_config(),
        module=_FakeOpen3d(write_bytes=b"same-mesh"),
    ).derive(dense, intended_uses=(SurfaceIntendedUse.MEASUREMENT,))
    second = Open3dTsdfSurfaceAdapter(
        output_root=tmp_path / "second",
        expected_environment=_environment("env-b"),
        config=_config(),
        module=_FakeOpen3d(write_bytes=b"same-mesh"),
    ).derive(dense, intended_uses=(SurfaceIntendedUse.MEASUREMENT,))

    assert first.surface_model.producer.producer.implementation == (
        OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION
    )
    assert first.surface_model.producer.producer.version == OPEN3D_SURFACE_PRODUCER_VERSION
    assert first.surface_model.producer.producer.revision == OPEN3D_SURFACE_SOURCE_REVISION
    assert first.surface_model.producer.configuration != second.surface_model.producer.configuration
    assert first.surface_model.artifact_ref != second.surface_model.artifact_ref


@pytest.mark.parametrize(
    ("projection", "intrinsics", "convention", "message"),
    [
        ("simple_radial", (4.0, 5.0, 0.75, 0.5), "camera-z", "pinhole"),
        ("pinhole", (4.0, 5.0, 0.75, 0.5, 0.01), "camera-z", "fx fy cx cy"),
        ("pinhole", (4.0, 5.0, 0.75, 0.5), "ray-distance", "camera-z"),
    ],
)
def test_adapter_rejects_unsupported_camera_or_depth_semantics(
    tmp_path: Path,
    projection: str,
    intrinsics: tuple[float, ...],
    convention: str,
    message: str,
) -> None:
    frame = LocalFrameId("frame:unsupported")
    camera = _camera(
        "unsupported",
        frame=frame,
        projection=projection,
        intrinsics=intrinsics,
    )
    source = _candidate("unsupported", cameras=(camera,))
    field = _depth_field(camera, "unsupported", convention=convention)
    dense = _dense(source, fields_value=(field,))

    with pytest.raises(Open3dSurfaceError, match=message):
        _adapter(tmp_path, _FakeOpen3d()).derive(
            dense,
            intended_uses=(SurfaceIntendedUse.COLLISION,),
        )
    assert not (tmp_path / "surface-output").exists()


@pytest.mark.parametrize(
    "mesh",
    [
        _FakeMesh(vertices=(), triangles=()),
        _FakeMesh(triangles=()),
        _FakeMesh(
            vertices=((0.0, 0.0, 1.0), (float("nan"), 0.0, 1.0), (0.0, 1.0, 1.0)),
        ),
    ],
)
def test_adapter_fails_closed_on_empty_or_nonfinite_mesh_and_cleans_output(
    tmp_path: Path,
    mesh: _FakeMesh,
) -> None:
    source = _candidate("bad-mesh")
    dense = _dense(source)

    with pytest.raises(Open3dSurfaceError, match="finite vertices"):
        _adapter(tmp_path, _FakeOpen3d(mesh=mesh)).derive(
            dense,
            intended_uses=(SurfaceIntendedUse.COLLISION,),
        )
    assert not (tmp_path / "surface-output").exists()


def test_adapter_cleans_output_when_geometry_only_writer_fails(tmp_path: Path) -> None:
    source = _candidate("writer")
    dense = _dense(source)

    with pytest.raises(Open3dSurfaceError, match="publish"):
        _adapter(tmp_path, _FakeOpen3d(write_success=False)).derive(
            dense,
            intended_uses=(SurfaceIntendedUse.COLLISION,),
        )
    assert not (tmp_path / "surface-output").exists()


def test_adapter_rejects_existing_or_symlink_output_root(tmp_path: Path) -> None:
    source = _candidate("paths")
    dense = _dense(source)

    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        Open3dTsdfSurfaceAdapter(
            output_root=existing,
            expected_environment=_environment(),
            config=_config(),
            module=_FakeOpen3d(),
        ).derive(dense, intended_uses=(SurfaceIntendedUse.COLLISION,))

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(Open3dSurfaceError, match="symlink"):
        Open3dTsdfSurfaceAdapter(
            output_root=link / "surface",
            expected_environment=_environment(),
            config=_config(),
            module=_FakeOpen3d(),
        ).derive(dense, intended_uses=(SurfaceIntendedUse.COLLISION,))


def test_intended_use_remains_metadata_not_suitability(tmp_path: Path) -> None:
    source = _candidate("intent", scale=GeometryScaleStatus.UNRESOLVED)
    dense = _dense(source)
    result = _adapter(tmp_path, _FakeOpen3d()).derive(
        dense,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
    )

    surface = result.surface_model
    assert surface.scale_status is GeometryScaleStatus.UNRESOLVED
    forbidden_fields = {
        "suitability",
        "score",
        "quality_decision",
        "watertight",
        "collision_mesh",
        "navmesh",
        "measurement_ready",
    }
    assert not forbidden_fields.intersection(field.name for field in fields(type(surface)))


def test_result_requires_materialization_bound_to_same_surface_artifact(tmp_path: Path) -> None:
    source = _candidate("result")
    dense = _dense(source)
    result = _adapter(tmp_path, _FakeOpen3d()).derive(
        dense,
        intended_uses=(SurfaceIntendedUse.COLLISION,),
    )

    other = cast(
        Any,
        type(result.materialization)(
            artifact_ref=ArtifactRef(
                ArtifactId("surface:other"),
                result.surface_model.artifact_ref.artifact_kind,
            ),
            entries=result.materialization.entries,
        ),
    )
    with pytest.raises(ValueError, match="produced SurfaceModel"):
        Open3dSurfaceResult(
            surface_model=result.surface_model,
            materialization=other,
        )
