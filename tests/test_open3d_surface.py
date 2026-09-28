from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.open3d_surface as surface_module
from wre.domain.artifact_materialization import ArtifactMaterializationMetadata
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
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import ProducerRef
from wre.domain.surfaces import (
    SURFACE_MODEL_ARTIFACT_KIND,
    SurfaceIntendedUse,
    SurfaceRepresentationName,
)
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND, DenseDepthArtifact
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate
from wre.reconstruction.open3d_surface import (
    OPEN3D_SURFACE_ADAPTER_ID,
    OPEN3D_SURFACE_CAPABILITY,
    OPEN3D_SURFACE_CAPABILITY_NAME,
    OPEN3D_SURFACE_DEPENDENCY_REF,
    OPEN3D_SURFACE_DEVICE,
    OPEN3D_SURFACE_ENVIRONMENT,
    OPEN3D_SURFACE_ENVIRONMENT_ARTIFACT_COUNT,
    OPEN3D_SURFACE_ENVIRONMENT_BYTE_LENGTH,
    OPEN3D_SURFACE_ENVIRONMENT_LOCK_SHA256,
    OPEN3D_SURFACE_MESH_PATH,
    OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
    OPEN3D_SURFACE_PRODUCER_VERSION,
    OPEN3D_SURFACE_ROOT_WHEEL,
    OPEN3D_SURFACE_ROOT_WHEEL_BYTE_LENGTH,
    OPEN3D_SURFACE_ROOT_WHEEL_SHA256,
    OPEN3D_SURFACE_SHIPPING_STATUS,
    OPEN3D_SURFACE_SOURCE_REVISION,
    Open3dSurfaceEnvironmentError,
    Open3dSurfaceError,
    Open3dSurfaceResult,
    Open3dTsdfSurfaceAdapter,
    Open3dTsdfSurfaceConfig,
)


_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"


def _entries_by_id() -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(_REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    entries = document["entries"]
    assert isinstance(entries, list)
    return {
        cast(str, entry["adapter_id"]): cast(dict[str, Any], entry)
        for entry in entries
        if isinstance(entry, dict)
    }


def _metrics() -> MetricVector:
    return MetricVector(observations=())


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _camera(
    token: str,
    *,
    frame: LocalFrameId,
    projection: str = "pinhole",
    intrinsics: tuple[float, ...] = (4.0, 5.0, 1.5, 1.0),
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> CameraSolution:
    return CameraSolution(
        solution_id=CameraSolutionId(f"camera:{token}"),
        observation_id=ObservationId(f"obs:{token}"),
        local_frame_id=frame,
        projection_model=CameraProjectionModelName(projection),
        dimensions=ImageDimensions(width_px=3, height_px=2),
        intrinsic_parameters=intrinsics,
        rotation_matrix=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        translation_xyz=translation,
        uncertainty_artifacts=(),
        metrics=_metrics(),
    )


def _candidate(
    token: str,
    *,
    cameras: tuple[CameraSolution, ...],
    depth_fields: tuple[DepthField, ...],
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    source_artifacts: tuple[ArtifactRef, ...],
) -> GeometrySolutionCandidate:
    frame = cameras[0].local_frame_id
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId(f"geometry:{token}"),
        local_frame_id=frame,
        scale_status=scale,
        camera_solution_ids=tuple(camera.solution_id for camera in cameras),
        depth_field_ids=tuple(depth.depth_field_id for depth in depth_fields),
        point_map_ids=(),
        metrics=_metrics(),
    )
    return GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=cameras,
        depth_fields=depth_fields,
        point_maps=(),
        producer=_producer(f"geometry:{token}"),
        source_artifacts=source_artifacts,
    )


def _depth(
    camera: CameraSolution,
    token: str,
    *,
    convention: str = "camera-z",
    confidence: tuple[float, ...] | None = None,
) -> DepthField:
    values = (1.0, 1.2, 0.0, 1.4, 1.6, 1.8)
    validity = (True, True, False, True, True, True)
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
    token: str = "surface",
    *,
    projection: str = "pinhole",
    intrinsics: tuple[float, ...] = (4.0, 5.0, 1.5, 1.0),
    convention: str = "camera-z",
    scale: GeometryScaleStatus = GeometryScaleStatus.UNRESOLVED,
    confidence_a: tuple[float, ...] | None = None,
    confidence_b: tuple[float, ...] | None = None,
) -> DenseDepthArtifact:
    frame = LocalFrameId(f"frame:{token}")
    camera_a = _camera(
        "a",
        frame=frame,
        projection=projection,
        intrinsics=intrinsics,
        translation=(0.25, -0.5, 1.5),
    )
    camera_b = _camera(
        "b",
        frame=frame,
        projection=projection,
        intrinsics=intrinsics,
        translation=(-1.0, 0.75, 2.0),
    )
    depth_a = _depth(
        camera_a,
        "a",
        convention=convention,
        confidence=confidence_a,
    )
    depth_b = _depth(
        camera_b,
        "b",
        convention=convention,
        confidence=confidence_b,
    )
    ancestor = _artifact("artifact:geometry", "geometry.solution")
    source = _candidate(
        token,
        cameras=(camera_a, camera_b),
        depth_fields=(depth_a, depth_b),
        scale=scale,
        source_artifacts=(ancestor,),
    )
    evidence = _artifact("artifact:dense-evidence", "evidence.depth")
    sources = tuple(
        sorted(
            (ancestor, evidence),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    return DenseDepthArtifact(
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId("artifact:dense"),
            artifact_kind=DENSE_DEPTH_ARTIFACT_KIND,
        ),
        source_geometry=source,
        depth_fields=(depth_a, depth_b),
        producer=_producer("dense"),
        source_artifacts=sources,
    )


def _config() -> Open3dTsdfSurfaceConfig:
    return Open3dTsdfSurfaceConfig(
        voxel_size_local_units=0.05,
        max_depth_local_units=4.0,
        trunc_voxel_multiplier=4.0,
        block_resolution=8,
        block_count=256,
        mesh_weight_threshold=1.0,
    )


class _FakeArray:
    def __init__(self, values: list[float]) -> None:
        self.values = values

    def reshape(self, _size: int) -> _FakeArray:
        return self

    def tolist(self) -> list[float]:
        return list(self.values)


class _FakeTensor:
    def __init__(self, values: object, *, dtype: object, device: object) -> None:
        self.values = values
        self.dtype = dtype
        self.device = device
        self.shape = self._shape(values)

    @staticmethod
    def _shape(values: object) -> tuple[int, ...]:
        if not isinstance(values, (tuple, list)):
            return ()
        if not values:
            return (0,)
        first = values[0]
        if isinstance(first, (tuple, list)):
            if not first:
                return (len(values), 0)
            if isinstance(first[0], (tuple, list)):
                return (len(values), len(first), len(first[0]))
            return (len(values), len(first))
        return (len(values),)

    @staticmethod
    def _flatten(values: object) -> list[float]:
        if isinstance(values, (tuple, list)):
            result: list[float] = []
            for item in values:
                result.extend(_FakeTensor._flatten(item))
            return result
        return [float(cast(float | int, values))]

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> _FakeArray:
        return _FakeArray(self._flatten(self.values))


class _FakeImage:
    def __init__(self, tensor: _FakeTensor) -> None:
        self.tensor = tensor


class _FakeMesh:
    def __init__(
        self,
        *,
        positions: tuple[tuple[float, float, float], ...],
        triangles: tuple[tuple[int, int, int], ...],
    ) -> None:
        self.vertex = SimpleNamespace(
            positions=_FakeTensor(positions, dtype="float32", device="CPU:0")
        )
        self.triangle = SimpleNamespace(
            indices=_FakeTensor(triangles, dtype="int32", device="CPU:0")
        )


class _FakeVoxelBlockGrid:
    def __init__(self, module: _FakeOpen3d, **kwargs: object) -> None:
        self.module = module
        self.kwargs = kwargs
        self.compute_calls: list[tuple[object, object, object, dict[str, object]]] = []
        self.integrate_calls: list[tuple[object, object, object, object, dict[str, object]]] = []
        self.extract_calls: list[dict[str, object]] = []

    def compute_unique_block_coordinates(
        self,
        depth: object,
        intrinsic: object,
        extrinsic: object,
        **kwargs: object,
    ) -> str:
        self.compute_calls.append((depth, intrinsic, extrinsic, kwargs))
        return f"blocks:{len(self.compute_calls)}"

    def integrate(
        self,
        block_coords: object,
        depth: object,
        intrinsic: object,
        extrinsic: object,
        **kwargs: object,
    ) -> None:
        self.integrate_calls.append((block_coords, depth, intrinsic, extrinsic, kwargs))
        if self.module.fail_integrate:
            raise RuntimeError("synthetic Open3D integration failure")

    def extract_triangle_mesh(self, **kwargs: object) -> _FakeMesh:
        self.extract_calls.append(kwargs)
        return _FakeMesh(
            positions=self.module.mesh_positions,
            triangles=self.module.mesh_triangles,
        )


class _FakeOpen3d:
    __version__ = "0.20.0"

    def __init__(
        self,
        *,
        mesh_positions: tuple[tuple[float, float, float], ...] = (
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        mesh_triangles: tuple[tuple[int, int, int], ...] = ((0, 1, 2),),
        fail_integrate: bool = False,
        writer_result: bool = True,
        add_color_property: bool = False,
        extra_output: bool = False,
    ) -> None:
        self.mesh_positions = mesh_positions
        self.mesh_triangles = mesh_triangles
        self.fail_integrate = fail_integrate
        self.writer_result = writer_result
        self.add_color_property = add_color_property
        self.extra_output = extra_output
        self.tensor_calls: list[_FakeTensor] = []
        self.device_calls: list[str] = []
        self.image_calls: list[_FakeImage] = []
        self.voxel_grids: list[_FakeVoxelBlockGrid] = []
        self.write_calls: list[tuple[str, object, dict[str, object]]] = []

        self.core = SimpleNamespace(
            float32="float32",
            float64="float64",
            Device=self._device,
            Tensor=self._tensor,
        )
        self.t = SimpleNamespace(
            geometry=SimpleNamespace(
                Image=self._image,
                VoxelBlockGrid=self._voxel_block_grid,
            ),
            io=SimpleNamespace(write_triangle_mesh=self._write_triangle_mesh),
        )

    def _device(self, value: str) -> str:
        self.device_calls.append(value)
        return value

    def _tensor(
        self,
        values: object,
        *,
        dtype: object,
        device: object,
    ) -> _FakeTensor:
        tensor = _FakeTensor(values, dtype=dtype, device=device)
        self.tensor_calls.append(tensor)
        return tensor

    def _image(self, tensor: _FakeTensor) -> _FakeImage:
        image = _FakeImage(tensor)
        self.image_calls.append(image)
        return image

    def _voxel_block_grid(self, **kwargs: object) -> _FakeVoxelBlockGrid:
        grid = _FakeVoxelBlockGrid(self, **kwargs)
        self.voxel_grids.append(grid)
        return grid

    def _write_triangle_mesh(
        self,
        filename: str,
        mesh: object,
        **kwargs: object,
    ) -> bool:
        self.write_calls.append((filename, mesh, kwargs))
        if not self.writer_result:
            return False

        header = [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 3",
            "property float x",
            "property float y",
            "property float z",
        ]
        if self.add_color_property:
            header.append("property uchar red")
        header.extend(
            (
                "element face 1",
                "property list uchar int vertex_indices",
                "end_header",
                "",
            )
        )
        path = Path(filename)
        path.write_bytes("\n".join(header).encode("ascii") + b"synthetic-mesh")
        if self.extra_output:
            (path.parent / "unexpected.txt").write_text("unexpected", encoding="utf-8")
        return True


def _adapter(
    tmp_path: Path,
    *,
    source: DenseDepthArtifact | None = None,
    module: object | None = None,
    intended_uses: tuple[SurfaceIntendedUse, ...] = (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.NAVIGATION,
    ),
) -> Open3dTsdfSurfaceAdapter:
    return Open3dTsdfSurfaceAdapter(
        source=source or _dense(),
        output_root=tmp_path / "surface-output",
        intended_uses=intended_uses,
        config=_config(),
        module=module or _FakeOpen3d(),
    )


def test_exact_constants_environment_and_capability_contract() -> None:
    assert OPEN3D_SURFACE_ADAPTER_ID == "open3d.tsdf_surface"
    assert OPEN3D_SURFACE_DEPENDENCY_REF == "open3d"
    assert OPEN3D_SURFACE_PRODUCER_VERSION == "0.20.0"
    assert OPEN3D_SURFACE_SOURCE_REVISION == "b6c5e196384ad71e75b6e6f9c5da22d046221f1d"
    assert OPEN3D_SURFACE_ROOT_WHEEL == ("open3d-0.20.0-cp312-cp312-manylinux_2_35_x86_64.whl")
    assert OPEN3D_SURFACE_ROOT_WHEEL_SHA256 == Sha256Digest(
        "f5cc6106d9c0c41beb8160aa99c2b663e7f08588a598ec7d464391a4cfe684d6"
    )
    assert OPEN3D_SURFACE_ROOT_WHEEL_BYTE_LENGTH == 400788891
    assert OPEN3D_SURFACE_ENVIRONMENT_LOCK_SHA256 == Sha256Digest(
        "ac9b2e1346e7b0a0bbed92eee22887e0e662cc303ea9f97df0e5fde835b07214"
    )
    assert OPEN3D_SURFACE_ENVIRONMENT_ARTIFACT_COUNT == 59
    assert OPEN3D_SURFACE_ENVIRONMENT_BYTE_LENGTH == 453384619
    assert OPEN3D_SURFACE_DEVICE == "CPU:0"
    assert OPEN3D_SURFACE_SHIPPING_STATUS == "experimental"
    assert OPEN3D_SURFACE_ENVIRONMENT.device == "CPU:0"
    assert isinstance(OPEN3D_SURFACE_ENVIRONMENT.sha256, Sha256Digest)

    assert OPEN3D_SURFACE_CAPABILITY.capability is OPEN3D_SURFACE_CAPABILITY_NAME
    assert OPEN3D_SURFACE_CAPABILITY.input_kinds == frozenset({DENSE_DEPTH_ARTIFACT_KIND})
    assert OPEN3D_SURFACE_CAPABILITY.output_kinds == frozenset({SURFACE_MODEL_ARTIFACT_KIND})


def test_config_has_exact_frozen_explicit_local_unit_shape_and_stable_hash() -> None:
    config = _config()
    assert tuple(item.name for item in fields(config)) == (
        "voxel_size_local_units",
        "max_depth_local_units",
        "trunc_voxel_multiplier",
        "block_resolution",
        "block_count",
        "mesh_weight_threshold",
        "schema_version",
        "device",
        "depth_scale",
    )
    assert config.device == "CPU:0"
    assert config.depth_scale == 1.0
    assert config.sha256 == _config().sha256

    changed = replace(config, voxel_size_local_units=0.06)
    assert changed.sha256 != config.sha256

    with pytest.raises(FrozenInstanceError):
        config.block_count = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"voxel_size_local_units": 0.0}, "positive and finite"),
        ({"voxel_size_local_units": float("inf")}, "positive and finite"),
        ({"max_depth_local_units": -1.0}, "positive and finite"),
        ({"trunc_voxel_multiplier": float("nan")}, "positive and finite"),
        ({"block_resolution": 0}, "positive integer"),
        ({"block_count": True}, "positive integer"),
        ({"mesh_weight_threshold": 0.0}, "positive and finite"),
        ({"schema_version": 2}, "schema_version"),
        ({"device": "CUDA:0"}, "CPU:0"),
        ({"depth_scale": 1000.0}, "exactly 1.0"),
    ],
)
def test_config_rejects_hidden_metric_gpu_and_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "voxel_size_local_units": 0.05,
        "max_depth_local_units": 4.0,
        "trunc_voxel_multiplier": 4.0,
        "block_resolution": 8,
        "block_count": 256,
        "mesh_weight_threshold": 1.0,
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        Open3dTsdfSurfaceConfig(**cast(Any, values))


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_dense(projection="simple-radial"), "pinhole"),
        (_dense(intrinsics=(4.0, 5.0, 1.5)), "exactly fx fy cx cy"),
        (_dense(convention="relative-depth"), "camera-z"),
    ],
)
def test_unsupported_source_semantics_fail_before_open3d_execution(
    tmp_path: Path,
    source: DenseDepthArtifact,
    message: str,
) -> None:
    module = _FakeOpen3d()
    with pytest.raises(Open3dSurfaceError, match=message):
        _adapter(tmp_path, source=source, module=module).derive()
    assert module.voxel_grids == []
    assert not (tmp_path / "surface-output").exists()


def test_wrong_open3d_version_fails_before_output_creation(tmp_path: Path) -> None:
    module = _FakeOpen3d()
    module.__version__ = "0.21.0"

    with pytest.raises(Open3dSurfaceEnvironmentError, match=r"exactly 0\\.20\\.0"):
        _adapter(tmp_path, module=module).derive()
    assert not (tmp_path / "surface-output").exists()


def test_success_forwards_exact_depth_intrinsics_extrinsics_and_depth_only_calls(
    tmp_path: Path,
) -> None:
    confidence_a = (0.9, 0.8, 0.0, 0.7, 0.6, 0.5)
    confidence_b = (0.1, 0.2, 0.0, 0.3, 0.4, 0.5)
    source = _dense(confidence_a=confidence_a, confidence_b=confidence_b)
    before_depths = source.depth_fields
    module = _FakeOpen3d()

    result = _adapter(tmp_path, source=source, module=module).derive()

    assert len(module.voxel_grids) == 1
    grid = module.voxel_grids[0]
    assert grid.kwargs == {
        "attr_names": ("tsdf", "weight"),
        "attr_dtypes": ("float32", "float32"),
        "attr_channels": (1, 1),
        "voxel_size": 0.05,
        "block_resolution": 8,
        "block_count": 256,
        "device": "CPU:0",
    }
    assert len(grid.compute_calls) == 2
    assert len(grid.integrate_calls) == 2
    assert grid.extract_calls == [{"weight_threshold": 1.0}]

    first_depth = cast(_FakeImage, grid.compute_calls[0][0]).tensor
    assert first_depth.dtype == "float32"
    assert first_depth.device == "CPU:0"
    assert first_depth.values == [
        [1.0, 1.2, 0.0],
        [1.4, 1.6, 1.8],
    ]

    first_intrinsic = cast(_FakeTensor, grid.compute_calls[0][1])
    assert first_intrinsic.values == (
        (4.0, 0.0, 1.5),
        (0.0, 5.0, 1.0),
        (0.0, 0.0, 1.0),
    )
    first_extrinsic = cast(_FakeTensor, grid.compute_calls[0][2])
    assert first_extrinsic.values == (
        (1.0, 0.0, 0.0, 0.25),
        (0.0, 1.0, 0.0, -0.5),
        (0.0, 0.0, 1.0, 1.5),
        (0.0, 0.0, 0.0, 1.0),
    )

    second_extrinsic = cast(_FakeTensor, grid.compute_calls[1][2])
    assert second_extrinsic.values == (
        (1.0, 0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0, 0.75),
        (0.0, 0.0, 1.0, 2.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    for compute_call, integrate_call in zip(
        grid.compute_calls,
        grid.integrate_calls,
        strict=True,
    ):
        assert compute_call[3] == {
            "depth_scale": 1.0,
            "depth_max": 4.0,
            "trunc_voxel_multiplier": 4.0,
        }
        assert integrate_call[4] == compute_call[3]
        assert integrate_call[1] is compute_call[0]
        assert integrate_call[2] is compute_call[1]
        assert integrate_call[3] is compute_call[2]

    assert source.depth_fields is before_depths
    assert source.depth_fields[0].confidence is confidence_a
    assert source.depth_fields[1].confidence is confidence_b
    assert result.surface_model.source_geometry is source.source_geometry


def test_confidence_never_changes_depth_support_or_open3d_calls(tmp_path: Path) -> None:
    low = _dense(
        token="low",
        confidence_a=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        confidence_b=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    high = _dense(
        token="high",
        confidence_a=(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
        confidence_b=(1.0, 1.0, 0.0, 1.0, 1.0, 1.0),
    )
    low_module = _FakeOpen3d()
    high_module = _FakeOpen3d()

    _adapter(tmp_path / "low", source=low, module=low_module).derive()
    _adapter(tmp_path / "high", source=high, module=high_module).derive()

    low_values = [
        cast(_FakeImage, call[0]).tensor.values for call in low_module.voxel_grids[0].compute_calls
    ]
    high_values = [
        cast(_FakeImage, call[0]).tensor.values for call in high_module.voxel_grids[0].compute_calls
    ]
    assert low_values == high_values


def test_geometry_only_surface_publication_preserves_exact_provenance_and_scale(
    tmp_path: Path,
) -> None:
    source = _dense(scale=GeometryScaleStatus.UNRESOLVED)
    module = _FakeOpen3d()
    adapter = _adapter(tmp_path, source=source, module=module)

    result = adapter.derive()

    assert isinstance(result, Open3dSurfaceResult)
    model = result.surface_model
    assert model.artifact_ref.artifact_kind == SURFACE_MODEL_ARTIFACT_KIND
    assert model.representation == SurfaceRepresentationName("mesh")
    assert model.intended_uses == (
        SurfaceIntendedUse.COLLISION,
        SurfaceIntendedUse.NAVIGATION,
    )
    assert model.source_geometry is source.source_geometry
    assert model.local_frame_id == source.source_geometry.geometry_solution.local_frame_id
    assert model.scale_status is GeometryScaleStatus.UNRESOLVED
    assert model.producer.producer == ProducerRef(
        implementation=OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
        version=OPEN3D_SURFACE_PRODUCER_VERSION,
        revision=OPEN3D_SURFACE_SOURCE_REVISION,
    )
    assert model.producer.configuration.sha256 != adapter.config.sha256

    expected_sources = tuple(
        sorted(
            {
                (item.artifact_id.value, item.artifact_kind.value): item
                for item in (
                    *source.source_geometry.source_artifacts,
                    source.artifact_ref,
                    *source.source_artifacts,
                )
            }.values(),
            key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
        )
    )
    assert model.source_artifacts == expected_sources
    assert source.artifact_ref in model.source_artifacts
    assert model.artifact_ref.artifact_id.value not in {
        item.artifact_id.value
        for item in (
            source.artifact_ref,
            *source.source_artifacts,
            *source.source_geometry.source_artifacts,
        )
    }

    assert isinstance(result.materialization, ArtifactMaterializationMetadata)
    assert result.materialization.artifact_ref == model.artifact_ref
    assert len(result.materialization.entries) == 1
    entry = result.materialization.entries[0]
    assert entry.relative_path == "surface.ply"
    payload = (adapter.output_root / "surface.ply").read_bytes()
    assert entry.byte_length == len(payload)
    assert entry.sha256 == Sha256Digest(hashlib.sha256(payload).hexdigest())

    assert len(module.write_calls) == 1
    _path, _mesh, kwargs = module.write_calls[0]
    assert kwargs == {
        "write_ascii": False,
        "compressed": False,
        "write_vertex_normals": False,
        "write_vertex_colors": False,
        "write_triangle_uvs": False,
        "print_progress": False,
    }
    header = payload.split(b"end_header", 1)[0].lower()
    assert b" red" not in header
    assert b" green" not in header
    assert b" blue" not in header
    assert b" nx" not in header
    assert b"texture" not in header


def test_metric_source_stays_metric_without_conversion(tmp_path: Path) -> None:
    source = _dense(scale=GeometryScaleStatus.METRIC)
    result = _adapter(tmp_path, source=source).derive()
    assert result.surface_model.scale_status is GeometryScaleStatus.METRIC


@pytest.mark.parametrize(
    ("positions", "triangles", "message"),
    [
        ((), ((0, 1, 2),), "vertex xyz"),
        (((float("nan"), 0.0, 1.0),), ((0, 0, 0),), "finite"),
        (((0.0, 0.0, 1.0),), (), "triangles"),
    ],
)
def test_empty_nonfinite_or_triangleless_mesh_fails_closed_and_cleans_output(
    tmp_path: Path,
    positions: tuple[tuple[float, float, float], ...],
    triangles: tuple[tuple[int, int, int], ...],
    message: str,
) -> None:
    module = _FakeOpen3d(mesh_positions=positions, mesh_triangles=triangles)
    output_root = tmp_path / "surface-output"

    with pytest.raises(Open3dSurfaceError, match=message):
        _adapter(tmp_path, module=module).derive()
    assert not output_root.exists()


def test_solver_or_writer_failure_cleans_only_adapter_owned_output(tmp_path: Path) -> None:
    integrate_module = _FakeOpen3d(fail_integrate=True)
    with pytest.raises(RuntimeError, match="synthetic Open3D integration failure"):
        _adapter(tmp_path / "integrate", module=integrate_module).derive()
    assert not (tmp_path / "integrate" / "surface-output").exists()

    writer_module = _FakeOpen3d(writer_result=False)
    with pytest.raises(Open3dSurfaceError, match="failed to write"):
        _adapter(tmp_path / "writer", module=writer_module).derive()
    assert not (tmp_path / "writer" / "surface-output").exists()


def test_existing_output_is_never_reused_or_deleted(tmp_path: Path) -> None:
    output_root = tmp_path / "surface-output"
    output_root.mkdir()
    sentinel = output_root / "sentinel.txt"
    sentinel.write_text("owned elsewhere", encoding="utf-8")
    module = _FakeOpen3d()

    with pytest.raises(ValueError, match="must not already exist"):
        _adapter(tmp_path, module=module).derive()

    assert sentinel.read_text(encoding="utf-8") == "owned elsewhere"
    assert module.voxel_grids == []


def test_symlink_output_path_is_rejected_without_touching_target(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("retained", encoding="utf-8")
    link = tmp_path / "surface-output"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")

    with pytest.raises(ValueError, match="must not already exist"):
        _adapter(tmp_path).derive()

    assert sentinel.read_text(encoding="utf-8") == "retained"


def test_extra_output_or_appearance_ply_fails_and_cleans_output(tmp_path: Path) -> None:
    extra = _FakeOpen3d(extra_output=True)
    with pytest.raises(Open3dSurfaceError, match=r"exactly surface\\.ply"):
        _adapter(tmp_path / "extra", module=extra).derive()
    assert not (tmp_path / "extra" / "surface-output").exists()

    appearance = _FakeOpen3d(add_color_property=True)
    with pytest.raises(Open3dSurfaceError, match="geometry only"):
        _adapter(tmp_path / "appearance", module=appearance).derive()
    assert not (tmp_path / "appearance" / "surface-output").exists()


def test_intended_use_is_metadata_not_suitability(tmp_path: Path) -> None:
    result = _adapter(
        tmp_path,
        intended_uses=(
            SurfaceIntendedUse.COLLISION,
            SurfaceIntendedUse.MEASUREMENT,
            SurfaceIntendedUse.NAVIGATION,
        ),
    ).derive()

    for attribute in (
        "collision_suitable",
        "measurement_suitable",
        "navigation_suitable",
        "suitability",
        "suitability_score",
        "threshold",
        "decision",
        "quality_decision",
        "watertight",
        "collision_mesh",
        "navmesh",
        "runtime_lod",
        "generated_regions",
        "hole_fill",
        "appearance",
    ):
        assert not hasattr(result.surface_model, attribute)


@pytest.mark.parametrize(
    ("uses", "message"),
    [
        (cast(Any, []), "immutable tuple"),
        ((), "non-empty"),
        (cast(Any, ("collision",)), "SurfaceIntendedUse"),
        (
            (SurfaceIntendedUse.COLLISION, SurfaceIntendedUse.COLLISION),
            "unique",
        ),
        (
            (SurfaceIntendedUse.NAVIGATION, SurfaceIntendedUse.COLLISION),
            "canonical",
        ),
    ],
)
def test_intended_use_contract_fails_before_execution(
    tmp_path: Path,
    uses: tuple[SurfaceIntendedUse, ...],
    message: str,
) -> None:
    module = _FakeOpen3d()
    with pytest.raises((TypeError, ValueError), match=message):
        Open3dTsdfSurfaceAdapter(
            source=_dense(),
            output_root=tmp_path / "surface",
            intended_uses=uses,
            config=_config(),
            module=module,
        )
    assert module.voxel_grids == []


def test_adapter_exposes_only_derive_as_public_execution_operation(tmp_path: Path) -> None:
    adapter = _adapter(tmp_path)
    public_callables = {
        name
        for name, value in type(adapter).__dict__.items()
        if not name.startswith("_") and callable(value)
    }
    assert public_callables == {"derive"}
    signature = inspect.signature(type(adapter).derive)
    assert tuple(signature.parameters) == ("self",)


def test_adapter_registry_contract_is_exact_experimental_open3d_020() -> None:
    entry = _entries_by_id()[OPEN3D_SURFACE_ADAPTER_ID]

    assert entry["capability"] == {
        "name": OPEN3D_SURFACE_CAPABILITY_NAME.value,
        "input_kinds": ["geometry.dense_depth"],
        "output_kinds": ["geometry.surface"],
    }
    assert entry["producer"] == {
        "implementation": OPEN3D_SURFACE_PRODUCER_IMPLEMENTATION,
        "version": OPEN3D_SURFACE_PRODUCER_VERSION,
        "revision": OPEN3D_SURFACE_SOURCE_REVISION,
    }
    assert entry["dependency_refs"] == [OPEN3D_SURFACE_DEPENDENCY_REF]
    assert entry["model"] is None
    assert entry["checkpoint"] is None
    assert entry["artifact_key_hardware_policy"] == "required"
    assert entry["license"]["direct"] == "MIT"
    assert entry["license"]["review"] == "pending"
    assert entry["shipping_status"] == OPEN3D_SURFACE_SHIPPING_STATUS
    assert entry["failure_signals"] == [
        "dependency_unavailable",
        "environment_incompatible",
    ]
    assert entry["metric_names"] == []
    assert entry["resume_mode"] == "unsupported"


def test_module_has_no_eager_open3d_numpy_quality_runtime_or_later_surface_dependencies() -> None:
    forbidden = {
        "open3d",
        "numpy",
        "torch",
        "pycolmap",
        "QualityDecision",
        "QualityPolicy",
        "MetricDescriptor",
        "CollisionMesh",
        "NavMesh",
        "MasterScene",
        "RuntimeScene",
        "Poisson",
        "ICP",
        "SLAM",
    }
    assert forbidden.isdisjoint(vars(surface_module))
    assert OPEN3D_SURFACE_MESH_PATH == "surface.ply"
