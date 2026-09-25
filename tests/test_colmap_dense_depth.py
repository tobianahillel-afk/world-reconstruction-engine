from __future__ import annotations

import hashlib
import inspect
import json
import os
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import yaml

import wre.reconstruction.colmap_dense_depth as dense_module
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import CameraProjectionModelName
from wre.domain.hardware_identity import HardwareRuntimeIdentity
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import DerivedArtifactProvenance, ProducerRef, ReconstructionRunId
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_canonical_geometry import canonicalize_colmap_sparse_model
from wre.reconstruction.colmap_dense_depth import (
    COLMAP_CAMERA_Z_CONVENTION,
    COLMAP_MVS_ENVIRONMENT_KIND,
    COLMAP_MVS_IMAGE_SET_KIND,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_ADAPTER_ID,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_DEPENDENCY_REF,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_IMPLEMENTATION,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_RUNTIME_BUILD,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_SHIPPING_STATUS,
    COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION,
    ColmapDenseDepthEnvironmentError,
    ColmapDenseDepthError,
    ColmapDenseDepthSource,
    ColmapPatchMatchDenseDepthAdapter,
    ColmapPatchMatchDenseDepthConfig,
    inspect_colmap_dense_depth_environment,
)
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_geometry_refinement import (
    colmap_native_sparse_model_artifact_ref,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapModelFileArtifact,
    ColmapReconstructionInput,
    ColmapSparseModelArtifact,
)
from wre.reconstruction.dense_depth import DENSE_DEPTH_ARTIFACT_KIND
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate

_ROOT = Path(__file__).resolve().parents[1]
_ADAPTER_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"
_DEPENDENCY_REGISTRY_PATH = _ROOT / "registry" / "dependencies.yaml"

_ROTATION = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_BUILD = COLMAP_PATCH_MATCH_DENSE_DEPTH_RUNTIME_BUILD


class _FakeCamera:
    model_name = "PINHOLE"
    width = 2
    height = 2
    params = (2.0, 2.0, 1.0, 1.0)


class _FakeRotation:
    @staticmethod
    def matrix() -> tuple[tuple[float, float, float], ...]:
        return _ROTATION


class _FakeTransform:
    def __init__(self, translation: tuple[float, float, float]) -> None:
        self.rotation = _FakeRotation()
        self.translation = translation


class _FakeImage:
    def __init__(self, name: str, translation: tuple[float, float, float]) -> None:
        self.name = name
        self.camera_id = 1
        self._translation = translation

    def cam_from_world(self) -> _FakeTransform:
        return _FakeTransform(self._translation)


class _FakePoint:
    def __init__(self, xyz: tuple[float, float, float]) -> None:
        self.xyz = xyz
        self.error = 0.1
        self.track = SimpleNamespace(elements=())


class _FakeReconstruction:
    def __init__(
        self,
        path: Path,
        *,
        workspace_width: int = 2,
    ) -> None:
        self.path = Path(path)
        self.workspace_width = workspace_width

    @staticmethod
    def is_valid() -> bool:
        return True

    @staticmethod
    def reg_image_ids() -> list[int]:
        return [1, 2]

    @staticmethod
    def point3D_ids() -> list[int]:
        return [1]

    def camera(self, camera_id: int) -> _FakeCamera:
        assert camera_id == 1
        camera = _FakeCamera()
        if self.path.name == "sparse":
            camera.width = self.workspace_width
        return camera

    @staticmethod
    def image(image_id: int) -> _FakeImage:
        names = {1: "000000-a.png", 2: "000001-b.png"}
        translations = {1: (0.0, 0.0, 0.0), 2: (-1.0, 0.0, 0.0)}
        return _FakeImage(names[image_id], translations[image_id])

    @staticmethod
    def point3D(point_id: int) -> _FakePoint:
        assert point_id == 1
        return _FakePoint((1.0, 2.0, 3.0))

    @staticmethod
    def num_reg_images() -> int:
        return 2

    @staticmethod
    def num_points3D() -> int:
        return 1


class _FakeUndistortOptions:
    def __init__(self) -> None:
        self.blank_pixels = -1.0
        self.min_scale = -1.0
        self.max_scale = -1.0
        self.max_image_size = 0
        self.roi_min_x = -1.0
        self.roi_min_y = -1.0
        self.roi_max_x = -1.0
        self.roi_max_y = -1.0


class _FakePatchMatchOptions:
    def __init__(self) -> None:
        self.max_image_size = 0
        self.gpu_index = "-1"
        self.depth_min = 0.0
        self.depth_max = 0.0
        self.window_radius = 0
        self.window_step = 0
        self.sigma_spatial = 0.0
        self.sigma_color = 0.0
        self.num_samples = 0
        self.ncc_sigma = 0.0
        self.min_triangulation_angle = 0.0
        self.incident_angle_sigma = 0.0
        self.num_iterations = 0
        self.geom_consistency = False
        self.geom_consistency_regularizer = 0.0
        self.geom_consistency_max_cost = 0.0
        self.filter = False
        self.filter_min_ncc = 0.0
        self.filter_min_triangulation_angle = 0.0
        self.filter_min_num_consistent = 0
        self.filter_geom_consistency_max_cost = 0.0
        self.cache_size = 0.0
        self.allow_missing_files = True
        self.write_consistency_graph = True
        self.num_threads = -1


class _FakeFileCopyType:
    copy = "copy"


class _FakeArray:
    def __init__(self, rows: tuple[tuple[float, ...], ...]) -> None:
        self._rows = rows
        self.shape = (len(rows), len(rows[0]))

    def reshape(self, value: int) -> _FakeArray:
        assert value == -1
        return self

    def tolist(self) -> list[float]:
        return [value for row in self._rows for value in row]


class _FakeDepthMap:
    arrays: ClassVar[dict[Path, _FakeArray]] = {}

    def __init__(self) -> None:
        self._path: Path | None = None

    def read(self, path: Path) -> None:
        candidate = Path(path)
        if candidate not in self.arrays:
            raise OSError("unknown synthetic depth map")
        self._path = candidate

    def to_array(self) -> _FakeArray:
        assert self._path is not None
        return self.arrays[self._path]


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = _BUILD
    __ceres_version__ = "2.2.0"
    has_cuda = True
    UndistortCameraOptions = _FakeUndistortOptions
    PatchMatchOptions = _FakePatchMatchOptions
    FileCopyType = _FakeFileCopyType
    DepthMap = _FakeDepthMap

    def __init__(
        self,
        *,
        emitted: tuple[str, ...] = ("000000-a.png", "000001-b.png"),
        depth_rows: tuple[tuple[float, ...], ...] = ((1.5, 0.0), (-2.0, 3.0)),
        fail_patch_match: bool = False,
        workspace_width: int = 2,
    ) -> None:
        self.emitted = emitted
        self.depth_rows = depth_rows
        self.fail_patch_match = fail_patch_match
        self.workspace_width = workspace_width
        self.calls: list[tuple[str, dict[str, Any]]] = []
        _FakeDepthMap.arrays = {}

    def Reconstruction(self, path: Path) -> _FakeReconstruction:
        return _FakeReconstruction(Path(path), workspace_width=self.workspace_width)

    def undistort_images(
        self,
        output_path: Path,
        input_path: Path,
        image_path: Path,
        **kwargs: Any,
    ) -> None:
        output = Path(output_path)
        source_model = Path(input_path)
        source_images = Path(image_path)
        self.calls.append(
            (
                "undistort_images",
                {
                    "output_path": output,
                    "input_path": source_model,
                    "image_path": source_images,
                    **kwargs,
                },
            )
        )
        (output / "sparse").mkdir(parents=True)
        for source_file in source_model.iterdir():
            if source_file.is_file():
                (output / "sparse" / source_file.name).write_bytes(source_file.read_bytes())
        (output / "images").mkdir()
        for source_file in source_images.iterdir():
            if source_file.is_file():
                (output / "images" / source_file.name).write_bytes(source_file.read_bytes())
        (output / "stereo" / "depth_maps").mkdir(parents=True)
        (output / "stereo" / "normal_maps").mkdir()
        (output / "stereo" / "consistency_graphs").mkdir()

    def patch_match_stereo(
        self,
        workspace_path: Path,
        **kwargs: Any,
    ) -> None:
        workspace = Path(workspace_path)
        self.calls.append(("patch_match_stereo", {"workspace_path": workspace, **kwargs}))
        if self.fail_patch_match:
            raise RuntimeError("synthetic PatchMatch failure")
        depth_root = workspace / "stereo" / "depth_maps"
        for image_name in self.emitted:
            path = depth_root / f"{image_name}.geometric.bin"
            path.write_bytes(b"synthetic-depth:" + image_name.encode("ascii"))
            _FakeDepthMap.arrays[path] = _FakeArray(self.depth_rows)

    def stereo_fusion(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2L15.2 must never call stereo_fusion")


def _cpu_module() -> _FakePycolmap:
    module = _FakePycolmap()
    module.has_cuda = False
    return module


def _environment() -> ColmapEnvironmentIdentity:
    return ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build=_BUILD,
        ceres_version="2.2.0",
        upstream_has_cuda=True,
    )


def _producer(token: str) -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation=f"test.colmap.{token}",
            version="1",
            revision=f"revision:{token}",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
        ),
    )


def _artifact(identifier: str, kind: str = "geometry.solution") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _observation(path: Path, observation_id: str) -> ImageObservation:
    digest = hash_file_content(path)
    return ImageObservation(
        observation_id=ObservationId(observation_id),
        asset=MediaAssetRef(
            uri=path.resolve().as_uri(),
            sha256=digest.sha256,
            byte_length=digest.byte_length,
            mime_type="image/png",
        ),
        source=SourceRef(source_id=SourceId("source:dense")),
        received_at=datetime(2026, 9, 24, tzinfo=UTC),
    )


def _write_native_model(root: Path) -> ColmapSparseModelArtifact:
    model_path = root / "0"
    model_path.mkdir(parents=True)
    for name, content in (
        ("cameras.bin", b"camera-source"),
        ("images.bin", b"images-source"),
        ("points3D.bin", b"points-source"),
    ):
        (model_path / name).write_bytes(content)
    files = tuple(
        ColmapModelFileArtifact(
            relative_path=path.name,
            sha256=hash_file_content(path).sha256,
            byte_length=hash_file_content(path).byte_length,
        )
        for path in sorted(model_path.iterdir(), key=lambda item: item.name)
    )
    return ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=2,
        num_points3d=1,
        files=files,
    )


def _features(
    observations: tuple[ImageObservation, ...],
    tmp_path: Path,
) -> ColmapFeatureExtractionResult:
    database = tmp_path / "features.db"
    database.write_bytes(b"feature-evidence")
    digest = hash_file_content(database)
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:dense-features"),
            source_observation_ids=tuple(item.observation_id for item in observations),
        ),
        environment=ColmapEnvironmentIdentity(
            pycolmap_version="4.2.0",
            colmap_version="COLMAP 4.2.0",
            colmap_build="Commit cpu-sparse-source",
            ceres_version="2.2.0",
            upstream_has_cuda=False,
        ),
        configuration_sha256=Sha256Digest("1" * 64),
        database_path=database,
        database_sha256=digest.sha256,
        database_byte_length=digest.byte_length,
        images=(
            ColmapImageFeatureSummary(
                observation_id=observations[0].observation_id,
                image_name="000000-a.png",
                keypoint_rows=1,
                keypoint_cols=4,
                descriptor_rows=1,
                descriptor_cols=128,
            ),
            ColmapImageFeatureSummary(
                observation_id=observations[1].observation_id,
                image_name="000001-b.png",
                keypoint_rows=1,
                keypoint_cols=4,
                descriptor_rows=1,
                descriptor_cols=128,
            ),
        ),
    )


def _fixture(
    tmp_path: Path,
    *,
    module: _FakePycolmap | None = None,
) -> tuple[_FakePycolmap, ColmapDenseDepthSource]:
    pycolmap = module or _FakePycolmap()
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_paths = (image_root / "000000-a.png", image_root / "000001-b.png")
    image_paths[0].write_bytes(b"image-a")
    image_paths[1].write_bytes(b"image-b")
    observations = (
        _observation(image_paths[0], "obs:a"),
        _observation(image_paths[1], "obs:b"),
    )
    inputs = (
        ColmapReconstructionInput(
            observation=observations[0],
            source_path=image_paths[0],
            image_name="000000-a.png",
        ),
        ColmapReconstructionInput(
            observation=observations[1],
            source_path=image_paths[1],
            image_name="000001-b.png",
        ),
    )
    features = _features(observations, tmp_path)
    model_root = tmp_path / "source-model"
    model_root.mkdir()
    model_artifact = _write_native_model(model_root)
    canonical = canonicalize_colmap_sparse_model(
        output_path=model_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=_environment(),
        module=pycolmap,
    )
    candidate = GeometrySolutionCandidate(
        geometry_solution=canonical.geometry_solution,
        camera_solutions=canonical.camera_solutions,
        depth_fields=(),
        point_maps=(canonical.point_map,),
        producer=_producer("source"),
        source_artifacts=(_artifact("artifact:source"),),
    )
    source = ColmapDenseDepthSource(
        model_root=model_root,
        model_artifact=model_artifact,
        source_geometry=candidate,
        features=features,
        image_root=image_root,
        images=inputs,
        expected_environment=_environment(),
        artifact_ref=colmap_native_sparse_model_artifact_ref(model_artifact),
    )
    return pycolmap, source


def _hardware(token: str = "a") -> HardwareRuntimeIdentity:
    return HardwareRuntimeIdentity(
        sha256=Sha256Digest(hashlib.sha256(token.encode("utf-8")).hexdigest())
    )


def _source_bytes(source: ColmapDenseDepthSource) -> tuple[dict[str, bytes], dict[str, bytes]]:
    model_path = source.model_root / source.model_artifact.relative_path
    models = {
        path.name: path.read_bytes()
        for path in sorted(model_path.iterdir(), key=lambda item: item.name)
        if path.is_file()
    }
    images = {
        item.image_name: (source.image_root / item.image_name).read_bytes()
        for item in source.images
    }
    return models, images


class _RecordingRealPycolmap:
    """Proxy the real binding while proving the dense donor call surface."""

    def __init__(self, module: Any) -> None:
        self._module = module
        self.calls: list[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def undistort_images(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("undistort_images")
        return self._module.undistort_images(*args, **kwargs)

    def patch_match_stereo(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append("patch_match_stereo")
        return self._module.patch_match_stereo(*args, **kwargs)

    def stereo_fusion(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("V2L15.2 real CUDA evidence must never call stereo_fusion")


def _real_cuda_sparse_model_artifact(
    model_root: Path,
    reconstruction: Any,
) -> ColmapSparseModelArtifact:
    model_path = model_root / "0"
    files = tuple(
        ColmapModelFileArtifact(
            relative_path=path.relative_to(model_path).as_posix(),
            sha256=hash_file_content(path).sha256,
            byte_length=hash_file_content(path).byte_length,
        )
        for path in sorted(model_path.rglob("*"), key=lambda item: item.as_posix())
        if path.is_file()
    )
    return ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=len(tuple(reconstruction.reg_image_ids())),
        num_points3d=len(tuple(reconstruction.point3D_ids())),
        files=files,
    )


def _real_cuda_source(
    root: Path,
    pycolmap: Any,
    environment: ColmapEnvironmentIdentity,
) -> tuple[ColmapDenseDepthSource, dict[str, object]]:
    seed = 20260924
    pycolmap.set_random_seed(seed)

    database_path = root / "synthetic.db"
    dataset_options = pycolmap.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 4
    dataset_options.num_points3D = 600
    dataset_options.track_length = -1
    dataset_options.sensor_from_rig_translation_stddev = 0.0
    dataset_options.sensor_from_rig_rotation_stddev = 0.0
    dataset_options.camera_width = 320
    dataset_options.camera_height = 240
    dataset_options.camera_model_id = pycolmap.CameraModelId.PINHOLE
    dataset_options.camera_params = [280.0, 280.0, 160.0, 120.0]
    dataset_options.camera_has_prior_focal_length = True
    dataset_options.num_points2D_without_point3D = 0
    dataset_options.inlier_match_ratio = 1.0
    dataset_options.match_config = pycolmap.SyntheticDatasetMatchConfig.EXHAUSTIVE

    with pycolmap.Database.open(database_path) as database:
        reconstruction = pycolmap.synthesize_dataset(dataset_options, database)

    assert bool(reconstruction.is_valid())
    assert int(reconstruction.num_reg_images()) == 4
    assert int(reconstruction.num_points3D()) == 600

    image_root = root / "images"
    image_root.mkdir()
    image_options = pycolmap.SyntheticImageOptions()
    image_options.feature_peak_radius = 1
    image_options.feature_patch_radius = 4
    image_options.feature_patch_max_brightness = 220
    pycolmap.synthesize_images(image_options, reconstruction, image_root)

    model_root = root / "source-model"
    model_path = model_root / "0"
    model_path.mkdir(parents=True)
    reconstruction.write(model_path)
    model_artifact = _real_cuda_sparse_model_artifact(model_root, reconstruction)

    registered = sorted(
        (
            str(reconstruction.image(int(image_id)).name),
            int(image_id),
        )
        for image_id in reconstruction.reg_image_ids()
    )
    observations: list[ImageObservation] = []
    inputs: list[ColmapReconstructionInput] = []
    summaries: list[ColmapImageFeatureSummary] = []
    for index, (image_name, image_id) in enumerate(registered):
        image_path = image_root / image_name
        assert image_path.is_file()
        observation = _observation(image_path, f"obs:cuda:{index:04d}")
        observations.append(observation)
        inputs.append(
            ColmapReconstructionInput(
                observation=observation,
                source_path=image_path,
                image_name=image_name,
            )
        )
        image = reconstruction.image(image_id)
        num_points = int(image.num_points2D())
        summaries.append(
            ColmapImageFeatureSummary(
                observation_id=observation.observation_id,
                image_name=image_name,
                keypoint_rows=num_points,
                keypoint_cols=4,
                descriptor_rows=num_points,
                descriptor_cols=128,
            )
        )

    database_digest = hash_file_content(database_path)
    canonical_observations = tuple(observations)
    features = ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:dense-real-cuda-features"),
            source_observation_ids=tuple(
                observation.observation_id for observation in canonical_observations
            ),
        ),
        environment=environment,
        configuration_sha256=Sha256Digest(
            hashlib.sha256(b"wre-v2l15.2-real-cuda-synthetic-feature-map-v1").hexdigest()
        ),
        database_path=database_path,
        database_sha256=database_digest.sha256,
        database_byte_length=database_digest.byte_length,
        images=tuple(summaries),
    )

    canonical = canonicalize_colmap_sparse_model(
        output_path=model_root,
        model_artifact=model_artifact,
        features=features,
        expected_environment=environment,
        module=pycolmap,
    )
    candidate = GeometrySolutionCandidate(
        geometry_solution=canonical.geometry_solution,
        camera_solutions=canonical.camera_solutions,
        depth_fields=(),
        point_maps=(canonical.point_map,),
        producer=_producer("real-cuda-source"),
        source_artifacts=(_artifact("artifact:source:real-cuda"),),
    )
    source = ColmapDenseDepthSource(
        model_root=model_root,
        model_artifact=model_artifact,
        source_geometry=candidate,
        features=features,
        image_root=image_root,
        images=tuple(inputs),
        expected_environment=environment,
        artifact_ref=colmap_native_sparse_model_artifact_ref(model_artifact),
    )
    fixture = {
        "camera_height": dataset_options.camera_height,
        "camera_model": "PINHOLE",
        "camera_params": list(dataset_options.camera_params),
        "camera_width": dataset_options.camera_width,
        "num_frames": dataset_options.num_frames_per_rig,
        "num_points3D": dataset_options.num_points3D,
        "seed": seed,
    }
    return source, fixture


def _sha256_json_document(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _real_cuda_output_evidence(
    result: Any,
    *,
    output_root: Path,
) -> dict[str, object]:
    depth_maps = []
    for path in sorted(
        (output_root / "stereo" / "depth_maps").glob("*.geometric.bin"),
        key=lambda item: item.name,
    ):
        digest = hash_file_content(path)
        depth_maps.append(
            {
                "byte_length": digest.byte_length,
                "name": path.name,
                "sha256": digest.sha256.value,
            }
        )

    fields = []
    for field in result.depth_fields:
        valid_values = [
            value for value, valid in zip(field.depth_values, field.validity, strict=True) if valid
        ]
        fields.append(
            {
                "depth_field_id": field.depth_field_id.value,
                "depth_payload_sha256": _sha256_json_document(
                    {
                        "depth_values": list(field.depth_values),
                        "validity": list(field.validity),
                    }
                ),
                "height_px": field.dimensions.height_px,
                "max_valid_depth": max(valid_values) if valid_values else None,
                "min_valid_depth": min(valid_values) if valid_values else None,
                "observation_id": field.observation_id.value,
                "valid_pixel_count": len(valid_values),
                "width_px": field.dimensions.width_px,
            }
        )

    return {
        "artifact_id": result.artifact_ref.artifact_id.value,
        "depth_fields": fields,
        "depth_maps": depth_maps,
    }


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_MVS_UNDISTORT_PREFLIGHT") != "1",
    reason="real PyCOLMAP undistortion preflight runs only in the dedicated MVS lane",
)
def test_real_cuda_wheel_undistort_workspace_preflight(tmp_path: Path) -> None:
    pycolmap = dense_module._load_pycolmap()
    environment = inspect_colmap_dense_depth_environment(pycolmap)
    assert environment.upstream_has_cuda is True

    evidence_path_raw = os.environ.get("WRE_COLMAP_MVS_UNDISTORT_PREFLIGHT_EVIDENCE")
    assert evidence_path_raw
    evidence_path = Path(evidence_path_raw).expanduser().resolve()
    assert not evidence_path.exists()

    source, fixture = _real_cuda_source(tmp_path, pycolmap, environment)
    source_before = _source_bytes(source)
    verified_images = dense_module._verified_source_images(source)
    source_model_path = dense_module._verified_native_model_path(
        source.model_root,
        source.model_artifact,
    )

    working_root = tmp_path / "private-input"
    working_root.mkdir()
    copied_model_path, copied_image_root = dense_module._copy_private_inputs(
        source=source,
        source_model_path=source_model_path,
        verified_images=verified_images,
        working_root=working_root,
    )

    output_root = tmp_path / "mvs-workspace"
    output_root.mkdir()
    config = ColmapPatchMatchDenseDepthConfig()
    undistort_options = dense_module._configure_undistortion(pycolmap, config)
    pycolmap.undistort_images(
        output_root,
        copied_model_path,
        copied_image_root,
        image_names=list(dense_module._registered_image_names(source)),
        output_type="COLMAP",
        copy_policy=pycolmap.FileCopyType.copy,
        num_patch_match_src_images=config.num_patch_match_src_images,
        undistort_options=undistort_options,
        jpeg_quality=config.jpeg_quality,
        num_threads=config.undistort_num_threads,
    )

    dense_module._validate_workspace_linkage(pycolmap, output_root, source)
    dense_module._verify_retained_sources(source, verified_images)
    assert _source_bytes(source) == source_before

    patch_options = dense_module._configure_patch_match(pycolmap, config)
    patchmatch_options = {
        "allow_missing_files": patch_options.allow_missing_files,
        "cache_size": patch_options.cache_size,
        "depth_max": patch_options.depth_max,
        "depth_min": patch_options.depth_min,
        "filter": patch_options.filter,
        "filter_geom_consistency_max_cost": patch_options.filter_geom_consistency_max_cost,
        "filter_min_ncc": patch_options.filter_min_ncc,
        "filter_min_num_consistent": patch_options.filter_min_num_consistent,
        "filter_min_triangulation_angle": patch_options.filter_min_triangulation_angle,
        "geom_consistency": patch_options.geom_consistency,
        "geom_consistency_max_cost": patch_options.geom_consistency_max_cost,
        "geom_consistency_regularizer": patch_options.geom_consistency_regularizer,
        "gpu_index": patch_options.gpu_index,
        "incident_angle_sigma": patch_options.incident_angle_sigma,
        "max_image_size": patch_options.max_image_size,
        "min_triangulation_angle": patch_options.min_triangulation_angle,
        "ncc_sigma": patch_options.ncc_sigma,
        "num_iterations": patch_options.num_iterations,
        "num_samples": patch_options.num_samples,
        "num_threads": patch_options.num_threads,
        "sigma_color": patch_options.sigma_color,
        "sigma_spatial": patch_options.sigma_spatial,
        "window_radius": patch_options.window_radius,
        "window_step": patch_options.window_step,
        "write_consistency_graph": patch_options.write_consistency_graph,
    }
    assert patchmatch_options == {
        "allow_missing_files": config.allow_missing_files,
        "cache_size": config.cache_size,
        "depth_max": config.depth_max,
        "depth_min": config.depth_min,
        "filter": config.filter,
        "filter_geom_consistency_max_cost": config.filter_geom_consistency_max_cost,
        "filter_min_ncc": config.filter_min_ncc,
        "filter_min_num_consistent": config.filter_min_num_consistent,
        "filter_min_triangulation_angle": config.filter_min_triangulation_angle,
        "geom_consistency": config.geom_consistency,
        "geom_consistency_max_cost": config.geom_consistency_max_cost,
        "geom_consistency_regularizer": config.geom_consistency_regularizer,
        "gpu_index": config.gpu_index,
        "incident_angle_sigma": config.incident_angle_sigma,
        "max_image_size": config.max_image_size,
        "min_triangulation_angle": config.min_triangulation_angle,
        "ncc_sigma": config.ncc_sigma,
        "num_iterations": config.num_iterations,
        "num_samples": config.num_samples,
        "num_threads": config.num_threads,
        "sigma_color": config.sigma_color,
        "sigma_spatial": config.sigma_spatial,
        "window_radius": config.window_radius,
        "window_step": config.window_step,
        "write_consistency_graph": config.write_consistency_graph,
    }
    assert patch_options.check() is True

    depth_root = output_root / "stereo" / "depth_maps"
    assert not depth_root.exists() or not tuple(depth_root.glob("*.bin"))

    workspace = pycolmap.Reconstruction(output_root / "sparse")
    assert bool(workspace.is_valid())
    registered_ids = tuple(sorted(int(value) for value in workspace.reg_image_ids()))
    assert len(registered_ids) == len(source.source_geometry.camera_solutions)

    workspace_images = []
    for image_id in registered_ids:
        image = workspace.image(image_id)
        camera = workspace.camera(int(image.camera_id))
        workspace_images.append(
            {
                "camera_model": str(camera.model_name),
                "height": int(camera.height),
                "image_name": str(image.name),
                "params": [float(value) for value in camera.params],
                "width": int(camera.width),
            }
        )

    sparse_files = []
    for path in sorted(
        (output_root / "sparse").glob("*"),
        key=lambda item: item.name,
    ):
        if not path.is_file():
            continue
        digest = hash_file_content(path)
        sparse_files.append(
            {
                "byte_length": digest.byte_length,
                "name": path.name,
                "sha256": digest.sha256.value,
            }
        )
    assert sparse_files

    evidence = {
        "schema_version": 1,
        "evidence_kind": "pycolmap_cuda12_4_2_0_cpu_undistort_preflight",
        "wheel_import_execution_evidence": True,
        "synthetic_fixture_execution_evidence": True,
        "undistort_execution_evidence": True,
        "workspace_linkage_execution_evidence": True,
        "patchmatch_options_binding_evidence": True,
        "patchmatch_options_check_evidence": True,
        "runtime_execution_evidence": True,
        "gpu_execution_evidence": False,
        "patchmatch_execution_evidence": False,
        "stereo_fusion_execution_evidence": False,
        "environment": {
            "ceres_version": environment.ceres_version,
            "colmap_build": environment.colmap_build,
            "colmap_version": environment.colmap_version,
            "pycolmap_version": environment.pycolmap_version,
            "upstream_has_cuda": environment.upstream_has_cuda,
        },
        "fixture": fixture,
        "patchmatch_options": patchmatch_options,
        "source_immutable_after_undistortion": True,
        "workspace": {
            "images": workspace_images,
            "sparse_files": sparse_files,
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_MVS_REAL_CUDA") != "1",
    reason="real CUDA PatchMatch evidence is manual and requires a trusted GPU runner",
)
def test_real_cuda_patchmatch_retained_fixture() -> None:
    pycolmap = dense_module._load_pycolmap()
    environment = inspect_colmap_dense_depth_environment(pycolmap)
    assert environment.upstream_has_cuda is True

    run_root_raw = os.environ.get("WRE_COLMAP_MVS_REAL_ROOT")
    hardware_path_raw = os.environ.get("WRE_COLMAP_MVS_HARDWARE_EVIDENCE")
    hardware_sha256 = os.environ.get("WRE_COLMAP_MVS_HARDWARE_SHA256")
    evidence_path_raw = os.environ.get("WRE_COLMAP_MVS_EVIDENCE_PATH")
    assert run_root_raw
    assert hardware_path_raw
    assert hardware_sha256
    assert evidence_path_raw

    run_root = Path(run_root_raw).expanduser().resolve()
    hardware_path = Path(hardware_path_raw).expanduser().resolve(strict=True)
    evidence_path = Path(evidence_path_raw).expanduser().resolve()
    assert not run_root.exists()
    assert not evidence_path.exists()
    run_root.mkdir(parents=True)

    hardware_digest = hash_file_content(hardware_path)
    assert hardware_digest.sha256.value == hardware_sha256
    hardware_document = json.loads(hardware_path.read_text(encoding="utf-8"))
    assert hardware_document["gpu_execution_evidence"] is True

    source, fixture = _real_cuda_source(run_root, pycolmap, environment)
    source_before = _source_bytes(source)
    proxy = _RecordingRealPycolmap(pycolmap)
    output_root = run_root / "dense-output"
    result = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=output_root,
        hardware_runtime=HardwareRuntimeIdentity(
            sha256=Sha256Digest(hardware_sha256),
        ),
        module=proxy,
    ).derive()

    assert proxy.calls == ["undistort_images", "patch_match_stereo"]
    assert _source_bytes(source) == source_before
    assert result.depth_fields
    assert sum(sum(field.validity) for field in result.depth_fields) > 0
    assert all(field.confidence is None for field in result.depth_fields)
    assert all(
        field.depth_value_convention is COLMAP_CAMERA_Z_CONVENTION for field in result.depth_fields
    )

    output_evidence = _real_cuda_output_evidence(result, output_root=output_root)
    assert output_evidence["depth_maps"]
    source_model_files = [
        {
            "byte_length": item.byte_length,
            "path": item.relative_path,
            "sha256": item.sha256.value,
        }
        for item in source.model_artifact.files
    ]
    source_images = [
        {
            "byte_length": item.observation.asset.byte_length,
            "image_name": item.image_name,
            "observation_id": item.observation.observation_id.value,
            "sha256": item.observation.asset.sha256.value,
        }
        for item in source.images
    ]
    evidence = {
        "schema_version": 1,
        "evidence_kind": "real_colmap_patchmatch_dense_depth_cuda",
        "runtime_execution_evidence": True,
        "gpu_execution_evidence": True,
        "adapter_id": COLMAP_PATCH_MATCH_DENSE_DEPTH_ADAPTER_ID,
        "dependency_ref": COLMAP_PATCH_MATCH_DENSE_DEPTH_DEPENDENCY_REF,
        "environment": {
            "ceres_version": environment.ceres_version,
            "colmap_build": environment.colmap_build,
            "colmap_version": environment.colmap_version,
            "pycolmap_version": environment.pycolmap_version,
            "upstream_has_cuda": environment.upstream_has_cuda,
        },
        "fixture": fixture,
        "hardware_evidence_sha256": hardware_sha256,
        "source": {
            "images": source_images,
            "model_files": source_model_files,
            "source_immutable_after_execution": True,
        },
        "execution_calls": proxy.calls,
        "result": output_evidence,
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_config_is_frozen_exact_and_deterministic() -> None:
    config = ColmapPatchMatchDenseDepthConfig()

    assert config.sha256 == ColmapPatchMatchDenseDepthConfig().sha256
    assert config.gpu_index == "0"
    assert config.max_image_size == -1
    assert config.undistort_max_image_size == -1
    assert config.geom_consistency is True
    assert config.filter is True
    assert config.allow_missing_files is False
    assert config.write_consistency_graph is False
    assert config.num_threads == 1
    assert set(config.canonical_document()) == {
        field.name for field in fields(ColmapPatchMatchDenseDepthConfig)
    }

    with pytest.raises(FrozenInstanceError):
        config.gpu_index = "1"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"gpu_index": "-1"}, "accelerator index"),
        ({"max_image_size": 2000}, "no-resize"),
        ({"undistort_max_image_size": 2000}, "no-resize"),
        ({"num_threads": 2}, "one host thread"),
        ({"geom_consistency": False}, "geometric-consistency"),
        ({"filter": False}, "filtering"),
        ({"allow_missing_files": True}, "missing PatchMatch"),
        ({"write_consistency_graph": True}, "consistency-graph"),
        ({"num_patch_match_src_images": 2}, "complete source-image"),
        ({"jpeg_quality": 90}, "recompress"),
        ({"num_iterations": 4}, "num_iterations"),
    ],
)
def test_config_rejects_unreviewed_modes(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ColmapPatchMatchDenseDepthConfig(**kwargs)


def test_environment_gate_requires_exact_cuda_42_and_public_apis() -> None:
    environment = inspect_colmap_dense_depth_environment(_FakePycolmap())
    assert environment == _environment()
    assert environment.upstream_has_cuda is True
    assert environment.colmap_build == "Commit be5e291 on 2026-08-31 with CUDA"
    assert COLMAP_PATCH_MATCH_DENSE_DEPTH_RUNTIME_BUILD == environment.colmap_build

    with pytest.raises(ColmapDenseDepthEnvironmentError, match="has_cuda"):
        inspect_colmap_dense_depth_environment(_cpu_module())

    wrong_commit = _FakePycolmap()
    wrong_commit.COLMAP_build = "Commit deadbee on 2026-08-31 with CUDA"
    with pytest.raises(ColmapDenseDepthEnvironmentError, match="exact official"):
        inspect_colmap_dense_depth_environment(wrong_commit)

    wrong_date = _FakePycolmap()
    wrong_date.COLMAP_build = "Commit be5e291 on 2026-08-30 with CUDA"
    with pytest.raises(ColmapDenseDepthEnvironmentError, match="exact official"):
        inspect_colmap_dense_depth_environment(wrong_date)

    wrong_accelerator = _FakePycolmap()
    wrong_accelerator.COLMAP_build = "Commit be5e291 on 2026-08-31 without GPU support"
    with pytest.raises(ColmapDenseDepthEnvironmentError, match="exact official"):
        inspect_colmap_dense_depth_environment(wrong_accelerator)

    missing = _FakePycolmap()
    missing.patch_match_stereo = None  # type: ignore[assignment]
    with pytest.raises(ColmapDenseDepthEnvironmentError, match="patch_match_stereo"):
        inspect_colmap_dense_depth_environment(missing)


def test_source_binding_is_exact_and_canonical(tmp_path: Path) -> None:
    _, source = _fixture(tmp_path)

    assert tuple(field.name for field in fields(source)) == (
        "model_root",
        "model_artifact",
        "source_geometry",
        "features",
        "image_root",
        "images",
        "expected_environment",
        "artifact_ref",
    )
    assert source.artifact_ref == colmap_native_sparse_model_artifact_ref(source.model_artifact)
    assert tuple(item.observation.observation_id.value for item in source.images) == (
        "obs:a",
        "obs:b",
    )

    with pytest.raises(ValueError, match="exact audited"):
        replace(source, artifact_ref=_artifact("artifact:other", "geometry.solution"))


def test_patchmatch_derives_camera_z_depth_without_fusion(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    before = _source_bytes(source)
    output_root = tmp_path / "dense-output"
    result = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=output_root,
        hardware_runtime=_hardware(),
        module=module,
    ).derive()

    assert result.artifact_ref.artifact_kind == DENSE_DEPTH_ARTIFACT_KIND
    assert result.source_geometry is source.source_geometry
    assert len(result.depth_fields) == 2
    assert {item.observation_id.value for item in result.depth_fields} == {
        "obs:a",
        "obs:b",
    }
    assert tuple(item.depth_field_id.value for item in result.depth_fields) == tuple(
        sorted(item.depth_field_id.value for item in result.depth_fields)
    )
    for depth in result.depth_fields:
        assert depth.depth_value_convention is COLMAP_CAMERA_Z_CONVENTION
        assert depth.depth_values == (1.5, 0.0, 0.0, 3.0)
        assert depth.validity == (True, False, False, True)
        assert depth.confidence is None
        assert depth.dimensions.width_px == 2
        assert depth.dimensions.height_px == 2

    assert result.producer.producer.implementation == (
        COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_IMPLEMENTATION
    )
    assert result.producer.producer.version == "4.2.0"
    assert result.producer.producer.revision == _BUILD
    kinds = {item.artifact_kind for item in result.source_artifacts}
    assert COLMAP_MVS_IMAGE_SET_KIND in kinds
    assert COLMAP_MVS_ENVIRONMENT_KIND in kinds
    assert colmap_native_sparse_model_artifact_ref(source.model_artifact) in (
        result.source_artifacts
    )
    assert source.source_geometry.source_artifacts[0] in result.source_artifacts
    assert _source_bytes(source) == before

    assert [name for name, _ in module.calls] == [
        "undistort_images",
        "patch_match_stereo",
    ]
    undistort = module.calls[0][1]
    assert undistort["output_path"] == output_root
    assert undistort["input_path"] != source.model_root / "0"
    assert undistort["image_path"] != source.image_root
    assert undistort["image_names"] == ["000000-a.png", "000001-b.png"]
    assert undistort["copy_policy"] == _FakeFileCopyType.copy
    assert undistort["num_patch_match_src_images"] == -1
    assert undistort["jpeg_quality"] == -1
    assert undistort["num_threads"] == 1
    assert undistort["undistort_options"].max_image_size == -1

    patch = module.calls[1][1]
    options = patch["options"]
    assert patch["workspace_path"] == output_root
    assert patch["workspace_format"] == "COLMAP"
    assert options.gpu_index == "0"
    assert options.max_image_size == -1
    assert options.geom_consistency is True
    assert options.allow_missing_files is False
    assert options.num_threads == 1


def test_partial_depth_coverage_is_preserved_without_fabrication(tmp_path: Path) -> None:
    module, source = _fixture(
        tmp_path,
        module=_FakePycolmap(emitted=("000001-b.png",)),
    )
    result = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=tmp_path / "partial",
        hardware_runtime=_hardware(),
        module=module,
    ).derive()

    assert len(source.source_geometry.camera_solutions) == 2
    assert len(result.depth_fields) == 1
    assert result.depth_fields[0].observation_id == ObservationId("obs:b")


def test_zero_depth_outputs_fail_and_cleanup_private_output(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path, module=_FakePycolmap(emitted=()))
    output_root = tmp_path / "zero"

    with pytest.raises(ColmapDenseDepthError, match="no source-linked"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=output_root,
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert not output_root.exists()


def test_nonfinite_depth_fails_closed(tmp_path: Path) -> None:
    module, source = _fixture(
        tmp_path,
        module=_FakePycolmap(depth_rows=((1.0, float("nan")), (2.0, 3.0))),
    )
    output_root = tmp_path / "nonfinite"

    with pytest.raises(ColmapDenseDepthError, match="must be finite"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=output_root,
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert not output_root.exists()


def test_source_image_tamper_is_rejected_before_solver_execution(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    (source.image_root / source.images[0].image_name).write_bytes(b"tampered")

    with pytest.raises(ColmapDenseDepthError, match="persisted observation"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=tmp_path / "tampered-image",
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert module.calls == []


def test_source_model_tamper_is_rejected_before_solver_execution(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    (source.model_root / "0" / "points3D.bin").write_bytes(b"tampered")

    with pytest.raises(Exception, match="changed after mapper publication"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=tmp_path / "tampered-model",
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert module.calls == []


def test_mismatched_geometry_is_rejected_before_workspace_execution(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    camera = source.source_geometry.camera_solutions[0]
    changed_camera = replace(
        camera,
        intrinsic_parameters=(3.0, 2.0, 1.0, 1.0),
    )
    changed_candidate = replace(
        source.source_geometry,
        camera_solutions=(changed_camera, source.source_geometry.camera_solutions[1]),
    )
    mismatched_source = replace(source, source_geometry=changed_candidate)

    with pytest.raises(ColmapDenseDepthError, match="cameras do not match"):
        ColmapPatchMatchDenseDepthAdapter(
            source=mismatched_source,
            output_root=tmp_path / "mismatch",
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert module.calls == []


def test_non_pinhole_source_is_rejected_before_execution(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    camera = source.source_geometry.camera_solutions[0]
    changed_camera = replace(
        camera,
        projection_model=CameraProjectionModelName("simple_pinhole"),
        intrinsic_parameters=(2.0, 1.0, 1.0),
    )
    changed_candidate = replace(
        source.source_geometry,
        camera_solutions=(changed_camera, source.source_geometry.camera_solutions[1]),
    )

    with pytest.raises(ColmapDenseDepthError, match="PINHOLE"):
        ColmapPatchMatchDenseDepthAdapter(
            source=replace(source, source_geometry=changed_candidate),
            output_root=tmp_path / "simple-pinhole",
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert module.calls == []


def test_workspace_camera_drift_fails_before_patchmatch(tmp_path: Path) -> None:
    module, source = _fixture(
        tmp_path,
        module=_FakePycolmap(workspace_width=3),
    )
    output_root = tmp_path / "workspace-drift"

    with pytest.raises(ColmapDenseDepthError, match="dimensions changed"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=output_root,
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert [name for name, _ in module.calls] == ["undistort_images"]
    assert not output_root.exists()


def test_patchmatch_failure_preserves_sources_and_cleans_output(tmp_path: Path) -> None:
    module, source = _fixture(
        tmp_path,
        module=_FakePycolmap(fail_patch_match=True),
    )
    before = _source_bytes(source)
    output_root = tmp_path / "patch-failure"

    with pytest.raises(ColmapDenseDepthError, match="PatchMatch stereo failed"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=output_root,
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    assert _source_bytes(source) == before
    assert not output_root.exists()


def test_output_root_must_be_fresh_and_disjoint(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)

    with pytest.raises(ValueError, match="overlap"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=source.image_root / "dense",
            hardware_runtime=_hardware(),
            module=module,
        ).derive()

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        ColmapPatchMatchDenseDepthAdapter(
            source=source,
            output_root=occupied,
            hardware_runtime=_hardware(),
            module=module,
        ).derive()


def test_artifact_identity_is_deterministic_and_hardware_sensitive(tmp_path: Path) -> None:
    module_a, source = _fixture(tmp_path)
    first = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=tmp_path / "first",
        hardware_runtime=_hardware("same"),
        module=module_a,
    ).derive()

    module_b = _FakePycolmap()
    second = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=tmp_path / "second",
        hardware_runtime=_hardware("same"),
        module=module_b,
    ).derive()
    assert first.artifact_ref == second.artifact_ref
    assert tuple(item.depth_field_id for item in first.depth_fields) == tuple(
        item.depth_field_id for item in second.depth_fields
    )

    module_c = _FakePycolmap()
    third = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=tmp_path / "third",
        hardware_runtime=_hardware("different"),
        module=module_c,
    ).derive()
    assert third.artifact_ref != first.artifact_ref
    assert tuple(item.depth_field_id for item in third.depth_fields) != tuple(
        item.depth_field_id for item in first.depth_fields
    )


def test_source_local_frame_and_scale_are_not_upgraded(tmp_path: Path) -> None:
    module, source = _fixture(tmp_path)
    result = ColmapPatchMatchDenseDepthAdapter(
        source=source,
        output_root=tmp_path / "scale",
        hardware_runtime=_hardware(),
        module=module,
    ).derive()

    assert result.source_geometry is source.source_geometry
    assert (
        result.source_geometry.geometry_solution.local_frame_id
        is source.source_geometry.geometry_solution.local_frame_id
    )
    assert (
        result.source_geometry.geometry_solution.scale_status
        is source.source_geometry.geometry_solution.scale_status
    )


def test_module_surface_contains_no_fusion_or_surface_execution_api() -> None:
    public_names = {name for name in vars(dense_module) if not name.startswith("_")}
    assert COLMAP_PATCH_MATCH_DENSE_DEPTH_ADAPTER_ID == "colmap.patch_match_dense_depth"
    assert "stereo_fusion" not in public_names
    assert not any("surface" in name.lower() for name in public_names)
    assert not any("mesh" in name.lower() for name in public_names)

    source = inspect.getsource(dense_module.ColmapPatchMatchDenseDepthAdapter.derive)
    assert "stereo_fusion" not in source
    assert "patch_match_stereo" in source


def test_registry_keeps_dense_depth_donor_experimental_and_cuda_isolated() -> None:
    adapter_registry = yaml.safe_load(_ADAPTER_REGISTRY_PATH.read_text(encoding="utf-8"))
    dependency_registry = yaml.safe_load(_DEPENDENCY_REGISTRY_PATH.read_text(encoding="utf-8"))

    entry = next(
        item
        for item in adapter_registry["entries"]
        if item["adapter_id"] == COLMAP_PATCH_MATCH_DENSE_DEPTH_ADAPTER_ID
    )
    assert entry["capability"] == {
        "name": "geometry.dense_depth.classical_mvs",
        "input_kinds": [
            "geometry.colmap_native_sparse_model",
            "geometry.solution",
            "image.observation",
        ],
        "output_kinds": ["geometry.dense_depth"],
    }
    assert entry["producer"] == {
        "implementation": COLMAP_PATCH_MATCH_DENSE_DEPTH_PRODUCER_IMPLEMENTATION,
        "version": "4.2.0",
        "revision": COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION,
    }
    assert entry["dependency_refs"] == [COLMAP_PATCH_MATCH_DENSE_DEPTH_DEPENDENCY_REF]
    assert entry["artifact_key_hardware_policy"] == "required"
    assert entry["license"]["review"] == "pending"
    assert entry["shipping_status"] == COLMAP_PATCH_MATCH_DENSE_DEPTH_SHIPPING_STATUS
    assert entry["model"] is None
    assert entry["checkpoint"] is None
    assert entry["metric_names"] == []
    assert entry["resume_mode"] == "unsupported"

    dependency = dependency_registry["dependencies"][COLMAP_PATCH_MATCH_DENSE_DEPTH_DEPENDENCY_REF]
    assert dependency["status"] == "candidate_optional"
    assert dependency["role"] == ["dense_depth_patchmatch_mvs"]
    assert dependency["integration"] == "official_pycolmap_cuda_external_environment"
    assert dependency["pinned_version"] == "4.2.0"
    assert dependency["python_package"] == "pycolmap-cuda12==4.2.0"
    assert dependency["source_revision"] == COLMAP_PATCH_MATCH_DENSE_DEPTH_SOURCE_REVISION
    assert dependency["upstream_repository"] == "colmap/colmap"
    assert dependency["license"] == "BSD-3-Clause"
    assert dependency["license_review"] == "pending"
