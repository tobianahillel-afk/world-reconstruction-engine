from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from wre.domain.estimated_geometry import LocalScaleStatus
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_environment import (
    ColmapEnvironmentIdentity,
    inspect_colmap_environment,
)
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_import import (
    COLMAP_IMPORTER_VERSION,
    ColmapReconstructionImportConfig,
    ColmapReconstructionImportError,
    ColmapReconstructionImportRequest,
    import_colmap_reconstruction,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionResult,
    ColmapModelFileArtifact,
    ColmapSparseModelArtifact,
)


class _FakeCamera:
    model_name = "SIMPLE_RADIAL"
    width = 640
    height = 480
    params = (500.0, 320.0, 240.0, 0.01)
    has_prior_focal_length = False


class _FakeRotation:
    @staticmethod
    def matrix() -> tuple[tuple[float, float, float], ...]:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


class _FakeTransform:
    rotation = _FakeRotation()
    translation = (0.0, 0.0, 0.0)


class _FakeImage:
    def __init__(self, name: str, camera_id: int = 1) -> None:
        self.name = name
        self.camera_id = camera_id

    @staticmethod
    def cam_from_world() -> _FakeTransform:
        return _FakeTransform()


class _FakePoint:
    def __init__(self, image_ids: tuple[int, ...]) -> None:
        self.xyz = (1.0, 2.0, 3.0)
        self.error = 0.2
        self.track = SimpleNamespace(
            elements=[
                SimpleNamespace(image_id=image_id, point2D_idx=index + 3)
                for index, image_id in enumerate(image_ids)
            ]
        )

    @staticmethod
    def has_error() -> bool:
        return True


class _FakeReconstruction:
    def __init__(
        self,
        image_names: tuple[str, ...] = ("000000-a.png", "000001-b.png"),
    ) -> None:
        self._images = {index + 1: _FakeImage(name) for index, name in enumerate(image_names)}
        self._points = {7: _FakePoint(tuple(self._images))}

    @staticmethod
    def is_valid() -> bool:
        return True

    def reg_image_ids(self) -> list[int]:
        return list(self._images)

    def point3D_ids(self) -> list[int]:
        return list(self._points)

    def image(self, image_id: int) -> _FakeImage:
        return self._images[image_id]

    @staticmethod
    def camera(camera_id: int) -> _FakeCamera:
        assert camera_id == 1
        return _FakeCamera()

    def point3D(self, point_id: int) -> _FakePoint:
        return self._points[point_id]


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-l36 without GPU support"
    __ceres_version__ = "2.2.0"
    has_cuda = False

    def __init__(self, reconstruction: _FakeReconstruction | None = None) -> None:
        self.reconstruction = reconstruction or _FakeReconstruction()
        self.read_paths: list[Path] = []

    def Reconstruction(self, path: Path) -> _FakeReconstruction:
        self.read_paths.append(Path(path))
        return self.reconstruction


def _environment() -> ColmapEnvironmentIdentity:
    return ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build=_FakePycolmap.COLMAP_build,
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )


def _model_artifact(root: Path, *, index: int = 0) -> ColmapSparseModelArtifact:
    model_path = root / str(index)
    model_path.mkdir(parents=True)
    files: list[ColmapModelFileArtifact] = []
    for name, content in (
        ("cameras.bin", b"camera"),
        ("images.bin", b"images"),
        ("points3D.bin", b"points"),
    ):
        path = model_path / name
        path.write_bytes(content)
        digest = hash_file_content(path)
        files.append(
            ColmapModelFileArtifact(
                relative_path=name,
                sha256=digest.sha256,
                byte_length=digest.byte_length,
            )
        )
    return ColmapSparseModelArtifact(
        model_index=index,
        relative_path=str(index),
        num_registered_images=2,
        num_points3d=1,
        files=tuple(files),
    )


def _feature_result(
    tmp_path: Path,
    *,
    image_names: tuple[str, ...] = ("000000-a.png", "000001-b.png"),
    environment: ColmapEnvironmentIdentity | None = None,
) -> ColmapFeatureExtractionResult:
    observation_ids = tuple(ObservationId(f"obs:{index}") for index in range(len(image_names)))
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:l32"),
            source_observation_ids=observation_ids,
        ),
        environment=environment or _environment(),
        configuration_sha256=Sha256Digest("1" * 64),
        database_path=tmp_path / "features.db",
        database_sha256=Sha256Digest("2" * 64),
        database_byte_length=1,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=image_name,
                keypoint_rows=10,
                keypoint_cols=4,
                descriptor_rows=10,
                descriptor_cols=128,
            )
            for observation_id, image_name in zip(observation_ids, image_names, strict=True)
        ),
    )


def _reconstruction_result(
    tmp_path: Path,
    *,
    models: tuple[ColmapSparseModelArtifact, ...],
    environment: ColmapEnvironmentIdentity | None = None,
) -> ColmapIncrementalReconstructionResult:
    return ColmapIncrementalReconstructionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:l35"),
            source_observation_ids=(ObservationId("obs:0"), ObservationId("obs:1")),
        ),
        environment=environment or _environment(),
        configuration_sha256=Sha256Digest("3" * 64),
        source_verification_database_sha256=Sha256Digest("4" * 64),
        output_path=tmp_path / "sparse",
        models=models,
    )


def _request(
    tmp_path: Path,
    *,
    module: _FakePycolmap | None = None,
) -> tuple[ColmapReconstructionImportRequest, _FakePycolmap]:
    output_path = tmp_path / "sparse"
    model = _model_artifact(output_path)
    reconstruction = _reconstruction_result(tmp_path, models=(model,))
    features = _feature_result(tmp_path)
    config = ColmapReconstructionImportConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l36"),
        producer=ProducerRef(
            implementation="wre.colmap_reconstruction_importer",
            version=COLMAP_IMPORTER_VERSION,
        ),
        input_observation_ids=reconstruction.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    return (
        ColmapReconstructionImportRequest(
            run=run,
            reconstruction=reconstruction,
            features=features,
            config=config,
        ),
        module or _FakePycolmap(),
    )


def test_import_maps_native_model_into_explicit_estimated_geometry(tmp_path: Path) -> None:
    request, module = _request(tmp_path)

    result = import_colmap_reconstruction(request, module=module)

    assert result.model_count == 1
    assert result.has_reconstruction is True
    assert len(module.read_paths) == 1
    imported = result.models[0]
    estimate = imported.estimate
    assert imported.source_model_index == 0
    assert estimate.scale_status is LocalScaleStatus.UNRESOLVED
    assert estimate.observation_ids == (ObservationId("obs:0"), ObservationId("obs:1"))
    assert estimate.observation_count == 2
    assert estimate.point_count == 1
    assert len(estimate.camera_calibrations) == 1
    assert tuple(item.observation_id for item in estimate.camera_poses) == estimate.observation_ids
    assert estimate.camera_poses[0].rotation_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert estimate.points3d[0].position_xyz == (1.0, 2.0, 3.0)
    assert tuple(item.observation_id for item in estimate.points3d[0].track) == (
        ObservationId("obs:0"),
        ObservationId("obs:1"),
    )


def test_import_rejects_tampered_native_model_before_pycolmap_read(tmp_path: Path) -> None:
    request, module = _request(tmp_path)
    (request.reconstruction.output_path / "0" / "images.bin").write_bytes(b"tampered")

    with pytest.raises(
        ColmapReconstructionImportError,
        match=r"changed after L3\.5",
    ):
        import_colmap_reconstruction(request, module=module)

    assert module.read_paths == []


def test_import_rejects_registered_image_without_l32_mapping(tmp_path: Path) -> None:
    module = _FakePycolmap(_FakeReconstruction(("000000-a.png", "unknown.png")))
    request, _ = _request(tmp_path, module=module)

    with pytest.raises(
        ColmapReconstructionImportError,
        match=r"no L3\.2 observation mapping",
    ):
        import_colmap_reconstruction(request, module=module)


def test_zero_solver_models_remain_explicit_unresolved_without_loading_pycolmap(
    tmp_path: Path,
) -> None:
    features = _feature_result(tmp_path)
    reconstruction = _reconstruction_result(tmp_path, models=())
    config = ColmapReconstructionImportConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:l36"),
        producer=ProducerRef(
            implementation="wre.colmap_reconstruction_importer",
            version=COLMAP_IMPORTER_VERSION,
        ),
        input_observation_ids=reconstruction.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 15, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )
    request = ColmapReconstructionImportRequest(
        run=run,
        reconstruction=reconstruction,
        features=features,
        config=config,
    )
    module = _FakePycolmap()

    result = import_colmap_reconstruction(request, module=module)

    assert result.models == ()
    assert result.model_count == 0
    assert result.has_reconstruction is False
    assert module.read_paths == []


def test_import_rejects_different_native_reader_environment(tmp_path: Path) -> None:
    request, module = _request(tmp_path)
    module.COLMAP_build = "different build"

    with pytest.raises(
        ColmapReconstructionImportError,
        match=r"exact L3\.5 PyCOLMAP environment",
    ):
        import_colmap_reconstruction(request, module=module)


@pytest.mark.skipif(
    os.environ.get("WRE_COLMAP_INTEGRATION") != "1",
    reason="real PyCOLMAP integration runs only in the dedicated COLMAP lane",
)
def test_real_pycolmap_reconstruction_can_be_imported_without_fragment_acceptance(
    tmp_path: Path,
) -> None:
    pycolmap = pytest.importorskip("pycolmap")
    environment = inspect_colmap_environment(pycolmap)
    database_path = tmp_path / "synthetic.db"
    with pycolmap.Database.open(database_path) as database:
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs = 2
        options.num_cameras_per_rig = 1
        options.num_frames_per_rig = 4
        options.num_points3D = 30
        reconstruction_native = pycolmap.synthesize_dataset(options, database)

    output_path = tmp_path / "sparse"
    model_path = output_path / "0"
    model_path.mkdir(parents=True)
    reconstruction_native.write(model_path)

    model_files: list[ColmapModelFileArtifact] = []
    for path in sorted(model_path.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file():
            continue
        digest = hash_file_content(path)
        model_files.append(
            ColmapModelFileArtifact(
                relative_path=path.relative_to(model_path).as_posix(),
                sha256=digest.sha256,
                byte_length=digest.byte_length,
            )
        )
    model = ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=reconstruction_native.num_reg_images(),
        num_points3d=reconstruction_native.num_points3D(),
        files=tuple(model_files),
    )

    native_images = sorted(
        (
            reconstruction_native.image(image_id)
            for image_id in reconstruction_native.reg_image_ids()
        ),
        key=lambda image: image.name,
    )
    observation_ids = tuple(
        ObservationId(f"obs:{index:03d}") for index in range(len(native_images))
    )
    features = ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:real-l32"),
            source_observation_ids=observation_ids,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("5" * 64),
        database_path=database_path,
        database_sha256=hash_file_content(database_path).sha256,
        database_byte_length=hash_file_content(database_path).byte_length,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=str(image.name),
                keypoint_rows=1,
                keypoint_cols=2,
                descriptor_rows=1,
                descriptor_cols=128,
            )
            for observation_id, image in zip(observation_ids, native_images, strict=True)
        ),
    )
    reconstruction = ColmapIncrementalReconstructionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:real-l35"),
            source_observation_ids=observation_ids,
        ),
        environment=environment,
        configuration_sha256=Sha256Digest("6" * 64),
        source_verification_database_sha256=Sha256Digest("7" * 64),
        output_path=output_path,
        models=(model,),
    )
    config = ColmapReconstructionImportConfig()
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:real-l36"),
        producer=ProducerRef(
            implementation="wre.colmap_reconstruction_importer",
            version=COLMAP_IMPORTER_VERSION,
        ),
        input_observation_ids=observation_ids,
        started_at=datetime(2026, 9, 15, tzinfo=UTC),
        configuration_sha256=config.sha256,
    )

    result = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=run,
            reconstruction=reconstruction,
            features=features,
            config=config,
        ),
        module=pycolmap,
    )

    assert result.model_count == 1
    estimate = result.models[0].estimate
    assert estimate.scale_status is LocalScaleStatus.UNRESOLVED
    assert estimate.observation_count == reconstruction_native.num_reg_images()
    assert estimate.point_count == reconstruction_native.num_points3D()
    assert all(pose.local_frame_id == estimate.local_frame_id for pose in estimate.camera_poses)
    assert all(point.local_frame_id == estimate.local_frame_id for point in estimate.points3d)
