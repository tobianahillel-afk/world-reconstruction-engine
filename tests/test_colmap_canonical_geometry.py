from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

import wre.reconstruction.colmap_canonical_geometry as canonical_module
from wre.domain.artifacts import ArtifactKind
from wre.domain.geometry_solutions import GeometryScaleStatus
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.runs import (
    DerivedArtifactProvenance,
    ProducerRef,
    ReconstructionRun,
    ReconstructionRunId,
)
from wre.ingestion.hashing import hash_file_content
from wre.reconstruction.colmap_canonical_geometry import (
    COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID,
    COLMAP_INCREMENTAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY,
    COLMAP_INCREMENTAL_CANONICAL_CAPABILITY,
    COLMAP_INCREMENTAL_CANONICAL_CAPABILITY_NAME,
    COLMAP_INCREMENTAL_CANONICAL_CHECKPOINT,
    COLMAP_INCREMENTAL_CANONICAL_DEPENDENCY_REF,
    COLMAP_INCREMENTAL_CANONICAL_MODEL,
    COLMAP_INCREMENTAL_CANONICAL_PRODUCER_IMPLEMENTATION,
    COLMAP_INCREMENTAL_CANONICAL_PRODUCER_VERSION,
    COLMAP_INCREMENTAL_CANONICAL_REPRODUCIBILITY_NOTES,
    COLMAP_INCREMENTAL_CANONICAL_SHIPPING_STATUS,
    CanonicalColmapSparseModel,
    ColmapCanonicalGeometryError,
    canonicalize_colmap_sparse_model,
    colmap_sparse_model_content_identity,
)
from wre.reconstruction.colmap_environment import ColmapEnvironmentIdentity
from wre.reconstruction.colmap_features import (
    ColmapFeatureExtractionResult,
    ColmapImageFeatureSummary,
)
from wre.reconstruction.colmap_import import (
    COLMAP_IMPORTER_VERSION,
    ColmapReconstructionImportConfig,
    ColmapReconstructionImportRequest,
    import_colmap_reconstruction,
)
from wre.reconstruction.colmap_reconstruction import (
    ColmapIncrementalReconstructionResult,
    ColmapModelFileArtifact,
    ColmapSparseModelArtifact,
)
from wre.reconstruction.legacy_geometry_conversion import (
    convert_sparse_reconstruction_estimate,
)

_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _ROOT / "registry" / "adapter-models.yaml"

_ROTATION = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


class _FakeCamera:
    def __init__(
        self,
        *,
        model_name: object = "PINHOLE",
        width: object = 640,
        height: object = 480,
        params: object = (500.0, 510.0, 320.0, 240.0),
    ) -> None:
        self.model_name = model_name
        self.width = width
        self.height = height
        self.params = params
        self.has_prior_focal_length = False


class _FakeRotation:
    def __init__(self, matrix: object) -> None:
        self._matrix = matrix

    def matrix(self) -> object:
        return self._matrix


class _FakeTransform:
    def __init__(self, rotation: object, translation: object) -> None:
        self.rotation = _FakeRotation(rotation)
        self.translation = translation


class _FakeImage:
    def __init__(
        self,
        name: object,
        *,
        camera_id: object = 1,
        rotation: object = _ROTATION,
        translation: object = (3.0, -2.0, 5.0),
    ) -> None:
        self.name = name
        self.camera_id = camera_id
        self._transform = _FakeTransform(rotation, translation)

    def cam_from_world(self) -> _FakeTransform:
        return self._transform


class _FakePoint:
    def __init__(
        self,
        xyz: object,
        *,
        image_ids: tuple[int, ...] = (7, 11),
    ) -> None:
        self.xyz = xyz
        self.error = 0.25
        self.track = SimpleNamespace(
            elements=tuple(
                SimpleNamespace(image_id=image_id, point2D_idx=index + 2)
                for index, image_id in enumerate(image_ids)
            )
        )

    @staticmethod
    def has_error() -> bool:
        return True


class _FakeReconstruction:
    def __init__(
        self,
        *,
        camera: _FakeCamera | None = None,
        images: dict[int, _FakeImage] | None = None,
        points: dict[int, _FakePoint] | None = None,
        valid: bool = True,
    ) -> None:
        self._camera = camera or _FakeCamera()
        self._images = images or {
            11: _FakeImage(
                "000001-b.png",
                translation=(-1.0, 0.5, 4.0),
            ),
            7: _FakeImage("000000-a.png"),
        }
        self._points = points or {
            9: _FakePoint((9.0, 8.0, 7.0)),
            2: _FakePoint((1.0, 2.0, 3.0)),
        }
        self._valid = valid

    def is_valid(self) -> bool:
        return self._valid

    def reg_image_ids(self) -> list[int]:
        return list(self._images)

    def point3D_ids(self) -> list[int]:
        return list(self._points)

    def image(self, image_id: int) -> _FakeImage:
        return self._images[image_id]

    def camera(self, camera_id: int) -> _FakeCamera:
        assert camera_id == 1
        return self._camera

    def point3D(self, point_id: int) -> _FakePoint:
        return self._points[point_id]


class _FakePycolmap:
    __version__ = "4.2.0"
    COLMAP_version = "COLMAP 4.2.0"
    COLMAP_build = "Commit fake-v2l12-3 without GPU support"
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


def _features(
    tmp_path: Path,
    *,
    names: tuple[str, str] = ("000000-a.png", "000001-b.png"),
) -> ColmapFeatureExtractionResult:
    observation_ids = (ObservationId("obs:a"), ObservationId("obs:b"))
    return ColmapFeatureExtractionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:features"),
            source_observation_ids=observation_ids,
        ),
        environment=_environment(),
        configuration_sha256=Sha256Digest("1" * 64),
        database_path=tmp_path / "features.db",
        database_sha256=Sha256Digest("2" * 64),
        database_byte_length=1,
        images=tuple(
            ColmapImageFeatureSummary(
                observation_id=observation_id,
                image_name=name,
                keypoint_rows=4,
                keypoint_cols=4,
                descriptor_rows=4,
                descriptor_cols=128,
            )
            for observation_id, name in zip(observation_ids, names, strict=True)
        ),
    )


def _model_artifact(
    tmp_path: Path,
    *,
    index: int = 0,
) -> tuple[Path, ColmapSparseModelArtifact]:
    output_path = tmp_path / "sparse"
    model_path = output_path / str(index)
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
    return (
        output_path,
        ColmapSparseModelArtifact(
            model_index=index,
            relative_path=str(index),
            num_registered_images=2,
            num_points3d=2,
            files=tuple(files),
        ),
    )


def _canonicalize(
    tmp_path: Path,
    *,
    reconstruction: _FakeReconstruction | None = None,
    features: ColmapFeatureExtractionResult | None = None,
) -> tuple[CanonicalColmapSparseModel, Path, ColmapSparseModelArtifact, _FakePycolmap]:
    output_path, model = _model_artifact(tmp_path)
    module = _FakePycolmap(reconstruction)
    result = canonicalize_colmap_sparse_model(
        output_path=output_path,
        model_artifact=model,
        features=features or _features(tmp_path),
        expected_environment=_environment(),
        module=module,
    )
    return result, output_path, model, module


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


def test_canonical_aggregate_has_exact_immutable_shape_and_one_local_frame(
    tmp_path: Path,
) -> None:
    result, _, _, _ = _canonicalize(tmp_path)

    assert tuple(field.name for field in fields(CanonicalColmapSparseModel)) == (
        "source_model_index",
        "source_model_identity_sha256",
        "camera_solutions",
        "point_map",
        "geometry_solution",
    )
    with pytest.raises(FrozenInstanceError):
        result.source_model_index = 9  # type: ignore[misc]

    local_frame = result.geometry_solution.local_frame_id
    assert result.point_map.local_frame_id is local_frame
    assert all(camera.local_frame_id is local_frame for camera in result.camera_solutions)


def test_direct_native_geometry_preserves_pose_intrinsics_points_and_unresolved_scale(
    tmp_path: Path,
) -> None:
    result, _, model, module = _canonicalize(tmp_path)

    assert result.source_model_index == model.model_index
    assert result.source_model_identity_sha256 == colmap_sparse_model_content_identity(model)
    assert len(module.read_paths) == 1

    cameras_by_observation = {
        camera.observation_id: camera for camera in result.camera_solutions
    }
    assert set(cameras_by_observation) == {
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    }
    first = cameras_by_observation[ObservationId("obs:a")]
    assert first.projection_model.value == "pinhole"
    assert first.dimensions.width_px == 640
    assert first.dimensions.height_px == 480
    assert first.intrinsic_parameters == (500.0, 510.0, 320.0, 240.0)
    assert first.rotation_matrix == _ROTATION
    assert first.translation_xyz == (3.0, -2.0, 5.0)
    assert first.uncertainty_artifacts == ()
    assert first.metrics.observations == ()

    assert result.point_map.source_observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
    )
    assert result.point_map.positions_xyz == (
        (1.0, 2.0, 3.0),
        (9.0, 8.0, 7.0),
    )
    assert result.point_map.confidence is None
    assert result.point_map.metrics.observations == ()

    assert result.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert result.geometry_solution.camera_solution_ids == tuple(
        camera.solution_id for camera in result.camera_solutions
    )
    assert result.geometry_solution.point_map_ids == (result.point_map.point_map_id,)
    assert result.geometry_solution.depth_field_ids == ()
    assert result.geometry_solution.metrics.observations == ()


def test_camera_from_local_pose_is_not_inverted_transposed_or_rescaled(
    tmp_path: Path,
) -> None:
    result, _, _, _ = _canonicalize(tmp_path)
    camera = next(
        item
        for item in result.camera_solutions
        if item.observation_id == ObservationId("obs:a")
    )
    point = (2.0, 7.0, -1.0)

    transformed = (
        camera.rotation_matrix[0][0] * point[0]
        + camera.rotation_matrix[0][1] * point[1]
        + camera.rotation_matrix[0][2] * point[2]
        + camera.translation_xyz[0],
        camera.rotation_matrix[1][0] * point[0]
        + camera.rotation_matrix[1][1] * point[1]
        + camera.rotation_matrix[1][2] * point[2]
        + camera.translation_xyz[1],
        camera.rotation_matrix[2][0] * point[0]
        + camera.rotation_matrix[2][1] * point[1]
        + camera.rotation_matrix[2][2] * point[2]
        + camera.translation_xyz[2],
    )

    assert transformed == (-4.0, 0.0, 4.0)


def test_source_model_identity_changes_when_audited_content_identity_changes(
    tmp_path: Path,
) -> None:
    _, model = _model_artifact(tmp_path)
    original = colmap_sparse_model_content_identity(model)
    first_file = model.files[0]
    changed_file = ColmapModelFileArtifact(
        relative_path=first_file.relative_path,
        sha256=Sha256Digest("f" * 64),
        byte_length=first_file.byte_length,
    )
    changed = ColmapSparseModelArtifact(
        model_index=model.model_index,
        relative_path=model.relative_path,
        num_registered_images=model.num_registered_images,
        num_points3d=model.num_points3d,
        files=(changed_file, *model.files[1:]),
    )

    assert colmap_sparse_model_content_identity(changed) != original


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("tamper", "changed after mapper publication"),
        ("missing", "membership changed"),
        ("extra", "membership changed"),
    ],
)
def test_native_manifest_is_verified_before_pycolmap_read(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    output_path, model = _model_artifact(tmp_path)
    model_path = output_path / "0"
    if mutation == "tamper":
        (model_path / "images.bin").write_bytes(b"tampered")
    elif mutation == "missing":
        (model_path / "images.bin").unlink()
    else:
        (model_path / "extra.bin").write_bytes(b"extra")
    module = _FakePycolmap()

    with pytest.raises(ColmapCanonicalGeometryError, match=message):
        canonicalize_colmap_sparse_model(
            output_path=output_path,
            model_artifact=model,
            features=_features(tmp_path),
            expected_environment=_environment(),
            module=module,
        )

    assert module.read_paths == []


def test_native_model_symlink_is_rejected_before_pycolmap_read(tmp_path: Path) -> None:
    real_output, model = _model_artifact(tmp_path / "real")
    linked_output = tmp_path / "linked"
    linked_output.mkdir()
    (linked_output / "0").symlink_to(real_output / "0", target_is_directory=True)
    module = _FakePycolmap()

    with pytest.raises(ColmapCanonicalGeometryError, match="symbolic link"):
        canonicalize_colmap_sparse_model(
            output_path=linked_output,
            model_artifact=model,
            features=_features(tmp_path),
            expected_environment=_environment(),
            module=module,
        )

    assert module.read_paths == []


def test_unknown_registered_image_fails_closed(tmp_path: Path) -> None:
    reconstruction = _FakeReconstruction(
        images={
            7: _FakeImage("000000-a.png"),
            11: _FakeImage("unknown.png"),
        }
    )

    with pytest.raises(ColmapCanonicalGeometryError, match="no observation mapping"):
        _canonicalize(tmp_path, reconstruction=reconstruction)


@pytest.mark.parametrize(
    ("camera", "message"),
    [
        (_FakeCamera(model_name="SIMPLE RADIAL"), "projection model"),
        (_FakeCamera(width=0), "camera width"),
        (_FakeCamera(params=(500.0, float("nan"))), "camera parameter"),
        (_FakeCamera(params=()), "parameters must not be empty"),
    ],
)
def test_malformed_camera_values_fail_closed(
    tmp_path: Path,
    camera: _FakeCamera,
    message: str,
) -> None:
    with pytest.raises(ColmapCanonicalGeometryError, match=message):
        _canonicalize(tmp_path, reconstruction=_FakeReconstruction(camera=camera))


@pytest.mark.parametrize(
    ("rotation", "translation"),
    [
        (((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)), (0.0, 0.0, 0.0)),
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            (0.0, float("inf"), 0.0),
        ),
    ],
)
def test_malformed_camera_pose_fails_closed(
    tmp_path: Path,
    rotation: object,
    translation: object,
) -> None:
    reconstruction = _FakeReconstruction(
        images={
            7: _FakeImage(
                "000000-a.png",
                rotation=rotation,
                translation=translation,
            ),
            11: _FakeImage("000001-b.png", translation=(-1.0, 0.5, 4.0)),
        }
    )

    with pytest.raises(ColmapCanonicalGeometryError):
        _canonicalize(tmp_path, reconstruction=reconstruction)


def test_direct_native_matches_retained_legacy_importer_and_bridge(
    tmp_path: Path,
) -> None:
    output_path, model = _model_artifact(tmp_path)
    features = _features(tmp_path)
    module = _FakePycolmap()

    direct = canonicalize_colmap_sparse_model(
        output_path=output_path,
        model_artifact=model,
        features=features,
        expected_environment=_environment(),
        module=module,
    )

    reconstruction = ColmapIncrementalReconstructionResult(
        provenance=DerivedArtifactProvenance(
            producing_run_id=ReconstructionRunId("run:incremental"),
            source_observation_ids=features.provenance.source_observation_ids,
        ),
        environment=_environment(),
        configuration_sha256=Sha256Digest("3" * 64),
        source_verification_database_sha256=Sha256Digest("4" * 64),
        output_path=output_path,
        models=(model,),
    )
    import_config = ColmapReconstructionImportConfig()
    import_run = ReconstructionRun(
        run_id=ReconstructionRunId("run:legacy-import"),
        producer=ProducerRef(
            implementation="wre.colmap_reconstruction_importer",
            version=COLMAP_IMPORTER_VERSION,
        ),
        input_observation_ids=features.provenance.source_observation_ids,
        started_at=datetime(2026, 9, 22, tzinfo=UTC),
        configuration_sha256=import_config.sha256,
    )
    legacy = import_colmap_reconstruction(
        ColmapReconstructionImportRequest(
            run=import_run,
            reconstruction=reconstruction,
            features=features,
            config=import_config,
        ),
        module=module,
    )
    bridged = convert_sparse_reconstruction_estimate(legacy.models[0].estimate)

    direct_cameras = {
        camera.observation_id: camera for camera in direct.camera_solutions
    }
    bridged_cameras = {
        camera.observation_id: camera for camera in bridged.camera_solutions
    }
    assert set(direct_cameras) == set(bridged_cameras)
    for observation_id in direct_cameras:
        assert (
            direct_cameras[observation_id].rotation_matrix
            == bridged_cameras[observation_id].rotation_matrix
        )
        assert (
            direct_cameras[observation_id].translation_xyz
            == bridged_cameras[observation_id].translation_xyz
        )
        assert (
            direct_cameras[observation_id].intrinsic_parameters
            == bridged_cameras[observation_id].intrinsic_parameters
        )
        assert (
            direct_cameras[observation_id].projection_model
            == bridged_cameras[observation_id].projection_model
        )

    assert direct.point_map.positions_xyz == bridged.point_map.positions_xyz
    assert direct.point_map.source_observation_ids == bridged.point_map.source_observation_ids
    assert (
        direct.geometry_solution.scale_status
        is bridged.geometry_solution.scale_status
        is GeometryScaleStatus.UNRESOLVED
    )


def test_incremental_canonical_adapter_registry_contract() -> None:
    assert COLMAP_INCREMENTAL_CANONICAL_CAPABILITY.capability is (
        COLMAP_INCREMENTAL_CANONICAL_CAPABILITY_NAME
    )
    assert COLMAP_INCREMENTAL_CANONICAL_CAPABILITY.input_kinds == frozenset(
        {
            ArtifactKind("image.observation"),
            ArtifactKind("evidence.geometric_verification"),
        }
    )
    assert COLMAP_INCREMENTAL_CANONICAL_CAPABILITY.output_kinds == frozenset(
        {
            ArtifactKind("geometry.camera_solution"),
            ArtifactKind("geometry.point_map"),
            ArtifactKind("geometry.solution"),
        }
    )

    entry = _entries_by_id()[COLMAP_INCREMENTAL_CANONICAL_ADAPTER_ID]
    assert entry["capability"] == {
        "name": COLMAP_INCREMENTAL_CANONICAL_CAPABILITY_NAME.value,
        "input_kinds": ["image.observation", "evidence.geometric_verification"],
        "output_kinds": [
            "geometry.camera_solution",
            "geometry.point_map",
            "geometry.solution",
        ],
    }
    assert entry["producer"] == {
        "implementation": COLMAP_INCREMENTAL_CANONICAL_PRODUCER_IMPLEMENTATION,
        "version": COLMAP_INCREMENTAL_CANONICAL_PRODUCER_VERSION,
        "revision": None,
    }
    assert entry["dependency_refs"] == [COLMAP_INCREMENTAL_CANONICAL_DEPENDENCY_REF]
    assert entry["model"] is COLMAP_INCREMENTAL_CANONICAL_MODEL
    assert entry["checkpoint"] is COLMAP_INCREMENTAL_CANONICAL_CHECKPOINT
    assert (
        entry["artifact_key_hardware_policy"]
        == COLMAP_INCREMENTAL_CANONICAL_ARTIFACT_KEY_HARDWARE_POLICY
    )
    assert entry["shipping_status"] == COLMAP_INCREMENTAL_CANONICAL_SHIPPING_STATUS
    assert entry["reproducibility_notes"] == (
        COLMAP_INCREMENTAL_CANONICAL_REPRODUCIBILITY_NOTES
    )


def test_existing_precision_and_legacy_registry_entries_remain_distinct() -> None:
    entries = _entries_by_id()
    assert entries["colmap.precision_geometry"]["capability"]["input_kinds"] == [
        "image.observation"
    ]
    assert entries["colmap.precision_geometry"]["capability"]["output_kinds"] == [
        "geometry.camera_solution",
        "geometry.point_map",
        "geometry.solution",
    ]
    assert entries["colmap.sparse_sfm"]["capability"]["output_kinds"] == [
        "geometry.sparse_reconstruction_estimate"
    ]


def test_canonicalization_module_does_not_depend_on_legacy_geometry_or_later_layers() -> None:
    forbidden_names = {
        "SparseReconstructionEstimate",
        "CameraCalibrationEstimate",
        "CameraPoseEstimate",
        "Point3DEstimate",
        "convert_sparse_reconstruction_estimate",
        "SurfaceModel",
        "SpatialFragment",
        "CameraId",
        "DepthField",
        "MasterScene",
        "RuntimeScene",
        "Router",
        "SQLiteLocalStore",
    }

    assert forbidden_names.isdisjoint(vars(canonical_module))
