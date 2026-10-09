from __future__ import annotations

import ast
import hashlib
import struct
import zlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import wre.reconstruction.static_appearance_baseline as baseline_module
from wre.domain import (
    LINEAR_SRGB_F64,
    ArtifactInputFingerprint,
    ArtifactKeyMaterial,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMetadata,
    ColorConversionRequest,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    DecodedPixelColorEncoding,
    ObservationKind,
    ObservationMetadata,
    PhotometricCompatibilityAssessment,
    PhotometricCompatibilityInput,
    PhotometricCompatibilityStatus,
    PhotometricNormalizationFactors,
    PhotometricNormalizationRequest,
    ProvenanceClass,
    RawMetadataEntry,
    SceneProjectId,
    SourceColorMetadata,
    SourceExposureMetadata,
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
    SourceWhiteBalanceMetadata,
    assess_color_conversion,
    assess_photometric_compatibility,
    assess_photometric_normalization,
    derive_artifact_key,
)
from wre.domain.appearance import APPEARANCE_MODEL_ARTIFACT_KIND, AppearanceRepresentationName
from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.camera_solutions import (
    CameraProjectionModelName,
    CameraSolution,
    CameraSolutionId,
)
from wre.domain.cameras import ImageDimensions
from wre.domain.fragments import LocalFrameId
from wre.domain.geometry_solutions import (
    GeometryScaleStatus,
    GeometrySolution,
    GeometrySolutionId,
)
from wre.domain.metrics import MetricVector
from wre.domain.observations import (
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.domain.point_maps import PointMap, PointMapId
from wre.domain.producer_identity import ArtifactProducerIdentity, ConfigurationIdentity
from wre.domain.runs import DerivedArtifactProvenance, ProducerRef, ReconstructionRunId
from wre.photometry import (
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)
from wre.reconstruction.colmap_canonical_geometry import (
    CanonicalColmapSparseModel,
    colmap_sparse_model_content_identity,
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
from wre.reconstruction.geometry_solution_comparison import GeometrySolutionCandidate
from wre.reconstruction.static_appearance_baseline import (
    GSPLAT_REFERENCE_PYCOLMAP_REVISION,
    GSPLAT_STATIC_APPEARANCE_REPRESENTATION,
    GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION,
    GSPLAT_STATIC_APPEARANCE_SOURCE_TREE,
    GsplatStaticAppearancePreflightError,
    GsplatStaticAppearancePreflightSource,
    GsplatStaticAppearanceTrainingProfile,
    materialize_verified_gsplat_ply,
    inspect_gsplat_static_appearance_shared_track_pixels,
    preflight_gsplat_static_appearance_inputs,
    verify_gsplat_reference_pycolmap_sources,
    verify_gsplat_static_appearance_matching_capture_metadata,
    verify_gsplat_static_appearance_native_geometry,
    verify_gsplat_static_appearance_reference_sources,
    verify_gsplat_static_appearance_shared_scene_tracks,
    verify_gsplat_static_appearance_source_photometry,
)
from wre.reconstruction.static_appearance_candidate import StaticAppearanceCandidateRequest


def _digest(data: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(data).hexdigest())


def _reviewed_source_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    """Small local synthetic tree; never pretends to be the real upstream commit."""

    root = tmp_path / "reference"
    data = {
        "LICENSE": b"synthetic license fixture",
        "examples/simple_trainer.py": b"print('fixture')\n",
        "examples/requirements.txt": b"# fixture requirements\n",
        "gsplat/version.py": b"__version__ = 'fixture'\n",
    }
    for path, payload in data.items():
        candidate = root / path
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(payload)
    monkeypatch.setattr(
        baseline_module,
        "_GSPLAT_REFERENCE_SOURCE_FILES",
        tuple(
            (path, len(payload), baseline_module._git_blob_sha1(payload))
            for path, payload in sorted(data.items())
        ),
    )
    return root, data


def test_gsplat_reference_source_git_pins_are_the_reviewed_upstream_identity() -> None:
    assert GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION == ("937e29912570c372bed6747a5c9bf85fed877bae")
    assert GSPLAT_STATIC_APPEARANCE_SOURCE_TREE == ("90c3f0b2352e6d2725bcba1ef0407168f922c0fa")
    pins = baseline_module._GSPLAT_REFERENCE_SOURCE_FILES
    assert len(pins) == 10
    assert tuple(path for path, _size, _sha in pins) == tuple(
        sorted(path for path, _size, _sha in pins)
    )
    assert (
        "examples/simple_trainer.py",
        49728,
        "6a30be737b5c9af53a140f64faf499d8d4d0933f",
    ) in pins
    assert (
        "examples/datasets/colmap.py",
        18447,
        "6c21f2c663b60d9dc38471a9963a3b5d092e5ed2",
    ) in pins
    assert baseline_module._git_blob_sha1(b'__version__ = "1.5.3"\n') == (
        "a06ff4e08777642c97011d6c993690dba5a7a02f"
    )


def test_gsplat_reference_source_gate_verifies_exact_bytes_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = _reviewed_source_fixture(tmp_path, monkeypatch)
    result = verify_gsplat_static_appearance_reference_sources(root)
    assert tuple(entry.relative_path for entry in result) == tuple(sorted(data))
    for entry in result:
        payload = data[entry.relative_path]
        assert entry.byte_length == len(payload)
        assert entry.sha256 == _digest(payload)
    assert result == verify_gsplat_static_appearance_reference_sources(root)


def test_gsplat_reference_source_gate_rejects_same_size_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _data = _reviewed_source_fixture(tmp_path, monkeypatch)
    target = root / "examples/simple_trainer.py"
    target.write_bytes(b"X" + target.read_bytes()[1:])
    with pytest.raises(GsplatStaticAppearancePreflightError, match="Git blob differs"):
        verify_gsplat_static_appearance_reference_sources(root)


def test_gsplat_reference_source_gate_rejects_missing_source_or_wrong_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, _data = _reviewed_source_fixture(tmp_path, monkeypatch)
    target = root / "gsplat/version.py"
    target.unlink()
    with pytest.raises(GsplatStaticAppearancePreflightError, match="missing"):
        verify_gsplat_static_appearance_reference_sources(root)
    target.write_bytes(b"bad")
    with pytest.raises(GsplatStaticAppearancePreflightError, match="byte length"):
        verify_gsplat_static_appearance_reference_sources(root)


def test_gsplat_reference_source_gate_rejects_symlink_directory_and_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = _reviewed_source_fixture(tmp_path, monkeypatch)
    alias = tmp_path / "source-alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        verify_gsplat_static_appearance_reference_sources(alias)
    target = root / "LICENSE"
    target.unlink()
    target.symlink_to(root / "examples/simple_trainer.py")
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        verify_gsplat_static_appearance_reference_sources(root)
    target.unlink()
    target.write_bytes(data["LICENSE"])
    (root / "examples").rename(root / "real-examples")
    (root / "examples").symlink_to(root / "real-examples", target_is_directory=True)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        verify_gsplat_static_appearance_reference_sources(root)


def _reference_pycolmap_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "reader-fork"
    data = {
        "LICENSE.txt": b"MIT synthetic fixture",
        "pyproject.toml": b"[project]\\nname = 'pycolmap'\\n",
        "pycolmap/__init__.py": b"from .scene_manager import SceneManager\\n",
        "pycolmap/scene_manager.py": b"class SceneManager:\\n    pass\\n",
    }
    for relative, payload in data.items():
        file = root / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(payload)
    monkeypatch.setattr(
        baseline_module,
        "_GSPLAT_REFERENCE_PYCOLMAP_FILES",
        tuple(
            (path, len(data[path]), baseline_module._git_blob_sha1(data[path]))
            for path in sorted(data)
        ),
    )
    return root, data


def test_gsplat_reference_reader_has_exact_pinned_source_identity() -> None:
    assert GSPLAT_REFERENCE_PYCOLMAP_REVISION == "cc7ea4b7301720ac29287dbe450952511b32125e"
    pins = baseline_module._GSPLAT_REFERENCE_PYCOLMAP_FILES
    assert len(pins) == 9
    assert tuple(path for path, _size, _sha in pins) == tuple(
        sorted(path for path, _size, _sha in pins)
    )
    assert (
        "LICENSE.txt",
        1084,
        "5156d3d49e0c312561c59680658b6261f635abe3",
    ) in pins
    assert (
        "pycolmap/scene_manager.py",
        26996,
        "352f051f71aad8bbad70c0b6cc63b83d3ab90ce5",
    ) in pins


def test_gsplat_reference_reader_checks_exact_files_without_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = _reference_pycolmap_fixture(tmp_path, monkeypatch)
    entries = verify_gsplat_reference_pycolmap_sources(root)
    assert tuple(item.relative_path for item in entries) == tuple(sorted(data))
    for item in entries:
        assert item.byte_length == len(data[item.relative_path])
        assert item.sha256 == _digest(data[item.relative_path])
    assert verify_gsplat_reference_pycolmap_sources(root) == entries


def test_gsplat_reference_reader_rejects_tampered_and_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, original = _reference_pycolmap_fixture(tmp_path, monkeypatch)
    path = root / "pycolmap/scene_manager.py"
    path.write_bytes(b"X" + original["pycolmap/scene_manager.py"][1:])
    with pytest.raises(GsplatStaticAppearancePreflightError, match="Git blob differs"):
        verify_gsplat_reference_pycolmap_sources(root)
    path.write_bytes(original["pycolmap/scene_manager.py"])
    (root / "LICENSE.txt").unlink()
    with pytest.raises(GsplatStaticAppearancePreflightError, match="missing"):
        verify_gsplat_reference_pycolmap_sources(root)


def test_gsplat_reference_reader_rejects_foreign_python_and_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, data = _reference_pycolmap_fixture(tmp_path, monkeypatch)
    extra = root / "pycolmap/foreign.py"
    extra.write_text("def unreviewed(): pass\\n")
    with pytest.raises(GsplatStaticAppearancePreflightError, match="closure differs"):
        verify_gsplat_reference_pycolmap_sources(root)
    extra.unlink()
    path = root / "pycolmap/scene_manager.py"
    path.unlink()
    path.symlink_to(root / "pycolmap/__init__.py")
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        verify_gsplat_reference_pycolmap_sources(root)
    path.unlink()
    path.write_bytes(data["pycolmap/scene_manager.py"])
    alias = tmp_path / "reader-symlink"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        verify_gsplat_reference_pycolmap_sources(alias)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _png(width: int = 2, height: int = 1, *, srgb: bool = True) -> bytes:
    image_header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + b"\x40\x80\xc0" * width) * height
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", image_header)
        + (_chunk(b"sRGB", b"\x00") if srgb else b"")
        + _chunk(b"IDAT", zlib.compress(pixels))
        + _chunk(b"IEND", b"")
    )


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(implementation="test.static_appearance", version="1"),
        configuration=ConfigurationIdentity(sha256=_digest(b"configuration")),
    )


def _parts(
    tmp_path: Path, *, png: bytes | None = None
) -> tuple[StaticAppearanceCandidateRequest, GsplatStaticAppearancePreflightSource]:
    image_bytes = png if png is not None else _png()
    image_root = tmp_path / "images"
    image_root.mkdir()
    source_file = image_root / "frame.png"
    source_file.write_bytes(image_bytes)
    observation_id = ObservationId("obs:frame")
    observation = ImageObservation(
        observation_id=observation_id,
        asset=MediaAssetRef(
            uri="file:///supplied/frame.png",
            sha256=_digest(image_bytes),
            byte_length=len(image_bytes),
            mime_type="image/png",
        ),
        source=SourceRef(SourceId("source:frame")),
        received_at=datetime(2026, 10, 8, tzinfo=UTC),
    )
    input_image = ColmapReconstructionInput(
        observation=observation,
        source_path=source_file,
        image_name="frame.png",
    )

    model_root = tmp_path / "models"
    model_dir = model_root / "0"
    model_dir.mkdir(parents=True)
    cameras = b"verified-native-cameras"
    points = b"verified-native-points"
    (model_dir / "cameras.bin").write_bytes(cameras)
    (model_dir / "points3D.bin").write_bytes(points)
    native = ColmapSparseModelArtifact(
        model_index=0,
        relative_path="0",
        num_registered_images=1,
        num_points3d=1,
        files=(
            ColmapModelFileArtifact("cameras.bin", _digest(cameras), len(cameras)),
            ColmapModelFileArtifact("points3D.bin", _digest(points), len(points)),
        ),
    )
    native_ref = colmap_native_sparse_model_artifact_ref(native)

    frame = LocalFrameId("frame:test")
    camera = CameraSolution(
        solution_id=CameraSolutionId("cam:frame"),
        observation_id=observation_id,
        local_frame_id=frame,
        projection_model=CameraProjectionModelName("pinhole"),
        dimensions=ImageDimensions(width_px=2, height_px=1),
        intrinsic_parameters=(2.0, 2.0, 1.0, 0.5),
        rotation_matrix=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        translation_xyz=(0.0, 0.0, 0.0),
        uncertainty_artifacts=(),
        metrics=MetricVector(observations=()),
    )
    point_map = PointMap(
        point_map_id=PointMapId("points:test"),
        local_frame_id=frame,
        source_observation_ids=(observation_id,),
        positions_xyz=((1.0, 2.0, 3.0),),
        confidence=None,
        metrics=MetricVector(observations=()),
    )
    geometry = GeometrySolution(
        geometry_solution_id=GeometrySolutionId("geometry:test"),
        local_frame_id=frame,
        scale_status=GeometryScaleStatus.UNRESOLVED,
        camera_solution_ids=(camera.solution_id,),
        depth_field_ids=(),
        point_map_ids=(point_map.point_map_id,),
        metrics=MetricVector(observations=()),
    )
    candidate = GeometrySolutionCandidate(
        geometry_solution=geometry,
        camera_solutions=(camera,),
        depth_fields=(),
        point_maps=(point_map,),
        producer=_producer(),
        source_artifacts=(ArtifactRef(ArtifactId("source:test"), ArtifactKind("geometry.input")),),
    )
    request = StaticAppearanceCandidateRequest(
        source_geometry=candidate,
        source_surface=None,
        source_observation_ids=(observation_id,),
        output_representation=AppearanceRepresentationName(GSPLAT_STATIC_APPEARANCE_REPRESENTATION),
        supporting_artifacts=(native_ref,),
    )
    source = GsplatStaticAppearancePreflightSource(
        image_root=image_root,
        images=(input_image,),
        native_model_root=model_root,
        native_model_artifact=native,
        native_model_ref=native_ref,
    )
    return request, source


def test_exact_srgb_png_and_native_manifest_byte_gate(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    evidence = preflight_gsplat_static_appearance_inputs(request, source)
    assert evidence.source_geometry_id == "geometry:test"
    assert evidence.native_model_ref == source.native_model_ref
    assert len(evidence.images) == 1
    assert evidence.images[0].width_px == 2
    assert evidence.images[0].height_px == 1
    assert evidence.images[0].sha256 == source.images[0].observation.asset.sha256
    assert request.source_geometry.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED
    assert GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION == ("937e29912570c372bed6747a5c9bf85fed877bae")


def test_rejects_tampered_image_bytes(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    (source.image_root / "frame.png").write_bytes(_png(2, 2))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="SHA-256"):
        preflight_gsplat_static_appearance_inputs(request, source)


def test_rejects_absent_srgb_tag_even_with_correct_image_hash(tmp_path: Path) -> None:
    request, source = _parts(tmp_path, png=_png(srgb=False))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="sRGB"):
        preflight_gsplat_static_appearance_inputs(request, source)


def test_rejects_png_dimensions_that_disagree_with_camera(tmp_path: Path) -> None:
    request, source = _parts(tmp_path, png=_png(3, 1))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="dimensions"):
        preflight_gsplat_static_appearance_inputs(request, source)


def test_rejects_corrupt_png_pixel_data_even_if_asset_digest_matches(tmp_path: Path) -> None:
    corrupted = bytearray(_png())
    pos = corrupted.index(b"IDAT") + 5
    corrupted[pos] ^= 0xFF
    request, source = _parts(tmp_path, png=bytes(corrupted))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="checksum"):
        preflight_gsplat_static_appearance_inputs(request, source)


def test_rejects_native_sparse_model_tampering(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    (source.native_model_root / "0" / "points3D.bin").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed after mapper publication"):
        preflight_gsplat_static_appearance_inputs(request, source)


def test_rejects_missing_native_support_and_foreign_observations(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="ArtifactRef"):
        preflight_gsplat_static_appearance_inputs(replace(request, supporting_artifacts=()), source)
    foreign = replace(
        request,
        source_observation_ids=(ObservationId("obs:unrelated"),),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="ObservationId"):
        preflight_gsplat_static_appearance_inputs(foreign, source)


def test_rejects_unknown_projection_and_implicit_resize(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    camera = request.source_geometry.camera_solutions[0]
    distorted = replace(camera, projection_model=CameraProjectionModelName("simple_radial"))
    alternative = replace(request.source_geometry, camera_solutions=(distorted,))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="PINHOLE"):
        preflight_gsplat_static_appearance_inputs(
            replace(request, source_geometry=alternative), source
        )


def test_rejects_image_root_symlink(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(source.image_root, target_is_directory=True)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="symlink"):
        preflight_gsplat_static_appearance_inputs(request, replace(source, image_root=alias))


def test_rejects_unknown_representation(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="representation"):
        preflight_gsplat_static_appearance_inputs(
            replace(request, output_representation=AppearanceRepresentationName("mesh")),
            source,
        )


def test_preflight_has_no_trainer_execution_or_optional_framework_import() -> None:
    import wre.reconstruction.static_appearance_baseline as module

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(value.name.split(".")[0] for value in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module.split(".")[0])
    assert imported <= {
        "__future__",
        "dataclasses",
        "hashlib",
        "json",
        "math",
        "pathlib",
        "struct",
        "typing",
        "wre",
        "zlib",
    }
    assert not {"torch", "gsplat", "numpy", "subprocess", "pycolmap"} & imported


def test_safe_training_profile_disables_unsafe_upstream_defaults() -> None:
    profile = GsplatStaticAppearanceTrainingProfile()
    overrides = profile.reference_overrides()
    assert profile.expected_ply_relative_path == "ply/point_cloud_29999.ply"
    assert overrides["data_factor"] == 1
    assert overrides["normalize_world_space"] is False
    assert overrides["global_scale"] == 1.0
    assert overrides["camera_model"] == "pinhole"
    assert overrides["pose_opt"] is False
    assert overrides["pose_noise"] == 0.0
    assert overrides["disable_viewer"] is True
    assert overrides["disable_video"] is True
    assert overrides["ckpt"] is None
    assert overrides["init_type"] == "sfm"
    assert overrides["sh_degree"] == 3
    assert overrides["save_ply"] is True
    assert overrides["ply_steps"] == [30_000]
    assert overrides["eval_steps"] == []
    assert overrides["save_steps"] == []
    assert overrides["steps_scaler"] == 1.0
    assert overrides["with_ut"] is False
    assert overrides["app_opt"] is False
    assert overrides["depth_loss"] is False
    assert "data_dir" not in overrides
    assert "result_dir" not in overrides


def test_safe_training_profile_canonical_hash_and_source_identity() -> None:
    profile = GsplatStaticAppearanceTrainingProfile(max_steps=400, test_every=4)
    assert (
        profile.configuration_sha256
        == GsplatStaticAppearanceTrainingProfile(max_steps=400, test_every=4).configuration_sha256
    )
    assert (
        profile.configuration_sha256
        != GsplatStaticAppearanceTrainingProfile(max_steps=401, test_every=4).configuration_sha256
    )
    assert (
        profile.configuration_sha256
        != GsplatStaticAppearanceTrainingProfile(max_steps=400, test_every=5).configuration_sha256
    )
    assert profile.expected_ply_relative_path == "ply/point_cloud_399.ply"
    document = profile.canonical_document()
    assert document["trainer_source_revision"] == GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION
    assert document["trainer_git_blob_sha1"] == "6a30be737b5c9af53a140f64faf499d8d4d0933f"
    assert document["upstream_config_overrides"] == profile.reference_overrides()


def test_safe_training_profile_is_immutable_and_returns_new_override_document() -> None:
    profile = GsplatStaticAppearanceTrainingProfile(max_steps=100)
    identity = profile.configuration_sha256
    overrides = profile.reference_overrides()
    overrides["normalize_world_space"] = True
    ply_steps = overrides["ply_steps"]
    assert isinstance(ply_steps, list)
    ply_steps.append(123)
    assert profile.reference_overrides()["normalize_world_space"] is False
    assert profile.reference_overrides()["ply_steps"] == [100]
    assert profile.configuration_sha256 == identity
    with pytest.raises(FrozenInstanceError):
        profile.max_steps = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    (
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": True}, "max_steps"),
        ({"max_steps": 1.5}, "max_steps"),
        ({"test_every": 1}, "test_every"),
        ({"test_every": True}, "test_every"),
        ({"schema_version": 2}, "schema_version"),
    ),
)
def test_safe_training_profile_refuses_invalid_parameters(
    kwargs: dict[str, object], reason: str
) -> None:
    with pytest.raises(ValueError, match=reason):
        GsplatStaticAppearanceTrainingProfile(**kwargs)  # type: ignore[arg-type]


def _gsplat_ply(*, properties: tuple[str, ...] | None = None, vertices: int = 1) -> bytes:
    """Exact uncompressed gsplat exporter layout with finite float32 content."""
    standard = (
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *(f"f_rest_{index}" for index in range(45)),
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    fields = properties or standard
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {vertices}\n"
        + "".join(f"property float {field}\n" for field in fields)
        + "end_header\n"
    ).encode("ascii")
    payload = tuple(float(index + 1) for index in range(len(fields)))
    return header + struct.pack("<" + "f" * len(payload), *payload) * vertices


def _gsplat_output(
    tmp_path: Path, data: bytes, *, steps: int = 20
) -> tuple[Path, GsplatStaticAppearanceTrainingProfile, ArtifactRef]:
    output = tmp_path / "output"
    (output / "ply").mkdir(parents=True)
    profile = GsplatStaticAppearanceTrainingProfile(max_steps=steps)
    (output / profile.expected_ply_relative_path).write_bytes(data)
    ref = ArtifactRef(ArtifactId("appearance:gsplat:fixture"), APPEARANCE_MODEL_ARTIFACT_KIND)
    return output, profile, ref


def test_gsplat_ply_materialization_hashes_real_output_and_retains_path(tmp_path: Path) -> None:
    data = _gsplat_ply(vertices=2)
    root, profile, ref = _gsplat_output(tmp_path, data)
    result = materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)
    assert result.artifact_ref == ref
    assert len(result.entries) == 1
    assert result.entries[0].relative_path == "ply/point_cloud_19.ply"
    assert result.entries[0].byte_length == len(data)
    assert result.entries[0].sha256 == _digest(data)
    assert (
        materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)
        == result
    )


@pytest.mark.parametrize("rest_count", (0, 3, 42, 48))
def test_gsplat_ply_rejects_wrong_degree_or_partial_harmonics(
    tmp_path: Path, rest_count: int
) -> None:
    """Even well-ordered RGB SH values cannot masquerade as the pinned degree-3 payload."""
    properties = (
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        *(f"f_rest_{index}" for index in range(rest_count)),
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    )
    root, profile, ref = _gsplat_output(tmp_path, _gsplat_ply(properties=properties), steps=20)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="property schema"):
        materialize_verified_gsplat_ply(
            artifact_ref=ref,
            output_root=root,
            profile=profile,
        )


@pytest.mark.parametrize(
    ("data", "reason"),
    [
        (b"not-ply\n", "not PLY"),
        (
            _gsplat_ply().replace(b"format binary_little_endian", b"format ascii"),
            "uncompressed little-endian",
        ),
        (_gsplat_ply(vertices=0), "non-empty vertex"),
        (
            _gsplat_ply().replace(b"property float opacity", b"property double opacity"),
            "unexpected element",
        ),
        (_gsplat_ply(properties=("x", "y", "z", "opacity")), "missing Gaussian"),
        (_gsplat_ply().replace(b"f_rest_1", b"f_rest_2"), "property schema"),
        (_gsplat_ply()[:-3], "truncated"),
        (_gsplat_ply() + b"extra", "unexpected bytes"),
    ],
)
def test_gsplat_ply_materialization_rejects_invalid_format_or_bytes(
    tmp_path: Path, data: bytes, reason: str
) -> None:
    root, profile, ref = _gsplat_output(tmp_path, data)
    with pytest.raises(GsplatStaticAppearancePreflightError, match=reason):
        materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)


def test_gsplat_ply_materialization_rejects_nonfinite_payload(tmp_path: Path) -> None:
    data = bytearray(_gsplat_ply())
    data[-4:] = struct.pack("<f", float("nan"))
    root, profile, ref = _gsplat_output(tmp_path, bytes(data))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="non-finite"):
        materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)


def test_gsplat_ply_materialization_rejects_zero_gaussians_and_forged_header(
    tmp_path: Path,
) -> None:
    data = _gsplat_ply(vertices=1).replace(b"element vertex 1", b"element vertex 0")
    root, profile, ref = _gsplat_output(tmp_path, data)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="non-empty vertex"):
        materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)


def test_gsplat_ply_materialization_rejects_symlink_and_wrong_artifact_kind(
    tmp_path: Path,
) -> None:
    root, profile, ref = _gsplat_output(tmp_path, _gsplat_ply())
    with pytest.raises(GsplatStaticAppearancePreflightError, match=r"appearance\.static"):
        materialize_verified_gsplat_ply(
            artifact_ref=ArtifactRef(ArtifactId("other"), ArtifactKind("geometry.input")),
            output_root=root,
            profile=profile,
        )
    expected = root / profile.expected_ply_relative_path
    destination = tmp_path / "replacement.ply"
    destination.write_bytes(expected.read_bytes())
    expected.unlink()
    expected.symlink_to(destination)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="missing"):
        materialize_verified_gsplat_ply(artifact_ref=ref, output_root=root, profile=profile)


def test_gsplat_ply_materialization_refuses_missing_final_step_file(
    tmp_path: Path,
) -> None:
    root, _profile, ref = _gsplat_output(tmp_path, _gsplat_ply(), steps=20)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="missing"):
        materialize_verified_gsplat_ply(
            artifact_ref=ref,
            output_root=root,
            profile=GsplatStaticAppearanceTrainingProfile(max_steps=21),
        )


def test_gsplat_ply_materialization_has_no_optional_training_import() -> None:
    source = Path(__file__).parents[1] / "src/wre/reconstruction/static_appearance_baseline.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(item.name.split(".")[0] for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not {"torch", "gsplat", "numpy", "subprocess", "pycolmap", "PIL"} & imported


def _native_feature_evidence(
    tmp_path: Path,
) -> tuple[ColmapFeatureExtractionResult, ColmapEnvironmentIdentity]:
    environment = ColmapEnvironmentIdentity(
        pycolmap_version="4.2.0",
        colmap_version="COLMAP 4.2.0",
        colmap_build="fixture",
        ceres_version="2.2.0",
        upstream_has_cuda=False,
    )
    return (
        ColmapFeatureExtractionResult(
            provenance=DerivedArtifactProvenance(
                producing_run_id=ReconstructionRunId("run:gsplat-features"),
                source_observation_ids=(ObservationId("obs:frame"),),
            ),
            environment=environment,
            configuration_sha256=_digest(b"features-config"),
            database_path=tmp_path / "features.db",
            database_sha256=_digest(b"features-db"),
            database_byte_length=1,
            images=(
                ColmapImageFeatureSummary(
                    observation_id=ObservationId("obs:frame"),
                    image_name="frame.png",
                    keypoint_rows=3,
                    keypoint_cols=4,
                    descriptor_rows=3,
                    descriptor_cols=128,
                ),
            ),
        ),
        environment,
    )


def _native_canonical(
    request: StaticAppearanceCandidateRequest,
    source: GsplatStaticAppearancePreflightSource,
) -> CanonicalColmapSparseModel:
    geometry = request.source_geometry
    return CanonicalColmapSparseModel(
        source_model_index=source.native_model_artifact.model_index,
        source_model_identity_sha256=colmap_sparse_model_content_identity(
            source.native_model_artifact
        ),
        camera_solutions=geometry.camera_solutions,
        point_map=geometry.point_maps[0],
        geometry_solution=geometry.geometry_solution,
    )


def test_native_geometry_reuses_audited_pycolmap_reader_and_preserves_exact_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    expected = _native_canonical(request, source)
    calls: list[dict[str, object]] = []

    def canonicalize(**kwargs: object) -> CanonicalColmapSparseModel:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(baseline_module, "canonicalize_colmap_sparse_model", canonicalize)
    result = verify_gsplat_static_appearance_native_geometry(
        request,
        source,
        features=features,
        expected_environment=environment,
        module=None,
    )
    assert result is expected
    assert len(calls) == 1
    assert calls[0]["model_artifact"] is source.native_model_artifact
    assert calls[0]["output_path"] == source.native_model_root
    assert calls[0]["features"] is features
    assert calls[0]["expected_environment"] is environment
    assert calls[0]["module"] is None


def test_native_geometry_rejects_foreign_camera_even_if_geometry_id_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    expected = _native_canonical(request, source)
    monkeypatch.setattr(
        baseline_module, "canonicalize_colmap_sparse_model", lambda **kwargs: expected
    )
    camera = request.source_geometry.camera_solutions[0]
    different_camera = replace(camera, intrinsic_parameters=(3.0, 2.0, 1.0, 0.5))
    different_geometry = replace(request.source_geometry, camera_solutions=(different_camera,))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="do not match"):
        verify_gsplat_static_appearance_native_geometry(
            replace(request, source_geometry=different_geometry),
            source,
            features=features,
            expected_environment=environment,
        )


def test_native_geometry_rejects_foreign_point_map_or_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    expected = _native_canonical(request, source)
    monkeypatch.setattr(
        baseline_module, "canonicalize_colmap_sparse_model", lambda **kwargs: expected
    )
    other_point = replace(request.source_geometry.point_maps[0], positions_xyz=((2.0, 2.0, 3.0),))
    other_map = replace(request.source_geometry, point_maps=(other_point,))
    metric = replace(
        request.source_geometry.geometry_solution, scale_status=GeometryScaleStatus.METRIC
    )
    other_scale = replace(request.source_geometry, geometry_solution=metric)
    for variant in (other_map, other_scale):
        with pytest.raises(GsplatStaticAppearancePreflightError, match="do not match"):
            verify_gsplat_static_appearance_native_geometry(
                replace(request, source_geometry=variant),
                source,
                features=features,
                expected_environment=environment,
            )


def test_native_geometry_fails_before_pycolmap_on_foreign_name_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    wrong = replace(
        features,
        images=(replace(features.images[0], image_name="different.png"),),
    )

    def fail_if_called(**kwargs: object) -> None:
        pytest.fail("native reader must not execute after a rejected image mapping")

    monkeypatch.setattr(baseline_module, "canonicalize_colmap_sparse_model", fail_if_called)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="naming"):
        verify_gsplat_static_appearance_native_geometry(
            request, source, features=wrong, expected_environment=environment
        )


def test_native_geometry_rejects_inconsistent_native_content_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    canonical = replace(
        _native_canonical(request, source),
        source_model_identity_sha256=_digest(b"foreign-native-model"),
    )
    monkeypatch.setattr(
        baseline_module, "canonicalize_colmap_sparse_model", lambda **kwargs: canonical
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="identity"):
        verify_gsplat_static_appearance_native_geometry(
            request, source, features=features, expected_environment=environment
        )


def _gsplat_photo_assessment(
    source: GsplatStaticAppearancePreflightSource,
    *,
    observation_id: ObservationId | None = None,
    asset_sha256: Sha256Digest | None = None,
    exposure_ev: float = 0.0,
    capture: tuple[float, float, float, float] | None = None,
    white_balance: tuple[str, float] | None = None,
) -> PhotometricCompatibilityAssessment:
    """Real V2L17 plans bound to the same source asset, not mocked compatible labels."""

    image = source.images[0]
    observation = observation_id or image.observation.observation_id
    source_digest = asset_sha256 or image.observation.asset.sha256
    rgb = bytes((64, 128, 192, 64, 128, 192))
    declaration = RawMetadataEntry("caller", "color", "sRGB")
    exposure_entry = (
        RawMetadataEntry("caller", "exposure", repr(capture)) if capture is not None else None
    )
    balance_entry = (
        RawMetadataEntry("caller", "white_balance", repr(white_balance))
        if white_balance is not None
        else None
    )
    raw_entries = tuple(
        item for item in (declaration, exposure_entry, balance_entry) if item is not None
    )
    photo = SourcePhotometryMetadata(
        observation_id=observation,
        source_metadata=ObservationMetadata(
            observation_id=observation,
            raw_entries=raw_entries,
        ),
        color=SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            declared_primaries="bt709_d65",
            declared_transfer_characteristic="srgb",
            declared_matrix_coefficients="identity",
            declared_range="full",
            evidence=(declaration,),
        ),
        exposure=(
            SourceExposureMetadata(
                status=SourcePhotometryInterpretationStatus.RESOLVED,
                iso_speed=capture[0],
                exposure_time_seconds=capture[1],
                f_number=capture[2],
                exposure_compensation_ev=capture[3],
                evidence=(exposure_entry,),
            )
            if capture is not None and exposure_entry is not None
            else SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT)
        ),
        white_balance=(
            SourceWhiteBalanceMetadata(
                status=SourcePhotometryInterpretationStatus.RESOLVED,
                mode=white_balance[0],
                color_temperature_kelvin=white_balance[1],
                evidence=(balance_entry,),
            )
            if white_balance is not None and balance_entry is not None
            else SourceWhiteBalanceMetadata(status=SourcePhotometryInterpretationStatus.ABSENT)
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=observation,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=source_digest,
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0,
                width_px=2,
                height_px=1,
                relative_path="levels/level-000000.rgb",
            ),
        ),
    )
    decoded_ref = ArtifactRef(
        ArtifactId("artifact:gsplat-photometric-decoded"),
        ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = _producer()
    decoded_artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:gsplat-photometric"),
        artifact_ref=decoded_ref,
        artifact_key=derive_artifact_key(
            ArtifactKeyMaterial(
                output_kind=decoded_ref.artifact_kind,
                input_fingerprints=(
                    ArtifactInputFingerprint(
                        artifact_kind=ArtifactKind("image.observation"),
                        sha256=source_digest,
                    ),
                ),
                producer=producer,
            )
        ),
        producer=producer,
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )
    materialization = ArtifactMaterializationMetadata(
        artifact_ref=decoded_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path="levels/level-000000.rgb",
                sha256=_digest(rgb),
                byte_length=len(rgb),
            ),
        ),
    )
    color_request = ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=decoded_artifact,
        decoded_materialization=materialization,
        source_photometry=photo,
        decoded_level_index=0,
        decoded_encoding=DecodedPixelColorEncoding.SRGB_FULL_RGB8,
        working_convention=LINEAR_SRGB_F64,
        output_convention=LINEAR_SRGB_F64,
    )
    color = assess_color_conversion(color_request)
    assert color.plan is not None
    converted = convert_rgb8_to_linear_reference(color.plan, rgb)
    normalized_request = PhotometricNormalizationRequest(
        source_plan=color.plan,
        source_content_sha256=converted.content_sha256,
        source_derived_sha256=converted.derived_sha256,
        source_photometry=photo,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=exposure_ev,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
    )
    normalization = assess_photometric_normalization(normalized_request)
    assert normalization.plan is not None
    normalized = normalize_linear_rgb_reference(normalization.plan, converted)
    assessment = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color,
            normalization_assessment=normalization,
            normalized_content_sha256=normalized.content_sha256,
            normalized_derived_sha256=normalized.derived_sha256,
        )
    )
    assert assessment.status is PhotometricCompatibilityStatus.COMPATIBLE
    return assessment


def test_gsplat_photo_gate_requires_exact_ready_identity_evidence(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    assessment = _gsplat_photo_assessment(source)
    assert verify_gsplat_static_appearance_source_photometry(request, source, (assessment,)) == (
        assessment.identity,
    )
    assert request.source_observation_ids == (ObservationId("obs:frame"),)


def test_gsplat_photo_gate_fails_on_mismatched_source_image_hash(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    other = _gsplat_photo_assessment(source, asset_sha256=_digest(b"other"))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="exact source PNG"):
        verify_gsplat_static_appearance_source_photometry(request, source, (other,))


def test_gsplat_photo_gate_fails_on_foreign_observation(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    foreign = _gsplat_photo_assessment(source, observation_id=ObservationId("obs:foreign"))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="exact source PNG"):
        verify_gsplat_static_appearance_source_photometry(request, source, (foreign,))


def test_gsplat_photo_gate_fails_on_unapplied_exposure_correction(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    assessment = _gsplat_photo_assessment(source, exposure_ev=1.0)
    with pytest.raises(GsplatStaticAppearancePreflightError, match="raw PNG"):
        verify_gsplat_static_appearance_source_photometry(request, source, (assessment,))


def test_gsplat_photo_gate_requires_complete_immutable_evidence(tmp_path: Path) -> None:
    request, source = _parts(tmp_path)
    with pytest.raises(TypeError, match="immutable tuple"):
        verify_gsplat_static_appearance_source_photometry(request, source, [])  # type: ignore[arg-type]
    with pytest.raises(GsplatStaticAppearancePreflightError, match="requires one"):
        verify_gsplat_static_appearance_source_photometry(request, source, ())
    with pytest.raises(TypeError, match="PhotometricCompatibilityAssessment"):
        verify_gsplat_static_appearance_source_photometry(
            request,
            source,
            ("compatible",),  # type: ignore[arg-type]
        )
    assessment = _gsplat_photo_assessment(source)
    false_label = PhotometricCompatibilityAssessment(
        status=PhotometricCompatibilityStatus.UNRESOLVED,
        compatibility_input=assessment.compatibility_input,
        reasons=("unverified",),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="unresolved"):
        verify_gsplat_static_appearance_source_photometry(request, source, (false_label,))


def _two_view_parts(
    tmp_path: Path,
) -> tuple[StaticAppearanceCandidateRequest, GsplatStaticAppearancePreflightSource]:
    request, source = _parts(tmp_path)
    first = source.images[0]
    second_id = ObservationId("obs:second")
    second_path = source.image_root / "second.png"
    second_path.write_bytes(first.source_path.read_bytes())
    second_observation = replace(
        first.observation,
        observation_id=second_id,
        asset=replace(first.observation.asset, uri="file:///supplied/second.png"),
        source=SourceRef(SourceId("source:second")),
    )
    second_image = replace(
        first,
        observation=second_observation,
        source_path=second_path,
        image_name="second.png",
    )
    native = replace(source.native_model_artifact, num_registered_images=2)
    native_ref = colmap_native_sparse_model_artifact_ref(native)
    geometry = request.source_geometry
    second_camera = replace(
        geometry.camera_solutions[0],
        observation_id=second_id,
        solution_id=CameraSolutionId("cam:second"),
        translation_xyz=(1.0, 0.0, 0.0),
    )
    point_map = replace(
        geometry.point_maps[0],
        source_observation_ids=(ObservationId("obs:frame"), second_id),
    )
    source_geometry = replace(
        geometry,
        camera_solutions=(*geometry.camera_solutions, second_camera),
        point_maps=(point_map,),
        geometry_solution=replace(
            geometry.geometry_solution,
            camera_solution_ids=(
                geometry.camera_solutions[0].solution_id,
                second_camera.solution_id,
            ),
        ),
    )
    return (
        replace(
            request,
            source_geometry=source_geometry,
            source_observation_ids=(ObservationId("obs:frame"), second_id),
            supporting_artifacts=(native_ref,),
        ),
        replace(
            source,
            images=(first, second_image),
            native_model_artifact=native,
            native_model_ref=native_ref,
        ),
    )


def test_multiview_capture_gate_accepts_only_matching_resolved_declarations(
    tmp_path: Path,
) -> None:
    request, source = _two_view_parts(tmp_path)
    capture = (100.0, 0.01, 4.0, 0.0)
    balance = ("manual", 5600.0)
    assessments = (
        _gsplat_photo_assessment(source, capture=capture, white_balance=balance),
        _gsplat_photo_assessment(
            source,
            observation_id=ObservationId("obs:second"),
            capture=capture,
            white_balance=balance,
        ),
    )
    assert verify_gsplat_static_appearance_matching_capture_metadata(
        request, source, assessments
    ) == tuple(item.identity for item in assessments)
    assert request.source_observation_ids == (
        ObservationId("obs:frame"),
        ObservationId("obs:second"),
    )


def test_multiview_capture_gate_rejects_missing_or_partial_declarations(
    tmp_path: Path,
) -> None:
    request, source = _two_view_parts(tmp_path)
    complete = _gsplat_photo_assessment(
        source,
        capture=(100.0, 0.01, 4.0, 0.0),
        white_balance=("manual", 5600.0),
    )
    missing = _gsplat_photo_assessment(source, observation_id=ObservationId("obs:second"))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="incomplete or unresolved"):
        verify_gsplat_static_appearance_matching_capture_metadata(
            request, source, (complete, missing)
        )
    partial = _gsplat_photo_assessment(
        source,
        observation_id=ObservationId("obs:second"),
        capture=(100.0, 0.01, 4.0, 0.0),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="incomplete or unresolved"):
        verify_gsplat_static_appearance_matching_capture_metadata(
            request, source, (complete, partial)
        )


@pytest.mark.parametrize(
    ("second_capture", "second_balance"),
    [
        ((200.0, 0.01, 4.0, 0.0), ("manual", 5600.0)),
        ((100.0, 0.02, 4.0, 0.0), ("manual", 5600.0)),
        ((100.0, 0.01, 5.6, 0.0), ("manual", 5600.0)),
        ((100.0, 0.01, 4.0, 1.0), ("manual", 5600.0)),
        ((100.0, 0.01, 4.0, 0.0), ("manual", 6500.0)),
        ((100.0, 0.01, 4.0, 0.0), ("auto", 5600.0)),
    ],
)
def test_multiview_capture_gate_rejects_any_declared_capture_drift(
    tmp_path: Path,
    second_capture: tuple[float, float, float, float],
    second_balance: tuple[str, float],
) -> None:
    request, source = _two_view_parts(tmp_path)
    assessments = (
        _gsplat_photo_assessment(
            source,
            capture=(100.0, 0.01, 4.0, 0.0),
            white_balance=("manual", 5600.0),
        ),
        _gsplat_photo_assessment(
            source,
            observation_id=ObservationId("obs:second"),
            capture=second_capture,
            white_balance=second_balance,
        ),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="settings differ"):
        verify_gsplat_static_appearance_matching_capture_metadata(request, source, assessments)


def test_multiview_capture_gate_requires_two_exact_distinct_source_observations(
    tmp_path: Path,
) -> None:
    request, source = _parts(tmp_path)
    one = _gsplat_photo_assessment(
        source,
        capture=(100.0, 0.01, 4.0, 0.0),
        white_balance=("manual", 5600.0),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="at least two"):
        verify_gsplat_static_appearance_matching_capture_metadata(request, source, (one,))
    second_root = tmp_path / "other"
    second_root.mkdir()
    multi_request, multi_source = _two_view_parts(second_root)
    foreign = _gsplat_photo_assessment(
        multi_source,
        observation_id=ObservationId("obs:foreign"),
        capture=(100.0, 0.01, 4.0, 0.0),
        white_balance=("manual", 5600.0),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="exact source PNG"):
        verify_gsplat_static_appearance_matching_capture_metadata(
            multi_request, multi_source, (one, foreign)
        )


def _shared_track_fixture(
    *,
    track_image_ids: tuple[int, ...] = (1, 2),
    point2d_index: int = 0,
    reciprocal_point_id: int = 7,
    second_image_name: str = "second.png",
    first_xy: tuple[float, float] = (0.5, 0.5),
    second_xy: tuple[float, float] = (0.5, 0.5),
    valid: bool = True,
) -> object:
    images = {
        1: SimpleNamespace(
            name="frame.png",
            num_points2D=lambda: 1,
            point2D=lambda _index: SimpleNamespace(
                point3D_id=reciprocal_point_id, xy=first_xy
            ),
        ),
        2: SimpleNamespace(
            name=second_image_name,
            num_points2D=lambda: 1,
            point2D=lambda _index: SimpleNamespace(
                point3D_id=reciprocal_point_id, xy=second_xy
            ),
        ),
    }
    track = SimpleNamespace(
        elements=tuple(
            SimpleNamespace(image_id=image_id, point2D_idx=point2d_index)
            for image_id in track_image_ids
        )
    )
    reconstruction = SimpleNamespace(
        is_valid=lambda: valid,
        reg_image_ids=lambda: (1, 2),
        point3D_ids=lambda: (7,),
        image=lambda image_id: images[image_id],
        point3D=lambda _point_id: SimpleNamespace(track=track),
    )
    return SimpleNamespace(Reconstruction=lambda _path: reconstruction)


def _matching_shared_track_photometry(
    source: GsplatStaticAppearancePreflightSource,
) -> tuple[PhotometricCompatibilityAssessment, ...]:
    return (
        _gsplat_photo_assessment(
            source,
            capture=(100.0, 0.01, 4.0, 0.0),
            white_balance=("manual", 5600.0),
        ),
        _gsplat_photo_assessment(
            source,
            observation_id=ObservationId("obs:second"),
            capture=(100.0, 0.01, 4.0, 0.0),
            white_balance=("manual", 5600.0),
        ),
    )


def test_shared_scene_tracks_retain_real_reciprocal_pair_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _two_view_parts(tmp_path)
    assessments = _matching_shared_track_photometry(source)
    features, env = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    result = verify_gsplat_static_appearance_shared_scene_tracks(
        request,
        source,
        assessments,
        features=features,
        expected_environment=env,
        module=_shared_track_fixture(),
    )
    assert len(result) == 1
    pair = result[0]
    assert pair.left_observation_id == ObservationId("obs:frame")
    assert pair.right_observation_id == ObservationId("obs:second")
    assert pair.shared_point_count == 1
    assert request.source_geometry.geometry_solution.scale_status is GeometryScaleStatus.UNRESOLVED


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"track_image_ids": (1,)}, "do not connect"),
        ({"track_image_ids": (1, 1)}, "duplicate registered"),
        ({"track_image_ids": (1, 3)}, "foreign or duplicate"),
        ({"point2d_index": 1}, "point2D index"),
        ({"reciprocal_point_id": 999}, "not reciprocal"),
        ({"second_image_name": "foreign.png"}, "names differ"),
        ({"valid": False}, "invalid during track"),
    ],
)
def test_shared_scene_track_gate_rejects_invalid_or_unconnected_native_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    reason: str,
) -> None:
    request, source = _two_view_parts(tmp_path)
    assessments = _matching_shared_track_photometry(source)
    features, env = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match=reason):
        verify_gsplat_static_appearance_shared_scene_tracks(
            request,
            source,
            assessments,
            features=features,
            expected_environment=env,
            module=_shared_track_fixture(**cast(Any, overrides)),
        )


def test_shared_scene_track_gate_never_skips_source_capture_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _two_view_parts(tmp_path)
    features, env = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    bad_assessments = (
        _matching_shared_track_photometry(source)[0],
        _gsplat_photo_assessment(source, observation_id=ObservationId("obs:second")),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="incomplete or unresolved"):
        verify_gsplat_static_appearance_shared_scene_tracks(
            request,
            source,
            bad_assessments,
            features=features,
            expected_environment=env,
            module=_shared_track_fixture(),
        )


def test_shared_track_pixel_evidence_samples_exact_identical_srgb_pngs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _two_view_parts(tmp_path)
    assessments = _matching_shared_track_photometry(source)
    features, environment = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    result = inspect_gsplat_static_appearance_shared_track_pixels(
        request, source, assessments,
        features=features, expected_environment=environment, module=_shared_track_fixture()
    )
    assert len(result) == 1
    assert result[0].left_observation_id == ObservationId("obs:frame")
    assert result[0].right_observation_id == ObservationId("obs:second")
    assert result[0].left_png_sha256 == source.images[0].observation.asset.sha256
    assert result[0].right_png_sha256 == source.images[1].observation.asset.sha256
    assert result[0].shared_track_count == 1
    assert result[0].mean_absolute_srgb_channel_delta == 0.0
    assert inspect_gsplat_static_appearance_shared_track_pixels(
        request, source, assessments,
        features=features, expected_environment=environment, module=_shared_track_fixture()
    ) == result


def test_shared_track_pixel_evidence_reports_actual_color_difference_without_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _two_view_parts(tmp_path)
    first = source.images[1]
    image_header = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
    second_png = (
        b"\\x89PNG\\r\\n\\x1a\\n"
        + _chunk(b"IHDR", image_header)
        + _chunk(b"sRGB", b"\\x00")
        + _chunk(b"IDAT", zlib.compress(b"\\x00\\x41\\x80\\xc0\\x40\\x80\\xc0"))
        + _chunk(b"IEND", b"")
    )
    first.source_path.write_bytes(second_png)
    modified = replace(
        first,
        observation=replace(
            first.observation,
            asset=replace(
                first.observation.asset,
                sha256=_digest(second_png),
                byte_length=len(second_png),
            ),
        ),
    )
    source = replace(source, images=(source.images[0], modified))
    assessments = _matching_shared_track_photometry(source)
    features, environment = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    result = inspect_gsplat_static_appearance_shared_track_pixels(
        request, source, assessments,
        features=features, expected_environment=environment, module=_shared_track_fixture()
    )
    assert result[0].shared_track_count == 1
    assert result[0].mean_absolute_srgb_channel_delta == pytest.approx(1 / 765)
    assert result[0].right_png_sha256 == _digest(second_png)
    assert not hasattr(result[0], "suitable_for_training")


@pytest.mark.parametrize(
    ("filter_type", "second_encoded"),
    [
        (0, b"\\x40\\x80\\xc0"),
        (1, b"\\x00\\x00\\x00"),
        (2, b"\\x40\\x80\\xc0"),
        (3, b"\\x20\\x40\\x60"),
        (4, b"\\x00\\x00\\x00"),
    ],
)
def test_shared_track_png_decoder_applies_all_five_png_filters(
    filter_type: int, second_encoded: bytes
) -> None:
    pixel = b"\\x40\\x80\\xc0"
    header = struct.pack(">IIBBBBB", 2, 1, 8, 2, 0, 0, 0)
    # These reference residuals reconstruct the second pixel to pixel
    # for every filter; the first sample for filters 1/3/4 must be its
    # literal RGB value, since the previous scanline is zero.
    if filter_type == 3:
        first_encoded = pixel
    else:
        first_encoded = pixel
    payload = bytes((filter_type,)) + first_encoded + second_encoded
    png = (
        b"\\x89PNG\\r\\n\\x1a\\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"sRGB", b"\\x00")
        + _chunk(b"IDAT", zlib.compress(payload))
        + _chunk(b"IEND", b"")
    )
    width, height, raster = baseline_module._decode_audited_gsplat_png_rgb8(png)
    assert (width, height) == (2, 1)
    assert raster == pixel + pixel


@pytest.mark.parametrize(
    ("second_xy", "reason"),
    [
        ((2.0, 0.5), "outside"),
        ((0.5, -0.1), "outside"),
        ((float("nan"), 0.5), "non-finite"),
        ((0.5, float("inf")), "non-finite"),
    ],
)
def test_shared_track_pixel_evidence_rejects_invalid_feature_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_xy: tuple[float, float],
    reason: str,
) -> None:
    request, source = _two_view_parts(tmp_path)
    features, environment = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match=reason):
        inspect_gsplat_static_appearance_shared_track_pixels(
            request, source, _matching_shared_track_photometry(source),
            features=features, expected_environment=environment,
            module=_shared_track_fixture(second_xy=second_xy),
        )


def test_shared_track_pixel_evidence_rechecks_source_byte_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request, source = _two_view_parts(tmp_path)
    assessments = _matching_shared_track_photometry(source)
    features, environment = _native_feature_evidence(tmp_path)
    monkeypatch.setattr(
        baseline_module,
        "verify_gsplat_static_appearance_native_geometry",
        lambda *_args, **_kwargs: _native_canonical(request, source),
    )
    source.images[1].source_path.write_bytes(_png(width=2, height=1, srgb=False))
    with pytest.raises(GsplatStaticAppearancePreflightError, match="size or SHA-256"):
        inspect_gsplat_static_appearance_shared_track_pixels(
            request, source, assessments,
            features=features, expected_environment=environment,
            module=_shared_track_fixture(),
        )
