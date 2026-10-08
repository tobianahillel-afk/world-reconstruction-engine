from __future__ import annotations

import ast
import hashlib
import struct
import zlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wre.domain.appearance import AppearanceRepresentationName
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
from wre.domain.runs import ProducerRef
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
    GSPLAT_STATIC_APPEARANCE_REPRESENTATION,
    GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION,
    GsplatStaticAppearancePreflightError,
    GsplatStaticAppearancePreflightSource,
    GsplatStaticAppearanceTrainingProfile,
    preflight_gsplat_static_appearance_inputs,
)
from wre.reconstruction.static_appearance_candidate import StaticAppearanceCandidateRequest


def _digest(data: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(data).hexdigest())


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
        "pathlib",
        "struct",
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
