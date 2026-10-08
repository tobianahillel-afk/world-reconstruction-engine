from __future__ import annotations

import ast
import hashlib
import struct
import zlib
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import wre.reconstruction.static_appearance_baseline as baseline_module
from wre.domain.appearance import APPEARANCE_MODEL_ARTIFACT_KIND, AppearanceRepresentationName
from wre.domain import (
    LINEAR_SRGB_F64,
    ArtifactInputFingerprint,
    ArtifactKeyMaterial,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMetadata,
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
    ColorConversionRequest,
    assess_color_conversion,
    assess_photometric_compatibility,
    assess_photometric_normalization,
    derive_artifact_key,
)
from wre.photometry import (
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)
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
    GSPLAT_STATIC_APPEARANCE_REPRESENTATION,
    GSPLAT_STATIC_APPEARANCE_SOURCE_REVISION,
    GsplatStaticAppearancePreflightError,
    GsplatStaticAppearancePreflightSource,
    GsplatStaticAppearanceTrainingProfile,
    materialize_verified_gsplat_ply,
    preflight_gsplat_static_appearance_inputs,
    verify_gsplat_static_appearance_native_geometry,
    verify_gsplat_static_appearance_source_photometry,
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
        "math",
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


def _gsplat_ply(*, properties: tuple[str, ...] | None = None, vertices: int = 1) -> bytes:
    """Exact uncompressed gsplat exporter layout with finite float32 content."""
    standard = (
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "f_rest_0",
        "f_rest_1",
        "f_rest_2",
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
) -> PhotometricCompatibilityAssessment:
    """Real V2L17 plans bound to the same source asset, not mocked compatible labels."""

    image = source.images[0]
    observation = observation_id or image.observation.observation_id
    source_digest = asset_sha256 or image.observation.asset.sha256
    rgb = bytes((64, 128, 192, 64, 128, 192))
    declaration = RawMetadataEntry("caller", "color", "sRGB")
    photo = SourcePhotometryMetadata(
        observation_id=observation,
        source_metadata=ObservationMetadata(
            observation_id=observation,
            raw_entries=(declaration,),
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
        exposure=SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
        ),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
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
    assert verify_gsplat_static_appearance_source_photometry(
        request, source, (assessment,)
    ) == (assessment.identity,)
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
            request, source, ("compatible",)  # type: ignore[arg-type]
        )
    assessment = _gsplat_photo_assessment(source)
    false_label = PhotometricCompatibilityAssessment(
        status=PhotometricCompatibilityStatus.UNRESOLVED,
        compatibility_input=assessment.compatibility_input,
        reasons=("unverified",),
    )
    with pytest.raises(GsplatStaticAppearancePreflightError, match="unresolved"):
        verify_gsplat_static_appearance_source_photometry(
            request, source, (false_label,)
        )
