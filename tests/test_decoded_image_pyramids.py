from __future__ import annotations

import os
import shutil
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

import wre.ingestion.decoded_images as decoded_images_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactMaterializationVerificationStatus,
    ArtifactRef,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    ImageObservation,
    MediaAssetRef,
    ObservationId,
    ObservationKind,
    Sha256Digest,
    SourceId,
    SourceRef,
    VideoFrameObservation,
    build_decoded_image_level_descriptors,
)
from wre.ingestion import (
    SUPPORTED_FFMPEG_VERSION,
    DecodedImagePyramidMaterializationRequest,
    FFmpegDecodedImagePyramidMaterializer,
    FFmpegToolchain,
    FFmpegToolchainIdentity,
    hash_file_content,
    inspect_toolchain,
)
from wre.materialization import verify_local_artifact_materialization

NOW = datetime(2026, 9, 18, 17, 30, tzinfo=UTC)


def _asset(path: Path) -> MediaAssetRef:
    content = hash_file_content(path)
    return MediaAssetRef(
        uri=path.resolve().as_uri(),
        sha256=content.sha256,
        byte_length=content.byte_length,
        mime_type="image/x-portable-pixmap",
    )


def _source(path: Path, *, video_frame: bool = False) -> ImageObservation | VideoFrameObservation:
    common = dict(
        observation_id=ObservationId("obs:decoded-pyramid"),
        asset=_asset(path),
        source=SourceRef(source_id=SourceId("source:decoded-pyramid")),
        received_at=NOW,
        captured_at=NOW,
    )
    if not video_frame:
        return ImageObservation(**common)
    return VideoFrameObservation(
        **common,
        video_asset=MediaAssetRef(
            uri="file:///parent-video.mkv",
            sha256=Sha256Digest("f" * 64),
            byte_length=123,
            mime_type="video/x-matroska",
        ),
        frame_index=7,
        frame_time_us=1_750_000,
    )


def _ref() -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("artifact:decoded-pyramid"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )


def _write_ppm(path: Path, *, width: int, height: int) -> None:
    header = f"P6\n{width} {height}\n255\n".encode()
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            pixels.extend(((x * 31) % 256, (y * 47) % 256, ((x + y) * 19) % 256))
    path.write_bytes(header + bytes(pixels))


def _require_pinned_ffmpeg() -> FFmpegToolchain:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail("GitHub CI must provide ffmpeg and ffprobe")
        pytest.skip("ffmpeg/ffprobe are not installed")
    toolchain = FFmpegToolchain(
        ffmpeg_path=ffmpeg,
        ffprobe_path=ffprobe,
        timeout_seconds=30,
    )
    inspect_toolchain(toolchain)
    return toolchain


def test_pyramid_spec_is_exact_immutable_hashable_positive_integer() -> None:
    spec = DecodedImagePyramidSpec(minimum_max_edge_px=512)

    assert tuple(field.name for field in fields(DecodedImagePyramidSpec)) == (
        "minimum_max_edge_px",
    )
    assert hash(spec) == hash(DecodedImagePyramidSpec(minimum_max_edge_px=512))
    with pytest.raises(FrozenInstanceError):
        spec.minimum_max_edge_px = 256  # type: ignore[misc]

    for value in (0, -1, True, 1.5, "512"):
        with pytest.raises(ValueError, match="positive integer"):
            DecodedImagePyramidSpec(minimum_max_edge_px=cast(Any, value))


def test_level_descriptor_requires_canonical_path_and_positive_dimensions() -> None:
    level = DecodedImageLevelDescriptor(
        level_index=2,
        width_px=10,
        height_px=5,
        relative_path="levels/level-000002.rgb",
    )
    assert hash(level) == hash(level)

    with pytest.raises(ValueError, match="canonical level path"):
        DecodedImageLevelDescriptor(
            level_index=2,
            width_px=10,
            height_px=5,
            relative_path="../level.rgb",
        )
    with pytest.raises(ValueError, match="positive integer"):
        DecodedImageLevelDescriptor(
            level_index=0,
            width_px=0,
            height_px=5,
            relative_path="levels/level-000000.rgb",
        )


@pytest.mark.parametrize(
    ("width", "height", "minimum", "expected"),
    [
        (8, 6, 2, ((8, 6), (4, 3), (2, 1))),
        (7, 5, 2, ((7, 5), (3, 2), (1, 1))),
        (5, 9, 3, ((5, 9), (2, 4), (1, 2))),
        (1, 1, 1, ((1, 1),)),
        (4, 3, 8, ((4, 3),)),
    ],
)
def test_pyramid_dimensions_are_deterministic_floor_halves(
    width: int,
    height: int,
    minimum: int,
    expected: tuple[tuple[int, int], ...],
) -> None:
    levels = build_decoded_image_level_descriptors(
        width,
        height,
        DecodedImagePyramidSpec(minimum_max_edge_px=minimum),
    )
    assert tuple((level.width_px, level.height_px) for level in levels) == expected
    assert tuple(level.level_index for level in levels) == tuple(range(len(expected)))


def test_manifest_is_typed_and_rejects_video_or_noncanonical_levels() -> None:
    spec = DecodedImagePyramidSpec(minimum_max_edge_px=2)
    levels = build_decoded_image_level_descriptors(7, 5, spec)
    manifest = DecodedImagePyramidManifest(
        source_observation_id=ObservationId("obs:image"),
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("a" * 64),
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=spec,
        levels=levels,
    )
    assert manifest.levels == levels

    with pytest.raises(TypeError, match="source_kind must be ObservationKind"):
        DecodedImagePyramidManifest(
            source_observation_id=ObservationId("obs:string-kind"),
            source_kind=cast(Any, "image"),
            source_asset_sha256=Sha256Digest("a" * 64),
            pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
            orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
            spec=spec,
            levels=levels,
        )

    with pytest.raises(ValueError, match="image or video_frame"):
        DecodedImagePyramidManifest(
            source_observation_id=ObservationId("obs:video"),
            source_kind=ObservationKind.VIDEO,
            source_asset_sha256=Sha256Digest("a" * 64),
            pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
            orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
            spec=spec,
            levels=levels,
        )

    broken = (
        levels[0],
        DecodedImageLevelDescriptor(
            level_index=1,
            width_px=2,
            height_px=2,
            relative_path="levels/level-000001.rgb",
        ),
    )
    with pytest.raises(ValueError, match="floor halving"):
        DecodedImagePyramidManifest(
            source_observation_id=ObservationId("obs:image"),
            source_kind=ObservationKind.IMAGE,
            source_asset_sha256=Sha256Digest("a" * 64),
            pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
            orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
            spec=spec,
            levels=broken,
        )


def test_reference_identity_is_content_config_and_producer_sensitive(tmp_path: Path) -> None:
    first_path = tmp_path / "first.ppm"
    second_path = tmp_path / "second.ppm"
    _write_ppm(first_path, width=7, height=5)
    _write_ppm(second_path, width=8, height=5)
    first = _source(first_path)
    second = _source(second_path)
    spec = DecodedImagePyramidSpec(minimum_max_edge_px=2)
    identity = FFmpegToolchainIdentity(
        ffmpeg_version=SUPPORTED_FFMPEG_VERSION,
        ffprobe_version=SUPPORTED_FFMPEG_VERSION,
    )

    first_producer, first_key = decoded_images_module._reference_artifact_identity(
        first,
        spec,
        identity,
    )
    same_producer, same_key = decoded_images_module._reference_artifact_identity(
        first,
        spec,
        identity,
    )
    _, source_changed = decoded_images_module._reference_artifact_identity(
        second,
        spec,
        identity,
    )
    _, config_changed = decoded_images_module._reference_artifact_identity(
        first,
        DecodedImagePyramidSpec(minimum_max_edge_px=1),
        identity,
    )
    other_identity = FFmpegToolchainIdentity(
        ffmpeg_version="8.0.1-test",
        ffprobe_version="8.0.1-test",
    )
    _, producer_changed = decoded_images_module._reference_artifact_identity(
        first,
        spec,
        other_identity,
    )

    assert first_producer == same_producer
    assert first_key == same_key
    assert source_changed != first_key
    assert config_changed != first_key
    assert producer_changed != first_key


def test_source_mismatch_fails_before_external_tool_execution(tmp_path: Path) -> None:
    source_path = tmp_path / "source.ppm"
    _write_ppm(source_path, width=4, height=4)
    source = _source(source_path)
    source_path.write_bytes(source_path.read_bytes() + b"changed")

    materializer = FFmpegDecodedImagePyramidMaterializer(
        FFmpegToolchain(ffmpeg_path="must-not-run", ffprobe_path="must-not-run")
    )
    request = DecodedImagePyramidMaterializationRequest(
        source=source,
        source_path=source_path,
        output_root=tmp_path / "out",
        artifact_ref=_ref(),
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
    )

    with pytest.raises(ValueError, match="source bytes do not match"):
        materializer.materialize(request)


def test_reference_materialization_is_verified_and_idempotent(tmp_path: Path) -> None:
    toolchain = _require_pinned_ffmpeg()
    source_path = tmp_path / "source.ppm"
    _write_ppm(source_path, width=7, height=5)
    source = _source(source_path)
    request = DecodedImagePyramidMaterializationRequest(
        source=source,
        source_path=source_path,
        output_root=tmp_path / "out",
        artifact_ref=_ref(),
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
    )
    materializer = FFmpegDecodedImagePyramidMaterializer(toolchain)

    first = materializer.materialize(request)
    first_bytes = {
        entry.relative_path: (request.output_root / entry.relative_path).read_bytes()
        for entry in first.materialization.entries
    }
    second = materializer.materialize(request)

    assert first == second
    assert first.manifest.source_observation_id == source.observation_id
    assert first.manifest.source_kind is ObservationKind.IMAGE
    assert first.manifest.source_asset_sha256 == source.asset.sha256
    assert first.manifest.pixel_layout is DecodedImagePixelLayout.RGB8_PACKED
    assert first.manifest.orientation_policy is DecodedImageOrientationPolicy.SOURCE_PIXELS
    assert tuple((level.width_px, level.height_px) for level in first.manifest.levels) == (
        (7, 5),
        (3, 2),
        (1, 1),
    )

    for level, entry in zip(
        first.manifest.levels,
        first.materialization.entries,
        strict=True,
    ):
        assert entry.relative_path == level.relative_path
        assert entry.byte_length == level.width_px * level.height_px * 3
        path = request.output_root / entry.relative_path
        assert path.read_bytes() == first_bytes[entry.relative_path]
        content = hash_file_content(path)
        assert content.sha256 == entry.sha256
        assert content.byte_length == entry.byte_length

    verification = verify_local_artifact_materialization(
        first.materialization,
        request.output_root,
    )
    assert verification.status is ArtifactMaterializationVerificationStatus.VERIFIED


def test_video_frame_observation_uses_same_contract_without_parent_video_selection(
    tmp_path: Path,
) -> None:
    toolchain = _require_pinned_ffmpeg()
    source_path = tmp_path / "frame.ppm"
    _write_ppm(source_path, width=4, height=3)
    source = _source(source_path, video_frame=True)
    result = FFmpegDecodedImagePyramidMaterializer(toolchain).materialize(
        DecodedImagePyramidMaterializationRequest(
            source=source,
            source_path=source_path,
            output_root=tmp_path / "frame-out",
            artifact_ref=_ref(),
            spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
        )
    )

    assert result.manifest.source_kind is ObservationKind.VIDEO_FRAME
    assert result.manifest.source_observation_id == source.observation_id
    assert tuple((level.width_px, level.height_px) for level in result.manifest.levels) == (
        (4, 3),
        (2, 1),
    )


def test_reference_surface_has_no_accelerated_or_future_pipeline_dependencies() -> None:
    for name in (
        "nvImageCodec",
        "nvJPEG",
        "NVDEC",
        "CUDA",
        "torch",
        "numpy",
        "Pillow",
        "OpenCV",
        "PyAV",
        "libvips",
        "PairCandidate",
        "QualityDecision",
        "JobSpec",
    ):
        assert name not in decoded_images_module.__dict__
