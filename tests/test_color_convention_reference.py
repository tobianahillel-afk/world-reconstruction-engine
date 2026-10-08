from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

import wre.photometry.color_conventions as reference_module
from wre.domain import (
    LINEAR_SRGB_F64,
    ArtifactMaterializationEntry,
    ColorConversionStatus,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    DecodedPixelColorEncoding,
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    RawMetadataEntry,
    Sha256Digest,
    SourceColorMetadata,
    SourceExposureMetadata,
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
    SourceWhiteBalanceMetadata,
    assess_color_conversion,
)
from wre.domain.color_conventions import ColorConversionRequest
from wre.photometry import convert_rgb8_to_linear_reference


def _reference_case() -> tuple[ColorConversionRequest, bytes]:
    pixels = bytes([0, 10, 128, 255, 64, 32])
    obs = ObservationId("obs:numeric-srgb")
    metadata_entry = RawMetadataEntry("caller", "color", "sRGB")
    photometry = SourcePhotometryMetadata(
        observation_id=obs,
        source_metadata=ObservationMetadata(observation_id=obs, raw_entries=(metadata_entry,)),
        color=SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            declared_primaries="bt709_d65",
            declared_transfer_characteristic="srgb",
            declared_matrix_coefficients="identity",
            declared_range="full",
            evidence=(metadata_entry,),
        ),
        exposure=SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=obs,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("e" * 64),
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0, width_px=2, height_px=1, relative_path="levels/level-000000.rgb"
            ),
        ),
    )
    return (
        ColorConversionRequest(
            decoded_manifest=manifest,
            source_photometry=photometry,
            decoded_level_index=0,
            decoded_level_entry=ArtifactMaterializationEntry(
                relative_path="levels/level-000000.rgb",
                byte_length=len(pixels),
                sha256=Sha256Digest(hashlib.sha256(pixels).hexdigest()),
            ),
            decoded_encoding=DecodedPixelColorEncoding.SRGB_FULL_RGB8,
            working_convention=LINEAR_SRGB_F64,
            output_convention=LINEAR_SRGB_F64,
        ),
        pixels,
    )


def test_independent_fixed_iec_srgb_numeric_reference_fixture() -> None:
    request, pixels = _reference_case()
    assessment = assess_color_conversion(request)
    assert assessment.status is ColorConversionStatus.READY
    assert assessment.plan is not None
    before = bytes(pixels)
    result = convert_rgb8_to_linear_reference(assessment.plan, pixels)

    # Fixed independent sRGB inverse-transfer reference values, not another call to the backend.
    expected = (
        0.0,
        0.003035269835488375,
        0.21586050011389926,
        1.0,
        0.05126945837404324,
        0.014443843596092545,
    )
    assert result.channels == pytest.approx(expected, abs=1e-13)
    assert pixels == before
    assert request.decoded_level_entry.sha256.value == hashlib.sha256(before).hexdigest()
    assert result.derived_sha256 != request.decoded_level_entry.sha256
    assert result.source_level_sha256 == request.decoded_level_entry.sha256
    assert result.plan.request.source_photometry is request.source_photometry


def test_reference_rejects_mismatched_bytes_and_does_not_write_source() -> None:
    request, pixels = _reference_case()
    plan = assess_color_conversion(request).plan
    assert plan is not None
    with pytest.raises(ValueError, match="byte length"):
        convert_rgb8_to_linear_reference(plan, pixels[:-1])
    with pytest.raises(ValueError, match="digest"):
        convert_rgb8_to_linear_reference(plan, bytes(reversed(pixels)))
    with pytest.raises(TypeError, match="immutable bytes"):
        convert_rgb8_to_linear_reference(plan, bytearray(pixels))  # type: ignore[arg-type]

    converted = convert_rgb8_to_linear_reference(plan, pixels)
    with pytest.raises(FrozenInstanceError):
        converted.channels = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="content digest"):
        replace(converted, content_sha256=Sha256Digest("0" * 64))


def test_backend_has_no_implicit_image_decode_or_photometry_normalization() -> None:
    source = reference_module.__file__
    assert source is not None
    assert not hasattr(reference_module, "normalize_exposure")
    assert not hasattr(reference_module, "normalize_white_balance")
    assert not hasattr(reference_module, "decode_image")
    assert not hasattr(reference_module, "convert_icc")
