from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

import wre.photometry.photometric_normalization as normalization_module
from wre.domain import (
    LINEAR_SRGB_F64,
    ArtifactMaterializationEntry,
    ColorConversionRequest,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    DecodedPixelColorEncoding,
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    PhotometricNormalizationFactors,
    PhotometricNormalizationRequest,
    PhotometricNormalizationStatus,
    RawMetadataEntry,
    Sha256Digest,
    SourceColorMetadata,
    SourceExposureMetadata,
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
    SourceWhiteBalanceMetadata,
    assess_color_conversion,
    assess_photometric_normalization,
)
from wre.photometry import (
    ConvertedLinearRgbBuffer,
    NormalizedLinearRgbBuffer,
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)


def _source_case() -> tuple[ConvertedLinearRgbBuffer, SourcePhotometryMetadata]:
    pixels = bytes((64, 128, 255))
    obs = ObservationId("obs:photometric-reference")
    color_entry = RawMetadataEntry("caller", "color", "sRGB")
    iso_entry = RawMetadataEntry("exif", "iso", "100")
    wb_entry = RawMetadataEntry("exif", "white_balance", "manual")
    entries = tuple(
        sorted(
            (color_entry, iso_entry, wb_entry),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    metadata = ObservationMetadata(observation_id=obs, raw_entries=entries)
    photometry = SourcePhotometryMetadata(
        observation_id=obs,
        source_metadata=metadata,
        color=SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            declared_primaries="bt709_d65",
            declared_transfer_characteristic="srgb",
            declared_matrix_coefficients="identity",
            declared_range="full",
            evidence=(color_entry,),
        ),
        exposure=SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            iso_speed=100.0,
            evidence=(iso_entry,),
        ),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            mode="manual",
            evidence=(wb_entry,),
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=obs,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("d" * 64),
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=1),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0,
                width_px=1,
                height_px=1,
                relative_path="levels/level-000000.rgb",
            ),
        ),
    )
    request = ColorConversionRequest(
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
    )
    plan = assess_color_conversion(request).plan
    assert plan is not None
    return convert_rgb8_to_linear_reference(plan, pixels), photometry


def _normalization_case() -> tuple[
    ConvertedLinearRgbBuffer,
    PhotometricNormalizationRequest,
]:
    source, photometry = _source_case()
    request = PhotometricNormalizationRequest(
        source_plan=source.plan,
        source_content_sha256=source.content_sha256,
        source_derived_sha256=source.derived_sha256,
        source_photometry=photometry,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=1.0,
            white_balance_rgb_gains=(1.0, 0.5, 2.0),
        ),
    )
    return source, request


def test_explicit_linear_reference_preserves_values_above_one_without_clipping() -> None:
    source, request = _normalization_case()
    assessment = assess_photometric_normalization(request)
    assert assessment.status is PhotometricNormalizationStatus.READY
    assert assessment.plan is not None

    before = tuple(source.channels)
    result = normalize_linear_rgb_reference(assessment.plan, source)

    expected = (
        0.10253891674808648,
        0.21586050011389926,
        4.0,
    )
    assert result.channels == pytest.approx(expected, abs=1e-13)
    assert result.channels[2] > 1.0
    assert source.channels == before
    assert result.source_content_sha256 == source.content_sha256
    assert result.source_derived_sha256 == source.derived_sha256
    assert result.derived_sha256 != source.derived_sha256
    assert result.plan.request.source_photometry is request.source_photometry


def test_reference_rejects_foreign_source_digests_and_plan() -> None:
    source, request = _normalization_case()

    wrong_content_request = replace(request, source_content_sha256=Sha256Digest("a" * 64))
    wrong_content_plan = assess_photometric_normalization(wrong_content_request).plan
    assert wrong_content_plan is not None
    with pytest.raises(ValueError, match="content digest"):
        normalize_linear_rgb_reference(wrong_content_plan, source)

    wrong_derived_request = replace(request, source_derived_sha256=Sha256Digest("b" * 64))
    wrong_derived_plan = assess_photometric_normalization(wrong_derived_request).plan
    assert wrong_derived_plan is not None
    with pytest.raises(ValueError, match="derived digest"):
        normalize_linear_rgb_reference(wrong_derived_plan, source)

    other_source, _ = _source_case()
    other_plan_request = replace(
        request,
        source_plan=other_source.plan,
        source_content_sha256=other_source.content_sha256,
        source_derived_sha256=other_source.derived_sha256,
    )
    other_plan = assess_photometric_normalization(other_plan_request).plan
    assert other_plan is not None
    if other_source.plan != source.plan:
        with pytest.raises(ValueError, match="color-conversion plan"):
            normalize_linear_rgb_reference(other_plan, source)


def test_normalized_result_is_immutable_and_digest_verified() -> None:
    source, request = _normalization_case()
    plan = assess_photometric_normalization(request).plan
    assert plan is not None
    result = normalize_linear_rgb_reference(plan, source)

    assert isinstance(result, NormalizedLinearRgbBuffer)
    with pytest.raises(FrozenInstanceError):
        result.channels = ()  # type: ignore[misc]
    with pytest.raises(ValueError, match="content digest"):
        replace(result, content_sha256=Sha256Digest("0" * 64))
    with pytest.raises(ValueError, match="derived identity"):
        replace(result, derived_sha256=Sha256Digest("0" * 64))


def test_reference_surface_contains_no_automatic_photometric_behavior() -> None:
    assert not hasattr(normalization_module, "auto_exposure")
    assert not hasattr(normalization_module, "auto_white_balance")
    assert not hasattr(normalization_module, "estimate_exposure")
    assert not hasattr(normalization_module, "estimate_white_balance")
    assert not hasattr(normalization_module, "chromatic_adaptation")
    assert not hasattr(normalization_module, "tone_map")
    assert not hasattr(normalization_module, "clip")
    assert not hasattr(normalization_module, "saturation_mask")
