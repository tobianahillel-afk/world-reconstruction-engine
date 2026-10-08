from __future__ import annotations

import hashlib
import struct
from dataclasses import FrozenInstanceError, replace

import pytest

import wre.photometry.photometric_validity as validity_module
from wre.domain import (
    LINEAR_SRGB_F64,
    ArtifactId,
    ArtifactInputFingerprint,
    ArtifactKeyMaterial,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    ColorConversionRequest,
    ConfigurationIdentity,
    DecodedImageLevelDescriptor,
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
    DecodedImagePyramidSpec,
    DecodedPixelColorEncoding,
    ImageDimensions,
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    PhotometricMaskKind,
    PhotometricNormalizationFactors,
    PhotometricNormalizationRequest,
    PhotometricValidityRequest,
    ProducerRef,
    ProvenanceClass,
    RawMetadataEntry,
    SceneProjectId,
    Sha256Digest,
    SourceColorMetadata,
    SourceExposureMetadata,
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
    SourceWhiteBalanceMetadata,
    assess_color_conversion,
    assess_photometric_normalization,
    derive_artifact_key,
)
from wre.photometry import (
    ConvertedLinearRgbBuffer,
    NormalizedLinearRgbBuffer,
    build_photometric_validity_masks,
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)


def _case(
    *,
    pixels: bytes = bytes((0, 1, 254, 255, 128, 255)),
    exposure_ev: float = 1.0,
) -> tuple[
    bytes,
    ConvertedLinearRgbBuffer,
    NormalizedLinearRgbBuffer,
    PhotometricValidityRequest,
]:
    observation_id = ObservationId("obs:photometric-validity-reference")
    color_entry = RawMetadataEntry("caller", "color", "sRGB")
    source_photometry = SourcePhotometryMetadata(
        observation_id=observation_id,
        source_metadata=ObservationMetadata(
            observation_id=observation_id,
            raw_entries=(color_entry,),
        ),
        color=SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            declared_primaries="bt709_d65",
            declared_transfer_characteristic="srgb",
            declared_matrix_coefficients="identity",
            declared_range="full",
            evidence=(color_entry,),
        ),
        exposure=SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=observation_id,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("1" * 64),
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
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:photometric-validity-reference"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="test.decoded.photometric-validity-reference",
            version="1.0.0",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("2" * 64)),
    )
    artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:photometric-validity-reference"),
        artifact_ref=artifact_ref,
        artifact_key=derive_artifact_key(
            ArtifactKeyMaterial(
                output_kind=artifact_ref.artifact_kind,
                input_fingerprints=(
                    ArtifactInputFingerprint(
                        artifact_kind=ArtifactKind("image.observation"),
                        sha256=manifest.source_asset_sha256,
                    ),
                ),
                producer=producer,
            )
        ),
        producer=producer,
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )
    materialization = ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path="levels/level-000000.rgb",
                byte_length=len(pixels),
                sha256=Sha256Digest(hashlib.sha256(pixels).hexdigest()),
            ),
        ),
    )
    color_request = ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=artifact,
        decoded_materialization=materialization,
        source_photometry=source_photometry,
        decoded_level_index=0,
        decoded_encoding=DecodedPixelColorEncoding.SRGB_FULL_RGB8,
        working_convention=LINEAR_SRGB_F64,
        output_convention=LINEAR_SRGB_F64,
    )
    color_plan = assess_color_conversion(color_request).plan
    assert color_plan is not None
    converted = convert_rgb8_to_linear_reference(color_plan, pixels)
    normalization_request = PhotometricNormalizationRequest(
        source_plan=color_plan,
        source_content_sha256=converted.content_sha256,
        source_derived_sha256=converted.derived_sha256,
        source_photometry=source_photometry,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=exposure_ev,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
    )
    normalization_plan = assess_photometric_normalization(normalization_request).plan
    assert normalization_plan is not None
    normalized = normalize_linear_rgb_reference(normalization_plan, converted)
    request = PhotometricValidityRequest(
        observation_id=observation_id,
        source_plan=color_plan,
        source_entry=color_request.decoded_level_entry,
        normalization_plan=normalization_plan,
        normalized_content_sha256=normalized.content_sha256,
        normalized_derived_sha256=normalized.derived_sha256,
        dimensions=ImageDimensions(width_px=2, height_px=1),
        channel_count=3,
    )
    return pixels, converted, normalized, request


def test_endpoint_and_normalized_range_masks_remain_separate_evidence() -> None:
    pixels, converted, normalized, request = _case()
    before_pixels = bytes(pixels)
    before_converted = bytes(converted.packed_rgb_f64_be)
    before_normalized = bytes(normalized.packed_rgb_f64_be)

    masks = build_photometric_validity_masks(request, pixels, normalized)

    assert masks.source_low_endpoint_candidate.kind is (
        PhotometricMaskKind.SOURCE_LOW_ENDPOINT_CANDIDATE
    )
    assert masks.source_low_endpoint_candidate.packed_mask == bytes((1, 0, 0, 0, 0, 0))
    assert masks.source_high_endpoint_candidate.packed_mask == bytes((0, 0, 0, 1, 0, 1))
    assert masks.normalized_above_one.packed_mask == bytes((0, 0, 1, 1, 0, 1))
    assert (
        len(
            {
                masks.source_low_endpoint_candidate.identity,
                masks.source_high_endpoint_candidate.identity,
                masks.normalized_above_one.identity,
                masks.identity,
            }
        )
        == 4
    )

    assert pixels == before_pixels
    assert converted.packed_rgb_f64_be == before_converted
    assert normalized.packed_rgb_f64_be == before_normalized
    assert "candidate" in masks.source_low_endpoint_candidate.kind.value
    assert "candidate" in masks.source_high_endpoint_candidate.kind.value
    assert "proven" not in masks.source_low_endpoint_candidate.kind.value
    assert "sensor" not in masks.source_high_endpoint_candidate.kind.value


def test_exact_one_is_not_normalized_over_range() -> None:
    pixels, _, normalized, request = _case(
        pixels=bytes((255, 128, 0, 64, 255, 1)),
        exposure_ev=0.0,
    )
    values = tuple(value for (value,) in struct.iter_unpack("!d", normalized.packed_rgb_f64_be))
    assert values[0] == 1.0
    assert values[4] == 1.0

    masks = build_photometric_validity_masks(request, pixels, normalized)
    assert masks.normalized_above_one.packed_mask == bytes((0, 0, 0, 0, 0, 0))
    assert masks.source_high_endpoint_candidate.packed_mask == bytes((1, 0, 0, 0, 1, 0))


def test_reference_rejects_foreign_source_and_normalized_identities() -> None:
    pixels, _, normalized, request = _case()

    with pytest.raises(ValueError, match="byte length"):
        build_photometric_validity_masks(request, pixels[:-1], normalized)
    with pytest.raises(ValueError, match="digest"):
        build_photometric_validity_masks(request, bytes(reversed(pixels)), normalized)
    with pytest.raises(TypeError, match="immutable bytes"):
        build_photometric_validity_masks(request, bytearray(pixels), normalized)  # type: ignore[arg-type]

    wrong_content = replace(request, normalized_content_sha256=Sha256Digest("3" * 64))
    with pytest.raises(ValueError, match="content digest"):
        build_photometric_validity_masks(wrong_content, pixels, normalized)

    wrong_derived = replace(request, normalized_derived_sha256=Sha256Digest("4" * 64))
    with pytest.raises(ValueError, match="derived identity"):
        build_photometric_validity_masks(wrong_derived, pixels, normalized)

    foreign_pixels, _, foreign_normalized, _ = _case(
        pixels=bytes((1, 2, 3, 4, 5, 6)),
        exposure_ev=0.0,
    )
    assert foreign_pixels != pixels
    with pytest.raises(ValueError, match="foreign normalization plan"):
        build_photometric_validity_masks(request, pixels, foreign_normalized)


def test_masks_are_immutable_and_identity_changes_with_mask_content() -> None:
    pixels, _, normalized, request = _case()
    masks = build_photometric_validity_masks(request, pixels, normalized)
    low = masks.source_low_endpoint_candidate

    with pytest.raises(FrozenInstanceError):
        low.packed_mask = b""  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        masks.identity = Sha256Digest("0" * 64)  # type: ignore[misc]

    changed_low = replace(low, packed_mask=bytes((0, 0, 0, 0, 0, 0)))
    assert changed_low.content_sha256 != low.content_sha256
    assert changed_low.identity != low.identity

    with pytest.raises(ValueError, match="canonical 0 or 1"):
        replace(low, packed_mask=bytes((2, 0, 0, 0, 0, 0)))


def test_reference_surface_contains_no_repair_or_appearance_policy() -> None:
    assert not hasattr(validity_module, "recover_highlights")
    assert not hasattr(validity_module, "repair_saturation")
    assert not hasattr(validity_module, "tone_map")
    assert not hasattr(validity_module, "clip")
    assert not hasattr(validity_module, "rescale")
    assert not hasattr(validity_module, "infer_camera_response")
    assert not hasattr(validity_module, "AppearanceModel")
    assert not hasattr(validity_module, "QualityDecision")
    assert not hasattr(validity_module, "route_observation")
