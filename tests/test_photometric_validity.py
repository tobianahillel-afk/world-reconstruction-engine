from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast

import pytest

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
    ColorConvention,
    ColorConversionRequest,
    ColorSampleStorage,
    ColorTransfer,
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
    PhotometricChannelMask,
    PhotometricCompatibilityInput,
    PhotometricCompatibilityStatus,
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
    assess_photometric_compatibility,
    assess_photometric_normalization,
    derive_artifact_key,
)
from wre.photometry import (
    ConvertedLinearRgbBuffer,
    NormalizedLinearRgbBuffer,
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)


PIXELS = bytes((0, 1, 254, 255, 128, 255))


def _source_case(
    *,
    observation_value: str = "obs:photometric-validity",
    pixels: bytes = PIXELS,
) -> tuple[
    ColorConversionRequest,
    ConvertedLinearRgbBuffer,
    SourcePhotometryMetadata,
]:
    obs = ObservationId(observation_value)
    color_entry = RawMetadataEntry("caller", "color", "sRGB")
    metadata = ObservationMetadata(observation_id=obs, raw_entries=(color_entry,))
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
        exposure=SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=obs,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("a" * 64),
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
        artifact_id=ArtifactId(f"artifact:{observation_value.split(':')[-1]}-decoded"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(implementation="test.decoded.photometric-validity", version="1.0.0"),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("b" * 64)),
    )
    artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:photometric-validity"),
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
    request = ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=artifact,
        decoded_materialization=materialization,
        source_photometry=photometry,
        decoded_level_index=0,
        decoded_encoding=DecodedPixelColorEncoding.SRGB_FULL_RGB8,
        working_convention=LINEAR_SRGB_F64,
        output_convention=LINEAR_SRGB_F64,
    )
    color_plan = assess_color_conversion(request).plan
    assert color_plan is not None
    return request, convert_rgb8_to_linear_reference(color_plan, pixels), photometry


def _ready_case(
    *,
    observation_value: str = "obs:photometric-validity",
    pixels: bytes = PIXELS,
    exposure_ev: float = 1.0,
) -> tuple[
    ColorConversionRequest,
    ConvertedLinearRgbBuffer,
    NormalizedLinearRgbBuffer,
    PhotometricValidityRequest,
]:
    color_request, converted, photometry = _source_case(
        observation_value=observation_value,
        pixels=pixels,
    )
    normalization_request = PhotometricNormalizationRequest(
        source_plan=converted.plan,
        source_content_sha256=converted.content_sha256,
        source_derived_sha256=converted.derived_sha256,
        source_photometry=photometry,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=exposure_ev,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
    )
    normalization_plan = assess_photometric_normalization(normalization_request).plan
    assert normalization_plan is not None
    normalized = normalize_linear_rgb_reference(normalization_plan, converted)
    level = color_request.decoded_manifest.levels[color_request.decoded_level_index]
    validity_request = PhotometricValidityRequest(
        observation_id=color_request.decoded_manifest.source_observation_id,
        source_plan=converted.plan,
        source_entry=color_request.decoded_level_entry,
        normalization_plan=normalization_plan,
        normalized_content_sha256=normalized.content_sha256,
        normalized_derived_sha256=normalized.derived_sha256,
        dimensions=ImageDimensions(width_px=level.width_px, height_px=level.height_px),
        channel_count=3,
    )
    return color_request, converted, normalized, validity_request


def test_contract_shapes_and_status_vocabularies_are_exact() -> None:
    assert tuple(item.value for item in PhotometricMaskKind) == (
        "source_low_endpoint_candidate",
        "source_high_endpoint_candidate",
        "normalized_above_one",
    )
    assert tuple(item.value for item in PhotometricCompatibilityStatus) == (
        "compatible",
        "unresolved",
        "incompatible",
    )
    assert [field.name for field in fields(PhotometricValidityRequest)] == [
        "observation_id",
        "source_plan",
        "source_entry",
        "normalization_plan",
        "normalized_content_sha256",
        "normalized_derived_sha256",
        "dimensions",
        "channel_count",
    ]
    assert [field.name for field in fields(PhotometricChannelMask)] == [
        "request",
        "kind",
        "packed_mask",
        "content_sha256",
        "identity",
    ]
    assert [field.name for field in fields(PhotometricCompatibilityInput)] == [
        "color_assessment",
        "normalization_assessment",
        "normalized_content_sha256",
        "normalized_derived_sha256",
    ]


def test_request_rejects_foreign_source_entry_observation_dimensions_and_plan() -> None:
    _, _, _, request = _ready_case()

    with pytest.raises(ValueError, match="source observation"):
        replace(request, observation_id=ObservationId("obs:foreign"))

    with pytest.raises(ValueError, match="source path byte length or digest"):
        replace(
            request,
            source_entry=replace(request.source_entry, relative_path="levels/foreign.rgb"),
        )
    with pytest.raises(ValueError, match="source path byte length or digest"):
        replace(request, source_entry=replace(request.source_entry, byte_length=3))
    with pytest.raises(ValueError, match="source path byte length or digest"):
        replace(
            request,
            source_entry=replace(request.source_entry, sha256=Sha256Digest("c" * 64)),
        )
    with pytest.raises(ValueError, match="dimensions"):
        replace(request, dimensions=ImageDimensions(width_px=1, height_px=2))
    with pytest.raises(ValueError, match="channel_count"):
        replace(request, channel_count=4)

    _, _, foreign_normalized, foreign_request = _ready_case(
        observation_value="obs:foreign-validity",
        pixels=bytes((1, 2, 3, 4, 5, 6)),
    )
    assert foreign_normalized.plan == foreign_request.normalization_plan
    with pytest.raises(ValueError, match="foreign source plan"):
        replace(request, normalization_plan=foreign_request.normalization_plan)


def test_request_and_compatibility_types_fail_closed() -> None:
    _, _, _, request = _ready_case()

    with pytest.raises(TypeError, match="channel_count"):
        replace(request, channel_count=cast(Any, True))
    with pytest.raises(TypeError, match="source_entry"):
        replace(request, source_entry=cast(Any, "levels/level-000000.rgb"))

    color_assessment = assess_color_conversion(request.source_plan.request)
    normalization_assessment = assess_photometric_normalization(
        request.normalization_plan.request
    )
    with pytest.raises(TypeError, match="color_assessment"):
        PhotometricCompatibilityInput(
            color_assessment=cast(Any, "ready"),
            normalization_assessment=normalization_assessment,
        )
    with pytest.raises(ValueError, match="present together"):
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=normalization_assessment,
            normalized_content_sha256=request.normalized_content_sha256,
        )


def test_compatibility_is_explicit_for_ready_unresolved_rejected_and_foreign_evidence() -> None:
    color_request, converted, normalized, request = _ready_case()
    color_assessment = assess_color_conversion(color_request)
    ready_normalization = assess_photometric_normalization(request.normalization_plan.request)

    compatible = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=ready_normalization,
            normalized_content_sha256=normalized.content_sha256,
            normalized_derived_sha256=normalized.derived_sha256,
        )
    )
    assert compatible.status is PhotometricCompatibilityStatus.COMPATIBLE
    assert compatible.reasons == ()

    unresolved_request = replace(
        request.normalization_plan.request,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=None,
            white_balance_rgb_gains=None,
        ),
    )
    unresolved = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=assess_photometric_normalization(unresolved_request),
        )
    )
    assert unresolved.status is PhotometricCompatibilityStatus.UNRESOLVED
    assert "unresolved" in unresolved.reasons[0]

    rejected_request = replace(
        request.normalization_plan.request,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=math.nan,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
    )
    rejected = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=assess_photometric_normalization(rejected_request),
        )
    )
    assert rejected.status is PhotometricCompatibilityStatus.INCOMPATIBLE
    assert any("rejected" in reason for reason in rejected.reasons)

    _, _, foreign_normalized, foreign_request = _ready_case(
        observation_value="obs:foreign-compatibility",
        pixels=bytes((2, 3, 4, 5, 6, 7)),
    )
    foreign = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=assess_photometric_normalization(
                foreign_request.normalization_plan.request
            ),
            normalized_content_sha256=foreign_normalized.content_sha256,
            normalized_derived_sha256=foreign_normalized.derived_sha256,
        )
    )
    assert foreign.status is PhotometricCompatibilityStatus.INCOMPATIBLE
    assert any("different observations" in reason for reason in foreign.reasons)

    unsupported_convention = ColorConvention(
        primaries=LINEAR_SRGB_F64.primaries,
        transfer=ColorTransfer.SRGB,
        storage=ColorSampleStorage.FLOAT64_RGB,
    )
    unsupported_color_request = replace(
        color_request,
        working_convention=unsupported_convention,
        output_convention=unsupported_convention,
    )
    unsupported_color = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=assess_color_conversion(unsupported_color_request),
            normalization_assessment=ready_normalization,
            normalized_content_sha256=normalized.content_sha256,
            normalized_derived_sha256=normalized.derived_sha256,
        )
    )
    assert unsupported_color.status is PhotometricCompatibilityStatus.INCOMPATIBLE

    assert converted.plan == request.source_plan


def test_mask_and_compatibility_identities_follow_complete_lineage() -> None:
    color_request, _, normalized, request = _ready_case()
    mask = PhotometricChannelMask(
        request=request,
        kind=PhotometricMaskKind.SOURCE_HIGH_ENDPOINT_CANDIDATE,
        packed_mask=bytes((0, 0, 0, 1, 0, 1)),
    )
    changed_mask = replace(mask, packed_mask=bytes((0, 0, 0, 1, 0, 0)))
    assert changed_mask.content_sha256 != mask.content_sha256
    assert changed_mask.identity != mask.identity

    changed_request = replace(
        request,
        normalized_content_sha256=Sha256Digest("d" * 64),
    )
    changed_lineage_mask = replace(mask, request=changed_request)
    assert changed_lineage_mask.identity != mask.identity

    color_assessment = assess_color_conversion(color_request)
    normalization_assessment = assess_photometric_normalization(
        request.normalization_plan.request
    )
    first = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=normalization_assessment,
            normalized_content_sha256=normalized.content_sha256,
            normalized_derived_sha256=normalized.derived_sha256,
        )
    )
    changed = assess_photometric_compatibility(
        PhotometricCompatibilityInput(
            color_assessment=color_assessment,
            normalization_assessment=normalization_assessment,
            normalized_content_sha256=Sha256Digest("e" * 64),
            normalized_derived_sha256=normalized.derived_sha256,
        )
    )
    assert first.identity != changed.identity
    with pytest.raises(FrozenInstanceError):
        first.status = PhotometricCompatibilityStatus.INCOMPATIBLE  # type: ignore[misc]
