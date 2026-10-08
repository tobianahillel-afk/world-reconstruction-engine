from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, fields, replace
from typing import Any, cast

import pytest

from wre.domain import (
    LINEAR_SRGB_F64,
    REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND,
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
    ObservationId,
    ObservationKind,
    ObservationMetadata,
    PhotometricNormalizationBackend,
    PhotometricNormalizationFactors,
    PhotometricNormalizationRequest,
    PhotometricNormalizationStatus,
    PhotometricNormalizationStep,
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
from wre.photometry import ConvertedLinearRgbBuffer, convert_rgb8_to_linear_reference

OBS = ObservationId("obs:photometric-normalization")
PIXELS = bytes((64, 128, 255))
COLOR = RawMetadataEntry("caller", "color", "sRGB")
ISO = RawMetadataEntry("exif", "iso", "100")
SHUTTER = RawMetadataEntry("exif", "exposure_time", "1/125")
APERTURE = RawMetadataEntry("exif", "f_number", "2.8")
COMPENSATION = RawMetadataEntry("exif", "exposure_compensation", "0")
WB_MODE = RawMetadataEntry("exif", "white_balance", "manual")
WB_TEMP = RawMetadataEntry("exif", "color_temperature", "5200")


def _source_case(
    *,
    pixels: bytes = PIXELS,
) -> tuple[ConvertedLinearRgbBuffer, SourcePhotometryMetadata]:
    raw_entries = tuple(
        sorted(
            (COLOR, ISO, SHUTTER, APERTURE, COMPENSATION, WB_MODE, WB_TEMP),
            key=lambda entry: (entry.namespace, entry.key, entry.value),
        )
    )
    source_metadata = ObservationMetadata(observation_id=OBS, raw_entries=raw_entries)
    photometry = SourcePhotometryMetadata(
        observation_id=OBS,
        source_metadata=source_metadata,
        color=SourceColorMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            declared_color_space="sRGB",
            declared_primaries="bt709_d65",
            declared_transfer_characteristic="srgb",
            declared_matrix_coefficients="identity",
            declared_range="full",
            evidence=(COLOR,),
        ),
        exposure=SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            iso_speed=100.0,
            exposure_time_seconds=1.0 / 125.0,
            f_number=2.8,
            exposure_compensation_ev=0.0,
            evidence=tuple(
                sorted(
                    (ISO, SHUTTER, APERTURE, COMPENSATION),
                    key=lambda entry: (entry.namespace, entry.key, entry.value),
                )
            ),
        ),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            mode="manual",
            color_temperature_kelvin=5200.0,
            evidence=tuple(
                sorted(
                    (WB_MODE, WB_TEMP),
                    key=lambda entry: (entry.namespace, entry.key, entry.value),
                )
            ),
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=OBS,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=Sha256Digest("a" * 64),
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
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:photometric-normalization-decoded"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(implementation="test.decoded.photometric", version="1.0.0"),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("c" * 64)),
    )
    artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:photometric-normalization"),
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
                sha256=Sha256Digest(hashlib.sha256(pixels).hexdigest()),
                byte_length=len(pixels),
            ),
        ),
    )
    color_request = ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=artifact,
        decoded_materialization=materialization,
        source_photometry=photometry,
        decoded_level_index=0,
        decoded_encoding=DecodedPixelColorEncoding.SRGB_FULL_RGB8,
        working_convention=LINEAR_SRGB_F64,
        output_convention=LINEAR_SRGB_F64,
    )
    color_plan = assess_color_conversion(color_request).plan
    assert color_plan is not None
    return convert_rgb8_to_linear_reference(color_plan, pixels), photometry


def _request(
    *,
    factors: PhotometricNormalizationFactors | None = None,
    pixels: bytes = PIXELS,
) -> PhotometricNormalizationRequest:
    source, photometry = _source_case(pixels=pixels)
    return PhotometricNormalizationRequest(
        source_plan=source.plan,
        source_content_sha256=source.content_sha256,
        source_derived_sha256=source.derived_sha256,
        source_photometry=photometry,
        factors=factors
        or PhotometricNormalizationFactors(
            exposure_adjustment_ev=1.0,
            white_balance_rgb_gains=(1.0, 0.5, 2.0),
        ),
    )


def test_contract_shapes_and_status_vocabulary_are_exact() -> None:
    assert tuple(item.value for item in PhotometricNormalizationStatus) == (
        "ready",
        "unresolved",
        "rejected",
    )
    assert tuple(item.value for item in PhotometricNormalizationStep) == (
        "exposure_ev_scale",
        "white_balance_rgb_gains",
    )
    assert [field.name for field in fields(PhotometricNormalizationFactors)] == [
        "exposure_adjustment_ev",
        "white_balance_rgb_gains",
    ]
    assert [field.name for field in fields(PhotometricNormalizationRequest)] == [
        "source_plan",
        "source_content_sha256",
        "source_derived_sha256",
        "source_photometry",
        "factors",
    ]


def test_capture_metadata_never_synthesizes_normalization_factors() -> None:
    source, photometry = _source_case()
    request = PhotometricNormalizationRequest(
        source_plan=source.plan,
        source_content_sha256=source.content_sha256,
        source_derived_sha256=source.derived_sha256,
        source_photometry=photometry,
        factors=PhotometricNormalizationFactors(
            exposure_adjustment_ev=None,
            white_balance_rgb_gains=None,
        ),
    )
    result = assess_photometric_normalization(request)
    assert photometry.exposure.status is SourcePhotometryInterpretationStatus.RESOLVED
    assert photometry.white_balance.status is SourcePhotometryInterpretationStatus.RESOLVED
    assert result.status is PhotometricNormalizationStatus.UNRESOLVED
    assert result.plan is None
    assert result.issue


@pytest.mark.parametrize(
    "factors",
    [
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=math.nan,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=math.inf,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=(0.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=(-1.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=(math.inf, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=1024.0,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=-1075.0,
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        ),
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=2.0,
            white_balance_rgb_gains=(1.0e308, 1.0, 1.0),
        ),
    ],
)
def test_invalid_or_unrepresentable_factors_are_rejected(
    factors: PhotometricNormalizationFactors,
) -> None:
    result = assess_photometric_normalization(_request(factors=factors))
    assert result.status is PhotometricNormalizationStatus.REJECTED
    assert result.plan is None
    assert result.issue


def test_factor_shapes_and_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="exposure_adjustment_ev"):
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=cast(Any, 1),
            white_balance_rgb_gains=(1.0, 1.0, 1.0),
        )
    with pytest.raises(TypeError, match="tuple"):
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=cast(Any, [1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="RGB gains"):
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=cast(Any, (1.0, 1.0)),
        )
    with pytest.raises(TypeError, match="members"):
        PhotometricNormalizationFactors(
            exposure_adjustment_ev=0.0,
            white_balance_rgb_gains=cast(Any, (1.0, 1, 1.0)),
        )


def test_request_rejects_foreign_photometry_lineage() -> None:
    request = _request()
    foreign_photometry = replace(
        request.source_photometry,
        exposure=SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT),
    )
    with pytest.raises(ValueError, match="lineage"):
        replace(request, source_photometry=foreign_photometry)


def test_plan_identity_is_deterministic_and_provenance_sensitive() -> None:
    request = _request()
    first = assess_photometric_normalization(request)
    second = assess_photometric_normalization(request)
    assert first.status is PhotometricNormalizationStatus.READY
    assert first.plan is not None
    assert second.plan == first.plan
    assert second.plan is not None
    assert second.plan.identity == first.plan.identity
    assert first.plan.exposure_scale == 2.0

    factor_changed = assess_photometric_normalization(
        replace(
            request,
            factors=PhotometricNormalizationFactors(
                exposure_adjustment_ev=-1.0,
                white_balance_rgb_gains=(1.0, 0.5, 2.0),
            ),
        )
    ).plan
    content_changed = assess_photometric_normalization(
        replace(request, source_content_sha256=Sha256Digest("b" * 64))
    ).plan
    derived_changed = assess_photometric_normalization(
        replace(request, source_derived_sha256=Sha256Digest("c" * 64))
    ).plan
    assert factor_changed is not None
    assert content_changed is not None
    assert derived_changed is not None
    assert factor_changed.identity != first.plan.identity
    assert content_changed.identity != first.plan.identity
    assert derived_changed.identity != first.plan.identity

    backend = replace(REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND, version="999.0.0")
    assert isinstance(backend, PhotometricNormalizationBackend)
    rejected = assess_photometric_normalization(request, backend=backend)
    assert rejected.status is PhotometricNormalizationStatus.REJECTED

    with pytest.raises(FrozenInstanceError):
        first.plan.identity = Sha256Digest("0" * 64)  # type: ignore[misc]
    with pytest.raises(ValueError, match="identity"):
        replace(first.plan, identity=Sha256Digest("0" * 64))
