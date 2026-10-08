from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from typing import Any, cast

import pytest

from wre.domain import (
    LINEAR_SRGB_F64,
    REFERENCE_COLOR_BACKEND,
    ArtifactId,
    ArtifactInputFingerprint,
    ArtifactKey,
    ArtifactKeyMaterial,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    ColorConvention,
    ColorConversionRequest,
    ColorConversionStatus,
    ColorPrimaries,
    ColorReferenceBackend,
    ColorSampleStorage,
    ColorTransfer,
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
    derive_artifact_key,
)

OBS = ObservationId("obs:color-reference")
BYTES = bytes((0, 10, 128, 255, 64, 32))
ENTRY = RawMetadataEntry(namespace="camera", key="colorspace", value="sRGB")


def _request(
    *,
    pixels: bytes = BYTES,
    status: SourcePhotometryInterpretationStatus = SourcePhotometryInterpretationStatus.RESOLVED,
    source_color: SourceColorMetadata | None = None,
    decoded_encoding: DecodedPixelColorEncoding = DecodedPixelColorEncoding.SRGB_FULL_RGB8,
    source_hash: Sha256Digest | None = None,
    observation: ObservationId = OBS,
    raw_entries: tuple[RawMetadataEntry, ...] = (ENTRY,),
    working: ColorConvention = LINEAR_SRGB_F64,
    output: ColorConvention = LINEAR_SRGB_F64,
) -> ColorConversionRequest:
    if source_color is None:
        if status is SourcePhotometryInterpretationStatus.RESOLVED:
            source_color = SourceColorMetadata(
                status=status,
                declared_color_space="sRGB",
                declared_primaries="bt709_d65",
                declared_transfer_characteristic="srgb",
                declared_matrix_coefficients="identity",
                declared_range="full",
                evidence=(ENTRY,),
            )
        elif status is SourcePhotometryInterpretationStatus.ABSENT:
            source_color = SourceColorMetadata(status=status)
        else:
            source_color = SourceColorMetadata(
                status=status, issue="camera did not resolve this color tag", evidence=(ENTRY,)
            )
    metadata = ObservationMetadata(observation_id=observation, raw_entries=raw_entries)
    photometry = SourcePhotometryMetadata(
        observation_id=observation,
        source_metadata=metadata,
        color=source_color,
        exposure=SourceExposureMetadata(status=SourcePhotometryInterpretationStatus.ABSENT),
        white_balance=SourceWhiteBalanceMetadata(
            status=SourcePhotometryInterpretationStatus.ABSENT
        ),
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=observation,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=source_hash or Sha256Digest("a" * 64),
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(minimum_max_edge_px=2),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0, width_px=2, height_px=1, relative_path="levels/level-000000.rgb"
            ),
        ),
    )
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:color-reference-decoded"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(implementation="test.decoded.reference", version="1.0.0"),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("c" * 64)),
    )
    artifact_key = derive_artifact_key(
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
    artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:color-reference"),
        artifact_ref=artifact_ref,
        artifact_key=artifact_key,
        producer=producer,
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )
    return ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=artifact,
        decoded_materialization=materialization,
        source_photometry=photometry,
        decoded_level_index=0,
        decoded_encoding=decoded_encoding,
        working_convention=working,
        output_convention=output,
    )


def test_no_pixel_layout_based_color_inference() -> None:
    for status in (
        SourcePhotometryInterpretationStatus.ABSENT,
        SourcePhotometryInterpretationStatus.UNKNOWN,
    ):
        result = assess_color_conversion(_request(status=status))
        assert result.status is ColorConversionStatus.UNRESOLVED
        assert result.plan is None
        assert result.issue

    missing_decoded = assess_color_conversion(
        _request(decoded_encoding=DecodedPixelColorEncoding.UNKNOWN)
    )
    assert missing_decoded.status is ColorConversionStatus.UNRESOLVED


def test_invalid_and_unsupported_color_declarations_fail_closed() -> None:
    invalid = assess_color_conversion(_request(status=SourcePhotometryInterpretationStatus.INVALID))
    assert invalid.status is ColorConversionStatus.REJECTED
    ready_source = _request().source_photometry.color
    for replacement in (
        replace(ready_source, declared_color_space="Display P3"),
        replace(ready_source, declared_range="limited"),
        replace(ready_source, declared_matrix_coefficients="bt709"),
        replace(ready_source, declared_transfer_characteristic="bt709"),
        replace(ready_source, declared_primaries="adobe_rgb"),
        replace(ready_source, profile_name="ICC profile"),
    ):
        result = assess_color_conversion(_request(source_color=replacement))
        assert result.status is ColorConversionStatus.REJECTED
        assert result.plan is None

    incomplete = replace(ready_source, declared_range=None)
    result = assess_color_conversion(_request(source_color=incomplete))
    assert result.status is ColorConversionStatus.UNRESOLVED


def test_hidden_conflicting_raw_color_declaration_is_rejected() -> None:
    conflicting = RawMetadataEntry(namespace="camera", key="colorspace", value="Display P3")
    result = assess_color_conversion(_request(raw_entries=(ENTRY, conflicting)))
    assert result.status is ColorConversionStatus.REJECTED
    assert result.plan is None
    assert result.issue == "source color evidence conflicts with bound raw metadata"


def test_rejects_unapproved_targets_backend_and_source_layout() -> None:
    srgb_float = ColorConvention(
        primaries=ColorPrimaries.SRGB_BT709_D65,
        transfer=ColorTransfer.SRGB,
        storage=ColorSampleStorage.FLOAT64_RGB,
    )
    uint8_linear = ColorConvention(
        primaries=ColorPrimaries.SRGB_BT709_D65,
        transfer=ColorTransfer.LINEAR,
        storage=ColorSampleStorage.UINT8_RGB,
    )
    assert (
        assess_color_conversion(_request(working=srgb_float)).status
        is ColorConversionStatus.REJECTED
    )
    assert (
        assess_color_conversion(_request(output=uint8_linear)).status
        is ColorConversionStatus.REJECTED
    )
    backend = replace(REFERENCE_COLOR_BACKEND, version="999.0.0")
    assert isinstance(backend, ColorReferenceBackend)
    assert (
        assess_color_conversion(_request(), backend=backend).status
        is ColorConversionStatus.REJECTED
    )


def test_request_binds_raw_observation_and_decoded_artifact_ownership() -> None:
    req = _request()
    with pytest.raises(ValueError, match="observation identities"):
        replace(
            req,
            source_photometry=replace(
                req.source_photometry,
                observation_id=ObservationId("obs:foreign"),
                source_metadata=ObservationMetadata(
                    observation_id=ObservationId("obs:foreign"), raw_entries=(ENTRY,)
                ),
            ),
        )

    foreign_ref = ArtifactRef(
        artifact_id=ArtifactId("artifact:foreign-decoded"),
        artifact_kind=req.decoded_artifact.artifact_ref.artifact_kind,
    )
    with pytest.raises(ValueError, match="refs disagree"):
        replace(
            req,
            decoded_materialization=replace(req.decoded_materialization, artifact_ref=foreign_ref),
        )

    with pytest.raises(ValueError, match="artifact key"):
        replace(
            req,
            decoded_artifact=replace(
                req.decoded_artifact,
                artifact_key=ArtifactKey(sha256=Sha256Digest("0" * 64)),
            ),
        )

    with pytest.raises(ValueError, match="materialization path"):
        replace(
            req,
            decoded_materialization=replace(
                req.decoded_materialization,
                entries=(
                    replace(req.decoded_level_entry, relative_path="levels/level-999999.rgb"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="materialization byte length"):
        replace(
            req,
            decoded_materialization=replace(
                req.decoded_materialization,
                entries=(replace(req.decoded_level_entry, byte_length=5),),
            ),
        )
    with pytest.raises(TypeError, match="decoded_encoding"):
        replace(req, decoded_encoding=cast(Any, "srgb"))
    with pytest.raises(ValueError, match="outside"):
        replace(req, decoded_level_index=1)


def test_plan_is_deterministic_provenance_bound_and_immutable() -> None:
    req = _request()
    first = assess_color_conversion(req)
    second = assess_color_conversion(req)
    assert first.status is ColorConversionStatus.READY
    first_plan = first.plan
    second_plan = second.plan
    assert first_plan is not None
    assert second_plan is not None
    assert first_plan == second_plan
    assert first_plan.identity == second_plan.identity
    assert len(first_plan.steps) == 2
    with pytest.raises(FrozenInstanceError):
        first_plan.identity = Sha256Digest("1" * 64)  # type: ignore[misc]

    source = req.source_photometry
    alternative = SourcePhotometryMetadata(
        observation_id=source.observation_id,
        source_metadata=source.source_metadata,
        color=source.color,
        exposure=SourceExposureMetadata(
            status=SourcePhotometryInterpretationStatus.RESOLVED,
            iso_speed=100.0,
            evidence=(ENTRY,),
        ),
        white_balance=source.white_balance,
    )
    changed_requests = (
        _request(source_hash=Sha256Digest("b" * 64)),
        _request(pixels=bytes(reversed(BYTES))),
        replace(req, source_photometry=alternative),
    )
    for changed_request in changed_requests:
        changed_plan = assess_color_conversion(changed_request).plan
        assert changed_plan is not None
        assert changed_plan.identity != first_plan.identity
