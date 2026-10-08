#!/usr/bin/env python3
"""Run the deterministic V2L17 photometric pipeline regression fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ColorConversionAssessment,
    ColorConversionRequest,
    ColorConversionStatus,
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
    PhotometricCompatibilityAssessment,
    PhotometricCompatibilityInput,
    PhotometricNormalizationAssessment,
    PhotometricNormalizationFactors,
    PhotometricNormalizationRequest,
    PhotometricNormalizationStatus,
    PhotometricValidityMasks,
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
    build_photometric_validity_masks,
    convert_rgb8_to_linear_reference,
    normalize_linear_rgb_reference,
)

_SCHEMA_VERSION = 1
_FIXTURE_ID = "fixture.photometric-pipeline-controlled.v1"
_LEVEL_PATH = "levels/level-000000.rgb"


@dataclass(frozen=True, slots=True)
class _CaseStage:
    case_id: str
    observation_id: ObservationId
    pixels: bytes
    source_photometry: SourcePhotometryMetadata
    color_assessment: ColorConversionAssessment
    converted: ConvertedLinearRgbBuffer | None
    normalization_assessment: PhotometricNormalizationAssessment | None
    normalized: NormalizedLinearRgbBuffer | None
    validity_masks: PhotometricValidityMasks | None
    record: dict[str, Any]


def _require_exact_keys(value: object, expected: tuple[str, ...], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    expected_set = set(expected)
    actual_set = set(value)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing or extra:
        raise ValueError(f"{context} fields mismatch: missing={missing} extra={extra}")
    if list(value) != sorted(value):
        raise ValueError(f"{context} fields must be in canonical lexical order")
    return value


def _require_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return value


def _raw_entry(value: object, context: str) -> RawMetadataEntry:
    item = _require_exact_keys(value, ("key", "namespace", "value"), context)
    if not all(isinstance(item[name], str) for name in ("key", "namespace", "value")):
        raise ValueError(f"{context} values must be strings")
    return RawMetadataEntry(
        namespace=item["namespace"],
        key=item["key"],
        value=item["value"],
    )


def _evidence(value: object, raw_entries: tuple[RawMetadataEntry, ...], context: str) -> tuple[RawMetadataEntry, ...]:
    entries = tuple(
        _raw_entry(item, f"{context}[{index}]")
        for index, item in enumerate(_require_list(value, context))
    )
    identities = tuple((item.namespace, item.key, item.value) for item in entries)
    if identities != tuple(sorted(identities)):
        raise ValueError(f"{context} must be in canonical namespace/key/value order")
    raw_set = set(raw_entries)
    if any(item not in raw_set for item in entries):
        raise ValueError(f"{context} must reference exact raw_metadata entries")
    return entries


def _status(value: object, context: str) -> SourcePhotometryInterpretationStatus:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return SourcePhotometryInterpretationStatus(value)
    except ValueError as exc:
        raise ValueError(f"{context} is unsupported") from exc


def _optional_float(value: object, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _source_photometry(case: dict[str, Any]) -> SourcePhotometryMetadata:
    observation_id = ObservationId(case["observation_id"])
    raw_entries = tuple(
        _raw_entry(item, f"{case['case_id']}.raw_metadata[{index}]")
        for index, item in enumerate(case["raw_metadata"])
    )
    raw_identity = tuple((item.namespace, item.key, item.value) for item in raw_entries)
    if raw_identity != tuple(sorted(raw_identity)):
        raise ValueError(f"{case['case_id']}.raw_metadata must be canonical and sorted")

    color_raw = _require_exact_keys(
        case["source_color"],
        (
            "declared_color_space",
            "declared_matrix_coefficients",
            "declared_primaries",
            "declared_range",
            "declared_transfer_characteristic",
            "evidence",
            "issue",
            "profile_name",
            "status",
        ),
        f"{case['case_id']}.source_color",
    )
    exposure_raw = _require_exact_keys(
        case["source_exposure"],
        (
            "evidence",
            "exposure_compensation_ev",
            "exposure_time_seconds",
            "f_number",
            "iso_speed",
            "issue",
            "status",
        ),
        f"{case['case_id']}.source_exposure",
    )
    white_balance_raw = _require_exact_keys(
        case["source_white_balance"],
        (
            "color_temperature_kelvin",
            "evidence",
            "issue",
            "mode",
            "status",
        ),
        f"{case['case_id']}.source_white_balance",
    )

    metadata = ObservationMetadata(observation_id=observation_id, raw_entries=raw_entries)
    color = SourceColorMetadata(
        status=_status(color_raw["status"], f"{case['case_id']}.source_color.status"),
        declared_color_space=color_raw["declared_color_space"],
        declared_primaries=color_raw["declared_primaries"],
        declared_transfer_characteristic=color_raw["declared_transfer_characteristic"],
        declared_matrix_coefficients=color_raw["declared_matrix_coefficients"],
        declared_range=color_raw["declared_range"],
        profile_name=color_raw["profile_name"],
        evidence=_evidence(
            color_raw["evidence"],
            raw_entries,
            f"{case['case_id']}.source_color.evidence",
        ),
        issue=color_raw["issue"],
    )
    exposure = SourceExposureMetadata(
        status=_status(exposure_raw["status"], f"{case['case_id']}.source_exposure.status"),
        iso_speed=_optional_float(exposure_raw["iso_speed"], "source_exposure.iso_speed"),
        exposure_time_seconds=_optional_float(
            exposure_raw["exposure_time_seconds"],
            "source_exposure.exposure_time_seconds",
        ),
        f_number=_optional_float(exposure_raw["f_number"], "source_exposure.f_number"),
        exposure_compensation_ev=_optional_float(
            exposure_raw["exposure_compensation_ev"],
            "source_exposure.exposure_compensation_ev",
        ),
        evidence=_evidence(
            exposure_raw["evidence"],
            raw_entries,
            f"{case['case_id']}.source_exposure.evidence",
        ),
        issue=exposure_raw["issue"],
    )
    white_balance = SourceWhiteBalanceMetadata(
        status=_status(
            white_balance_raw["status"],
            f"{case['case_id']}.source_white_balance.status",
        ),
        mode=white_balance_raw["mode"],
        color_temperature_kelvin=_optional_float(
            white_balance_raw["color_temperature_kelvin"],
            "source_white_balance.color_temperature_kelvin",
        ),
        evidence=_evidence(
            white_balance_raw["evidence"],
            raw_entries,
            f"{case['case_id']}.source_white_balance.evidence",
        ),
        issue=white_balance_raw["issue"],
    )
    return SourcePhotometryMetadata(
        observation_id=observation_id,
        source_metadata=metadata,
        color=color,
        exposure=exposure,
        white_balance=white_balance,
    )


def _pixels(case: dict[str, Any]) -> bytes:
    dimensions = case["dimensions"]
    width = dimensions["width_px"]
    height = dimensions["height_px"]
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError(f"{case['case_id']}.dimensions must be positive integers")
    values = _require_list(case["rgb8"], f"{case['case_id']}.rgb8")
    if len(values) != width * height * 3:
        raise ValueError(f"{case['case_id']}.rgb8 does not match dimensions")
    if any(type(value) is not int or not 0 <= value <= 255 for value in values):
        raise ValueError(f"{case['case_id']}.rgb8 must contain canonical uint8 values")
    return bytes(values)


def _color_request(
    case: dict[str, Any],
    pixels: bytes,
    source_photometry: SourcePhotometryMetadata,
) -> ColorConversionRequest:
    case_id = case["case_id"]
    observation_id = source_photometry.observation_id
    dimensions = case["dimensions"]
    source_asset_sha256 = Sha256Digest(
        hashlib.sha256(f"v2l17.5.source-asset\0{case_id}".encode("ascii")).hexdigest()
    )
    manifest = DecodedImagePyramidManifest(
        source_observation_id=observation_id,
        source_kind=ObservationKind.IMAGE,
        source_asset_sha256=source_asset_sha256,
        pixel_layout=DecodedImagePixelLayout.RGB8_PACKED,
        orientation_policy=DecodedImageOrientationPolicy.SOURCE_PIXELS,
        spec=DecodedImagePyramidSpec(
            minimum_max_edge_px=max(dimensions["width_px"], dimensions["height_px"])
        ),
        levels=(
            DecodedImageLevelDescriptor(
                level_index=0,
                width_px=dimensions["width_px"],
                height_px=dimensions["height_px"],
                relative_path=_LEVEL_PATH,
            ),
        ),
    )
    artifact_ref = ArtifactRef(
        artifact_id=ArtifactId(f"artifact:v2l17-5:{case_id}:decoded"),
        artifact_kind=ArtifactKind("media.decoded_image_pyramid"),
    )
    producer = ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.fixture.photometric-pipeline-controlled.decoded",
            version="1.0.0",
        ),
        configuration=ConfigurationIdentity(
            sha256=Sha256Digest(
                hashlib.sha256(f"v2l17.5.decoded-config\0{case_id}".encode("ascii")).hexdigest()
            )
        ),
    )
    artifact = ArtifactMetadata(
        project_id=SceneProjectId("project:v2l17-5:photometric-controlled"),
        artifact_ref=artifact_ref,
        artifact_key=derive_artifact_key(
            ArtifactKeyMaterial(
                output_kind=artifact_ref.artifact_kind,
                input_fingerprints=(
                    ArtifactInputFingerprint(
                        artifact_kind=ArtifactKind("image.observation"),
                        sha256=source_asset_sha256,
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
                relative_path=_LEVEL_PATH,
                byte_length=len(pixels),
                sha256=Sha256Digest(hashlib.sha256(pixels).hexdigest()),
            ),
        ),
    )
    try:
        decoded_encoding = DecodedPixelColorEncoding(case["decoded_encoding"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{case_id}.decoded_encoding is unsupported") from exc

    return ColorConversionRequest(
        decoded_manifest=manifest,
        decoded_artifact=artifact,
        decoded_materialization=materialization,
        source_photometry=source_photometry,
        decoded_level_index=0,
        decoded_encoding=decoded_encoding,
        working_convention=LINEAR_SRGB_F64,
        output_convention=LINEAR_SRGB_F64,
    )


def _normalization_factors(case: dict[str, Any]) -> PhotometricNormalizationFactors:
    raw = case["normalization_factors"]
    ev = _optional_float(
        raw["exposure_adjustment_ev"],
        f"{case['case_id']}.normalization_factors.exposure_adjustment_ev",
    )
    gains_raw = raw["white_balance_rgb_gains"]
    gains: tuple[float, float, float] | None
    if gains_raw is None:
        gains = None
    else:
        values = _require_list(
            gains_raw,
            f"{case['case_id']}.normalization_factors.white_balance_rgb_gains",
        )
        if len(values) != 3:
            raise ValueError("white_balance_rgb_gains must contain exactly three values")
        converted = tuple(
            _optional_float(
                value,
                f"{case['case_id']}.normalization_factors.white_balance_rgb_gains",
            )
            for value in values
        )
        if any(value is None for value in converted):
            raise ValueError("white_balance_rgb_gains must not contain null")
        gains = (converted[0], converted[1], converted[2])  # type: ignore[arg-type]
    return PhotometricNormalizationFactors(
        exposure_adjustment_ev=ev,
        white_balance_rgb_gains=gains,
    )


def _float_values(buffer: bytes) -> list[float]:
    return [value for (value,) in struct.iter_unpack("!d", buffer)]


def _raw_record(entries: tuple[RawMetadataEntry, ...]) -> list[dict[str, str]]:
    return [
        {"namespace": item.namespace, "key": item.key, "value": item.value}
        for item in entries
    ]


def _source_record(
    request: ColorConversionRequest,
    source_photometry: SourcePhotometryMetadata,
) -> dict[str, Any]:
    entry = request.decoded_level_entry
    return {
        "decoded_artifact_id": request.decoded_artifact.artifact_ref.artifact_id.value,
        "decoded_artifact_key": request.decoded_artifact.artifact_key.value,
        "decoded_level": {
            "byte_length": entry.byte_length,
            "relative_path": entry.relative_path,
            "sha256": entry.sha256.value,
        },
        "observation_id": source_photometry.observation_id.value,
        "raw_metadata": _raw_record(source_photometry.source_metadata.raw_entries),
        "source_asset_sha256": request.decoded_manifest.source_asset_sha256.value,
        "source_photometry_statuses": {
            "color": source_photometry.color.status.value,
            "exposure": source_photometry.exposure.status.value,
            "white_balance": source_photometry.white_balance.status.value,
        },
    }


def _color_record(
    assessment: ColorConversionAssessment,
    converted: ConvertedLinearRgbBuffer | None,
) -> dict[str, Any]:
    return {
        "converted_content_sha256": (
            converted.content_sha256.value if converted is not None else None
        ),
        "converted_derived_sha256": (
            converted.derived_sha256.value if converted is not None else None
        ),
        "converted_values": (
            _float_values(converted.packed_rgb_f64_be) if converted is not None else None
        ),
        "issue": assessment.issue,
        "plan_identity": assessment.plan.identity.value if assessment.plan is not None else None,
        "status": assessment.status.value,
    }


def _normalization_record(
    assessment: PhotometricNormalizationAssessment | None,
    normalized: NormalizedLinearRgbBuffer | None,
) -> dict[str, Any] | None:
    if assessment is None:
        return None
    return {
        "issue": assessment.issue,
        "normalized_content_sha256": (
            normalized.content_sha256.value if normalized is not None else None
        ),
        "normalized_derived_sha256": (
            normalized.derived_sha256.value if normalized is not None else None
        ),
        "normalized_values": (
            _float_values(normalized.packed_rgb_f64_be) if normalized is not None else None
        ),
        "plan_identity": assessment.plan.identity.value if assessment.plan is not None else None,
        "status": assessment.status.value,
    }


def _mask_record(mask: Any) -> dict[str, Any]:
    return {
        "content_sha256": mask.content_sha256.value,
        "identity": mask.identity.value,
        "kind": mask.kind.value,
        "packed_mask": list(mask.packed_mask),
    }


def _validity_record(masks: PhotometricValidityMasks | None) -> dict[str, Any] | None:
    if masks is None:
        return None
    return {
        "identity": masks.identity.value,
        "normalized_above_one": _mask_record(masks.normalized_above_one),
        "source_high_endpoint_candidate": _mask_record(masks.source_high_endpoint_candidate),
        "source_low_endpoint_candidate": _mask_record(masks.source_low_endpoint_candidate),
    }


def _compatibility_record(
    assessment: PhotometricCompatibilityAssessment,
) -> dict[str, Any]:
    value = assessment.compatibility_input
    return {
        "color_observation_id": (
            value.color_assessment.request.decoded_manifest.source_observation_id.value
        ),
        "identity": assessment.identity.value,
        "normalization_observation_id": (
            value.normalization_assessment.request.source_photometry.observation_id.value
        ),
        "reasons": list(assessment.reasons),
        "status": assessment.status.value,
    }


def _assert_float_values(actual: list[float] | None, expected: object, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{context} unexpectedly produced values")
        return
    expected_values = _require_list(expected, context)
    if actual is None or len(actual) != len(expected_values):
        raise ValueError(f"{context} value count mismatch")
    for index, (left, right) in enumerate(zip(actual, expected_values, strict=True)):
        if isinstance(right, bool) or not isinstance(right, (int, float)):
            raise ValueError(f"{context}[{index}] expected value must be numeric")
        if not math.isclose(left, float(right), rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{context}[{index}] differs from checked-in expectation")


def _assert_mask(actual: list[int] | None, expected: object, context: str) -> None:
    if expected is None:
        if actual is not None:
            raise ValueError(f"{context} unexpectedly produced a mask")
        return
    expected_mask = _require_list(expected, context)
    if actual != expected_mask:
        raise ValueError(f"{context} differs from checked-in expectation")


def _validate_case(value: object, index: int) -> dict[str, Any]:
    context = f"cases[{index}]"
    case = _require_exact_keys(
        value,
        (
            "case_id",
            "compatibility_normalization_case_id",
            "decoded_encoding",
            "dimensions",
            "expected",
            "normalization_factors",
            "observation_id",
            "raw_metadata",
            "rgb8",
            "source_color",
            "source_exposure",
            "source_white_balance",
        ),
        context,
    )
    if not isinstance(case["case_id"], str) or not case["case_id"]:
        raise ValueError(f"{context}.case_id must be non-blank")
    if not isinstance(case["observation_id"], str) or not case["observation_id"]:
        raise ValueError(f"{context}.observation_id must be non-blank")
    if case["compatibility_normalization_case_id"] is not None and not isinstance(
        case["compatibility_normalization_case_id"], str
    ):
        raise ValueError(f"{context}.compatibility_normalization_case_id must be string or null")
    _require_exact_keys(case["dimensions"], ("height_px", "width_px"), f"{context}.dimensions")
    _require_exact_keys(
        case["normalization_factors"],
        ("exposure_adjustment_ev", "white_balance_rgb_gains"),
        f"{context}.normalization_factors",
    )
    _require_exact_keys(
        case["expected"],
        (
            "color_status",
            "compatibility_reason_substrings",
            "compatibility_status",
            "converted_values",
            "normalization_status",
            "normalized_above_one_mask",
            "normalized_values",
            "source_high_endpoint_candidate_mask",
            "source_low_endpoint_candidate_mask",
        ),
        f"{context}.expected",
    )
    for index_raw, raw in enumerate(_require_list(case["raw_metadata"], f"{context}.raw_metadata")):
        _require_exact_keys(raw, ("key", "namespace", "value"), f"{context}.raw_metadata[{index_raw}]")
    return case


def _build_stage(case: dict[str, Any]) -> _CaseStage:
    pixels = _pixels(case)
    photometry = _source_photometry(case)
    color_request = _color_request(case, pixels, photometry)
    color_assessment = assess_color_conversion(color_request)

    converted: ConvertedLinearRgbBuffer | None = None
    normalization_assessment: PhotometricNormalizationAssessment | None = None
    normalized: NormalizedLinearRgbBuffer | None = None
    validity_masks: PhotometricValidityMasks | None = None

    if color_assessment.status is ColorConversionStatus.READY:
        if color_assessment.plan is None:
            raise RuntimeError("ready color assessment unexpectedly lacks plan")
        converted = convert_rgb8_to_linear_reference(color_assessment.plan, pixels)
        normalization_request = PhotometricNormalizationRequest(
            source_plan=color_assessment.plan,
            source_content_sha256=converted.content_sha256,
            source_derived_sha256=converted.derived_sha256,
            source_photometry=photometry,
            factors=_normalization_factors(case),
        )
        normalization_assessment = assess_photometric_normalization(normalization_request)
        if normalization_assessment.status is PhotometricNormalizationStatus.READY:
            if normalization_assessment.plan is None:
                raise RuntimeError("ready normalization assessment unexpectedly lacks plan")
            normalized = normalize_linear_rgb_reference(
                normalization_assessment.plan,
                converted,
            )
            dimensions = case["dimensions"]
            validity_request = PhotometricValidityRequest(
                observation_id=photometry.observation_id,
                source_plan=color_assessment.plan,
                source_entry=color_request.decoded_level_entry,
                normalization_plan=normalization_assessment.plan,
                normalized_content_sha256=normalized.content_sha256,
                normalized_derived_sha256=normalized.derived_sha256,
                dimensions=ImageDimensions(
                    width_px=dimensions["width_px"],
                    height_px=dimensions["height_px"],
                ),
                channel_count=3,
            )
            validity_masks = build_photometric_validity_masks(
                validity_request,
                pixels,
                normalized,
            )

    record = {
        "case_id": case["case_id"],
        "color": _color_record(color_assessment, converted),
        "normalization": _normalization_record(normalization_assessment, normalized),
        "source": _source_record(color_request, photometry),
        "validity": _validity_record(validity_masks),
    }
    return _CaseStage(
        case_id=case["case_id"],
        observation_id=photometry.observation_id,
        pixels=pixels,
        source_photometry=photometry,
        color_assessment=color_assessment,
        converted=converted,
        normalization_assessment=normalization_assessment,
        normalized=normalized,
        validity_masks=validity_masks,
        record=record,
    )


def _assert_expected(case: dict[str, Any], stage: _CaseStage, compatibility: PhotometricCompatibilityAssessment) -> None:
    expected = case["expected"]
    if stage.color_assessment.status.value != expected["color_status"]:
        raise ValueError(f"{stage.case_id} color status differs from expectation")

    actual_normalization_status = (
        stage.normalization_assessment.status.value
        if stage.normalization_assessment is not None
        else None
    )
    if actual_normalization_status != expected["normalization_status"]:
        raise ValueError(f"{stage.case_id} normalization status differs from expectation")
    if compatibility.status.value != expected["compatibility_status"]:
        raise ValueError(f"{stage.case_id} compatibility status differs from expectation")

    _assert_float_values(
        stage.record["color"]["converted_values"],
        expected["converted_values"],
        f"{stage.case_id}.expected.converted_values",
    )
    normalization_record = stage.record["normalization"]
    actual_normalized_values = (
        normalization_record["normalized_values"] if normalization_record is not None else None
    )
    _assert_float_values(
        actual_normalized_values,
        expected["normalized_values"],
        f"{stage.case_id}.expected.normalized_values",
    )

    validity = stage.record["validity"]
    _assert_mask(
        None if validity is None else validity["source_low_endpoint_candidate"]["packed_mask"],
        expected["source_low_endpoint_candidate_mask"],
        f"{stage.case_id}.expected.source_low_endpoint_candidate_mask",
    )
    _assert_mask(
        None if validity is None else validity["source_high_endpoint_candidate"]["packed_mask"],
        expected["source_high_endpoint_candidate_mask"],
        f"{stage.case_id}.expected.source_high_endpoint_candidate_mask",
    )
    _assert_mask(
        None if validity is None else validity["normalized_above_one"]["packed_mask"],
        expected["normalized_above_one_mask"],
        f"{stage.case_id}.expected.normalized_above_one_mask",
    )

    substrings = _require_list(
        expected["compatibility_reason_substrings"],
        f"{stage.case_id}.expected.compatibility_reason_substrings",
    )
    for substring in substrings:
        if not isinstance(substring, str) or not substring:
            raise ValueError("compatibility reason substring must be non-blank")
        if not any(substring in reason for reason in compatibility.reasons):
            raise ValueError(
                f"{stage.case_id} compatibility reasons lack expected substring {substring!r}"
            )
    if not substrings and compatibility.reasons:
        raise ValueError(f"{stage.case_id} unexpectedly carries compatibility reasons")


def run_fixture(fixture_path: Path) -> dict[str, Any]:
    """Compose V2L17.1-V2L17.4 without decoding media, inferring factors, or repairing values."""

    if not isinstance(fixture_path, Path):
        raise TypeError("fixture_path must be pathlib.Path")
    raw_bytes = fixture_path.read_bytes()
    try:
        document = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fixture must be UTF-8 canonical JSON") from exc

    top = _require_exact_keys(document, ("cases", "fixture_id", "schema_version"), "fixture")
    canonical_bytes = (json.dumps(top, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if raw_bytes != canonical_bytes:
        raise ValueError("fixture JSON must use canonical sorted two-space encoding")
    if top["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("fixture schema_version is unsupported")
    if top["fixture_id"] != _FIXTURE_ID:
        raise ValueError("fixture_id is unsupported")

    cases = [
        _validate_case(item, index)
        for index, item in enumerate(_require_list(top["cases"], "fixture.cases"))
    ]
    case_ids = [case["case_id"] for case in cases]
    if not case_ids or case_ids != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
        raise ValueError("fixture cases must be non-empty unique and canonically sorted")

    stages = {case["case_id"]: _build_stage(case) for case in cases}
    records: list[dict[str, Any]] = []
    for case in cases:
        stage = stages[case["case_id"]]
        compatibility_case_id = case["compatibility_normalization_case_id"] or case["case_id"]
        if compatibility_case_id not in stages:
            raise ValueError(
                f"{case['case_id']} references unknown compatibility normalization case"
            )
        normalization_stage = stages[compatibility_case_id]
        normalization_assessment = normalization_stage.normalization_assessment
        if normalization_assessment is None:
            raise ValueError(
                f"{case['case_id']} compatibility normalization evidence is not constructible"
            )
        compatibility = assess_photometric_compatibility(
            PhotometricCompatibilityInput(
                color_assessment=stage.color_assessment,
                normalization_assessment=normalization_assessment,
                normalized_content_sha256=(
                    normalization_stage.normalized.content_sha256
                    if normalization_stage.normalized is not None
                    else None
                ),
                normalized_derived_sha256=(
                    normalization_stage.normalized.derived_sha256
                    if normalization_stage.normalized is not None
                    else None
                ),
            )
        )
        _assert_expected(case, stage, compatibility)
        record = dict(stage.record)
        record["compatibility"] = _compatibility_record(compatibility)
        records.append(record)

    return {
        "cases": records,
        "fixture_id": top["fixture_id"],
        "fixture_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "schema_version": _SCHEMA_VERSION,
    }


def write_evidence(fixture_path: Path, output_path: Path) -> dict[str, Any]:
    """Run the controlled fixture and persist canonical machine-readable evidence."""

    evidence = run_fixture(fixture_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_evidence(args.fixture, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
