from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from wre.domain.artifact_materialization import ArtifactMaterializationEntry
from wre.domain.cameras import ImageDimensions
from wre.domain.color_conventions import (
    LINEAR_SRGB_F64,
    ColorConversionAssessment,
    ColorConversionPlan,
    ColorConversionStatus,
    assess_color_conversion,
)
from wre.domain.observations import ObservationId, Sha256Digest
from wre.domain.photometric_normalization import (
    PhotometricNormalizationAssessment,
    PhotometricNormalizationPlan,
    PhotometricNormalizationStatus,
    assess_photometric_normalization,
)


class PhotometricMaskKind(StrEnum):
    """Descriptive per-channel evidence; endpoint candidates are not sensor-clipping truth."""

    SOURCE_LOW_ENDPOINT_CANDIDATE = "source_low_endpoint_candidate"
    SOURCE_HIGH_ENDPOINT_CANDIDATE = "source_high_endpoint_candidate"
    NORMALIZED_ABOVE_ONE = "normalized_above_one"


class PhotometricCompatibilityStatus(StrEnum):
    COMPATIBLE = "compatible"
    UNRESOLVED = "unresolved"
    INCOMPATIBLE = "incompatible"


@dataclass(frozen=True, slots=True)
class PhotometricValidityRequest:
    """Exact V2L17.2/V2L17.3 lineage required to construct diagnostic masks."""

    observation_id: ObservationId
    source_plan: ColorConversionPlan
    source_entry: ArtifactMaterializationEntry
    normalization_plan: PhotometricNormalizationPlan
    normalized_content_sha256: Sha256Digest
    normalized_derived_sha256: Sha256Digest
    dimensions: ImageDimensions
    channel_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, ObservationId):
            raise TypeError("photometric_validity_request.observation_id must be ObservationId")
        if not isinstance(self.source_plan, ColorConversionPlan):
            raise TypeError("photometric_validity_request.source_plan must be ColorConversionPlan")
        if not isinstance(self.source_entry, ArtifactMaterializationEntry):
            raise TypeError(
                "photometric_validity_request.source_entry must be ArtifactMaterializationEntry"
            )
        if not isinstance(self.normalization_plan, PhotometricNormalizationPlan):
            raise TypeError(
                "photometric_validity_request.normalization_plan must be "
                "PhotometricNormalizationPlan"
            )
        if not isinstance(self.normalized_content_sha256, Sha256Digest):
            raise TypeError(
                "photometric_validity_request.normalized_content_sha256 must be Sha256Digest"
            )
        if not isinstance(self.normalized_derived_sha256, Sha256Digest):
            raise TypeError(
                "photometric_validity_request.normalized_derived_sha256 must be Sha256Digest"
            )
        if not isinstance(self.dimensions, ImageDimensions):
            raise TypeError("photometric_validity_request.dimensions must be ImageDimensions")
        if type(self.channel_count) is not int:
            raise TypeError("photometric_validity_request.channel_count must be int")
        if self.channel_count != 3:
            raise ValueError("photometric_validity_request.channel_count must be exactly 3")

        source_request = self.source_plan.request
        if source_request.decoded_manifest.source_observation_id != self.observation_id:
            raise ValueError("photometric_validity_request source observation identity disagrees")
        if source_request.source_photometry.observation_id != self.observation_id:
            raise ValueError(
                "photometric_validity_request photometry observation identity disagrees"
            )
        if self.source_entry != source_request.decoded_level_entry:
            raise ValueError(
                "photometric_validity_request source path byte length or digest disagrees"
            )
        level = source_request.decoded_manifest.levels[source_request.decoded_level_index]
        if self.dimensions != ImageDimensions(width_px=level.width_px, height_px=level.height_px):
            raise ValueError("photometric_validity_request dimensions disagree with decoded level")
        if self.source_entry.byte_length != (
            self.dimensions.width_px * self.dimensions.height_px * self.channel_count
        ):
            raise ValueError(
                "photometric_validity_request source byte length disagrees with dimensions"
            )

        source_assessment = assess_color_conversion(
            source_request,
            backend=self.source_plan.backend,
        )
        if (
            source_assessment.status is not ColorConversionStatus.READY
            or source_assessment.plan != self.source_plan
            or source_request.working_convention != LINEAR_SRGB_F64
            or source_request.output_convention != LINEAR_SRGB_F64
        ):
            raise ValueError(
                "photometric_validity_request source plan is not the reviewed linear-sRGB route"
            )

        normalization_request = self.normalization_plan.request
        if normalization_request.source_plan != self.source_plan:
            raise ValueError(
                "photometric_validity_request normalization plan has foreign source plan"
            )
        if normalization_request.source_photometry.observation_id != self.observation_id:
            raise ValueError(
                "photometric_validity_request normalization observation identity disagrees"
            )
        normalization_assessment = assess_photometric_normalization(
            normalization_request,
            backend=self.normalization_plan.backend,
        )
        if (
            normalization_assessment.status is not PhotometricNormalizationStatus.READY
            or normalization_assessment.plan != self.normalization_plan
        ):
            raise ValueError(
                "photometric_validity_request normalization plan is not an approved ready plan"
            )


def _mask_content_sha256(packed_mask: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(packed_mask).hexdigest())


def _mask_identity(
    request: PhotometricValidityRequest,
    kind: PhotometricMaskKind,
    content_sha256: Sha256Digest,
) -> Sha256Digest:
    payload = {
        "domain": "wre.photometric_channel_mask",
        "schema_version": 1,
        "kind": kind.value,
        "observation_id": request.observation_id.value,
        "source_plan_identity": request.source_plan.identity.value,
        "source_entry": asdict(request.source_entry),
        "normalization_plan_identity": request.normalization_plan.identity.value,
        "normalized_content_sha256": request.normalized_content_sha256.value,
        "normalized_derived_sha256": request.normalized_derived_sha256.value,
        "dimensions": asdict(request.dimensions),
        "channel_count": request.channel_count,
        "mask_content_sha256": content_sha256.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Sha256Digest(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class PhotometricChannelMask:
    """One immutable per-channel boolean mask encoded as canonical 0/1 bytes."""

    request: PhotometricValidityRequest
    kind: PhotometricMaskKind
    packed_mask: bytes
    content_sha256: Sha256Digest = field(init=False)
    identity: Sha256Digest = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, PhotometricValidityRequest):
            raise TypeError("photometric_mask.request must be PhotometricValidityRequest")
        if not isinstance(self.kind, PhotometricMaskKind):
            raise TypeError("photometric_mask.kind must be PhotometricMaskKind")
        if type(self.packed_mask) is not bytes:
            raise TypeError("photometric_mask.packed_mask must be immutable bytes")
        expected_channels = (
            self.request.dimensions.width_px
            * self.request.dimensions.height_px
            * self.request.channel_count
        )
        if len(self.packed_mask) != expected_channels:
            raise ValueError("photometric_mask packed mask does not match request dimensions")
        if any(value not in (0, 1) for value in self.packed_mask):
            raise ValueError("photometric_mask bytes must contain only canonical 0 or 1 values")
        content_sha256 = _mask_content_sha256(self.packed_mask)
        object.__setattr__(self, "content_sha256", content_sha256)
        object.__setattr__(
            self,
            "identity",
            _mask_identity(self.request, self.kind, content_sha256),
        )


def _mask_set_identity(
    request: PhotometricValidityRequest,
    masks: tuple[PhotometricChannelMask, PhotometricChannelMask, PhotometricChannelMask],
) -> Sha256Digest:
    payload = {
        "domain": "wre.photometric_validity_masks",
        "schema_version": 1,
        "observation_id": request.observation_id.value,
        "source_plan_identity": request.source_plan.identity.value,
        "normalization_plan_identity": request.normalization_plan.identity.value,
        "normalized_content_sha256": request.normalized_content_sha256.value,
        "normalized_derived_sha256": request.normalized_derived_sha256.value,
        "mask_identities": [mask.identity.value for mask in masks],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Sha256Digest(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class PhotometricValidityMasks:
    request: PhotometricValidityRequest
    source_low_endpoint_candidate: PhotometricChannelMask
    source_high_endpoint_candidate: PhotometricChannelMask
    normalized_above_one: PhotometricChannelMask
    identity: Sha256Digest = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.request, PhotometricValidityRequest):
            raise TypeError("photometric_validity_masks.request must be PhotometricValidityRequest")
        expected = (
            (
                self.source_low_endpoint_candidate,
                PhotometricMaskKind.SOURCE_LOW_ENDPOINT_CANDIDATE,
            ),
            (
                self.source_high_endpoint_candidate,
                PhotometricMaskKind.SOURCE_HIGH_ENDPOINT_CANDIDATE,
            ),
            (self.normalized_above_one, PhotometricMaskKind.NORMALIZED_ABOVE_ONE),
        )
        for mask, kind in expected:
            if not isinstance(mask, PhotometricChannelMask):
                raise TypeError("photometric_validity_masks members must be PhotometricChannelMask")
            if mask.request != self.request or mask.kind is not kind:
                raise ValueError(
                    "photometric_validity_masks members must share request and exact mask kinds"
                )
        masks = (
            self.source_low_endpoint_candidate,
            self.source_high_endpoint_candidate,
            self.normalized_above_one,
        )
        object.__setattr__(self, "identity", _mask_set_identity(self.request, masks))


@dataclass(frozen=True, slots=True)
class PhotometricCompatibilityInput:
    """Color/normalization evidence for one observation; no repair or routing instruction."""

    color_assessment: ColorConversionAssessment
    normalization_assessment: PhotometricNormalizationAssessment
    normalized_content_sha256: Sha256Digest | None = None
    normalized_derived_sha256: Sha256Digest | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.color_assessment, ColorConversionAssessment):
            raise TypeError(
                "photometric_compatibility_input.color_assessment must be ColorConversionAssessment"
            )
        if not isinstance(
            self.normalization_assessment,
            PhotometricNormalizationAssessment,
        ):
            raise TypeError(
                "photometric_compatibility_input.normalization_assessment must be "
                "PhotometricNormalizationAssessment"
            )
        for name in ("normalized_content_sha256", "normalized_derived_sha256"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Sha256Digest):
                raise TypeError(
                    f"photometric_compatibility_input.{name} must be Sha256Digest or None"
                )
        if (self.normalized_content_sha256 is None) != (self.normalized_derived_sha256 is None):
            raise ValueError(
                "photometric_compatibility_input normalized content and derived identities "
                "must be present together"
            )


def _compatibility_identity(
    compatibility_input: PhotometricCompatibilityInput,
    status: PhotometricCompatibilityStatus,
    reasons: tuple[str, ...],
) -> Sha256Digest:
    color_plan = compatibility_input.color_assessment.plan
    normalization_plan = compatibility_input.normalization_assessment.plan
    payload = {
        "domain": "wre.photometric_compatibility_assessment",
        "schema_version": 1,
        "status": status.value,
        "color_status": compatibility_input.color_assessment.status.value,
        "color_observation_id": (
            compatibility_input.color_assessment.request.decoded_manifest.source_observation_id.value
        ),
        "color_plan_identity": color_plan.identity.value if color_plan is not None else None,
        "normalization_status": compatibility_input.normalization_assessment.status.value,
        "normalization_observation_id": (
            compatibility_input.normalization_assessment.request.source_photometry.observation_id.value
        ),
        "normalization_plan_identity": (
            normalization_plan.identity.value if normalization_plan is not None else None
        ),
        "normalized_content_sha256": (
            compatibility_input.normalized_content_sha256.value
            if compatibility_input.normalized_content_sha256 is not None
            else None
        ),
        "normalized_derived_sha256": (
            compatibility_input.normalized_derived_sha256.value
            if compatibility_input.normalized_derived_sha256 is not None
            else None
        ),
        "reasons": list(reasons),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Sha256Digest(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


@dataclass(frozen=True, slots=True)
class PhotometricCompatibilityAssessment:
    status: PhotometricCompatibilityStatus
    compatibility_input: PhotometricCompatibilityInput
    reasons: tuple[str, ...]
    identity: Sha256Digest = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhotometricCompatibilityStatus):
            raise TypeError(
                "photometric_compatibility_assessment.status must be PhotometricCompatibilityStatus"
            )
        if not isinstance(self.compatibility_input, PhotometricCompatibilityInput):
            raise TypeError(
                "photometric_compatibility_assessment.compatibility_input must be "
                "PhotometricCompatibilityInput"
            )
        if not isinstance(self.reasons, tuple):
            raise TypeError("photometric_compatibility_assessment.reasons must be tuple")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons):
            raise ValueError(
                "photometric_compatibility_assessment reasons must be non-blank strings"
            )
        if self.status is PhotometricCompatibilityStatus.COMPATIBLE:
            if self.reasons:
                raise ValueError("compatible photometric assessment must not carry reasons")
        elif not self.reasons:
            raise ValueError("unresolved or incompatible photometric assessment requires reasons")
        object.__setattr__(
            self,
            "identity",
            _compatibility_identity(self.compatibility_input, self.status, self.reasons),
        )


def assess_photometric_compatibility(
    compatibility_input: PhotometricCompatibilityInput,
) -> PhotometricCompatibilityAssessment:
    """Classify evidence only; never mutate, drop, repair, renormalize or route observations."""

    if not isinstance(compatibility_input, PhotometricCompatibilityInput):
        raise TypeError("compatibility_input must be PhotometricCompatibilityInput")

    color = compatibility_input.color_assessment
    normalization = compatibility_input.normalization_assessment
    color_observation = color.request.decoded_manifest.source_observation_id
    normalization_observation = normalization.request.source_photometry.observation_id

    incompatible_reasons: list[str] = []
    unresolved_reasons: list[str] = []

    if color_observation != normalization_observation:
        incompatible_reasons.append(
            "color and normalization evidence refer to different observations"
        )
    if normalization.request.source_plan.request != color.request:
        incompatible_reasons.append("normalization evidence is bound to a foreign color request")

    if color.status is ColorConversionStatus.REJECTED:
        incompatible_reasons.append("color conversion is rejected")
    elif color.status is ColorConversionStatus.UNRESOLVED:
        unresolved_reasons.append("color conversion is unresolved")

    if normalization.status is PhotometricNormalizationStatus.REJECTED:
        incompatible_reasons.append("photometric normalization is rejected")
    elif normalization.status is PhotometricNormalizationStatus.UNRESOLVED:
        unresolved_reasons.append("photometric normalization is unresolved")

    if color.status is ColorConversionStatus.READY:
        if color.plan is None:
            incompatible_reasons.append("ready color assessment lacks a plan")
        elif (
            color.plan.request.working_convention != LINEAR_SRGB_F64
            or color.plan.request.output_convention != LINEAR_SRGB_F64
        ):
            incompatible_reasons.append("color conversion uses an unsupported working convention")

    if normalization.status is PhotometricNormalizationStatus.READY:
        if normalization.plan is None:
            incompatible_reasons.append("ready normalization assessment lacks a plan")
        elif color.plan is None or normalization.plan.request.source_plan != color.plan:
            incompatible_reasons.append("ready normalization plan is foreign to the color plan")

    normalized_ids_present = compatibility_input.normalized_content_sha256 is not None
    if (
        color.status is ColorConversionStatus.READY
        and normalization.status is PhotometricNormalizationStatus.READY
        and not normalized_ids_present
    ):
        unresolved_reasons.append("normalized output identities are absent")
    if normalization.status is not PhotometricNormalizationStatus.READY and normalized_ids_present:
        incompatible_reasons.append(
            "normalized output identities are present without a ready normalization"
        )

    if incompatible_reasons:
        return PhotometricCompatibilityAssessment(
            status=PhotometricCompatibilityStatus.INCOMPATIBLE,
            compatibility_input=compatibility_input,
            reasons=tuple(incompatible_reasons),
        )
    if unresolved_reasons:
        return PhotometricCompatibilityAssessment(
            status=PhotometricCompatibilityStatus.UNRESOLVED,
            compatibility_input=compatibility_input,
            reasons=tuple(unresolved_reasons),
        )
    return PhotometricCompatibilityAssessment(
        status=PhotometricCompatibilityStatus.COMPATIBLE,
        compatibility_input=compatibility_input,
        reasons=(),
    )
