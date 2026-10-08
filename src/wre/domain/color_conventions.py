from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum

from wre.domain.artifact_materialization import ArtifactMaterializationEntry
from wre.domain.decoded_images import (
    DecodedImageOrientationPolicy,
    DecodedImagePixelLayout,
    DecodedImagePyramidManifest,
)
from wre.domain.observations import Sha256Digest
from wre.domain.source_photometry import (
    SourcePhotometryInterpretationStatus,
    SourcePhotometryMetadata,
)


class DecodedPixelColorEncoding(StrEnum):
    """Caller-declared interpretation of decoded RGB8 bytes, never inferred from their layout."""

    UNKNOWN = "unknown"
    SRGB_FULL_RGB8 = "srgb_iec61966_2_1_full_rgb8"


class ColorPrimaries(StrEnum):
    SRGB_BT709_D65 = "srgb_bt709_d65"


class ColorTransfer(StrEnum):
    SRGB = "srgb_iec61966_2_1"
    LINEAR = "linear"


class ColorSampleStorage(StrEnum):
    UINT8_RGB = "uint8_rgb"
    FLOAT64_RGB = "float64_rgb"


class ColorConversionStatus(StrEnum):
    READY = "ready"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


class ColorConversionStep(StrEnum):
    SRGB_EOTF = "srgb_eotf_iec61966_2_1"
    LINEAR_IDENTITY = "linear_identity"


@dataclass(frozen=True, slots=True)
class ColorConvention:
    """A concrete RGB convention, not merely an RGB channel layout."""

    primaries: ColorPrimaries
    transfer: ColorTransfer
    storage: ColorSampleStorage

    def __post_init__(self) -> None:
        if not isinstance(self.primaries, ColorPrimaries):
            raise TypeError("color_convention.primaries must be ColorPrimaries")
        if not isinstance(self.transfer, ColorTransfer):
            raise TypeError("color_convention.transfer must be ColorTransfer")
        if not isinstance(self.storage, ColorSampleStorage):
            raise TypeError("color_convention.storage must be ColorSampleStorage")


@dataclass(frozen=True, slots=True)
class ColorReferenceBackend:
    """Exact bounded reference implementation and numeric conventions."""

    implementation: str
    version: str
    precision: str
    quantization: str
    specification: str

    def __post_init__(self) -> None:
        for field in ("implementation", "version", "precision", "quantization", "specification"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"color_backend.{field} must be non-blank")


REFERENCE_COLOR_BACKEND = ColorReferenceBackend(
    implementation="wre.photometry.color_conventions.python_stdlib",
    version="1.0.0",
    precision="python_float_binary64",
    quantization="none",
    specification="IEC 61966-2-1 sRGB inverse transfer; W3C sRGB color definition",
)

LINEAR_SRGB_F64 = ColorConvention(
    primaries=ColorPrimaries.SRGB_BT709_D65,
    transfer=ColorTransfer.LINEAR,
    storage=ColorSampleStorage.FLOAT64_RGB,
)


@dataclass(frozen=True, slots=True)
class ColorConversionRequest:
    """One explicitly declared source and target, bound to an immutable decoded level."""

    decoded_manifest: DecodedImagePyramidManifest
    source_photometry: SourcePhotometryMetadata
    decoded_level_index: int
    decoded_level_entry: ArtifactMaterializationEntry
    decoded_encoding: DecodedPixelColorEncoding
    working_convention: ColorConvention
    output_convention: ColorConvention

    def __post_init__(self) -> None:
        if not isinstance(self.decoded_manifest, DecodedImagePyramidManifest):
            raise TypeError("color_request.decoded_manifest must be DecodedImagePyramidManifest")
        if not isinstance(self.source_photometry, SourcePhotometryMetadata):
            raise TypeError("color_request.source_photometry must be SourcePhotometryMetadata")
        if not isinstance(self.decoded_level_entry, ArtifactMaterializationEntry):
            raise TypeError(
                "color_request.decoded_level_entry must be ArtifactMaterializationEntry"
            )
        if not isinstance(self.decoded_encoding, DecodedPixelColorEncoding):
            raise TypeError("color_request.decoded_encoding must be DecodedPixelColorEncoding")
        if not isinstance(self.working_convention, ColorConvention):
            raise TypeError("color_request.working_convention must be ColorConvention")
        if not isinstance(self.output_convention, ColorConvention):
            raise TypeError("color_request.output_convention must be ColorConvention")
        if type(self.decoded_level_index) is not int:
            raise TypeError("color_request.decoded_level_index must be int")
        if not 0 <= self.decoded_level_index < len(self.decoded_manifest.levels):
            raise ValueError("color_request.decoded_level_index is outside the manifest")
        if self.decoded_manifest.source_observation_id != self.source_photometry.observation_id:
            raise ValueError("color_request observation identities disagree")
        level = self.decoded_manifest.levels[self.decoded_level_index]
        if self.decoded_level_entry.relative_path != level.relative_path:
            raise ValueError("color_request level path does not match decoded manifest")
        if self.decoded_level_entry.byte_length != level.width_px * level.height_px * 3:
            raise ValueError("color_request decoded level must contain exactly 3 bytes per pixel")


@dataclass(frozen=True, slots=True)
class ColorConversionPlan:
    request: ColorConversionRequest
    backend: ColorReferenceBackend
    steps: tuple[ColorConversionStep, ...]
    identity: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.request, ColorConversionRequest):
            raise TypeError("color_plan.request must be ColorConversionRequest")
        if not isinstance(self.backend, ColorReferenceBackend):
            raise TypeError("color_plan.backend must be ColorReferenceBackend")
        if self.steps != (
            ColorConversionStep.SRGB_EOTF,
            ColorConversionStep.LINEAR_IDENTITY,
        ):
            raise ValueError("color_plan steps must be the reviewed sRGB reference sequence")
        if not isinstance(self.identity, Sha256Digest):
            raise TypeError("color_plan.identity must be Sha256Digest")
        if self.identity != _plan_identity(self.request, self.backend, self.steps):
            raise ValueError("color_plan identity does not match request and backend")


@dataclass(frozen=True, slots=True)
class ColorConversionAssessment:
    status: ColorConversionStatus
    request: ColorConversionRequest
    plan: ColorConversionPlan | None = None
    issue: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ColorConversionStatus):
            raise TypeError("color_assessment.status must be ColorConversionStatus")
        if not isinstance(self.request, ColorConversionRequest):
            raise TypeError("color_assessment.request must be ColorConversionRequest")
        if self.status is ColorConversionStatus.READY:
            if self.plan is None or self.issue is not None or self.plan.request != self.request:
                raise ValueError("ready color assessment requires matching plan and no issue")
        elif self.plan is not None or not isinstance(self.issue, str) or not self.issue.strip():
            raise ValueError("unresolved or rejected color assessment requires issue and no plan")


def _plan_identity(
    request: ColorConversionRequest,
    backend: ColorReferenceBackend,
    steps: tuple[ColorConversionStep, ...],
) -> Sha256Digest:
    payload = {
        "domain": "wre.color_conversion_plan",
        "schema_version": 1,
        "request": asdict(request),
        "backend": asdict(backend),
        "steps": [step.value for step in steps],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Sha256Digest(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


def assess_color_conversion(
    request: ColorConversionRequest,
    *,
    backend: ColorReferenceBackend = REFERENCE_COLOR_BACKEND,
) -> ColorConversionAssessment:
    """Qualify only fully evidenced sRGB RGB8 -> linear sRGB float64; no guessing."""

    if not isinstance(request, ColorConversionRequest):
        raise TypeError("color_request must be ColorConversionRequest")
    if not isinstance(backend, ColorReferenceBackend):
        raise TypeError("color_backend must be ColorReferenceBackend")

    def unresolved(issue: str) -> ColorConversionAssessment:
        return ColorConversionAssessment(
            status=ColorConversionStatus.UNRESOLVED, request=request, issue=issue
        )

    def rejected(issue: str) -> ColorConversionAssessment:
        return ColorConversionAssessment(
            status=ColorConversionStatus.REJECTED, request=request, issue=issue
        )

    color = request.source_photometry.color
    if color.status in (
        SourcePhotometryInterpretationStatus.ABSENT,
        SourcePhotometryInterpretationStatus.UNKNOWN,
    ):
        return unresolved("source color evidence is absent or unknown")
    if color.status is SourcePhotometryInterpretationStatus.INVALID:
        return rejected("source color declaration is invalid")
    declarations = (
        color.declared_color_space,
        color.declared_primaries,
        color.declared_transfer_characteristic,
        color.declared_matrix_coefficients,
        color.declared_range,
    )
    if any(value is None for value in declarations):
        return unresolved("source color declaration is incomplete")
    if declarations != ("sRGB", "bt709_d65", "srgb", "identity", "full"):
        return rejected("source color declaration is contradictory or unsupported")
    if color.profile_name is not None:
        return rejected("ICC/profile interpretation is outside this reference route")
    if request.decoded_encoding is DecodedPixelColorEncoding.UNKNOWN:
        return unresolved("decoded RGB8 encoding is not independently declared")
    if request.decoded_encoding is not DecodedPixelColorEncoding.SRGB_FULL_RGB8:
        return rejected("decoded pixel encoding is unsupported")
    if request.decoded_manifest.pixel_layout is not DecodedImagePixelLayout.RGB8_PACKED:
        return rejected("decoded pixel layout is unsupported")
    if (
        request.decoded_manifest.orientation_policy
        is not DecodedImageOrientationPolicy.SOURCE_PIXELS
    ):
        return rejected("decoded orientation policy is unsupported")
    if request.working_convention != LINEAR_SRGB_F64:
        return rejected("working color convention is unsupported")
    if request.output_convention != LINEAR_SRGB_F64:
        return rejected("output color convention is unsupported")
    if backend != REFERENCE_COLOR_BACKEND:
        return rejected("unreviewed color backend or precision configuration")
    steps = (ColorConversionStep.SRGB_EOTF, ColorConversionStep.LINEAR_IDENTITY)
    plan = ColorConversionPlan(
        request=request,
        backend=backend,
        steps=steps,
        identity=_plan_identity(request, backend, steps),
    )
    return ColorConversionAssessment(status=ColorConversionStatus.READY, request=request, plan=plan)
