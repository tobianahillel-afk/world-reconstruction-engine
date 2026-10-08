from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from enum import StrEnum

from wre.domain.color_conventions import (
    LINEAR_SRGB_F64,
    ColorConversionPlan,
    ColorConversionStatus,
    assess_color_conversion,
)
from wre.domain.observations import Sha256Digest
from wre.domain.source_photometry import SourcePhotometryMetadata


class PhotometricNormalizationStatus(StrEnum):
    READY = "ready"
    UNRESOLVED = "unresolved"
    REJECTED = "rejected"


class PhotometricNormalizationStep(StrEnum):
    EXPOSURE_EV_SCALE = "exposure_ev_scale"
    WHITE_BALANCE_RGB_GAINS = "white_balance_rgb_gains"


@dataclass(frozen=True, slots=True)
class PhotometricNormalizationFactors:
    """Caller-supplied factors; capture metadata never synthesizes these values."""

    exposure_adjustment_ev: float | None
    white_balance_rgb_gains: tuple[float, float, float] | None

    def __post_init__(self) -> None:
        if (
            self.exposure_adjustment_ev is not None
            and type(self.exposure_adjustment_ev) is not float
        ):
            raise TypeError(
                "photometric_factors.exposure_adjustment_ev must be float or None"
            )
        if self.white_balance_rgb_gains is not None:
            if not isinstance(self.white_balance_rgb_gains, tuple):
                raise TypeError("photometric_factors.white_balance_rgb_gains must be tuple or None")
            if len(self.white_balance_rgb_gains) != 3:
                raise ValueError(
                    "photometric_factors.white_balance_rgb_gains must contain RGB gains"
                )
            if any(type(value) is not float for value in self.white_balance_rgb_gains):
                raise TypeError("photometric_factors.white_balance_rgb_gains members must be float")


@dataclass(frozen=True, slots=True)
class PhotometricNormalizationBackend:
    implementation: str
    version: str
    precision: str
    quantization: str
    specification: str

    def __post_init__(self) -> None:
        for field in ("implementation", "version", "precision", "quantization", "specification"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"photometric_backend.{field} must be non-blank")


REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND = PhotometricNormalizationBackend(
    implementation="wre.photometry.photometric_normalization.python_stdlib",
    version="1.0.0",
    precision="python_float_binary64",
    quantization="none",
    specification="explicit linear-light 2^EV scale followed by caller-supplied RGB gains",
)


@dataclass(frozen=True, slots=True)
class PhotometricNormalizationRequest:
    """Normalization intent bound to one exact V2L17.2 linear-color result identity."""

    source_plan: ColorConversionPlan
    source_content_sha256: Sha256Digest
    source_derived_sha256: Sha256Digest
    source_photometry: SourcePhotometryMetadata
    factors: PhotometricNormalizationFactors

    def __post_init__(self) -> None:
        if not isinstance(self.source_plan, ColorConversionPlan):
            raise TypeError("photometric_request.source_plan must be ColorConversionPlan")
        if not isinstance(self.source_content_sha256, Sha256Digest):
            raise TypeError("photometric_request.source_content_sha256 must be Sha256Digest")
        if not isinstance(self.source_derived_sha256, Sha256Digest):
            raise TypeError("photometric_request.source_derived_sha256 must be Sha256Digest")
        if not isinstance(self.source_photometry, SourcePhotometryMetadata):
            raise TypeError(
                "photometric_request.source_photometry must be SourcePhotometryMetadata"
            )
        if not isinstance(self.factors, PhotometricNormalizationFactors):
            raise TypeError("photometric_request.factors must be PhotometricNormalizationFactors")
        if self.source_plan.request.source_photometry != self.source_photometry:
            raise ValueError("photometric_request source photometry lineage disagrees")
        if (
            self.source_plan.request.decoded_manifest.source_observation_id
            != self.source_photometry.observation_id
        ):
            raise ValueError("photometric_request observation identities disagree")


@dataclass(frozen=True, slots=True)
class PhotometricNormalizationPlan:
    request: PhotometricNormalizationRequest
    backend: PhotometricNormalizationBackend
    steps: tuple[PhotometricNormalizationStep, ...]
    exposure_scale: float
    identity: Sha256Digest

    def __post_init__(self) -> None:
        if not isinstance(self.request, PhotometricNormalizationRequest):
            raise TypeError("photometric_plan.request must be PhotometricNormalizationRequest")
        if not isinstance(self.backend, PhotometricNormalizationBackend):
            raise TypeError("photometric_plan.backend must be PhotometricNormalizationBackend")
        if self.steps != (
            PhotometricNormalizationStep.EXPOSURE_EV_SCALE,
            PhotometricNormalizationStep.WHITE_BALANCE_RGB_GAINS,
        ):
            raise ValueError("photometric_plan steps must match the reviewed reference sequence")
        if type(self.exposure_scale) is not float:
            raise TypeError("photometric_plan.exposure_scale must be float")
        if not math.isfinite(self.exposure_scale) or self.exposure_scale <= 0.0:
            raise ValueError("photometric_plan.exposure_scale must be finite and positive")
        if not isinstance(self.identity, Sha256Digest):
            raise TypeError("photometric_plan.identity must be Sha256Digest")
        if self.identity != _plan_identity(
            self.request,
            self.backend,
            self.steps,
            self.exposure_scale,
        ):
            raise ValueError("photometric_plan identity does not match request and backend")


@dataclass(frozen=True, slots=True)
class PhotometricNormalizationAssessment:
    status: PhotometricNormalizationStatus
    request: PhotometricNormalizationRequest
    plan: PhotometricNormalizationPlan | None = None
    issue: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhotometricNormalizationStatus):
            raise TypeError("photometric_assessment.status must be PhotometricNormalizationStatus")
        if not isinstance(self.request, PhotometricNormalizationRequest):
            raise TypeError(
                "photometric_assessment.request must be PhotometricNormalizationRequest"
            )
        if self.status is PhotometricNormalizationStatus.READY:
            if self.plan is None or self.issue is not None or self.plan.request != self.request:
                raise ValueError("ready photometric assessment requires matching plan and no issue")
        elif self.plan is not None or not isinstance(self.issue, str) or not self.issue.strip():
            raise ValueError(
                "unresolved or rejected photometric assessment requires issue and no plan"
            )


def _plan_identity(
    request: PhotometricNormalizationRequest,
    backend: PhotometricNormalizationBackend,
    steps: tuple[PhotometricNormalizationStep, ...],
    exposure_scale: float,
) -> Sha256Digest:
    payload = {
        "domain": "wre.photometric_normalization_plan",
        "schema_version": 1,
        "source_plan_identity": request.source_plan.identity.value,
        "source_content_sha256": request.source_content_sha256.value,
        "source_derived_sha256": request.source_derived_sha256.value,
        "source_photometry": asdict(request.source_photometry),
        "factors": asdict(request.factors),
        "backend": asdict(backend),
        "steps": [step.value for step in steps],
        "exposure_scale": exposure_scale,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return Sha256Digest(hashlib.sha256(encoded.encode("utf-8")).hexdigest())


def assess_photometric_normalization(
    request: PhotometricNormalizationRequest,
    *,
    backend: PhotometricNormalizationBackend = REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND,
) -> PhotometricNormalizationAssessment:
    """Approve explicit finite linear-light factors without inferring them from metadata."""

    if not isinstance(request, PhotometricNormalizationRequest):
        raise TypeError("photometric_request must be PhotometricNormalizationRequest")
    if not isinstance(backend, PhotometricNormalizationBackend):
        raise TypeError("photometric_backend must be PhotometricNormalizationBackend")

    def unresolved(issue: str) -> PhotometricNormalizationAssessment:
        return PhotometricNormalizationAssessment(
            status=PhotometricNormalizationStatus.UNRESOLVED,
            request=request,
            issue=issue,
        )

    def rejected(issue: str) -> PhotometricNormalizationAssessment:
        return PhotometricNormalizationAssessment(
            status=PhotometricNormalizationStatus.REJECTED,
            request=request,
            issue=issue,
        )

    source_assessment = assess_color_conversion(
        request.source_plan.request,
        backend=request.source_plan.backend,
    )
    if (
        source_assessment.status is not ColorConversionStatus.READY
        or source_assessment.plan != request.source_plan
        or request.source_plan.request.output_convention != LINEAR_SRGB_F64
    ):
        return rejected("source color conversion is not the reviewed linear-sRGB result")

    exposure_ev = request.factors.exposure_adjustment_ev
    gains = request.factors.white_balance_rgb_gains
    if exposure_ev is None or gains is None:
        return unresolved("explicit exposure adjustment and RGB white-balance gains are required")
    if not math.isfinite(exposure_ev):
        return rejected("exposure adjustment EV must be finite")
    if any(not math.isfinite(value) or value <= 0.0 for value in gains):
        return rejected("RGB white-balance gains must be finite and strictly positive")

    try:
        exposure_scale = math.pow(2.0, exposure_ev)
    except OverflowError:
        return rejected("exposure adjustment overflows binary64")
    if not math.isfinite(exposure_scale) or exposure_scale <= 0.0:
        return rejected("exposure adjustment is not representable as a positive binary64 scale")
    try:
        combined_scales = tuple(exposure_scale * gain for gain in gains)
    except OverflowError:
        return rejected("combined exposure and white-balance scale overflows binary64")
    if any(not math.isfinite(value) or value <= 0.0 for value in combined_scales):
        return rejected("combined exposure and white-balance scale is not representable")

    if backend != REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND:
        return rejected("unreviewed photometric normalization backend or precision configuration")

    steps = (
        PhotometricNormalizationStep.EXPOSURE_EV_SCALE,
        PhotometricNormalizationStep.WHITE_BALANCE_RGB_GAINS,
    )
    plan = PhotometricNormalizationPlan(
        request=request,
        backend=backend,
        steps=steps,
        exposure_scale=exposure_scale,
        identity=_plan_identity(request, backend, steps, exposure_scale),
    )
    return PhotometricNormalizationAssessment(
        status=PhotometricNormalizationStatus.READY,
        request=request,
        plan=plan,
    )
