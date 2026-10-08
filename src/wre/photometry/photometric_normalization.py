"""Bounded explicit exposure/white-balance normalization over V2L17.2 linear RGB."""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

from wre.domain.observations import Sha256Digest
from wre.domain.photometric_normalization import (
    REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND,
    PhotometricNormalizationPlan,
    PhotometricNormalizationStatus,
    assess_photometric_normalization,
)
from wre.photometry.color_conventions import ConvertedLinearRgbBuffer


@dataclass(frozen=True, slots=True)
class NormalizedLinearRgbBuffer:
    """Distinct immutable linear-light result; values above 1.0 are intentionally preserved."""

    plan: PhotometricNormalizationPlan
    source_content_sha256: Sha256Digest
    source_derived_sha256: Sha256Digest
    content_sha256: Sha256Digest
    derived_sha256: Sha256Digest
    channels: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, PhotometricNormalizationPlan):
            raise TypeError("normalized_color.plan must be PhotometricNormalizationPlan")
        for name in (
            "source_content_sha256",
            "source_derived_sha256",
            "content_sha256",
            "derived_sha256",
        ):
            if not isinstance(getattr(self, name), Sha256Digest):
                raise TypeError(f"normalized_color.{name} must be Sha256Digest")
        if not isinstance(self.channels, tuple):
            raise TypeError("normalized_color.channels must be an immutable tuple")
        if any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in self.channels
        ):
            raise ValueError(
                "normalized_color.channels must be finite non-negative float64 values"
            )

        source_level = self.plan.request.source_plan.request.decoded_manifest.levels[
            self.plan.request.source_plan.request.decoded_level_index
        ]
        if len(self.channels) != source_level.width_px * source_level.height_px * 3:
            raise ValueError("normalized_color.channels do not match source dimensions")
        if self.source_content_sha256 != self.plan.request.source_content_sha256:
            raise ValueError("normalized_color source content digest mismatch")
        if self.source_derived_sha256 != self.plan.request.source_derived_sha256:
            raise ValueError("normalized_color source derived digest mismatch")

        encoded = b"".join(struct.pack("!d", channel) for channel in self.channels)
        content_digest = Sha256Digest(hashlib.sha256(encoded).hexdigest())
        if self.content_sha256 != content_digest:
            raise ValueError("normalized_color output content digest mismatch")
        derived_digest = Sha256Digest(
            hashlib.sha256(
                b"wre.photometric_normalization_result.v1\0"
                + self.plan.identity.value.encode("ascii")
                + b"\0"
                + encoded
            ).hexdigest()
        )
        if self.derived_sha256 != derived_digest:
            raise ValueError("normalized_color derived identity mismatch")


def normalize_linear_rgb_reference(
    plan: PhotometricNormalizationPlan,
    source: ConvertedLinearRgbBuffer,
) -> NormalizedLinearRgbBuffer:
    """Apply only the reviewed explicit linear-light EV scale and RGB gains."""

    if not isinstance(plan, PhotometricNormalizationPlan):
        raise TypeError("photometric reference requires a PhotometricNormalizationPlan")
    if not isinstance(source, ConvertedLinearRgbBuffer):
        raise TypeError("photometric reference source must be ConvertedLinearRgbBuffer")

    assessment = assess_photometric_normalization(plan.request, backend=plan.backend)
    if (
        assessment.status is not PhotometricNormalizationStatus.READY
        or assessment.plan != plan
        or plan.backend != REFERENCE_PHOTOMETRIC_NORMALIZATION_BACKEND
    ):
        raise ValueError("photometric normalization plan is not an approved ready conversion")

    if source.plan != plan.request.source_plan:
        raise ValueError("source color-conversion plan does not match normalization request")
    if source.content_sha256 != plan.request.source_content_sha256:
        raise ValueError("source color content digest does not match normalization request")
    if source.derived_sha256 != plan.request.source_derived_sha256:
        raise ValueError("source color derived digest does not match normalization request")

    gains = plan.request.factors.white_balance_rgb_gains
    if gains is None:
        raise ValueError("ready normalization plan unexpectedly lacks RGB gains")

    channels_list: list[float] = []
    for index, value in enumerate(source.channels):
        gain = gains[index % 3]
        normalized = value * plan.exposure_scale * gain
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                "normalized channel is not representable as finite non-negative binary64"
            )
        channels_list.append(normalized)

    channels = tuple(channels_list)
    encoded = b"".join(struct.pack("!d", value) for value in channels)
    content_sha = Sha256Digest(hashlib.sha256(encoded).hexdigest())
    derived_sha = Sha256Digest(
        hashlib.sha256(
            b"wre.photometric_normalization_result.v1\0"
            + plan.identity.value.encode("ascii")
            + b"\0"
            + encoded
        ).hexdigest()
    )
    return NormalizedLinearRgbBuffer(
        plan=plan,
        source_content_sha256=source.content_sha256,
        source_derived_sha256=source.derived_sha256,
        content_sha256=content_sha,
        derived_sha256=derived_sha,
        channels=channels,
    )
