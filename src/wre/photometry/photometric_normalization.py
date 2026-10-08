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
    """Distinct immutable packed linear-light result; values above 1.0 are preserved."""

    plan: PhotometricNormalizationPlan
    source_content_sha256: Sha256Digest
    source_derived_sha256: Sha256Digest
    content_sha256: Sha256Digest
    derived_sha256: Sha256Digest
    packed_rgb_f64_be: bytes

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
        if type(self.packed_rgb_f64_be) is not bytes:
            raise TypeError("normalized_color.packed_rgb_f64_be must be immutable bytes")

        source_level = self.plan.request.source_plan.request.decoded_manifest.levels[
            self.plan.request.source_plan.request.decoded_level_index
        ]
        expected_channels = source_level.width_px * source_level.height_px * 3
        if len(self.packed_rgb_f64_be) != expected_channels * 8:
            raise ValueError("normalized_color packed buffer does not match source dimensions")
        for (value,) in struct.iter_unpack("!d", self.packed_rgb_f64_be):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "normalized_color packed buffer must contain finite non-negative float64 values"
                )

        if self.source_content_sha256 != self.plan.request.source_content_sha256:
            raise ValueError("normalized_color source content digest mismatch")
        if self.source_derived_sha256 != self.plan.request.source_derived_sha256:
            raise ValueError("normalized_color source derived digest mismatch")

        content_digest = Sha256Digest(hashlib.sha256(self.packed_rgb_f64_be).hexdigest())
        if self.content_sha256 != content_digest:
            raise ValueError("normalized_color output content digest mismatch")

        derived_hasher = hashlib.sha256()
        derived_hasher.update(b"wre.photometric_normalization_result.v1\0")
        derived_hasher.update(self.plan.identity.value.encode("ascii"))
        derived_hasher.update(b"\0")
        derived_hasher.update(self.packed_rgb_f64_be)
        if self.derived_sha256 != Sha256Digest(derived_hasher.hexdigest()):
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

    packed_mutable = bytearray(len(source.packed_rgb_f64_be))
    offset = 0
    for index, (value,) in enumerate(struct.iter_unpack("!d", source.packed_rgb_f64_be)):
        gain = gains[index % 3]
        normalized = value * plan.exposure_scale * gain
        if not math.isfinite(normalized) or normalized < 0.0:
            raise ValueError(
                "normalized channel is not representable as finite non-negative binary64"
            )
        struct.pack_into("!d", packed_mutable, offset, normalized)
        offset += 8

    packed_rgb_f64_be = bytes(packed_mutable)
    del packed_mutable

    content_sha = Sha256Digest(hashlib.sha256(packed_rgb_f64_be).hexdigest())
    derived_hasher = hashlib.sha256()
    derived_hasher.update(b"wre.photometric_normalization_result.v1\0")
    derived_hasher.update(plan.identity.value.encode("ascii"))
    derived_hasher.update(b"\0")
    derived_hasher.update(packed_rgb_f64_be)
    derived_sha = Sha256Digest(derived_hasher.hexdigest())

    return NormalizedLinearRgbBuffer(
        plan=plan,
        source_content_sha256=source.content_sha256,
        source_derived_sha256=source.derived_sha256,
        content_sha256=content_sha,
        derived_sha256=derived_sha,
        packed_rgb_f64_be=packed_rgb_f64_be,
    )
