"""Bounded IEC 61966-2-1 RGB8 to linear RGB reference, not a general ICC engine."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

from wre.domain.color_conventions import (
    REFERENCE_COLOR_BACKEND,
    ColorConversionPlan,
    ColorConversionStatus,
    assess_color_conversion,
)
from wre.domain.observations import Sha256Digest


@dataclass(frozen=True, slots=True)
class ConvertedLinearRgbBuffer:
    """Distinct immutable float64 output with source and transform identities."""

    plan: ColorConversionPlan
    source_level_sha256: Sha256Digest
    content_sha256: Sha256Digest
    derived_sha256: Sha256Digest
    channels: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ColorConversionPlan):
            raise TypeError("converted_color.plan must be ColorConversionPlan")
        for name in ("source_level_sha256", "content_sha256", "derived_sha256"):
            if not isinstance(getattr(self, name), Sha256Digest):
                raise TypeError(f"converted_color.{name} must be Sha256Digest")
        if not isinstance(self.channels, tuple) or any(
            type(value) is not float or not 0.0 <= value <= 1.0 for value in self.channels
        ):
            raise ValueError("converted_color.channels must be immutable normalized float64 values")
        level = self.plan.request.decoded_manifest.levels[self.plan.request.decoded_level_index]
        if len(self.channels) != level.width_px * level.height_px * 3:
            raise ValueError("converted_color.channels do not match decoded dimensions")
        if self.source_level_sha256 != self.plan.request.decoded_level_entry.sha256:
            raise ValueError("converted_color source level hash mismatch")
        encoded = b"".join(struct.pack("!d", channel) for channel in self.channels)
        content_digest = hashlib.sha256(encoded).hexdigest()
        if self.content_sha256 != Sha256Digest(content_digest):
            raise ValueError("converted_color output content digest mismatch")
        derived_digest = hashlib.sha256(
            b"wre.color_conversion_result.v1\0"
            + self.plan.identity.value.encode("ascii")
            + b"\0"
            + encoded
        ).hexdigest()
        if self.derived_sha256 != Sha256Digest(derived_digest):
            raise ValueError("converted_color derived identity mismatch")


def _srgb_to_linear(channel: int) -> float:
    """Standard sRGB inverse transfer, with C <= 0.04045 linear segment."""

    value = channel / 255.0
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def convert_rgb8_to_linear_reference(
    plan: ColorConversionPlan,
    decoded_rgb8: bytes,
) -> ConvertedLinearRgbBuffer:
    """Convert one caller-verified canonical RGB8 level without changing its bytes."""

    if not isinstance(plan, ColorConversionPlan):
        raise TypeError("color reference requires a ColorConversionPlan")
    if type(decoded_rgb8) is not bytes:
        raise TypeError("decoded RGB8 source must be immutable bytes")
    assessment = assess_color_conversion(plan.request, backend=plan.backend)
    if (
        assessment.status is not ColorConversionStatus.READY
        or assessment.plan != plan
        or plan.backend != REFERENCE_COLOR_BACKEND
    ):
        raise ValueError("color reference plan is not an approved ready conversion")
    entry = plan.request.decoded_level_entry
    if len(decoded_rgb8) != entry.byte_length:
        raise ValueError("decoded RGB8 byte length does not match the bound materialization")
    if Sha256Digest(hashlib.sha256(decoded_rgb8).hexdigest()) != entry.sha256:
        raise ValueError("decoded RGB8 bytes do not match the bound level digest")

    channels = tuple(_srgb_to_linear(value) for value in decoded_rgb8)
    encoded = b"".join(struct.pack("!d", value) for value in channels)
    content_sha = Sha256Digest(hashlib.sha256(encoded).hexdigest())
    derived_sha = Sha256Digest(
        hashlib.sha256(
            b"wre.color_conversion_result.v1\0"
            + plan.identity.value.encode("ascii")
            + b"\0"
            + encoded
        ).hexdigest()
    )
    return ConvertedLinearRgbBuffer(
        plan=plan,
        source_level_sha256=entry.sha256,
        content_sha256=content_sha,
        derived_sha256=derived_sha,
        channels=channels,
    )
