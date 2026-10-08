"""Deterministic diagnostic masks over exact V2L17.2/V2L17.3 photometric evidence."""

from __future__ import annotations

import hashlib
import struct

from wre.domain.observations import Sha256Digest
from wre.domain.photometric_validity import (
    PhotometricChannelMask,
    PhotometricMaskKind,
    PhotometricValidityMasks,
    PhotometricValidityRequest,
)
from wre.photometry.photometric_normalization import NormalizedLinearRgbBuffer


def build_photometric_validity_masks(
    request: PhotometricValidityRequest,
    decoded_rgb8: bytes,
    normalized: NormalizedLinearRgbBuffer,
) -> PhotometricValidityMasks:
    """Build separate endpoint-candidate and normalized-over-one masks without repair."""

    if not isinstance(request, PhotometricValidityRequest):
        raise TypeError("photometric validity reference requires PhotometricValidityRequest")
    if type(decoded_rgb8) is not bytes:
        raise TypeError("decoded RGB8 source must be immutable bytes")
    if not isinstance(normalized, NormalizedLinearRgbBuffer):
        raise TypeError("normalized source must be NormalizedLinearRgbBuffer")

    if len(decoded_rgb8) != request.source_entry.byte_length:
        raise ValueError("decoded RGB8 byte length does not match the bound source entry")
    decoded_sha256 = Sha256Digest(hashlib.sha256(decoded_rgb8).hexdigest())
    if decoded_sha256 != request.source_entry.sha256:
        raise ValueError("decoded RGB8 digest does not match the bound source entry")

    if normalized.plan != request.normalization_plan:
        raise ValueError("normalized buffer uses a foreign normalization plan")
    if normalized.content_sha256 != request.normalized_content_sha256:
        raise ValueError("normalized content digest does not match the validity request")
    if normalized.derived_sha256 != request.normalized_derived_sha256:
        raise ValueError("normalized derived identity does not match the validity request")
    if normalized.plan.request.source_plan != request.source_plan:
        raise ValueError("normalized buffer is bound to a foreign source plan")

    expected_channels = (
        request.dimensions.width_px * request.dimensions.height_px * request.channel_count
    )
    if len(decoded_rgb8) != expected_channels:
        raise ValueError("decoded RGB8 channel count does not match request dimensions")
    if len(normalized.packed_rgb_f64_be) != expected_channels * 8:
        raise ValueError("normalized channel count does not match request dimensions")

    source_low = bytes(1 if channel == 0 else 0 for channel in decoded_rgb8)
    source_high = bytes(1 if channel == 255 else 0 for channel in decoded_rgb8)
    normalized_above_one = bytes(
        1 if value > 1.0 else 0
        for (value,) in struct.iter_unpack("!d", normalized.packed_rgb_f64_be)
    )

    low_mask = PhotometricChannelMask(
        request=request,
        kind=PhotometricMaskKind.SOURCE_LOW_ENDPOINT_CANDIDATE,
        packed_mask=source_low,
    )
    high_mask = PhotometricChannelMask(
        request=request,
        kind=PhotometricMaskKind.SOURCE_HIGH_ENDPOINT_CANDIDATE,
        packed_mask=source_high,
    )
    above_mask = PhotometricChannelMask(
        request=request,
        kind=PhotometricMaskKind.NORMALIZED_ABOVE_ONE,
        packed_mask=normalized_above_one,
    )
    return PhotometricValidityMasks(
        request=request,
        source_low_endpoint_candidate=low_mask,
        source_high_endpoint_candidate=high_mask,
        normalized_above_one=above_mask,
    )
