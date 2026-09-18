from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wre.domain.observations import ObservationId, ObservationKind, Sha256Digest

DECODED_IMAGE_PYRAMID_KIND = "media.decoded_image_pyramid"


class DecodedImagePixelLayout(StrEnum):
    """Canonical decoded pixel byte layout."""

    RGB8_PACKED = "rgb8_packed"


class DecodedImageOrientationPolicy(StrEnum):
    """Orientation semantics for canonical decoded pixels."""

    SOURCE_PIXELS = "source_pixels"


@dataclass(frozen=True, slots=True)
class DecodedImagePyramidSpec:
    """Deterministic pyramid extent requested for one decoded source image."""

    minimum_max_edge_px: int

    def __post_init__(self) -> None:
        if type(self.minimum_max_edge_px) is not int or self.minimum_max_edge_px <= 0:
            raise ValueError("decoded_image_pyramid.minimum_max_edge_px must be a positive integer")


@dataclass(frozen=True, slots=True)
class DecodedImageLevelDescriptor:
    """One canonical raw-RGB pyramid level."""

    level_index: int
    width_px: int
    height_px: int
    relative_path: str

    def __post_init__(self) -> None:
        if type(self.level_index) is not int or self.level_index < 0:
            raise ValueError("decoded_image_level.level_index must be a non-negative integer")
        if type(self.width_px) is not int or self.width_px <= 0:
            raise ValueError("decoded_image_level.width_px must be a positive integer")
        if type(self.height_px) is not int or self.height_px <= 0:
            raise ValueError("decoded_image_level.height_px must be a positive integer")
        expected_path = f"levels/level-{self.level_index:06d}.rgb"
        if self.relative_path != expected_path:
            raise ValueError(
                "decoded_image_level.relative_path must match the canonical level path"
            )


@dataclass(frozen=True, slots=True)
class DecodedImagePyramidManifest:
    """Logical manifest for one reusable decoded-image pyramid."""

    source_observation_id: ObservationId
    source_kind: ObservationKind
    source_asset_sha256: Sha256Digest
    pixel_layout: DecodedImagePixelLayout
    orientation_policy: DecodedImageOrientationPolicy
    spec: DecodedImagePyramidSpec
    levels: tuple[DecodedImageLevelDescriptor, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_observation_id, ObservationId):
            raise TypeError("decoded_image_manifest.source_observation_id must be ObservationId")
        if not isinstance(self.source_kind, ObservationKind):
            raise TypeError("decoded_image_manifest.source_kind must be ObservationKind")
        if self.source_kind not in (ObservationKind.IMAGE, ObservationKind.VIDEO_FRAME):
            raise ValueError("decoded_image_manifest.source_kind must be image or video_frame")
        if not isinstance(self.source_asset_sha256, Sha256Digest):
            raise TypeError("decoded_image_manifest.source_asset_sha256 must be Sha256Digest")
        if not isinstance(self.pixel_layout, DecodedImagePixelLayout):
            raise TypeError("decoded_image_manifest.pixel_layout must be DecodedImagePixelLayout")
        if not isinstance(self.orientation_policy, DecodedImageOrientationPolicy):
            raise TypeError(
                "decoded_image_manifest.orientation_policy must be DecodedImageOrientationPolicy"
            )
        if not isinstance(self.spec, DecodedImagePyramidSpec):
            raise TypeError("decoded_image_manifest.spec must be DecodedImagePyramidSpec")
        if not isinstance(self.levels, tuple):
            raise TypeError("decoded_image_manifest.levels must be an immutable tuple")
        if not self.levels:
            raise ValueError("decoded_image_manifest.levels must not be empty")
        if any(not isinstance(level, DecodedImageLevelDescriptor) for level in self.levels):
            raise TypeError(
                "decoded_image_manifest.levels members must be DecodedImageLevelDescriptor"
            )

        for expected_index, level in enumerate(self.levels):
            if level.level_index != expected_index:
                raise ValueError(
                    "decoded_image_manifest level indices must be contiguous from zero"
                )
            if expected_index == 0:
                continue
            previous = self.levels[expected_index - 1]
            if max(previous.width_px, previous.height_px) <= self.spec.minimum_max_edge_px:
                raise ValueError("decoded_image_manifest contains levels after the pyramid stop")
            expected_width = max(1, previous.width_px // 2)
            expected_height = max(1, previous.height_px // 2)
            if (level.width_px, level.height_px) != (expected_width, expected_height):
                raise ValueError(
                    "decoded_image_manifest dimensions must follow deterministic floor halving"
                )

        if max(self.levels[-1].width_px, self.levels[-1].height_px) > self.spec.minimum_max_edge_px:
            raise ValueError("decoded_image_manifest must continue until the configured stop")


def build_decoded_image_level_descriptors(
    native_width_px: int,
    native_height_px: int,
    spec: DecodedImagePyramidSpec,
) -> tuple[DecodedImageLevelDescriptor, ...]:
    """Build deterministic native-plus-floor-halved pyramid dimensions."""

    if type(native_width_px) is not int or native_width_px <= 0:
        raise ValueError("native_width_px must be a positive integer")
    if type(native_height_px) is not int or native_height_px <= 0:
        raise ValueError("native_height_px must be a positive integer")
    if not isinstance(spec, DecodedImagePyramidSpec):
        raise TypeError("spec must be DecodedImagePyramidSpec")

    levels: list[DecodedImageLevelDescriptor] = []
    width = native_width_px
    height = native_height_px
    index = 0

    while True:
        levels.append(
            DecodedImageLevelDescriptor(
                level_index=index,
                width_px=width,
                height_px=height,
                relative_path=f"levels/level-{index:06d}.rgb",
            )
        )
        if max(width, height) <= spec.minimum_max_edge_px:
            break
        width = max(1, width // 2)
        height = max(1, height // 2)
        index += 1

    return tuple(levels)
