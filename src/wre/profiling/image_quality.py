from __future__ import annotations

from dataclasses import dataclass

from wre.domain.metrics import (
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
)

_BLACK_CLIP_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.exposure.black_clip_fraction"),
    dimension=MetricDimension("exposure"),
    unit=MetricUnit("fraction"),
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("fraction"),
)
_MEAN_LUMA_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.exposure.mean_luma"),
    dimension=MetricDimension("exposure"),
    unit=MetricUnit("normalized_luma"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("mean"),
)
_WHITE_CLIP_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.exposure.white_clip_fraction"),
    dimension=MetricDimension("exposure"),
    unit=MetricUnit("fraction"),
    direction=MetricDirection.LOWER_IS_BETTER,
    aggregation=MetricAggregation("fraction"),
)
_LAPLACIAN_VARIANCE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.sharpness.laplacian_variance"),
    dimension=MetricDimension("sharpness"),
    unit=MetricUnit("normalized_luma_squared"),
    direction=MetricDirection.HIGHER_IS_BETTER,
    aggregation=MetricAggregation("variance"),
)


@dataclass(frozen=True, slots=True)
class LumaRaster:
    """Immutable decoded 8-bit luma raster supplied explicitly by the caller."""

    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width < 3:
            raise ValueError("luma_raster.width must be an integer >= 3")
        if type(self.height) is not int or self.height < 3:
            raise ValueError("luma_raster.height must be an integer >= 3")
        if type(self.pixels) is not bytes:
            raise TypeError("luma_raster.pixels must be immutable bytes")
        if len(self.pixels) != self.width * self.height:
            raise ValueError("luma_raster.pixels length must equal width * height")


def _laplacian_variance(raster: LumaRaster) -> float:
    width = raster.width
    height = raster.height
    pixels = raster.pixels
    values: list[float] = []

    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            index = row + x
            center = pixels[index] / 255.0
            laplacian = (
                4.0 * center
                - pixels[index - 1] / 255.0
                - pixels[index + 1] / 255.0
                - pixels[index - width] / 255.0
                - pixels[index + width] / 255.0
            )
            values.append(laplacian)

    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / len(values)


def evaluate_image_quality(
    raster: LumaRaster,
    provenance: MetricProvenance,
) -> MetricVector:
    """Measure deterministic luma quality without decoding media or making decisions."""

    if not isinstance(raster, LumaRaster):
        raise TypeError("image_quality.raster must be LumaRaster")
    if not isinstance(provenance, MetricProvenance):
        raise TypeError("image_quality.provenance must be MetricProvenance")

    sample_count = len(raster.pixels)
    mean_luma = sum(raster.pixels) / (255.0 * sample_count)
    black_clip_fraction = sum(sample == 0 for sample in raster.pixels) / sample_count
    white_clip_fraction = sum(sample == 255 for sample in raster.pixels) / sample_count
    laplacian_variance = _laplacian_variance(raster)

    return MetricVector(
        observations=(
            MetricObservation(
                descriptor=_BLACK_CLIP_DESCRIPTOR,
                value=float(black_clip_fraction),
                provenance=provenance,
            ),
            MetricObservation(
                descriptor=_MEAN_LUMA_DESCRIPTOR,
                value=float(mean_luma),
                provenance=provenance,
            ),
            MetricObservation(
                descriptor=_WHITE_CLIP_DESCRIPTOR,
                value=float(white_clip_fraction),
                provenance=provenance,
            ),
            MetricObservation(
                descriptor=_LAPLACIAN_VARIANCE_DESCRIPTOR,
                value=float(laplacian_variance),
                provenance=provenance,
            ),
        )
    )
