from __future__ import annotations

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
from wre.profiling.image_quality import LumaRaster

_GRID_SIZE = 8
_GRID_SAMPLE_COUNT = _GRID_SIZE * _GRID_SIZE

_AVERAGE_HASH_HAMMING_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.visual.average_hash_hamming_fraction"),
    dimension=MetricDimension("visual_similarity"),
    unit=MetricUnit("fraction"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("fraction"),
)
_GRID_LUMA_MAE_DESCRIPTOR = MetricDescriptor(
    name=MetricName("media.visual.grid_luma_mae"),
    dimension=MetricDimension("visual_similarity"),
    unit=MetricUnit("normalized_luma"),
    direction=MetricDirection.INFORMATIONAL,
    aggregation=MetricAggregation("mean"),
)


def _sample_grid(raster: LumaRaster) -> tuple[int, ...]:
    x_coordinates = tuple(
        ((2 * grid_index + 1) * raster.width) // (2 * _GRID_SIZE)
        for grid_index in range(_GRID_SIZE)
    )
    y_coordinates = tuple(
        ((2 * grid_index + 1) * raster.height) // (2 * _GRID_SIZE)
        for grid_index in range(_GRID_SIZE)
    )
    return tuple(raster.pixels[y * raster.width + x] for y in y_coordinates for x in x_coordinates)


def _average_hash_bits(samples: tuple[int, ...]) -> tuple[bool, ...]:
    mean = sum(samples) / _GRID_SAMPLE_COUNT
    return tuple(sample >= mean for sample in samples)


def evaluate_visual_similarity(
    left: LumaRaster,
    right: LumaRaster,
    provenance: MetricProvenance,
) -> MetricVector:
    """Measure threshold-free pairwise visual difference over explicit luma rasters."""

    if not isinstance(left, LumaRaster):
        raise TypeError("visual_similarity.left must be LumaRaster")
    if not isinstance(right, LumaRaster):
        raise TypeError("visual_similarity.right must be LumaRaster")
    if not isinstance(provenance, MetricProvenance):
        raise TypeError("visual_similarity.provenance must be MetricProvenance")

    left_samples = _sample_grid(left)
    right_samples = _sample_grid(right)

    left_hash = _average_hash_bits(left_samples)
    right_hash = _average_hash_bits(right_samples)
    hamming_fraction = (
        sum(
            left_bit != right_bit for left_bit, right_bit in zip(left_hash, right_hash, strict=True)
        )
        / _GRID_SAMPLE_COUNT
    )
    grid_luma_mae = sum(
        abs(left_sample - right_sample)
        for left_sample, right_sample in zip(left_samples, right_samples, strict=True)
    ) / (255.0 * _GRID_SAMPLE_COUNT)

    return MetricVector(
        observations=(
            MetricObservation(
                descriptor=_AVERAGE_HASH_HAMMING_DESCRIPTOR,
                value=float(hamming_fraction),
                provenance=provenance,
            ),
            MetricObservation(
                descriptor=_GRID_LUMA_MAE_DESCRIPTOR,
                value=float(grid_luma_mae),
                provenance=provenance,
            ),
        )
    )
