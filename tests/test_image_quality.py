from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.profiling.image_quality as image_quality_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    MetricDirection,
    MetricProvenance,
    ProducerRef,
    Sha256Digest,
)
from wre.profiling import LumaRaster, evaluate_image_quality


def _provenance() -> MetricProvenance:
    return MetricProvenance(
        evaluator=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="wre.profiling.image_quality",
                version="1.0.0",
                revision="v2l5.2",
            ),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
        ),
        input_artifacts=(
            ArtifactRef(
                artifact_id=ArtifactId("artifact:decoded-luma"),
                artifact_kind=ArtifactKind("media.decoded_luma"),
            ),
        ),
    )


def _metric_values(raster: LumaRaster) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value
        for observation in evaluate_image_quality(raster, _provenance()).observations
    }


def test_luma_raster_is_immutable_and_requires_exact_dimensions_and_bytes() -> None:
    raster = LumaRaster(width=3, height=3, pixels=bytes([0] * 9))

    assert tuple(field.name for field in fields(LumaRaster)) == ("width", "height", "pixels")
    with pytest.raises(FrozenInstanceError):
        raster.width = 4  # type: ignore[misc]

    for width in (0, 2, -1):
        with pytest.raises(ValueError, match=r"width must be an integer >= 3"):
            LumaRaster(width=width, height=3, pixels=bytes(9))

    for height in (0, 2, -1):
        with pytest.raises(ValueError, match=r"height must be an integer >= 3"):
            LumaRaster(width=3, height=height, pixels=bytes(9))

    with pytest.raises(ValueError, match="width must be an integer"):
        LumaRaster(width=cast(Any, True), height=3, pixels=bytes(9))

    with pytest.raises(ValueError, match="height must be an integer"):
        LumaRaster(width=3, height=cast(Any, 3.0), pixels=bytes(9))

    with pytest.raises(TypeError, match="immutable bytes"):
        LumaRaster(width=3, height=3, pixels=cast(Any, bytearray(9)))

    with pytest.raises(ValueError, match=r"length must equal width \* height"):
        LumaRaster(width=3, height=3, pixels=bytes(8))


def test_evaluator_requires_explicit_typed_inputs() -> None:
    provenance = _provenance()
    raster = LumaRaster(width=3, height=3, pixels=bytes(9))

    with pytest.raises(TypeError, match="raster must be LumaRaster"):
        evaluate_image_quality(cast(Any, bytes(9)), provenance)

    with pytest.raises(TypeError, match="provenance must be MetricProvenance"):
        evaluate_image_quality(raster, cast(Any, None))


@pytest.mark.parametrize(
    ("sample", "mean_luma", "black_clip", "white_clip"),
    [
        (0, 0.0, 1.0, 0.0),
        (128, 128.0 / 255.0, 0.0, 0.0),
        (255, 1.0, 0.0, 1.0),
    ],
)
def test_flat_rasters_have_exact_exposure_metrics_and_zero_sharpness(
    sample: int,
    mean_luma: float,
    black_clip: float,
    white_clip: float,
) -> None:
    values = _metric_values(LumaRaster(width=3, height=3, pixels=bytes([sample] * 9)))

    assert values["media.exposure.mean_luma"] == pytest.approx(mean_luma)
    assert values["media.exposure.black_clip_fraction"] == black_clip
    assert values["media.exposure.white_clip_fraction"] == white_clip
    assert values["media.sharpness.laplacian_variance"] == 0.0


def test_structured_edge_fixture_has_exact_statistics_and_positive_sharpness() -> None:
    raster = LumaRaster(
        width=4,
        height=3,
        pixels=bytes(
            [
                0,
                0,
                0,
                0,
                0,
                255,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    values = _metric_values(raster)

    assert values["media.exposure.mean_luma"] == pytest.approx(1.0 / 12.0)
    assert values["media.exposure.black_clip_fraction"] == pytest.approx(11.0 / 12.0)
    assert values["media.exposure.white_clip_fraction"] == pytest.approx(1.0 / 12.0)
    assert values["media.sharpness.laplacian_variance"] == pytest.approx(6.25)
    assert values["media.sharpness.laplacian_variance"] > 0.0


def test_metric_vector_contract_is_exact_canonical_and_reuses_provenance() -> None:
    provenance = _provenance()
    result = evaluate_image_quality(
        LumaRaster(width=3, height=3, pixels=bytes([128] * 9)),
        provenance,
    )

    assert tuple(item.descriptor.name.value for item in result.observations) == (
        "media.exposure.black_clip_fraction",
        "media.exposure.mean_luma",
        "media.exposure.white_clip_fraction",
        "media.sharpness.laplacian_variance",
    )
    assert tuple(item.descriptor.dimension.value for item in result.observations) == (
        "exposure",
        "exposure",
        "exposure",
        "sharpness",
    )
    assert tuple(item.descriptor.unit.value for item in result.observations) == (
        "fraction",
        "normalized_luma",
        "fraction",
        "normalized_luma_squared",
    )
    assert tuple(item.descriptor.direction for item in result.observations) == (
        MetricDirection.LOWER_IS_BETTER,
        MetricDirection.INFORMATIONAL,
        MetricDirection.LOWER_IS_BETTER,
        MetricDirection.HIGHER_IS_BETTER,
    )
    assert tuple(item.descriptor.aggregation.value for item in result.observations) == (
        "fraction",
        "mean",
        "fraction",
        "variance",
    )
    assert all(item.provenance is provenance for item in result.observations)
    assert all(
        type(item.value) is float and math.isfinite(item.value)
        for item in result.observations
    )


def test_evaluation_is_deterministic_and_does_not_mutate_input() -> None:
    pixels = bytes(
        [
            0,
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
        ]
    )
    raster = LumaRaster(width=3, height=3, pixels=pixels)
    provenance = _provenance()

    first = evaluate_image_quality(raster, provenance)
    second = evaluate_image_quality(raster, provenance)

    assert first == second
    assert raster.pixels is pixels
    assert raster.pixels == pixels


def test_public_surface_contains_no_decode_threshold_routing_or_profile_mutation() -> None:
    assert tuple(signature(evaluate_image_quality).parameters) == ("raster", "provenance")
    assert image_quality_module.__dict__.get("MediaProfile") is None
    for name in (
        "decode_image",
        "decode_media",
        "open",
        "Path",
        "threshold",
        "QualityDecision",
        "QualityPolicy",
        "route",
        "select_frames",
    ):
        assert name not in image_quality_module.__dict__
