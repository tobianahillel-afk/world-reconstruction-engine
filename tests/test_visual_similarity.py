from __future__ import annotations

import math
from inspect import signature
from typing import Any, cast

import pytest

import wre.profiling.visual_similarity as visual_similarity_module
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
from wre.profiling import LumaRaster, evaluate_visual_similarity


def _provenance() -> MetricProvenance:
    return MetricProvenance(
        evaluator=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="wre.profiling.visual_similarity",
                version="1.0.0",
                revision="v2l5.3",
            ),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("b" * 64)),
        ),
        input_artifacts=(
            ArtifactRef(
                artifact_id=ArtifactId("artifact:luma:left"),
                artifact_kind=ArtifactKind("media.decoded_luma"),
            ),
            ArtifactRef(
                artifact_id=ArtifactId("artifact:luma:right"),
                artifact_kind=ArtifactKind("media.decoded_luma"),
            ),
        ),
    )


def _raster(samples: list[int]) -> LumaRaster:
    assert len(samples) == 64
    return LumaRaster(width=8, height=8, pixels=bytes(samples))


def _values(left: LumaRaster, right: LumaRaster) -> dict[str, float]:
    return {
        observation.descriptor.name.value: observation.value
        for observation in evaluate_visual_similarity(left, right, _provenance()).observations
    }


def test_public_evaluator_signature_and_typed_inputs_are_exact() -> None:
    assert tuple(signature(evaluate_visual_similarity).parameters) == (
        "left",
        "right",
        "provenance",
    )
    raster = _raster([0] * 64)
    provenance = _provenance()

    with pytest.raises(TypeError, match="left must be LumaRaster"):
        evaluate_visual_similarity(cast(Any, bytes(64)), raster, provenance)

    with pytest.raises(TypeError, match="right must be LumaRaster"):
        evaluate_visual_similarity(raster, cast(Any, bytes(64)), provenance)

    with pytest.raises(TypeError, match="provenance must be MetricProvenance"):
        evaluate_visual_similarity(raster, raster, cast(Any, None))


def test_identical_raster_is_deterministic_symmetric_and_exactly_zero() -> None:
    raster = _raster([(index * 29) % 256 for index in range(64)])
    provenance = _provenance()

    first = evaluate_visual_similarity(raster, raster, provenance)
    second = evaluate_visual_similarity(raster, raster, provenance)

    assert first == second
    assert tuple(item.value for item in first.observations) == (0.0, 0.0)


def test_pairwise_metrics_are_symmetric() -> None:
    left = _raster([(index * 11) % 256 for index in range(64)])
    right = _raster([(index * 23 + 7) % 256 for index in range(64)])
    provenance = _provenance()

    forward = evaluate_visual_similarity(left, right, provenance)
    reverse = evaluate_visual_similarity(right, left, provenance)

    assert tuple(item.value for item in forward.observations) == pytest.approx(
        tuple(item.value for item in reverse.observations)
    )


def test_uniform_black_and_white_prove_signals_are_complementary() -> None:
    values = _values(_raster([0] * 64), _raster([255] * 64))

    assert values["media.visual.average_hash_hamming_fraction"] == 0.0
    assert values["media.visual.grid_luma_mae"] == 1.0


def test_inverted_structural_fixture_has_exact_maximal_distances() -> None:
    left = _raster([0] * 32 + [255] * 32)
    right = _raster([255] * 32 + [0] * 32)

    values = _values(left, right)

    assert values["media.visual.average_hash_hamming_fraction"] == 1.0
    assert values["media.visual.grid_luma_mae"] == 1.0


def test_metric_vector_contract_is_canonical_informational_and_finite() -> None:
    provenance = _provenance()
    result = evaluate_visual_similarity(
        _raster([0] * 32 + [255] * 32),
        _raster([255] * 32 + [0] * 32),
        provenance,
    )

    assert tuple(item.descriptor.name.value for item in result.observations) == (
        "media.visual.average_hash_hamming_fraction",
        "media.visual.grid_luma_mae",
    )
    assert tuple(item.descriptor.dimension.value for item in result.observations) == (
        "visual_similarity",
        "visual_similarity",
    )
    assert tuple(item.descriptor.unit.value for item in result.observations) == (
        "fraction",
        "normalized_luma",
    )
    assert tuple(item.descriptor.direction for item in result.observations) == (
        MetricDirection.INFORMATIONAL,
        MetricDirection.INFORMATIONAL,
    )
    assert tuple(item.descriptor.aggregation.value for item in result.observations) == (
        "fraction",
        "mean",
    )
    assert all(item.provenance is provenance for item in result.observations)
    assert all(
        type(item.value) is float and math.isfinite(item.value) and 0.0 <= item.value <= 1.0
        for item in result.observations
    )


def test_non_eight_pixel_inputs_use_the_documented_center_sampling_rule() -> None:
    left = LumaRaster(width=3, height=3, pixels=bytes([0] * 9))
    right = LumaRaster(
        width=3,
        height=3,
        pixels=bytes(
            [
                255,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        ),
    )

    values = _values(left, right)

    assert 0.0 <= values["media.visual.average_hash_hamming_fraction"] <= 1.0
    assert 0.0 < values["media.visual.grid_luma_mae"] <= 1.0


def test_public_surface_does_not_reimplement_hashing_or_make_duplicate_decisions() -> None:
    for name in (
        "hash_file_content",
        "DuplicateObservationLookup",
        "LocalImageIngestor",
        "is_duplicate",
        "is_near_duplicate",
        "threshold",
        "QualityDecision",
        "MediaProfile",
        "route",
        "cluster",
        "select_frames",
        "Path",
        "open",
    ):
        assert name not in visual_similarity_module.__dict__
