from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.frame_selection.quality_diversity as quality_diversity_module
from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    FrameSelectionPolicy,
    FrameSelectionPolicyId,
    FrameSelectionPolicyRevision,
    MetricAggregation,
    MetricDescriptor,
    MetricDimension,
    MetricDirection,
    MetricName,
    MetricObservation,
    MetricProvenance,
    MetricUnit,
    MetricVector,
    ProducerRef,
    Sha256Digest,
)
from wre.frame_selection import (
    AdjacentFrameDiversityEvidence,
    FrameSelectionCandidate,
    QualityDiversitySelectionConfig,
    select_quality_diversity_frames,
)
from wre.ingestion.keyframes import ProbedVideoFrame
from wre.profiling import LumaRaster, evaluate_image_quality, evaluate_visual_similarity


def _provenance(label: str = "candidate") -> MetricProvenance:
    return MetricProvenance(
        evaluator=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation=f"tests.frame_selection.{label}",
                version="1.0.0",
                revision="v2l6.3",
            ),
            configuration=ConfigurationIdentity(sha256=Sha256Digest("d" * 64)),
        ),
        input_artifacts=(
            ArtifactRef(
                artifact_id=ArtifactId(f"artifact:{label}"),
                artifact_kind=ArtifactKind("media.metric_evidence"),
            ),
        ),
    )


def _descriptor(
    name: str,
    direction: MetricDirection,
    *,
    unit: str = "unitless",
) -> MetricDescriptor:
    return MetricDescriptor(
        name=MetricName(name),
        dimension=MetricDimension("frame_selection_test"),
        unit=MetricUnit(unit),
        direction=direction,
        aggregation=MetricAggregation("scalar"),
    )


def _metric_vector(
    *items: tuple[str, float, MetricDirection],
    provenance: MetricProvenance | None = None,
) -> MetricVector:
    actual_provenance = provenance or _provenance()
    observations = tuple(
        sorted(
            (
                MetricObservation(
                    descriptor=_descriptor(name, direction),
                    value=value,
                    provenance=actual_provenance,
                )
                for name, value, direction in items
            ),
            key=lambda item: item.descriptor.name.value,
        )
    )
    return MetricVector(observations=observations)


def _candidate(
    index: int,
    time_us: int,
    *metrics: tuple[str, float, MetricDirection],
) -> FrameSelectionCandidate:
    return FrameSelectionCandidate(
        frame=ProbedVideoFrame(frame_index=index, frame_time_us=time_us),
        metrics=_metric_vector(*metrics, provenance=_provenance(f"candidate-{index}")),
    )


def _policy(*names: str, optional: tuple[str, ...] = ()) -> FrameSelectionPolicy:
    return FrameSelectionPolicy(
        policy_id=FrameSelectionPolicyId("quality-diversity-baseline"),
        revision=FrameSelectionPolicyRevision(1),
        required_metric_names=tuple(MetricName(name) for name in sorted(names)),
        optional_metric_names=tuple(MetricName(name) for name in sorted(optional)),
    )


def _adjacent(
    left: int,
    right: int,
    value: float,
    *,
    direction: MetricDirection = MetricDirection.INFORMATIONAL,
) -> AdjacentFrameDiversityEvidence:
    return AdjacentFrameDiversityEvidence(
        left_frame_index=left,
        right_frame_index=right,
        metrics=_metric_vector(
            ("media.visual.grid_luma_mae", value, direction),
            provenance=_provenance(f"pair-{left}-{right}"),
        ),
    )


def test_candidate_and_adjacent_evidence_are_exact_immutable_values() -> None:
    candidate = _candidate(
        1,
        100,
        (
            "media.sharpness.laplacian_variance",
            1.0,
            MetricDirection.HIGHER_IS_BETTER,
        ),
    )
    adjacent = _adjacent(1, 2, 0.5)

    assert tuple(field.name for field in fields(FrameSelectionCandidate)) == (
        "frame",
        "metrics",
    )
    assert tuple(field.name for field in fields(AdjacentFrameDiversityEvidence)) == (
        "left_frame_index",
        "right_frame_index",
        "metrics",
    )
    assert hash(candidate) == hash(candidate)
    assert hash(adjacent) == hash(adjacent)

    with pytest.raises(FrozenInstanceError):
        candidate.frame = ProbedVideoFrame(2, 200)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        adjacent.left_frame_index = 0  # type: ignore[misc]


def test_candidate_and_adjacent_evidence_fail_closed_on_wrong_types_and_indices() -> None:
    metrics = MetricVector(observations=())

    with pytest.raises(TypeError, match="frame must be ProbedVideoFrame"):
        FrameSelectionCandidate(frame=cast(Any, (0, 0)), metrics=metrics)
    with pytest.raises(TypeError, match="metrics must be MetricVector"):
        FrameSelectionCandidate(
            frame=ProbedVideoFrame(0, 0),
            metrics=cast(Any, ()),
        )

    for left, right in ((-1, 1), (0, -1), (1, 1), (2, 1)):
        with pytest.raises(ValueError, match="frame_index"):
            AdjacentFrameDiversityEvidence(
                left_frame_index=cast(Any, left),
                right_frame_index=cast(Any, right),
                metrics=metrics,
            )

    with pytest.raises(ValueError, match="left_frame_index"):
        AdjacentFrameDiversityEvidence(
            left_frame_index=cast(Any, True),
            right_frame_index=1,
            metrics=metrics,
        )
    with pytest.raises(TypeError, match="metrics must be MetricVector"):
        AdjacentFrameDiversityEvidence(
            left_frame_index=0,
            right_frame_index=1,
            metrics=cast(Any, ()),
        )


def test_selection_config_is_exact_immutable_and_validated() -> None:
    config = QualityDiversitySelectionConfig(
        minimum_sharpness_laplacian_variance=0.5,
        maximum_black_clip_fraction=0.2,
        maximum_white_clip_fraction=0.3,
        minimum_adjacent_grid_luma_mae=0.4,
    )

    assert tuple(field.name for field in fields(QualityDiversitySelectionConfig)) == (
        "minimum_sharpness_laplacian_variance",
        "maximum_black_clip_fraction",
        "maximum_white_clip_fraction",
        "minimum_adjacent_grid_luma_mae",
    )
    assert hash(config) == hash(config)
    with pytest.raises(FrozenInstanceError):
        config.maximum_black_clip_fraction = 0.5  # type: ignore[misc]

    with pytest.raises(ValueError, match="at least one threshold"):
        QualityDiversitySelectionConfig()

    for value in (0, True, 1, "0.5"):
        with pytest.raises(TypeError, match="must be float or None"):
            QualityDiversitySelectionConfig(
                minimum_sharpness_laplacian_variance=cast(Any, value)
            )

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="must be finite"):
            QualityDiversitySelectionConfig(
                minimum_sharpness_laplacian_variance=value
            )

    with pytest.raises(ValueError, match=r"must be >= 0\.0"):
        QualityDiversitySelectionConfig(
            minimum_sharpness_laplacian_variance=-0.01
        )

    for field_name in (
        "maximum_black_clip_fraction",
        "maximum_white_clip_fraction",
        "minimum_adjacent_grid_luma_mae",
    ):
        with pytest.raises(ValueError, match=r"within \[0.0, 1.0\]"):
            QualityDiversitySelectionConfig(**{field_name: 1.01})


def test_selector_signature_candidate_collection_and_types_are_exact() -> None:
    assert tuple(signature(select_quality_diversity_frames).parameters) == (
        "candidates",
        "policy",
        "config",
        "adjacent_diversity_evidence",
    )

    candidate = _candidate(
        0,
        0,
        (
            "media.sharpness.laplacian_variance",
            1.0,
            MetricDirection.HIGHER_IS_BETTER,
        ),
    )
    config = QualityDiversitySelectionConfig(
        minimum_sharpness_laplacian_variance=0.5
    )
    policy = _policy("media.sharpness.laplacian_variance")

    with pytest.raises(TypeError, match="immutable tuple"):
        select_quality_diversity_frames(
            cast(Any, [candidate]), policy, config
        )
    with pytest.raises(ValueError, match="must not be empty"):
        select_quality_diversity_frames((), policy, config)
    with pytest.raises(TypeError, match="members must be FrameSelectionCandidate"):
        select_quality_diversity_frames(
            (cast(Any, candidate.frame),), policy, config
        )
    with pytest.raises(TypeError, match="policy must be FrameSelectionPolicy"):
        select_quality_diversity_frames(
            (candidate,), cast(Any, "policy"), config
        )
    with pytest.raises(TypeError, match="config must be QualityDiversitySelectionConfig"):
        select_quality_diversity_frames(
            (candidate,), policy, cast(Any, {})
        )


def test_candidates_must_remain_in_source_order() -> None:
    metrics = (
        "media.sharpness.laplacian_variance",
        1.0,
        MetricDirection.HIGHER_IS_BETTER,
    )
    policy = _policy("media.sharpness.laplacian_variance")
    config = QualityDiversitySelectionConfig(
        minimum_sharpness_laplacian_variance=0.5
    )

    with pytest.raises(ValueError, match="indices must be strictly increasing"):
        select_quality_diversity_frames(
            (_candidate(1, 0, metrics), _candidate(1, 1, metrics)),
            policy,
            config,
        )
    with pytest.raises(ValueError, match="timestamps must be non-decreasing"):
        select_quality_diversity_frames(
            (_candidate(0, 10, metrics), _candidate(1, 9, metrics)),
            policy,
            config,
        )


def test_policy_must_exactly_declare_configured_metric_names() -> None:
    candidate = _candidate(
        0,
        0,
        (
            "media.sharpness.laplacian_variance",
            1.0,
            MetricDirection.HIGHER_IS_BETTER,
        ),
        (
            "media.exposure.black_clip_fraction",
            0.0,
            MetricDirection.LOWER_IS_BETTER,
        ),
    )
    config = QualityDiversitySelectionConfig(
        minimum_sharpness_laplacian_variance=0.5,
        maximum_black_clip_fraction=0.5,
    )
    expected_policy = _policy(
        "media.exposure.black_clip_fraction",
        "media.sharpness.laplacian_variance",
    )

    assert select_quality_diversity_frames(
        (candidate,), expected_policy, config
    ) == (candidate.frame,)

    with pytest.raises(ValueError, match="must exactly match"):
        select_quality_diversity_frames(
            (candidate,),
            _policy("media.sharpness.laplacian_variance"),
            config,
        )
    with pytest.raises(ValueError, match="empty optional_metric_names"):
        select_quality_diversity_frames(
            (candidate,),
            _policy(
                "media.exposure.black_clip_fraction",
                "media.sharpness.laplacian_variance",
                optional=("media.exposure.mean_luma",),
            ),
            config,
        )


def test_sharpness_threshold_is_inclusive_and_quality_only_preserves_order() -> None:
    name = "media.sharpness.laplacian_variance"
    candidates = (
        _candidate(0, 0, (name, 0.49, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(1, 100, (name, 0.50, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(2, 200, (name, 0.75, MetricDirection.HIGHER_IS_BETTER)),
    )

    selected = select_quality_diversity_frames(
        candidates,
        _policy(name),
        QualityDiversitySelectionConfig(
            minimum_sharpness_laplacian_variance=0.50
        ),
    )

    assert selected == (candidates[1].frame, candidates[2].frame)
    assert selected[0] is candidates[1].frame
    assert selected[1] is candidates[2].frame


@pytest.mark.parametrize(
    ("metric_name", "threshold"),
    [
        ("media.exposure.black_clip_fraction", 0.20),
        ("media.exposure.white_clip_fraction", 0.30),
    ],
)
def test_clipping_thresholds_are_inclusive(
    metric_name: str,
    threshold: float,
) -> None:
    candidates = (
        _candidate(
            0,
            0,
            (metric_name, threshold, MetricDirection.LOWER_IS_BETTER),
        ),
        _candidate(
            1,
            100,
            (metric_name, threshold + 0.01, MetricDirection.LOWER_IS_BETTER),
        ),
    )
    kwargs = (
        {"maximum_black_clip_fraction": threshold}
        if "black" in metric_name
        else {"maximum_white_clip_fraction": threshold}
    )

    selected = select_quality_diversity_frames(
        candidates,
        _policy(metric_name),
        QualityDiversitySelectionConfig(**kwargs),
    )

    assert selected == (candidates[0].frame,)


def test_combined_quality_thresholds_are_conjunctive() -> None:
    names = (
        "media.exposure.black_clip_fraction",
        "media.exposure.white_clip_fraction",
        "media.sharpness.laplacian_variance",
    )
    candidates = (
        _candidate(
            0,
            0,
            (names[0], 0.10, MetricDirection.LOWER_IS_BETTER),
            (names[1], 0.10, MetricDirection.LOWER_IS_BETTER),
            (names[2], 0.80, MetricDirection.HIGHER_IS_BETTER),
        ),
        _candidate(
            1,
            100,
            (names[0], 0.10, MetricDirection.LOWER_IS_BETTER),
            (names[1], 0.40, MetricDirection.LOWER_IS_BETTER),
            (names[2], 0.80, MetricDirection.HIGHER_IS_BETTER),
        ),
    )

    selected = select_quality_diversity_frames(
        candidates,
        _policy(*names),
        QualityDiversitySelectionConfig(
            minimum_sharpness_laplacian_variance=0.5,
            maximum_black_clip_fraction=0.2,
            maximum_white_clip_fraction=0.3,
        ),
    )

    assert selected == (candidates[0].frame,)


def test_adjacent_diversity_uses_fixed_source_candidate_pairs_and_inclusive_boundary() -> None:
    sharpness = "media.sharpness.laplacian_variance"
    diversity = "media.visual.grid_luma_mae"
    candidates = (
        _candidate(10, 0, (sharpness, 0.1, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(20, 100, (sharpness, 1.0, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(30, 200, (sharpness, 1.0, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(40, 300, (sharpness, 1.0, MetricDirection.HIGHER_IS_BETTER)),
    )
    evidence = (
        _adjacent(10, 20, 0.90),
        _adjacent(20, 30, 0.50),
        _adjacent(30, 40, 0.49),
    )

    selected = select_quality_diversity_frames(
        candidates,
        _policy(sharpness, diversity),
        QualityDiversitySelectionConfig(
            minimum_sharpness_laplacian_variance=0.5,
            minimum_adjacent_grid_luma_mae=0.5,
        ),
        evidence,
    )

    assert selected == (
        candidates[1].frame,
        candidates[2].frame,
    )
    assert selected[0] is candidates[1].frame
    assert selected[1] is candidates[2].frame


def test_adjacent_diversity_rejects_missing_extra_and_wrong_pairs() -> None:
    diversity = "media.visual.grid_luma_mae"
    candidates = (
        _candidate(0, 0),
        _candidate(2, 100),
        _candidate(4, 200),
    )
    policy = _policy(diversity)
    config = QualityDiversitySelectionConfig(
        minimum_adjacent_grid_luma_mae=0.5
    )

    with pytest.raises(ValueError, match="exactly one entry"):
        select_quality_diversity_frames(
            candidates,
            policy,
            config,
            (_adjacent(0, 2, 0.5),),
        )
    with pytest.raises(ValueError, match="exactly one entry"):
        select_quality_diversity_frames(
            candidates,
            policy,
            config,
            (
                _adjacent(0, 2, 0.5),
                _adjacent(2, 4, 0.5),
                _adjacent(4, 5, 0.5),
            ),
        )
    with pytest.raises(ValueError, match="match consecutive candidate"):
        select_quality_diversity_frames(
            candidates,
            policy,
            config,
            (
                _adjacent(0, 4, 0.5),
                _adjacent(2, 4, 0.5),
            ),
        )


def test_adjacent_evidence_is_rejected_when_diversity_threshold_is_disabled() -> None:
    sharpness = "media.sharpness.laplacian_variance"
    candidates = (
        _candidate(0, 0, (sharpness, 1.0, MetricDirection.HIGHER_IS_BETTER)),
        _candidate(1, 100, (sharpness, 1.0, MetricDirection.HIGHER_IS_BETTER)),
    )

    with pytest.raises(ValueError, match="must be empty"):
        select_quality_diversity_frames(
            candidates,
            _policy(sharpness),
            QualityDiversitySelectionConfig(
                minimum_sharpness_laplacian_variance=0.5
            ),
            (_adjacent(0, 1, 1.0),),
        )


def test_missing_and_direction_incompatible_metrics_fail_closed() -> None:
    sharpness = "media.sharpness.laplacian_variance"
    policy = _policy(sharpness)
    config = QualityDiversitySelectionConfig(
        minimum_sharpness_laplacian_variance=0.5
    )

    with pytest.raises(ValueError, match="metric is missing"):
        select_quality_diversity_frames(
            (_candidate(0, 0),),
            policy,
            config,
        )
    with pytest.raises(ValueError, match="incompatible direction"):
        select_quality_diversity_frames(
            (
                _candidate(
                    0,
                    0,
                    (sharpness, 1.0, MetricDirection.LOWER_IS_BETTER),
                ),
            ),
            policy,
            config,
        )


def test_metric_semantic_ranges_fail_closed() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        select_quality_diversity_frames(
            (
                _candidate(
                    0,
                    0,
                    (
                        "media.sharpness.laplacian_variance",
                        -0.1,
                        MetricDirection.HIGHER_IS_BETTER,
                    ),
                ),
            ),
            _policy("media.sharpness.laplacian_variance"),
            QualityDiversitySelectionConfig(
                minimum_sharpness_laplacian_variance=0.0
            ),
        )

    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        select_quality_diversity_frames(
            (
                _candidate(
                    0,
                    0,
                    (
                        "media.exposure.black_clip_fraction",
                        1.1,
                        MetricDirection.LOWER_IS_BETTER,
                    ),
                ),
            ),
            _policy("media.exposure.black_clip_fraction"),
            QualityDiversitySelectionConfig(
                maximum_black_clip_fraction=1.0
            ),
        )


def test_real_v2l5_metric_vectors_compose_with_selection_baseline() -> None:
    quality_provenance = _provenance("quality")
    diversity_provenance = _provenance("diversity")
    dark = LumaRaster(width=8, height=8, pixels=bytes([32] * 64))
    edge = LumaRaster(
        width=8,
        height=8,
        pixels=bytes([0] * 32 + [255] * 32),
    )

    dark_metrics = evaluate_image_quality(dark, quality_provenance)
    edge_metrics = evaluate_image_quality(edge, quality_provenance)
    diversity_metrics = evaluate_visual_similarity(
        dark,
        edge,
        diversity_provenance,
    )
    candidates = (
        FrameSelectionCandidate(
            frame=ProbedVideoFrame(frame_index=0, frame_time_us=0),
            metrics=dark_metrics,
        ),
        FrameSelectionCandidate(
            frame=ProbedVideoFrame(frame_index=1, frame_time_us=100_000),
            metrics=edge_metrics,
        ),
    )

    selected = select_quality_diversity_frames(
        candidates,
        _policy(
            "media.exposure.black_clip_fraction",
            "media.exposure.white_clip_fraction",
            "media.visual.grid_luma_mae",
        ),
        QualityDiversitySelectionConfig(
            maximum_black_clip_fraction=1.0,
            maximum_white_clip_fraction=1.0,
            minimum_adjacent_grid_luma_mae=0.1,
        ),
        (
            AdjacentFrameDiversityEvidence(
                left_frame_index=0,
                right_frame_index=1,
                metrics=diversity_metrics,
            ),
        ),
    )

    assert selected == (candidates[0].frame, candidates[1].frame)


def test_public_surface_has_no_score_budget_route_decode_or_persistence_behavior() -> None:
    for name in (
        "score",
        "rank",
        "weight",
        "max_frames",
        "frames_per_second",
        "chunk",
        "budget",
        "QualityDecision",
        "route",
        "Path",
        "open",
        "subprocess",
        "FFmpegToolchain",
        "LocalKeyframeExtractor",
        "VideoFrameObservation",
        "ArtifactRef",
        "persist",
        "registry",
    ):
        assert name not in quality_diversity_module.__dict__
