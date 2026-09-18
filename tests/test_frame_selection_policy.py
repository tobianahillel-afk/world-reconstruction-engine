from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from inspect import signature
from typing import Any, cast

import pytest

import wre.domain.frame_selection as frame_selection_module
from wre.domain import (
    FrameSelectionPolicy,
    FrameSelectionPolicyId,
    FrameSelectionPolicyRevision,
    MetricName,
    VideoObservation,
)
from wre.ingestion.keyframes import (
    KeyframeSelectionPolicy,
    ProbedVideoFrame,
    select_keyframes,
)


def test_policy_id_is_typed_hashable_stringifiable_and_validated() -> None:
    policy_id = FrameSelectionPolicyId("adaptive.quality-diversity:v1")

    assert str(policy_id) == "adaptive.quality-diversity:v1"
    assert policy_id == FrameSelectionPolicyId("adaptive.quality-diversity:v1")
    assert hash(policy_id) == hash(FrameSelectionPolicyId("adaptive.quality-diversity:v1"))

    for value in ("", "has space", "bad/slash", "x" * 129):
        with pytest.raises(ValueError, match="frame_selection_policy_id"):
            FrameSelectionPolicyId(value)

    with pytest.raises(ValueError, match="frame_selection_policy_id"):
        FrameSelectionPolicyId(cast(Any, 123))


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
def test_policy_revision_requires_positive_exact_integer(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        FrameSelectionPolicyRevision(cast(Any, value))

    revision = FrameSelectionPolicyRevision(2)
    assert revision == FrameSelectionPolicyRevision(2)
    assert hash(revision) == hash(FrameSelectionPolicyRevision(2))


def test_empty_policy_is_exact_immutable_hashable_value() -> None:
    policy = FrameSelectionPolicy(
        policy_id=FrameSelectionPolicyId("interval-baseline"),
        revision=FrameSelectionPolicyRevision(1),
    )

    assert tuple(field.name for field in fields(FrameSelectionPolicy)) == (
        "policy_id",
        "revision",
        "required_metric_names",
        "optional_metric_names",
    )
    assert policy.required_metric_names == ()
    assert policy.optional_metric_names == ()
    assert hash(policy) == hash(
        FrameSelectionPolicy(
            policy_id=FrameSelectionPolicyId("interval-baseline"),
            revision=FrameSelectionPolicyRevision(1),
        )
    )

    with pytest.raises(FrozenInstanceError):
        policy.revision = FrameSelectionPolicyRevision(2)  # type: ignore[misc]


def test_policy_requires_typed_identity_and_revision() -> None:
    with pytest.raises(TypeError, match="policy_id"):
        FrameSelectionPolicy(
            policy_id=cast(Any, "interval-baseline"),
            revision=FrameSelectionPolicyRevision(1),
        )

    with pytest.raises(TypeError, match="revision"):
        FrameSelectionPolicy(
            policy_id=FrameSelectionPolicyId("interval-baseline"),
            revision=cast(Any, 1),
        )


@pytest.mark.parametrize("field_name", ["required_metric_names", "optional_metric_names"])
def test_metric_name_collections_must_be_immutable_typed_canonical_unique(
    field_name: str,
) -> None:
    policy_id = FrameSelectionPolicyId("quality-diversity")
    revision = FrameSelectionPolicyRevision(1)

    with pytest.raises(TypeError, match="immutable tuple"):
        FrameSelectionPolicy(
            policy_id=policy_id,
            revision=revision,
            **{field_name: cast(Any, [MetricName("media.exposure.mean_luma")])},
        )

    with pytest.raises(TypeError, match="members must be MetricName"):
        FrameSelectionPolicy(
            policy_id=policy_id,
            revision=revision,
            **{field_name: (cast(Any, "media.exposure.mean_luma"),)},
        )

    duplicate = MetricName("media.exposure.mean_luma")
    with pytest.raises(ValueError, match="must be unique"):
        FrameSelectionPolicy(
            policy_id=policy_id,
            revision=revision,
            **{field_name: (duplicate, duplicate)},
        )

    with pytest.raises(ValueError, match="canonical MetricName order"):
        FrameSelectionPolicy(
            policy_id=policy_id,
            revision=revision,
            **{
                field_name: (
                    MetricName("media.visual.grid_luma_mae"),
                    MetricName("media.exposure.mean_luma"),
                )
            },
        )


def test_required_and_optional_metric_sets_must_be_disjoint() -> None:
    shared = MetricName("media.sharpness.laplacian_variance")

    with pytest.raises(ValueError, match="must be disjoint"):
        FrameSelectionPolicy(
            policy_id=FrameSelectionPolicyId("quality-diversity"),
            revision=FrameSelectionPolicyRevision(1),
            required_metric_names=(shared,),
            optional_metric_names=(shared,),
        )


def test_policy_can_declare_current_v2l5_metric_names_without_values() -> None:
    policy = FrameSelectionPolicy(
        policy_id=FrameSelectionPolicyId("quality-diversity"),
        revision=FrameSelectionPolicyRevision(3),
        required_metric_names=(
            MetricName("media.exposure.mean_luma"),
            MetricName("media.sharpness.laplacian_variance"),
        ),
        optional_metric_names=(
            MetricName("media.temporal.grid_luma_change_mean"),
            MetricName("media.visual.grid_luma_mae"),
        ),
    )

    assert tuple(str(name) for name in policy.required_metric_names) == (
        "media.exposure.mean_luma",
        "media.sharpness.laplacian_variance",
    )
    assert tuple(str(name) for name in policy.optional_metric_names) == (
        "media.temporal.grid_luma_change_mean",
        "media.visual.grid_luma_mae",
    )
    assert not hasattr(policy, "metrics")
    assert not hasattr(policy, "metric_vector")


def test_public_contract_has_no_selection_execution_or_configuration_surface() -> None:
    assert tuple(signature(FrameSelectionPolicy).parameters) == (
        "policy_id",
        "revision",
        "required_metric_names",
        "optional_metric_names",
    )

    policy = FrameSelectionPolicy(
        policy_id=FrameSelectionPolicyId("minimal"),
        revision=FrameSelectionPolicyRevision(1),
    )

    for attribute in (
        "frames",
        "selected_frames",
        "source_path",
        "video",
        "min_interval_us",
        "max_interval_us",
        "cadence",
        "threshold",
        "weight",
        "score",
        "budget",
        "route",
        "quality_decision",
        "adapter_id",
        "configuration",
        "parameters",
        "execute",
        "selector",
    ):
        assert not hasattr(policy, attribute)

    for name in (
        "Path",
        "open",
        "subprocess",
        "select_keyframes",
        "KeyframeSelectionPolicy",
        "VideoObservation",
        "QualityDecision",
    ):
        assert name not in frame_selection_module.__dict__


def test_retained_interval_keyframe_donor_behavior_is_unchanged() -> None:
    donor_policy = KeyframeSelectionPolicy(min_interval_us=1_000_000)
    frames = (
        ProbedVideoFrame(frame_index=0, frame_time_us=0),
        ProbedVideoFrame(frame_index=1, frame_time_us=500_000),
        ProbedVideoFrame(frame_index=2, frame_time_us=1_000_000),
        ProbedVideoFrame(frame_index=3, frame_time_us=2_000_000),
    )

    assert select_keyframes(frames, donor_policy) == (
        frames[0],
        frames[2],
        frames[3],
    )
    assert tuple(field.name for field in fields(KeyframeSelectionPolicy)) == (
        "min_interval_us",
    )
    assert tuple(field.name for field in fields(VideoObservation)) == (
        "observation_id",
        "asset",
        "source",
        "received_at",
        "camera_id",
        "metadata",
        "gps",
        "capture_time",
        "kind",
    )
