from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactProducerIdentity,
    ConfigurationIdentity,
    FailureCategory,
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
    QualityDecision,
    Sha256Digest,
)
from wre.domain.quality_policy import (
    QualityEvaluation,
    QualityEvaluationReason,
    QualityEvaluationReasonKind,
    QualityPolicy,
    QualityPolicyFailureRule,
    QualityPolicyId,
    QualityPolicyMetricRule,
    QualityPolicyRevision,
    evaluate_quality_policy,
)


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.quality.policy.tests",
            version="1.0.0",
            revision="policy:1",
        ),
        configuration=ConfigurationIdentity(sha256=Sha256Digest("a" * 64)),
    )


def _metric(
    name: str,
    value: float,
    *,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
) -> MetricObservation:
    return MetricObservation(
        descriptor=MetricDescriptor(
            name=MetricName(name),
            dimension=MetricDimension("quality"),
            unit=MetricUnit("scalar"),
            direction=direction,
            aggregation=MetricAggregation("value"),
        ),
        value=value,
        provenance=MetricProvenance(evaluator=_producer(), input_artifacts=()),
    )


def _metric_rule(
    name: str,
    *,
    required: bool = True,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
    threshold: float | None = 1.0,
    decision: QualityDecision | None = QualityDecision.RETRY,
) -> QualityPolicyMetricRule:
    return QualityPolicyMetricRule(
        metric_name=MetricName(name),
        required=required,
        expected_direction=direction,
        threshold=threshold,
        violation_decision=decision,
    )


def _failure_rule(
    failure: FailureCategory,
    decision: QualityDecision = QualityDecision.ESCALATE,
) -> QualityPolicyFailureRule:
    return QualityPolicyFailureRule(failure_category=failure, decision=decision)


def _policy(
    *,
    metric_rules: tuple[QualityPolicyMetricRule, ...] = (),
    failure_rules: tuple[QualityPolicyFailureRule, ...] = (),
) -> QualityPolicy:
    return QualityPolicy(
        policy_id=QualityPolicyId("quality.default"),
        revision=QualityPolicyRevision("2026-09-17"),
        metric_rules=metric_rules,
        failure_rules=failure_rules,
    )


def test_policy_identity_and_revision_are_typed_immutable_and_hashable() -> None:
    policy_id = QualityPolicyId("quality.master")
    revision = QualityPolicyRevision("r1")

    assert str(policy_id) == "quality.master"
    assert str(revision) == "r1"
    assert hash(policy_id) == hash(QualityPolicyId("quality.master"))
    assert hash(revision) == hash(QualityPolicyRevision("r1"))

    with pytest.raises(FrozenInstanceError):
        policy_id.value = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        revision.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "token_type",
    [QualityPolicyId, QualityPolicyRevision],
)
@pytest.mark.parametrize(
    "value",
    ["", " leading", ".leading", "has/slash", "has space", "a" * 129],
)
def test_policy_identity_tokens_reject_invalid_values(token_type: type[Any], value: str) -> None:
    with pytest.raises(ValueError):
        token_type(value)


def test_metric_rule_is_explicit_immutable_and_hashable() -> None:
    rule = _metric_rule("geometry.reprojection_error")

    assert hash(rule) == hash(_metric_rule("geometry.reprojection_error"))
    with pytest.raises(FrozenInstanceError):
        rule.required = False  # type: ignore[misc]


def test_metric_rule_supports_known_metric_without_threshold() -> None:
    rule = _metric_rule(
        "runtime.wall_time",
        required=False,
        threshold=None,
        decision=None,
    )

    assert rule.threshold is None
    assert rule.violation_decision is None


def test_metric_rule_rejects_invalid_threshold_contracts() -> None:
    with pytest.raises(ValueError, match="requires a threshold"):
        _metric_rule("a.metric", threshold=None, decision=QualityDecision.RETRY)
    with pytest.raises(ValueError, match="requires violation_decision"):
        _metric_rule("a.metric", threshold=1.0, decision=None)
    with pytest.raises(ValueError, match="non-PASS"):
        _metric_rule("a.metric", threshold=1.0, decision=QualityDecision.PASS)
    with pytest.raises(ValueError, match="informational metrics cannot define a threshold"):
        _metric_rule(
            "a.metric",
            direction=MetricDirection.INFORMATIONAL,
            threshold=1.0,
            decision=QualityDecision.RETRY,
        )
    with pytest.raises(ValueError, match="must be finite"):
        _metric_rule("a.metric", threshold=float("inf"))
    with pytest.raises(TypeError, match="threshold must be float"):
        _metric_rule("a.metric", threshold=cast(Any, 1))


def test_metric_rule_rejects_untyped_members() -> None:
    with pytest.raises(TypeError, match="metric_name must be MetricName"):
        QualityPolicyMetricRule(
            metric_name=cast(Any, "a.metric"),
            required=True,
            expected_direction=MetricDirection.LOWER_IS_BETTER,
        )
    with pytest.raises(TypeError, match="required must be bool"):
        QualityPolicyMetricRule(
            metric_name=MetricName("a.metric"),
            required=cast(Any, 1),
            expected_direction=MetricDirection.LOWER_IS_BETTER,
        )
    with pytest.raises(TypeError, match="expected_direction must be MetricDirection"):
        QualityPolicyMetricRule(
            metric_name=MetricName("a.metric"),
            required=True,
            expected_direction=cast(Any, "lower_is_better"),
        )


def test_failure_rule_requires_typed_failure_and_non_pass_decision() -> None:
    rule = _failure_rule(FailureCategory.TIMEOUT, QualityDecision.RETRY)

    assert rule.failure_category is FailureCategory.TIMEOUT
    assert rule.decision is QualityDecision.RETRY
    with pytest.raises(FrozenInstanceError):
        rule.decision = QualityDecision.ESCALATE  # type: ignore[misc]

    with pytest.raises(TypeError, match="failure_category must be FailureCategory"):
        QualityPolicyFailureRule(
            failure_category=cast(Any, "timeout"),
            decision=QualityDecision.RETRY,
        )
    with pytest.raises(ValueError, match="non-PASS"):
        QualityPolicyFailureRule(
            failure_category=FailureCategory.TIMEOUT,
            decision=QualityDecision.PASS,
        )


def test_policy_requires_immutable_unique_canonical_rule_collections() -> None:
    metric_a = _metric_rule("a.metric")
    metric_b = _metric_rule("b.metric")
    failure_a = _failure_rule(FailureCategory.MEMORY_EXHAUSTION)
    failure_b = _failure_rule(FailureCategory.TIMEOUT)

    policy = _policy(
        metric_rules=(metric_a, metric_b),
        failure_rules=(failure_a, failure_b),
    )
    assert policy.metric_rules == (metric_a, metric_b)
    assert policy.failure_rules == (failure_a, failure_b)

    with pytest.raises(FrozenInstanceError):
        policy.metric_rules = ()  # type: ignore[misc]
    with pytest.raises(TypeError, match="metric_rules must be an immutable tuple"):
        _policy(metric_rules=cast(Any, [metric_a]))
    with pytest.raises(TypeError, match="failure_rules must be an immutable tuple"):
        _policy(failure_rules=cast(Any, [failure_a]))
    with pytest.raises(ValueError, match="metric names must be unique"):
        _policy(metric_rules=(metric_a, metric_a))
    with pytest.raises(ValueError, match="canonical MetricName order"):
        _policy(metric_rules=(metric_b, metric_a))
    with pytest.raises(ValueError, match="failure categories must be unique"):
        _policy(failure_rules=(failure_a, failure_a))
    with pytest.raises(ValueError, match="canonical FailureCategory order"):
        _policy(failure_rules=(failure_b, failure_a))


def test_evaluator_passes_when_required_metrics_satisfy_thresholds() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule("appearance.psnr", direction=MetricDirection.HIGHER_IS_BETTER),
            _metric_rule("geometry.error", threshold=2.0),
        )
    )
    metrics = MetricVector(
        observations=(
            _metric(
                "appearance.psnr",
                1.0,
                direction=MetricDirection.HIGHER_IS_BETTER,
            ),
            _metric("geometry.error", 2.0),
        )
    )

    result = evaluate_quality_policy(policy, metrics, ())

    assert result.decision is QualityDecision.PASS
    assert result.reasons == ()
    assert result.policy_id == policy.policy_id
    assert result.revision == policy.revision


@pytest.mark.parametrize(
    "decision",
    [
        QualityDecision.ACCEPT_WITH_WARNINGS,
        QualityDecision.RETRY,
        QualityDecision.ESCALATE,
        QualityDecision.UNRESOLVED,
    ],
)
def test_threshold_violation_returns_explicit_non_pass_decision(
    decision: QualityDecision,
) -> None:
    policy = _policy(metric_rules=(_metric_rule("geometry.error", decision=decision),))
    metrics = MetricVector(observations=(_metric("geometry.error", 1.5),))

    result = evaluate_quality_policy(policy, metrics, ())

    assert result.decision is decision
    assert len(result.reasons) == 1
    reason = result.reasons[0]
    assert reason.kind is QualityEvaluationReasonKind.METRIC_THRESHOLD_VIOLATION
    assert reason.decision is decision
    assert reason.metric_name == MetricName("geometry.error")
    assert reason.observed_value == 1.5
    assert reason.threshold == 1.0
    assert reason.expected_direction is MetricDirection.LOWER_IS_BETTER


def test_higher_is_better_threshold_uses_exact_supplied_value() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule(
                "appearance.psnr",
                direction=MetricDirection.HIGHER_IS_BETTER,
                threshold=30.0,
                decision=QualityDecision.ACCEPT_WITH_WARNINGS,
            ),
        )
    )

    equal = evaluate_quality_policy(
        policy,
        MetricVector(
            observations=(
                _metric(
                    "appearance.psnr",
                    30.0,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                ),
            )
        ),
        (),
    )
    below = evaluate_quality_policy(
        policy,
        MetricVector(
            observations=(
                _metric(
                    "appearance.psnr",
                    29.999,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                ),
            )
        ),
        (),
    )

    assert equal.decision is QualityDecision.PASS
    assert below.decision is QualityDecision.ACCEPT_WITH_WARNINGS


def test_missing_required_metric_fails_closed_unresolved() -> None:
    policy = _policy(metric_rules=(_metric_rule("geometry.error"),))

    result = evaluate_quality_policy(policy, MetricVector(observations=()), ())

    assert result.decision is QualityDecision.UNRESOLVED
    assert result.reasons == (
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.MISSING_REQUIRED_METRIC,
            decision=QualityDecision.UNRESOLVED,
            metric_name=MetricName("geometry.error"),
        ),
    )


@pytest.mark.parametrize(
    "metric_name",
    ("unknown.metric", "runtime.unregistered", "geometry.future_metric"),
)
def test_unknown_supplied_metric_fails_closed_unresolved(metric_name: str) -> None:
    policy = _policy()
    metrics = MetricVector(observations=(_metric(metric_name, 1.0),))

    result = evaluate_quality_policy(policy, metrics, ())

    assert result.decision is QualityDecision.UNRESOLVED
    assert result.reasons == (
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.UNKNOWN_METRIC,
            decision=QualityDecision.UNRESOLVED,
            metric_name=MetricName(metric_name),
        ),
    )


def test_reason_rejects_false_direction_mismatch_and_false_threshold_violation() -> None:
    with pytest.raises(ValueError, match="different expected and actual directions"):
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.METRIC_DIRECTION_MISMATCH,
            decision=QualityDecision.UNRESOLVED,
            metric_name=MetricName("geometry.error"),
            expected_direction=MetricDirection.LOWER_IS_BETTER,
            actual_direction=MetricDirection.LOWER_IS_BETTER,
        )

    with pytest.raises(ValueError, match="requires an observed value that violates threshold"):
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.METRIC_THRESHOLD_VIOLATION,
            decision=QualityDecision.RETRY,
            metric_name=MetricName("geometry.error"),
            observed_value=0.5,
            threshold=1.0,
            expected_direction=MetricDirection.LOWER_IS_BETTER,
        )


def test_metric_direction_mismatch_fails_closed_without_threshold_comparison() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule(
                "appearance.psnr",
                direction=MetricDirection.HIGHER_IS_BETTER,
                threshold=30.0,
                decision=QualityDecision.RETRY,
            ),
        )
    )
    metrics = MetricVector(observations=(_metric("appearance.psnr", 1.0),))

    result = evaluate_quality_policy(policy, metrics, ())

    assert result.decision is QualityDecision.UNRESOLVED
    assert result.reasons == (
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.METRIC_DIRECTION_MISMATCH,
            decision=QualityDecision.UNRESOLVED,
            metric_name=MetricName("appearance.psnr"),
            expected_direction=MetricDirection.HIGHER_IS_BETTER,
            actual_direction=MetricDirection.LOWER_IS_BETTER,
        ),
    )


def test_mapped_failure_returns_explicit_decision_without_executing_action() -> None:
    policy = _policy(failure_rules=(_failure_rule(FailureCategory.TIMEOUT, QualityDecision.RETRY),))

    result = evaluate_quality_policy(
        policy,
        MetricVector(observations=()),
        (FailureCategory.TIMEOUT,),
    )

    assert result.decision is QualityDecision.RETRY
    assert result.reasons == (
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.MAPPED_FAILURE,
            decision=QualityDecision.RETRY,
            failure_category=FailureCategory.TIMEOUT,
        ),
    )
    assert not hasattr(result, "route")
    assert not hasattr(result, "retry_count")
    assert not hasattr(result, "fallback")


@pytest.mark.parametrize("failure", tuple(FailureCategory))
def test_every_unmapped_stable_failure_fails_closed_unresolved(
    failure: FailureCategory,
) -> None:
    result = evaluate_quality_policy(
        _policy(),
        MetricVector(observations=()),
        (failure,),
    )

    assert result.decision is QualityDecision.UNRESOLVED
    assert result.reasons == (
        QualityEvaluationReason(
            kind=QualityEvaluationReasonKind.UNMAPPED_FAILURE,
            decision=QualityDecision.UNRESOLVED,
            failure_category=failure,
        ),
    )


def test_precedence_is_unresolved_escalate_retry_warning_pass() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule(
                "a.warning",
                decision=QualityDecision.ACCEPT_WITH_WARNINGS,
            ),
            _metric_rule("b.retry", decision=QualityDecision.RETRY),
            _metric_rule("c.escalate", decision=QualityDecision.ESCALATE),
        ),
        failure_rules=(_failure_rule(FailureCategory.TIMEOUT, QualityDecision.UNRESOLVED),),
    )
    metrics = MetricVector(
        observations=(
            _metric("a.warning", 2.0),
            _metric("b.retry", 2.0),
            _metric("c.escalate", 2.0),
        )
    )

    result = evaluate_quality_policy(policy, metrics, (FailureCategory.TIMEOUT,))

    assert result.decision is QualityDecision.UNRESOLVED
    assert [reason.decision for reason in result.reasons] == [
        QualityDecision.UNRESOLVED,
        QualityDecision.ESCALATE,
        QualityDecision.RETRY,
        QualityDecision.ACCEPT_WITH_WARNINGS,
    ]


def test_unresolved_evidence_dominates_mapped_retry_and_escalate_conditions() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule(
                "a.missing",
                decision=QualityDecision.ACCEPT_WITH_WARNINGS,
            ),
            _metric_rule(
                "b.mismatch",
                direction=MetricDirection.HIGHER_IS_BETTER,
                decision=QualityDecision.RETRY,
            ),
            _metric_rule("c.escalate", decision=QualityDecision.ESCALATE),
        ),
        failure_rules=(
            _failure_rule(FailureCategory.TIMEOUT, QualityDecision.RETRY),
        ),
    )
    metrics = MetricVector(
        observations=(
            _metric("b.mismatch", 0.5),
            _metric("c.escalate", 2.0),
            _metric("z.unknown", 1.0),
        )
    )

    result = evaluate_quality_policy(
        policy,
        metrics,
        (FailureCategory.MEMORY_EXHAUSTION, FailureCategory.TIMEOUT),
    )

    assert result.decision is QualityDecision.UNRESOLVED
    assert {
        reason.kind for reason in result.reasons if reason.decision is QualityDecision.UNRESOLVED
    } == {
        QualityEvaluationReasonKind.MISSING_REQUIRED_METRIC,
        QualityEvaluationReasonKind.METRIC_DIRECTION_MISMATCH,
        QualityEvaluationReasonKind.UNKNOWN_METRIC,
        QualityEvaluationReasonKind.UNMAPPED_FAILURE,
    }
    assert any(
        reason.kind is QualityEvaluationReasonKind.METRIC_THRESHOLD_VIOLATION
        and reason.decision is QualityDecision.ESCALATE
        for reason in result.reasons
    )
    assert any(
        reason.kind is QualityEvaluationReasonKind.MAPPED_FAILURE
        and reason.decision is QualityDecision.RETRY
        for reason in result.reasons
    )


def test_optional_absent_and_informational_present_are_deterministic() -> None:
    policy = _policy(
        metric_rules=(
            _metric_rule(
                "runtime.note",
                required=False,
                direction=MetricDirection.INFORMATIONAL,
                threshold=None,
                decision=None,
            ),
        )
    )

    absent = evaluate_quality_policy(policy, MetricVector(observations=()), ())
    present = evaluate_quality_policy(
        policy,
        MetricVector(
            observations=(
                _metric(
                    "runtime.note",
                    5.0,
                    direction=MetricDirection.INFORMATIONAL,
                ),
            )
        ),
        (),
    )

    assert absent.decision is QualityDecision.PASS
    assert present.decision is QualityDecision.PASS
    assert absent.reasons == present.reasons == ()


def test_failure_input_must_be_typed_immutable_unique_and_canonical() -> None:
    policy = _policy(
        failure_rules=(
            _failure_rule(FailureCategory.MEMORY_EXHAUSTION),
            _failure_rule(FailureCategory.TIMEOUT),
        )
    )
    metrics = MetricVector(observations=())

    with pytest.raises(TypeError, match="failures must be an immutable tuple"):
        evaluate_quality_policy(policy, metrics, cast(Any, [FailureCategory.TIMEOUT]))
    with pytest.raises(TypeError, match="failures members must be FailureCategory"):
        evaluate_quality_policy(policy, metrics, cast(Any, ("timeout",)))
    with pytest.raises(ValueError, match="failures must be unique"):
        evaluate_quality_policy(
            policy,
            metrics,
            (FailureCategory.TIMEOUT, FailureCategory.TIMEOUT),
        )
    with pytest.raises(ValueError, match="canonical FailureCategory order"):
        evaluate_quality_policy(
            policy,
            metrics,
            (FailureCategory.TIMEOUT, FailureCategory.MEMORY_EXHAUSTION),
        )


def test_evaluator_rejects_untyped_policy_and_metrics() -> None:
    policy = _policy()
    metrics = MetricVector(observations=())

    with pytest.raises(TypeError, match="policy must be QualityPolicy"):
        evaluate_quality_policy(cast(Any, "policy"), metrics, ())
    with pytest.raises(TypeError, match="metrics must be MetricVector"):
        evaluate_quality_policy(policy, cast(Any, "metrics"), ())


def test_evaluation_is_immutable_and_decision_must_match_reason_precedence() -> None:
    reason = QualityEvaluationReason(
        kind=QualityEvaluationReasonKind.MAPPED_FAILURE,
        decision=QualityDecision.RETRY,
        failure_category=FailureCategory.TIMEOUT,
    )
    evaluation = QualityEvaluation(
        policy_id=QualityPolicyId("quality.default"),
        revision=QualityPolicyRevision("r1"),
        decision=QualityDecision.RETRY,
        reasons=(reason,),
    )

    with pytest.raises(FrozenInstanceError):
        evaluation.decision = QualityDecision.PASS  # type: ignore[misc]
    with pytest.raises(TypeError, match="reasons must be an immutable tuple"):
        QualityEvaluation(
            policy_id=QualityPolicyId("quality.default"),
            revision=QualityPolicyRevision("r1"),
            decision=QualityDecision.RETRY,
            reasons=cast(Any, [reason]),
        )
    with pytest.raises(ValueError, match="must match deterministic reason precedence"):
        QualityEvaluation(
            policy_id=QualityPolicyId("quality.default"),
            revision=QualityPolicyRevision("r1"),
            decision=QualityDecision.ESCALATE,
            reasons=(reason,),
        )


def test_evaluator_does_not_mutate_metric_vector_or_infer_failures() -> None:
    observation = _metric("geometry.error", 0.5)
    metrics = MetricVector(observations=(observation,))
    before = metrics.observations
    policy = _policy(metric_rules=(_metric_rule("geometry.error"),))

    result = evaluate_quality_policy(policy, metrics, ())

    assert result.decision is QualityDecision.PASS
    assert metrics.observations == before
    assert metrics.observations[0] is observation
    assert not hasattr(result, "failure_category")
    assert not hasattr(result, "benchmark_record")
