from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from wre.domain.failures import FailureCategory
from wre.domain.metrics import MetricDirection, MetricName, MetricVector
from wre.domain.quality import QualityDecision

_POLICY_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_DECISION_PRECEDENCE = {
    QualityDecision.PASS: 0,
    QualityDecision.ACCEPT_WITH_WARNINGS: 1,
    QualityDecision.RETRY: 2,
    QualityDecision.ESCALATE: 3,
    QualityDecision.UNRESOLVED: 4,
}


def _require_policy_token(value: object, context: str) -> None:
    if not isinstance(value, str) or _POLICY_TOKEN_RE.fullmatch(value) is None:
        raise ValueError(
            f"{context} must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
        )


def _require_non_pass_decision(value: object, context: str) -> None:
    if not isinstance(value, QualityDecision):
        raise TypeError(f"{context} must be QualityDecision")
    if value is QualityDecision.PASS:
        raise ValueError(f"{context} must be a non-PASS QualityDecision")


@dataclass(frozen=True, slots=True, order=True)
class QualityPolicyId:
    """Caller-supplied stable identity for one quality policy family."""

    value: str

    def __post_init__(self) -> None:
        _require_policy_token(self.value, "quality_policy_id")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True, order=True)
class QualityPolicyRevision:
    """Caller-supplied immutable revision identity for one quality policy."""

    value: str

    def __post_init__(self) -> None:
        _require_policy_token(self.value, "quality_policy_revision")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class QualityPolicyMetricRule:
    """Explicit admission and optional threshold gate for one metric."""

    metric_name: MetricName
    required: bool
    expected_direction: MetricDirection
    threshold: float | None = None
    violation_decision: QualityDecision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.metric_name, MetricName):
            raise TypeError("quality_policy_metric_rule.metric_name must be MetricName")
        if type(self.required) is not bool:
            raise TypeError("quality_policy_metric_rule.required must be bool")
        if not isinstance(self.expected_direction, MetricDirection):
            raise TypeError("quality_policy_metric_rule.expected_direction must be MetricDirection")

        if self.threshold is None:
            if self.violation_decision is not None:
                raise ValueError(
                    "quality_policy_metric_rule.violation_decision requires a threshold"
                )
            return

        if type(self.threshold) is not float:
            raise TypeError("quality_policy_metric_rule.threshold must be float when present")
        if not math.isfinite(self.threshold):
            raise ValueError("quality_policy_metric_rule.threshold must be finite")
        if self.expected_direction is MetricDirection.INFORMATIONAL:
            raise ValueError(
                "quality_policy_metric_rule informational metrics cannot define a threshold"
            )
        if self.violation_decision is None:
            raise ValueError("quality_policy_metric_rule.threshold requires violation_decision")
        _require_non_pass_decision(
            self.violation_decision,
            "quality_policy_metric_rule.violation_decision",
        )


@dataclass(frozen=True, slots=True)
class QualityPolicyFailureRule:
    """Exact stable failure-category mapping to one non-PASS quality decision."""

    failure_category: FailureCategory
    decision: QualityDecision

    def __post_init__(self) -> None:
        if not isinstance(self.failure_category, FailureCategory):
            raise TypeError("quality_policy_failure_rule.failure_category must be FailureCategory")
        _require_non_pass_decision(self.decision, "quality_policy_failure_rule.decision")


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    """Immutable versioned caller-selected quality-evaluation policy."""

    policy_id: QualityPolicyId
    revision: QualityPolicyRevision
    metric_rules: tuple[QualityPolicyMetricRule, ...]
    failure_rules: tuple[QualityPolicyFailureRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, QualityPolicyId):
            raise TypeError("quality_policy.policy_id must be QualityPolicyId")
        if not isinstance(self.revision, QualityPolicyRevision):
            raise TypeError("quality_policy.revision must be QualityPolicyRevision")
        if not isinstance(self.metric_rules, tuple):
            raise TypeError("quality_policy.metric_rules must be an immutable tuple")
        if any(not isinstance(rule, QualityPolicyMetricRule) for rule in self.metric_rules):
            raise TypeError("quality_policy.metric_rules members must be QualityPolicyMetricRule")
        if not isinstance(self.failure_rules, tuple):
            raise TypeError("quality_policy.failure_rules must be an immutable tuple")
        if any(not isinstance(rule, QualityPolicyFailureRule) for rule in self.failure_rules):
            raise TypeError("quality_policy.failure_rules members must be QualityPolicyFailureRule")

        metric_names = tuple(rule.metric_name.value for rule in self.metric_rules)
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("quality_policy.metric_rules metric names must be unique")
        if metric_names != tuple(sorted(metric_names)):
            raise ValueError("quality_policy.metric_rules must be in canonical MetricName order")

        failure_categories = tuple(rule.failure_category.value for rule in self.failure_rules)
        if len(failure_categories) != len(set(failure_categories)):
            raise ValueError("quality_policy.failure_rules failure categories must be unique")
        if failure_categories != tuple(sorted(failure_categories)):
            raise ValueError(
                "quality_policy.failure_rules must be in canonical FailureCategory order"
            )


class QualityEvaluationReasonKind(StrEnum):
    """Closed machine-readable reasons emitted by deterministic quality evaluation."""

    MISSING_REQUIRED_METRIC = "missing_required_metric"
    UNKNOWN_METRIC = "unknown_metric"
    METRIC_DIRECTION_MISMATCH = "metric_direction_mismatch"
    METRIC_THRESHOLD_VIOLATION = "metric_threshold_violation"
    MAPPED_FAILURE = "mapped_failure"
    UNMAPPED_FAILURE = "unmapped_failure"


@dataclass(frozen=True, slots=True)
class QualityEvaluationReason:
    """Typed evidence explaining one quality-policy evaluation condition."""

    kind: QualityEvaluationReasonKind
    decision: QualityDecision
    metric_name: MetricName | None = None
    failure_category: FailureCategory | None = None
    observed_value: float | None = None
    threshold: float | None = None
    expected_direction: MetricDirection | None = None
    actual_direction: MetricDirection | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, QualityEvaluationReasonKind):
            raise TypeError("quality_evaluation_reason.kind must be QualityEvaluationReasonKind")
        if not isinstance(self.decision, QualityDecision):
            raise TypeError("quality_evaluation_reason.decision must be QualityDecision")
        if self.decision is QualityDecision.PASS:
            raise ValueError("quality_evaluation_reason.decision must be non-PASS")

        if self.metric_name is not None and not isinstance(self.metric_name, MetricName):
            raise TypeError("quality_evaluation_reason.metric_name must be MetricName when present")
        if self.failure_category is not None and not isinstance(
            self.failure_category, FailureCategory
        ):
            raise TypeError(
                "quality_evaluation_reason.failure_category must be FailureCategory when present"
            )
        for field_name, value in (
            ("observed_value", self.observed_value),
            ("threshold", self.threshold),
        ):
            if value is not None and type(value) is not float:
                raise TypeError(
                    f"quality_evaluation_reason.{field_name} must be float when present"
                )
            if value is not None and not math.isfinite(value):
                raise ValueError(
                    f"quality_evaluation_reason.{field_name} must be finite when present"
                )
        for field_name, value in (
            ("expected_direction", self.expected_direction),
            ("actual_direction", self.actual_direction),
        ):
            if value is not None and not isinstance(value, MetricDirection):
                raise TypeError(
                    f"quality_evaluation_reason.{field_name} must be MetricDirection when present"
                )

        self._validate_shape()

    def _validate_shape(self) -> None:
        metric_only = {
            QualityEvaluationReasonKind.MISSING_REQUIRED_METRIC,
            QualityEvaluationReasonKind.UNKNOWN_METRIC,
        }
        if self.kind in metric_only:
            if self.decision is not QualityDecision.UNRESOLVED:
                raise ValueError(f"{self.kind.value} reason must use UNRESOLVED")
            if self.metric_name is None or any(
                value is not None
                for value in (
                    self.failure_category,
                    self.observed_value,
                    self.threshold,
                    self.expected_direction,
                    self.actual_direction,
                )
            ):
                raise ValueError(f"{self.kind.value} reason has invalid fields")
            return

        if self.kind is QualityEvaluationReasonKind.METRIC_DIRECTION_MISMATCH:
            if self.decision is not QualityDecision.UNRESOLVED:
                raise ValueError("metric_direction_mismatch reason must use UNRESOLVED")
            if (
                self.metric_name is None
                or self.expected_direction is None
                or self.actual_direction is None
                or any(
                    value is not None
                    for value in (
                        self.failure_category,
                        self.observed_value,
                        self.threshold,
                    )
                )
            ):
                raise ValueError("metric_direction_mismatch reason has invalid fields")
            if self.expected_direction is self.actual_direction:
                raise ValueError(
                    "metric_direction_mismatch reason requires different expected "
                    "and actual directions"
                )
            return

        if self.kind is QualityEvaluationReasonKind.METRIC_THRESHOLD_VIOLATION:
            if (
                self.metric_name is None
                or self.observed_value is None
                or self.threshold is None
                or self.expected_direction
                not in (
                    MetricDirection.HIGHER_IS_BETTER,
                    MetricDirection.LOWER_IS_BETTER,
                )
                or self.failure_category is not None
                or self.actual_direction is not None
            ):
                raise ValueError("metric_threshold_violation reason has invalid fields")
            if not _threshold_is_violated(
                self.observed_value,
                self.threshold,
                self.expected_direction,
            ):
                raise ValueError(
                    "metric_threshold_violation reason requires an observed value "
                    "that violates threshold"
                )
            return

        if self.kind in (
            QualityEvaluationReasonKind.MAPPED_FAILURE,
            QualityEvaluationReasonKind.UNMAPPED_FAILURE,
        ):
            if self.kind is QualityEvaluationReasonKind.UNMAPPED_FAILURE and (
                self.decision is not QualityDecision.UNRESOLVED
            ):
                raise ValueError("unmapped_failure reason must use UNRESOLVED")
            if self.failure_category is None or any(
                value is not None
                for value in (
                    self.metric_name,
                    self.observed_value,
                    self.threshold,
                    self.expected_direction,
                    self.actual_direction,
                )
            ):
                raise ValueError(f"{self.kind.value} reason has invalid fields")
            return

        raise ValueError("unsupported quality evaluation reason kind")


@dataclass(frozen=True, slots=True)
class QualityEvaluation:
    """Immutable auditable result of one pure quality-policy evaluation."""

    policy_id: QualityPolicyId
    revision: QualityPolicyRevision
    decision: QualityDecision
    reasons: tuple[QualityEvaluationReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.policy_id, QualityPolicyId):
            raise TypeError("quality_evaluation.policy_id must be QualityPolicyId")
        if not isinstance(self.revision, QualityPolicyRevision):
            raise TypeError("quality_evaluation.revision must be QualityPolicyRevision")
        if not isinstance(self.decision, QualityDecision):
            raise TypeError("quality_evaluation.decision must be QualityDecision")
        if not isinstance(self.reasons, tuple):
            raise TypeError("quality_evaluation.reasons must be an immutable tuple")
        if any(not isinstance(reason, QualityEvaluationReason) for reason in self.reasons):
            raise TypeError("quality_evaluation.reasons members must be QualityEvaluationReason")
        if self.reasons != tuple(sorted(self.reasons, key=_reason_sort_key)):
            raise ValueError("quality_evaluation.reasons must use canonical reason order")
        expected = _select_decision(self.reasons)
        if self.decision is not expected:
            raise ValueError(
                "quality_evaluation.decision must match deterministic reason precedence"
            )


def _reason_sort_key(reason: QualityEvaluationReason) -> tuple[int, str, str, str]:
    return (
        -_DECISION_PRECEDENCE[reason.decision],
        reason.kind.value,
        reason.metric_name.value if reason.metric_name is not None else "",
        reason.failure_category.value if reason.failure_category is not None else "",
    )


def _select_decision(reasons: tuple[QualityEvaluationReason, ...]) -> QualityDecision:
    if not reasons:
        return QualityDecision.PASS
    return max(reasons, key=lambda reason: _DECISION_PRECEDENCE[reason.decision]).decision


def _threshold_is_violated(
    value: float,
    threshold: float,
    direction: MetricDirection,
) -> bool:
    if direction is MetricDirection.HIGHER_IS_BETTER:
        return value < threshold
    if direction is MetricDirection.LOWER_IS_BETTER:
        return value > threshold
    raise ValueError("threshold comparison requires a directional metric")


def evaluate_quality_policy(
    policy: QualityPolicy,
    metrics: MetricVector,
    failures: tuple[FailureCategory, ...],
) -> QualityEvaluation:
    """Evaluate explicit typed evidence without performing any external action."""

    if not isinstance(policy, QualityPolicy):
        raise TypeError("policy must be QualityPolicy")
    if not isinstance(metrics, MetricVector):
        raise TypeError("metrics must be MetricVector")
    if not isinstance(failures, tuple):
        raise TypeError("failures must be an immutable tuple")
    if any(not isinstance(failure, FailureCategory) for failure in failures):
        raise TypeError("failures members must be FailureCategory")

    failure_values = tuple(failure.value for failure in failures)
    if len(failure_values) != len(set(failure_values)):
        raise ValueError("failures must be unique")
    if failure_values != tuple(sorted(failure_values)):
        raise ValueError("failures must be in canonical FailureCategory order")

    metric_rules = {rule.metric_name: rule for rule in policy.metric_rules}
    failure_rules = {rule.failure_category: rule for rule in policy.failure_rules}
    observations = {
        observation.descriptor.name: observation for observation in metrics.observations
    }

    reasons: list[QualityEvaluationReason] = []

    for rule in policy.metric_rules:
        if rule.required and rule.metric_name not in observations:
            reasons.append(
                QualityEvaluationReason(
                    kind=QualityEvaluationReasonKind.MISSING_REQUIRED_METRIC,
                    decision=QualityDecision.UNRESOLVED,
                    metric_name=rule.metric_name,
                )
            )

    for observation in metrics.observations:
        metric_name = observation.descriptor.name
        rule = metric_rules.get(metric_name)
        if rule is None:
            reasons.append(
                QualityEvaluationReason(
                    kind=QualityEvaluationReasonKind.UNKNOWN_METRIC,
                    decision=QualityDecision.UNRESOLVED,
                    metric_name=metric_name,
                )
            )
            continue

        actual_direction = observation.descriptor.direction
        if actual_direction is not rule.expected_direction:
            reasons.append(
                QualityEvaluationReason(
                    kind=QualityEvaluationReasonKind.METRIC_DIRECTION_MISMATCH,
                    decision=QualityDecision.UNRESOLVED,
                    metric_name=metric_name,
                    expected_direction=rule.expected_direction,
                    actual_direction=actual_direction,
                )
            )
            continue

        if rule.threshold is not None and _threshold_is_violated(
            observation.value,
            rule.threshold,
            rule.expected_direction,
        ):
            assert rule.violation_decision is not None
            reasons.append(
                QualityEvaluationReason(
                    kind=QualityEvaluationReasonKind.METRIC_THRESHOLD_VIOLATION,
                    decision=rule.violation_decision,
                    metric_name=metric_name,
                    observed_value=observation.value,
                    threshold=rule.threshold,
                    expected_direction=rule.expected_direction,
                )
            )

    for failure in failures:
        rule = failure_rules.get(failure)
        if rule is None:
            reasons.append(
                QualityEvaluationReason(
                    kind=QualityEvaluationReasonKind.UNMAPPED_FAILURE,
                    decision=QualityDecision.UNRESOLVED,
                    failure_category=failure,
                )
            )
            continue
        reasons.append(
            QualityEvaluationReason(
                kind=QualityEvaluationReasonKind.MAPPED_FAILURE,
                decision=rule.decision,
                failure_category=failure,
            )
        )

    canonical_reasons = tuple(sorted(reasons, key=_reason_sort_key))
    return QualityEvaluation(
        policy_id=policy.policy_id,
        revision=policy.revision,
        decision=_select_decision(canonical_reasons),
        reasons=canonical_reasons,
    )
