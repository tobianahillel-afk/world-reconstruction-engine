from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.routing.fallback as fallback_module
from wre.domain import AdapterCapabilityName, FailureCategory, QualityDecision
from wre.routing import (
    FallbackAction,
    FallbackPolicy,
    FallbackResolution,
    FallbackRule,
    FallbackTrigger,
    RouteEdge,
    RouteGraph,
    RouteNode,
    RouteNodeId,
    resolve_fallback,
)


def _node(node_id: str) -> RouteNode:
    return RouteNode(
        node_id=RouteNodeId(node_id),
        required_capabilities=(AdapterCapabilityName(f"cap.{node_id}"),),
    )


def _graph() -> RouteGraph:
    return RouteGraph(
        nodes=(
            _node("a"),
            _node("b"),
            _node("c"),
        ),
        edges=(
            RouteEdge(RouteNodeId("a"), RouteNodeId("b")),
            RouteEdge(RouteNodeId("b"), RouteNodeId("c")),
        ),
    )


def _failure_trigger(
    category: FailureCategory = FailureCategory.TIMEOUT,
) -> FallbackTrigger:
    return FallbackTrigger(failure_category=category)


def _quality_trigger(
    decision: QualityDecision = QualityDecision.ESCALATE,
) -> FallbackTrigger:
    return FallbackTrigger(quality_decision=decision)


def _rule(
    source: str,
    trigger: FallbackTrigger,
    action: FallbackAction,
    target: str | None,
) -> FallbackRule:
    return FallbackRule(
        source_node_id=RouteNodeId(source),
        trigger=trigger,
        action=action,
        target_node_id=RouteNodeId(target) if target is not None else None,
    )


def _policy(
    *,
    rules: tuple[FallbackRule, ...],
    max_transitions: int = 3,
) -> FallbackPolicy:
    return FallbackPolicy(
        route_graph=_graph(),
        rules=rules,
        max_transitions=max_transitions,
    )


def test_fallback_trigger_is_exact_frozen_contract() -> None:
    trigger = _failure_trigger()

    assert tuple(field.name for field in fields(FallbackTrigger)) == (
        "failure_category",
        "quality_decision",
    )
    assert trigger.failure_category is FailureCategory.TIMEOUT
    assert trigger.quality_decision is None
    with pytest.raises(FrozenInstanceError):
        trigger.failure_category = FailureCategory.CORRUPTED_MEDIA  # type: ignore[misc]


def test_fallback_trigger_requires_exactly_one_stable_source() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        FallbackTrigger()
    with pytest.raises(ValueError, match="exactly one"):
        FallbackTrigger(
            failure_category=FailureCategory.TIMEOUT,
            quality_decision=QualityDecision.RETRY,
        )
    with pytest.raises(TypeError, match="failure_category must be FailureCategory"):
        FallbackTrigger(failure_category=cast(Any, "timeout"))
    with pytest.raises(TypeError, match="quality_decision must be QualityDecision"):
        FallbackTrigger(quality_decision=cast(Any, "retry"))


@pytest.mark.parametrize(
    "decision",
    [
        QualityDecision.RETRY,
        QualityDecision.ESCALATE,
        QualityDecision.UNRESOLVED,
    ],
)
def test_fallback_trigger_allows_only_non_success_quality_outcomes(
    decision: QualityDecision,
) -> None:
    assert FallbackTrigger(quality_decision=decision).quality_decision is decision


@pytest.mark.parametrize(
    "decision",
    [
        QualityDecision.PASS,
        QualityDecision.ACCEPT_WITH_WARNINGS,
    ],
)
def test_fallback_trigger_rejects_positive_quality_outcomes(
    decision: QualityDecision,
) -> None:
    with pytest.raises(ValueError, match="RETRY ESCALATE or UNRESOLVED"):
        FallbackTrigger(quality_decision=decision)


def test_fallback_action_has_exact_closed_vocabulary() -> None:
    assert [(item.name, item.value) for item in FallbackAction] == [
        ("RETRY", "retry"),
        ("ESCALATE", "escalate"),
        ("UNRESOLVED", "unresolved"),
    ]


def test_fallback_rule_is_exact_frozen_contract() -> None:
    rule = _rule(
        "a",
        _failure_trigger(),
        FallbackAction.ESCALATE,
        "b",
    )

    assert tuple(field.name for field in fields(FallbackRule)) == (
        "source_node_id",
        "trigger",
        "action",
        "target_node_id",
    )
    with pytest.raises(FrozenInstanceError):
        rule.action = FallbackAction.UNRESOLVED  # type: ignore[misc]


def test_fallback_rule_enforces_action_target_invariants() -> None:
    trigger = _failure_trigger()

    assert _rule("a", trigger, FallbackAction.RETRY, "a").target_node_id == RouteNodeId("a")
    assert _rule("a", trigger, FallbackAction.ESCALATE, "b").target_node_id == RouteNodeId("b")
    assert _rule("a", trigger, FallbackAction.UNRESOLVED, None).target_node_id is None

    with pytest.raises(ValueError, match="RETRY target_node_id must equal"):
        _rule("a", trigger, FallbackAction.RETRY, "b")
    with pytest.raises(ValueError, match="ESCALATE requires"):
        _rule("a", trigger, FallbackAction.ESCALATE, None)
    with pytest.raises(ValueError, match="must differ"):
        _rule("a", trigger, FallbackAction.ESCALATE, "a")
    with pytest.raises(ValueError, match="UNRESOLVED requires"):
        _rule("a", trigger, FallbackAction.UNRESOLVED, "b")


def test_fallback_rule_rejects_wrong_types() -> None:
    with pytest.raises(TypeError, match="source_node_id"):
        FallbackRule(
            source_node_id=cast(Any, "a"),
            trigger=_failure_trigger(),
            action=FallbackAction.UNRESOLVED,
            target_node_id=None,
        )
    with pytest.raises(TypeError, match="trigger"):
        FallbackRule(
            source_node_id=RouteNodeId("a"),
            trigger=cast(Any, "timeout"),
            action=FallbackAction.UNRESOLVED,
            target_node_id=None,
        )
    with pytest.raises(TypeError, match="action"):
        FallbackRule(
            source_node_id=RouteNodeId("a"),
            trigger=_failure_trigger(),
            action=cast(Any, "unresolved"),
            target_node_id=None,
        )


def test_fallback_policy_is_exact_frozen_contract() -> None:
    rules = (
        _rule("a", _failure_trigger(), FallbackAction.ESCALATE, "b"),
    )
    policy = _policy(rules=rules)

    assert tuple(field.name for field in fields(FallbackPolicy)) == (
        "route_graph",
        "rules",
        "max_transitions",
    )
    assert policy.rules is rules
    with pytest.raises(FrozenInstanceError):
        policy.max_transitions = 4  # type: ignore[misc]


def test_fallback_policy_validates_bound_collection_order_and_graph_nodes() -> None:
    failure_rule = _rule("a", _failure_trigger(), FallbackAction.ESCALATE, "b")
    quality_rule = _rule("a", _quality_trigger(), FallbackAction.ESCALATE, "c")

    with pytest.raises(TypeError, match="route_graph"):
        FallbackPolicy(
            route_graph=cast(Any, "graph"),
            rules=(),
            max_transitions=1,
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        FallbackPolicy(
            route_graph=_graph(),
            rules=cast(Any, [failure_rule]),
            max_transitions=1,
        )
    with pytest.raises(TypeError, match="members must be FallbackRule"):
        FallbackPolicy(
            route_graph=_graph(),
            rules=cast(Any, ("rule",)),
            max_transitions=1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        _policy(rules=(), max_transitions=0)
    with pytest.raises(ValueError, match="positive integer"):
        _policy(rules=(), max_transitions=cast(Any, True))
    with pytest.raises(ValueError, match="unique by source node and trigger"):
        _policy(rules=(failure_rule, failure_rule))

    canonical = tuple(
        sorted(
            (failure_rule, quality_rule),
            key=lambda rule: (
                rule.source_node_id.value,
                "failure" if rule.trigger.failure_category is not None else "quality",
                (
                    rule.trigger.failure_category.value
                    if rule.trigger.failure_category is not None
                    else cast(QualityDecision, rule.trigger.quality_decision).value
                ),
            ),
        )
    )
    assert _policy(rules=canonical).rules == canonical
    with pytest.raises(ValueError, match="canonical source/trigger order"):
        _policy(rules=tuple(reversed(canonical)))

    with pytest.raises(ValueError, match="source_node_id must exist"):
        _policy(
            rules=(
                _rule(
                    "z",
                    _failure_trigger(),
                    FallbackAction.UNRESOLVED,
                    None,
                ),
            )
        )
    with pytest.raises(ValueError, match="target_node_id must exist"):
        _policy(
            rules=(
                _rule(
                    "a",
                    _failure_trigger(),
                    FallbackAction.ESCALATE,
                    "z",
                ),
            )
        )


def test_multiple_explicit_triggers_on_one_source_are_valid() -> None:
    rules = (
        _rule(
            "a",
            _failure_trigger(FailureCategory.CAMERA_AMBIGUITY),
            FallbackAction.ESCALATE,
            "b",
        ),
        _rule(
            "a",
            _failure_trigger(FailureCategory.TIMEOUT),
            FallbackAction.RETRY,
            "a",
        ),
        _rule(
            "a",
            _quality_trigger(QualityDecision.ESCALATE),
            FallbackAction.ESCALATE,
            "c",
        ),
    )

    policy = _policy(rules=rules)

    assert len(policy.rules) == 3


def test_fallback_resolution_is_exact_frozen_contract() -> None:
    resolution = FallbackResolution(
        action=FallbackAction.ESCALATE,
        target_node_id=RouteNodeId("b"),
    )

    assert tuple(field.name for field in fields(FallbackResolution)) == (
        "action",
        "target_node_id",
    )
    with pytest.raises(FrozenInstanceError):
        resolution.action = FallbackAction.RETRY  # type: ignore[misc]

    with pytest.raises(ValueError, match="UNRESOLVED requires"):
        FallbackResolution(
            action=FallbackAction.UNRESOLVED,
            target_node_id=RouteNodeId("a"),
        )
    with pytest.raises(ValueError, match="requires target_node_id"):
        FallbackResolution(
            action=FallbackAction.RETRY,
            target_node_id=None,
        )


def test_resolver_returns_exact_retry_escalate_and_unresolved_rules() -> None:
    retry_trigger = _failure_trigger(FailureCategory.TIMEOUT)
    escalate_trigger = _failure_trigger(FailureCategory.CAMERA_AMBIGUITY)
    unresolved_trigger = _quality_trigger(QualityDecision.UNRESOLVED)
    rules = (
        _rule("a", escalate_trigger, FallbackAction.ESCALATE, "b"),
        _rule("a", retry_trigger, FallbackAction.RETRY, "a"),
        _rule("b", unresolved_trigger, FallbackAction.UNRESOLVED, None),
    )
    policy = _policy(rules=rules)

    retry = resolve_fallback(policy, RouteNodeId("a"), retry_trigger, 0)
    escalate = resolve_fallback(policy, RouteNodeId("a"), escalate_trigger, 0)
    unresolved = resolve_fallback(policy, RouteNodeId("b"), unresolved_trigger, 0)

    assert retry == FallbackResolution(FallbackAction.RETRY, RouteNodeId("a"))
    assert escalate == FallbackResolution(FallbackAction.ESCALATE, RouteNodeId("b"))
    assert unresolved == FallbackResolution(FallbackAction.UNRESOLVED, None)


def test_resolver_missing_rule_and_transition_bound_fail_closed_to_unresolved() -> None:
    trigger = _failure_trigger()
    policy = _policy(
        rules=(
            _rule("a", trigger, FallbackAction.ESCALATE, "b"),
        ),
        max_transitions=2,
    )

    missing = resolve_fallback(
        policy,
        RouteNodeId("b"),
        trigger,
        0,
    )
    boundary = resolve_fallback(
        policy,
        RouteNodeId("a"),
        trigger,
        2,
    )
    above = resolve_fallback(
        policy,
        RouteNodeId("a"),
        trigger,
        3,
    )

    expected = FallbackResolution(FallbackAction.UNRESOLVED, None)
    assert missing == expected
    assert boundary == expected
    assert above == expected


def test_resolver_rejects_foreign_nodes_invalid_transition_counts_and_types() -> None:
    trigger = _failure_trigger()
    policy = _policy(rules=())

    with pytest.raises(ValueError, match="must exist"):
        resolve_fallback(policy, RouteNodeId("z"), trigger, 0)
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_fallback(policy, RouteNodeId("a"), trigger, -1)
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_fallback(policy, RouteNodeId("a"), trigger, cast(Any, True))
    with pytest.raises(TypeError, match="policy must be FallbackPolicy"):
        resolve_fallback(cast(Any, "policy"), RouteNodeId("a"), trigger, 0)
    with pytest.raises(TypeError, match="current_node_id"):
        resolve_fallback(policy, cast(Any, "a"), trigger, 0)
    with pytest.raises(TypeError, match="trigger must be FallbackTrigger"):
        resolve_fallback(policy, RouteNodeId("a"), cast(Any, "timeout"), 0)


def test_fallback_policy_has_no_execution_decision_artifact_or_truth_mutation_surface() -> None:
    policy = _policy(rules=())

    for attribute in (
        "adapter",
        "adapter_id",
        "model",
        "checkpoint",
        "schedule",
        "execute",
        "decision_artifact",
        "reason",
        "score",
        "rank",
        "winner",
        "quality_mode",
        "threshold",
        "provenance",
        "metadata",
    ):
        assert not hasattr(policy, attribute)


def test_fallback_module_has_no_io_persistence_or_external_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "FFmpegToolchain",
        "ArtifactProducerIdentity",
        "HardwareRuntimeIdentity",
    }

    assert forbidden_symbols.isdisjoint(fallback_module.__dict__)
