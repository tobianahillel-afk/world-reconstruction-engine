from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wre.domain import FailureCategory, QualityDecision
from wre.routing.graph import RouteGraph, RouteNodeId

_ALLOWED_QUALITY_TRIGGERS = frozenset(
    (
        QualityDecision.RETRY,
        QualityDecision.ESCALATE,
        QualityDecision.UNRESOLVED,
    )
)


@dataclass(frozen=True, slots=True)
class FallbackTrigger:
    """One explicit stable failure or non-success quality outcome."""

    failure_category: FailureCategory | None = None
    quality_decision: QualityDecision | None = None

    def __post_init__(self) -> None:
        populated = int(self.failure_category is not None) + int(
            self.quality_decision is not None
        )
        if populated != 1:
            raise ValueError(
                "fallback_trigger requires exactly one of failure_category or quality_decision"
            )
        if self.failure_category is not None and not isinstance(
            self.failure_category,
            FailureCategory,
        ):
            raise TypeError(
                "fallback_trigger.failure_category must be FailureCategory when present"
            )
        if self.quality_decision is not None:
            if not isinstance(self.quality_decision, QualityDecision):
                raise TypeError(
                    "fallback_trigger.quality_decision must be QualityDecision when present"
                )
            if self.quality_decision not in _ALLOWED_QUALITY_TRIGGERS:
                raise ValueError(
                    "fallback_trigger.quality_decision must be RETRY ESCALATE or UNRESOLVED"
                )


class FallbackAction(StrEnum):
    RETRY = "retry"
    ESCALATE = "escalate"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class FallbackRule:
    """One exact allowlisted fallback transition for one source node and trigger."""

    source_node_id: RouteNodeId
    trigger: FallbackTrigger
    action: FallbackAction
    target_node_id: RouteNodeId | None

    def __post_init__(self) -> None:
        if not isinstance(self.source_node_id, RouteNodeId):
            raise TypeError("fallback_rule.source_node_id must be RouteNodeId")
        if not isinstance(self.trigger, FallbackTrigger):
            raise TypeError("fallback_rule.trigger must be FallbackTrigger")
        if not isinstance(self.action, FallbackAction):
            raise TypeError("fallback_rule.action must be FallbackAction")
        if self.target_node_id is not None and not isinstance(
            self.target_node_id,
            RouteNodeId,
        ):
            raise TypeError("fallback_rule.target_node_id must be RouteNodeId when present")

        if self.action is FallbackAction.RETRY:
            if self.target_node_id != self.source_node_id:
                raise ValueError(
                    "fallback_rule RETRY target_node_id must equal source_node_id"
                )
        elif self.action is FallbackAction.ESCALATE:
            if self.target_node_id is None:
                raise ValueError("fallback_rule ESCALATE requires target_node_id")
            if self.target_node_id == self.source_node_id:
                raise ValueError(
                    "fallback_rule ESCALATE target_node_id must differ from source_node_id"
                )
        elif self.target_node_id is not None:
            raise ValueError("fallback_rule UNRESOLVED requires target_node_id None")


def _trigger_key(trigger: FallbackTrigger) -> tuple[str, str]:
    if trigger.failure_category is not None:
        return ("failure", trigger.failure_category.value)
    if trigger.quality_decision is None:
        raise ValueError("validated fallback trigger unexpectedly has no source")
    return ("quality", trigger.quality_decision.value)


def _rule_key(rule: FallbackRule) -> tuple[str, str, str]:
    trigger_kind, trigger_value = _trigger_key(rule.trigger)
    return (rule.source_node_id.value, trigger_kind, trigger_value)


@dataclass(frozen=True, slots=True)
class FallbackPolicy:
    """Explicit graph-bound fallback allowlist with a hard transition ceiling."""

    route_graph: RouteGraph
    rules: tuple[FallbackRule, ...]
    max_transitions: int

    def __post_init__(self) -> None:
        if not isinstance(self.route_graph, RouteGraph):
            raise TypeError("fallback_policy.route_graph must be RouteGraph")
        if not isinstance(self.rules, tuple):
            raise TypeError("fallback_policy.rules must be an immutable tuple")
        if any(not isinstance(rule, FallbackRule) for rule in self.rules):
            raise TypeError("fallback_policy.rules members must be FallbackRule")
        if (
            isinstance(self.max_transitions, bool)
            or not isinstance(self.max_transitions, int)
            or self.max_transitions <= 0
        ):
            raise ValueError("fallback_policy.max_transitions must be a positive integer")

        rule_keys = tuple(_rule_key(rule) for rule in self.rules)
        if len(rule_keys) != len(set(rule_keys)):
            raise ValueError(
                "fallback_policy.rules must be unique by source node and trigger"
            )
        if rule_keys != tuple(sorted(rule_keys)):
            raise ValueError("fallback_policy.rules must use canonical source/trigger order")

        graph_node_ids = {node.node_id for node in self.route_graph.nodes}
        for rule in self.rules:
            if rule.source_node_id not in graph_node_ids:
                raise ValueError(
                    "fallback_policy rule source_node_id must exist in route_graph"
                )
            if (
                rule.target_node_id is not None
                and rule.target_node_id not in graph_node_ids
            ):
                raise ValueError(
                    "fallback_policy rule target_node_id must exist in route_graph"
                )


@dataclass(frozen=True, slots=True)
class FallbackResolution:
    """One policy result; it does not execute or recursively follow the target."""

    action: FallbackAction
    target_node_id: RouteNodeId | None

    def __post_init__(self) -> None:
        if not isinstance(self.action, FallbackAction):
            raise TypeError("fallback_resolution.action must be FallbackAction")
        if self.target_node_id is not None and not isinstance(
            self.target_node_id,
            RouteNodeId,
        ):
            raise TypeError(
                "fallback_resolution.target_node_id must be RouteNodeId when present"
            )
        if self.action is FallbackAction.UNRESOLVED:
            if self.target_node_id is not None:
                raise ValueError(
                    "fallback_resolution UNRESOLVED requires target_node_id None"
                )
        elif self.target_node_id is None:
            raise ValueError(
                "fallback_resolution RETRY or ESCALATE requires target_node_id"
            )


def resolve_fallback(
    policy: FallbackPolicy,
    current_node_id: RouteNodeId,
    trigger: FallbackTrigger,
    transitions_used: int,
) -> FallbackResolution:
    """Resolve at most one exact allowlisted transition without executing it."""

    if not isinstance(policy, FallbackPolicy):
        raise TypeError("policy must be FallbackPolicy")
    if not isinstance(current_node_id, RouteNodeId):
        raise TypeError("current_node_id must be RouteNodeId")
    if not isinstance(trigger, FallbackTrigger):
        raise TypeError("trigger must be FallbackTrigger")
    if (
        isinstance(transitions_used, bool)
        or not isinstance(transitions_used, int)
        or transitions_used < 0
    ):
        raise ValueError("transitions_used must be a non-negative integer")

    graph_node_ids = {node.node_id for node in policy.route_graph.nodes}
    if current_node_id not in graph_node_ids:
        raise ValueError("current_node_id must exist in fallback policy route_graph")

    if transitions_used >= policy.max_transitions:
        return FallbackResolution(
            action=FallbackAction.UNRESOLVED,
            target_node_id=None,
        )

    for rule in policy.rules:
        if rule.source_node_id == current_node_id and rule.trigger == trigger:
            return FallbackResolution(
                action=rule.action,
                target_node_id=rule.target_node_id,
            )

    return FallbackResolution(
        action=FallbackAction.UNRESOLVED,
        target_node_id=None,
    )
