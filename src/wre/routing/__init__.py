"""Routing contracts built on stable WRE domain vocabularies."""

from wre.domain import QualityMode
from wre.routing.decision import (
    ROUTE_DECISION_ARTIFACT_KIND,
    RouteDecisionArtifact,
    RouteDecisionReason,
    RouteDecisionReasonCode,
)
from wre.routing.fallback import (
    FallbackAction,
    FallbackPolicy,
    FallbackResolution,
    FallbackRule,
    FallbackTrigger,
    resolve_fallback,
)
from wre.routing.graph import RouteEdge, RouteGraph, RouteNode, RouteNodeId
from wre.routing.inputs import RouteResourceBudget, RouterInputSnapshot
from wre.routing.quality_mode import RouteQualityRequest

__all__ = [
    "ROUTE_DECISION_ARTIFACT_KIND",
    "FallbackAction",
    "FallbackPolicy",
    "FallbackResolution",
    "FallbackRule",
    "FallbackTrigger",
    "QualityMode",
    "RouteDecisionArtifact",
    "RouteDecisionReason",
    "RouteDecisionReasonCode",
    "RouteEdge",
    "RouteGraph",
    "RouteNode",
    "RouteNodeId",
    "RouteQualityRequest",
    "RouteResourceBudget",
    "RouterInputSnapshot",
    "resolve_fallback",
]
