"""Routing contracts built on stable WRE domain vocabularies."""

from wre.domain import QualityMode
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
    "FallbackAction",
    "FallbackPolicy",
    "FallbackResolution",
    "FallbackRule",
    "FallbackTrigger",
    "QualityMode",
    "RouteEdge",
    "RouteGraph",
    "RouteNode",
    "RouteNodeId",
    "RouteQualityRequest",
    "RouteResourceBudget",
    "RouterInputSnapshot",
    "resolve_fallback",
]
