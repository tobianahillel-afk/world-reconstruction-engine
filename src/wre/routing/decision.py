from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain import ArtifactKind, ArtifactRef
from wre.routing.fallback import FallbackPolicy
from wre.routing.graph import RouteGraph
from wre.routing.inputs import RouterInputSnapshot

_REASON_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{0,127}$")

ROUTE_DECISION_ARTIFACT_KIND = ArtifactKind("routing.route_decision")


@dataclass(frozen=True, slots=True, order=True)
class RouteDecisionReasonCode:
    """Stable open semantic identity for one auditable route-decision reason."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _REASON_CODE_RE.fullmatch(self.value) is None:
            raise ValueError(
                "route_decision_reason_code must be a 1-128 character lowercase token using "
                "letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RouteDecisionReason:
    """One typed decision reason with optional exact input evidence references."""

    code: RouteDecisionReasonCode
    evidence_refs: tuple[ArtifactRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.code, RouteDecisionReasonCode):
            raise TypeError("route_decision_reason.code must be RouteDecisionReasonCode")
        if not isinstance(self.evidence_refs, tuple):
            raise TypeError("route_decision_reason.evidence_refs must be an immutable tuple")
        if any(not isinstance(ref, ArtifactRef) for ref in self.evidence_refs):
            raise TypeError("route_decision_reason.evidence_refs members must be ArtifactRef")

        evidence_keys = tuple(
            (ref.artifact_id.value, ref.artifact_kind.value) for ref in self.evidence_refs
        )
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("route_decision_reason.evidence_refs must be unique")
        if evidence_keys != tuple(sorted(evidence_keys)):
            raise ValueError(
                "route_decision_reason.evidence_refs must use canonical "
                "ArtifactId/ArtifactKind order"
            )

        kinds_by_id: dict[str, str] = {}
        for ref in self.evidence_refs:
            previous = kinds_by_id.setdefault(ref.artifact_id.value, ref.artifact_kind.value)
            if previous != ref.artifact_kind.value:
                raise ValueError(
                    "route_decision_reason.evidence_refs must not declare conflicting "
                    "ArtifactKind values for one ArtifactId"
                )


@dataclass(frozen=True, slots=True)
class RouteDecisionArtifact:
    """Auditable declaration of one caller-supplied route decision."""

    decision_ref: ArtifactRef
    router_input: RouterInputSnapshot
    route_graph: RouteGraph
    fallback_policy: FallbackPolicy
    reasons: tuple[RouteDecisionReason, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.decision_ref, ArtifactRef):
            raise TypeError("route_decision.decision_ref must be ArtifactRef")
        if self.decision_ref.artifact_kind != ROUTE_DECISION_ARTIFACT_KIND:
            raise ValueError(
                "route_decision.decision_ref artifact_kind must be routing.route_decision"
            )
        if not isinstance(self.router_input, RouterInputSnapshot):
            raise TypeError("route_decision.router_input must be RouterInputSnapshot")
        if not isinstance(self.route_graph, RouteGraph):
            raise TypeError("route_decision.route_graph must be RouteGraph")
        if not isinstance(self.fallback_policy, FallbackPolicy):
            raise TypeError("route_decision.fallback_policy must be FallbackPolicy")
        if self.fallback_policy.route_graph != self.route_graph:
            raise ValueError(
                "route_decision.fallback_policy route_graph must equal route_decision.route_graph"
            )
        if not isinstance(self.reasons, tuple):
            raise TypeError("route_decision.reasons must be an immutable tuple")
        if not self.reasons:
            raise ValueError("route_decision.reasons must not be empty")
        if any(not isinstance(reason, RouteDecisionReason) for reason in self.reasons):
            raise TypeError("route_decision.reasons members must be RouteDecisionReason")

        reason_codes = tuple(reason.code.value for reason in self.reasons)
        if len(reason_codes) != len(set(reason_codes)):
            raise ValueError("route_decision.reasons must use unique reason codes")
        if reason_codes != tuple(sorted(reason_codes)):
            raise ValueError("route_decision.reasons must use canonical reason-code order")

        allowed_evidence = set(self.router_input.existing_artifacts)
        for profile in self.router_input.media_profiles:
            allowed_evidence.update(profile.evidence_artifacts)

        for reason in self.reasons:
            if any(ref not in allowed_evidence for ref in reason.evidence_refs):
                raise ValueError(
                    "route_decision reason evidence_refs must come from router input evidence"
                )
