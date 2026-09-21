from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.routing.decision as decision_module
from wre.domain import (
    AdapterCapabilityName,
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    FailureCategory,
    MediaProfile,
    ObservationId,
    QualityMode,
)
from wre.routing import (
    ROUTE_DECISION_ARTIFACT_KIND,
    FallbackAction,
    FallbackPolicy,
    FallbackRule,
    FallbackTrigger,
    RouteDecisionArtifact,
    RouteDecisionReason,
    RouteDecisionReasonCode,
    RouteEdge,
    RouteGraph,
    RouteNode,
    RouteNodeId,
    RouteQualityRequest,
    RouteResourceBudget,
    RouterInputSnapshot,
)


def _ref(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _profile(
    observation_id: str,
    evidence: tuple[ArtifactRef, ...] = (),
) -> MediaProfile:
    return MediaProfile(
        observation_id=ObservationId(observation_id),
        evidence_artifacts=evidence,
    )


def _router_input(
    *,
    profiles: tuple[MediaProfile, ...] | None = None,
    existing_artifacts: tuple[ArtifactRef, ...] = (),
    failures: tuple[FailureCategory, ...] = (),
) -> RouterInputSnapshot:
    return RouterInputSnapshot(
        quality=RouteQualityRequest(QualityMode.QUALITY),
        media_profiles=profiles or (_profile("obs:a"),),
        resource_budget=RouteResourceBudget(
            cpu_threads=8,
            ram_bytes=16_000_000_000,
            gpu_count=1,
            gpu_vram_bytes=8_000_000_000,
            scratch_storage_bytes=100_000_000_000,
        ),
        existing_artifacts=existing_artifacts,
        prior_failures=failures,
    )


def _node(node_id: str, capability: str) -> RouteNode:
    return RouteNode(
        node_id=RouteNodeId(node_id),
        required_capabilities=(AdapterCapabilityName(capability),),
    )


def _graph() -> RouteGraph:
    return RouteGraph(
        nodes=(
            _node("a", "geometry.fast"),
            _node("b", "geometry.precise"),
        ),
        edges=(RouteEdge(RouteNodeId("a"), RouteNodeId("b")),),
    )


def _fallback_policy(graph: RouteGraph) -> FallbackPolicy:
    return FallbackPolicy(
        route_graph=graph,
        rules=(
            FallbackRule(
                source_node_id=RouteNodeId("a"),
                trigger=FallbackTrigger(failure_category=FailureCategory.TIMEOUT),
                action=FallbackAction.ESCALATE,
                target_node_id=RouteNodeId("b"),
            ),
        ),
        max_transitions=2,
    )


def _decision_ref(kind: ArtifactKind = ROUTE_DECISION_ARTIFACT_KIND) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId("route-decision:fixture"),
        artifact_kind=kind,
    )


def _reason(
    code: str = "quality.mode",
    evidence_refs: tuple[ArtifactRef, ...] = (),
) -> RouteDecisionReason:
    return RouteDecisionReason(
        code=RouteDecisionReasonCode(code),
        evidence_refs=evidence_refs,
    )


def _artifact(
    *,
    router_input: RouterInputSnapshot | None = None,
    graph: RouteGraph | None = None,
    policy: FallbackPolicy | None = None,
    reasons: tuple[RouteDecisionReason, ...] | None = None,
    decision_ref: ArtifactRef | None = None,
) -> RouteDecisionArtifact:
    selected_graph = graph or _graph()
    return RouteDecisionArtifact(
        decision_ref=decision_ref or _decision_ref(),
        router_input=router_input or _router_input(),
        route_graph=selected_graph,
        fallback_policy=policy or _fallback_policy(selected_graph),
        reasons=reasons or (_reason(),),
    )


def test_route_decision_artifact_kind_is_exact_stable_value() -> None:
    assert ROUTE_DECISION_ARTIFACT_KIND == ArtifactKind("routing.route_decision")
    assert str(ROUTE_DECISION_ARTIFACT_KIND) == "routing.route_decision"


def test_route_decision_reason_code_is_immutable_hashable_and_exact_string() -> None:
    code = RouteDecisionReasonCode("profile.sparse-view")
    same = RouteDecisionReasonCode("profile.sparse-view")

    assert str(code) == "profile.sparse-view"
    assert code == same
    assert hash(code) == hash(same)
    with pytest.raises(FrozenInstanceError):
        code.value = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "invalid",
    [
        "",
        "Uppercase",
        "has space",
        "has/slash",
        ".leading",
        "a" * 129,
    ],
)
def test_route_decision_reason_code_rejects_invalid_tokens(invalid: str) -> None:
    with pytest.raises(ValueError, match="lowercase token"):
        RouteDecisionReasonCode(invalid)


def test_route_decision_reason_is_exact_frozen_contract() -> None:
    reason = _reason()

    assert tuple(field.name for field in fields(RouteDecisionReason)) == (
        "code",
        "evidence_refs",
    )
    assert reason.evidence_refs == ()
    with pytest.raises(FrozenInstanceError):
        reason.code = RouteDecisionReasonCode("other")  # type: ignore[misc]


def test_route_decision_reason_validates_canonical_evidence() -> None:
    ref_a = _ref("a", "profile.evidence")
    ref_b = _ref("b", "profile.evidence")

    assert _reason(evidence_refs=(ref_a, ref_b)).evidence_refs == (ref_a, ref_b)

    with pytest.raises(TypeError, match="code must be RouteDecisionReasonCode"):
        RouteDecisionReason(
            code=cast(Any, "quality.mode"),
            evidence_refs=(),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        RouteDecisionReason(
            code=RouteDecisionReasonCode("quality.mode"),
            evidence_refs=cast(Any, [ref_a]),
        )
    with pytest.raises(TypeError, match="members must be ArtifactRef"):
        _reason(evidence_refs=cast(Any, ("a",)))
    with pytest.raises(ValueError, match="must be unique"):
        _reason(evidence_refs=(ref_a, ref_a))
    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        _reason(evidence_refs=(ref_b, ref_a))

    conflicting_a = _ref("same", "kind.a")
    conflicting_b = _ref("same", "kind.b")
    with pytest.raises(ValueError, match="conflicting ArtifactKind"):
        _reason(evidence_refs=(conflicting_a, conflicting_b))


def test_route_decision_artifact_is_exact_frozen_contract() -> None:
    artifact = _artifact()

    assert tuple(field.name for field in fields(RouteDecisionArtifact)) == (
        "decision_ref",
        "router_input",
        "route_graph",
        "fallback_policy",
        "reasons",
    )
    with pytest.raises(FrozenInstanceError):
        artifact.reasons = ()  # type: ignore[misc]


def test_route_decision_artifact_rejects_wrong_core_types() -> None:
    graph = _graph()
    policy = _fallback_policy(graph)
    router_input = _router_input()
    reasons = (_reason(),)

    with pytest.raises(TypeError, match="decision_ref"):
        RouteDecisionArtifact(
            decision_ref=cast(Any, "decision"),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=reasons,
        )
    with pytest.raises(TypeError, match="router_input"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=cast(Any, "input"),
            route_graph=graph,
            fallback_policy=policy,
            reasons=reasons,
        )
    with pytest.raises(TypeError, match="route_graph"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=router_input,
            route_graph=cast(Any, "graph"),
            fallback_policy=policy,
            reasons=reasons,
        )
    with pytest.raises(TypeError, match="fallback_policy"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=cast(Any, "policy"),
            reasons=reasons,
        )


def test_route_decision_requires_exact_artifact_kind() -> None:
    with pytest.raises(ValueError, match=r"routing\.route_decision"):
        _artifact(decision_ref=_decision_ref(ArtifactKind("routing.other")))


def test_route_decision_requires_fallback_policy_for_exact_same_graph() -> None:
    graph = _graph()
    other_graph = RouteGraph(
        nodes=(
            _node("a", "geometry.fast"),
            _node("b", "geometry.precise"),
            _node("c", "geometry.other"),
        ),
        edges=(
            RouteEdge(RouteNodeId("a"), RouteNodeId("b")),
            RouteEdge(RouteNodeId("b"), RouteNodeId("c")),
        ),
    )

    with pytest.raises(ValueError, match="must equal"):
        _artifact(graph=graph, policy=_fallback_policy(other_graph))


def test_route_decision_requires_nonempty_canonical_unique_reasons() -> None:
    router_input = _router_input()
    graph = _graph()
    policy = _fallback_policy(graph)

    with pytest.raises(TypeError, match="immutable tuple"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=cast(Any, [_reason()]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=(),
        )
    with pytest.raises(TypeError, match="RouteDecisionReason"):
        RouteDecisionArtifact(
            decision_ref=_decision_ref(),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=cast(Any, ("reason",)),
        )

    first = _reason("a")
    second = _reason("b")
    assert _artifact(reasons=(first, second)).reasons == (first, second)

    with pytest.raises(ValueError, match="unique reason codes"):
        _artifact(reasons=(first, first))
    with pytest.raises(ValueError, match="canonical reason-code order"):
        _artifact(reasons=(second, first))


def test_route_decision_reason_evidence_must_come_from_router_input() -> None:
    profile_ref = _ref("a", "profile.evidence")
    existing_ref = _ref("b", "existing.artifact")
    router_input = _router_input(
        profiles=(_profile("obs:a", (profile_ref,)),),
        existing_artifacts=(existing_ref,),
    )

    reasons = (
        _reason("artifact.reuse", (existing_ref,)),
        _reason("profile.signal", (profile_ref,)),
    )
    artifact = _artifact(router_input=router_input, reasons=reasons)
    assert artifact.reasons == reasons

    foreign_ref = _ref("z", "foreign.evidence")
    with pytest.raises(ValueError, match="router input evidence"):
        _artifact(
            router_input=router_input,
            reasons=(_reason("foreign.signal", (foreign_ref,)),),
        )


def test_route_decision_preserves_exact_caller_objects() -> None:
    profile_ref = _ref("a", "profile.evidence")
    router_input = _router_input(
        profiles=(_profile("obs:a", (profile_ref,)),),
    )
    graph = _graph()
    policy = _fallback_policy(graph)
    decision_ref = _decision_ref()
    reasons = (_reason("profile.signal", (profile_ref,)),)

    artifact = RouteDecisionArtifact(
        decision_ref=decision_ref,
        router_input=router_input,
        route_graph=graph,
        fallback_policy=policy,
        reasons=reasons,
    )

    assert artifact.decision_ref is decision_ref
    assert artifact.router_input is router_input
    assert artifact.route_graph is graph
    assert artifact.fallback_policy is policy
    assert artifact.reasons is reasons
    assert artifact.reasons[0] is reasons[0]


def test_route_decision_has_no_selection_execution_or_generic_metadata_surface() -> None:
    artifact = _artifact()

    for attribute in (
        "select",
        "candidate_routes",
        "score",
        "rank",
        "weight",
        "probability",
        "winner",
        "adapter",
        "adapter_id",
        "model",
        "checkpoint",
        "execution_result",
        "timestamp",
        "explanation",
        "text",
        "metadata",
        "provenance",
        "threshold",
    ):
        assert not hasattr(artifact, attribute)


def test_route_decision_module_has_no_io_persistence_or_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "ArtifactMetadata",
        "ArtifactProducerIdentity",
        "resolve_fallback",
    }

    assert forbidden_symbols.isdisjoint(decision_module.__dict__)
