from __future__ import annotations

from dataclasses import fields
from typing import Any, cast

import pytest

import wre.routing as routing_module
from wre.domain import (
    AdapterCapabilityName,
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    FailureCategory,
    MediaProfile,
    ObservationId,
    QualityDecision,
    QualityMode,
)
from wre.routing import (
    ROUTE_DECISION_ARTIFACT_KIND,
    FallbackAction,
    FallbackPolicy,
    FallbackResolution,
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
    resolve_fallback,
)


def _ref(identifier: str, kind: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(identifier),
        artifact_kind=ArtifactKind(kind),
    )


def _profile(
    observation_id: str,
    evidence: tuple[ArtifactRef, ...],
) -> MediaProfile:
    return MediaProfile(
        observation_id=ObservationId(observation_id),
        evidence_artifacts=evidence,
    )


def _node(node_id: str, capability: str) -> RouteNode:
    return RouteNode(
        node_id=RouteNodeId(node_id),
        required_capabilities=(AdapterCapabilityName(capability),),
    )


def _canonical_fixture() -> tuple[
    RouterInputSnapshot,
    RouteGraph,
    FallbackPolicy,
    tuple[RouteDecisionReason, ...],
    RouteDecisionArtifact,
]:
    profile_a_ref = _ref("artifact:profile-a", "profile.media")
    profile_b_ref = _ref("artifact:profile-b", "profile.media")
    existing_a = _ref("artifact:reuse-a", "retrieval.pairs")
    existing_b = _ref("artifact:reuse-b", "scene.cluster")

    router_input = RouterInputSnapshot(
        quality=RouteQualityRequest(QualityMode.QUALITY),
        media_profiles=(
            _profile("obs:a", (profile_a_ref,)),
            _profile("obs:b", (profile_b_ref,)),
        ),
        resource_budget=RouteResourceBudget(
            cpu_threads=8,
            ram_bytes=16_000_000_000,
            gpu_count=1,
            gpu_vram_bytes=8_000_000_000,
            scratch_storage_bytes=100_000_000_000,
        ),
        existing_artifacts=(existing_a, existing_b),
        prior_failures=(
            FailureCategory.CAMERA_AMBIGUITY,
            FailureCategory.TIMEOUT,
        ),
    )

    graph = RouteGraph(
        nodes=(
            _node("a", "geometry.preview"),
            _node("b", "geometry.precise"),
            _node("c", "geometry.refine"),
        ),
        edges=(
            RouteEdge(RouteNodeId("a"), RouteNodeId("b")),
            RouteEdge(RouteNodeId("b"), RouteNodeId("c")),
        ),
    )

    policy = FallbackPolicy(
        route_graph=graph,
        rules=(
            FallbackRule(
                source_node_id=RouteNodeId("a"),
                trigger=FallbackTrigger(
                    failure_category=FailureCategory.CAMERA_AMBIGUITY,
                ),
                action=FallbackAction.ESCALATE,
                target_node_id=RouteNodeId("b"),
            ),
            FallbackRule(
                source_node_id=RouteNodeId("b"),
                trigger=FallbackTrigger(
                    quality_decision=QualityDecision.ESCALATE,
                ),
                action=FallbackAction.ESCALATE,
                target_node_id=RouteNodeId("c"),
            ),
        ),
        max_transitions=2,
    )

    reasons = (
        RouteDecisionReason(
            code=RouteDecisionReasonCode("artifact.reuse"),
            evidence_refs=(existing_a, existing_b),
        ),
        RouteDecisionReason(
            code=RouteDecisionReasonCode("profile.signal"),
            evidence_refs=(profile_a_ref, profile_b_ref),
        ),
    )

    decision = RouteDecisionArtifact(
        decision_ref=_ref(
            "route-decision:fixture",
            ROUTE_DECISION_ARTIFACT_KIND.value,
        ),
        router_input=router_input,
        route_graph=graph,
        fallback_policy=policy,
        reasons=reasons,
    )
    return router_input, graph, policy, reasons, decision


def test_repeated_canonical_fixture_is_equal_and_wire_stable() -> None:
    first = _canonical_fixture()
    second = _canonical_fixture()

    assert first == second

    router_input, graph, policy, reasons, decision = first
    assert router_input.quality.quality_mode.value == "quality"
    assert tuple(failure.value for failure in router_input.prior_failures) == (
        "camera_ambiguity",
        "timeout",
    )
    assert tuple(node.node_id.value for node in graph.nodes) == ("a", "b", "c")
    assert tuple((edge.source.value, edge.target.value) for edge in graph.edges) == (
        ("a", "b"),
        ("b", "c"),
    )
    assert tuple(rule.action.value for rule in policy.rules) == (
        "escalate",
        "escalate",
    )
    assert tuple(reason.code.value for reason in reasons) == (
        "artifact.reuse",
        "profile.signal",
    )
    assert decision.decision_ref.artifact_kind.value == "routing.route_decision"


def test_noncanonical_router_input_collections_fail_closed() -> None:
    router_input, _, _, _, _ = _canonical_fixture()

    with pytest.raises(ValueError, match="canonical ObservationId order"):
        RouterInputSnapshot(
            quality=router_input.quality,
            media_profiles=tuple(reversed(router_input.media_profiles)),
            resource_budget=router_input.resource_budget,
            existing_artifacts=router_input.existing_artifacts,
            prior_failures=router_input.prior_failures,
        )

    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        RouterInputSnapshot(
            quality=router_input.quality,
            media_profiles=router_input.media_profiles,
            resource_budget=router_input.resource_budget,
            existing_artifacts=tuple(reversed(router_input.existing_artifacts)),
            prior_failures=router_input.prior_failures,
        )

    with pytest.raises(ValueError, match="canonical FailureCategory order"):
        RouterInputSnapshot(
            quality=router_input.quality,
            media_profiles=router_input.media_profiles,
            resource_budget=router_input.resource_budget,
            existing_artifacts=router_input.existing_artifacts,
            prior_failures=tuple(reversed(router_input.prior_failures)),
        )


def test_noncanonical_graph_policy_and_decision_collections_fail_closed() -> None:
    router_input, graph, policy, reasons, decision = _canonical_fixture()

    with pytest.raises(ValueError, match="canonical RouteNodeId order"):
        RouteGraph(
            nodes=tuple(reversed(graph.nodes)),
            edges=graph.edges,
        )

    with pytest.raises(ValueError, match="canonical source/target order"):
        RouteGraph(
            nodes=graph.nodes,
            edges=tuple(reversed(graph.edges)),
        )

    with pytest.raises(ValueError, match="canonical source/trigger order"):
        FallbackPolicy(
            route_graph=graph,
            rules=tuple(reversed(policy.rules)),
            max_transitions=policy.max_transitions,
        )

    with pytest.raises(ValueError, match="canonical reason-code order"):
        RouteDecisionArtifact(
            decision_ref=decision.decision_ref,
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=tuple(reversed(reasons)),
        )


def test_noncanonical_reason_evidence_fails_closed() -> None:
    _, _, _, reasons, _ = _canonical_fixture()

    with pytest.raises(ValueError, match="canonical ArtifactId/ArtifactKind order"):
        RouteDecisionReason(
            code=reasons[1].code,
            evidence_refs=tuple(reversed(reasons[1].evidence_refs)),
        )


def test_invalid_cross_contract_values_fail_closed() -> None:
    router_input, graph, policy, reasons, decision = _canonical_fixture()
    foreign_ref = _ref("artifact:foreign", "foreign.evidence")

    with pytest.raises(ValueError, match="router input evidence"):
        RouteDecisionArtifact(
            decision_ref=decision.decision_ref,
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=(
                RouteDecisionReason(
                    code=RouteDecisionReasonCode("foreign.evidence"),
                    evidence_refs=(foreign_ref,),
                ),
            ),
        )

    with pytest.raises(ValueError, match=r"routing\.route_decision"):
        RouteDecisionArtifact(
            decision_ref=_ref("route-decision:bad", "routing.other"),
            router_input=router_input,
            route_graph=graph,
            fallback_policy=policy,
            reasons=reasons,
        )

    other_graph = RouteGraph(
        nodes=(
            _node("a", "geometry.preview"),
            _node("b", "geometry.precise"),
        ),
        edges=(RouteEdge(RouteNodeId("a"), RouteNodeId("b")),),
    )
    with pytest.raises(ValueError, match="must equal"):
        RouteDecisionArtifact(
            decision_ref=decision.decision_ref,
            router_input=router_input,
            route_graph=graph,
            fallback_policy=FallbackPolicy(
                route_graph=other_graph,
                rules=(),
                max_transitions=1,
            ),
            reasons=reasons,
        )

    with pytest.raises(ValueError, match="must be acyclic"):
        RouteGraph(
            nodes=graph.nodes,
            edges=(
                RouteEdge(RouteNodeId("a"), RouteNodeId("b")),
                RouteEdge(RouteNodeId("b"), RouteNodeId("c")),
                RouteEdge(RouteNodeId("c"), RouteNodeId("a")),
            ),
        )

    with pytest.raises(TypeError, match="must be QualityMode"):
        RouteQualityRequest(cast(Any, "quality"))

    with pytest.raises(ValueError, match="must be zero when gpu_count is zero"):
        RouteResourceBudget(
            cpu_threads=8,
            ram_bytes=16_000_000_000,
            gpu_count=0,
            gpu_vram_bytes=1,
            scratch_storage_bytes=0,
        )


def test_fallback_resolution_is_exact_allowlisted_and_bounded() -> None:
    _, _, policy, _, _ = _canonical_fixture()
    trigger = FallbackTrigger(
        failure_category=FailureCategory.CAMERA_AMBIGUITY,
    )

    exact = resolve_fallback(
        policy,
        RouteNodeId("a"),
        trigger,
        0,
    )
    missing = resolve_fallback(
        policy,
        RouteNodeId("c"),
        trigger,
        0,
    )
    exhausted = resolve_fallback(
        policy,
        RouteNodeId("a"),
        trigger,
        policy.max_transitions,
    )

    assert exact == FallbackResolution(
        action=FallbackAction.ESCALATE,
        target_node_id=RouteNodeId("b"),
    )
    assert missing == FallbackResolution(
        action=FallbackAction.UNRESOLVED,
        target_node_id=None,
    )
    assert exhausted == FallbackResolution(
        action=FallbackAction.UNRESOLVED,
        target_node_id=None,
    )


@pytest.mark.parametrize(
    "decision",
    [
        QualityDecision.PASS,
        QualityDecision.ACCEPT_WITH_WARNINGS,
    ],
)
def test_positive_quality_outcomes_cannot_trigger_fallback(
    decision: QualityDecision,
) -> None:
    with pytest.raises(ValueError, match="RETRY ESCALATE or UNRESOLVED"):
        FallbackTrigger(quality_decision=decision)


def test_complete_routing_fixture_has_no_execution_or_reconstruction_truth_surface() -> None:
    router_input, graph, policy, reasons, decision = _canonical_fixture()

    objects = (router_input, graph, policy, *reasons, decision)
    forbidden_attributes = {
        "timestamp",
        "random_seed",
        "environment",
        "metadata",
        "score",
        "rank",
        "winner",
        "adapter_id",
        "model",
        "checkpoint",
        "execution_result",
        "provenance",
        "camera_solution",
        "depth",
        "geometry",
        "surface",
        "appearance",
        "temporal_state",
    }
    for value in objects:
        assert forbidden_attributes.isdisjoint(field.name for field in fields(value))


def test_routing_package_exposes_no_selection_execution_or_persistence_api() -> None:
    forbidden_symbols = {
        "select_route",
        "rank_routes",
        "resolve_adapter",
        "execute_route",
        "run_route",
        "schedule_route",
        "persist_route_decision",
        "CameraSolution",
        "DepthField",
        "GeometrySolution",
        "ProvenanceClass",
    }

    assert forbidden_symbols.isdisjoint(routing_module.__dict__)
