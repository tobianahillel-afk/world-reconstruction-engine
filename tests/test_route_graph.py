from __future__ import annotations

from dataclasses import FrozenInstanceError, fields
from typing import Any, cast

import pytest

import wre.routing.graph as route_graph_module
from wre.domain import AdapterCapabilityName
from wre.routing import RouteEdge, RouteGraph, RouteNode, RouteNodeId


def _capability(value: str) -> AdapterCapabilityName:
    return AdapterCapabilityName(value)


def _node(node_id: str, *capabilities: str) -> RouteNode:
    return RouteNode(
        node_id=RouteNodeId(node_id),
        required_capabilities=tuple(_capability(value) for value in capabilities),
    )


def _edge(source: str, target: str) -> RouteEdge:
    return RouteEdge(source=RouteNodeId(source), target=RouteNodeId(target))


def test_route_node_id_uses_opaque_identity_contract_and_is_distinct() -> None:
    first = RouteNodeId("Route:Node_01.v1")
    same = RouteNodeId("Route:Node_01.v1")

    assert str(first) == "Route:Node_01.v1"
    assert first == same
    assert hash(first) == hash(same)
    assert first != cast(Any, AdapterCapabilityName("route.node_01.v1"))

    with pytest.raises(FrozenInstanceError):
        first.value = "changed"  # type: ignore[misc]

    for invalid in ("", " leading", "trailing ", ".leading", "has/slash", "a" * 129):
        with pytest.raises(ValueError):
            RouteNodeId(invalid)


def test_route_node_is_exact_frozen_contract() -> None:
    node = _node("a", "geometry.camera", "geometry.depth")

    assert tuple(field.name for field in fields(RouteNode)) == (
        "node_id",
        "required_capabilities",
    )
    assert node.required_capabilities == (
        AdapterCapabilityName("geometry.camera"),
        AdapterCapabilityName("geometry.depth"),
    )
    with pytest.raises(FrozenInstanceError):
        node.node_id = RouteNodeId("b")  # type: ignore[misc]


def test_route_node_requires_nonempty_canonical_unique_typed_capabilities() -> None:
    with pytest.raises(TypeError, match="node_id must be RouteNodeId"):
        RouteNode(
            node_id=cast(Any, "a"),
            required_capabilities=(_capability("geometry.camera"),),
        )
    with pytest.raises(TypeError, match="immutable tuple"):
        RouteNode(
            node_id=RouteNodeId("a"),
            required_capabilities=cast(Any, [_capability("geometry.camera")]),
        )
    with pytest.raises(ValueError, match="must not be empty"):
        RouteNode(node_id=RouteNodeId("a"), required_capabilities=())
    with pytest.raises(TypeError, match="AdapterCapabilityName"):
        RouteNode(
            node_id=RouteNodeId("a"),
            required_capabilities=cast(Any, ("geometry.camera",)),
        )
    with pytest.raises(ValueError, match="must be unique"):
        RouteNode(
            node_id=RouteNodeId("a"),
            required_capabilities=(
                _capability("geometry.camera"),
                _capability("geometry.camera"),
            ),
        )
    with pytest.raises(ValueError, match="canonical lexical order"):
        RouteNode(
            node_id=RouteNodeId("a"),
            required_capabilities=(
                _capability("geometry.depth"),
                _capability("geometry.camera"),
            ),
        )


def test_route_edge_is_exact_frozen_typed_contract() -> None:
    edge = _edge("a", "b")
    same = _edge("a", "b")

    assert tuple(field.name for field in fields(RouteEdge)) == ("source", "target")
    assert edge == same
    assert hash(edge) == hash(same)

    with pytest.raises(FrozenInstanceError):
        edge.target = RouteNodeId("c")  # type: ignore[misc]
    with pytest.raises(TypeError, match="source must be RouteNodeId"):
        RouteEdge(source=cast(Any, "a"), target=RouteNodeId("b"))
    with pytest.raises(TypeError, match="target must be RouteNodeId"):
        RouteEdge(source=RouteNodeId("a"), target=cast(Any, "b"))
    with pytest.raises(ValueError, match="endpoints must be distinct"):
        _edge("a", "a")


def test_route_graph_is_exact_frozen_contract() -> None:
    graph = RouteGraph(
        nodes=(
            _node("a", "cap.a"),
            _node("b", "cap.b"),
        ),
        edges=(_edge("a", "b"),),
    )

    assert tuple(field.name for field in fields(RouteGraph)) == ("nodes", "edges")
    assert graph.nodes[0].node_id == RouteNodeId("a")
    with pytest.raises(FrozenInstanceError):
        graph.edges = ()  # type: ignore[misc]


def test_route_graph_requires_canonical_unique_nodes_and_edges() -> None:
    node_a = _node("a", "cap.a")
    node_b = _node("b", "cap.b")
    edge = _edge("a", "b")

    with pytest.raises(TypeError, match="nodes must be an immutable tuple"):
        RouteGraph(nodes=cast(Any, [node_a]), edges=())
    with pytest.raises(ValueError, match="nodes must not be empty"):
        RouteGraph(nodes=(), edges=())
    with pytest.raises(TypeError, match="nodes members must be RouteNode"):
        RouteGraph(nodes=cast(Any, ("a",)), edges=())
    with pytest.raises(ValueError, match="unique node IDs"):
        RouteGraph(nodes=(node_a, node_a), edges=())
    with pytest.raises(ValueError, match="canonical RouteNodeId order"):
        RouteGraph(nodes=(node_b, node_a), edges=())

    with pytest.raises(TypeError, match="edges must be an immutable tuple"):
        RouteGraph(nodes=(node_a, node_b), edges=cast(Any, [edge]))
    with pytest.raises(TypeError, match="edges members must be RouteEdge"):
        RouteGraph(nodes=(node_a, node_b), edges=cast(Any, ("a",)))
    with pytest.raises(ValueError, match="edges must be unique"):
        RouteGraph(nodes=(node_a, node_b), edges=(edge, edge))


def test_route_graph_rejects_decreasing_or_foreign_edges() -> None:
    nodes = (
        _node("a", "cap.a"),
        _node("b", "cap.b"),
        _node("c", "cap.c"),
    )

    with pytest.raises(ValueError, match="canonical source/target order"):
        RouteGraph(
            nodes=nodes,
            edges=(
                _edge("b", "c"),
                _edge("a", "b"),
            ),
        )
    with pytest.raises(ValueError, match="reference nodes in the graph"):
        RouteGraph(nodes=nodes, edges=(_edge("a", "z"),))


def test_one_node_linear_branch_merge_parallel_and_multi_root_graphs_are_valid() -> None:
    assert RouteGraph(nodes=(_node("a", "cap.a"),), edges=()).edges == ()

    linear = RouteGraph(
        nodes=(
            _node("a", "cap.a"),
            _node("b", "cap.b"),
            _node("c", "cap.c"),
        ),
        edges=(
            _edge("a", "b"),
            _edge("b", "c"),
        ),
    )
    assert len(linear.edges) == 2

    branch_merge = RouteGraph(
        nodes=(
            _node("a", "cap.a"),
            _node("b", "cap.b"),
            _node("c", "cap.c"),
            _node("d", "cap.d"),
        ),
        edges=(
            _edge("a", "b"),
            _edge("a", "c"),
            _edge("b", "d"),
            _edge("c", "d"),
        ),
    )
    assert len(branch_merge.edges) == 4

    parallel_multi_root = RouteGraph(
        nodes=(
            _node("a", "cap.a"),
            _node("b", "cap.b"),
            _node("c", "cap.c"),
            _node("d", "cap.d"),
        ),
        edges=(
            _edge("a", "c"),
            _edge("b", "c"),
        ),
    )
    assert RouteNodeId("d") in tuple(node.node_id for node in parallel_multi_root.nodes)


@pytest.mark.parametrize(
    "edges",
    [
        (
            _edge("a", "b"),
            _edge("b", "a"),
        ),
        (
            _edge("a", "b"),
            _edge("b", "c"),
            _edge("c", "a"),
        ),
    ],
)
def test_route_graph_rejects_directed_cycles(edges: tuple[RouteEdge, ...]) -> None:
    nodes = (
        _node("a", "cap.a"),
        _node("b", "cap.b"),
        _node("c", "cap.c"),
    )

    canonical_edges = tuple(sorted(edges, key=lambda item: (item.source.value, item.target.value)))
    with pytest.raises(ValueError, match="must be acyclic"):
        RouteGraph(nodes=nodes, edges=canonical_edges)


def test_route_graph_contract_has_no_selection_execution_or_future_router_surface() -> None:
    node = _node("a", "cap.a")
    graph = RouteGraph(nodes=(node,), edges=())

    for value in (node, graph):
        for attribute in (
            "adapter_id",
            "model",
            "checkpoint",
            "profile",
            "budget",
            "artifacts",
            "failure",
            "quality_mode",
            "quality_decision",
            "fallback",
            "retry",
            "escalation",
            "reason",
            "score",
            "rank",
            "winner",
            "metadata",
        ):
            assert not hasattr(value, attribute)


def test_route_graph_module_has_no_io_persistence_or_execution_surface() -> None:
    forbidden_symbols = {
        "Path",
        "subprocess",
        "socket",
        "requests",
        "sqlite3",
        "FFmpegToolchain",
        "ArtifactProducerIdentity",
        "HardwareRuntimeIdentity",
        "QualityDecision",
    }

    assert forbidden_symbols.isdisjoint(route_graph_module.__dict__)
