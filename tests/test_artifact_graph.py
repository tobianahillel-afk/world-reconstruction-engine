from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import ArtifactId, ArtifactKind, ArtifactRef
from wre.domain.artifact_graph import ArtifactDependencyEdge, ArtifactDependencyGraph


def _ref(name: str, kind: str = "artifact.generic") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:{name}"),
        artifact_kind=ArtifactKind(kind),
    )


def _edge(artifact: ArtifactRef, dependency: ArtifactRef) -> ArtifactDependencyEdge:
    return ArtifactDependencyEdge(artifact=artifact, dependency=dependency)


def test_dependency_edge_is_typed_immutable_and_directional() -> None:
    output = _ref("output")
    source = _ref("source", "media.image")
    edge = _edge(output, source)

    assert edge.artifact == output
    assert edge.dependency == source
    assert edge != _edge(source, output)
    assert hash(edge) == hash(_edge(output, source))

    with pytest.raises(FrozenInstanceError):
        edge.artifact = source  # type: ignore[misc]
    with pytest.raises(TypeError, match=r"artifact_dependency_edge\.artifact"):
        ArtifactDependencyEdge(artifact=cast(Any, "artifact:output"), dependency=source)
    with pytest.raises(TypeError, match=r"artifact_dependency_edge\.dependency"):
        ArtifactDependencyEdge(artifact=output, dependency=cast(Any, "artifact:source"))


def test_valid_dag_accepts_isolated_chain_diamond_and_disconnected_components() -> None:
    raw = _ref("raw", "media.image")
    features = _ref("features", "feature.set")
    matches = _ref("matches", "feature.matches")
    cameras = _ref("cameras", "camera.solution")
    report = _ref("report", "report.summary")
    isolated = _ref("isolated")

    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, features, matches, cameras, report, isolated}),
        edges=frozenset(
            {
                _edge(features, raw),
                _edge(matches, raw),
                _edge(cameras, features),
                _edge(cameras, matches),
                _edge(report, cameras),
            }
        ),
    )

    assert isolated in graph.nodes
    assert _edge(cameras, features) in graph.edges
    assert _edge(cameras, matches) in graph.edges


def test_graph_equality_and_hashing_are_insertion_order_independent() -> None:
    a = _ref("a")
    b = _ref("b")
    c = _ref("c")
    first = ArtifactDependencyGraph(
        nodes=frozenset((a, b, c)),
        edges=frozenset((_edge(c, b), _edge(b, a))),
    )
    second = ArtifactDependencyGraph(
        nodes=frozenset((c, a, b)),
        edges=frozenset((_edge(b, a), _edge(c, b))),
    )

    assert first == second
    assert hash(first) == hash(second)


def test_graph_rejects_undeclared_edge_endpoints() -> None:
    a = _ref("a")
    b = _ref("b")

    with pytest.raises(ValueError, match="endpoints must be declared"):
        ArtifactDependencyGraph(
            nodes=frozenset({a}),
            edges=frozenset({_edge(a, b)}),
        )


def test_graph_rejects_self_dependency() -> None:
    a = _ref("a")

    with pytest.raises(ValueError, match="self-dependency"):
        ArtifactDependencyGraph(
            nodes=frozenset({a}),
            edges=frozenset({_edge(a, a)}),
        )


def test_graph_rejects_two_node_cycle() -> None:
    a = _ref("a")
    b = _ref("b")

    with pytest.raises(ValueError, match="directed cycle"):
        ArtifactDependencyGraph(
            nodes=frozenset({a, b}),
            edges=frozenset({_edge(a, b), _edge(b, a)}),
        )


def test_graph_rejects_longer_cycle() -> None:
    a = _ref("a")
    b = _ref("b")
    c = _ref("c")

    with pytest.raises(ValueError, match="directed cycle"):
        ArtifactDependencyGraph(
            nodes=frozenset({a, b, c}),
            edges=frozenset({_edge(a, b), _edge(b, c), _edge(c, a)}),
        )


def test_graph_rejects_conflicting_kind_for_same_artifact_id() -> None:
    artifact_id = ArtifactId("artifact:shared")
    image = ArtifactRef(artifact_id=artifact_id, artifact_kind=ArtifactKind("media.image"))
    depth = ArtifactRef(artifact_id=artifact_id, artifact_kind=ArtifactKind("depth.field"))

    with pytest.raises(ValueError, match="conflicting ArtifactKind"):
        ArtifactDependencyGraph(nodes=frozenset({image, depth}), edges=frozenset())


def test_graph_contract_rejects_mutable_or_untyped_collections_and_members() -> None:
    a = _ref("a")

    with pytest.raises(TypeError, match="nodes must be an immutable frozenset"):
        ArtifactDependencyGraph(nodes=cast(Any, [a]), edges=frozenset())
    with pytest.raises(TypeError, match="nodes must contain ArtifactRef"):
        ArtifactDependencyGraph(nodes=cast(Any, frozenset({"artifact:a"})), edges=frozenset())
    with pytest.raises(TypeError, match="edges must be an immutable frozenset"):
        ArtifactDependencyGraph(nodes=frozenset({a}), edges=cast(Any, []))
    with pytest.raises(TypeError, match="edges must contain ArtifactDependencyEdge"):
        ArtifactDependencyGraph(
            nodes=frozenset({a}),
            edges=cast(Any, frozenset({("artifact:a", "artifact:b")})),
        )


def test_graph_is_immutable_and_contains_only_direct_claims() -> None:
    a = _ref("a")
    b = _ref("b")
    c = _ref("c")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({a, b, c}),
        edges=frozenset({_edge(c, b), _edge(b, a)}),
    )

    assert _edge(c, a) not in graph.edges
    with pytest.raises(FrozenInstanceError):
        graph.nodes = frozenset()  # type: ignore[misc]
