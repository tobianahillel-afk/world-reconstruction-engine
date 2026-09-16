from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactDependencyEdge,
    ArtifactDependencyGraph,
    ArtifactId,
    ArtifactInvalidationPlan,
    ArtifactKind,
    ArtifactRef,
    plan_artifact_invalidation,
)


def _ref(name: str, kind: str = "artifact.generic") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:{name}"),
        artifact_kind=ArtifactKind(kind),
    )


def _edge(artifact: ArtifactRef, dependency: ArtifactRef) -> ArtifactDependencyEdge:
    return ArtifactDependencyEdge(artifact=artifact, dependency=dependency)


def test_invalidation_plan_is_typed_immutable_and_requires_roots_in_affected() -> None:
    root = _ref("root")
    dependent = _ref("dependent")
    plan = ArtifactInvalidationPlan(
        roots=frozenset({root}),
        affected_artifacts=frozenset({root, dependent}),
    )

    assert plan.roots == frozenset({root})
    assert plan.affected_artifacts == frozenset({root, dependent})
    assert hash(plan) == hash(
        ArtifactInvalidationPlan(
            roots=frozenset({root}),
            affected_artifacts=frozenset({dependent, root}),
        )
    )

    with pytest.raises(FrozenInstanceError):
        plan.roots = frozenset()  # type: ignore[misc]
    with pytest.raises(TypeError, match="roots must be an immutable frozenset"):
        ArtifactInvalidationPlan(
            roots=cast(Any, [root]),
            affected_artifacts=frozenset({root}),
        )
    with pytest.raises(TypeError, match="roots must contain ArtifactRef"):
        ArtifactInvalidationPlan(
            roots=cast(Any, frozenset({"artifact:root"})),
            affected_artifacts=frozenset({root}),
        )
    with pytest.raises(TypeError, match="affected_artifacts must be an immutable frozenset"):
        ArtifactInvalidationPlan(
            roots=frozenset({root}),
            affected_artifacts=cast(Any, [root]),
        )
    with pytest.raises(TypeError, match="affected_artifacts must contain ArtifactRef"):
        ArtifactInvalidationPlan(
            roots=frozenset(),
            affected_artifacts=cast(Any, frozenset({"artifact:root"})),
        )
    with pytest.raises(ValueError, match="must include every root"):
        ArtifactInvalidationPlan(
            roots=frozenset({root}),
            affected_artifacts=frozenset({dependent}),
        )


def test_chain_root_invalidates_all_downstream_dependents() -> None:
    raw = _ref("raw", "media.image")
    features = _ref("features", "feature.set")
    geometry = _ref("geometry", "geometry.solution")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, features, geometry}),
        edges=frozenset({_edge(features, raw), _edge(geometry, features)}),
    )

    plan = plan_artifact_invalidation(graph, frozenset({raw}))

    assert plan.roots == frozenset({raw})
    assert plan.affected_artifacts == frozenset({raw, features, geometry})


def test_changed_downstream_artifact_does_not_invalidate_its_upstream_dependencies() -> None:
    raw = _ref("raw", "media.image")
    features = _ref("features", "feature.set")
    geometry = _ref("geometry", "geometry.solution")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, features, geometry}),
        edges=frozenset({_edge(features, raw), _edge(geometry, features)}),
    )

    plan = plan_artifact_invalidation(graph, frozenset({geometry}))

    assert plan.affected_artifacts == frozenset({geometry})
    assert raw not in plan.affected_artifacts
    assert features not in plan.affected_artifacts


def test_diamond_and_branched_graph_invalidates_only_reachable_reverse_closure() -> None:
    raw = _ref("raw")
    left = _ref("left")
    right = _ref("right")
    merged = _ref("merged")
    left_child = _ref("left-child")
    unrelated = _ref("unrelated")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, left, right, merged, left_child, unrelated}),
        edges=frozenset(
            {
                _edge(left, raw),
                _edge(right, raw),
                _edge(merged, left),
                _edge(merged, right),
                _edge(left_child, left),
            }
        ),
    )

    plan = plan_artifact_invalidation(graph, frozenset({left}))

    assert plan.affected_artifacts == frozenset({left, merged, left_child})
    assert raw not in plan.affected_artifacts
    assert right not in plan.affected_artifacts
    assert unrelated not in plan.affected_artifacts


def test_multiple_roots_union_their_reverse_dependency_closures() -> None:
    raw_a = _ref("raw-a")
    raw_b = _ref("raw-b")
    a = _ref("a")
    b = _ref("b")
    merged = _ref("merged")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw_a, raw_b, a, b, merged}),
        edges=frozenset(
            {
                _edge(a, raw_a),
                _edge(b, raw_b),
                _edge(merged, a),
                _edge(merged, b),
            }
        ),
    )

    plan = plan_artifact_invalidation(graph, frozenset({raw_a, raw_b}))

    assert plan.roots == frozenset({raw_a, raw_b})
    assert plan.affected_artifacts == frozenset({raw_a, raw_b, a, b, merged})


def test_downstream_root_duplicate_does_not_change_final_union() -> None:
    raw = _ref("raw")
    mid = _ref("mid")
    output = _ref("output")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, mid, output}),
        edges=frozenset({_edge(mid, raw), _edge(output, mid)}),
    )

    from_raw = plan_artifact_invalidation(graph, frozenset({raw}))
    from_raw_and_mid = plan_artifact_invalidation(graph, frozenset({raw, mid}))

    assert from_raw.affected_artifacts == from_raw_and_mid.affected_artifacts
    assert from_raw_and_mid.roots == frozenset({raw, mid})


def test_isolated_and_empty_roots_are_explicit_no_surprises() -> None:
    isolated = _ref("isolated")
    other = _ref("other")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({isolated, other}),
        edges=frozenset(),
    )

    isolated_plan = plan_artifact_invalidation(graph, frozenset({isolated}))
    empty_plan = plan_artifact_invalidation(graph, frozenset())

    assert isolated_plan.affected_artifacts == frozenset({isolated})
    assert empty_plan == ArtifactInvalidationPlan(
        roots=frozenset(),
        affected_artifacts=frozenset(),
    )


def test_planner_rejects_undeclared_or_kind_mismatched_roots() -> None:
    declared = _ref("shared", "geometry.solution")
    graph = ArtifactDependencyGraph(nodes=frozenset({declared}), edges=frozenset())
    undeclared = _ref("missing")
    wrong_kind = ArtifactRef(
        artifact_id=declared.artifact_id,
        artifact_kind=ArtifactKind("surface.model"),
    )

    with pytest.raises(ValueError, match="exact declared graph node"):
        plan_artifact_invalidation(graph, frozenset({undeclared}))
    with pytest.raises(ValueError, match="exact declared graph node"):
        plan_artifact_invalidation(graph, frozenset({wrong_kind}))


def test_planner_rejects_untyped_graph_roots_collection_and_members() -> None:
    root = _ref("root")
    graph = ArtifactDependencyGraph(nodes=frozenset({root}), edges=frozenset())

    with pytest.raises(TypeError, match="graph must be ArtifactDependencyGraph"):
        plan_artifact_invalidation(cast(Any, "graph"), frozenset({root}))
    with pytest.raises(TypeError, match="roots must be an immutable frozenset"):
        plan_artifact_invalidation(graph, cast(Any, [root]))
    with pytest.raises(TypeError, match="roots must contain ArtifactRef"):
        plan_artifact_invalidation(graph, cast(Any, frozenset({"artifact:root"})))


def test_planning_is_pure_and_does_not_modify_graph() -> None:
    root = _ref("root")
    dependent = _ref("dependent")
    graph = ArtifactDependencyGraph(
        nodes=frozenset({root, dependent}),
        edges=frozenset({_edge(dependent, root)}),
    )
    before_hash = hash(graph)
    before_nodes = graph.nodes
    before_edges = graph.edges

    plan_artifact_invalidation(graph, frozenset({root}))

    assert hash(graph) == before_hash
    assert graph.nodes is before_nodes
    assert graph.edges is before_edges


def test_long_chain_planning_does_not_depend_on_python_recursion_limit() -> None:
    nodes = tuple(_ref(f"n{index:04d}") for index in range(1500))
    graph = ArtifactDependencyGraph(
        nodes=frozenset(nodes),
        edges=frozenset(
            _edge(nodes[index], nodes[index - 1]) for index in range(1, len(nodes))
        ),
    )

    plan = plan_artifact_invalidation(graph, frozenset({nodes[0]}))

    assert len(plan.affected_artifacts) == 1500
    assert nodes[-1] in plan.affected_artifacts
