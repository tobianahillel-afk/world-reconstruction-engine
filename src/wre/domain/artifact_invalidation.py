from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from wre.domain.artifact_graph import ArtifactDependencyGraph
from wre.domain.artifacts import ArtifactRef


@dataclass(frozen=True, slots=True)
class ArtifactInvalidationPlan:
    """Pure invalidation scope: changed roots plus all downstream dependents."""

    roots: frozenset[ArtifactRef]
    affected_artifacts: frozenset[ArtifactRef]

    def __post_init__(self) -> None:
        if not isinstance(self.roots, frozenset):
            raise TypeError("artifact_invalidation_plan.roots must be an immutable frozenset")
        if not all(isinstance(root, ArtifactRef) for root in self.roots):
            raise TypeError("artifact_invalidation_plan.roots must contain ArtifactRef values")
        if not isinstance(self.affected_artifacts, frozenset):
            raise TypeError(
                "artifact_invalidation_plan.affected_artifacts must be an immutable frozenset"
            )
        if not all(
            isinstance(artifact, ArtifactRef) for artifact in self.affected_artifacts
        ):
            raise TypeError(
                "artifact_invalidation_plan.affected_artifacts must contain ArtifactRef values"
            )
        if not self.roots.issubset(self.affected_artifacts):
            raise ValueError("artifact invalidation plan affected_artifacts must include every root")


def plan_artifact_invalidation(
    graph: ArtifactDependencyGraph,
    roots: frozenset[ArtifactRef],
) -> ArtifactInvalidationPlan:
    """Return roots plus every direct/transitive dependent without mutating state."""

    if not isinstance(graph, ArtifactDependencyGraph):
        raise TypeError("graph must be ArtifactDependencyGraph")
    if not isinstance(roots, frozenset):
        raise TypeError("roots must be an immutable frozenset")
    if not all(isinstance(root, ArtifactRef) for root in roots):
        raise TypeError("roots must contain ArtifactRef values")

    undeclared = roots.difference(graph.nodes)
    if undeclared:
        raise ValueError("every invalidation root must be an exact declared graph node")

    dependents_by_dependency: dict[ArtifactRef, set[ArtifactRef]] = {
        node: set() for node in graph.nodes
    }
    for edge in graph.edges:
        dependents_by_dependency[edge.dependency].add(edge.artifact)

    affected = set(roots)
    pending = deque(roots)
    while pending:
        dependency = pending.popleft()
        for dependent in dependents_by_dependency[dependency]:
            if dependent in affected:
                continue
            affected.add(dependent)
            pending.append(dependent)

    return ArtifactInvalidationPlan(
        roots=roots,
        affected_artifacts=frozenset(affected),
    )
