from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from wre.domain.artifacts import ArtifactId, ArtifactKind, ArtifactRef


@dataclass(frozen=True, slots=True)
class ArtifactDependencyEdge:
    """One direct dependency claim: artifact depends on dependency."""

    artifact: ArtifactRef
    dependency: ArtifactRef

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, ArtifactRef):
            raise TypeError("artifact_dependency_edge.artifact must be ArtifactRef")
        if not isinstance(self.dependency, ArtifactRef):
            raise TypeError("artifact_dependency_edge.dependency must be ArtifactRef")


@dataclass(frozen=True, slots=True)
class ArtifactDependencyGraph:
    """Immutable direct artifact-dependency DAG with fail-closed validation."""

    nodes: frozenset[ArtifactRef]
    edges: frozenset[ArtifactDependencyEdge]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, frozenset):
            raise TypeError("artifact_dependency_graph.nodes must be an immutable frozenset")
        if not all(isinstance(node, ArtifactRef) for node in self.nodes):
            raise TypeError("artifact_dependency_graph.nodes must contain ArtifactRef values")
        if not isinstance(self.edges, frozenset):
            raise TypeError("artifact_dependency_graph.edges must be an immutable frozenset")
        if not all(isinstance(edge, ArtifactDependencyEdge) for edge in self.edges):
            raise TypeError(
                "artifact_dependency_graph.edges must contain ArtifactDependencyEdge values"
            )

        self._validate_kind_consistency()
        self._validate_edges()
        self._validate_acyclic()

    def _validate_kind_consistency(self) -> None:
        kinds_by_id: dict[ArtifactId, ArtifactKind] = {}
        for node in self.nodes:
            existing = kinds_by_id.get(node.artifact_id)
            if existing is not None and existing != node.artifact_kind:
                raise ValueError(
                    "artifact dependency graph cannot declare one ArtifactId with conflicting "
                    "ArtifactKind values"
                )
            kinds_by_id[node.artifact_id] = node.artifact_kind

    def _validate_edges(self) -> None:
        for edge in self.edges:
            if edge.artifact not in self.nodes or edge.dependency not in self.nodes:
                raise ValueError("artifact dependency edge endpoints must be declared graph nodes")
            if edge.artifact == edge.dependency:
                raise ValueError("artifact dependency graph cannot contain a self-dependency")

    def _validate_acyclic(self) -> None:
        adjacency: dict[ArtifactRef, set[ArtifactRef]] = {node: set() for node in self.nodes}
        incoming_count: dict[ArtifactRef, int] = {node: 0 for node in self.nodes}
        for edge in self.edges:
            adjacency[edge.artifact].add(edge.dependency)
            incoming_count[edge.dependency] += 1

        ready = deque(node for node, count in incoming_count.items() if count == 0)
        visited_count = 0
        while ready:
            node = ready.popleft()
            visited_count += 1
            for dependency in adjacency[node]:
                incoming_count[dependency] -= 1
                if incoming_count[dependency] == 0:
                    ready.append(dependency)

        if visited_count != len(self.nodes):
            raise ValueError("artifact dependency graph cannot contain a directed cycle")
