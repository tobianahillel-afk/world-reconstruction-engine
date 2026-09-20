from __future__ import annotations

import re
from dataclasses import dataclass

from wre.domain import AdapterCapabilityName

_OPAQUE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


@dataclass(frozen=True, slots=True, order=True)
class RouteNodeId:
    """Opaque stable identity for one structural route node."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str) or _OPAQUE_ID_RE.fullmatch(self.value) is None:
            raise ValueError(
                "route_node_id must be 1-128 characters using letters, digits, '.', '_', ':' or '-'"
            )

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class RouteNode:
    """One route node expressed only through semantic capability requirements."""

    node_id: RouteNodeId
    required_capabilities: tuple[AdapterCapabilityName, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.node_id, RouteNodeId):
            raise TypeError("route_node.node_id must be RouteNodeId")
        if not isinstance(self.required_capabilities, tuple):
            raise TypeError("route_node.required_capabilities must be an immutable tuple")
        if not self.required_capabilities:
            raise ValueError("route_node.required_capabilities must not be empty")
        if any(
            not isinstance(capability, AdapterCapabilityName)
            for capability in self.required_capabilities
        ):
            raise TypeError(
                "route_node.required_capabilities members must be AdapterCapabilityName"
            )

        values = tuple(capability.value for capability in self.required_capabilities)
        if len(values) != len(set(values)):
            raise ValueError("route_node.required_capabilities must be unique")
        if values != tuple(sorted(values)):
            raise ValueError("route_node.required_capabilities must use canonical lexical order")


@dataclass(frozen=True, slots=True)
class RouteEdge:
    """One directed structural dependency between route nodes."""

    source: RouteNodeId
    target: RouteNodeId

    def __post_init__(self) -> None:
        if not isinstance(self.source, RouteNodeId):
            raise TypeError("route_edge.source must be RouteNodeId")
        if not isinstance(self.target, RouteNodeId):
            raise TypeError("route_edge.target must be RouteNodeId")
        if self.source == self.target:
            raise ValueError("route_edge endpoints must be distinct")


@dataclass(frozen=True, slots=True)
class RouteGraph:
    """Immutable acyclic route structure with no execution or selection semantics."""

    nodes: tuple[RouteNode, ...]
    edges: tuple[RouteEdge, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.nodes, tuple):
            raise TypeError("route_graph.nodes must be an immutable tuple")
        if not self.nodes:
            raise ValueError("route_graph.nodes must not be empty")
        if any(not isinstance(node, RouteNode) for node in self.nodes):
            raise TypeError("route_graph.nodes members must be RouteNode")

        node_ids = tuple(node.node_id.value for node in self.nodes)
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("route_graph.nodes must use unique node IDs")
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("route_graph.nodes must use canonical RouteNodeId order")

        if not isinstance(self.edges, tuple):
            raise TypeError("route_graph.edges must be an immutable tuple")
        if any(not isinstance(edge, RouteEdge) for edge in self.edges):
            raise TypeError("route_graph.edges members must be RouteEdge")

        edge_keys = tuple((edge.source.value, edge.target.value) for edge in self.edges)
        if len(edge_keys) != len(set(edge_keys)):
            raise ValueError("route_graph.edges must be unique")
        if edge_keys != tuple(sorted(edge_keys)):
            raise ValueError("route_graph.edges must use canonical source/target order")

        known_ids = set(node_ids)
        for edge in self.edges:
            if edge.source.value not in known_ids or edge.target.value not in known_ids:
                raise ValueError("route_graph edge endpoints must reference nodes in the graph")

        adjacency: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
        indegree: dict[str, int] = {node_id: 0 for node_id in node_ids}
        for edge in self.edges:
            adjacency[edge.source.value].append(edge.target.value)
            indegree[edge.target.value] += 1

        ready = sorted(node_id for node_id, count in indegree.items() if count == 0)
        visited = 0
        while ready:
            current = ready.pop(0)
            visited += 1
            for target in adjacency[current]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()

        if visited != len(node_ids):
            raise ValueError("route_graph must be acyclic")
