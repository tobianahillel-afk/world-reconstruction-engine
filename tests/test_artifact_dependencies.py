from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactDependencyEdge,
    ArtifactDependencyGraph,
    ArtifactId,
    ArtifactKind,
    ArtifactRef,
    SceneProjectId,
)
from wre.persistence import (
    PersistenceConflictError,
    PersistenceError,
    SQLiteLocalStore,
    UnsupportedSchemaVersionError,
)
from wre.persistence.codec import (
    canonical_json,
    decode_artifact_dependencies,
    encode_artifact_dependencies,
)


def _ref(name: str, kind: str = "artifact.generic") -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(f"artifact:{name}"),
        artifact_kind=ArtifactKind(kind),
    )


def _edge(artifact: ArtifactRef, dependency: ArtifactRef) -> ArtifactDependencyEdge:
    return ArtifactDependencyEdge(artifact=artifact, dependency=dependency)


def _graph(
    nodes: set[ArtifactRef],
    edges: set[ArtifactDependencyEdge] | None = None,
) -> ArtifactDependencyGraph:
    return ArtifactDependencyGraph(
        nodes=frozenset(nodes),
        edges=frozenset() if edges is None else frozenset(edges),
    )


def _store(tmp_path: Path) -> SQLiteLocalStore:
    return SQLiteLocalStore(tmp_path / "state" / "wre.sqlite3")


def test_dependency_codec_is_deterministic_and_preserves_empty_sets() -> None:
    artifact = _ref("output", "geometry.solution")
    first = _ref("a", "media.image")
    second = _ref("b", "feature.matches")

    payload = encode_artifact_dependencies(artifact, frozenset({second, first}))

    assert payload == {
        "artifact_ref": {
            "artifact_id": "artifact:output",
            "artifact_kind": "geometry.solution",
        },
        "dependencies": [
            {"artifact_id": "artifact:a", "artifact_kind": "media.image"},
            {"artifact_id": "artifact:b", "artifact_kind": "feature.matches"},
        ],
    }
    assert canonical_json(payload) == canonical_json(
        encode_artifact_dependencies(artifact, frozenset({first, second}))
    )
    assert decode_artifact_dependencies(payload) == (artifact, frozenset({first, second}))
    assert decode_artifact_dependencies(encode_artifact_dependencies(artifact, frozenset())) == (
        artifact,
        frozenset(),
    )


def test_dependency_codec_rejects_untyped_and_duplicate_wire_members() -> None:
    artifact = _ref("output")
    dependency = _ref("input")

    with pytest.raises(TypeError, match="artifact_ref must be ArtifactRef"):
        encode_artifact_dependencies(cast(Any, "artifact:output"), frozenset())
    with pytest.raises(TypeError, match="dependencies must be an immutable frozenset"):
        encode_artifact_dependencies(artifact, cast(Any, [dependency]))
    with pytest.raises(TypeError, match="dependencies must contain ArtifactRef"):
        encode_artifact_dependencies(artifact, cast(Any, frozenset({"artifact:input"})))

    dependency_payload = {
        "artifact_id": dependency.artifact_id.value,
        "artifact_kind": dependency.artifact_kind.value,
    }
    with pytest.raises(ValueError, match="must not contain duplicate"):
        decode_artifact_dependencies(
            {
                "artifact_ref": {
                    "artifact_id": artifact.artifact_id.value,
                    "artifact_kind": artifact.artifact_kind.value,
                },
                "dependencies": [dependency_payload, dependency_payload],
            }
        )


def test_store_round_trip_direct_dependencies_and_dependents_after_reopen(tmp_path: Path) -> None:
    raw = _ref("raw", "media.image")
    features = _ref("features", "feature.set")
    matches = _ref("matches", "feature.matches")
    cameras = _ref("cameras", "camera.solution")
    report = _ref("report", "report.summary")
    isolated = _ref("isolated")
    graph = _graph(
        {raw, features, matches, cameras, report, isolated},
        {
            _edge(features, raw),
            _edge(matches, raw),
            _edge(cameras, features),
            _edge(cameras, matches),
            _edge(report, cameras),
        },
    )
    path = tmp_path / "state" / "wre.sqlite3"

    SQLiteLocalStore(path).put_artifact_dependency_graph(graph)
    reopened = SQLiteLocalStore(path)

    assert reopened.get_artifact_dependencies(raw) == frozenset()
    assert reopened.get_artifact_dependencies(isolated) == frozenset()
    assert reopened.get_artifact_dependencies(cameras) == frozenset({features, matches})
    assert reopened.get_artifact_dependencies(_ref("missing")) is None
    assert reopened.find_direct_artifact_dependents(raw) == frozenset({features, matches})
    assert reopened.find_direct_artifact_dependents(cameras) == frozenset({report})
    assert reopened.find_direct_artifact_dependents(report) == frozenset()
    assert cameras not in reopened.find_direct_artifact_dependents(raw)
    assert report not in reopened.find_direct_artifact_dependents(raw)


def test_repeated_and_overlapping_graph_writes_are_idempotent(tmp_path: Path) -> None:
    a = _ref("a")
    b = _ref("b")
    c = _ref("c")
    first = _graph({a, b}, {_edge(b, a)})
    extended = _graph({a, b, c}, {_edge(b, a), _edge(c, b)})
    store = _store(tmp_path)

    store.put_artifact_dependency_graph(first)
    store.put_artifact_dependency_graph(first)
    store.put_artifact_dependency_graph(extended)
    store.put_artifact_dependency_graph(extended)

    assert store.get_artifact_dependencies(a) == frozenset()
    assert store.get_artifact_dependencies(b) == frozenset({a})
    assert store.get_artifact_dependencies(c) == frozenset({b})


def test_dependency_set_conflict_rolls_back_entire_graph_write(tmp_path: Path) -> None:
    new_first = _ref("aa-new")
    conflicting = _ref("b-conflict")
    original_dependency = _ref("z-dependency")
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(
        _graph(
            {conflicting, original_dependency},
            {_edge(conflicting, original_dependency)},
        )
    )

    conflicting_attempt = _graph(
        {new_first, conflicting, original_dependency},
        {_edge(new_first, conflicting)},
    )
    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_artifact_dependency_graph(conflicting_attempt)

    assert store.get_artifact_dependencies(new_first) is None
    assert store.get_artifact_dependencies(conflicting) == frozenset({original_dependency})


def test_artifact_kind_conflict_rolls_back_entire_graph_write(tmp_path: Path) -> None:
    new_first = _ref("aa-new")
    shared_id = ArtifactId("artifact:shared")
    original = ArtifactRef(shared_id, ArtifactKind("media.image"))
    conflicting = ArtifactRef(shared_id, ArtifactKind("depth.field"))
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(_graph({original}))

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_artifact_dependency_graph(
            _graph({new_first, conflicting}, {_edge(new_first, conflicting)})
        )

    assert store.get_artifact_dependencies(new_first) is None
    assert store.get_artifact_dependencies(original) == frozenset()


def test_dependency_storage_does_not_create_project_or_artifact_metadata(tmp_path: Path) -> None:
    source = _ref("source")
    output = _ref("output")
    store = _store(tmp_path)

    store.put_artifact_dependency_graph(_graph({source, output}, {_edge(output, source)}))

    assert store.get_artifact_metadata(source.artifact_id) is None
    assert store.get_artifact_metadata(output.artifact_id) is None
    assert store.get_scene_project(SceneProjectId("project:missing")) is None


def test_lookup_rejects_requested_kind_mismatch(tmp_path: Path) -> None:
    artifact = _ref("shared", "media.image")
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(_graph({artifact}))
    wrong_kind = ArtifactRef(artifact.artifact_id, ArtifactKind("depth.field"))

    with pytest.raises(PersistenceError, match="does not match the requested ArtifactRef"):
        store.get_artifact_dependencies(wrong_kind)


def test_malformed_dependency_payload_fails_closed(tmp_path: Path) -> None:
    artifact = _ref("corrupt")
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(_graph({artifact}))

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE wre_records
            SET payload_json = ?
            WHERE record_type = 'artifact_dependencies' AND record_id = ?
            """,
            ('{"artifact_ref":{"artifact_id":"artifact:corrupt"}}', artifact.artifact_id.value),
        )

    with pytest.raises(PersistenceError, match="cannot be decoded"):
        store.get_artifact_dependencies(artifact)
    with pytest.raises(PersistenceError, match="cannot be decoded"):
        store.find_direct_artifact_dependents(_ref("anything"))


def test_dependency_record_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    artifact = _ref("original")
    replacement = _ref("replacement")
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(_graph({artifact}))

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE wre_records
            SET payload_json = ?
            WHERE record_type = 'artifact_dependencies' AND record_id = ?
            """,
            (
                canonical_json(encode_artifact_dependencies(replacement, frozenset())),
                artifact.artifact_id.value,
            ),
        )

    with pytest.raises(PersistenceError, match="does not match the requested ArtifactRef"):
        store.get_artifact_dependencies(artifact)
    with pytest.raises(PersistenceError, match="does not match its payload"):
        store.find_direct_artifact_dependents(_ref("anything"))


def test_unsupported_dependency_record_version_fails_closed(tmp_path: Path) -> None:
    artifact = _ref("versioned")
    store = _store(tmp_path)
    store.put_artifact_dependency_graph(_graph({artifact}))

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE wre_records
            SET schema_version = 999
            WHERE record_type = 'artifact_dependencies' AND record_id = ?
            """,
            (artifact.artifact_id.value,),
        )

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported"):
        store.get_artifact_dependencies(artifact)
    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported"):
        store.find_direct_artifact_dependents(_ref("anything"))


def test_store_rejects_untyped_graph_and_lookup_inputs(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(TypeError, match="graph must be ArtifactDependencyGraph"):
        store.put_artifact_dependency_graph(cast(Any, {}))
    with pytest.raises(TypeError, match="artifact_ref must be ArtifactRef"):
        store.get_artifact_dependencies(cast(Any, "artifact:x"))
    with pytest.raises(TypeError, match="dependency_ref must be ArtifactRef"):
        store.find_direct_artifact_dependents(cast(Any, "artifact:x"))
