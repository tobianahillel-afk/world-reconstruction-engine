from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKey,
    ArtifactKind,
    ArtifactMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    ProducerRef,
    ProvenanceClass,
    SceneProjectId,
    Sha256Digest,
)
from wre.persistence import PersistenceError, SQLiteLocalStore, UnsupportedSchemaVersionError
from wre.persistence.codec import canonical_json, encode_artifact_metadata


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _key(character: str = "c") -> ArtifactKey:
    return ArtifactKey(sha256=_digest(character))


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.adapters.cache-fixture",
            version="1.0.0",
            revision="commit:test",
        ),
        configuration=ConfigurationIdentity(sha256=_digest("a")),
    )


def _metadata(
    *,
    project_id: str,
    artifact_id: str,
    artifact_kind: str = "geometry.solution",
    key_character: str = "c",
) -> ArtifactMetadata:
    return ArtifactMetadata(
        project_id=SceneProjectId(project_id),
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(artifact_id),
            artifact_kind=ArtifactKind(artifact_kind),
        ),
        artifact_key=_key(key_character),
        producer=_producer(),
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )


def _path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "wre.sqlite3"


def test_exact_key_lookup_returns_empty_tuple_for_no_match(tmp_path: Path) -> None:
    store = SQLiteLocalStore(_path(tmp_path))
    store.put_artifact_metadata(
        _metadata(
            project_id="project:one",
            artifact_id="artifact:one",
            key_character="c",
        )
    )

    assert store.find_artifact_metadata_by_key(_key("d")) == ()


def test_exact_key_lookup_returns_single_exact_match_and_rejects_near_key(
    tmp_path: Path,
) -> None:
    store = SQLiteLocalStore(_path(tmp_path))
    exact = _metadata(
        project_id="project:one",
        artifact_id="artifact:exact",
        key_character="c",
    )
    near = _metadata(
        project_id="project:one",
        artifact_id="artifact:near",
        key_character="d",
    )
    store.put_artifact_metadata(near)
    store.put_artifact_metadata(exact)

    assert store.find_artifact_metadata_by_key(_key("c")) == (exact,)


def test_exact_key_lookup_preserves_all_cross_project_matches_in_deterministic_order(
    tmp_path: Path,
) -> None:
    store = SQLiteLocalStore(_path(tmp_path))
    project_z = _metadata(
        project_id="project:z",
        artifact_id="artifact:02",
        key_character="e",
    )
    project_a_late = _metadata(
        project_id="project:a",
        artifact_id="artifact:03",
        key_character="e",
    )
    project_a_early = _metadata(
        project_id="project:a",
        artifact_id="artifact:01",
        artifact_kind="surface.model",
        key_character="e",
    )

    for metadata in (project_z, project_a_late, project_a_early):
        store.put_artifact_metadata(metadata)

    assert store.find_artifact_metadata_by_key(_key("e")) == (
        project_a_early,
        project_a_late,
        project_z,
    )


def test_exact_key_lookup_order_does_not_depend_on_insertion_history(tmp_path: Path) -> None:
    first_path = tmp_path / "first.sqlite3"
    second_path = tmp_path / "second.sqlite3"
    candidates = (
        _metadata(project_id="project:b", artifact_id="artifact:02", key_character="f"),
        _metadata(project_id="project:a", artifact_id="artifact:03", key_character="f"),
        _metadata(project_id="project:a", artifact_id="artifact:01", key_character="f"),
    )

    first = SQLiteLocalStore(first_path)
    second = SQLiteLocalStore(second_path)
    for metadata in candidates:
        first.put_artifact_metadata(metadata)
    for metadata in reversed(candidates):
        second.put_artifact_metadata(metadata)

    assert first.find_artifact_metadata_by_key(_key("f")) == second.find_artifact_metadata_by_key(
        _key("f")
    )


def test_exact_key_lookup_survives_store_reopen(tmp_path: Path) -> None:
    path = _path(tmp_path)
    expected = _metadata(
        project_id="project:reopen",
        artifact_id="artifact:reopen",
        key_character="7",
    )
    store = SQLiteLocalStore(path)
    store.put_artifact_metadata(expected)

    reopened = SQLiteLocalStore(path)

    assert reopened.find_artifact_metadata_by_key(_key("7")) == (expected,)


def test_exact_key_lookup_is_read_only(tmp_path: Path) -> None:
    path = _path(tmp_path)
    store = SQLiteLocalStore(path)
    metadata = _metadata(
        project_id="project:read-only",
        artifact_id="artifact:read-only",
        key_character="8",
    )
    store.put_artifact_metadata(metadata)

    with sqlite3.connect(path) as connection:
        before = connection.execute(
            "SELECT record_type, record_id, schema_version, payload_json "
            "FROM wre_records ORDER BY record_type, record_id"
        ).fetchall()

    assert store.find_artifact_metadata_by_key(_key("8")) == (metadata,)

    with sqlite3.connect(path) as connection:
        after = connection.execute(
            "SELECT record_type, record_id, schema_version, payload_json "
            "FROM wre_records ORDER BY record_type, record_id"
        ).fetchall()

    assert after == before


def test_exact_key_lookup_rejects_untyped_input(tmp_path: Path) -> None:
    store = SQLiteLocalStore(_path(tmp_path))

    with pytest.raises(TypeError, match="artifact_key must be ArtifactKey"):
        store.find_artifact_metadata_by_key(cast(Any, "c" * 64))
    with pytest.raises(TypeError, match="artifact_key must be ArtifactKey"):
        store.find_artifact_metadata_by_key(cast(Any, _digest("c")))


def test_exact_key_lookup_fails_closed_on_malformed_metadata_row(tmp_path: Path) -> None:
    path = _path(tmp_path)
    store = SQLiteLocalStore(path)
    store.put_artifact_metadata(
        _metadata(
            project_id="project:valid",
            artifact_id="artifact:valid",
            key_character="9",
        )
    )

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_metadata', 'artifact:malformed', 1, ?)
            """,
            (canonical_json({"artifact_key_sha256": "9" * 64}),),
        )

    with pytest.raises(PersistenceError, match="cannot be decoded"):
        store.find_artifact_metadata_by_key(_key("9"))


def test_exact_key_lookup_fails_closed_on_unsupported_metadata_version(tmp_path: Path) -> None:
    path = _path(tmp_path)
    store = SQLiteLocalStore(path)
    metadata = _metadata(
        project_id="project:version",
        artifact_id="artifact:version",
        key_character="a",
    )

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_metadata', ?, 999, ?)
            """,
            (
                metadata.artifact_ref.artifact_id.value,
                canonical_json(encode_artifact_metadata(metadata)),
            ),
        )

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported"):
        store.find_artifact_metadata_by_key(_key("a"))


def test_exact_key_lookup_fails_closed_on_record_identity_mismatch(tmp_path: Path) -> None:
    path = _path(tmp_path)
    store = SQLiteLocalStore(path)
    metadata = _metadata(
        project_id="project:mismatch",
        artifact_id="artifact:payload",
        key_character="b",
    )

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_metadata', 'artifact:row-key', 1, ?)
            """,
            (canonical_json(encode_artifact_metadata(metadata)),),
        )

    with pytest.raises(PersistenceError, match="does not match its payload"):
        store.find_artifact_metadata_by_key(_key("b"))
