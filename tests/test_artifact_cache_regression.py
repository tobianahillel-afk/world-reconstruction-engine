from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from wre.domain import (
    ArtifactDependencyEdge,
    ArtifactDependencyGraph,
    ArtifactId,
    ArtifactInputFingerprint,
    ArtifactKey,
    ArtifactKeyMaterial,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMaterializationVerificationStatus,
    ArtifactMetadata,
    ArtifactProducerIdentity,
    ArtifactRef,
    ConfigurationIdentity,
    ProducerRef,
    ProvenanceClass,
    SceneProjectId,
    Sha256Digest,
    derive_artifact_key,
    plan_artifact_invalidation,
)
from wre.materialization import verify_local_artifact_materialization
from wre.persistence import SQLiteLocalStore


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _sha256_bytes(value: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(value).hexdigest())


def _producer(configuration_character: str = "a") -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.adapters.v2l2-regression",
            version="1.0.0",
            revision="fixture:v1",
        ),
        configuration=ConfigurationIdentity(sha256=_digest(configuration_character)),
    )


def _key(configuration_character: str = "a") -> ArtifactKey:
    return derive_artifact_key(
        ArtifactKeyMaterial(
            output_kind=ArtifactKind("geometry.solution"),
            input_fingerprints=(
                ArtifactInputFingerprint(
                    artifact_kind=ArtifactKind("media.image"),
                    sha256=_digest("1"),
                ),
            ),
            producer=_producer(configuration_character),
        )
    )


def _metadata(
    *,
    project_id: str,
    artifact_id: str,
    key: ArtifactKey,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        project_id=SceneProjectId(project_id),
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(artifact_id),
            artifact_kind=ArtifactKind("geometry.solution"),
        ),
        artifact_key=key,
        producer=_producer(),
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )


def _materialization(
    artifact_ref: ArtifactRef,
    *,
    relative_path: str,
    expected: bytes,
) -> ArtifactMaterializationMetadata:
    return ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref,
        entries=(
            ArtifactMaterializationEntry(
                relative_path=relative_path,
                sha256=_sha256_bytes(expected),
                byte_length=len(expected),
            ),
        ),
    )


def _verified_candidates(
    store: SQLiteLocalStore,
    key: ArtifactKey,
    root: Path,
) -> tuple[ArtifactMetadata, ...]:
    """Test-only composition proving key equality alone is not reuse eligibility."""

    verified: list[ArtifactMetadata] = []
    for candidate in store.find_artifact_metadata_by_key(key):
        materialization = store.get_artifact_materialization(candidate.artifact_ref)
        if materialization is None:
            continue
        outcome = verify_local_artifact_materialization(materialization, root)
        if outcome.status is ArtifactMaterializationVerificationStatus.VERIFIED:
            verified.append(candidate)
    return tuple(verified)


def _database_rows(path: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(path) as connection:
        return connection.execute(
            "SELECT record_type, record_id, schema_version, payload_json "
            "FROM wre_records ORDER BY record_type, record_id"
        ).fetchall()


def test_exact_key_candidates_require_verified_materialization_before_fixture_reuse(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    root = tmp_path / "artifacts"
    root.mkdir()
    store = SQLiteLocalStore(database)
    key = _key()
    verified = _metadata(
        project_id="project:a",
        artifact_id="artifact:verified",
        key=key,
    )
    metadata_only = _metadata(
        project_id="project:b",
        artifact_id="artifact:metadata-only",
        key=key,
    )
    store.put_artifact_metadata(metadata_only)
    store.put_artifact_metadata(verified)

    payload = b"verified geometry bytes"
    verified_path = root / "verified" / "geometry.bin"
    verified_path.parent.mkdir()
    verified_path.write_bytes(payload)
    store.put_artifact_materialization(
        _materialization(
            verified.artifact_ref,
            relative_path="verified/geometry.bin",
            expected=payload,
        )
    )

    candidates = store.find_artifact_metadata_by_key(key)

    assert candidates == (verified, metadata_only)
    assert _verified_candidates(store, key, root) == (verified,)
    assert store.get_artifact_materialization(metadata_only.artifact_ref) is None


def test_missing_and_corrupted_materializations_never_satisfy_verified_precondition(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    root = tmp_path / "artifacts"
    root.mkdir()
    store = SQLiteLocalStore(database)
    key = _key()
    missing = _metadata(
        project_id="project:a",
        artifact_id="artifact:missing",
        key=key,
    )
    corrupted = _metadata(
        project_id="project:b",
        artifact_id="artifact:corrupted",
        key=key,
    )
    for candidate in (missing, corrupted):
        store.put_artifact_metadata(candidate)

    store.put_artifact_materialization(
        _materialization(
            missing.artifact_ref,
            relative_path="missing/output.bin",
            expected=b"expected missing bytes",
        )
    )
    corrupted_path = root / "corrupted" / "output.bin"
    corrupted_path.parent.mkdir()
    corrupted_path.write_bytes(b"wrong bytes")
    store.put_artifact_materialization(
        _materialization(
            corrupted.artifact_ref,
            relative_path="corrupted/output.bin",
            expected=b"expected correct bytes",
        )
    )

    missing_manifest = store.get_artifact_materialization(missing.artifact_ref)
    corrupted_manifest = store.get_artifact_materialization(corrupted.artifact_ref)

    assert missing_manifest is not None
    assert corrupted_manifest is not None
    assert (
        verify_local_artifact_materialization(missing_manifest, root).status
        is ArtifactMaterializationVerificationStatus.MISSING
    )
    assert (
        verify_local_artifact_materialization(corrupted_manifest, root).status
        is ArtifactMaterializationVerificationStatus.CORRUPTED
    )
    assert _verified_candidates(store, key, root) == ()


def test_changed_computation_identity_does_not_discover_stale_candidate(tmp_path: Path) -> None:
    store = SQLiteLocalStore(tmp_path / "state.sqlite3")
    previous_key = _key("a")
    changed_key = _key("b")
    stale = _metadata(
        project_id="project:one",
        artifact_id="artifact:old-computation",
        key=previous_key,
    )
    store.put_artifact_metadata(stale)

    assert changed_key != previous_key
    assert store.find_artifact_metadata_by_key(previous_key) == (stale,)
    assert store.find_artifact_metadata_by_key(changed_key) == ()


def test_post_verification_byte_corruption_preserves_persisted_expectations(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    root = tmp_path / "artifacts"
    root.mkdir()
    store = SQLiteLocalStore(database)
    candidate = _metadata(
        project_id="project:one",
        artifact_id="artifact:mutable-bytes",
        key=_key(),
    )
    store.put_artifact_metadata(candidate)
    expected = b"original artifact bytes"
    target = root / "candidate.bin"
    target.write_bytes(expected)
    manifest = _materialization(
        candidate.artifact_ref,
        relative_path="candidate.bin",
        expected=expected,
    )
    store.put_artifact_materialization(manifest)

    first = verify_local_artifact_materialization(manifest, root)
    persisted_before = store.get_artifact_materialization(candidate.artifact_ref)
    metadata_before = store.get_artifact_metadata(candidate.artifact_ref.artifact_id)
    target.write_bytes(b"corrupted after verification")
    second = verify_local_artifact_materialization(manifest, root)

    assert first.status is ArtifactMaterializationVerificationStatus.VERIFIED
    assert second.status is ArtifactMaterializationVerificationStatus.CORRUPTED
    assert (
        store.get_artifact_materialization(candidate.artifact_ref) == persisted_before == manifest
    )
    assert (
        store.get_artifact_metadata(candidate.artifact_ref.artifact_id)
        == metadata_before
        == candidate
    )


def test_invalidation_planning_is_downstream_scoped_and_non_destructive(
    tmp_path: Path,
) -> None:
    database = tmp_path / "state.sqlite3"
    root = tmp_path / "artifacts"
    root.mkdir()
    store = SQLiteLocalStore(database)

    raw = ArtifactRef(ArtifactId("artifact:raw"), ArtifactKind("media.image"))
    features = ArtifactRef(ArtifactId("artifact:features"), ArtifactKind("feature.set"))
    geometry = ArtifactRef(ArtifactId("artifact:geometry"), ArtifactKind("geometry.solution"))
    unrelated = ArtifactRef(ArtifactId("artifact:unrelated"), ArtifactKind("surface.model"))
    graph = ArtifactDependencyGraph(
        nodes=frozenset({raw, features, geometry, unrelated}),
        edges=frozenset(
            {
                ArtifactDependencyEdge(artifact=features, dependency=raw),
                ArtifactDependencyEdge(artifact=geometry, dependency=features),
            }
        ),
    )
    store.put_artifact_dependency_graph(graph)

    geometry_metadata = _metadata(
        project_id="project:geometry",
        artifact_id=geometry.artifact_id.value,
        key=_key(),
    )
    store.put_artifact_metadata(geometry_metadata)
    payload = b"geometry survives planning"
    target = root / "geometry.bin"
    target.write_bytes(payload)
    geometry_manifest = _materialization(
        geometry,
        relative_path="geometry.bin",
        expected=payload,
    )
    store.put_artifact_materialization(geometry_manifest)

    rows_before = _database_rows(database)
    bytes_before = target.read_bytes()
    plan = plan_artifact_invalidation(graph, frozenset({features}))
    rows_after = _database_rows(database)

    assert plan.affected_artifacts == frozenset({features, geometry})
    assert raw not in plan.affected_artifacts
    assert unrelated not in plan.affected_artifacts
    assert rows_after == rows_before
    assert target.read_bytes() == bytes_before
    assert store.get_artifact_metadata(geometry.artifact_id) == geometry_metadata
    assert store.get_artifact_materialization(geometry) == geometry_manifest
    assert store.get_artifact_dependencies(geometry) == frozenset({features})


def test_integrated_expectations_survive_store_reopen_deterministically(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    root = tmp_path / "artifacts"
    root.mkdir()
    key = _key()
    candidate = _metadata(
        project_id="project:reopen",
        artifact_id="artifact:reopen",
        key=key,
    )
    payload = b"reopen fixture"
    target = root / "reopen.bin"
    target.write_bytes(payload)
    manifest = _materialization(
        candidate.artifact_ref,
        relative_path="reopen.bin",
        expected=payload,
    )

    store = SQLiteLocalStore(database)
    store.put_artifact_metadata(candidate)
    store.put_artifact_materialization(manifest)
    store.put_artifact_dependency_graph(
        ArtifactDependencyGraph(nodes=frozenset({candidate.artifact_ref}), edges=frozenset())
    )

    reopened = SQLiteLocalStore(database)

    assert reopened.find_artifact_metadata_by_key(key) == (candidate,)
    assert reopened.get_artifact_materialization(candidate.artifact_ref) == manifest
    assert _verified_candidates(reopened, key, root) == (candidate,)
    assert reopened.get_artifact_dependencies(candidate.artifact_ref) == frozenset()
