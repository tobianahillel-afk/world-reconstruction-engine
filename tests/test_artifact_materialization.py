from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any, cast

import pytest

from wre.domain import (
    ArtifactId,
    ArtifactKind,
    ArtifactMaterializationEntry,
    ArtifactMaterializationMetadata,
    ArtifactMaterializationVerification,
    ArtifactMaterializationVerificationStatus,
    ArtifactRef,
    Sha256Digest,
)
from wre.materialization import verify_local_artifact_materialization
from wre.persistence import (
    PersistenceConflictError,
    PersistenceError,
    SQLiteLocalStore,
    UnsupportedSchemaVersionError,
)
from wre.persistence.codec import (
    canonical_json,
    decode_artifact_materialization,
    encode_artifact_materialization,
)


def _digest_bytes(value: bytes) -> Sha256Digest:
    return Sha256Digest(hashlib.sha256(value).hexdigest())


def _ref(
    artifact_id: str = "artifact:materialized",
    artifact_kind: str = "geometry.solution",
) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=ArtifactId(artifact_id),
        artifact_kind=ArtifactKind(artifact_kind),
    )


def _entry(path: str, value: bytes = b"payload") -> ArtifactMaterializationEntry:
    return ArtifactMaterializationEntry(
        relative_path=path,
        sha256=_digest_bytes(value),
        byte_length=len(value),
    )


def _metadata(
    *entries: ArtifactMaterializationEntry,
    artifact_ref: ArtifactRef | None = None,
) -> ArtifactMaterializationMetadata:
    return ArtifactMaterializationMetadata(
        artifact_ref=artifact_ref or _ref(),
        entries=tuple(entries) if entries else (_entry("output.bin"),),
    )


def _store_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "wre.sqlite3"


def test_materialization_entry_is_immutable_typed_and_hashable() -> None:
    entry = _entry("nested/output.bin")

    assert hash(entry) == hash(_entry("nested/output.bin"))
    with pytest.raises(FrozenInstanceError):
        entry.relative_path = "other.bin"  # type: ignore[misc]
    with pytest.raises(TypeError, match="sha256 must be Sha256Digest"):
        ArtifactMaterializationEntry(
            relative_path="output.bin",
            sha256=cast(Any, "0" * 64),
            byte_length=1,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        ArtifactMaterializationEntry(
            relative_path="output.bin",
            sha256=_digest_bytes(b"x"),
            byte_length=-1,
        )
    with pytest.raises(ValueError, match="non-negative integer"):
        ArtifactMaterializationEntry(
            relative_path="output.bin",
            sha256=_digest_bytes(b"x"),
            byte_length=cast(Any, True),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        "/absolute.bin",
        "./output.bin",
        "nested/./output.bin",
        "../output.bin",
        "nested/../output.bin",
        "nested//output.bin",
        "nested/output.bin/",
        "nested\\output.bin",
        "C:/output.bin",
        "C:\\output.bin",
        "nul\x00byte.bin",
    ],
)
def test_materialization_entry_rejects_noncanonical_or_unsafe_paths(relative_path: str) -> None:
    with pytest.raises(ValueError):
        ArtifactMaterializationEntry(
            relative_path=relative_path,
            sha256=_digest_bytes(b"x"),
            byte_length=1,
        )


def test_materialization_metadata_requires_sorted_unique_immutable_entries() -> None:
    first = _entry("a.bin", b"a")
    second = _entry("b.bin", b"b")
    metadata = _metadata(first, second)

    assert metadata.entries == (first, second)
    assert hash(metadata) == hash(_metadata(first, second))

    with pytest.raises(TypeError, match="entries must be an immutable tuple"):
        ArtifactMaterializationMetadata(
            artifact_ref=_ref(),
            entries=cast(Any, [first]),
        )
    with pytest.raises(ValueError, match="at least one entry"):
        ArtifactMaterializationMetadata(artifact_ref=_ref(), entries=())
    with pytest.raises(TypeError, match="ArtifactMaterializationEntry values"):
        ArtifactMaterializationMetadata(
            artifact_ref=_ref(),
            entries=cast(Any, ("a.bin",)),
        )
    with pytest.raises(ValueError, match="paths must be unique"):
        _metadata(first, first)
    with pytest.raises(ValueError, match="sorted by relative path"):
        _metadata(second, first)


def test_materialization_verification_status_is_closed_and_stable() -> None:
    assert tuple(ArtifactMaterializationVerificationStatus) == (
        ArtifactMaterializationVerificationStatus.VERIFIED,
        ArtifactMaterializationVerificationStatus.MISSING,
        ArtifactMaterializationVerificationStatus.CORRUPTED,
    )
    assert [status.value for status in ArtifactMaterializationVerificationStatus] == [
        "verified",
        "missing",
        "corrupted",
    ]


def test_verifier_returns_verified_for_exact_single_and_multi_file_content(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "a.bin").write_bytes(b"alpha")
    nested = root / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"beta")
    metadata = _metadata(
        _entry("a.bin", b"alpha"),
        _entry("nested/b.bin", b"beta"),
    )

    result = verify_local_artifact_materialization(metadata, root)

    assert result == ArtifactMaterializationVerification(
        artifact_ref=metadata.artifact_ref,
        status=ArtifactMaterializationVerificationStatus.VERIFIED,
    )


def test_verifier_returns_missing_when_declared_file_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    metadata = _metadata(_entry("missing.bin", b"expected"))

    result = verify_local_artifact_materialization(metadata, root)

    assert result.status is ArtifactMaterializationVerificationStatus.MISSING


def test_verifier_reports_corrupted_for_length_or_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    target = root / "output.bin"
    target.write_bytes(b"observed")

    wrong_length = ArtifactMaterializationMetadata(
        artifact_ref=_ref("artifact:length"),
        entries=(
            ArtifactMaterializationEntry(
                relative_path="output.bin",
                sha256=_digest_bytes(b"observed"),
                byte_length=999,
            ),
        ),
    )
    wrong_digest = ArtifactMaterializationMetadata(
        artifact_ref=_ref("artifact:digest"),
        entries=(
            ArtifactMaterializationEntry(
                relative_path="output.bin",
                sha256=_digest_bytes(b"different"),
                byte_length=len(b"observed"),
            ),
        ),
    )

    assert (
        verify_local_artifact_materialization(wrong_length, root).status
        is ArtifactMaterializationVerificationStatus.CORRUPTED
    )
    assert (
        verify_local_artifact_materialization(wrong_digest, root).status
        is ArtifactMaterializationVerificationStatus.CORRUPTED
    )


def test_corruption_takes_precedence_over_missing_in_multi_file_manifest(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "corrupt.bin").write_bytes(b"wrong")
    metadata = _metadata(
        _entry("corrupt.bin", b"expected"),
        _entry("missing.bin", b"missing"),
    )

    result = verify_local_artifact_materialization(metadata, root)

    assert result.status is ArtifactMaterializationVerificationStatus.CORRUPTED


def test_verifier_rejects_non_regular_target_and_non_directory_root(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    (root / "directory.bin").mkdir()
    metadata = _metadata(_entry("directory.bin", b"ignored"))

    assert (
        verify_local_artifact_materialization(metadata, root).status
        is ArtifactMaterializationVerificationStatus.CORRUPTED
    )

    root_file = tmp_path / "root-file"
    root_file.write_bytes(b"not-a-directory")
    assert (
        verify_local_artifact_materialization(_metadata(), root_file).status
        is ArtifactMaterializationVerificationStatus.CORRUPTED
    )


def test_verifier_fails_closed_when_symlink_resolves_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"payload")
    (root / "escape.bin").symlink_to(outside)
    metadata = _metadata(_entry("escape.bin", b"payload"))

    result = verify_local_artifact_materialization(metadata, root)

    assert result.status is ArtifactMaterializationVerificationStatus.CORRUPTED


def test_verification_is_read_only_for_bytes_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "materialized"
    root.mkdir()
    target = root / "output.bin"
    target.write_bytes(b"payload")
    metadata = _metadata(_entry("output.bin", b"payload"))
    before_bytes = target.read_bytes()
    before_metadata = metadata

    result = verify_local_artifact_materialization(metadata, root)

    assert result.status is ArtifactMaterializationVerificationStatus.VERIFIED
    assert target.read_bytes() == before_bytes
    assert metadata == before_metadata


def test_verifier_rejects_untyped_public_inputs(tmp_path: Path) -> None:
    metadata = _metadata()

    with pytest.raises(TypeError, match="metadata must be ArtifactMaterializationMetadata"):
        verify_local_artifact_materialization(cast(Any, "metadata"), tmp_path)
    with pytest.raises(TypeError, match="root must be pathlib.Path"):
        verify_local_artifact_materialization(metadata, cast(Any, str(tmp_path)))


def test_materialization_codec_round_trips_deterministically() -> None:
    metadata = _metadata(
        _entry("a.bin", b"a"),
        _entry("nested/b.bin", b"beta"),
    )

    encoded = encode_artifact_materialization(metadata)

    assert encoded == {
        "artifact_ref": {
            "artifact_id": "artifact:materialized",
            "artifact_kind": "geometry.solution",
        },
        "entries": [
            {
                "byte_length": 1,
                "relative_path": "a.bin",
                "sha256": _digest_bytes(b"a").value,
            },
            {
                "byte_length": 4,
                "relative_path": "nested/b.bin",
                "sha256": _digest_bytes(b"beta").value,
            },
        ],
    }
    assert decode_artifact_materialization(encoded) == metadata
    assert canonical_json(encoded) == canonical_json(encode_artifact_materialization(metadata))


def test_materialization_codec_rejects_duplicate_unsorted_and_malformed_entries() -> None:
    first = _entry("a.bin", b"a")
    second = _entry("b.bin", b"b")
    encoded = encode_artifact_materialization(_metadata(first, second))

    duplicate = dict(encoded)
    duplicate["entries"] = [encoded["entries"][0], encoded["entries"][0]]  # type: ignore[index]
    with pytest.raises(ValueError, match="paths must be unique"):
        decode_artifact_materialization(duplicate)

    unsorted = dict(encoded)
    unsorted["entries"] = list(reversed(encoded["entries"]))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sorted by relative path"):
        decode_artifact_materialization(unsorted)

    malformed = dict(encoded)
    malformed["entries"] = [
        {
            "byte_length": True,
            "relative_path": "a.bin",
            "sha256": _digest_bytes(b"a").value,
        }
    ]
    with pytest.raises(ValueError, match="must be an integer"):
        decode_artifact_materialization(malformed)


def test_store_put_get_reopen_and_idempotent_materialization_metadata(tmp_path: Path) -> None:
    path = _store_path(tmp_path)
    metadata = _metadata(
        _entry("a.bin", b"a"),
        _entry("nested/b.bin", b"beta"),
    )
    store = SQLiteLocalStore(path)

    assert store.get_artifact_materialization(metadata.artifact_ref) is None
    store.put_artifact_materialization(metadata)
    store.put_artifact_materialization(metadata)
    assert store.get_artifact_materialization(metadata.artifact_ref) == metadata

    reopened = SQLiteLocalStore(path)
    assert reopened.get_artifact_materialization(metadata.artifact_ref) == metadata


def test_store_materialization_conflicts_on_any_content_change(tmp_path: Path) -> None:
    store = SQLiteLocalStore(_store_path(tmp_path))
    original = _metadata(_entry("output.bin", b"first"))
    store.put_artifact_materialization(original)

    changed_digest = _metadata(_entry("output.bin", b"second"))
    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_artifact_materialization(changed_digest)

    changed_kind = _metadata(
        _entry("output.bin", b"first"),
        artifact_ref=_ref(artifact_kind="surface.model"),
    )
    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_artifact_materialization(changed_kind)


def test_store_materialization_does_not_require_other_v2_records(tmp_path: Path) -> None:
    path = _store_path(tmp_path)
    metadata = _metadata()
    store = SQLiteLocalStore(path)

    store.put_artifact_materialization(metadata)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT record_type, record_id FROM wre_records ORDER BY record_type, record_id"
        ).fetchall()
    assert rows == [("artifact_materialization", metadata.artifact_ref.artifact_id.value)]


def test_store_materialization_lookup_fails_closed_on_kind_mismatch(tmp_path: Path) -> None:
    store = SQLiteLocalStore(_store_path(tmp_path))
    metadata = _metadata()
    store.put_artifact_materialization(metadata)
    wrong_kind = ArtifactRef(
        artifact_id=metadata.artifact_ref.artifact_id,
        artifact_kind=ArtifactKind("surface.model"),
    )

    with pytest.raises(PersistenceError, match="does not match the requested ArtifactRef"):
        store.get_artifact_materialization(wrong_kind)


def test_store_materialization_fails_closed_on_corrupted_payload(tmp_path: Path) -> None:
    path = _store_path(tmp_path)
    store = SQLiteLocalStore(path)
    artifact_ref = _ref("artifact:corrupt")

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_materialization', ?, 1, ?)
            """,
            (artifact_ref.artifact_id.value, canonical_json({"entries": []})),
        )

    with pytest.raises(PersistenceError, match="cannot be decoded"):
        store.get_artifact_materialization(artifact_ref)


def test_store_materialization_fails_closed_on_unsupported_record_version(tmp_path: Path) -> None:
    path = _store_path(tmp_path)
    store = SQLiteLocalStore(path)
    metadata = _metadata(artifact_ref=_ref("artifact:version"))

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_materialization', ?, 999, ?)
            """,
            (
                metadata.artifact_ref.artifact_id.value,
                canonical_json(encode_artifact_materialization(metadata)),
            ),
        )

    with pytest.raises(UnsupportedSchemaVersionError, match="unsupported"):
        store.get_artifact_materialization(metadata.artifact_ref)


def test_store_materialization_fails_closed_on_record_id_payload_mismatch(tmp_path: Path) -> None:
    path = _store_path(tmp_path)
    store = SQLiteLocalStore(path)
    requested = _ref("artifact:row-key")
    payload_metadata = _metadata(artifact_ref=_ref("artifact:payload"))

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
            VALUES('artifact_materialization', ?, 1, ?)
            """,
            (
                requested.artifact_id.value,
                canonical_json(encode_artifact_materialization(payload_metadata)),
            ),
        )

    with pytest.raises(PersistenceError, match="does not match the requested ArtifactRef"):
        store.get_artifact_materialization(requested)


def test_store_materialization_rejects_untyped_public_inputs(tmp_path: Path) -> None:
    store = SQLiteLocalStore(_store_path(tmp_path))

    with pytest.raises(TypeError, match="metadata must be ArtifactMaterializationMetadata"):
        store.put_artifact_materialization(cast(Any, "metadata"))
    with pytest.raises(TypeError, match="artifact_ref must be ArtifactRef"):
        store.get_artifact_materialization(cast(Any, "artifact:materialized"))
