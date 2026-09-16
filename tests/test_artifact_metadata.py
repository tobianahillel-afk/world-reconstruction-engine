from __future__ import annotations

import sqlite3
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
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
    CheckpointIdentity,
    ConfigurationIdentity,
    ImageObservation,
    MediaAssetRef,
    ModelIdentity,
    ObservationId,
    ProducerRef,
    ProvenanceClass,
    SceneProject,
    SceneProjectId,
    Sha256Digest,
    SourceId,
    SourceRef,
)
from wre.persistence import PersistenceConflictError, SQLiteLocalStore
from wre.persistence.codec import (
    decode_artifact_metadata,
    decode_scene_project,
    encode_artifact_metadata,
    encode_scene_project,
)


def _digest(character: str) -> Sha256Digest:
    return Sha256Digest(character * 64)


def _producer() -> ArtifactProducerIdentity:
    return ArtifactProducerIdentity(
        producer=ProducerRef(
            implementation="wre.adapters.example",
            version="1.2.3",
            revision="commit:abc",
        ),
        configuration=ConfigurationIdentity(sha256=_digest("a")),
        model=ModelIdentity(
            name="example.model",
            version="4.5",
            revision="commit:def",
        ),
        checkpoint=CheckpointIdentity(
            identifier="weights:quality",
            sha256=_digest("b"),
        ),
    )


def _metadata(
    *,
    project_id: str = "project:one",
    artifact_id: str = "artifact:geometry-01",
    artifact_kind: str = "geometry.solution",
    key_character: str = "c",
    provenance_class: ProvenanceClass = ProvenanceClass.OBSERVED_RECONSTRUCTED,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        project_id=SceneProjectId(project_id),
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId(artifact_id),
            artifact_kind=ArtifactKind(artifact_kind),
        ),
        artifact_key=ArtifactKey(sha256=_digest(key_character)),
        producer=_producer(),
        provenance_class=provenance_class,
    )


def _store(tmp_path: Path) -> SQLiteLocalStore:
    return SQLiteLocalStore(tmp_path / "state" / "wre.sqlite3")


def test_artifact_metadata_is_exact_typed_immutable_aggregate() -> None:
    metadata = _metadata()

    assert metadata.project_id == SceneProjectId("project:one")
    assert metadata.artifact_ref == ArtifactRef(
        artifact_id=ArtifactId("artifact:geometry-01"),
        artifact_kind=ArtifactKind("geometry.solution"),
    )
    assert metadata.artifact_key == ArtifactKey(sha256=_digest("c"))
    assert metadata.producer == _producer()
    assert metadata.provenance_class is ProvenanceClass.OBSERVED_RECONSTRUCTED
    assert hash(metadata) == hash(_metadata())

    with pytest.raises(FrozenInstanceError):
        metadata.project_id = SceneProjectId("project:other")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("project_id", "project:one", r"artifact_metadata\.project_id"),
        ("artifact_ref", "artifact:one", r"artifact_metadata\.artifact_ref"),
        ("artifact_key", "key", r"artifact_metadata\.artifact_key"),
        ("producer", "producer", r"artifact_metadata\.producer"),
        ("provenance_class", "generated", r"artifact_metadata\.provenance_class"),
    ],
)
def test_artifact_metadata_rejects_untyped_members(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "project_id": SceneProjectId("project:one"),
        "artifact_ref": ArtifactRef(
            artifact_id=ArtifactId("artifact:one"),
            artifact_kind=ArtifactKind("geometry.solution"),
        ),
        "artifact_key": ArtifactKey(sha256=_digest("c")),
        "producer": _producer(),
        "provenance_class": ProvenanceClass.OBSERVED_RECONSTRUCTED,
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match=message):
        ArtifactMetadata(**cast(Any, kwargs))


def test_scene_project_codec_round_trip_is_exact() -> None:
    project = SceneProject(project_id=SceneProjectId("project:codec"))

    payload = encode_scene_project(project)

    assert payload == {"project_id": "project:codec"}
    assert decode_scene_project(payload) == project


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"project_id": 7},
        {"project_id": "bad project id"},
    ],
)
def test_scene_project_codec_rejects_malformed_payload(payload: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        decode_scene_project(payload)


def test_artifact_metadata_codec_round_trip_preserves_exact_values() -> None:
    metadata = _metadata(key_character="f", provenance_class=ProvenanceClass.GENERATED)

    payload = encode_artifact_metadata(metadata)
    decoded = decode_artifact_metadata(payload)

    assert decoded == metadata
    assert decoded.artifact_key == ArtifactKey(sha256=_digest("f"))
    assert payload["artifact_key_sha256"] == "f" * 64
    assert payload["provenance_class"] == "generated"
    assert payload["project_id"] == "project:one"


def test_artifact_metadata_codec_preserves_native_null_model_and_checkpoint() -> None:
    native = ArtifactMetadata(
        project_id=SceneProjectId("project:native"),
        artifact_ref=ArtifactRef(
            artifact_id=ArtifactId("artifact:native"),
            artifact_kind=ArtifactKind("geometry.solution"),
        ),
        artifact_key=ArtifactKey(sha256=_digest("d")),
        producer=ArtifactProducerIdentity(
            producer=ProducerRef(
                implementation="wre.reconstruction.colmap",
                version="4.2.0",
                revision="adapter:1",
            ),
            configuration=ConfigurationIdentity(sha256=_digest("e")),
        ),
        provenance_class=ProvenanceClass.OBSERVED_RECONSTRUCTED,
    )

    payload = encode_artifact_metadata(native)
    producer_payload = cast(dict[str, object], payload["producer"])

    assert producer_payload["model"] is None
    assert producer_payload["checkpoint"] is None
    assert decode_artifact_metadata(payload) == native


def test_artifact_metadata_codec_fails_closed_on_malformed_payloads() -> None:
    valid = encode_artifact_metadata(_metadata())

    missing_key = dict(valid)
    missing_key.pop("artifact_key_sha256")
    with pytest.raises((TypeError, ValueError)):
        decode_artifact_metadata(missing_key)

    unknown_provenance = dict(valid)
    unknown_provenance["provenance_class"] = "unknown"
    with pytest.raises(ValueError):
        decode_artifact_metadata(unknown_provenance)

    malformed_ref = dict(valid)
    malformed_ref["artifact_ref"] = {"artifact_id": "artifact:one", "artifact_kind": 4}
    with pytest.raises((TypeError, ValueError)):
        decode_artifact_metadata(malformed_ref)

    checkpoint_without_model = dict(valid)
    producer = cast(dict[str, object], checkpoint_without_model["producer"])
    producer = dict(producer)
    producer["model"] = None
    checkpoint_without_model["producer"] = producer
    with pytest.raises(ValueError, match="checkpoint requires model identity"):
        decode_artifact_metadata(checkpoint_without_model)


def test_store_round_trips_project_and_artifact_metadata_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state" / "wre.sqlite3"
    project = SceneProject(project_id=SceneProjectId("project:persisted"))
    metadata = _metadata(project_id=project.project_id.value)

    store = SQLiteLocalStore(path)
    store.put_scene_project(project)
    store.put_artifact_metadata(metadata)

    reopened = SQLiteLocalStore(path)
    assert reopened.get_scene_project(project.project_id) == project
    assert reopened.get_artifact_metadata(metadata.artifact_ref.artifact_id) == metadata


def test_artifact_metadata_persistence_does_not_require_or_create_project_record(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    metadata = _metadata(project_id="project:not-stored")

    store.put_artifact_metadata(metadata)

    assert store.get_artifact_metadata(metadata.artifact_ref.artifact_id) == metadata
    assert store.get_scene_project(metadata.project_id) is None


def test_missing_v2_records_return_none(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.get_scene_project(SceneProjectId("project:missing")) is None
    assert store.get_artifact_metadata(ArtifactId("artifact:missing")) is None


def test_v2_identical_writes_are_idempotent_and_artifact_conflicts_fail_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    project = SceneProject(project_id=SceneProjectId("project:stable"))
    metadata = _metadata(project_id=project.project_id.value)
    conflicting = _metadata(
        project_id=project.project_id.value,
        artifact_id=metadata.artifact_ref.artifact_id.value,
        key_character="9",
        provenance_class=ProvenanceClass.INFERRED,
    )

    store.put_scene_project(project)
    store.put_scene_project(project)
    store.put_artifact_metadata(metadata)
    store.put_artifact_metadata(metadata)

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_artifact_metadata(conflicting)

    assert store.get_scene_project(project.project_id) == project
    assert store.get_artifact_metadata(metadata.artifact_ref.artifact_id) == metadata


def test_v2_records_use_dedicated_record_types_and_coexist_with_retained_v1(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "wre.sqlite3"
    store = SQLiteLocalStore(path)
    project = SceneProject(project_id=SceneProjectId("project:coexist"))
    metadata = _metadata(project_id=project.project_id.value)
    observation = ImageObservation(
        observation_id=ObservationId("obs:coexist"),
        asset=MediaAssetRef(
            uri="file:///coexist.jpg",
            sha256=_digest("8"),
            byte_length=128,
            mime_type="image/jpeg",
        ),
        source=SourceRef(source_id=SourceId("upload:coexist")),
        received_at=datetime(2026, 9, 16, 0, 0, tzinfo=UTC),
        captured_at=None,
    )

    store.put_observation(observation)
    store.put_scene_project(project)
    store.put_artifact_metadata(metadata)

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT record_type, record_id FROM wre_records ORDER BY record_type, record_id"
        ).fetchall()

    assert rows == [
        ("artifact_metadata", "artifact:geometry-01"),
        ("observation", "obs:coexist"),
        ("scene_project", "project:coexist"),
    ]
    assert store.get_observation(observation.observation_id) == observation
    assert store.get_scene_project(project.project_id) == project
    assert store.get_artifact_metadata(metadata.artifact_ref.artifact_id) == metadata
