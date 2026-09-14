from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from wre.domain import (
    Camera,
    CameraId,
    DerivedArtifactProvenance,
    ImageDimensions,
    ImageObservation,
    LocalFrameId,
    MediaAssetRef,
    ObservationId,
    ObservationMetadata,
    ProducerRef,
    RawMetadataEntry,
    ReconstructionRun,
    ReconstructionRunId,
    Sha256Digest,
    SourceId,
    SourceRef,
    SpatialFragment,
    SpatialFragmentId,
    VideoFrameObservation,
)
from wre.persistence import (
    DATABASE_SCHEMA_VERSION,
    PersistenceConflictError,
    PersistenceError,
    SQLiteLocalStore,
    UnsupportedSchemaVersionError,
)
from wre.persistence.codec import (
    canonical_json,
    decode_derived_artifact_provenance,
    encode_derived_artifact_provenance,
)

NOW = datetime(2026, 9, 14, 19, 0, tzinfo=UTC)
SHA_A = "a" * 64
SHA_B = "b" * 64


def _asset(
    uri: str = "file:///observations/image.jpg",
    sha256: str = SHA_A,
    *,
    byte_length: int = 1024,
    mime_type: str | None = "image/jpeg",
) -> MediaAssetRef:
    return MediaAssetRef(
        uri=uri,
        sha256=Sha256Digest(sha256),
        byte_length=byte_length,
        mime_type=mime_type,
    )


def _source() -> SourceRef:
    return SourceRef(source_id=SourceId("upload:local"), locator="incoming/image.jpg")


def _image(observation_id: str = "obs:image:001") -> ImageObservation:
    return ImageObservation(
        observation_id=ObservationId(observation_id),
        asset=_asset(),
        source=_source(),
        received_at=NOW,
        captured_at=NOW - timedelta(seconds=3),
    )


def _store(tmp_path: Path) -> SQLiteLocalStore:
    return SQLiteLocalStore(tmp_path / "state" / "wre.sqlite3")


def test_image_observation_round_trip_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state" / "wre.sqlite3"
    observation = _image()

    store = SQLiteLocalStore(path)
    store.put_observation(observation)

    assert store.get_observation(observation.observation_id) == observation
    assert SQLiteLocalStore(path).get_observation(observation.observation_id) == observation


def test_video_frame_round_trip_preserves_parent_video_and_offset(tmp_path: Path) -> None:
    store = _store(tmp_path)
    observation = VideoFrameObservation(
        observation_id=ObservationId("obs:frame:012"),
        asset=_asset(
            uri="file:///frames/frame-0012.png",
            byte_length=40_000,
            mime_type="image/png",
        ),
        source=SourceRef(source_id=SourceId("video:import-1")),
        received_at=NOW,
        captured_at=None,
        video_asset=_asset(
            uri="file:///videos/source.mp4",
            sha256=SHA_B,
            byte_length=8_000_000,
            mime_type="video/mp4",
        ),
        frame_index=12,
        frame_time_us=400_000,
    )

    store.put_observation(observation)

    assert store.get_observation(observation.observation_id) == observation


def test_camera_and_observation_metadata_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    camera = Camera(
        camera_id=CameraId("camera:phone-01"),
        manufacturer="Example Corp",
        model="Model X",
        serial_number="ABC-123",
    )
    metadata = ObservationMetadata(
        observation_id=ObservationId("obs:image:001"),
        camera_id=camera.camera_id,
        dimensions=ImageDimensions(width_px=4032, height_px=3024),
        raw_entries=(
            RawMetadataEntry(namespace="exif", key="Make", value="Example Corp"),
            RawMetadataEntry(namespace="exif", key="DateTimeOriginal", value="2026:09:14 21:00:00"),
        ),
    )

    store.put_camera(camera)
    store.put_observation_metadata(metadata)

    assert store.get_camera(camera.camera_id) == camera
    assert store.get_observation_metadata(metadata.observation_id) == metadata


def test_spatial_fragment_round_trip_preserves_canonical_membership(tmp_path: Path) -> None:
    store = _store(tmp_path)
    fragment = SpatialFragment(
        fragment_id=SpatialFragmentId("fragment:001"),
        local_frame_id=LocalFrameId("frame:001"),
        observation_ids=(
            ObservationId("obs:c"),
            ObservationId("obs:a"),
            ObservationId("obs:b"),
        ),
    )

    store.put_spatial_fragment(fragment)
    loaded = store.get_spatial_fragment(fragment.fragment_id)

    assert loaded == fragment
    assert loaded is not None
    assert loaded.observation_ids == (
        ObservationId("obs:a"),
        ObservationId("obs:b"),
        ObservationId("obs:c"),
    )


def test_reconstruction_run_round_trip_canonicalizes_times_and_inputs(tmp_path: Path) -> None:
    store = _store(tmp_path)
    plus_two = timezone(timedelta(hours=2))
    started = datetime(2026, 9, 14, 21, 0, tzinfo=plus_two)
    run = ReconstructionRun(
        run_id=ReconstructionRunId("run:001"),
        producer=ProducerRef(
            implementation="wre.reconstruction.colmap",
            version="4.2.0",
            revision="adapter:1",
        ),
        input_observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
        started_at=started,
        completed_at=started + timedelta(seconds=4),
        configuration_sha256=Sha256Digest("c" * 64),
    )

    store.put_reconstruction_run(run)
    loaded = store.get_reconstruction_run(run.run_id)

    assert loaded == run
    assert loaded is not None
    assert loaded.started_at.tzinfo == UTC
    assert loaded.input_observation_ids == (ObservationId("obs:a"), ObservationId("obs:b"))


def test_missing_records_return_none(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.get_observation(ObservationId("obs:missing")) is None
    assert store.get_camera(CameraId("camera:missing")) is None
    assert store.get_observation_metadata(ObservationId("obs:missing")) is None
    assert store.get_spatial_fragment(SpatialFragmentId("fragment:missing")) is None
    assert store.get_reconstruction_run(ReconstructionRunId("run:missing")) is None


def test_identical_write_is_idempotent_but_conflicting_identity_is_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    original = Camera(camera_id=CameraId("camera:stable"), model="Model A")
    conflicting = Camera(camera_id=CameraId("camera:stable"), model="Model B")

    store.put_camera(original)
    store.put_camera(original)

    with pytest.raises(PersistenceConflictError, match="already has different content"):
        store.put_camera(conflicting)

    assert store.get_camera(original.camera_id) == original


def test_payload_json_is_canonical_and_database_schema_is_versioned(tmp_path: Path) -> None:
    store = _store(tmp_path)
    camera = Camera(
        camera_id=CameraId("camera:canonical"),
        manufacturer="Example",
        model="X",
    )
    store.put_camera(camera)

    with sqlite3.connect(store.path) as connection:
        schema_version = connection.execute(
            "SELECT value FROM wre_meta WHERE key = 'database_schema_version'"
        ).fetchone()
        payload_row = connection.execute(
            "SELECT payload_json FROM wre_records WHERE record_type = 'camera' AND record_id = ?",
            (camera.camera_id.value,),
        ).fetchone()

    assert schema_version == (str(DATABASE_SCHEMA_VERSION),)
    assert payload_row == (
        '{"camera_id":"camera:canonical","manufacturer":"Example","model":"X",'
        '"serial_number":null}',
    )
    assert payload_row[0] == canonical_json(
        {
            "serial_number": None,
            "model": "X",
            "manufacturer": "Example",
            "camera_id": "camera:canonical",
        }
    )


def test_unsupported_database_schema_is_rejected_before_record_use(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE wre_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute(
            "INSERT INTO wre_meta(key, value) VALUES('database_schema_version', '999')"
        )

    with pytest.raises(UnsupportedSchemaVersionError, match="999"):
        SQLiteLocalStore(path)


def test_unsupported_record_schema_and_corrupt_payload_are_rejected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    observation = _image("obs:image:corrupt")
    store.put_observation(observation)

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE wre_records
            SET schema_version = 999
            WHERE record_type = 'observation' AND record_id = ?
            """,
            (observation.observation_id.value,),
        )

    with pytest.raises(UnsupportedSchemaVersionError, match="999"):
        store.get_observation(observation.observation_id)

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            UPDATE wre_records
            SET schema_version = 1, payload_json = 'not-json'
            WHERE record_type = 'observation' AND record_id = ?
            """,
            (observation.observation_id.value,),
        )

    with pytest.raises(PersistenceError, match="payload is invalid"):
        store.get_observation(observation.observation_id)


def test_derived_artifact_provenance_codec_is_deterministic() -> None:
    provenance = DerivedArtifactProvenance(
        producing_run_id=ReconstructionRunId("run:001"),
        source_observation_ids=(ObservationId("obs:b"), ObservationId("obs:a")),
    )

    payload = encode_derived_artifact_provenance(provenance)
    decoded = decode_derived_artifact_provenance(payload)

    assert decoded == provenance
    assert canonical_json(payload) == (
        '{"producing_run_id":"run:001","source_observation_ids":["obs:a","obs:b"]}'
    )
