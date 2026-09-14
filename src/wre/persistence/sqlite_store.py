from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from wre.domain.cameras import Camera, CameraId, ObservationMetadata
from wre.domain.fragments import SpatialFragment, SpatialFragmentId
from wre.domain.observations import Observation, ObservationId, Sha256Digest
from wre.domain.runs import ReconstructionRun, ReconstructionRunId
from wre.persistence.codec import (
    JsonObject,
    canonical_json,
    decode_camera,
    decode_observation,
    decode_observation_metadata,
    decode_reconstruction_run,
    decode_spatial_fragment,
    encode_camera,
    encode_observation,
    encode_observation_metadata,
    encode_reconstruction_run,
    encode_spatial_fragment,
    parse_json_object,
)

DATABASE_SCHEMA_VERSION = 1
RECORD_SCHEMA_VERSION = 1

T = TypeVar("T")


class PersistenceError(RuntimeError):
    """Base error for the local persistence boundary."""


class UnsupportedSchemaVersionError(PersistenceError):
    """Raised when the database or record schema is not supported."""


class PersistenceConflictError(PersistenceError):
    """Raised when one stable record identity is reused with different content."""


class SQLiteLocalStore:
    """Small transactional local store for immutable L1 domain records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if self.path.parent != Path(""):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS wre_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            row = connection.execute(
                "SELECT value FROM wre_meta WHERE key = 'database_schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO wre_meta(key, value) VALUES('database_schema_version', ?)",
                    (str(DATABASE_SCHEMA_VERSION),),
                )
            else:
                try:
                    version = int(row[0])
                except (TypeError, ValueError) as exc:
                    raise UnsupportedSchemaVersionError(
                        "database schema version metadata is invalid"
                    ) from exc
                if version != DATABASE_SCHEMA_VERSION:
                    raise UnsupportedSchemaVersionError(
                        f"database schema version {version} is unsupported; "
                        f"expected {DATABASE_SCHEMA_VERSION}"
                    )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS wre_records (
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY(record_type, record_id)
                )
                """
            )

    def _put(self, record_type: str, record_id: str, payload: Mapping[str, object]) -> None:
        payload_json = canonical_json(payload)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT schema_version, payload_json
                FROM wre_records
                WHERE record_type = ? AND record_id = ?
                """,
                (record_type, record_id),
            ).fetchone()
            if row is None:
                connection.execute(
                    """
                    INSERT INTO wre_records(record_type, record_id, schema_version, payload_json)
                    VALUES(?, ?, ?, ?)
                    """,
                    (record_type, record_id, RECORD_SCHEMA_VERSION, payload_json),
                )
                return

            existing_version, existing_payload = row
            if existing_version != RECORD_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"record schema version {existing_version} is unsupported for "
                    f"{record_type}:{record_id}"
                )
            if existing_payload != payload_json:
                raise PersistenceConflictError(
                    f"record identity {record_type}:{record_id} already has different content"
                )

    def _get_payload(self, record_type: str, record_id: str) -> JsonObject | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT schema_version, payload_json
                FROM wre_records
                WHERE record_type = ? AND record_id = ?
                """,
                (record_type, record_id),
            ).fetchone()
        if row is None:
            return None

        schema_version, payload_json = row
        if schema_version != RECORD_SCHEMA_VERSION:
            raise UnsupportedSchemaVersionError(
                f"record schema version {schema_version} is unsupported for "
                f"{record_type}:{record_id}"
            )
        if not isinstance(payload_json, str):
            raise PersistenceError(f"record payload is not text for {record_type}:{record_id}")
        try:
            return parse_json_object(payload_json)
        except ValueError as exc:
            raise PersistenceError(
                f"record payload is invalid for {record_type}:{record_id}"
            ) from exc

    def _get(
        self,
        record_type: str,
        record_id: str,
        decoder: Callable[[Mapping[str, object]], T],
    ) -> T | None:
        payload = self._get_payload(record_type, record_id)
        if payload is None:
            return None
        try:
            return decoder(payload)
        except (TypeError, ValueError) as exc:
            raise PersistenceError(
                f"record payload cannot be decoded for {record_type}:{record_id}"
            ) from exc

    def put_observation(self, observation: Observation) -> None:
        self._put("observation", observation.observation_id.value, encode_observation(observation))

    def get_observation(self, observation_id: ObservationId) -> Observation | None:
        return self._get("observation", observation_id.value, decode_observation)

    def find_observation_ids_by_content(
        self,
        *,
        sha256: Sha256Digest,
        byte_length: int,
    ) -> tuple[ObservationId, ...]:
        if isinstance(byte_length, bool) or not isinstance(byte_length, int) or byte_length < 0:
            raise ValueError("byte_length must be a non-negative integer")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_id, schema_version, payload_json
                FROM wre_records
                WHERE record_type = 'observation'
                ORDER BY record_id
                """
            ).fetchall()

        matches: list[ObservationId] = []
        for record_id, schema_version, payload_json in rows:
            if schema_version != RECORD_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"record schema version {schema_version} is unsupported for "
                    f"observation:{record_id}"
                )
            if not isinstance(record_id, str) or not isinstance(payload_json, str):
                raise PersistenceError(
                    "observation duplicate index encountered invalid storage data"
                )
            try:
                observation = decode_observation(parse_json_object(payload_json))
            except (TypeError, ValueError) as exc:
                raise PersistenceError(
                    f"record payload cannot be decoded for observation:{record_id}"
                ) from exc
            if observation.observation_id.value != record_id:
                raise PersistenceError(
                    f"record identity observation:{record_id} does not match its payload"
                )
            if (
                observation.asset.sha256 == sha256
                and observation.asset.byte_length == byte_length
            ):
                matches.append(observation.observation_id)

        return tuple(matches)

    def put_camera(self, camera: Camera) -> None:
        self._put("camera", camera.camera_id.value, encode_camera(camera))

    def get_camera(self, camera_id: CameraId) -> Camera | None:
        return self._get("camera", camera_id.value, decode_camera)

    def put_observation_metadata(self, metadata: ObservationMetadata) -> None:
        self._put(
            "observation_metadata",
            metadata.observation_id.value,
            encode_observation_metadata(metadata),
        )

    def get_observation_metadata(self, observation_id: ObservationId) -> ObservationMetadata | None:
        return self._get(
            "observation_metadata",
            observation_id.value,
            decode_observation_metadata,
        )

    def put_spatial_fragment(self, fragment: SpatialFragment) -> None:
        self._put(
            "spatial_fragment",
            fragment.fragment_id.value,
            encode_spatial_fragment(fragment),
        )

    def get_spatial_fragment(self, fragment_id: SpatialFragmentId) -> SpatialFragment | None:
        return self._get("spatial_fragment", fragment_id.value, decode_spatial_fragment)

    def put_reconstruction_run(self, run: ReconstructionRun) -> None:
        self._put("reconstruction_run", run.run_id.value, encode_reconstruction_run(run))

    def get_reconstruction_run(self, run_id: ReconstructionRunId) -> ReconstructionRun | None:
        return self._get("reconstruction_run", run_id.value, decode_reconstruction_run)
