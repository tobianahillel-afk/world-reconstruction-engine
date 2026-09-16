from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from wre.domain.artifact_graph import ArtifactDependencyGraph
from wre.domain.artifact_keys import ArtifactKey
from wre.domain.artifact_materialization import ArtifactMaterializationMetadata
from wre.domain.artifact_metadata import ArtifactMetadata
from wre.domain.artifacts import ArtifactId, ArtifactRef
from wre.domain.cameras import Camera, CameraId, ObservationMetadata
from wre.domain.fragments import SpatialFragment, SpatialFragmentId
from wre.domain.metadata import ObservationMetadataInterpretation
from wre.domain.observations import Observation, ObservationId, Sha256Digest
from wre.domain.projects import SceneProject, SceneProjectId
from wre.domain.runs import ReconstructionRun, ReconstructionRunId
from wre.persistence.codec import (
    JsonObject,
    canonical_json,
    decode_artifact_dependencies,
    decode_artifact_materialization,
    decode_artifact_metadata,
    decode_camera,
    decode_metadata_interpretation,
    decode_observation,
    decode_observation_metadata,
    decode_reconstruction_run,
    decode_scene_project,
    decode_spatial_fragment,
    encode_artifact_dependencies,
    encode_artifact_materialization,
    encode_artifact_metadata,
    encode_camera,
    encode_metadata_interpretation,
    encode_observation,
    encode_observation_metadata,
    encode_reconstruction_run,
    encode_scene_project,
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
    """Small transactional local store for immutable solver-independent domain records."""

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
        self._put_many(((record_type, record_id, payload),))

    def _put_many(
        self,
        records: tuple[tuple[str, str, Mapping[str, object]], ...],
    ) -> None:
        canonical_records = tuple(
            (record_type, record_id, canonical_json(payload))
            for record_type, record_id, payload in records
        )
        if not canonical_records:
            return

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            for record_type, record_id, payload_json in canonical_records:
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
                        INSERT INTO wre_records(
                            record_type,
                            record_id,
                            schema_version,
                            payload_json
                        )
                        VALUES(?, ?, ?, ?)
                        """,
                        (record_type, record_id, RECORD_SCHEMA_VERSION, payload_json),
                    )
                    continue

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
            if observation.asset.sha256 == sha256 and observation.asset.byte_length == byte_length:
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

    def put_metadata_interpretation(
        self,
        interpretation: ObservationMetadataInterpretation,
    ) -> None:
        self._put(
            "metadata_interpretation",
            interpretation.observation_id.value,
            encode_metadata_interpretation(interpretation),
        )

    def get_metadata_interpretation(
        self,
        observation_id: ObservationId,
    ) -> ObservationMetadataInterpretation | None:
        return self._get(
            "metadata_interpretation",
            observation_id.value,
            decode_metadata_interpretation,
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

    def put_scene_project(self, project: SceneProject) -> None:
        self._put("scene_project", project.project_id.value, encode_scene_project(project))

    def get_scene_project(self, project_id: SceneProjectId) -> SceneProject | None:
        return self._get("scene_project", project_id.value, decode_scene_project)

    def put_artifact_metadata(self, metadata: ArtifactMetadata) -> None:
        self._put(
            "artifact_metadata",
            metadata.artifact_ref.artifact_id.value,
            encode_artifact_metadata(metadata),
        )

    def get_artifact_metadata(self, artifact_id: ArtifactId) -> ArtifactMetadata | None:
        return self._get("artifact_metadata", artifact_id.value, decode_artifact_metadata)

    def put_artifact_materialization(self, metadata: ArtifactMaterializationMetadata) -> None:
        if not isinstance(metadata, ArtifactMaterializationMetadata):
            raise TypeError("metadata must be ArtifactMaterializationMetadata")
        self._put(
            "artifact_materialization",
            metadata.artifact_ref.artifact_id.value,
            encode_artifact_materialization(metadata),
        )

    def get_artifact_materialization(
        self,
        artifact_ref: ArtifactRef,
    ) -> ArtifactMaterializationMetadata | None:
        if not isinstance(artifact_ref, ArtifactRef):
            raise TypeError("artifact_ref must be ArtifactRef")
        metadata = self._get(
            "artifact_materialization",
            artifact_ref.artifact_id.value,
            decode_artifact_materialization,
        )
        if metadata is None:
            return None
        if metadata.artifact_ref != artifact_ref:
            raise PersistenceError(
                "artifact materialization record identity does not match the requested ArtifactRef"
            )
        return metadata

    def find_artifact_metadata_by_key(
        self,
        artifact_key: ArtifactKey,
    ) -> tuple[ArtifactMetadata, ...]:
        if not isinstance(artifact_key, ArtifactKey):
            raise TypeError("artifact_key must be ArtifactKey")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_id, schema_version, payload_json
                FROM wre_records
                WHERE record_type = 'artifact_metadata'
                ORDER BY record_id
                """
            ).fetchall()

        matches: list[ArtifactMetadata] = []
        for record_id, schema_version, payload_json in rows:
            if schema_version != RECORD_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"record schema version {schema_version} is unsupported for "
                    f"artifact_metadata:{record_id}"
                )
            if not isinstance(record_id, str) or not isinstance(payload_json, str):
                raise PersistenceError(
                    "artifact metadata cache lookup encountered invalid storage data"
                )
            try:
                metadata = decode_artifact_metadata(parse_json_object(payload_json))
            except (TypeError, ValueError) as exc:
                raise PersistenceError(
                    f"record payload cannot be decoded for artifact_metadata:{record_id}"
                ) from exc
            if metadata.artifact_ref.artifact_id.value != record_id:
                raise PersistenceError(
                    f"record identity artifact_metadata:{record_id} does not match its payload"
                )
            if metadata.artifact_key == artifact_key:
                matches.append(metadata)

        matches.sort(
            key=lambda item: (
                item.project_id.value,
                item.artifact_ref.artifact_id.value,
                item.artifact_ref.artifact_kind.value,
            )
        )
        return tuple(matches)

    def put_artifact_dependency_graph(self, graph: ArtifactDependencyGraph) -> None:
        if not isinstance(graph, ArtifactDependencyGraph):
            raise TypeError("graph must be ArtifactDependencyGraph")

        dependencies_by_node: dict[ArtifactRef, set[ArtifactRef]] = {
            node: set() for node in graph.nodes
        }
        for edge in graph.edges:
            dependencies_by_node[edge.artifact].add(edge.dependency)

        records = tuple(
            (
                "artifact_dependencies",
                node.artifact_id.value,
                encode_artifact_dependencies(node, frozenset(dependencies_by_node[node])),
            )
            for node in sorted(
                graph.nodes,
                key=lambda item: (item.artifact_id.value, item.artifact_kind.value),
            )
        )
        self._put_many(records)

    def get_artifact_dependencies(
        self,
        artifact_ref: ArtifactRef,
    ) -> frozenset[ArtifactRef] | None:
        if not isinstance(artifact_ref, ArtifactRef):
            raise TypeError("artifact_ref must be ArtifactRef")

        stored = self._get(
            "artifact_dependencies",
            artifact_ref.artifact_id.value,
            decode_artifact_dependencies,
        )
        if stored is None:
            return None
        stored_artifact_ref, dependencies = stored
        if stored_artifact_ref != artifact_ref:
            raise PersistenceError(
                "artifact dependency record identity does not match the requested ArtifactRef"
            )
        return dependencies

    def find_direct_artifact_dependents(
        self,
        dependency_ref: ArtifactRef,
    ) -> frozenset[ArtifactRef]:
        if not isinstance(dependency_ref, ArtifactRef):
            raise TypeError("dependency_ref must be ArtifactRef")

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT record_id, schema_version, payload_json
                FROM wre_records
                WHERE record_type = 'artifact_dependencies'
                ORDER BY record_id
                """
            ).fetchall()

        dependents: set[ArtifactRef] = set()
        for record_id, schema_version, payload_json in rows:
            if schema_version != RECORD_SCHEMA_VERSION:
                raise UnsupportedSchemaVersionError(
                    f"record schema version {schema_version} is unsupported for "
                    f"artifact_dependencies:{record_id}"
                )
            if not isinstance(record_id, str) or not isinstance(payload_json, str):
                raise PersistenceError(
                    "artifact dependency lookup encountered invalid storage data"
                )
            try:
                artifact_ref, dependencies = decode_artifact_dependencies(
                    parse_json_object(payload_json)
                )
            except (TypeError, ValueError) as exc:
                raise PersistenceError(
                    f"record payload cannot be decoded for artifact_dependencies:{record_id}"
                ) from exc
            if artifact_ref.artifact_id.value != record_id:
                raise PersistenceError(
                    f"record identity artifact_dependencies:{record_id} does not match its payload"
                )
            if dependency_ref in dependencies:
                dependents.add(artifact_ref)

        return frozenset(dependents)
