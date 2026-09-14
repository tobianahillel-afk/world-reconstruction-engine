# Local persistence

L1.5 provides a small local persistence boundary for the solver-independent L1 domain. It is intentionally not an ingestion database, reconstruction job queue, geometry cache or world graph.

## Storage model

The implementation uses Python's standard-library `sqlite3` module. No runtime dependency is added.

SQLite owns durability and transactions. Domain records are stored as canonical JSON payloads so serialization remains explicit and versioned rather than depending on Python dataclass internals or an ORM mapping.

The database contains two tables:

- `wre_meta` stores the database schema version.
- `wre_records` stores `(record_type, record_id, schema_version, payload_json)` with a composite primary key on `(record_type, record_id)`.

Database schema version and record schema version are currently `1`. Unsupported versions are rejected instead of being interpreted optimistically.

## Canonical serialization

Payload JSON is emitted with sorted keys, compact separators and UTF-8 characters preserved. Time instants are serialized as timezone-aware ISO-8601 values normalized to UTC with microsecond precision. Domain constructors are used again during decode so their invariants remain authoritative after persistence.

Explicit codecs currently cover:

- `ImageObservation` and `VideoFrameObservation`;
- `Camera`;
- `ObservationMetadata`;
- `SpatialFragment`;
- `ReconstructionRun`;
- `DerivedArtifactProvenance` as a composable value object.

`DerivedArtifactProvenance` has no standalone persistence identity in L1, so L1.5 defines its deterministic codec but does not invent a database record key for it.

## Identity and write semantics

Stable identities are protected against silent replacement.

A `put_*` operation behaves as follows:

1. no existing record with that `(record_type, record_id)` → insert atomically;
2. existing record with exactly the same canonical payload → idempotent success;
3. existing record with different content → `PersistenceConflictError`.

L1.5 deliberately does not add an implicit last-write-wins update policy. If later work requires mutable/versioned records, that policy must be introduced explicitly with appropriate provenance rather than hidden inside the storage layer.

## Transactions and reopen behavior

Writes use an SQLite transaction with `BEGIN IMMEDIATE`, so the identity check and insert are one local atomic operation. Connections are short-lived per operation, and a store reopened from the same database file reads the same records.

Missing records return `None`. Corrupted JSON, incompatible record versions and incompatible database versions fail explicitly.

## Boundary

L1.5 does **not** implement:

- file/media discovery or copying;
- image/video decoding;
- duplicate detection or ingestion hashing workflows;
- EXIF/XMP parsing;
- GPS/time interpretation;
- feature extraction, matching or reconstruction;
- solver artifact storage;
- job scheduling or run orchestration;
- global world geometry or fragment merging.

Those remain in their owning roadmap lots. The persistence layer stores L1 domain state; it does not change the epistemic meaning of that state.
