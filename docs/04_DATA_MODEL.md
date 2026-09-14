# Initial data model

The initial model is intentionally small. Add fields/types only when a work item requires them.

Planned core concepts:

- `Observation`
- `ImageObservation`
- `VideoObservation`
- `VideoFrameObservation`
- `Camera`
- `FeatureSet`
- `PairCandidate`
- `GeometricMatch`
- `Track`
- `CameraPoseEstimate`
- `SpatialFragment`
- `FragmentPlacement`
- `MergeHypothesis`
- `SpatialConstraint`
- `ReconstructionRun`
- `RouteDecision` (planned L4)
- absolute-anchor/GCP constraint types (planned L9)
- `TemporalEpoch` / temporal evidence (planned L15)
- `GeometryChangeHypothesis` (planned L15)
- `GeometryState` (planned L15)

Every derived object must be able to identify its source observations/evidence and producing run/version. IDs from different domains must not be casually interchangeable.

## L1.1 observation contract

The Observation layer represents source media made available to WRE. It is deliberately solver-independent and must not contain feature matches, camera poses, fragment placement or any other inferred geometry.

Implemented observation primitives:

- `ObservationId` — opaque typed identifier for an observation.
- `SourceId` / `SourceRef` — opaque source identity plus an optional source locator. This is intentionally smaller than the later provenance/run model.
- `Sha256Digest` — validated canonical content digest.
- `MediaAssetRef` — immutable reference to bytes by URI, SHA-256 digest, byte length and optional MIME type.
- `Observation` — abstract raw-observation base carrying an asset, source, receipt time and optional capture instant.
- `ImageObservation` — still-image observation.
- `VideoObservation` — whole source-video observation before any frame or keyframe extraction.
- `VideoFrameObservation` — image-like frame observation that additionally retains the parent video asset, deterministic frame index and integer microsecond offset.

`received_at` and `captured_at`, when present, are timezone-aware instants. An ambiguous local capture timestamp must not be coerced into an instant merely to populate the model; later metadata ingestion may retain the raw value until its time semantics are known.

Video frame offsets use integer microseconds rather than floating-point seconds so frame identity and ordering do not depend on accidental floating-point equality.

L2.5 makes the whole source video a first-class raw observation. That preserves source identity/provenance before any frame selection occurs. It does not imply that the bytes have been decoded or that container metadata, frame count, duration, codec, MIME type or capture time have been inferred. L2.6 later owns deterministic frame/keyframe extraction and creates `VideoFrameObservation` records that retain their parent `video_asset`.

### Explicit L1.1 boundary

Not part of the Observation model:

- EXIF parsing or raw metadata extraction;
- geographic metadata interpretation or georeferencing;
- camera intrinsics, calibration or camera identity;
- feature descriptors, matches, tracks or geometric verification;
- camera-pose estimates, fragments or placements;
- persistence/database representation;
- reconstruction-run provenance.

Those concepts remain in their owning later work items. A raw observation may eventually reference richer metadata/provenance objects, but L1.1 does not silently pull those later contracts forward.

## L1.2 camera and metadata contract

L1.2 adds solver-independent representation for camera identity and metadata that later ingest stages can populate. Representation is deliberately separate from parsing and inference.

Implemented primitives:

- `CameraId` — opaque typed identifier for a camera identity assigned by an explicit upstream policy.
- `Camera` — optional descriptive manufacturer/model/serial fields attached to a `CameraId`.
- `ImageDimensions` — positive integer width/height in pixels.
- `RawMetadataEntry` — one namespaced, uninterpreted string metadata value preserved as supplied.
- `ObservationMetadata` — immutable metadata associated with one `ObservationId`, optionally referencing a camera and image dimensions plus raw entries.

A `CameraId` is not derived automatically from manufacturer/model/serial strings. Two devices can share those strings, serials can be absent or unreliable, and metadata can be edited. Any later identity-resolution policy must make its evidence explicit rather than hiding an entity merge inside this model.

Raw metadata values are intentionally not normalized here. A local capture-time string can therefore be preserved verbatim even when its timezone semantics are unknown. Geographic or other device metadata likewise remains uninterpreted until the owning ingestion work item validates it.

### Explicit L1.2 boundary

Not implemented by this contract:

- reading EXIF/XMP/container metadata from media bytes;
- interpreting or normalizing geographic coordinates;
- converting ambiguous local timestamps into instants;
- inferring that two observations came from the same physical camera;
- camera intrinsics, distortion or solver camera models;
- calibration priors or pose priors;
- persistence/database schemas;
- `SpatialFragment` or estimated geometry.

Those behaviors remain in later work items. This keeps raw observations and source metadata usable independently of reconstruction engines and prevents metadata convenience from becoming hidden geometric truth.

## L1.3 SpatialFragment contract

A `SpatialFragment` is the solver-independent identity of one local spatial component and the set of raw observations that belong to it. It deliberately does not claim a world placement.

Implemented primitives:

- `SpatialFragmentId` — opaque typed fragment identity.
- `LocalFrameId` — opaque typed identity for the fragment's local coordinate frame.
- `SpatialFragment` — immutable fragment identity, local-frame identity and observation membership.

Observation membership has set semantics: it must be non-empty and duplicate-free. The stored tuple is sorted canonically by `ObservationId` so equality and later serialization do not depend on insertion order.

`LocalFrameId` names a local coordinate frame only. L1.3 does not claim that the frame is metric, Earth-aligned, georeferenced or compatible with another fragment's frame. Those properties require later evidence and transforms.

### Explicit L1.3 boundary

Not implemented by this contract:

- camera poses or 3D points inside the fragment;
- reconstruction solver output/import;
- global/Earth placement or geographic anchors;
- scale resolution between independent local frames;
- fragment lifecycle or automatic membership changes;
- fragment merge candidates or merge transforms;
- persistence/database representation;
- reconstruction-run provenance.

Those behaviors remain in their owning work items. A fragment may therefore exist as an internally named local component without any claim about where that component belongs in the global world.

## L1.4 ReconstructionRun and provenance contract

A `ReconstructionRun` records the immutable provenance envelope for one reconstruction computation. It describes who produced the computation, which raw observations were inputs, when it ran and, when available, the digest of the exact configuration representation used by the producer. It does not execute or schedule the computation.

Implemented primitives:

- `ReconstructionRunId` — opaque typed identity for one computation run.
- `ProducerRef` — implementation name plus required version and optional revision/build identity.
- `ReconstructionRun` — immutable run identity, producer, canonical raw-input membership, timezone-aware timing and optional configuration SHA-256.
- `DerivedArtifactProvenance` — composable reference from a derived artifact to its producing run and raw source observations.

Run input and provenance observation membership use set semantics: they are non-empty, duplicate-free and stored in canonical `ObservationId` order. `completed_at`, when present, cannot precede `started_at`.

`configuration_sha256` is a content digest only. L1.4 does not define configuration serialization or infer meaning from configuration bytes; the producer that supplies the digest must define the canonical representation being hashed.

`DerivedArtifactProvenance` records traceability, not truth. Referencing a run says which computation produced an artifact and which observations it ultimately depends on; it does not assert that the artifact is correct, accepted or promoted into a higher epistemic layer.

### Explicit L1.4 boundary

Not implemented by this contract:

- launching, scheduling or retrying reconstruction jobs;
- solver command lines or solver-private run objects;
- persistence/database schemas;
- artifact storage locations;
- run status machines or orchestration state;
- automatic acceptance of derived evidence or estimated geometry;
- local persistence or repository/database adapters.

Those behaviors remain in later work items. The core contract only guarantees reproducible identity and provenance links that later evidence/geometry models can embed without depending on a specific reconstruction engine.

## Planned L4 route-decision contract

The Strategy Router will eventually produce a solver-independent route decision such as FAST, STANDARD or ESCALATED. That decision is **derived orchestration evidence**, not geometry and not acceptance. It should be able to retain the considered signals, route reason, configuration/threshold identity and escalation outcome.

No route class may encode a lower truth/evidence threshold.

## Planned L9 absolute-anchor contracts

GPS, GCP/known-landmark and reference-fragment anchoring will be represented as world-graph constraints with provenance and uncertainty. A fragment remains valid in a local frame even when no absolute anchor exists.

Absolute anchors must not mutate a fragment into a world coordinate silently; the relation between local geometry and world frame remains an estimated constraint/placement with residuals.

## Planned L15 temporal / 4D contracts

The temporal layer will introduce concepts along these semantic lines (exact implementation fields belong to their work items):

### TemporalEpoch / temporal evidence

A period/grouping of observations or geometry used for cross-epoch analysis. Epoch membership can be uncertain; publication/upload time must not be silently treated as capture time.

### GeometryChangeHypothesis

An evidence-bearing hypothesis that geometry differs between epochs/states. It should retain compared geometry/state references, measured change evidence, candidate transition interval, residuals/uncertainty, supporting/contradicting observations, producer/version/configuration and status.

### GeometryState

A time-valid estimated-geometry state. It should be able to retain a stable state identity, spatial lineage/object/fragment reference, geometry product reference, validity interval (possibly uncertain), supporting evidence/provenance, uncertainty/confidence and transition/supersession relationships.

Historical geometry is never destructively overwritten merely because a later state is accepted. Multiple states can coexist when their validity intervals and evidence support physical change.

See [`13_TEMPORAL_WORLD.md`](13_TEMPORAL_WORLD.md) for the complete architectural contract.
