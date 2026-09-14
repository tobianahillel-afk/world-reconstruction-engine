# Initial data model

The initial model is intentionally small. Add fields/types only when a work item requires them.

Planned core concepts:

- `Observation`
- `ImageObservation`
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
- `VideoFrameObservation` — image-like frame observation that additionally retains the parent video asset, deterministic frame index and integer microsecond offset.

`received_at` and `captured_at`, when present, are timezone-aware instants. An ambiguous local capture timestamp must not be coerced into an instant merely to populate the model; later metadata ingestion may retain the raw value until its time semantics are known.

Video frame offsets use integer microseconds rather than floating-point seconds so frame identity and ordering do not depend on accidental floating-point equality.

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
