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
- GPS parsing or georeferencing;
- camera intrinsics, calibration or camera identity;
- feature descriptors, matches, tracks or geometric verification;
- camera-pose estimates, fragments or placements;
- persistence/database representation;
- reconstruction-run provenance.

Those concepts remain in their owning later work items. A raw observation may eventually reference richer metadata/provenance objects, but L1.1 does not silently pull those later contracts forward.
