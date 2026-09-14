# Media ingestion

Lot L2 turns externally supplied media into the raw-observation contracts defined by L1. The work is intentionally staged so byte handling, content identity, metadata interpretation and video/keyframe logic do not collapse into one opaque ingestion step.

## L2.1 image-ingestion boundary

L2.1 introduces the smallest deterministic bridge from an already-addressed image asset to the raw `ImageObservation` model and local persistence.

`ImageIngestRequest` carries only explicit inputs:

- `ObservationId`;
- an existing `MediaAssetRef`;
- `SourceRef`;
- timezone-aware `received_at`;
- optional timezone-aware `captured_at`.

`ImageIngestor` constructs an `ImageObservation` and writes it through the structural `ObservationSink` protocol. `SQLiteLocalStore` satisfies that protocol, but the ingestion service has no SQLite dependency and can be reused with another sink.

No wall clock is consulted. If capture time is unknown, it remains `None`; L2.1 does not substitute receipt time or infer a local timestamp.

## Why L2.1 requires an existing MediaAssetRef

The L1 observation contract deliberately requires media content identity (`Sha256Digest`), byte length and URI as part of `MediaAssetRef`. L2.1 does not fabricate or calculate that digest because the roadmap assigns file hashing and duplicate detection to L2.2.

This means L2.1 is the deterministic ingestion core for media that is already content-addressed by an upstream source or test fixture. L2.2 will add the workflow that reads raw media bytes, computes/verifies SHA-256 and performs duplicate detection before invoking this same ingestion boundary.

A supplied digest is therefore an input contract at L2.1, not evidence that WRE itself has independently verified the bytes. Code that needs WRE-calculated content identity must use the later L2.2 path once implemented.

## Identity and retry behavior

L2.1 inherits L1 persistence semantics:

1. a new observation ID and payload are stored atomically;
2. repeating the exact same observation is idempotent;
3. reusing the same observation ID with different content raises the persistence conflict instead of silently replacing the raw observation.

The ingestor does not weaken or reinterpret those rules.

## Explicitly deferred work

L2.1 does **not** implement:

- reading a local file to calculate SHA-256;
- hash indexes or duplicate detection;
- media discovery or directory crawling;
- byte copying into a managed media object store;
- image decoding or image-validity inference;
- EXIF/XMP parsing;
- GPS or capture-time interpretation;
- camera identity resolution;
- video ingestion or keyframe extraction;
- feature extraction, matching or reconstruction.

Those behaviors remain in their owning L2/L3 work items. L2.1 only creates a raw image observation from explicit already-addressed inputs and persists it through the established domain boundary.
