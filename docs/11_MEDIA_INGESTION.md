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

This means L2.1 is the deterministic ingestion core for media that is already content-addressed by an upstream source or test fixture. L2.2 adds the workflow that reads raw media bytes, computes SHA-256 and performs duplicate detection before invoking this same ingestion boundary.

A supplied digest is therefore an input contract at L2.1, not evidence that WRE itself has independently verified the bytes. Code that needs WRE-calculated content identity uses the L2.2 local-file path.

## L2.2 hashing and duplicate detection

`hash_file_content` reads a local file in bounded chunks using Python's standard-library SHA-256 implementation. The digest and exact byte length are produced from the same read, so the resulting `MediaAssetRef` is grounded in the bytes WRE actually consumed rather than caller-supplied metadata.

`LocalImageIngestor` composes four existing boundaries instead of replacing them:

1. resolve the requested local file and fail before persistence if it does not exist;
2. stream the bytes through SHA-256 while counting their exact length;
3. ask a `DuplicateObservationLookup` for already-persisted observations with the same digest and byte length;
4. build a verified `MediaAssetRef` and delegate persistence to the L2.1 `ImageIngestor`.

The default chunk size is one MiB and can be overridden with a positive integer. The implementation never needs to load the whole media asset into memory.

### Duplicate semantics

A content duplicate means **same SHA-256 and same byte length**. Duplicate observation IDs are returned deterministically in canonical ID order.

Duplicate detection is not observation deduplication. Two observations may contain identical bytes while carrying different source, receipt-time or later metadata evidence. L2.2 therefore reports prior exact-content matches but still stores the new observation when its `ObservationId` is distinct. A retry of the same observation excludes its own ID from the duplicate list and remains idempotent through the L1 persistence contract.

This distinction preserves provenance: equal bytes do not silently erase distinct observation events.

### Local duplicate lookup baseline

`SQLiteLocalStore.find_observation_ids_by_content` currently performs a deterministic scan of persisted observation records and validates record/payload integrity while looking for matches. L2.2 intentionally does not migrate the database or introduce a separate content index yet. This is a simple correctness-first local baseline; a future scaling work item may add an index without changing the duplicate semantics or the `DuplicateObservationLookup` interface.

An explicit `asset_uri` or MIME type supplied to local ingestion is preserved verbatim. L2.2 does not infer MIME type from a filename and does not interpret media metadata.

## L2.3 raw EXIF metadata

L2.3 reuses ExifRead `3.5.1` behind a small WRE adapter rather than implementing TIFF/EXIF parsing. The runtime dependency is exactly pinned in `pyproject.toml` and `uv.lock`; the BSD-3-Clause license decision is recorded in `registry/dependencies.yaml`.

`extract_exif_entries` opens the local media file and asks ExifRead for standard metadata with `strict=True`, `details=False` and thumbnail extraction disabled. Each returned standard tag is stored as one immutable `RawMetadataEntry` with:

- namespace `exif`;
- the complete ExifRead key, including its IFD prefix such as `Image`, `EXIF` or `GPS`;
- a deterministic string representation of the parser's raw `values` payload.

Entries are sorted by their complete tag key before they enter `ObservationMetadata`, so persistence equality does not depend on parser dictionary iteration order. ExifRead pseudo-entries for filenames and thumbnails are not metadata records and are excluded.

`details=False` deliberately excludes MakerNote decoding from the L2.3 baseline. MakerNotes are vendor-specific and can require fragile proprietary interpretation; L2.3 needs the stable Image/EXIF/GPS fields that feed later deterministic processing, not camera-vendor heuristics.

### Raw means uninterpreted

L2.3 does not promote metadata text into semantic facts. In particular:

- `EXIF DateTimeOriginal` remains its source string and is not converted into `ImageObservation.captured_at`;
- GPS reference/rational tags remain raw entries and are not converted to decimal latitude/longitude;
- `Image Make` and `Image Model` do not create or resolve a `CameraId`;
- image dimensions are not populated by this EXIF stage;
- an image with no standard EXIF produces an `ObservationMetadata` record with an empty raw-entry tuple.

This matters because raw observations and their persistence identity are immutable. A later interpretation stage must not rewrite an observed record merely because new semantics were derived from its metadata.

## L2.4 deterministic GPS and capture-time interpretation

L2.4 consumes the immutable raw EXIF entries from L2.3 and creates a separate `ObservationMetadataInterpretation` record. It does **not** mutate `ObservationMetadata` or populate `ImageObservation.captured_at` retroactively.

GPS interpretation has four explicit states:

- `absent` — no relevant EXIF GPS evidence exists;
- `incomplete` — some required coordinate evidence exists but the four latitude/longitude DMS + reference tags are not all present;
- `invalid` — relevant evidence exists but is contradictory, duplicated, malformed or out of range;
- `resolved` — complete validated DMS/reference evidence deterministically yields decimal latitude/longitude.

DMS components are parsed with exact rational arithmetic using `Fraction` before the final float conversion. N/S/E/W references control signs explicitly. Incomplete and invalid states never expose partial usable coordinates. `GPSMapDatum`, when present, is retained as evidence; absence of a datum is not silently replaced by an assumed CRS.

Capture-time interpretation likewise has explicit states:

- `absent` — no targeted capture-time evidence exists;
- `local_ambiguous` — `EXIF DateTimeOriginal` is syntactically valid but has no explicit offset, so no global instant is asserted;
- `invalid` — the source date/offset pair is malformed, duplicated or internally inconsistent;
- `resolved` — `DateTimeOriginal` plus a valid `OffsetTimeOriginal` yields a timezone-aware instant normalized to UTC.

A local time without an offset remains local evidence. WRE does not guess a timezone from GPS, machine locale, source website or current timezone during L2.4.

Duplicate targeted GPS/time keys are treated as invalid evidence rather than last-write-wins. Each interpretation records the EXIF keys that support the result and, for incomplete/invalid states, an explicit issue string.

### Interpretation persistence

`MetadataInterpreter` is storage-agnostic and writes through `MetadataInterpretationSink`. `SQLiteLocalStore` stores the derived interpretation as a separate immutable `metadata_interpretation` record in the existing generic table, so no SQLite schema migration is required.

Repeated storage of the same canonical interpretation is idempotent. Reusing the same observation identity for a different interpretation is rejected as a persistence conflict instead of silently rewriting previously derived evidence.

## Identity and retry behavior

The image-ingestion path inherits L1 persistence semantics:

1. a new observation ID and payload are stored atomically;
2. repeating the exact same observation, metadata or interpretation payload is idempotent;
3. reusing a stable record identity with different content raises the persistence conflict instead of silently replacing previously recorded evidence.

Hashing, duplicate reporting, EXIF extraction and semantic interpretation do not weaken or reinterpret those rules.

## Explicitly deferred work

Through L2.4, media ingestion does **not** implement:

- media discovery or directory crawling;
- byte copying into a managed media object store;
- image decoding or image-validity inference;
- MIME-type inference;
- XMP parsing;
- MakerNote interpretation;
- automatic CRS/georeferencing policy beyond deterministic EXIF coordinate interpretation;
- guessing a timezone for ambiguous local timestamps;
- camera identity resolution;
- video ingestion or keyframe extraction;
- feature extraction, matching or reconstruction;
- automatic merge/deletion of observations merely because bytes are identical.

Those behaviors remain in their owning work items. L2.5 next owns video ingestion; deterministic keyframe extraction remains L2.6.
