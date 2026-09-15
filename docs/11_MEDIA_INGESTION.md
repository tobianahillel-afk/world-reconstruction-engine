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

## L2.5 deterministic source-video ingestion

L2.5 adds whole-video source ingestion without performing frame extraction. A video file is first represented as a raw `VideoObservation`; later L2.6 frame/keyframe extraction creates `VideoFrameObservation` records that retain the source video's `MediaAssetRef`.

`VideoIngestRequest` mirrors the existing image boundary and accepts only explicit inputs:

- `ObservationId`;
- an already-addressed `MediaAssetRef`;
- `SourceRef`;
- timezone-aware `received_at`;
- optional timezone-aware `captured_at`.

`VideoIngestor` constructs and persists the `VideoObservation` through the same structural `ObservationSink` used by image ingestion. `SQLiteLocalStore` therefore needs no new table or schema migration; the observation codec gains only the new stable discriminator `video`.

For local source files, `LocalVideoIngestor` reuses the L2.2 `hash_file_content` primitive and duplicate lookup. It resolves the path, streams the bytes through SHA-256, preserves an explicit asset URI/MIME type when supplied, reports prior exact-content observations and persists the new raw video observation.

The duplicate semantics are intentionally identical to images: exact same bytes do not collapse distinct provenance events. A retry of the same observation ID/content is idempotent and excludes itself from duplicate reporting.

### No decoding or container inference in L2.5

L2.5 deliberately treats source-video bytes as opaque media. It does not:

- invoke FFmpeg or ffprobe;
- validate that the container can be decoded;
- inspect codecs, stream count, duration, dimensions, frame rate or frame timestamps;
- infer MIME type from the filename/container;
- infer capture time from container metadata;
- select or extract frames/keyframes;
- create any `VideoFrameObservation`.

This boundary is intentional. Source-byte identity and provenance can be stored deterministically before frame-selection policy exists. FFmpeg remains the planned reuse choice for video decode/extraction when L2.6 implements deterministic keyframe extraction; that work must review the exact binary/integration/license assumptions it adopts.

## L2.6 deterministic frame/keyframe extraction

L2.6 reuses the external `ffmpeg`/`ffprobe` command-line tools instead of implementing video demuxing or decoding in WRE. The supported baseline is the Ubuntu Noble package `ffmpeg=7:6.1.1-3ubuntu5`, whose CLI reports version `6.1.1-3ubuntu5`. The exact system package and CLI versions are verified in fast CI and the dependency/license decision is recorded in `registry/dependencies.yaml`.

The selected Debian/Ubuntu binary build is GPL-2.0-or-later because that distribution build enables GPL-licensed FFmpeg parts. WRE executes the tools only as external subprocesses: it does not link to FFmpeg libraries and does not bundle the binary. A future change to another package/build, library linkage or binary redistribution requires a fresh version/license review.

`LocalKeyframeExtractor` performs the following deterministic sequence:

1. resolve the local source path and recompute SHA-256 + byte length;
2. reject the operation if those bytes do not exactly match the supplied persisted `VideoObservation.asset`;
3. verify that `ffmpeg` and `ffprobe` report the same required version;
4. ask `ffprobe` for `best_effort_timestamp_time` for every decoded frame of the first video stream;
5. reject missing, non-finite or non-monotone timestamps rather than fabricating a timeline;
6. keep source frame indices and round observed timestamp seconds to integer microseconds using decimal round-half-even semantics;
7. ignore only leading negative-time preroll frames because the current `VideoFrameObservation` contract requires a non-negative offset;
8. select the first representable frame and subsequent frames separated by at least the configured positive `min_interval_us`;
9. ask `ffmpeg` to extract exactly those source frame indices as RGB PNG files;
10. SHA-256 content-address each PNG and persist a `VideoFrameObservation` containing the exact parent video asset, source frame index and integer-microsecond offset.

Frame observation IDs are deterministic hashes of parent observation identity, parent video content digest, source frame index and integer-microsecond offset. Repeating extraction with identical source bytes, policy and supported toolchain therefore yields the same observation IDs and immutable persistence payloads.

### Time semantics remain conservative

`frame_time_us` is a relative coordinate in the decoded source-video timeline. L2.6 does **not** silently treat `VideoObservation.captured_at` as the start of that timeline. Even when the parent video has a caller-supplied `captured_at`, extracted `VideoFrameObservation.captured_at` remains `None` until a later contract establishes an evidence-backed mapping from the video timeline to absolute capture instants.

This avoids turning a convenient parent timestamp into unsupported per-frame temporal truth.

### Explicit L2.6 boundary

L2.6 does not:

- use scene-content scoring, learned models or motion magnitude to choose frames;
- infer frame timestamps from nominal frame rate when timestamps are missing;
- reinterpret duplicate or non-monotone timeline evidence;
- extract audio or secondary video streams;
- infer camera calibration, rolling shutter, IMU or trajectory state;
- track features or motion across frames;
- perform visual odometry, matching, geometric verification or reconstruction.

Those capabilities remain in later lots. L3 begins the COLMAP baseline; the richer video/motion model remains L11.

## Identity and retry behavior

The image/video-ingestion path inherits L1 persistence semantics:

1. a new observation ID and payload are stored atomically;
2. repeating the exact same observation, metadata, interpretation or extracted-frame payload is idempotent;
3. reusing a stable record identity with different content raises the persistence conflict instead of silently replacing previously recorded evidence.

Hashing, duplicate reporting, EXIF extraction, semantic interpretation, source-video ingestion and frame extraction do not weaken or reinterpret those rules.

## Explicitly deferred work

Through L2.6, media ingestion does **not** implement:

- media discovery or directory crawling;
- byte copying into a managed media object store;
- image-validity inference beyond owning parsers' explicit failures;
- MIME-type inference;
- XMP parsing;
- MakerNote interpretation;
- automatic CRS/georeferencing policy beyond deterministic EXIF coordinate interpretation;
- guessing a timezone for ambiguous local timestamps;
- camera identity resolution;
- video scene-change scoring or adaptive keyframe selection;
- audio/secondary-stream ingestion;
- feature extraction, matching, tracking, visual odometry or reconstruction;
- automatic merge/deletion of observations merely because bytes are identical.

Those behaviors remain in their owning work items. L3 next owns the COLMAP reconstruction baseline; later lots own richer video motion, difficult matching, absolute anchoring and temporal world behavior.
