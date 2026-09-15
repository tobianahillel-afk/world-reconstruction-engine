# L2 media-ingestion lot review

Date: 2026-09-15
Scope: lot `L2` — Media ingestion
Implementation head reviewed: `d6308faaf0135f6484152a03579d4b284077b636`
Verdict: **PASS**

## Review objective

Verify that L2 turns supplied image/video media into deterministic, provenance-preserving raw observations and explicit metadata interpretations without silently inventing time, identity or geometry, and without pulling L3 reconstruction work forward.

## L2.1 — Image ingestion

**PASS.** PR #12 adds the storage-agnostic `ImageIngestor` for already content-addressed media. Inputs are explicit, missing capture time stays missing, and the stage does not calculate hashes, parse metadata or invoke reconstruction.

## L2.2 — Hashing and duplicate detection

**PASS.** PR #13 adds streaming SHA-256 plus exact byte count and deterministic duplicate lookup. Exact-content equality is evidence only: distinct observations and their provenance remain persistable, while same-observation retries stay idempotent.

## L2.3 — Raw EXIF metadata

**PASS.** PR #14 reuses pinned ExifRead 3.5.1 rather than implementing TIFF/EXIF parsing. Standard EXIF is preserved as canonical raw entries. GPS rationals, local date strings and camera descriptors are not promoted into semantic truth.

## L2.4 — GPS and capture-time interpretation

**PASS.** PR #15 creates a separate immutable interpretation record. GPS and capture time expose explicit absent/incomplete/invalid/resolved or local-ambiguous states. Partial coordinates and timezone guesses are not emitted; raw observation/metadata records are never rewritten.

## L2.5 — Source-video ingestion

**PASS.** PR #17 adds `VideoObservation`, content-addressed local source ingestion and the same exact-content duplicate semantics as images. Source bytes remain opaque at this boundary; no decoder, frame selection, MIME inference or motion/reconstruction logic is pulled forward.

## L2.6 — Deterministic frame/keyframe extraction

**PASS.** PR #18 reuses external FFmpeg/ffprobe through an explicitly pinned Ubuntu 24.04 system package. Before tool invocation, local source bytes must match the persisted parent video digest and byte length. `ffprobe` supplies observed first-video-stream frame timestamps; missing, non-finite or non-monotone timestamps fail instead of producing a synthetic timeline.

Frame selection is deterministic: retain the first representable frame and subsequent frames at the configured positive minimum integer-microsecond spacing. FFmpeg extracts those exact source frame indices to PNG; each PNG is content-addressed and persisted as a `VideoFrameObservation` retaining the parent video asset, source frame index and relative integer-microsecond coordinate.

The review caught and removed an unsafe temporal inference: parent `VideoObservation.captured_at` is **not** assumed to be the start instant of the decoded stream. Extracted frames therefore retain `frame_time_us` but keep `captured_at=None` until a later evidence-backed timeline-to-absolute-time contract exists.

## FFmpeg reuse and supply-chain review

The selected CI/reference environment is Ubuntu 24.04 with package `ffmpeg=7:6.1.1-3ubuntu5`; both `ffmpeg` and `ffprobe` must report `6.1.1-3ubuntu5`. The package is recorded as GPL-2.0-or-later for this distribution build. WRE uses subprocess invocation only and neither links nor bundles FFmpeg binaries. A different build/package, binary redistribution or library linkage requires a fresh version/license review.

## Real integration evidence

Implementation head `d6308faaf0135f6484152a03579d4b284077b636`:

- fast-ci run `34912604396`: **PASS**;
- exact FFmpeg apt install/version assertions: **PASS**;
- repository validator: **PASS**;
- Ruff lint/format: **PASS**;
- Pyright: **PASS**;
- pytest including real FFmpeg synthetic-video extraction: **PASS**;
- actionlint: **PASS**;
- CodeQL run `34912604416`: **PASS**;
- dependency-review run `34912604439`: **PASS** under the repository's existing dependency-graph capability policy.

The FFmpeg integration fixture synthesizes eight PPM frames, encodes deterministic FFV1 Matroska input, extracts the selected frames twice and checks source indices, integer-microsecond timestamps, stable observation identity, PNG content hashes, parent-video linkage, persistence and idempotence.

## Epistemic and scope audit

**PASS.** L2 preserves the required separation:

- media bytes and observations remain raw/source-facing state;
- raw EXIF remains separate from semantic GPS/time interpretation;
- duplicate content does not merge provenance;
- video-relative timestamps do not become unsupported absolute capture instants;
- no features, correspondences, poses, fragments, tracking, visual odometry, matching or reconstruction are implemented by L2.

The lot therefore hands L3 deterministic media observations and extracted video frames without claiming any geometric truth.

## Residual limitations

These are explicit follow-up scope, not blockers for L2:

- video extraction currently targets only the first video stream;
- audio and secondary video streams are not ingested;
- keyframe selection is a deterministic temporal-spacing baseline, not scene-change or motion-adaptive selection;
- no managed immutable media-object store exists yet;
- the reference FFmpeg system-package baseline is Ubuntu 24.04 specific;
- richer video tracking, rolling-shutter/rig/IMU and trajectory behavior remains owned by L11.

## Decision

**L2 lot review: PASS.** L2.1 through L2.6 form a coherent, tested, deterministic media-ingestion pipeline with explicit provenance and uncertainty boundaries. The next permitted implementation item after the state handoff is `L3.1` — COLMAP environment adapter. Milestone M1 remains pending because the entire L3 COLMAP baseline and its review are still outstanding.
