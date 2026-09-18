# V2L6 — Adaptive frame selection review

**Review status:** PASS  
**Reviewed:** 2026-09-18  
**Lot:** `V2L6 — Adaptive frame selection`

## Review objective

Verify that V2L6.1 through V2L6.6 compose into a deterministic, evidence-driven and reusable media-selection boundary without collapsing quality evidence, frame-selection policy, resource controls, extraction provenance or decoded-image execution into one hidden algorithm.

The lot must establish a solver-independent policy contract, retain the useful interval donor behind an adapter, add explicit quality/diversity selection, bound long-video density after selection, preserve exact selected-frame provenance/timing during materialization and expose canonical reusable decoded-image pyramids. It must stop before pair retrieval, scene identity, synchronization, routing or geometry.

## V2L6 implementation sequence

V2L6 was intentionally split into six bounded work items:

1. PR #71 — `V2L6.1 Frame selection policy contract`: immutable policy identity/revision plus canonical required/optional `MetricName` declarations only.
2. PR #72 — `V2L6.2 Existing interval keyframe donor adapter`: thin V2 wrapper over the retained deterministic interval selector with no fake metric dependency.
3. PR #73 — `V2L6.3 Quality diversity aware frame selection baseline`: deterministic conjunctive quality thresholds over explicit V2L5 metric evidence plus adjacent source-candidate diversity evidence.
4. PR #74 — `V2L6.4 Long video frame density resource bounds`: pure post-selection minimum-spacing and/or hard-count reduction without rereading quality or inferring hardware budgets.
5. PR #75 — `V2L6.5 Frame selection provenance timing regression`: explicit already-selected frame materialization preserving parent-video provenance, frame index/time, deterministic frame identity and idempotent persistence without re-probing or re-selecting.
6. PR #76 — `V2L6.6 Canonical reusable decoded image pyramid artifacts`: vendor-neutral canonical RGB8/source-pixel pyramids, content/config/producer artifact identity and one retained exact FFmpeg software reference materializer.

No V2L6 item may establish scene identity, retrieval truth, route choice, geometry truth or a universal accelerated media backend.

## Composition findings

**PASS.**

- `FrameSelectionPolicy` remains declarative. It contains typed policy identity/revision and canonical required/optional metric names, not frame candidates, scalar scores, thresholds, paths, adapters or execution hooks.
- The interval adapter retains the exact deterministic minimum-spacing donor semantics and requires an empty metric declaration instead of inventing fake quality dependencies.
- Quality/diversity selection consumes only explicit precomputed `MetricVector` evidence and explicitly supplied adjacent visual-diversity evidence. Sharpness/exposure thresholds are conjunctive; there is no global weighted score, hidden ranking or selected-set optimization.
- Adjacent diversity is defined against the immediately preceding **source candidate**, not the preceding selected frame. This keeps evidence semantics deterministic and prevents order-dependent hidden global optimization.
- Long-video density bounds run only after a selection already exists. Minimum spacing precedes hard-count thinning; count thinning uses deterministic integer floor quantiles; returned values retain the exact original `ProbedVideoFrame` objects and timestamps.
- Resource bounds are representation limits only. V2L6.4 does not infer RAM/VRAM, inspect hardware or become the later scheduler/resource estimator.
- Explicit selected-frame extraction accepts a non-empty immutable tuple of already selected `ProbedVideoFrame` values, verifies the source-video bytes first and materializes exactly those frame indices. It performs no `ffprobe` frame discovery and re-enters no interval/quality/diversity/density selection.
- Extracted `VideoFrameObservation` values preserve exact parent video asset, source, received time, frame index, microsecond timing, `captured_at=None`, deterministic observation identity and exact PNG hash/length. Repeating the same request is idempotent through the retained persistence boundary.
- Existing policy-based `LocalKeyframeExtractor.extract` remains available and retains its donor behavior; the new explicit selected-frame path factors materialization without changing the donor selection semantics.
- `DecodedImagePyramidSpec`, level descriptors and the logical pyramid manifest are vendor-neutral domain contracts. They contain no FFmpeg, GPU or vendor type.
- The first decoded-image artifact version is explicitly packed unsigned RGB8, three bytes per pixel, row-major top-to-bottom and source-pixel orientation. It deliberately makes no sRGB/ICC/white-balance/exposure/HDR/photometric-normalization truth claim.
- Pyramid level zero uses native decoded dimensions; later levels floor-halve each positive dimension until the configured maximum-edge stop, never upscale and never duplicate dimensions.
- The FFmpeg reference materializer verifies source bytes before tool execution, requires the retained exact FFmpeg toolchain identity, disables WRE autorotation semantics and emits exact raw RGB level bytes. It reuses the mature external decoder/scaler instead of implementing codecs or resampling.
- Decoded-image computation identity reuses existing `ArtifactKeyMaterial`, source-content fingerprints and exact producer/configuration identity. Distinct `ObservationId` values with the same source bytes and observation kind share the same computation key; changing the observation kind, source bytes, spec, producer version or decode/preprocess configuration changes identity.
- Logical provenance and cache bytes remain separate: the manifest may name the source observation while reusable materialized level bytes are content/config/producer-addressed and do not embed an `ObservationId`.
- Exact level hashes/lengths are represented through the existing `ArtifactMaterializationMetadata` contract and verify through the generic materialization verifier rather than a V2L6-specific cache system.
- The adapter registry adds one approved `ffmpeg.decoded_image_pyramid` reference capability using the already reviewed external FFmpeg subprocess dependency. No new runtime dependency or license posture is introduced.
- The official FFmpeg baseline remains 6.1.1-3ubuntu5. FFmpeg 8 compatibility lanes are green, but issue #54 and a future explicit baseline migration decision remain separate.
- No nvImageCodec, nvJPEG, NVDEC, CUDA, ROCm, Metal, DirectML, Pillow, OpenCV, PyAV, libvips, NumPy or Torch runtime dependency is introduced by the decoded-image reference path.
- V2L6 establishes reusable decode work so later retrieval/matching/geometry need not repeat equivalent decode/resize work, while accelerated execution profiles remain later benchmarked alternatives rather than product-contract requirements.
- The lot introduces no `PairCandidate`, retrieval index, scene cluster, route graph, camera/depth/geometry solution, scheduler, residency policy or V2L7 behavior.

## Cross-cutting invariant audit

**PASS.**

- Raw `ImageObservation`, `VideoObservation` and `VideoFrameObservation` values remain immutable source evidence.
- Selection policy, measured quality evidence, selection decisions, post-selection density bounds and extraction execution remain separately owned.
- Missing or unused metric evidence is not fabricated.
- Selected-frame timing is inherited exactly from probed source evidence rather than recomputed from output cadence.
- Source-byte identity is checked before materialization so paths cannot silently substitute different media.
- Reusable decoded-image artifacts compose the existing artifact-key/materialization substrate instead of creating a second cache identity system.
- Vendor/runtime choices remain behind adapters and producer identity; stable domain contracts stay portable.
- No generated/inferred content is promoted to observed truth.
- Pair proposal and scene identity remain later responsibilities. V2L6 does not infer that two observations depict the same scene merely because frames were selected or decoded.
- No work item starts a later lot early.

## Review findings resolved during the lot

**PASS after correction.**

- PR #71 initially froze an obsolete historical `VideoObservation` test field set; the regression was corrected to the actual retained observation shape without changing product code.
- PR #73 required only mechanical Ruff/YAML lifecycle corrections after implementation; selector behavior remained unchanged.
- PR #74 review confirmed spacing-before-count semantics, endpoint-preserving deterministic floor quantiles and no quality/hardware/routing leakage.
- PR #75 required only Ruff lint/format corrections before the selected-frame extraction regression passed under both the official FFmpeg baseline and the separate FFmpeg 8 compatibility lane.
- PR #76 required Ruff formatting and a Pyright-only fixture typing correction. The final review also added an explicit regression proving content-addressed reuse across different observation IDs while preserving observation-kind significance. No product-contract change was required by those mechanical findings.
- No unresolved PR review thread remains at the implementation review boundary.

## Evidence

Exact V2L6 work-item evidence:

- V2L6.1 / PR #71 / head `aea4b90fbe163ee04a7ee7a676b4c154f396b4fb`: fast-ci #546, native COLMAP #259, CodeQL #484 — **PASS**.
- V2L6.2 / PR #72 / head `71ba4d76e1932f980e669ade9027de6d0aa45850`: fast-ci #551, native COLMAP #262, CodeQL #489 — **PASS**.
- V2L6.3 / PR #73 / head `77d42d25a99b079be998e5dfcedad0f9f5bb91e5`: fast-ci #560, native COLMAP #265, CodeQL #498 — **PASS**.
- V2L6.4 / PR #74 / final handoff head `5e8b63ef5282d0b57f9cde05f7acd871c062c3e6`: fast-ci #567, native COLMAP #267, CodeQL #505 — **PASS**.
- V2L6.5 / PR #75 / final handoff head `67264d795a2243e437c97e8baade849f0db17e0f`: fast-ci #572, FFmpeg 8 #8, native COLMAP #269, CodeQL #510 — **PASS**.
- V2L6.6 / PR #76 / reviewed implementation head `9802845f4486a72de6b3d204105d6c5ac77dd840`: fast-ci #579, FFmpeg 8 #14, native COLMAP #276, CodeQL #517 — **PASS**.

The final PR #76 lifecycle-handoff head changes review/state/registry metadata after the already-green reviewed implementation. It must independently pass every triggered exact-head lane before merge. Those final run identifiers belong in PR #76 merge evidence rather than in this self-referential review commit.

## V2L7.1 handoff decision

After this PASS, `V2L7.1 — Canonical PairCandidate contract` may become the sole `ready` item.

The first V2L7 responsibility is intentionally representation-only. It defines canonical unordered observation-pair identity, explicit candidate-source/evidence references and deterministic deduplication/union semantics. A candidate means only **“worth considering together”**.

V2L7.1 must not encode sequential distance, GPS distance, visual-retrieval score, geolocation, capture time, geometric verification, scene identity or route choice as generic pair truth. Source-specific measurements stay in exact referenced evidence artifacts. The retained sequential and GPS donors are wrapped only in V2L7.2 and V2L7.3.

## Decision

**V2L6 lot review: PASS.**

Adaptive frame selection now composes policy declaration, deterministic donors, explicit evidence-driven selection, bounded representation size, provenance-preserving selected-frame materialization and reusable decoded-image artifacts without coupling those responsibilities to retrieval, geometry or one acceleration stack. V2L7 may begin with the canonical PairCandidate/candidate-source contract only after PR #76 final exact-head gates pass.
