# V2L5 — Media profiling review

**Review status:** PASS  
**Reviewed:** 2026-09-18  
**Lot:** `V2L5 — Media profiling`

## Review objective

Verify that V2L5.1 through V2L5.6 compose into a useful, solver-independent media-profiling evidence layer without collapsing measurements, candidates or missing evidence into hidden scene truth.

The lot must establish a minimal `MediaProfile` evidence-link boundary, deterministic image-quality measurements, threshold-free pairwise visual-similarity measurements, evidence-preserving input-class signals and threshold-free collection/sequence summaries. It must stop before adaptive frame selection, retrieval, clustering, temporal synchronization, routing, geometry or dynamic reconstruction.

## V2L5 implementation sequence

V2L5 was intentionally split into six bounded work items:

1. PR #65 — `V2L5.1 MediaProfile contract`: immutable `ObservationId` plus canonical exact `ArtifactRef` evidence links only.
2. PR #66 — `V2L5.2 Blur sharpness exposure clipping metrics`: explicit decoded-luma input and deterministic typed image-quality metrics.
3. PR #67 — `V2L5.3 Near duplicate and visual diversity signals`: threshold-free average-hash Hamming fraction and fixed-grid luma MAE while exact-content duplicate handling remains SHA-256 plus byte length.
4. PR #68 — `V2L5.4 Camera and specialist input class signals`: observed still/video evidence separated from candidate 360, UAV, fisheye and rolling-shutter hints.
5. PR #69 — `V2L5.5 Dynamic long sequence sparse coverage signals`: informational collection/source counts plus optional duration, temporal visual-change and coverage-diversity summaries without dynamic/long/sparse labels.
6. PR #70 — `V2L5.6 Media profile fixture matrix`: cross-contract composition fixtures and this lot review.

No V2L5 item is allowed to select frames, choose a route, establish scene identity, infer geometric overlap or execute a solver.

## Composition findings

**PASS.**

- `MediaProfile` remains exactly one `ObservationId` plus immutable canonical `ArtifactRef` evidence links. It contains no metric bag, classifier result, route, mutable metadata or catch-all future field.
- `LumaRaster` is an explicit already-decoded 8-bit luma boundary. Image-quality evaluation reads no path or media source and emits only typed `MetricVector` evidence with exact evaluator/input provenance.
- Exposure clipping and mean luma are explicit measurements; sharpness is the deterministic population variance of the normalized four-neighbour Laplacian. The V2L5.2 review finding that the first implementation allocated all Laplacian samples was fixed with an O(1) streaming Welford accumulator and regression coverage.
- Pairwise visual similarity remains threshold-free. Average-hash Hamming and grid-luma MAE deliberately carry complementary evidence: uniform black versus uniform white has Hamming fraction zero but luma MAE one.
- Exact-content duplicates remain owned by the retained SHA-256 plus byte-length ingestion donor. V2L5 does not reinterpret luma similarity as exact duplicate truth and never deletes or collapses source observations.
- Input classes preserve evidence strength. Still/video is observed from `ObservationKind`; equirectangular-360, drone/aerial, fisheye and rolling-shutter are candidates only.
- The V2L5.4 review finding that manufacturer-only `DJI` metadata could falsely imply aerial capture was fixed. The final baseline requires recognizable UAV model-family evidence with lexical boundaries and retains Osmo, Pocket, Ronin and accidental-substring cases as unresolved/non-candidates.
- Missing specialist evidence creates no negative class claim. Candidate classes may coexist because 360, UAV, fisheye and rolling-shutter properties are not mutually exclusive.
- Profile-summary metrics remain INFORMATIONAL. Observation/source count, optional duration, temporal grid-luma change and coverage grid-luma diversity are evidence for later policy; they do not label a sequence dynamic, long or sparsely covered and do not assert geometric overlap.
- The V2L5.6 sequence fixture now feeds the actual `media.visual.grid_luma_mae` emitted by V2L5.3 into the V2L5.5 summary input. This directly verifies the intended cross-contract composition rather than merely testing the two evaluators independently.
- Missing duration/change/diversity evidence is omitted, not fabricated as zero, false, rejected or unsupported.
- All composed metric outputs preserve exact caller-supplied `MetricProvenance`; immutable raster, metadata and summary inputs remain unchanged.
- The lot introduces no total quality score, confidence score, rank, threshold, `QualityDecision`, route graph, frame-selection decision, camera solve, motion classification, retrieval model, persistence mutation or new runtime dependency.

## Fixture-matrix findings

**PASS.** The V2L5.6 matrix exercises representative compositions without introducing a new production orchestrator:

- ordinary still — MediaProfile links, image-quality metrics, exact observed STILL evidence and collection counts coexist with no fabricated specialist claim;
- specialist capture — exact 2:1 dimensions plus explicit fisheye, rolling-shutter and recognized UAV-family evidence coexist as candidates while STILL remains separately observed;
- video sequence — VIDEO evidence composes with actual V2L5.3 visual-distance output, duration and profile-summary metrics while all summary metrics remain informational;
- missing evidence — optional metadata, duration and visual aggregates remain absent rather than becoming zero/false semantic claims;
- visual similarity — identical rasters produce zero distances while black-versus-white demonstrates complementary Hamming/MAE behavior without a duplicate decision;
- boundary/non-mutation — MediaProfile field shape remains minimal and composed evaluators preserve typed inputs and provenance.

## Cross-cutting invariant audit

**PASS.**

- Raw observations remain immutable source truth and are not rewritten by profiling.
- Derived evidence remains distinct from observed truth and keeps exact provenance.
- Unknown or absent evidence stays absent/unresolved rather than being guessed.
- Visual similarity does not define scene identity; later retrieval and geometric verification retain that responsibility.
- Candidate camera/specialist evidence does not become calibration, route or reconstruction truth.
- Dynamic-likelihood proxies do not contaminate static/dynamic reconstruction semantics.
- No algorithm or vendor-specific implementation is embedded in the stable MediaProfile boundary.
- V2L6 adaptive frame selection remains a later responsibility and has not been implemented in this lot.

## Evidence

Exact merged V2L5 work-item heads:

- V2L5.1 / PR #65 / `7a0426ecf6f824a4f0585bc3220ef17d613517cc`: fast-ci #506, native COLMAP #245, CodeQL #444 — **PASS**.
- V2L5.2 / PR #66 / `d25993e653291d6c23af6779b56afb22674ac7fd`: fast-ci #513, native COLMAP #249, CodeQL #451 — **PASS**.
- V2L5.3 / PR #67 / `8c666f53b7912144ec7853da121f7d9aa0d848c0`: fast-ci #518, native COLMAP #251, CodeQL #456 — **PASS**.
- V2L5.4 / PR #68 / `5667b3348ee577bd1c7a640c00c24a8680e8dbbe`: fast-ci #527, native COLMAP #253, CodeQL #465 — **PASS**.
- V2L5.5 / PR #69 / `966fe3c3e3e02c870e3e180b3224e945ff916ddc`: fast-ci #532, native COLMAP #255, CodeQL #470 — **PASS**.
- V2L5.6 fixture-matrix implementation / PR #70 / `7256a3aa8cd5dc39d3a38cef9ab32062396b94cb`: fast-ci #538 and CodeQL #476 — **PASS**.

PR #70 review identified one P2 composition gap: the first sequence fixture hard-coded visual-distance scalars rather than consuming V2L5.3 output. The fixture was corrected so actual `media.visual.grid_luma_mae` values feed V2L5.5 summary evidence before this review was recorded.

The final PR #70 handoff head changes review/state/registry metadata after the already-green matrix implementation. It must pass fast CI, the triggered native COLMAP integration lane and CodeQL on that exact final head before merge. Those final run identifiers belong in PR #70 merge evidence rather than in this self-referential review commit.

## V2L6.1 handoff decision

After this PASS, `V2L6.1 — Frame selection policy contract` may become the sole `ready` item.

The first V2L6 responsibility is deliberately declarative. It defines immutable policy identity/revision plus canonical required/optional `MetricName` evidence declarations. An empty metric declaration remains valid so the retained deterministic interval donor can be wrapped in V2L6.2 without pretending it consumes quality evidence.

V2L6.1 does **not** select frames, expose FFmpeg or source paths, encode interval spacing, thresholds, quality weights, diversity thresholds, frame-density/resource budgets, decode images, persist selected frames or execute routing. Those responsibilities remain split across V2L6.2 through V2L6.6.

## Decision

**V2L5 lot review: PASS.**

The media-profiling layer now provides explicit, deterministic and composable evidence for later selection/routing work while preserving uncertainty and ownership boundaries. V2L6 may begin with the policy contract only after PR #70 final exact-head gates pass and all review findings are resolved.
