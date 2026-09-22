# V2L12 — Classical precision geometry baseline review

**Review status:** PASS  
**Reviewed:** 2026-09-22  
**Lot:** `V2L12 — Classical precision geometry baseline`  
**Milestone:** `V2M2 — Reliable static physical reconstruction` remains in progress

## Review objective

Verify that V2L12.1 through V2L12.5 compose the current pinned COLMAP/PyCOLMAP 4.2.0 classical precision baseline behind V2 artifact and geometry contracts without restoring legacy sparse estimated geometry as the production target, hiding mapper selection, inventing geometry for failed/disconnected outcomes, or crossing into feed-forward geometry, hybrid refinement, dense reconstruction, physical surface, appearance, dynamic, chronology or runtime responsibilities.

Passing this lot means WRE has a validated classical precision geometry baseline with explicit V2 evidence identities, direct native-to-canonical normalization, distinct incremental/global executable mapper identities and truthful zero/single/disconnected model outcomes.

Passing V2L12 does **not** mean COLMAP is the permanent or globally preferred reconstruction algorithm. It is the current classical precision baseline against which later feed-forward, hybrid/global and specialist routes can be compared under the same product contracts.

## V2L12 implementation sequence

V2L12 was split into five bounded work items:

1. PR #107 — `V2L12.1 Current COLMAP adapter descriptor and V2 artifact normalization`: pure solver-independent `colmap.precision_geometry` capability over canonical V2 output kinds while retaining `colmap.sparse_sfm` as historical donor evidence.
2. PR #108 — `V2L12.2 Current COLMAP evidence stages use V2 artifact identities`: explicit V2 planning/publication contracts for local features, raw pair matching and geometric verification using canonical artifact keys, producer/configuration/hardware identity and immutable donor evidence.
3. PR #109 — `V2L12.3 COLMAP incremental and global mappers emit canonical geometry`: direct audited native-model normalization to canonical `CameraSolution`, `PointMap` and `GeometrySolution` plus distinct pinned incremental/global executable routes.
4. PR #110 — `V2L12.4 Disconnected and zero model explicit outcomes`: pure positive-domain `ColmapGeometryOutcome` contract representing no model, one model or multiple disconnected independent models without ranking, merge or repair.
5. PR #111 — `V2L12.5 COLMAP current route regression comparison and lot review`: shared-source image-based native regression across retained incremental donor evidence and the current incremental/global V2 paths, conservative shipping-status decision and lot closure.

No V2L12 item introduces feed-forward geometry, learned geometry, cross-model alignment, scale solving, world placement, dense depth, physical surface, appearance, dynamic reconstruction or route-selection heuristics.

## V2 artifact/evidence findings

**PASS.**

V2L12.1 and V2L12.2 establish the production-facing classical evidence boundary before mapper execution.

- `colmap.precision_geometry` declares canonical V2 geometry output kinds rather than `geometry.sparse_reconstruction_estimate`.
- The retained `colmap.sparse_sfm` entry remains distinct and addressable as historical donor evidence.
- Local feature, pair-match and geometric-verification stages have distinct adapter identities and solver-independent artifact kinds.
- Artifact identities use the existing canonical key material: ordered input fingerprints, exact producer/configuration identity and explicit hardware-runtime identity.
- Filenames, timestamps, database row IDs, paths and mutable object identity do not become semantic cache identity.
- Publication preserves exact already-computed database SHA-256 and byte length.
- Evidence publication does not promote local features, matches or two-view geometry into accepted camera/geometry truth.
- Existing deterministic PyCOLMAP donor execution semantics remain unchanged.

The V2 wrappers therefore make classical evidence auditable without duplicating COLMAP internals or replacing the stable WRE artifact contracts.

## Direct canonical geometry findings

**PASS.**

V2L12.3 reads each native sparse model only after exact audited file membership, SHA-256 and byte-length checks and normalizes it directly into V2 geometry.

For each audited native model:

- one deterministic source-model content identity is derived from the exact model manifest;
- one explicit `LocalFrameId` is created for that independent native model;
- every registered COLMAP image maps to exactly one known WRE `ObservationId`;
- camera extrinsics preserve `image.cam_from_world()` exactly as camera-from-local-frame:
  `x_camera = R * x_local + t`;
- camera projection token normalization is lowercase-only and invalid canonical syntax fails closed;
- point coordinates are copied directly in deterministic native point-ID order;
- one canonical `PointMap` retains exact registered observation support;
- one `GeometrySolution` references the exact emitted cameras and point map;
- local scale remains `GeometryScaleStatus.UNRESOLVED`;
- no depth field, physical surface, fragment/world placement, metric upgrade or geographic claim is invented.

The production normalization path does not construct `SparseReconstructionEstimate`, `CameraCalibrationEstimate`, `CameraPoseEstimate` or `Point3DEstimate`. The retained importer and V2L11 one-way bridge are regression evidence only.

## Incremental and global mapper findings

**PASS.**

The current executable mapper routes remain intentionally distinct:

- `colmap.incremental_precision_geometry` uses the retained deterministic PyCOLMAP 4.2.0 incremental mapper behavior and direct V2 canonicalization.
- `colmap.global_precision_geometry` uses pinned PyCOLMAP 4.2.0 `global_mapping` plus explicit global pipeline options.

Both consume image observations plus verified geometric evidence and emit the same canonical V2 camera/point/geometry responsibilities.

The global route:

- verifies the caller geometric-verification database before use;
- copies it into a fresh private workspace before view-graph calibration or mapping;
- never mutates the caller database;
- fixes the exposed random seed and mapper thread count;
- disables exposed global-positioning GPU execution;
- selects Ceres bundle adjustment explicitly and disables its exposed GPU path;
- records the PyCOLMAP 4.2.0 limitation that view-graph calibration exposes a random seed but not a separate internal Ceres thread setting.

Accelerated/Caspar behavior is not silently selected under the reference adapter identities.

## Explicit mapper-outcome findings

**PASS.**

V2L12.4 introduces one immutable cardinality-derived `ColmapGeometryOutcome` with exactly:

- `NO_MODEL`;
- `SINGLE_MODEL`;
- `DISCONNECTED_MODELS`.

The state cannot be supplied independently by a caller; it is derived after validating exact native/canonical model cardinality, index order and source-model content identity.

Consequences:

- zero native models contain no placeholder `CameraSolution`, `PointMap` or `GeometrySolution`;
- exactly one native model preserves exactly one independent canonical local-frame solution;
- multiple native models remain an ordered tuple of disconnected independent local-frame solutions;
- mutable collections, duplicate/non-canonical model indices, missing/extra/reordered canonical models and foreign content identities fail closed;
- no ranking, selection, merge, alignment, filtering, rescaling, retry, fallback or cross-frame compatibility claim occurs in the outcome layer.

This is outcome representation, not route policy.

## V2L12.5 shared-source native regression findings

**PASS.**

The final native fixture reuses the existing versioned L3.7 image-based synthetic scene rather than starting from a preconstructed sparse model.

The exact PyCOLMAP 4.2.0 chain is:

```text
versioned synthetic PGM scene
  -> SIFT feature extraction
  -> exhaustive raw pair matching
  -> geometric verification
  -> one shared verified source database
       -> retained/current incremental execution
            -> direct native -> canonical V2
            -> retained legacy import -> V2L11 one-way bridge
       -> current global execution
            -> direct native -> canonical V2
```

The fixture proves:

- both mapper executions preserve the caller verification database byte-for-byte;
- every emitted native model exactly matches its audited file manifest;
- canonical source-model identity matches the audited native manifest identity;
- direct incremental and retained-import-plus-bridge paths agree on registered observation membership, camera rotations, translations, projection models, image dimensions, intrinsic parameters, local 3D point coordinates and unresolved-scale semantics;
- the legacy donor and direct V2 path may use different opaque target IDs and different point sequence canonicalization rules, so equivalence compares the point-coordinate multiset with multiplicity rather than incorrectly requiring identical tuple order;
- incremental/global canonical results are represented through `ColmapGeometryOutcome`;
- every current canonical model remains an independent unresolved local frame;
- comparison facts are descriptive model/registered-observation counts only.

No overall winner score, preferred/default mapper, cross-local-frame error, hidden alignment or automatic route-selection heuristic is created.

The fixture begins from image-derived verified evidence and exercises extraction/matching/verification, but it remains a deterministic synthetic integration regression. It is **not** a natural-image photogrammetric quality benchmark and must not be interpreted as evidence that one mapper is universally superior.

## Shipping-status decision

**PASS — retain all V2L12 adapters as `experimental`.**

The following V2L12-owned registry entries remain explicitly `experimental`:

- `colmap.precision_geometry`;
- `colmap.local_features`;
- `colmap.pair_matching`;
- `colmap.geometric_verification`;
- `colmap.incremental_precision_geometry`;
- `colmap.global_precision_geometry`.

This is an evidence-based non-promotion, not missing review.

Reasons:

- the full V2L12 composition is validated on deterministic synthetic evidence, not yet on the representative natural-image benchmark matrix required to judge geometry quality;
- V2L12 does not own production route selection or a default incremental/global choice;
- comparable geometry/camera metrics versus feed-forward and later hybrid/global candidates are future V2L13/V2L14 responsibilities;
- disconnected components remain intentionally independent rather than aligned/ranked;
- failure taxonomy, quality-policy decisions, retry/fallback and production scheduling are separate later responsibilities;
- the generic `colmap.precision_geometry` entry is a capability descriptor rather than an executable default selector;
- retaining `experimental` prevents successful integration evidence from being misread as a permanent “best algorithm” decision.

The direct COLMAP dependency/license evidence remains approved and unchanged. No shipping decision changes capability, producer, dependency, model/checkpoint, hardware-key policy, geometry semantics or route preference.

The retained historical `colmap.sparse_sfm` donor remains `approved` under its existing legacy baseline evidence; V2L12 does not reinterpret that status as V2 production preference.

## Cross-cutting invariant audit

**PASS.**

- Original observations and parent evidence remain immutable.
- V2 evidence identity remains distinct from solver-private database/layout details.
- Legacy estimated geometry remains donor/regression evidence, not the V2 production target.
- Camera-from-local-frame convention remains exactly the V2L11 convention.
- Local frame identity remains distinct from world/Earth/geographic identity.
- Local scale remains unresolved unless independently proved.
- Independent native models remain independent local frames.
- Zero-model output remains truthful absence of geometry.
- Global/incremental mapper identity remains explicit rather than hidden behind one mutable implementation.
- GPU/accelerated BA is not silently enabled under the reference adapters.
- No route winner/default is inferred from the final fixture.
- No feature/match/verification evidence is promoted directly to physical scene truth.
- No `SurfaceModel`, mesh, collision/measurement suitability or appearance representation is created.
- No feed-forward/hybrid geometry, dynamic 4D, historical chronology, master scene or runtime work enters V2L12.

## Explicit V2L12 non-capabilities

After V2L12, WRE still does **not** yet provide:

- feed-forward camera/depth geometry adapters or preview candidates;
- a benchmark comparing learned preview geometry to the classical baseline;
- a generic competing-`GeometrySolution` comparison contract;
- hybrid/global refinement or consensus/disagreement policy;
- cross-local-frame alignment or scale reconciliation;
- metric-scale inference or world/Earth/ECEF/ENU/GPS/GCP anchoring;
- dense depth/MVS/fusion;
- explicit physical `SurfaceModel`;
- collision/navigation/measurement guarantees;
- photometric calibration or photorealistic appearance;
- materials/environment reconstruction;
- dynamic 4D reconstruction;
- historical chronology;
- `MasterScene` assembly or runtime compilation;
- a production router decision that chooses incremental or global mapping as a universal default.

Those remain V2L13 and later lots.

## Review findings resolved during V2L12

**PASS after development corrections.**

- V2L12.1 and V2L12.2 closed without remaining semantic defects.
- V2L12.3 development required mechanical import ordering/Ruff formatting and one Pyright test-annotation correction before the full native/global lane became green. The native fixture also corrected its workflow inclusion so the required real global mapper test actually executed. No camera/point/scale semantics were weakened.
- V2L12.4 required one Ruff formatting correction only; no outcome semantics changed.
- The first V2L12.5 native comparison incorrectly required the direct V2 and retained legacy bridge `PointMap.positions_xyz` tuples to have identical order. Real PyCOLMAP evidence showed the direct route preserves numeric native `point3D_id` order while the historical legacy estimate reorders its opaque string point IDs lexically. The comparison was corrected to require the same coordinate multiset with multiplicity, preserving each path's accepted internal ordering contract.
- The first comparison also implicitly required opaque local-frame identity equality between the two independently-derived regression paths. That requirement was removed: equivalence is on pose/intrinsics/points/membership/unresolved scale, while the donor and V2 production path intentionally derive different opaque target IDs.
- These V2L12.5 corrections changed only regression comparison assumptions; no production reconstruction code was changed.
- No unresolved review thread or known semantic regression remains at the V2L12 review boundary.
- Automated code-review availability is not treated as semantic evidence; exact-head CI plus deny-by-default diff review govern closure.

## Evidence

Accepted V2L12 work-item evidence:

- V2L12.1 / PR #107 / final head `47ce86df60b8ec640f47f83fad3b99b4d01fbf65`: fast-ci #780, CodeQL #719, native PyCOLMAP #418, FFmpeg-compat #41 — **PASS**.
- V2L12.2 / PR #108 / final head `f8ccfae0bccadba619109f451e3ae4c0a433f06f`: fast-ci #794, CodeQL #733, native PyCOLMAP #432, FFmpeg-compat #54 — **PASS**.
- V2L12.3 / PR #109 / final head `b0c326c447dc8d60952b50c69cd1b19bac0f4d5e`: fast-ci #811, CodeQL #750, native PyCOLMAP #449, dependency-review #149, FFmpeg-compat #70 — **PASS**.
- V2L12.4 / PR #110 / final head `790e544994329363c6c16139d285d22b45280e21`: fast-ci #817, CodeQL #756, native PyCOLMAP #455 — **PASS**.
- V2L12.5 / PR #111 / reviewed native-comparison head `43a7fd52ff0123235ee4cab6c31a048c89e1862b`: fast-ci #820, CodeQL #759, native PyCOLMAP #458, dependency-review #151 — **PASS**.
- V2L12.5 / PR #111 / explicit shipping-decision head `e50840ead932438bf466edafde481c0fd70b75ac`: fast-ci #821, CodeQL #760, native PyCOLMAP #459, dependency-review #152 — **PASS**.

The final PR #111 lifecycle/review head changes review/state/work-item metadata after this already-green implementation/review evidence. It must independently pass every triggered exact-head lane before merge.

## V2L13.1 handoff decision

After this PASS, `V2L13.1 — Feed forward geometry adapter interface` may become the sole `ready` item.

V2L13.1 is contract/interface work only. It must define the solver-independent boundary needed for later feed-forward camera/depth candidates while preserving the V2L11/V2L12 canonical geometry semantics.

It must not:

- integrate or select a learned checkpoint yet;
- promote a current research candidate before V2L13.2 freshness/license/checkpoint review;
- change `CameraSolution`, `DepthField`, `PointMap` or `GeometrySolution` semantics;
- reinterpret predicted geometry as metric/world/geographic truth without explicit evidence;
- rank a feed-forward route against COLMAP;
- add hybrid refinement, dense MVS, physical surface or appearance responsibility;
- make the classical V2L12 routes a hidden fallback/default.

## Decision

**V2L12 lot review: PASS.**

WRE now has an audited classical precision geometry baseline using exact PyCOLMAP/COLMAP 4.2.0, V2 artifact identities, direct canonical camera/point/geometry normalization, distinct incremental/global executable routes and explicit no-model/single/disconnected outcomes.

The V2L12 adapters intentionally remain experimental. V2M2 remains in progress and may continue with V2L13.1 only after PR #111 final exact-head gates pass.
