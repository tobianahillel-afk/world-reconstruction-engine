# V2L16 — Physical surface and V2M2 milestone review

**Review status:** PASS  
**Reviewed:** 2026-09-30  
**Lot:** `V2L16 — Physical surface`  
**Milestone:** `V2M2 — Reliable static physical reconstruction`

## Review objective

Verify that V2L16.1 through V2L16.6 form one coherent physical-surface layer over the accepted V2M2 geometry and dense-depth contracts, while preserving immutable provenance, exact local-frame and scale semantics, explicit unsupported/hole evidence, descriptive quality observations and a mandatory CPU-capable product path.

The milestone review also verifies that accepted V2L11 through V2L16 evidence composes into a reliable static physical-reconstruction substrate without collapsing solver outputs into unquestioned truth or silently claiming appearance, runtime or safety capabilities that belong to later milestones.

Passing V2L16 means WRE now has an explicit canonical physical `SurfaceModel`, one bounded CPU surface-generation baseline, descriptive topology evidence, an optional hybrid candidate boundary, explicit negative support semantics and one retained controlled CPU benchmark.

Passing V2M2 means WRE can carry a static reconstruction from canonical camera/depth/point/geometry evidence through competing/refined geometry and dense-depth evidence into an explicit physical surface while retaining uncertainty, holes, provenance and unresolved scale. It does **not** mean the surface is photorealistic, complete, watertight, safe for collision/navigation, accurate for measurement or selected as a universal best backend.

## V2L16 implementation sequence

1. PR #173 — `V2L16.1 SurfaceModel contract and intended use metadata`: immutable physical-surface identity/provenance envelope with explicit representation and intended uses, while preserving exact source geometry, local frame and scale status.
2. PR #175 — `V2L16.2 Open3D depth-fusion surface baseline`: exact Open3D 0.20.0 CPU depth-only TSDF baseline over canonical `DenseDepthArtifact` evidence with one geometry-only `surface.ply`, exact materialization metadata and no inferred metric scale.
3. PR #176 — `V2L16.3 Descriptive surface suitability metrics`: pure topology observations over caller-supplied `SurfaceMeshInspection`; no mesh parsing, repair, threshold or safety verdict.
4. PR #178 — `V2L16.4 Hybrid surface candidate boundary`: pure optional candidate request/result boundary with no concrete learned/hybrid surface solver selected.
5. PR #179 — `V2L16.5 Unsupported region and hole semantics`: immutable negative support annotations distinguishing `unsupported` from `hole`; absence remains unknown.
6. PR #180 — `V2L16.6 Controlled physical surface benchmark and V2M2 review`: deterministic CPU-only benchmark that composes V2L16.3 topology metrics with optional V2L16.5 negative support evidence without selection, safety or completeness semantics.

No V2L16 item implements appearance, materials, environment, MasterScene, RuntimeScene, dynamic 4D, chronology or generated completion.

## V2L16.1 SurfaceModel findings

**PASS.**

`SurfaceModel` is an immutable metadata/provenance envelope for one explicit physical-surface artifact. It retains:

- one exact `ArtifactRef` of kind `geometry.surface`;
- one complete source `GeometrySolutionCandidate`;
- an open representation token;
- canonical intended-use metadata for collision, measurement and/or navigation;
- the exact source `LocalFrameId`;
- the exact source `GeometryScaleStatus`;
- exact producer/configuration identity;
- canonical source-artifact ancestry.

The contract deliberately does not inline solver-private mesh/TSDF/SDF payloads and does not turn intended use into suitability proof. A surface marked as intended for measurement is not therefore measurement-accurate; a surface intended for collision is not therefore collision-safe.

Unresolved source scale remains unresolved.

## V2L16.2 CPU Open3D baseline findings

**PASS as one bounded baseline, not as a universal default.**

The accepted concrete baseline uses Open3D 0.20.0 in an optional exact external environment and performs depth-only CPU tensor TSDF integration.

The route:

- consumes one canonical `DenseDepthArtifact`;
- accepts the reviewed pinhole + camera-z path only;
- forwards canonical camera-from-local extrinsics with the audited convention;
- keeps invalid depth as exact zero;
- does not use confidence as a hidden support threshold;
- uses geometry-only TSDF state;
- rejects empty, non-finite or triangle-free results;
- publishes one `surface.ply` plus exact `ArtifactMaterializationMetadata`;
- preserves source geometry, frame, scale and complete ancestry.

Real retained integration evidence exists from V2L16.2, including a non-empty triangle mesh generated on CPU. That evidence proves the bounded adapter can execute; it does not prove that Open3D is globally best, that all scenes reconstruct well, that unresolved scale becomes metric, or that the result is collision/measurement/navigation ready.

The optional Open3D dependency remains isolated from WRE's mandatory environment.

## V2L16.3 descriptive topology findings

**PASS.**

`SurfaceMeshInspection` is explicit caller-supplied canonical evidence. The evaluator computes only descriptive topology observations:

- finite-vertex ratio;
- degenerate-triangle ratio;
- boundary-edge ratio when a unique-edge denominator exists;
- non-manifold-edge ratio when a unique-edge denominator exists;
- largest-component triangle ratio;
- connected-component count.

The evaluator:

- does not open surface payload bytes;
- does not use Open3D, NumPy, Trimesh or PyMeshLab;
- does not repair, fill, remesh or orient geometry;
- omits edge-ratio observations when the denominator is unavailable rather than fabricating a value;
- preserves exact evaluator and source-artifact provenance;
- does not infer watertightness, safety, measurement accuracy or a pass/fail decision.

Metric scale changes none of those semantics: the observations remain ratios/counts and are not converted into length, area, volume or tolerance claims.

## V2L16.4 optional hybrid candidate findings

**PASS as a boundary only.**

The hybrid-surface candidate contract can express one optional alternate `SurfaceModel` produced from exact canonical source geometry and explicit support evidence while preserving:

- source geometry;
- local frame;
- scale state;
- representation;
- intended uses;
- exact canonical ancestry;
- output materialization identity.

No concrete 2DGS, OMeGa, SurfaceSplat, MeshSplatting or other learned/hybrid surface technology is selected or integrated by this work item.

The boundary therefore increases replaceability without manufacturing a second production solver.

## V2L16.5 negative support findings

**PASS.**

`SurfaceSupportMap` attaches caller-supplied selector-artifact identities to one exact `SurfaceModel` and distinguishes only:

- `unsupported` — the selected region must not be treated as supported physical truth even if geometry exists there;
- `hole` — no retained physical-surface coverage is claimed for the selected region.

The contract preserves exact surface/frame identity and canonical source ancestry.

Critically:

- selector payloads are identities only and are not interpreted by the contract;
- no fill, repair, interpolation, generated geometry or completion is requested;
- absence of a support map or region entry remains **unknown**;
- absence never becomes positive completeness/support/watertightness/safety evidence.

## V2L16.6 controlled physical-surface benchmark

**PASS.**

The controlled benchmark is a pure standard-library/WRE-contract composition layer. It does not parse mesh or selector payloads and does not invoke Open3D or another solver.

The retained fixture contains three canonically ordered physical-surface candidates:

| Candidate | Scale state | Negative support evidence | Topology character |
| --- | --- | --- | --- |
| `surface:candidate-hole` | metric | one explicit `hole` region | fully finite vertices, one component, no degenerate/non-manifold triangles |
| `surface:candidate-unknown` | unresolved | annotation absent → explicit `unknown` | lower finite ratio, two components, degenerate and non-manifold evidence |
| `surface:candidate-unsupported` | metric | two explicit `unsupported` regions | distinct boundary/degenerate/component observations |

For every candidate the benchmark:

- calls `evaluate_surface_suitability` unchanged;
- retains the complete V2L16.3 `MetricVector` and evaluator provenance;
- keeps support annotations separate from topology metrics;
- retains exact optional support-map artifact identity and ordered selector/status entries;
- emits one canonical `BenchmarkRecord` using the shared fixture, shared hardware identity, exact `SurfaceModel` producer and `QualityMode.QUALITY`;
- does not let support status change the topology metric vector, producer provenance or benchmark identity.

The absence case is represented explicitly as:

- support state `unknown`;
- no support-map artifact identity;
- no region tuple.

It is not represented as zero unsupported regions or complete support.

The retained JSON contains no winner, rank, score, threshold, preferred/default route, retry/fallback, `QualityDecision`, supported-area/fraction, completeness, watertightness, safety, collision-ready, navigation-ready, measurement-ready, generated completion or shipping-promotion field.

### Retained implementation evidence

Implementation head `868806bbe92e1c33a1e1ce385025d48c688d1b3e` passed:

- fast-ci #1291 — repository metadata, Ruff lint/format, type check, unit tests and actionlint;
- CodeQL #1119;
- dependency-review #353;
- physical-surface-controlled-benchmark #5;
- dense-depth-controlled-benchmark #71;
- COLMAP integration #858;
- COLMAP MVS integration #132;
- DA3 reference #273;
- DA3 execution profile #232;
- feed-forward/classical comparison #253.

Retained physical-surface benchmark artifact:

- artifact id `11093533691`;
- artifact name `v2l16-6-controlled-physical-surface-benchmark`;
- artifact digest `sha256:95dd77bd624b8e91d732397f675bbb9d61103e6fe24c4bd96bd40733176ef0e6`;
- size 2,418 bytes;
- exact head `868806bbe92e1c33a1e1ce385025d48c688d1b3e`.

### Unrelated Open3D environment-lock drift

`open3d-surface-integration #62` fails before executing the V2L16.6 benchmark because a pre-existing PyPI lock no longer regenerates byte-identically:

- `charset-normalizer 3.5.1 -> 3.5.2`;
- `platformdirs 4.12.1 -> 4.12.2`.

The same external resolver drift was already documented before V2L16.6. This PR does not modify the Open3D resolver, dependency lock or V2L16.2 adapter. The V2L16.6 controlled benchmark deliberately requires no Open3D import or network package resolution and passes independently.

Updating that external lock is a separate maintenance action, not evidence that the physical-surface benchmark or V2M2 semantics regressed.

## V2M2 composition audit — V2L11 through V2L16

**PASS.**

V2M2 is reviewed as a composed capability, not as a count of finished work items.

### V2L11 — canonical geometry contracts

V2L11 established the canonical camera, depth, point-map and `GeometrySolution` vocabulary with explicit local-frame and scale semantics. This gives later solvers a stable representation boundary and prevents solver-private structures from becoming product truth.

### V2L12 — classical precision geometry

V2L12 established the audited classical COLMAP precision path behind canonical contracts. Native solver state is retained as explicit evidence and normalized into WRE types rather than replacing the domain model.

### V2L13 — feed-forward geometry

V2L13 added an approved feed-forward preview candidate under exact source/checkpoint/runtime evidence. The learned route remains a candidate behind canonical geometry semantics; model output does not bypass provenance, scale or quality evidence.

### V2L14 — competing geometry and refinement

V2L14 established:

- complete competing `GeometrySolutionCandidate` values;
- a pure refinement boundary;
- audited CPU COLMAP bundle-adjustment refinement;
- gauge-invariant descriptive consensus/disagreement evidence;
- one controlled benchmark over multiple geometry hypotheses.

Its hybrid/global accelerator gate validly concluded deferred optional rather than blocking the mandatory CPU product path.

No universal geometry winner was promoted.

### V2L15 — dense depth / MVS / fusion

V2L15 established:

- a canonical dense-depth artifact over one exact source geometry hypothesis;
- an experimental optional COLMAP PatchMatch donor boundary;
- pure multi-artifact fusion and learned-prior boundaries;
- explicit coverage, holes and confidence evidence;
- a controlled trusted-reference dense-depth benchmark.

Real CUDA PatchMatch remains explicitly deferred/unverified. That deferred optional evidence is not treated as passed.

### V2L16 — explicit physical surface

V2L16 makes physical surface a first-class artifact separate from sparse/dense geometry and separate from appearance.

The milestone therefore reaches an explicit physical-surface boundary while preserving:

- exact source geometry ancestry;
- immutable evidence;
- local-frame identity;
- unresolved or metric scale as explicit state;
- exact dense-depth lineage where used;
- explicit materialization identity;
- descriptive topology observations;
- explicit negative `unsupported`/`hole` evidence;
- unknown state when negative annotations are absent;
- no hidden alignment, scale conversion or completion.

## V2M2 invariant audit

**PASS.**

Across V2L11 through V2L16:

- raw observations and retained solver evidence remain immutable;
- derived artifacts never destructively overwrite their evidence;
- solver-private state stays behind adapters/evidence boundaries;
- camera/depth/point/surface responsibilities remain explicit;
- multiple geometry hypotheses may coexist without an implicit universal winner;
- refinement retains initialization lineage;
- local frames are not silently promoted to world/Earth/geographic coordinates;
- unresolved scale remains unresolved unless separate evidence resolves it;
- holes and unsupported regions remain representable instead of being silently filled;
- missing evidence remains unknown rather than becoming zero/false/complete;
- quality observations remain descriptive until an explicit later policy owns a decision;
- the mandatory path retains CPU-capable evidence and does not require an unavailable accelerator;
- physical geometry remains distinct from photorealistic appearance;
- generated completion remains absent from reconstructed physical truth.

## Explicit V2M2 non-capabilities

After this milestone WRE still does **not** provide:

- a canonical source-color/exposure interpretation layer;
- decode/working/output color-space normalization;
- exposure or white-balance normalization artifacts;
- a canonical `AppearanceModel`;
- Gaussian/radiance appearance reconstruction;
- material decomposition, PBR or relighting;
- an `EnvironmentModel` for sky/horizon/distant illumination;
- `MasterScene` assembly/versioning;
- `RuntimeScene` compilation, compression, chunking, LOD or streaming;
- a reference interactive viewer;
- dynamic 4D reconstruction;
- persistent dynamic entities or multi-video event reconstruction;
- historical chronology or cross-epoch change semantics;
- generated completion of missing physical regions;
- a universal geometry, depth or surface solver winner/default;
- guaranteed watertight surface reconstruction;
- certified collision safety;
- certified navigation readiness;
- certified measurement accuracy;
- world/Earth/geographic anchoring merely from a local reconstruction;
- proof from deferred/unavailable accelerator lanes such as real CUDA PatchMatch.

These are later V2M3+ responsibilities or explicit future policies.

## V2L17.1 handoff decision

After this PASS, only `V2L17.1 — Source color exposure metadata contract` may become ready.

V2L17.1 must be a pure solver-independent source-photometry metadata contract over exact observation identity and retained raw metadata evidence. It may describe source color declarations and exposure/white-balance capture hints with explicit absent/unknown/invalid states, but it must not decode pixels, transform color, normalize exposure/white balance or create an `AppearanceModel`.

In particular V2L17.1 must preserve the distinction between:

- raw metadata values;
- interpreted color/exposure evidence;
- decoded pixel convention;
- later working/output color-space conversion;
- later photometric normalization.

No missing EXIF/video metadata may be guessed into sRGB, linear light, white balance, ISO, shutter speed, aperture or exposure compensation merely because such defaults are common.

## Decision

**V2L16 lot review: PASS.**

**V2M2 milestone review: PASS.**

WRE now has a reliable static physical-reconstruction substrate through explicit physical-surface evidence. The milestone remains descriptive and provenance-first: it does not claim photorealism, completion, runtime packaging or certified safety.

The final PR #180 lifecycle/review head must independently pass repository validation, the dedicated physical-surface controlled benchmark and all applicable exact-head checks before merge. No V2L17 implementation is included in this review commit.
