# V2L13 — Feed-forward geometry review

**Review status:** PASS  
**Reviewed:** 2026-09-23  
**Lot:** `V2L13 — Feed-forward geometry`  
**Milestone:** `V2M2 — Reliable static physical reconstruction` remains in progress

## Review objective

Verify that V2L13.1 through V2L13.6 compose one solver-independent feed-forward geometry capability behind the frozen V2 camera/depth/geometry contracts, with one exact experimental DA3-BASE preview candidate, exact runtime/checkpoint/hardware reproducibility, frame/scale-invariant camera-quality evidence, a controlled comparison against the current classical precision baseline, and retained execution-profile evidence.

Passing this lot means WRE can execute one bounded learned preview geometry route and compare its camera evidence under canonical contracts without leaking model-private state into core semantics or promoting one solver/backend as universal truth.

Passing V2L13 does **not** mean DA3-BASE is the permanent preview default, that it is globally better or worse than COLMAP, that its relative depth is metric, or that its CPU reference execution is a production performance target. V2L14 owns generic competing-`GeometrySolution` comparison/refinement/consensus work.

## V2L13 implementation sequence

V2L13 was split into six bounded work items:

1. PR #112 — `V2L13.1 Feed forward geometry adapter interface`: pure solver-independent `geometry.feed_forward` capability and `FeedForwardGeometryResult` referential-integrity contract.
2. PR #113 — `V2L13.2 First approved feed forward preview candidate`: pure DA3-BASE candidate-native prediction semantics and direct normalization into canonical V2 cameras/depth/geometry.
3. PR #114 — `V2L13.3 Feed forward checkpoint hardware license reproducibility`: exact DA3 source/checkpoint/runtime identity, verified decoded-image execution boundary and real CPU reference lane.
4. PR #115 — `V2L13.4 Feed forward geometry camera quality metrics`: solver-independent frame/scale-invariant camera-quality evaluator.
5. PR #116 — `V2L13.5 Preview versus classical baseline comparison`: controlled exact DA3-BASE versus current incremental PyCOLMAP 4.2.0 camera-pose comparison without winner/default promotion.
6. PR #117 — `V2L13.6 Learned execution profile and stage profiling benchmark`: retained cold/steady/stage/resource benchmark evidence, explicit unavailable accelerator evidence and lot closure.

No V2L13 item introduces generic multi-solution refinement, hybrid/global geometry consensus, dense MVS, physical surface, appearance, dynamic reconstruction, chronology or runtime-scene responsibility.

## Generic feed-forward contract findings

**PASS.**

V2L13.1 defines one stable solver-independent boundary:

- input capability kind: `media.decoded_image_pyramid`;
- canonical outputs: `CameraSolution`, `DepthField`, optional `PointMap`, and one `GeometrySolution`;
- one explicit local-frame domain per aggregate;
- exact child reference integrity;
- canonical observation ordering and uniqueness;
- no hidden scale inference, coordinate conversion, alignment, routing or solver execution.

`FeedForwardGeometryResult` is deliberately not a DA3 type. Later learned candidates can normalize through the same contract if they satisfy the same canonical semantics.

## DA3-BASE candidate normalization findings

**PASS.**

V2L13.2 selects DA3-BASE only as the first bounded PREVIEW integration candidate after a current candidate/license refresh.

The pure normalization boundary:

- accepts immutable `Da3BaseObservationPrediction` values;
- preserves reviewed OpenCV world-to-camera extrinsics directly as WRE camera-from-local:
  `x_camera = R * x_local + t`;
- preserves reviewed pinhole `fx, fy, cx, cy` exactly;
- retains DA3 depth as `da3.relative-depth`;
- keeps local scale `UNRESOLVED`;
- creates one camera and one depth field per observation in one explicit local frame;
- fabricates no `PointMap` because no separate reviewed unprojection convention was admitted;
- derives stable role-separated canonical IDs only from an explicit normalization identity.

The candidate-private DA3 “world” is only a local reconstruction frame. It is not project world, Earth, ECEF, ENU or geographic truth.

## Exact runtime and reproducibility findings

**PASS.**

V2L13.3 establishes one conservative exact reference execution profile:

- DA3 source revision `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`;
- reviewed package-tree identity `be87985cb286f0747d4fd9bfaec47f1b765457ea`;
- checkpoint repository identity `depth-anything/DA3-BASE`;
- `model.safetensors` SHA-256 `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`;
- checkpoint byte length `541518028`;
- Python `3.12.12`;
- Torch `2.4.1+cpu`;
- torchvision `0.19.1+cpu`;
- CPU float32 execution;
- exact optional external package closure;
- explicit caller-supplied `HardwareRuntimeIdentity`.

Execution is fail-closed:

- source must be an exact clean local Git checkout;
- package tree must match the reviewed tree;
- unexpected executable/bytecode/native shadows and source symlinks are rejected;
- checkpoint must be an exact local regular non-symlink file with exact bytes;
- canonical decoded-image materialization must verify before pixel use;
- no runtime Git/Hugging Face/network fallback exists;
- learned packages are imported lazily and ordinary WRE core import remains Torch-free.

The reviewed official checkpoint follows upstream `strict=False` loading behavior with exactly six expected missing auxiliary-head keys. WRE accepts exactly that reviewed allowlist and rejects any additional missing or unexpected state key.

DA3 native confidence uses `exp(x)+1`, not a calibrated probability. WRE therefore emits canonical `confidence=None` instead of inventing a sigmoid, threshold or rescale.

The adapter remains `experimental`; direct source/checkpoint terms are recorded as Apache-2.0 while transitive redistribution review remains pending.

## Camera-quality findings

**PASS.**

V2L13.4 adds pure camera-quality measurements against explicit trusted reference cameras.

The retained metric vocabulary is:

- `geometry.camera.observation_coverage_ratio`;
- `geometry.camera.relative_rotation_error_deg_median`;
- `geometry.camera.translation_pair_coverage_ratio`;
- `geometry.camera.relative_translation_direction_error_deg_median`;
- `geometry.camera.focal_relative_error_median`;
- `geometry.camera.principal_point_error_normalized_median`.

Relative-pose metrics are intentionally invariant to a common proper rigid transform and positive local scale:

- relative rotation uses pairwise camera-from-local rotations;
- camera centers use `C = -R^T t`;
- relative translation compares directed baseline directions in the first camera frame.

The evaluator never compares absolute candidate/reference translations across unrelated local frames and never estimates a Sim(3) alignment.

Pinhole-only intrinsic metrics fail closed for unsupported projection families rather than coercing distortion models into pinhole.

The evaluator returns `MetricVector` evidence only. It defines no threshold, `QualityDecision`, winner, retry, fallback or route change.

## Preview versus classical comparison findings

**PASS.**

V2L13.5 compares the exact DA3-BASE preview route with `colmap.incremental_precision_geometry` on one shared controlled 12-view synthetic fixture using the same rendered source observations and trusted reference cameras.

Both routes matched all 12 reference observations.

Retained pose/coverage evidence from the accepted fixture:

| Metric | DA3-BASE | COLMAP incremental |
| --- | ---: | ---: |
| observation coverage | 1.0 | 1.0 |
| relative rotation median error | 1.4394171023° | 0.4603724911° |
| translation-pair coverage | 1.0 | 1.0 |
| relative translation-direction median error | 12.2050836330° | 1.8912699527° |

On this one deterministic synthetic fixture, the angular errors are numerically lower for the classical route. This is descriptive evidence only.

It does **not** establish:

- an overall solver score;
- a winner/rank;
- a default route;
- universal superiority;
- a natural-image quality conclusion;
- cross-frame absolute translation accuracy.

The controlled COLMAP route uses `SIMPLE_RADIAL` while DA3 emits `pinhole`. V2L13.5 therefore does not manufacture symmetric intrinsic evidence by discarding distortion or aliasing projection families.

The full comparison limitations remain recorded in `docs/reviews/V2L13-preview-classical-comparison.md`.

## Execution-profile and profiling findings

**PASS with no accelerated profile validated on the available benchmark hardware.**

V2L13.6 adds a pure execution-profile benchmark layer using canonical `BenchmarkRecord`, `MetricVector`, `BenchmarkFixtureIdentity`, `HardwareRuntimeIdentity` and performance-evidence artifact references.

The accepted reference profile is:

`da3.cpu.float32.eager`

The benchmark records:

- explicit cold-start sample;
- explicit steady-state sample policy;
- source verification;
- checkpoint verification;
- decoded-input verification;
- environment verification;
- learned-runtime import;
- model construction;
- checkpoint load;
- preprocessing;
- model execution;
- postprocess/candidate construction;
- canonical normalization;
- end-to-end wall time;
- peak process RSS;
- conditional GPU-memory evidence only when a GPU profile actually exists.

The latest retained exact-head profiling artifact for PR #117 run `da3-execution-profile #7` has:

- artifact name: `v2l13-6-da3-execution-profile`;
- artifact id: `10737667873`;
- artifact digest: `sha256:51dda7b2181084feaacdbe85cc420252226cab0512a9a620705fe34d0279cc7d`;
- fixture: `fixture.da3-execution-504.v1`;
- fixture SHA-256: `d6c502e0f565b349166107bf41e2dabf8ed4948b5c729173506cdcb78a9136a7`;
- material hardware identity: `e2d62fedf656cdc056e19e1d934e8dae5cd5f77ee2d4ba65a619f25bb51c89d6`;
- retained benchmark record:
  `benchmark:da3:4e4ace40c9361b57cdf2b670de64b13474f344a62b3ce3ac`.

Measured reference-profile values on that exact runner:

| Metric | Value |
| --- | ---: |
| end-to-end cold | 8.163169823 s |
| end-to-end steady median | 4.473515958 s |
| model execution cold | 3.369469243 s |
| model execution steady median | 3.271559976 s |
| runtime import cold | 3.231493471 s |
| runtime import steady median | 0.006843673 s |
| model construction steady median | 0.583651224 s |
| checkpoint verification steady median | 0.377826333 s |
| checkpoint load steady median | 0.137496158 s |
| canonical normalization steady median | 0.023709847 s |
| peak process RSS | 1,880,207,360 bytes (~1.75 GiB) |

The recorded material stages sum to approximately 4.462 s for the 4.474 s steady end-to-end sample, so the retained stage evidence accounts for essentially all measured wall time on this run. Model execution is approximately 73% of the steady end-to-end time on this hardware.

Cold start is approximately 3.69 s slower than steady on this run, with learned-runtime import accounting for most of that observed cold-only delta.

These are **machine-specific benchmark observations**, not portable latency guarantees.

The same benchmark retains camera-quality evidence for the one-view profiling fixture:

- observation coverage: `1.0`;
- focal relative error median: `0.6994413757324218`;
- normalized principal-point error median: `0.0`.

Because the profiling fixture contains one camera, pairwise relative-rotation and translation-direction metrics are unavailable rather than fabricated.

### Accelerator candidate evidence

The bounded candidate profile `da3.cuda.float32.eager` was explicitly probed in the exact benchmark environment.

Result:

- status: `unavailable`;
- reason: `cuda_not_available_in_exact_cpu_reference_environment`;
- measured candidate record: none.

The exact runtime uses `torch==2.4.1+cpu`, and the available GitHub runner exposes no CUDA execution for this profile.

Therefore this lot makes **no** claim of:

- validated CUDA execution;
- accelerated-profile speedup;
- compiled-profile speedup;
- mixed-precision equivalence;
- TensorRT/Torch-TensorRT support;
- GPU-memory performance;
- preferred execution backend.

The benchmark result explicitly retains:

- `accelerated_profile_validated = false`;
- `speedup_claimed = false`;
- `winner = null`;
- `default_profile = null`;
- `shipping_promotion = false`.

This is the required fail-closed outcome when no genuinely executable accelerator candidate exists on the declared comparison hardware.

## Cross-cutting invariant audit

**PASS.**

- Feed-forward model-private state does not become a core WRE type.
- Candidate “world” remains a local reconstruction frame.
- Local scale remains unresolved for DA3-BASE.
- Relative depth is not relabeled metric.
- No `PointMap` is fabricated.
- Confidence is not invented from an uncalibrated native signal.
- Exact source/checkpoint/runtime/hardware identity participates in reproducibility.
- Ordinary WRE core import remains independent of Torch/DA3.
- Camera quality is measured independently from appearance.
- Cross-local-frame comparison uses invariant relative-pose evidence rather than raw absolute translations.
- Classical and learned solutions remain distinct alternatives with retained provenance.
- Performance evidence never changes geometry truth semantics.
- Unsupported execution profiles remain explicit unavailable/failed evidence.
- No speedup is computed across different material hardware identities.
- No benchmark record carries winner/rank/default/threshold/route-decision semantics.
- No V2L13 item mutates COLMAP configuration or shipping status.
- No route fallback/default promotion is introduced.
- Losing or weaker evidence is not deleted.

## Explicit V2L13 non-capabilities

After V2L13, WRE still does **not** yet provide:

- a generic contract for multiple competing `GeometrySolution` alternatives;
- generic geometry alignment or Sim(3) registration between local frames;
- refinement of a feed-forward initialization by classical/global optimization;
- hybrid learned + geometric global reconstruction;
- consensus/disagreement metrics across arbitrary geometry solutions;
- a policy selecting a geometry winner;
- metric-scale solving or Earth/ECEF/ENU/GPS/GCP anchoring;
- dense MVS/fusion;
- explicit physical `SurfaceModel`;
- collision/navigation/measurement guarantees;
- appearance/material/environment reconstruction;
- dynamic 4D reconstruction;
- historical chronology;
- `MasterScene` or `RuntimeScene`;
- validated GPU/compiled/mixed-precision DA3 execution on representative accelerator hardware;
- a permanent DA3 or COLMAP default.

Those remain V2L14 and later work.

## Findings resolved during V2L13

**PASS after development corrections.**

- V2L13.1 established the generic aggregate without learned dependencies.
- V2L13.2 locked the camera-direction, relative-depth, unresolved-scale and no-PointMap semantics before runtime execution.
- V2L13.3 real integration exposed two upstream details that were corrected without weakening WRE semantics:
  - the official checkpoint intentionally omits six auxiliary-head state keys accepted by upstream `strict=False`; WRE now permits exactly that reviewed set and rejects any other mismatch;
  - `InputProcessor` returns an `N×3×H×W` image stack and the official API adds the batch dimension; WRE mirrors that reviewed behavior.
- V2L13.3 also moved metric-marker validation to the raw output mapping because upstream `addict.Dict` attribute access can materialize an empty value for a missing key.
- V2L13.4 required no geometry-semantic correction after the invariant camera-quality fixtures.
- V2L13.5 retained the projection-family boundary rather than coercing COLMAP `SIMPLE_RADIAL` into DA3 pinhole semantics.
- V2L13.6 initially failed fast type checking because the optional integration test imported Torch directly. The test was corrected to use dynamic import only inside the explicitly provisioned optional path, preserving the core no-Torch dependency boundary.
- The exact profiling values vary between GitHub runners/hardware identities. The review therefore records hardware identity and does not compare those separate runs as a controlled speedup.
- No unresolved semantic defect or known regression remains at the V2L13 review boundary.

## Evidence

Accepted V2L13 work-item evidence:

- V2L13.1 / PR #112 / final head `5d2f340d429b10b9f857e2ab436f402dc498edd9`: fast-ci #837, CodeQL #776, PyCOLMAP #475 — **PASS**.
- V2L13.2 / PR #113 / final head `4e692fb3e92edf15af2ed781beb2fd6e21b47603`: fast-ci #856, CodeQL #795, PyCOLMAP #494 — **PASS**.
- V2L13.3 / PR #114 / final head `3cdc7014f4907abd4b436af861ed7ae618ca813c`: fast-ci #889, CodeQL #828, DA3 reference #14, PyCOLMAP #527, FFmpeg8 compatibility #102, dependency-review #172 — **PASS**.
- V2L13.4 / PR #115 / final head `5e7ac36c6241cb4ddfba8784d7f2e1595633b03f`: fast-ci #899, CodeQL #838, DA3 reference #23, PyCOLMAP #537 — **PASS**.
- V2L13.5 / PR #116 / final head `13465ee317e07aa838be7156d0e1834f53a9f72c`: fast-ci #917, CodeQL #856, DA3 reference #40, PyCOLMAP #555, feed-forward/classical comparison #15, dependency-review #187 — **PASS**.
- V2L13.6 / PR #117 / implementation head `11fa5210aa2ba10b44fe4eead56e8ef31753556b`: fast-ci #925, CodeQL #864, DA3 reference #47, DA3 execution profile #7, PyCOLMAP #563, feed-forward/classical comparison #22, dependency-review #194 — **PASS**.

The final PR #117 lifecycle/review head changes lot-review/state/work-item metadata after this green implementation evidence and must independently pass every triggered exact-head lane before merge.

## V2L14.1 handoff decision

After this PASS, `V2L14.1 — Competing GeometrySolution comparison contract` may become the sole `ready` item.

V2L14.1 must define a solver-independent immutable way to retain multiple canonical geometry hypotheses as alternatives without choosing, aligning, refining or fusing them.

It must preserve:

- each source `GeometrySolution` and its exact local-frame identity;
- each candidate's camera/depth/point child references;
- scale status per candidate;
- producer/provenance identity;
- source evidence;
- explicit alternative identity and canonical ordering.

It must not:

- compare raw coordinates across unrelated local frames;
- infer an alignment/Sim(3) transform;
- rescale or mutate a candidate;
- choose a winner;
- apply quality thresholds;
- delete losing hypotheses;
- run bundle adjustment/global refinement;
- introduce the first hybrid/global candidate;
- compute consensus/disagreement metrics owned by V2L14.5.

## Decision

**V2L13 lot review: PASS.**

WRE now has a solver-independent feed-forward geometry boundary, one exact experimental DA3-BASE preview candidate, exact runtime/reproducibility evidence, invariant camera-quality measurement, a controlled comparison against the current classical precision baseline, and retained execution-profile/stage/resource evidence.

DA3-BASE remains experimental and non-default. No accelerator profile was validated on the available benchmark hardware. V2M2 remains in progress and may continue with V2L14.1 only after PR #117 final exact-head gates pass.
