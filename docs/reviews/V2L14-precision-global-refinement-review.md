# V2L14 — Precision / global refinement review

**Review status:** PASS  
**Reviewed:** 2026-09-24  
**Lot:** `V2L14 — Precision / global refinement and competing geometry`  
**Milestone:** `V2M2 — Reliable static physical reconstruction` remains in progress

## Review objective

Verify that V2L14.1 through V2L14.6 compose a solver-independent competing-geometry and refinement layer over the canonical V2 camera/depth/point/geometry contracts, while preserving immutable evidence, unresolved local-frame scale, explicit provenance and a mandatory CPU-capable product path.

Passing this lot means WRE can now retain multiple complete geometry hypotheses, refine an audited classical hypothesis, compare hypotheses through gauge-invariant camera disagreement, and run one controlled CPU benchmark against shared trusted reference cameras without inventing a universal winner, default backend or cross-frame alignment.

Passing V2L14 does **not** mean that one DA3/COLMAP route is globally best, that the deferred GLUEMAP/VGGT accelerator specialist is production-ready, or that dense depth, fusion or a physical surface exists. Those remain later responsibilities.

## V2L14 implementation sequence

V2L14 was split into six bounded work items:

1. PR #118 — `V2L14.1 Competing GeometrySolution comparison contract`: immutable complete geometry hypotheses, canonical competing sets and deterministic unordered pair topology without compatibility, alignment or winner semantics.
2. PR #120 — `V2L14.2 Geometry refinement adapter boundary`: pure solver-independent initialization-to-refined-candidate boundary with explicit support evidence and distinct output identity.
3. PR #121 — `V2L14.3 Classical COLMAP bundle-adjustment refinement`: exact PyCOLMAP 4.2.0 one-thread CPU Ceres bundle adjustment over an audited COLMAP native sparse model, with immutable initialization lineage and direct canonical output.
4. V2L14.4 — hybrid/global specialist technology gate: extensive exact-source, dependency, checkpoint and redistribution qualification culminated in PR #146, which records a **NO-GO / deferred optional** outcome for the current accelerator-dependent GLUEMAP/VGGT specialist on the mandatory path and establishes a CPU-portable mandatory product baseline.
5. PR #147 — `V2L14.5 Geometry consensus disagreement metrics`: pure CPU/standard-library pairwise camera disagreement evidence over canonical competing hypotheses, invariant to independent rigid local-frame gauges and containing no rank/score/selection semantics.
6. PR #148 — `V2L14.6 Controlled geometry comparison benchmark`: one shared deterministic CPU fixture exercising DA3 preview, COLMAP incremental, COLMAP BA refinement and the single-model COLMAP global route with trusted-reference camera evidence plus complete V2L14.5 pairwise consensus.

No V2L14 work item implements dense MVS/depth fusion, physical surface, appearance, dynamic reconstruction, chronology, MasterScene/runtime compilation or V2L15 behavior.

## Competing-geometry contract findings

**PASS.**

V2L14.1 establishes one complete `GeometrySolutionCandidate` as a retained immutable hypothesis containing:

- one canonical `GeometrySolution`;
- the exact referenced canonical `CameraSolution`, `DepthField` and `PointMap` values;
- exact producer/configuration identity;
- canonical unique source `ArtifactRef` evidence.

`CompetingGeometrySolutions` requires at least two distinct candidates and preserves deterministic `GeometrySolutionId` order. `GeometrySolutionPair` and pair derivation represent the complete unordered comparison topology.

Critically, grouping hypotheses does **not** assert:

- common `LocalFrameId`;
- equal or resolved scale;
- compatible projection/intrinsic models;
- point/depth coordinate comparability;
- an alignment transform;
- a winner or preferred/default route.

This keeps competing evidence richer than any later policy decision.

## Solver-independent refinement boundary findings

**PASS.**

V2L14.2 defines a minimal refinement boundary:

- one complete immutable initialization candidate;
- an explicit canonical tuple of supporting artifacts;
- one complete refined candidate;
- a required distinct output `GeometrySolutionId`.

The boundary does not expose solver-private state, does not mutate/relabel the initialization and does not require same-frame, same-scale or coordinate compatibility as a generic interface property.

It contains no scoring, quality gate, route selection, retry/fallback or persistence behavior.

## Classical bundle-adjustment refinement findings

**PASS.**

V2L14.3 binds the pure refinement boundary to exact audited COLMAP native state rather than pretending arbitrary canonical geometry can be reconstructed into a valid bundle-adjustment problem.

The accepted classical refinement route:

- uses PyCOLMAP 4.2.0;
- requires verified native COLMAP sparse-model evidence corresponding exactly to the canonical initialization;
- copies native input state into a private output workspace before refinement;
- executes one-thread CPU Ceres bundle adjustment;
- keeps the caller native model byte-identical;
- emits a new canonical candidate with a distinct `GeometrySolutionId`;
- preserves unresolved local scale;
- retains exact initialization/support/producer provenance;
- does not select, align or promote the result.

This is a concrete precision-refinement baseline, not a claim that classical BA is universally optimal.

## Hybrid/global specialist technology-gate findings

**PASS — current GLUEMAP/VGGT candidate is deferred optional, not mandatory.**

V2L14.4 performed substantially deeper qualification than a normal adapter integration before any solver code was accepted. The retained evidence includes exact source/materialization identities, external environment locks, package/license inventories, bounded LightGlue/ALIKED source and local-checkpoint constructor proofs, VGGT source-import compatibility evidence, and explicit redistribution surfaces.

The gate exposed real integration constraints, including accelerator availability, gated/custom-license model assets and transitive redistribution obligations. The final mandatory-path decision in PR #146 is therefore:

- every mandatory WRE capability must retain at least one supported CPU route;
- GPU/CUDA/NPU-only candidates are optional specialists;
- a technology gate may validly conclude NO-GO/deferred optional;
- current GLUEMAP/VGGT evidence remains retained for future specialist work;
- the mandatory V2 roadmap does not block on provisioning accelerator hardware;
- no GLUEMAP/VGGT adapter is claimed to exist.

This NO-GO is a successful technology-gate outcome, not a failed lot.

## Consensus/disagreement findings

**PASS.**

V2L14.5 adds deterministic pairwise disagreement evidence without importing NumPy, Torch, PyCOLMAP, CUDA or another solver.

For every canonical unordered pair, the evaluator may retain:

- shared-observation-over-union coverage;
- median relative-rotation disagreement;
- translation-pair coverage;
- median relative-translation-direction disagreement when the pair geometry is eligible.

The evaluator deliberately does not:

- align local frames;
- solve SE3/Sim3/ICP;
- compare absolute translation across unrelated frames;
- compare raw depth or point coordinates across unrelated frames;
- compare or coerce incompatible intrinsics/projections;
- convert disagreement into a score, threshold, rank or decision.

Low overlap and degenerate-baseline cases remain partial descriptive metric vectors rather than fabricated values.

## V2L14.6 controlled CPU benchmark

**PASS.**

The retained benchmark reuses the existing deterministic `colmap-l3-end-to-end` synthetic source/reference fixture and one shared classical evidence chain.

Execution identity:

- Python 3.12.12;
- DA3-BASE exact reviewed Torch CPU route;
- PyCOLMAP 4.2.0 without GPU support;
- shared feature extraction, exhaustive matching and geometric verification executed once;
- incremental COLMAP baseline;
- V2L14.3 CPU bundle-adjustment refinement over the exact audited incremental native model;
- global COLMAP attempted from the same shared verified evidence;
- no GLUEMAP/VGGT or accelerator requirement.

The retained run produced four complete candidates because the global route returned exactly one model:

| Route | Reference observation coverage | Median relative rotation error | Median relative translation-direction error | Translation-pair coverage |
| --- | ---: | ---: | ---: | ---: |
| COLMAP BA refinement | 1.0 | 0.2921350370° | 1.4986289788° | 1.0 |
| COLMAP incremental | 1.0 | 0.4603724911° | 1.8912699527° | 1.0 |
| COLMAP global | 1.0 | 0.2517321655° | 1.2156099985° | 1.0 |
| DA3-BASE preview | 1.0 | 1.4394171023° | 12.2050836330° | 1.0 |

These values describe **one controlled synthetic fixture only**. They are not a route ranking, production score or default-selection result.

All four candidates use the same 12 trusted reference camera `ObservationId` values.

With four candidates, V2L14.5 correctly retained all six unordered consensus pairs:

| Candidate pair | Shared-observation ratio | Median rotation disagreement | Translation-pair coverage | Median translation-direction disagreement |
| --- | ---: | ---: | ---: | ---: |
| BA refinement ↔ incremental | 1.0 | 0.1688363377° | 1.0 | 1.8615819693° |
| BA refinement ↔ global | 1.0 | 0.0488824379° | 1.0 | 0.4704020944° |
| BA refinement ↔ DA3 | 1.0 | 1.2183501354° | 1.0 | 12.2588359084° |
| incremental ↔ global | 1.0 | 0.1978158934° | 1.0 | 1.9033848718° |
| incremental ↔ DA3 | 1.0 | 1.1444314473° | 1.0 | 12.2683025708° |
| global ↔ DA3 | 1.0 | 1.2623361907° | 1.0 | 12.1623612583° |

Again, these are symmetric disagreement observations, not preference signals.

The machine-readable benchmark explicitly marks these dimensions unavailable rather than fabricating comparability:

- absolute translation across unrelated local frames;
- intrinsic-parameter comparison across incompatible projections;
- raw depth-value comparison across candidates;
- raw point-coordinate comparison across candidates;
- cross-candidate scale comparison.

The retained JSON has no winner, rank, preferred/default route, overall score, threshold, retry/fallback, `QualityDecision` or shipping-promotion field.

## Controlled-benchmark implementation corrections

**PASS after two canonical-order corrections.**

The first benchmark run exposed that DA3 normalization returns canonical geometry references sorted by child IDs while its returned child tuples preserve inference order. Creating `GeometrySolutionCandidate` directly from that tuple correctly failed the candidate contract. The benchmark now creates a canonical comparison view by sorting DA3 camera/depth/point children by their canonical IDs; no scientific value is changed.

The second run progressed through all reconstruction routes and consensus, then correctly exposed that `CameraPoseQualityRequest` has a different input ordering requirement: cameras must be supplied in canonical `ObservationId` order. The benchmark now creates a temporary ObservationId-ordered metric view without mutating the `GeometrySolutionCandidate`, whose own canonical order remains by child identity.

These corrections are benchmark normalization only. No DA3, COLMAP, bundle-adjustment, consensus or production geometry implementation changed.

## Evidence integrity findings

**PASS.**

The controlled benchmark verifies:

- rendered source assets remain byte-identical;
- the DA3 checkpoint remains byte-identical;
- the reviewed DA3 source checkout remains clean;
- the shared COLMAP verification database remains byte-identical after incremental and global use;
- the audited native incremental model remains byte-identical after BA refinement;
- the BA refined candidate has a distinct geometry identity and exact initialization lineage;
- every complete candidate retains producer/configuration/source-artifact evidence;
- every reference metric retains the same trusted-reference artifact provenance;
- every pairwise consensus metric retains the V2L14.5 evaluator and canonical input-artifact provenance.

The global route yielded one model in this retained run and was therefore included. The benchmark contract still requires truthful zero/disconnected reporting with no component selection if a future controlled execution yields another cardinality.

## Cross-cutting invariant audit

**PASS.**

- Raw observations and retained solver evidence remain immutable.
- Derived evidence does not overwrite source evidence.
- Competing hypotheses coexist without deletion or hidden preference.
- Local-frame identity remains distinct from world/Earth/geographic identity.
- Unresolved scale remains a valid explicit state.
- No cross-frame point/depth/absolute-translation comparison is manufactured.
- Routing chooses work, not truth.
- Disagreement/reference metrics remain descriptive and separate.
- No metric vector is collapsed into an overall score.
- Mandatory V2 behavior remains CPU-capable.
- Accelerator specialists remain optional and cannot block mandatory roadmap progress.
- Surface/collision/measurement truth is not inferred from sparse/depth evidence.
- No dense MVS, fusion or physical surface enters V2L14.

## Explicit V2L14 non-capabilities

After V2L14, WRE still does **not** yet provide:

- a dense-depth/MVS artifact composition contract for later dense solvers;
- COLMAP PatchMatch/MVS integration under the V2 dense-depth boundary;
- multi-view depth consistency/fusion;
- learned dense-depth priors as a V2L15 adapter family;
- dense coverage/hole/confidence quality outputs;
- an explicit physical `SurfaceModel`;
- collision/navigation/measurement suitability;
- a production geometry router that selects a universal backend;
- cross-local-frame alignment/scale reconciliation;
- world/Earth/geographic anchoring;
- production-ready GLUEMAP/VGGT execution;
- static appearance/material/environment reconstruction;
- dynamic 4D or chronology;
- MasterScene/runtime compilation.

These remain V2L15 and later lots.

## Evidence

Accepted lot evidence includes:

- V2L14.1 — PR #118, final head `6ba7b1e900a8b56b1551eeba2a0343145d21a45d`.
- V2L14.2 — PR #120, final head `717bf49fc7fcb14aed61e4b5aa44fc55c1e814a4`.
- V2L14.3 — PR #121, final head `9c5d50155a660de4229c49101d07f1fb41e9e4bf`.
- V2L14.4 — retained qualification PR series beginning with PR #122 and final mandatory-path decision PR #146, head `ee5bc601feb00c8619868f09d624795660c19e6b`.
- V2L14.5 — PR #147, final lifecycle head `e0cb95586096893c1a8b8f4b6e64a572d46fc9cf`.
- V2L14.6 — PR #148, green implementation/benchmark head `716bb9bcac4a845c57c24d549040a987c68d0739`:
  - fast-ci #1071 — **PASS**;
  - CodeQL #924 — **PASS**;
  - dependency-review #257 — **PASS**;
  - geometry-controlled-benchmark #5 — **PASS**.
- Retained V2L14.6 Actions artifact `v2l14-6-controlled-geometry-benchmark`, artifact digest `sha256:f5c8d9a8ecc28e6805746ec4fb21a13007044d276c09eaa35c0d9d8347a877e1`.
- Retained benchmark JSON SHA-256 `dd947799161225ba16de557b7dc8fc56f58d218d8fe59b6d517fc7ae2dfa0307`.

The final PR #148 lifecycle/review head changes only review/registry/state metadata after this already-green benchmark evidence. It must independently pass every triggered exact-head lane before merge.

## V2L15.1 handoff decision

After this PASS, `V2L15.1 — Dense depth artifact contract` may become the sole `ready` item.

V2L15.1 is pure solver-independent evidence-contract work. It must reuse the canonical `DepthField`, `CameraSolution`, `GeometrySolutionCandidate`, producer/configuration identity and `ArtifactRef` contracts rather than creating a solver-private dense representation.

Its first contract must represent one coherent non-empty set of dense depth fields derived from one complete canonical source geometry hypothesis while preserving:

- exact source `GeometrySolutionCandidate` identity and cameras;
- canonical unique `DepthField` order;
- exact camera/observation/dimension linkage for every dense field;
- exact per-field depth convention, validity and confidence without conversion;
- producer/configuration/source-artifact provenance;
- partial camera coverage as explicit evidence rather than fabricated depth;
- unresolved source scale/frame semantics.

It must not:

- execute COLMAP MVS/PatchMatch yet;
- fuse multiple depth fields or point maps;
- convert depth convention or infer metric scale;
- create or promote a `SurfaceModel`;
- fill holes;
- add a learned prior;
- compare/rank/select depth routes;
- infer collision/measurement suitability;
- cross into V2L15.2 or later implementation.

## Decision

**V2L14 lot review: PASS.**

WRE now has a canonical competing-geometry model, a pure refinement boundary, an audited classical CPU BA refinement route, gauge-invariant pairwise consensus evidence and one retained controlled CPU benchmark over four complete geometry hypotheses.

The benchmark remains descriptive and synthetic. No route is promoted to universal truth or default. The current accelerator specialist remains deferred optional. V2M2 remains in progress and may continue with V2L15.1 only after PR #148's final lifecycle head passes all exact-head gates.
