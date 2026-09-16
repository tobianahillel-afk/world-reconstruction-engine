# WRE v2.0 executable roadmap

This roadmap turns the v2 blueprint into implementation-sized work. It is intentionally capability-oriented rather than paper/model-oriented. Technologies are candidates behind contracts; lots own product responsibilities.

## Roadmap rules

- A **milestone** closes only when an end-to-end user capability is demonstrable.
- A **lot** groups one coherent capability and should normally contain 3–6 work items.
- A **work item** must be implementable/testable/reviewable by one agent in one development run and normally maps to one PR.
- If a work item discovers a second independent responsibility or external integration, split it before coding.
- Scope is deny-by-default: a work item permits only its objective/acceptance criteria and minimum supporting changes.
- Contracts precede alternative adapters; adapters precede benchmark-based default promotion.
- Legacy v1 code is only an implementation donor/regression source. Reuse it when it cleanly satisfies the owning v2 contract; never preserve or wrap a v1 API merely for backward compatibility.
- Before activating any work item that integrates or promotes a solver/model, refresh the candidate shortlist from `docs/14_TECHNOLOGY_SELECTION.md`, `docs/27_RESEARCH_CANDIDATE_COVERAGE.md`, the current adapter/model registry and current research evidence. Planned item names never freeze a 2026 winner.

## V2M0 — Production substrate ready

### V2L0 — V1→V2 migration and donor baseline

Purpose: establish v2.0 repository authority while preserving only useful legacy implementation donors, regression fixtures and historical evidence.

- `V2L0.1` — Inventory and migration map: classify every implemented v1 component as reusable donor/generalize/temporary wrap/deprecate/remove-later and archive the v1 machine roadmap/review state.
- `V2L0.2` — Replace active roadmap/state/components/reviews with v2 identifiers and validator coverage while preserving historical evidence.
- `V2L0.3` — Migration regression review: prove current fast CI and useful COLMAP/media donor fixtures still pass under v2 governance without making v1 compatibility a product requirement.

### V2L1 — SceneProject and artifact identity

Purpose: create the project/artifact vocabulary that all later production work shares.

- `V2L1.1` — `SceneProjectId` / `SceneProject` minimal immutable project contract.
- `V2L1.2` — `ArtifactId`, artifact kind and exact input-artifact references.
- `V2L1.3` — producer/model/checkpoint/configuration identity primitives.
- `V2L1.4` — provenance-class contract (`OBSERVED_RECONSTRUCTED`, `INFERRED`, `GENERATED`).
- `V2L1.5` — canonical artifact key/content-addressing contract and deterministic tests.
- `V2L1.6` — local codec/persistence round-trip for project/artifact metadata.

### V2L2 — Artifact DAG and cache

Purpose: make expensive work reusable and invalidation explicit.

- `V2L2.1` — immutable artifact dependency DAG contract with cycle rejection.
- `V2L2.2` — local artifact metadata store and dependency lookup.
- `V2L2.3` — cache lookup by canonical artifact key; no filename/time heuristics.
- `V2L2.4` — dependency-scoped invalidation planning without destructive deletion.
- `V2L2.5` — artifact materialization/verification metadata and corrupted-artifact failure path.
- `V2L2.6` — cache reuse/invalidation regression fixture and lot review.

### V2L3 — Adapter, model and shipping registries

Purpose: make algorithms replaceable without changing core semantics.

- `V2L3.1` — solver-independent adapter capability descriptor.
- `V2L3.2` — model/checkpoint identity + hardware/runtime requirements.
- `V2L3.3` — shipping status/license/reproducibility registry schema.
- `V2L3.4` — adapter failure/metrics/resume capability declaration.
- `V2L3.5` — registry validator: reject floating/unapproved defaults and incomplete production metadata.
- `V2L3.6` — register useful COLMAP/FFmpeg/ExifRead donor baselines through the v2 registry without making their v1 wrappers authoritative.

### V2L4 — Quality, failure and benchmark records

Purpose: create one common language for gates, escalation and comparison.

- `V2L4.1` — stable failure taxonomy contract.
- `V2L4.2` — typed multidimensional metric vector/metric provenance contract.
- `V2L4.3` — quality decision contract: PASS / ACCEPT_WITH_WARNINGS / RETRY / ESCALATE / UNRESOLVED.
- `V2L4.4` — benchmark record schema with fixture, adapter/model/config, quality mode, hardware and metric vector.
- `V2L4.5` — deterministic quality-policy evaluation baseline.
- `V2L4.6` — negative tests proving unknown metrics/failures do not silently pass and lot review.

## V2M1 — Arbitrary media can be understood and organized

### V2L5 — Media profiling

- `V2L5.1` — `MediaProfile` contract and evidence links.
- `V2L5.2` — deterministic image blur/sharpness + exposure/clipping metrics.
- `V2L5.3` — visual duplicate/near-duplicate and diversity signals behind an explicit profile artifact.
- `V2L5.4` — camera/input-class flags: still/video/360/drone/fisheye/rolling-shutter candidates without pretending certainty.
- `V2L5.5` — dynamic-likelihood / long-sequence / sparse-coverage profile signals.
- `V2L5.6` — profile fixture matrix and lot review.

### V2L6 — Adaptive frame selection

- `V2L6.1` — frame-selection policy contract separate from raw video ingestion.
- `V2L6.2` — donor adapter for the existing deterministic interval-keyframe implementation; no legacy API compatibility promise.
- `V2L6.3` — quality/diversity-aware selection baseline using only explicit profile metrics.
- `V2L6.4` — bounded density/resource controls for long videos.
- `V2L6.5` — provenance/timing preservation + selection regression fixtures and lot review.

### V2L7 — Pair candidate retrieval

- `V2L7.1` — canonical `PairCandidate`/candidate-source contract and dedup semantics.
- `V2L7.2` — reuse the sequential-pairing donor behind the canonical contract where still useful.
- `V2L7.3` — reuse the GPS-pairing donor behind the canonical contract where still useful.
- `V2L7.4` — deterministic classical vocabulary retrieval baseline.
- `V2L7.5` — modern visual/place-retrieval evidence adapter contract + first benchmark candidate; optional learned geo/time signals may contribute evidence but never define scene identity or capture time by themselves.
- `V2L7.6` — candidate-union/dedup/retrieval-recall benchmark and lot review.

### V2L8 — Scene identity and clustering

- `V2L8.1` — `SceneCluster` / scene-relationship hypothesis contract.
- `V2L8.2` — local-match/geometric-verification evidence adapter using the useful COLMAP donor baseline where appropriate.
- `V2L8.3` — deterministic cluster construction from verified relationships.
- `V2L8.4` — repeated/symmetric look-alike fail-closed fixture; weak similarity must remain separate/unresolved.
- `V2L8.5` — scalable clustering boundary for large candidate graphs and lot review.

### V2L9 — Temporal grouping and synchronization foundations

- `V2L9.1` — `TemporalGroup` and `SyncHypothesis` contracts.
- `V2L9.2` — explicit metadata/timecode synchronization evidence.
- `V2L9.3` — audio-correlation synchronization adapter using external media tooling.
- `V2L9.4` — visual-event synchronization interface; no implicit fusion into one timeline.
- `V2L9.5` — ambiguity/contradiction fixtures and lot review.

### V2L10 — Quality modes and route graph

- `V2L10.1` — `QualityMode` contract: PREVIEW / FAST / QUALITY / MASTER.
- `V2L10.2` — route-node/route-graph contract with explicit adapter capability requirements.
- `V2L10.3` — deterministic router inputs from data profile, budget, existing artifacts and prior failures.
- `V2L10.4` — allowlisted fallback/escalation policy with bounded termination.
- `V2L10.5` — auditable route-decision artifact and reasons.
- `V2L10.6` — route determinism/fail-closed tests and M1 review.

## V2M2 — Reliable static physical reconstruction

### V2L11 — Canonical camera/depth/geometry contracts

- `V2L11.1` — canonical `CameraSolution` contract with coordinate frame, intrinsics/extrinsics, uncertainty hooks and metrics.
- `V2L11.2` — `DepthField` and validity/confidence contract.
- `V2L11.3` — `PointMap` contract.
- `V2L11.4` — `GeometrySolution` assembly contract and local-scale semantics.
- `V2L11.5` — one-way migration/conversion from legacy `SparseReconstructionEstimate` when useful for donor fixtures or historical artifacts; no permanent legacy representation requirement.
- `V2L11.6` — coordinate-convention regression fixture and lot review.

### V2L12 — Classical precision geometry baseline

- `V2L12.1` — COLMAP adapter descriptor and input/output normalization through v2 artifact contracts.
- `V2L12.2` — useful existing feature/matching/verification donor stages consume v2 artifact/cache identities.
- `V2L12.3` — the classical incremental SfM baseline emits canonical `CameraSolution`/`GeometrySolution`.
- `V2L12.4` — disconnected/zero-model outcomes remain explicit.
- `V2L12.5` — real COLMAP end-to-end regression fixture migrated to v2 contracts and lot review.

### V2L13 — Feed-forward geometry

- `V2L13.1` — feed-forward camera/depth adapter interface and normalized outputs.
- `V2L13.2` — refresh the current shortlist, then integrate one approved preview candidate behind the contract.
- `V2L13.3` — exact checkpoint/hardware/license registry + reproducibility fixture.
- `V2L13.4` — geometry/camera quality metrics against controlled holdout/reference.
- `V2L13.5` — preview-vs-classical-baseline comparison without default promotion and lot review.

### V2L14 — Precision/global refinement and competing solutions

- `V2L14.1` — competing `GeometrySolution` comparison contract.
- `V2L14.2` — geometry-initialization/refinement adapter boundary.
- `V2L14.3` — classical BA/global refinement baseline path behind the boundary.
- `V2L14.4` — refresh the current shortlist, then integrate the first approved hybrid/global candidate.
- `V2L14.5` — consensus/disagreement metrics and unresolved policy.
- `V2L14.6` — controlled geometry benchmark and lot review.

### V2L15 — Dense depth / MVS / fusion

- `V2L15.1` — dense-depth artifact contract.
- `V2L15.2` — COLMAP MVS/PatchMatch donor baseline adapter where still useful under v2 contracts.
- `V2L15.3` — depth consistency/fusion contract.
- `V2L15.4` — learned depth-prior adapter boundary without automatic truth promotion.
- `V2L15.5` — coverage/holes/confidence outputs.
- `V2L15.6` — dense-depth benchmark and lot review.

### V2L16 — Explicit physical surface

- `V2L16.1` — `SurfaceModel` contract and intended-use metadata.
- `V2L16.2` — baseline depth-fusion-to-surface adapter.
- `V2L16.3` — collision/measurement suitability metrics.
- `V2L16.4` — optional hybrid surface candidate adapter boundary.
- `V2L16.5` — unsupported-region/hole semantics; no generated collision by default.
- `V2L16.6` — surface benchmark and M2 review.

## V2M3 — Photorealistic master scene and interactive runtime

### V2L17 — Color and photometric pipeline

- `V2L17.1` — source color/exposure metadata contract.
- `V2L17.2` — decode/working/output color-space convention layer.
- `V2L17.3` — exposure/white-balance normalization artifact.
- `V2L17.4` — clipping/saturation masks and incompatible-camera handling.
- `V2L17.5` — deterministic photometric fixture and lot review.

### V2L18 — Static appearance

- `V2L18.1` — canonical `AppearanceModel` contract distinct from surface.
- `V2L18.2` — Gaussian/radiance adapter capability contract.
- `V2L18.3` — refresh the current shortlist, then integrate one approved static appearance baseline.
- `V2L18.4` — held-out render metrics + geometry disagreement diagnostics.
- `V2L18.5` — in-the-wild exposure/distractor specialist adapter boundary.
- `V2L18.6` — appearance benchmark/default candidate review.

### V2L19 — Materials and environment

- `V2L19.1` — `EnvironmentModel` and sky/horizon separation contract.
- `V2L19.2` — baseline environment extraction/representation.
- `V2L19.3` — `MaterialModel` contract (albedo/normal/roughness/metallic with provenance).
- `V2L19.4` — inverse-rendering/material adapter boundary.
- `V2L19.5` — inferred-vs-measured material provenance tests and lot review.

### V2L20 — MasterScene assembly/versioning

- `V2L20.1` — `MasterScene` immutable component-reference contract.
- `V2L20.2` — scene version/reason/parent semantics.
- `V2L20.3` — compatibility checks across coordinate/color/time/provenance classes.
- `V2L20.4` — incremental new-media version creation without unrelated recomputation.
- `V2L20.5` — competing component selection evidence retained in master metadata.
- `V2L20.6` — master-scene versioning fixture and lot review.

### V2L21 — Runtime compiler core

- `V2L21.1` — `RuntimeTargetProfile` / `RuntimeScene` contracts.
- `V2L21.2` — geometry simplification/collision output boundary.
- `V2L21.3` — appearance packaging/compression boundary.
- `V2L21.4` — spatial chunk and LOD manifest.
- `V2L21.5` — global target memory/download/FPS budget validation.
- `V2L21.6` — runtime compilation provenance + master-link regression and lot review.

### V2L22 — Reference interactive viewer

- `V2L22.1` — viewer/runtime integration choice and exact dependency review.
- `V2L22.2` — orbit + first-person walk using explicit collision surface.
- `V2L22.3` — free-fly/drone camera and saved camera path.
- `V2L22.4` — appearance + environment rendering integration.
- `V2L22.5` — expert provenance/confidence inspection overlay.
- `V2L22.6` — representative load/FPS/memory fixture and lot review.

### V2L23 — Static end-to-end vertical slice

- `V2L23.1` — arbitrary static media -> SceneCluster -> geometry -> surface -> appearance -> MasterScene -> RuntimeScene integration fixture.
- `V2L23.2` — PREVIEW/FAST/QUALITY route comparison on the same fixture.
- `V2L23.3` — failure/holes/unresolved behavior fixture.
- `V2L23.4` — retained benchmark report and M3 review.

## V2M4 — Dynamic 4D reconstruction

### V2L24 — Dense tracking and static/dynamic decomposition

- `V2L24.1` — dense track/motion evidence contract.
- `V2L24.2` — refresh the current shortlist, then integrate one approved dense tracker.
- `V2L24.3` — dynamic/static mask/decomposition contract.
- `V2L24.4` — transient/semi-static/dynamic classification baseline.
- `V2L24.5` — contamination/occlusion fixtures and lot review.

### V2L25 — Short-video 4D preview

- `V2L25.1` — dynamic geometry/preview adapter contract.
- `V2L25.2` — refresh the current shortlist, then integrate one approved fast 4D preview candidate.
- `V2L25.3` — camera/depth/motion normalization into WRE artifacts.
- `V2L25.4` — temporal preview metrics and resource bounds.
- `V2L25.5` — fallback/unresolved fixture and lot review.

### V2L26 — Quality 4D reconstruction

- `V2L26.1` — quality dynamic-solution contract and comparison metrics.
- `V2L26.2` — refresh the current shortlist, then integrate the first approved quality 4D candidate.
- `V2L26.3` — geometry/appearance temporal consistency gate.
- `V2L26.4` — competing preview/quality solution comparison/fusion policy.
- `V2L26.5` — held-out/free-viewpoint dynamic fixture and lot review.

### V2L27 — Persistent dynamic entities and dynamic surface

- `V2L27.1` — `DynamicEntity`, `Trajectory` and identity contract.
- `V2L27.2` — occlusion persistence with explicit inferred uncertainty.
- `V2L27.3` — dynamic `SurfaceModel`/motion-field contract.
- `V2L27.4` — persistent-object/dynamic-surface adapter boundary.
- `V2L27.5` — identity/occlusion/deformation fixtures and lot review.

### V2L28 — Long video / streaming geometry

- `V2L28.1` — chunk/window/stateful processing contract.
- `V2L28.2` — refresh the current shortlist, then integrate one approved long-sequence/stateful geometry baseline.
- `V2L28.3` — drift metric and chunk-boundary artifacts.
- `V2L28.4` — overlap/loop/global refinement boundary.
- `V2L28.5` — bounded-memory long-sequence fixture and lot review.

### V2L29 — Multi-video event reconstruction

- `V2L29.1` — combine metadata/audio/visual sync evidence into explicit hypotheses.
- `V2L29.2` — cross-camera track graph contract.
- `V2L29.3` — per-camera trajectory/calibration composition.
- `V2L29.4` — joint/merged 4D adapter boundary.
- `V2L29.5` — ambiguous-sync and known-offset multi-camera fixture and lot review.

### V2L30 — Dynamic runtime/timeline

- `V2L30.1` — temporal runtime chunk/LOD contract.
- `V2L30.2` — playback/pause/frame-step timeline metadata.
- `V2L30.3` — dynamic collision bounds/interaction proxy boundary.
- `V2L30.4` — temporal streaming/prefetch policy.
- `V2L30.5` — dynamic FPS/memory/scrubbing benchmark and M4 review.

## V2M5 — Historical chronology

### V2L31 — Temporal evidence and epochs

- `V2L31.1` — capture-date evidence contract with uncertain intervals; metadata, archive context and learned geo/time signals may contribute evidence, but publication time or model prediction never silently becomes exact capture time.
- `V2L31.2` — `TemporalEpoch` grouping hypothesis.
- `V2L31.3` — contradictory/unknown date handling; publication time never silently becomes capture time.
- `V2L31.4` — epoch grouping fixture and lot review.

### V2L32 — Cross-epoch registration and change measurement

- `V2L32.1` — frame/scale-compatible cross-epoch alignment contract.
- `V2L32.2` — robust registration baseline adapter.
- `V2L32.3` — mature point/surface change-measurement adapter.
- `V2L32.4` — registration-error/occlusion/confidence outputs.
- `V2L32.5` — unchanged/true-change/misregistration fixture and lot review.

### V2L33 — TemporalState and ChangeEvent

- `V2L33.1` — `TemporalState` immutable validity/support contract.
- `V2L33.2` — `ChangeEvent` evidence/interval/confidence contract.
- `V2L33.3` — change-vs-registration gate.
- `V2L33.4` — reversible state transitions/supersession without deleting history.
- `V2L33.5` — contradiction/change-point fixture and lot review.

### V2L34 — Cross-temporal reconstruction support

- `V2L34.1` — sparse-epoch support/transfer contract preserving provenance.
- `V2L34.2` — refresh the current shortlist, then integrate one approved cross-temporal geometry/appearance candidate.
- `V2L34.3` — state-specific geometry/appearance ownership rules.
- `V2L34.4` — transfer hallucination/coverage safeguards and lot review.

### V2L35 — Historical runtime timeline

- `V2L35.1` — master timeline/state-index contract.
- `V2L35.2` — runtime epoch switching and historical chunk selection.
- `V2L35.3` — uncertain interval/provenance visualization.
- `V2L35.4` — chronology runtime/benchmark fixture and M5 review.

## V2M6 — Specialist routes and scale

### V2L36 — Sparse-view specialist route

- `V2L36.1` — sparse-coverage classifier threshold policy.
- `V2L36.2` — refresh the current shortlist, then integrate the first approved sparse-view specialist.
- `V2L36.3` — uncertainty/coverage artifact and conservative completion boundary.
- `V2L36.4` — sparse-view benchmark + routing policy update and lot review.

### V2L37 — Very large unordered collections

- `V2L37.1` — scalable retrieval-index contract.
- `V2L37.2` — submap/chunk reconstruction contract.
- `V2L37.3` — refresh the current shortlist, then integrate the first approved scalable/stateful mapping path.
- `V2L37.4` — global/submap/crowd-sourced reconstruction merge alignment/refinement and long-tail escalation boundary.
- `V2L37.5` — memory/time/candidate-recall large-collection benchmark and lot review.

### V2L38 — 360 / omnidirectional route

- `V2L38.1` — native equirectangular camera/calibration contract.
- `V2L38.2` — 360 profiling/routing signal.
- `V2L38.3` — refresh the current shortlist, then integrate the first approved omnidirectional reconstruction path.
- `V2L38.4` — environment separation and 360 benchmark/lot review.

### V2L39 — Drone / aerial route

- `V2L39.1` — aerial profile/scale/trajectory contract.
- `V2L39.2` — static aerial geometry baseline route.
- `V2L39.3` — dynamic UAV specialist adapter boundary.
- `V2L39.4` — altitude/scale/dynamic contamination benchmark and lot review.

### V2L40 — Blur / rolling shutter / HDR / low-light routes

- `V2L40.1` — specialist-condition profile evidence contract.
- `V2L40.2` — rolling-shutter camera-model/refinement adapter boundary.
- `V2L40.3` — blur-aware specialist adapter boundary and first candidate.
- `V2L40.4` — HDR/low-light specialist adapter boundary and first candidate.
- `V2L40.5` — specialist-trigger false-positive/quality benchmark and lot review.

### V2L41 — Difficult materials and repeated structure

- `V2L41.1` — reflective/translucent/water/weak-texture difficulty signals.
- `V2L41.2` — repeated/symmetric-structure disambiguation specialist boundary.
- `V2L41.3` — difficult-matcher/geometry specialist adapter boundary.
- `V2L41.4` — failure-specific benchmark and route update.
- `V2L41.5` — fail-closed specialist lot review.

### V2L42 — Advanced relighting / PBR

- `V2L42.1` — relightable environment/material capability contract.
- `V2L42.2` — refresh the current shortlist, then integrate one approved relighting candidate.
- `V2L42.3` — material-fusion/inverse-rendering candidate adapter.
- `V2L42.4` — relighting/material benchmark and lot review.

### V2L43 — Generated completion profiles

- `V2L43.1` — STRICT / REALISTIC / CINEMATIC completion-policy contract.
- `V2L43.2` — `CompletionArtifact` region/provenance linkage.
- `V2L43.3` — conservative static completion adapter boundary.
- `V2L43.4` — generated-vs-reconstructed visibility/export/runtime preservation.
- `V2L43.5` — provenance-confusion negative fixture and M6 review.

## V2M7 — Professional production and release hardening

### V2L44 — Human review and expert overrides

- `V2L44.1` — versioned manual-decision artifact contract.
- `V2L44.2` — media/frame and scene-cluster review operations.
- `V2L44.3` — camera/mask/region/temporal corrections.
- `V2L44.4` — competing-solver comparison/selection operation.
- `V2L44.5` — AUTO/ASSISTED/EXPERT state flow + reproducibility fixture and lot review.

### V2L45 — Interoperability/export

- `V2L45.1` — exchange-format capability/metadata contract.
- `V2L45.2` — COLMAP import/export compatibility.
- `V2L45.3` — mesh/material glTF/GLB export.
- `V2L45.4` — approved splat package export.
- `V2L45.5` — DCC/game-engine package boundary and round-trip/validation tests.

### V2L46 — Resource estimation and scheduling

- `V2L46.1` — `JobSpec` resource requirement contract.
- `V2L46.2` — local deterministic scheduler baseline.
- `V2L46.3` — empirical resource estimator inputs/output contract.
- `V2L46.4` — OOM/disk/time predicted-failure policy and chunk/downscale alternatives.
- `V2L46.5` — resource scheduling benchmark and lot review.

### V2L47 — Checkpoint/resume and distributed execution

- `V2L47.1` — checkpoint identity/compatibility contract.
- `V2L47.2` — resume/retry state machine with bounded attempts.
- `V2L47.3` — adapter checkpoint integration baseline.
- `V2L47.4` — remote/distributed worker interface without changing artifact semantics.
- `V2L47.5` — interruption/incompatible-checkpoint fixture and lot review.

### V2L48 — Security and untrusted execution hardening

- `V2L48.1` — media/parser/native-tool resource limits.
- `V2L48.2` — sandbox/container boundary for high-risk external execution where practical.
- `V2L48.3` — pinned/verified remote model/checkpoint download policy.
- `V2L48.4` — path/command/injection/oversized-media negative tests.
- `V2L48.5` — supply-chain/security review and lot closure.

### V2L49 — Continuous benchmark/default refresh

- `V2L49.1` — benchmark suite manifest and retained result store.
- `V2L49.2` — candidate-vs-default comparison runner.
- `V2L49.3` — promotion eligibility policy by profile/quality mode.
- `V2L49.4` — human approval/audit record for default changes.
- `V2L49.5` — regression guard preventing benchmark score from bypassing license/invariant checks.

### V2L50 — Packaging and reproducibility

- `V2L50.1` — supported installation/runtime profiles.
- `V2L50.2` — reproducibility manifest/export for a completed project.
- `V2L50.3` — project reopen/rehydration from retained artifacts months later.
- `V2L50.4` — upgrade/migration policy for artifact schemas/models.
- `V2L50.5` — clean-machine installation/reproduction fixture and lot review.

### V2L51 — Final validation program

- `V2L51.1` — complete profile coverage audit against the canonical blueprint.
- `V2L51.2` — full static geometry/appearance/runtime validation set.
- `V2L51.3` — full dynamic/long/multi-video validation set.
- `V2L51.4` — full historical/specialist/generated-provenance validation set.
- `V2L51.5` — security/reproducibility/performance validation.
- `V2L51.6` — v2.0 milestone review: no known blueprint capability omitted or silently conflated.

## Dependency philosophy

The roadmap deliberately leaves exact research winners out of work-item identity. For example, `V2L13.2` means “first approved feed-forward geometry candidate”, not “DA3 forever”. Before each solver/model integration item is activated, the owning PR must refresh the shortlist from `docs/14_TECHNOLOGY_SELECTION.md`, `docs/27_RESEARCH_CANDIDATE_COVERAGE.md`, the adapter/model registry and current evidence, then benchmark the selected candidate under the stable WRE contract.

The goal is to integrate the strongest currently approved specialist for each responsibility without turning the roadmap into a list of papers. New research normally changes the shortlist, benchmark result or default promotion—not the frozen architecture or work-item identity.

## Why there are many lots

The count is intentional. Geometry, surface, appearance, dynamics, chronology, runtime, orchestration and specialists have different failure modes and evidence. Combining them into fewer giant lots would make PRs ambiguous and encourage hidden coupling. The lots are shallow; work-item dependencies carry the sequencing.
