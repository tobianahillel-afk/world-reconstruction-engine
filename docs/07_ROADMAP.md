# Roadmap — transition notice

> **Blueprint v2 transition (2026-09-15):** the product target has been redefined by `01_PRODUCT.md`, `02_ARCHITECTURE.md`, `03_PIPELINE.md`, `14_TECHNOLOGY_SELECTION.md` and `15_PRODUCTION_RUNTIME.md` as an independent visual spatiotemporal reconstruction engine. The lot sequence below records the existing implementation roadmap and completed/in-progress engineering history; it is **not** the final roadmap for the v2 product target. A dedicated roadmap/state migration must map retained work into the new architecture, remove obsolete MONDE/world-graph assumptions, introduce missing production/runtime/appearance/4D lanes, and update `registry/work-items.yaml` plus `PROJECT_STATE.yaml` atomically. Until that migration merges, do not start a future legacy lot solely because it appears below.

The current machine-readable implementation source of truth is `../registry/work-items.yaml`. This document preserves the pre-v2 progression so completed work remains interpretable during migration.

## Legacy implementation lots

- **L0 Bootstrap & development system** — resumable repository, tooling, CI, reuse registry, fixtures, review gate.
- **L1 Core domain** — observations, cameras, fragments, provenance and persistence.
- **L2 Media ingestion** — images, hashes, EXIF/GPS/time, video/keyframes.
- **L3 COLMAP baseline** — features, matching, verification, incremental reconstruction, import, E2E fixture.
- **L4 Candidate retrieval & strategy routing** — sequence/GPS/vocabulary/graph candidate generation, ranking, deterministic FAST/STANDARD/ESCALATED routing and router benchmark.
- **L5 Evidence geometry** — pair hypotheses, residuals, competing F/E/H models, robust estimation, triplets/cycles.
- **L6 Fragment engine** — fragment creation, registration, unknown/disconnected lifecycle, local refinement.
- **L7 LIMAP structural geometry** — lines, hybrid registration, triangulation, VP/planes, holistic BA.
- **L8 Fragment merging** — candidate generation, Sim3, TEASER++, GICP, reversible merge, false-merge tests.
- **L9 World graph & absolute anchoring** — spatial constraints, GTSAM factors, GPS, GCP/known-landmark and reference-fragment anchors, loop closures, incremental optimization/covariance.
- **L10 Hypotheses & uncertainty** — competing placements, covariance, scoring, unresolved state, invalidation.
- **L11 Video/motion** — temporal frame graph, tracking, rigs, rolling shutter/IMU interfaces, trajectory priors.
- **L12 Multi-solver validation** — independent mapping engines and consensus/disagreement handling.
- **L13 Hard-case escalation** — affine/DSP/view-synthesis/aggressive escalation and controller; optional specialist proposal engines remain geometrically verified.
- **L14 Dense reconstruction** — depth, fusion, mesh and texture as optional layer.
- **L15 Temporal / 4D world** — epochs, cross-epoch alignment, scale/frame compatibility, py4dgeo change measurements, change hypotheses, change-point inference, time-valid GeometryState and reversible temporal transitions.
- **L16 Validation program** — synthetic/real holdouts, absolute-anchor validation, repeated facades, historical-change validation, video, regression/performance benchmark.
- **L17 MONDE boundary** — observation/evidence/estimated-geometry/provenance/temporal contracts.

## Legacy milestones

- **M0 — Repo ready:** L0.
- **M1 — Reconstruct a place:** L1–L3.
- **M2 — Build incrementally:** L4–L6.
- **M3 — High-accuracy world engine:** L7–L10.
- **M4 — Advanced temporal and integration:** L11–L17.

## What is expected to survive the migration

Completed work is not discarded merely because the product grew. The following capabilities are broadly compatible with blueprint v2 and should normally be mapped forward rather than rewritten without evidence:

- repository bootstrap, CI, fixtures and review discipline;
- immutable media ingestion, hashing and provenance;
- camera/media metadata contracts;
- video decoding/keyframe foundations;
- COLMAP adapter and sparse reconstruction baseline;
- sequential/GPS/visual retrieval infrastructure where useful;
- deterministic/cacheable artifacts;
- geometry validation concepts;
- explicit unresolved/failure states.

The migration should evaluate each existing component against new contracts rather than deleting or preserving it by name alone.

## Capabilities missing from the legacy roadmap that v2 must plan explicitly

At minimum the migrated roadmap must introduce first-class work for:

- media quality profiling and adaptive keyframing;
- same-scene clustering and scalable large-collection organization;
- model/adapter/benchmark registries;
- PREVIEW / FAST / QUALITY / MASTER routing;
- feed-forward geometry foundation-model adapters;
- long-sequence/stateful/streaming geometry;
- dense tracking and multi-video synchronization;
- static/dynamic decomposition;
- explicit physical surface pipeline;
- photorealistic Gaussian/radiance appearance pipeline;
- material, photometric and environment/sky layers;
- short-event 4D reconstruction and persistent objects;
- long-term chronological states and change detection;
- sparse-view, 360, drone, blur/HDR and other specialist routes;
- artifact DAG/content-addressed cache and checkpoint/resume;
- resource estimation and hardware scheduling;
- automatic quality gates and failure classification;
- human assisted/expert review workflows;
- master-scene versioning;
- LOD/compression/streaming and runtime compiler;
- collision/navigation assets independent of photorealistic appearance;
- web/game/XR reference runtime;
- strict/realistic/cinematic generated-completion policy;
- product-level benchmark program spanning geometry, rendering, temporal quality and runtime performance.

## Migration requirement

The next roadmap rewrite must be treated as architecture work, not as a cosmetic renumbering. It must:

1. inventory completed code and tests;
2. map reusable components to v2 responsibilities;
3. identify contracts that need expansion or deprecation;
4. define new lots/work items with dependencies and acceptance criteria;
5. include research/adaptor evaluation as explicit benchmark work rather than preselecting every 2026 model;
6. preserve small reviewable PRs and lot/milestone review gates;
7. update `registry/work-items.yaml`, `PROJECT_STATE.yaml`, component/dependency registries and relevant ADRs together;
8. prevent future agents from resuming obsolete MONDE-centric work solely because it existed in this legacy list.
