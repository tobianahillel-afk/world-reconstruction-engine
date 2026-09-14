# Roadmap

The machine-readable source of truth is `../registry/work-items.yaml`. This document describes the intended progression.

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
- **L15 Temporal / 4D world** — epochs, cross-epoch alignment, py4dgeo change measurements, change hypotheses, change-point inference, time-valid GeometryState and reversible temporal transitions.
- **L16 Validation program** — synthetic/real holdouts, repeated facades, historical-change validation, video, regression/performance benchmark.
- **L17 MONDE boundary** — observation/evidence/estimated-geometry/provenance/temporal contracts.

## Milestones

- **M0 — Repo ready:** L0.
- **M1 — Reconstruct a place:** L1–L3.
- **M2 — Build incrementally:** L4–L6.
- **M3 — High-accuracy world engine:** L7–L10.
- **M4 — Advanced, temporal and integration:** L11–L17.

## Sequencing notes

- The Strategy Router is implemented in L4 before the heavy L13 escalation engines. It selects FAST/STANDARD/ESCALATED routes but never lowers acceptance thresholds.
- Absolute world placement is optional until evidence exists. L9 adds GPS/GCP/reference-fragment factors rather than requiring GPS during reconstruction.
- L15 follows dense reconstruction so multi-epoch point-cloud change analysis can reuse stable dense products where needed; temporal models may also reference sparse/structural geometry when appropriate.
- L16 historical-change fixtures validate the L15 temporal engine rather than standing in for it.
- Candidate dependencies recorded for future lots are not installed by roadmap changes; each owning work item performs its own version/license/integration review.
