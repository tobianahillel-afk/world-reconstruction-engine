# Roadmap

The machine-readable source of truth is `../registry/work-items.yaml`. This document describes the intended progression.

- **L0 Bootstrap & development system** — resumable repository, tooling, CI, reuse registry, fixtures, review gate.
- **L1 Core domain** — observations, cameras, fragments, provenance and persistence.
- **L2 Media ingestion** — images, hashes, EXIF/GPS/time, video/keyframes.
- **L3 COLMAP baseline** — features, matching, verification, incremental reconstruction, import, E2E fixture.
- **L4 Candidate retrieval** — sequence/GPS/vocabulary/graph candidate generation and ranking.
- **L5 Evidence geometry** — pair hypotheses, residuals, competing F/E/H models, robust estimation, triplets/cycles.
- **L6 Fragment engine** — fragment creation, registration, unknown/disconnected lifecycle, local refinement.
- **L7 LIMAP structural geometry** — lines, hybrid registration, triangulation, VP/planes, holistic BA.
- **L8 Fragment merging** — candidate generation, Sim3, TEASER++, GICP, reversible merge, false-merge tests.
- **L9 World graph** — spatial constraints, GTSAM factors, GPS/loop closures, incremental optimization/covariance.
- **L10 Hypotheses & uncertainty** — competing placements, covariance, scoring, unresolved state, invalidation.
- **L11 Video/motion** — temporal graph, tracking, rigs, rolling shutter/IMU interfaces, trajectory priors.
- **L12 Multi-solver validation** — independent mapping engines and consensus/disagreement handling.
- **L13 Hard-case escalation** — affine/DSP/view-synthesis escalation and controller.
- **L14 Dense reconstruction** — depth, fusion, mesh and texture as optional layer.
- **L15 Validation program** — synthetic/real holdouts, repeated facades, historical change, regression/performance benchmark.
- **L16 MONDE boundary** — observation/evidence/estimated-geometry/provenance/temporal contracts.
