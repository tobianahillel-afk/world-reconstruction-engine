# Product definition

## Goal

Build an incremental spatial reconstruction engine that can ingest arbitrary images/video, reconstruct geometrically supported local fragments, place observations relative to known geometry, retain unknown fragments, and merge fragments only when sufficient evidence exists.

## Primary success criterion

Minimize incorrect placements and false fragment merges. Coverage is secondary to correctness: an unresolved result is preferable to a confident-looking wrong result.

## Initial product milestones

- **M0 — Repo ready:** resumable agent workflow, testing/CI foundations and dependency/reuse policy.
- **M1 — Reconstruct a place:** media-to-`SpatialFragment` baseline using reused reconstruction engines.
- **M2 — Build incrementally:** retrieval, placement, disconnected fragments and bridging.
- **M3 — High-accuracy world engine:** structural geometry, robust merging, world graph and uncertainty.
- **M4 — Advanced/research:** video/IMU, multi-solver validation, hard-case escalation, dense reconstruction, certifiable audits and MONDE boundary contracts.

## Non-goals for early milestones

- Reimplementing mature photogrammetry algorithms.
- Generating plausible geometry where observations do not support it.
- Building a UI before the geometry/evidence core is testable.
- Planet-scale deployment before fragment-level correctness is demonstrated.
