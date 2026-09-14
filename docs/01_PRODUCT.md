# Product definition

## Goal

Build an incremental spatial reconstruction engine that can ingest arbitrary images/video, reconstruct geometrically supported local fragments, place observations relative to known geometry, retain unknown fragments, merge fragments only when sufficient evidence exists, anchor locally valid reconstructions into an absolute world frame when independent control evidence arrives, and preserve multiple time-valid geometry states when the physical world changes.

## Primary success criterion

Minimize incorrect placements, false fragment merges, false absolute anchoring and false physical-change claims. Coverage is secondary to correctness: an unresolved result is preferable to a confident-looking wrong result, and a later state must not erase earlier supported geometry.

## Initial product milestones

- **M0 — Repo ready:** resumable agent workflow, testing/CI foundations and dependency/reuse policy.
- **M1 — Reconstruct a place:** media-to-`SpatialFragment` baseline using reused reconstruction engines.
- **M2 — Build incrementally:** retrieval, adaptive strategy routing, placement, disconnected fragments and bridging.
- **M3 — High-accuracy world engine:** structural geometry, robust merging, absolute anchoring, world graph and uncertainty.
- **M4 — Advanced temporal and integration:** video/IMU, multi-solver validation, hard-case escalation, dense reconstruction, temporal/4D world states, serious validation and MONDE boundary contracts.

## Non-goals for early milestones

- Reimplementing mature photogrammetry or 3D/4D change-measurement algorithms.
- Generating plausible geometry where observations do not support it.
- Requiring GPS or an absolute Earth frame for a locally valid reconstruction.
- Treating publication/upload time as capture time without evidence.
- Replacing historical geometry merely because a newer state is accepted.
- Building a UI before the geometry/evidence core is testable.
- Planet-scale deployment before fragment-level correctness is demonstrated.
