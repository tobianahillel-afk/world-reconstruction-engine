# ADR 0001 — Adaptive routing, absolute anchoring and temporal world states

- Status: Accepted
- Date: 2026-09-15
- Scope: future WRE architecture and roadmap; no solver implementation in this ADR

## Context

WRE must ingest heterogeneous photos and videos, including unordered Internet imagery with little or no GPS, and build spatial models without forcing unsupported placements. The existing pipeline already separates raw observations, derived evidence and estimated geometry and already prefers cheap methods before expensive escalation.

Three gaps need to be explicit before implementation continues far into reconstruction:

1. easy observations should not pay the cost of every available solver path when a cheaper path can reach the same evidence threshold;
2. reconstructions without native GPS need a first-class way to become absolutely anchored later through ground-control points, known landmarks or already-georeferenced fragments;
3. historical imagery needs more than a validation fixture: WRE needs first-class time-valid geometry states and change hypotheses so real physical change is not misclassified as bad evidence.

## Decision 1 — Add a deterministic Reconstruction Strategy Router

WRE will add a lightweight strategy-routing layer after candidate generation/ranking and before expensive matching/reconstruction escalation.

The initial route classes are conceptually:

- `FAST` — narrowly targeted work for observations with strong, low-ambiguity priors/evidence or reusable cached products;
- `STANDARD` — the normal retrieval/matching/geometry path;
- `ESCALATED` — broadened or more expensive processing for unresolved/hard cases.

The exact names remain an implementation detail until L4.6, but the semantics are fixed.

### Invariant

A cheaper route may reduce the **number or cost of operations**, but it must never reduce the evidence/acceptance threshold required to accept a placement, correspondence, reconstruction or merge.

Routing decisions must be deterministic and auditable. Persist or expose enough information to explain:

- inputs/signals considered;
- selected route;
- thresholds/configuration version;
- reasons for the decision;
- escalation conditions and outcome.

Learned retrieval/matching components may propose candidates where explicitly integrated, but geometric verification remains authoritative.

Heavy hard-case algorithms remain owned by L13. L4 routing decides when to request escalation; it does not duplicate L13.

## Decision 2 — Add first-class absolute anchoring constraints

The world graph will represent absolute anchors as constraints/factors rather than overwriting fragment geometry.

In addition to GPS factors, L9 will explicitly support:

- ground-control-point / known-landmark factors;
- reference-fragment / already-georeferenced-world anchors.

This allows a collection reconstructed entirely without GPS to remain locally valid and later acquire absolute world placement when independent evidence becomes available.

Absolute anchoring must preserve provenance, uncertainty and residuals. An anchor is evidence, not unquestionable truth.

## Decision 3 — Add a Temporal / 4D World lot

A new L15 will model historical geometric state explicitly. The former validation lot becomes L16 and the MONDE boundary becomes L17.

The temporal layer will distinguish:

- temporal evidence / epoch association;
- cross-epoch alignment;
- measured geometric change;
- change hypotheses/change points;
- time-valid `GeometryState` records;
- contradictory temporal evidence;
- reversible state transitions.

Historical disagreement is not automatically an outlier. If independent observations support incompatible geometry at different times, WRE should be able to represent multiple geometry states with validity intervals instead of overwriting history.

## Decision 4 — Reuse py4dgeo for 3D/4D change measurement

WRE will plan an adapter to py4dgeo for mature multi-epoch point-cloud change analysis such as M3C2-family measurements. WRE will not reimplement those foundational change-measurement algorithms without a later ADR.

py4dgeo measurements are evidence. WRE remains responsible for provenance, temporal hypotheses, state validity, contradiction handling and acceptance policy.

## Decision 5 — Optional specialist tools remain behind adapters

The dependency registry will record, but not install in this architecture PR, candidates including:

- Kimera-RPGO for robust pose-graph/outlier rejection;
- Basalt for optional visual-inertial/video trajectory work;
- Kalibr for camera/IMU/rolling-shutter calibration;
- hloc and LightGlue for optional difficult visual retrieval/matching proposals;
- Nerfstudio for optional photorealistic visualization/view synthesis.

These tools do not become canonical authorities merely because they produce an answer. Licenses, model-weight licenses and integration modes must be reviewed when their owning work item is implemented.

Nerfstudio/Gaussian-splat-style rendering, if used, is a visualization product. It must not replace canonical sparse/dense WRE geometry or its provenance.

## Consequences

- L4 gains a Strategy Router and its benchmark.
- L9 gains explicit GCP/known-landmark and reference-fragment absolute anchors.
- A new L15 Temporal / 4D World lot is inserted.
- Validation shifts to L16; MONDE boundary shifts to L17.
- M4 expands through L17.
- New component and dependency registry entries are required.
- `PROJECT_STATE.yaml` remains on the current active implementation item; this roadmap amendment does not jump development ahead.

## Non-goals

This ADR does not implement:

- the strategy router;
- GCP factors;
- py4dgeo integration;
- temporal-state inference;
- learned feature/matching models;
- Nerfstudio/Gaussian splatting;
- dynamic-object 4D reconstruction of continuously moving people/vehicles.

Those remain future work items and must pass their normal PR/lot/milestone gates.
