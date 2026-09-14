# Architecture

## Epistemic layers

WRE separates three layers:

1. **Observation** — raw media and source metadata (bytes/hash, timestamps, EXIF/GPS, camera metadata, provenance).
2. **Derived evidence** — features, matches, verified two-view geometry, tracks, residuals, priors, change measurements and routing/validation evidence.
3. **Estimated geometry** — camera poses, local geometry, fragments, placements, transforms, merge assertions and time-valid geometry states.

Derived data must point back to its inputs and producing run/version.

## Module boundaries

Planned package domains:

- `ingest` — image/video and metadata ingestion
- `features` — feature-engine adapters
- `retrieval` — candidate pair generation
- `routing` — deterministic FAST/STANDARD/ESCALATED strategy decisions and escalation reasons
- `matching` — pair matching adapters/orchestration
- `geometry` — robust geometric verification and evidence
- `reconstruction` — COLMAP/LIMAP/other reconstruction adapters
- `fragments` — local fragment lifecycle
- `registration` — incremental image placement
- `merging` — fragment alignment and reversible merge assertions
- `graph` — world-level constraints/optimization, including absolute anchors
- `uncertainty` — covariance/confidence representation
- `temporal` — epochs, cross-epoch change evidence, GeometryState and temporal transitions
- `validation` — holdouts, consistency and regression metrics
- `dense` — optional dense depth/mesh/texturing
- `visualization` — optional non-canonical photorealistic rendering/view-synthesis products
- `cli` — thin user/developer interface

External engines remain behind adapters. Core domain models must not depend on a particular solver's private representation.

## Global spatial strategy

Never construct one monolithic world reconstruction. Maintain local fragments/submaps and optimize world-level transforms/constraints. Open/re-optimize local geometry only when required.

A fragment is allowed to exist without absolute world coordinates. GPS, ground-control points, known landmarks and already-georeferenced reference fragments later enter the world graph as provenance-bearing constraints with uncertainty rather than destructively rewriting fragment geometry.

## Adaptive compute strategy

WRE chooses the cheapest route likely to reach the normal evidence threshold, then escalates when needed. A FAST route may reduce candidates or reuse cached products, but it does not lower geometric acceptance requirements. Routing decisions must remain deterministic/auditable enough to explain what was tried and why.

See [`12_ADAPTIVE_RECONSTRUCTION.md`](12_ADAPTIVE_RECONSTRUCTION.md) and ADR 0001.

## Temporal / 4D strategy

Canonical world geometry is time-aware when evidence supports physical change. Historical disagreement is not automatically an outlier: WRE may represent multiple `GeometryState` records with validity intervals and reversible change hypotheses.

Mature 3D/4D change-measurement algorithms should be reused behind adapters; WRE owns provenance, temporal state, contradiction handling and acceptance policy. The first temporal target is historical/static-world change, not arbitrary continuous dynamic-object reconstruction.

See [`13_TEMPORAL_WORLD.md`](13_TEMPORAL_WORLD.md) and ADR 0001.

## Optional learned and rendering components

Learned retrieval/matching systems may propose candidates when an owning work item explicitly integrates them. Their outputs remain derived evidence and must pass geometric verification before acceptance.

Photorealistic neural/Gaussian-splat-style rendering is optional visualization. It must never replace canonical WRE geometry, uncertainty or provenance.
