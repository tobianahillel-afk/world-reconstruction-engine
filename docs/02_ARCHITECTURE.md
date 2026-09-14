# Architecture

## Epistemic layers

WRE separates three layers:

1. **Observation** — raw media and source metadata (bytes/hash, timestamps, EXIF/GPS, camera metadata, provenance).
2. **Derived evidence** — features, matches, verified two-view geometry, tracks, residuals and priors.
3. **Estimated geometry** — camera poses, local geometry, fragments, placements, transforms and merge assertions.

Derived data must point back to its inputs and producing run/version.

## Module boundaries

Planned package domains:

- `ingest` — image/video and metadata ingestion
- `features` — feature-engine adapters
- `retrieval` — candidate pair generation
- `matching` — pair matching adapters/orchestration
- `geometry` — robust geometric verification and evidence
- `reconstruction` — COLMAP/LIMAP/other reconstruction adapters
- `fragments` — local fragment lifecycle
- `registration` — incremental image placement
- `merging` — fragment alignment and reversible merge assertions
- `graph` — world-level constraints/optimization
- `uncertainty` — covariance/confidence representation
- `validation` — holdouts, consistency and regression metrics
- `dense` — optional dense depth/mesh/texturing
- `cli` — thin user/developer interface

External engines remain behind adapters. Core domain models must not depend on a particular solver's private representation.

## Global strategy

Never construct one monolithic world reconstruction. Maintain local fragments/submaps and optimize world-level transforms/constraints. Open/re-optimize local geometry only when required.
