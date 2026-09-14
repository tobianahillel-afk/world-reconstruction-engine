# Adaptive reconstruction and absolute anchoring

This document defines the future contracts for L4 strategy routing and L9 absolute anchoring. It is architectural guidance, not an implementation of either capability.

## Why route work adaptively

WRE should spend only the computation needed to reach the normal evidence standard. Some observations arrive with strong sequence, GPS, graph-neighbour or cached-feature context; others are visually ambiguous, historically changed or far from existing viewpoints.

The route must therefore control **cost**, never truth criteria.

```text
candidate signals / cached evidence
             |
             v
      Candidate ranking
             |
             v
      Strategy Router
       /      |       \
    FAST   STANDARD  ESCALATED
       \      |       /
             v
   geometric/evidence checks
             |
       sufficient?
       /        \
     yes        no
     accept   escalate / unresolved
```

## Strategy Router contract

The router belongs to L4 after candidate ranking. It consumes cheap/reusable signals rather than running a second reconstruction engine.

Candidate signals may include:

- exact-content identity / previously computed products;
- video/sequence neighbourhood;
- resolved GPS or other spatial priors;
- graph-neighbour candidates;
- visual-retrieval rank/score;
- known fragment candidates;
- cached features/matches/geometric checks;
- ambiguity measures such as multiple competing fragments;
- previously observed failure/escalation state.

The router emits a route decision with auditable reasons and configuration/version identity.

### FAST

FAST is appropriate when a small targeted candidate set can plausibly reach the normal acceptance standard. It may reuse cached features, restrict pair candidates or try incremental registration first.

FAST is **not** permission to skip required geometric checks.

### STANDARD

STANDARD uses the normal candidate retrieval/matching/verification/reconstruction path designed for ordinary imagery.

### ESCALATED

ESCALATED asks later hard-case machinery for broader or more expensive processing. L13 owns specialist escalation algorithms; L4 only selects/escalates.

### Required safety invariant

FAST, STANDARD and ESCALATED must satisfy the **same acceptance policy and evidence standard** for an accepted claim. They may use different solver-specific thresholds or intermediate measurements when those are calibrated to equivalent semantics and documented; a cheaper route must never weaken the final proof required for acceptance.

The route changes how evidence is sought, not what counts as sufficient evidence.

If evidence remains insufficient or contradictory, the outcome is `UNRESOLVED`, not a forced placement.

## Existing tools to reuse

The router should orchestrate existing capabilities rather than duplicate them. Expected examples include COLMAP sequential/spatial/vocabulary/transitive matching, incremental image registration, existing WRE retrieval signals and later specialist adapters.

Optional learned retrieval/matching tools may improve candidate proposals, especially for difficult Internet/archive imagery, but their output must still pass deterministic geometric verification.

## Benchmarking

L4.7 must measure both quality and avoided work. Useful metrics include:

- candidate pairs evaluated;
- feature/match work reused versus recomputed;
- successful registrations by route;
- escalations and failed fast attempts;
- false-acceptance rate (must not regress versus baseline);
- runtime/compute saved on easy sequences and near-duplicate viewpoints;
- unresolved rate on hard/ambiguous fixtures.

Performance improvements do not justify lowering geometric quality gates.

# Absolute anchoring

A reconstruction can be internally coherent without knowing where it belongs on Earth. WRE must preserve that state rather than requiring GPS at ingestion time.

Absolute placement may arrive later from:

- GPS observations;
- surveyed ground-control points (GCPs);
- known landmarks/control points with world coordinates;
- an already-georeferenced reference fragment;
- other explicit world-frame constraints.

## Constraint semantics

Absolute anchors enter the world graph as evidence-bearing constraints/factors. They should carry, where applicable:

- source/provenance;
- coordinate reference system / world-frame identity;
- covariance or uncertainty;
- validity/time information when relevant;
- residuals after optimization;
- producing parser/adapter/configuration version.

They do not destructively rewrite local fragment geometry.

## GCP / known-landmark factors

L9.5 will model correspondences between reconstructed/local geometry and known world points. Existing OpenSfM GCP/checkpoint patterns are a useful reference/adapter candidate; WRE owns the solver-independent contract and evidence policy.

## Reference-fragment anchors

L9.6 will allow a fragment with an established world placement to constrain another fragment through validated relative geometry. This is especially important when the newly arriving collection has no native GPS.

The local fragment may still have unresolved metric scale. The reference relationship must therefore retain the actual transform model used (for example a validated Sim(3) when scale is unknown, or a rigid/metric transform when scale is independently established), together with scale/transform uncertainty and residual evidence.

A reference anchor must not become a hidden merge. Relative placement evidence and the absolute anchor remain separately traceable.

## Failure behavior

Conflicting absolute anchors should produce residuals/competing hypotheses or an unresolved state. WRE must not average incompatible anchors merely to obtain a coordinate.
