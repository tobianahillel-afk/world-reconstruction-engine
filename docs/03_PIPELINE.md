# Pipeline

Target pipeline, introduced incrementally by roadmap lots:

```text
media
  -> ingest / metadata / provenance
  -> candidate retrieval (time, sequence, GPS, visual index, graph)
  -> candidate ranking
  -> deterministic strategy routing (FAST / STANDARD / ESCALATED)
  -> point/line feature evidence
  -> matching
  -> robust multi-model geometric verification
  -> tracks / triplet / cycle consistency
  -> local reconstruction (COLMAP baseline, LIMAP structural extension)
  -> SpatialFragment
  -> incremental placement or new unknown fragment
  -> robust fragment merge (Sim3 / TEASER++ / ICP evidence)
  -> world factor graph
       -> relative fragment/loop constraints
       -> GPS constraints
       -> GCP / known-landmark constraints
       -> reference-fragment / absolute anchors
  -> uncertainty / competing hypotheses
  -> optional dense reconstruction
  -> temporal epochs / cross-epoch alignment
  -> multi-epoch geometric change measurement
  -> GeometryChangeHypothesis / GeometryState validity
  -> validation / holdouts / regression
  -> optional non-canonical photorealistic visualization
```

## Strategy routing and escalation

Cheap/high-confidence methods run first. Expensive methods run only for unresolved hard cases.

The L4 Strategy Router makes this policy explicit and auditable. It chooses how much work to attempt first, but **never lowers the evidence threshold** required to accept a result. FAST, STANDARD and ESCALATED paths differ in cost/candidate breadth/reuse, not in truth criteria.

Examples:

- cached/sequence/strong-prior evidence before broad retrieval;
- standard SIFT/COLMAP capabilities before affine/DSP/view-synthesis escalation;
- local validation before multi-solver consensus;
- sparse geometry before dense reconstruction;
- targeted incremental registration before rebuilding a larger local model when appropriate.

L13 owns the expensive hard-case algorithms. L4 owns when to request them.

See [`12_ADAPTIVE_RECONSTRUCTION.md`](12_ADAPTIVE_RECONSTRUCTION.md).

## Absolute anchoring

Local reconstruction does not require GPS. A fragment may remain valid in its own local frame until absolute evidence arrives.

GPS, ground-control points, known landmarks and already-georeferenced reference fragments enter the world graph as constraints/factors with provenance, uncertainty and residuals. Conflicting anchors remain diagnosable/competing evidence; they are not silently averaged into truth.

## Temporal / 4D pipeline

Historical imagery is grouped/associated into evidence-backed epochs only when timing evidence permits it. Comparable geometry must be aligned before change measurement. Mature multi-epoch change algorithms (planned via py4dgeo adapter) produce measurements; WRE turns those measurements into provenance-bearing change hypotheses and time-valid geometry states only after validation.

An apparent difference can remain `UNRESOLVED` if registration error, occlusion, uncertain dating or competing geometry can explain it. Accepted changes never delete the previous supported state.

See [`13_TEMPORAL_WORLD.md`](13_TEMPORAL_WORLD.md).

## Acceptance philosophy

A solver, learned matcher, router or change detector result is a measurement/hypothesis, not automatically truth. Acceptance is based on geometric support, residuals, uncertainty, provenance and consistency with independent evidence.
