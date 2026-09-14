# Pipeline

Target pipeline, introduced incrementally by roadmap lots:

```text
media
  -> ingest / metadata / provenance
  -> candidate retrieval (time, sequence, GPS, visual index, graph)
  -> point/line feature evidence
  -> matching
  -> robust multi-model geometric verification
  -> tracks / triplet / cycle consistency
  -> local reconstruction (COLMAP baseline, LIMAP structural extension)
  -> SpatialFragment
  -> incremental placement or new unknown fragment
  -> robust fragment merge (Sim3 / TEASER++ / ICP evidence)
  -> world factor graph
  -> uncertainty / competing hypotheses / validation
  -> optional dense reconstruction
```

## Escalation policy

Cheap/high-confidence methods run first. Expensive methods run only for unresolved hard cases. Examples: standard SIFT before affine/DSP/view-synthesis escalation; local validation before multi-solver consensus; sparse geometry before dense reconstruction.

## Acceptance philosophy

A solver result is a measurement/hypothesis, not automatically truth. Acceptance is based on geometric support, residuals, uncertainty and consistency with independent evidence.
