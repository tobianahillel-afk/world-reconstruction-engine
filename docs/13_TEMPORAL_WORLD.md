# Temporal / 4D world model

WRE's temporal goal is to represent how physical geometry changes over time without overwriting earlier supported states. This document defines the future L15 contract.

The first target is **historical/static-world change** (for example a facade, roof or extension changing between epochs), not continuous reconstruction of arbitrary moving people or vehicles.

## Core principle

Spatial disagreement across time is not automatically bad evidence.

If observations from different periods independently support incompatible geometry, WRE should be able to represent multiple time-valid states rather than forcing one timeless model or rejecting one period as an outlier.

```text
observations / reconstructions
           |
           v
    temporal evidence
           |
        epochs
           |
  cross-epoch alignment
           |
  change measurements
           |
 GeometryChangeHypothesis
           |
   validated change point
        /       \
 GeometryState A  GeometryState B
 [valid interval] [valid interval]
```

## Temporal evidence and epochs

A `TemporalEpoch` is a grouping/reference frame for evidence that is believed to describe a sufficiently coherent period for change analysis. Epoch membership must remain evidence-backed; uncertain dates may yield broad/overlapping intervals rather than invented exact timestamps.

Potential temporal evidence includes:

- resolved EXIF capture instants;
- ambiguous local capture strings;
- source publication/upload dates;
- archive catalog dates;
- externally supplied date intervals;
- video sequence relationships;
- geometric/appearance consistency that supports or contradicts proposed epoch grouping.

Source publication time is not automatically capture time. Each temporal observation retains provenance and semantics.

## GeometryState

The eventual domain model should provide a stable `GeometryState` concept with at least the semantic ability to record:

- a state identity;
- the spatial/world object or fragment lineage it describes;
- validity interval (`valid_from` / `valid_to`) or an explicitly uncertain interval;
- a reference to canonical geometry products;
- supporting observations/evidence;
- producing run/configuration;
- confidence/status and uncertainty;
- supersession/transition relationships without destructive history rewriting.

A `GeometryState` is estimated geometry, not a raw observation.

## GeometryChangeHypothesis

A change hypothesis links incompatible but potentially time-consistent geometry. It should be able to record:

- compared states/epochs/geometry references;
- measured change evidence;
- candidate transition interval/change point;
- residuals and uncertainty;
- supporting and contradicting observations;
- algorithm/version/configuration provenance;
- status such as candidate, accepted, rejected or unresolved.

Acceptance must remain reversible and evidence-backed.

## Reuse py4dgeo instead of reimplementing change measurement

L15 plans a py4dgeo adapter for mature multi-epoch point-cloud change analysis (including M3C2-family measurements and related change-analysis facilities).

WRE does not treat a py4dgeo output as a semantic world-state decision. The split is:

```text
py4dgeo / mature geometry algorithms
  -> measured multi-epoch geometric change

WRE temporal layer
  -> provenance
  -> evidence association
  -> hypotheses
  -> validity intervals
  -> contradiction handling
  -> acceptance/rejection/unresolved state
```

No custom M3C2 implementation should be added without an ADR.

## Cross-epoch alignment

Change measurement is only meaningful after comparable geometry is placed in compatible frames. L15.3 therefore owns explicit cross-epoch alignment policy and uncertainty before the py4dgeo adapter is treated as valid evidence.

Alignment must explicitly account for **coordinate frame, orientation, translation and scale**. A monocular/local SfM fragment whose scale is unresolved cannot be interpreted with metric 3D-change distances until scale has been independently established or jointly estimated with sufficient evidence. The transform model used for alignment (for example Sim(3) versus rigid/metric) and its uncertainty must remain traceable.

Alignment must distinguish true physical change from registration error. Stable areas/control geometry may be used to estimate alignment, while changed regions must not dominate that estimate. Change measurements whose alignment uncertainty is too large relative to the measured effect remain unresolved.

## Change-point inference

A transition time may be:

- directly observed;
- bounded between a last-known-old and first-known-new observation;
- supported probabilistically by multiple observations;
- unresolved.

WRE must not invent an exact change timestamp when the evidence only supports an interval.

## Contradictory temporal evidence

Contradictions are first-class. Examples:

- two images claimed to be from the same date support incompatible geometry;
- archive metadata and visual/geometric evidence disagree;
- two candidate alignments imply different changes;
- apparent change can be explained by occlusion or reconstruction uncertainty.

Such cases remain competing/unresolved until independent evidence resolves them.

## Reversible transitions

Accepting a new historical state must not destroy the previous state or its evidence. Temporal transitions and state lineage must be reversible/auditable in the same spirit as fragment merges.

## Validation

The shifted L16 Validation Program must include a historical-change fixture that validates the temporal engine itself, not merely the presence of differently dated observations. Tests should include:

- unchanged structure across epochs;
- known synthetic change;
- uncertain/interval-only dating;
- false apparent change from misregistration or unresolved scale;
- contradictory dates;
- repeated facade/ambiguous geometry across time.

## Optional visualization

Photorealistic view synthesis (for example Nerfstudio/Gaussian-splat-style outputs) may be added as an optional visualization product. It never becomes WRE's canonical world geometry or replaces provenance/validity semantics.
