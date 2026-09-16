# WRE v2.0 product and architecture freeze

Status: **FROZEN** for the WRE v2.0 target unless an explicit blueprint-change decision supersedes this document.

## Why this document exists

WRE v2.0 is the authoritative product. The repository is no longer developing toward completion of the legacy v1 architecture and then evolving it into v2. The v1 implementation is only a source of proven code, tests, fixtures and operational lessons that may be reused when they cleanly satisfy a v2 contract.

The purpose of the freeze is to keep the product stable while allowing the scientific implementation to improve continuously.

## What is frozen

The following are product/architecture commitments for v2.0:

- WRE is an independent visual spatiotemporal reconstruction engine for arbitrary real-world photos and videos.
- The target outcome is a photorealistic, temporal, freely navigable 3D/4D scene, not a point cloud, mesh, Gaussian Splat, one solver or one fixed pipeline.
- Observation, organization/evidence, geometry, surface, appearance, materials, environment, dynamics/history and runtime representations remain separate responsibilities with explicit contracts.
- Continuous event-time 4D and long-term historical chronology are distinct temporal regimes.
- Physical geometry and photorealistic appearance remain distinct even when one method can estimate both.
- `MasterScene` and target-specific `RuntimeScene` are distinct products.
- `OBSERVED_RECONSTRUCTED`, `INFERRED` and `GENERATED` provenance remain distinguishable; generated completion never silently becomes measured reality.
- PREVIEW, FAST, QUALITY and MASTER remain explicit quality modes with different compute/quality tradeoffs.
- The engine profiles the data and routes work through interchangeable specialists rather than forcing one universal solver chain.
- Core semantics are expressed through stable WRE contracts and versioned artifacts; external solvers/models remain behind adapters.
- Expensive work is represented by an artifact DAG with reproducible identities, dependency-aware reuse/invalidation and explicit quality evidence.
- Quality gates, failure semantics, fallback/escalation, benchmark evidence, human review, resumability, scheduling, compression/LOD/streaming and reproducibility are first-class production capabilities.
- Unknown, ambiguous or unsupported states fail closed rather than being converted into plausible-looking truth.
- New media may incrementally enrich a project without gratuitously recomputing unrelated validated artifacts.

The canonical detailed definitions remain in `01_PRODUCT.md`, `02_ARCHITECTURE.md`, `03_PIPELINE.md`, `15_PRODUCTION_RUNTIME.md`, `24_SYSTEM_INVARIANTS.md` and `25_V2_ROADMAP.md`.

## What is intentionally not frozen

The following are implementation choices, not product architecture:

- the current best geometry, matching, depth, tracking, 4D, appearance, material, relighting, compression or streaming model;
- which research paper or library is the current default adapter;
- model/checkpoint versions;
- benchmark winners and per-profile default promotions;
- hardware-specific kernels and implementation optimizations;
- calibrated thresholds that are owned by explicit quality/benchmark policy;
- specialist candidates that may appear after this freeze.

A future method can replace a current specialist without a blueprint rewrite when it implements the same WRE contract, passes the relevant benchmark/quality gates, has acceptable licensing/shipping/reproducibility metadata, and preserves system invariants.

In short: **freeze the product and contracts; continuously improve the specialists.**

## V1 is a legacy implementation donor, not a compatibility target

The legacy v1 implementation is not the product specification and is not a backward-compatibility target for v2.

Rules:

- No v2 domain contract, data model, route, API or architecture decision may be weakened or distorted merely to preserve a v1 interface or sequencing assumption.
- Reuse v1 code only when it is independently useful and cleanly satisfies the owning v2 contract.
- A retained v1 component may be wrapped temporarily when that is the cheapest safe migration path, but the wrapper is not a promise of permanent compatibility.
- If a clean v2 implementation is simpler, more correct or better aligned with the frozen architecture, it may replace the v1 implementation once acceptance/regression evidence exists.
- Obsolete v1 code may be deprecated and removed after its required v2 replacement and migration coverage are validated.
- Passing v1 fixtures may remain as regression evidence where they test behavior that v2 still promises; they do not make the legacy API authoritative.
- Git history, old PRs and review evidence are preserved for traceability even when the corresponding code is later deleted.

Therefore, references to a **retained V1 baseline** mean a temporary regression/reference baseline and implementation donor. They do **not** mean “finish V1”, “preserve V1 architecture”, or “maintain V1 compatibility indefinitely”.

## Scientific replacement rule

New research enters WRE as a candidate behind an existing responsibility whenever possible:

```text
new method
  -> adapter/capability declaration
  -> exact model/license/reproducibility metadata
  -> benchmark under the stable WRE contract
  -> quality and resource comparison
  -> optional default/router promotion
```

A better paper should normally change a registry/default, not core product semantics.

## Change control for the frozen blueprint

The freeze does not prohibit correcting a genuine architectural omission. It prevents silent drift.

A proposed change to a frozen product responsibility, canonical representation boundary, provenance rule, temporal model, quality-mode meaning or other system invariant requires all of the following:

1. an explicit blueprint-change rationale explaining why the current frozen contract cannot represent the required product capability;
2. impact analysis across architecture, roadmap, persisted artifacts and completed work;
3. an explicit roadmap/state migration rather than reinterpretation of an already-active work item;
4. migration/compatibility decisions for already-produced v2 artifacts where applicable;
5. updated acceptance evidence and review before the new blueprint becomes authoritative.

Adding or replacing a solver behind an existing contract does not require a blueprint revision.

## Authority

For WRE v2.0 development, authority is ordered as follows:

1. the frozen product/architecture documents and system invariants;
2. the active machine-readable work-item contract and `PROJECT_STATE.yaml`;
3. accepted v2 domain contracts and review evidence;
4. reusable implementation code, including legacy v1 donors;
5. historical v1 architecture and plans.

If legacy v1 structure conflicts with the frozen v2 architecture, **v2 wins**.