# Roadmap — v2 transition

The canonical product/architecture target is defined by `01_PRODUCT.md`, `02_ARCHITECTURE.md`, `03_PIPELINE.md`, `14_TECHNOLOGY_SELECTION.md`, `15_PRODUCTION_RUNTIME.md`, `23_ENGINEERING_EXECUTION.md` and `24_SYSTEM_INVARIANTS.md`.

The executable **v2.0 capability roadmap** is now defined in [`25_V2_ROADMAP.md`](25_V2_ROADMAP.md). It decomposes the product into shallow capability lots and one-run work items. Model/paper names are deliberately not baked into work-item identity; exact candidates are chosen through the technology/model registry and benchmark process.

## Transition state

The pre-v2 machine-readable roadmap remains authoritative **only until the dedicated V2L0.2 state/registry migration merges**. Do not implement the legacy active item `L4.3` merely because `PROJECT_STATE.yaml` still points to it during this short transition.

V2L0 is intentionally staged:

1. `V2L0.1` — inventory/migration map + executable roadmap;
2. `V2L0.2` — replace active machine roadmap/state/components/reviews with v2 identifiers;
3. `V2L0.3` — run compatibility review proving retained code/fixtures/CI remain sound, then hand off to `V2L1.1`.

The explicit V1→V2 component decisions live in `registry/v1-v2-migration.yaml`.

## Migration principles

- Completed code is not discarded because the architecture vocabulary evolved.
- Each implemented v1 component is classified as `retain`, `generalize`, `wrap_compatibly`, `deprecate` or `remove_later`.
- A deletion/rewrite requires replacement evidence and migration coverage.
- Existing passing regression/native fixtures remain evidence baselines.
- Planned-but-unimplemented legacy lots are not obligations; useful ideas are redistributed to v2 capability lots.
- MONDE-specific organization is removed from the product core; optional future integration remains external.

## Why the roadmap contains many lots

This is deliberate. Geometry, surface, appearance, runtime, dynamic 4D, chronology and specialist routes have independent failure/evidence domains. Combining them into giant lots would make agent scope ambiguous and encourage hidden coupling.

A lot should normally contain 3–6 work items. A work item should normally fit one branch/PR and one agent development run. If implementation reveals a second independent responsibility, the roadmap must be split before coding it.

## Historical v1 progression

The pre-v2 roadmap and exact machine state remain recoverable from Git history at commit `415f28d8f2a157d3227b2c73e139150901608df2`. Historical PR/review evidence is also retained in GitHub and `docs/reviews/`.

Implemented v1 foundations mapped forward include:

- repository bootstrap, CI, security/review gates and fixture harness;
- immutable image/video observations, hashing and provenance;
- raw EXIF plus fail-closed GPS/time interpretation;
- local deterministic persistence;
- FFmpeg-based deterministic keyframe baseline;
- exact PyCOLMAP/COLMAP 4.2.0 environment, features, matching, geometric verification, incremental SfM and import;
- real native COLMAP end-to-end fixture;
- sequential and GPS candidate retrieval baselines.

See `registry/v1-v2-migration.yaml` for the precise forward decision for each component.
