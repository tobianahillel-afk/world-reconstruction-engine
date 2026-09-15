# V2L0 V2 migration and retained-baseline compatibility review

Date: 2026-09-15
Scope: lot `V2L0` — V2 migration and compatibility
Legacy implementation baseline: `415f28d8f2a157d3227b2c73e139150901608df2`
V2 machine-migration merge reviewed: `6d5e61525c829b6fa945ba16b21fd133e13c5b19`
Current review state: **IN REVIEW**

## Review objective

Verify that the WRE V2 blueprint and machine-governance migration change the product architecture and development control plane without silently deleting, weakening or behaviorally rewriting the retained V1 implementation. V2L0 closes only when retained ingestion, persistence, COLMAP reconstruction and candidate-retrieval baselines remain available and tested, the V1-to-V2 mapping is traceable, V2 governance is internally consistent, and the first product implementation item is handed off through a complete deny-by-default contract.

This review does not approve any new reconstruction algorithm or product-domain implementation.

## Reviewed migration sequence

The migration is intentionally split into reviewable steps:

1. PR #29 merged the V2 product/architecture blueprint and engineering rules without replacing product implementation.
2. PR #30 (`V2L0.1`) added the executable V2 roadmap and explicit V1-to-V2 migration inventory.
3. PR #31 (`V2L0.2`) migrated the machine-readable roadmap/state/component/review registries and strengthened the repository validator.
4. `V2L0.3` is this compatibility review and the only place permitted to close V2L0 and prepare the first V2 product work item.

## Retained product-source audit

**PASS for the migration diff.** Comparing legacy implementation baseline `415f28d8f2a157d3227b2c73e139150901608df2` with V2 machine-migration `main` at `6d5e61525c829b6fa945ba16b21fd133e13c5b19` shows no changes to the retained product modules under:

- `src/wre/ingestion/`;
- `src/wre/persistence/`;
- `src/wre/reconstruction/`;
- `src/wre/retrieval/`;
- retained domain modules such as observations, cameras, reconstruction runs and estimated geometry.

The only changed file under `src/wre/` during V2L0.1/V2L0.2 is `src/wre/repo_validation.py`, whose responsibility is repository governance rather than reconstruction behavior. Its focused tests are correspondingly updated in `tests/test_repo_validation.py`.

No retained product source module was deleted by the migration. The migration therefore changes development governance and target architecture without pretending that already-working V1 code ceased to exist.

## Retained capability mapping audit

**PASS.** `registry/v1-v2-migration.yaml` explicitly classifies implemented V1 capabilities instead of deleting or silently reinterpreting them. The retained baseline includes:

- repository bootstrap, CI, supply-chain checks, deterministic fixtures and review discipline;
- typed observations, media/source identity, hashing and provenance;
- camera/media metadata and EXIF/GPS/time interpretation;
- image and video ingestion plus deterministic FFmpeg keyframe foundations;
- the exact COLMAP/PyCOLMAP environment adapter;
- feature extraction, raw matching, geometric verification, sparse reconstruction and solver-independent import;
- the deterministic native COLMAP end-to-end fixture;
- sequential and GPS pair-candidate retrieval;
- persistence and existing regression infrastructure.

Where V2 needs broader contracts, the migration map uses retain-and-generalize / compatible-wrapper decisions rather than a rewrite-by-default policy.

## Historical review evidence

**PASS.** The V1 reviews remain present and referenced by `registry/reviews.yaml`:

- `docs/reviews/L0-M0-bootstrap-review.md`;
- `docs/reviews/L1-core-domain-review.md`;
- `docs/reviews/L2-media-ingestion-review.md`;
- `docs/reviews/L3-M1-colmap-baseline-review.md`.

The V2 migration does not relabel those historical reviews as V2 completion. They remain evidence for retained implementation behavior and provenance boundaries.

## Governance and deny-by-default audit

**PASS.** V2L0.2 establishes machine-enforced V2 governance with the following relevant properties:

- one active work item only;
- at most six work items per lot;
- planned future items may remain concise;
- an executable item must define objective, allowed scope, out-of-scope, dependencies, required reading, reuse, acceptance and tests;
- work-item scope is deny-by-default;
- active dependencies must be complete;
- an item marked `done` cannot depend on unfinished work;
- lot/milestone review gates remain blocking;
- component and review registries cover the V2 roadmap;
- V1 completion history is separated from V2 `last_completed` state.

This allows the research/model shortlist to evolve without making vague implementation work executable.

## Retained CI and native integration evidence

**PASS for the V2L0.2 implementation head.** Exact PR #31 head `6ccdd3f2a9a92f7645da6130f4e096436956c49a` passed all required checks before squash merge:

- fast-ci run `35021407532` / run #282: **PASS**;
  - repository metadata validator;
  - Ruff lint;
  - Ruff format;
  - Pyright;
  - full pytest suite;
  - actionlint;
- native COLMAP integration run `35021406737` / run #120: **PASS**;
  - exact external `pycolmap==4.2.0` environment;
  - retained L3 reconstruction path;
  - retained L4 retrieval references;
- CodeQL run `35021406660` / run #233: **PASS**.

The squash merge changes commit identity, not the reviewed tree contents. Nevertheless this V2L0.3 PR must itself pass its exact-head fast and native integration lanes before the lot review can be changed from `in_review` to `passed`.

## Compatibility conclusions

The evidence currently supports the following conclusions:

1. **No restart-from-zero is justified.** Retained V1 implementation remains the compatibility baseline for V2.
2. **No legacy roadmap continuation is implied.** Retained code is reused under new V2 contracts; obsolete MONDE/world-graph sequencing stays historical.
3. **No learned/classical ideology is encoded into the core.** V2 may approve either through adapters and benchmarks while retaining the classical COLMAP baseline.
4. **No appearance representation is confused with physical geometry.** Future splat/radiance work is first-class appearance, while physical surface remains separately owned.
5. **No future item becomes executable merely because it appears in the roadmap.** Activation requires a complete contract and satisfied dependencies/reviews.

## V2L1.1 handoff requirements

Before this review may close, `V2L1.1 — SceneProject contract` must be expanded from a concise planned entry into a complete executable contract. The contract must remain deliberately minimal:

- establish only the stable project-root identity/boundary required for later project metadata;
- reuse existing WRE ID and validation conventions where appropriate;
- avoid inventing placeholder artifact, cluster, temporal, runtime or master-scene types owned by later work items;
- avoid persistence, artifact DAG/cache, quality routing, solver selection, runtime compilation and media-processing behavior;
- require explicit positive and negative unit tests;
- keep all unlisted behavior out of scope.

No `SceneProject` source code is permitted in the V2L0.3 PR.

## Remaining checks before PASS

- exact-head fast CI on the V2L0.3 handoff branch;
- exact-head native PyCOLMAP integration on the V2L0.3 handoff branch;
- CodeQL for the V2L0.3 handoff branch;
- expansion of `V2L1.1` into a complete executable contract;
- atomic final state update: V2L0.3 `done`, V2L0 review `passed`, migration-governance component `implemented`, and V2L1.1 the sole `ready` item.

## Decision

**V2L0 lot review: IN REVIEW.** The migration diff, historical evidence, retained source tree and PR #31 validation support compatibility. Final PASS is intentionally withheld until the V2L0.3 exact handoff head passes its own required checks and the next work item is fully specified without implementation leakage.
