# V2L0 V2 migration and retained-baseline compatibility review

Date: 2026-09-15
Scope: lot `V2L0` — V2 migration and compatibility
Legacy implementation baseline: `415f28d8f2a157d3227b2c73e139150901608df2`
V2 machine-migration merge reviewed: `6d5e61525c829b6fa945ba16b21fd133e13c5b19`
Review PR: #32
Verdict: **PASS**

## Review objective

Verify that the WRE V2 blueprint and machine-governance migration change the product architecture and development control plane without silently deleting, weakening or behaviorally rewriting the retained V1 implementation. V2L0 closes only when retained ingestion, persistence, COLMAP reconstruction and candidate-retrieval baselines remain available and tested, the V1-to-V2 mapping is traceable, V2 governance is internally consistent, and the first product implementation item is handed off through a complete deny-by-default contract.

This review does not approve any new reconstruction algorithm or product-domain implementation.

## Reviewed migration sequence

The migration is intentionally split into reviewable steps:

1. PR #29 merged the V2 product/architecture blueprint and engineering rules without replacing product implementation.
2. PR #30 (`V2L0.1`) added the executable V2 roadmap and explicit V1-to-V2 migration inventory.
3. PR #31 (`V2L0.2`) migrated the machine-readable roadmap/state/component/review registries and strengthened the repository validator.
4. PR #32 (`V2L0.3`) audits retained compatibility, closes V2L0, and prepares only the contract for the first V2 product work item.

## Retained product-source audit

**PASS.** Comparing legacy implementation baseline `415f28d8f2a157d3227b2c73e139150901608df2` with V2 machine-migration `main` at `6d5e61525c829b6fa945ba16b21fd133e13c5b19` shows no changes to the retained product modules under:

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

**PASS.** Exact PR #31 head `6ccdd3f2a9a92f7645da6130f4e096436956c49a` passed the checks relevant to the Python governance migration before squash merge:

- fast-ci run `35021407532` / run #282: **PASS**;
- native COLMAP integration run `35021406737` / run #120: **PASS** under exact external `pycolmap==4.2.0`;
- CodeQL run `35021406660` / run #233: **PASS**.

PR #32 then revalidated compatibility after the migration was on `main`. Exact initial review head `463e0c317298e225d321767444d8e9a34375a4bb` passed:

- fast-ci run `35023566802` / run #284: **PASS**;
- native COLMAP integration run `35023566747` / run #122: **PASS**.

CodeQL is **not applicable to the docs/YAML-only V2L0.3 changes** because `.github/workflows/codeql.yml` explicitly triggers pull-request analysis only when Python source/scripts/tests, `pyproject.toml`, `uv.lock`, or the CodeQL workflow itself changes. The last V2 migration change touching Python was V2L0.2, and its CodeQL run #233 passed.

The final V2L0.3 handoff commit changes only review/state/registry metadata. PR #32 must still have its final-head fast and native integration checks green before merge; the PR checks are the merge-gate evidence for that final metadata state and do not require another self-referential evidence commit.

## Compatibility conclusions

The evidence supports the following conclusions:

1. **No restart-from-zero is justified.** Retained V1 implementation remains the compatibility baseline for V2.
2. **No legacy roadmap continuation is implied.** Retained code is reused under new V2 contracts; obsolete MONDE/world-graph sequencing stays historical.
3. **No learned/classical ideology is encoded into the core.** V2 may approve either through adapters and benchmarks while retaining the classical COLMAP baseline.
4. **No appearance representation is confused with physical geometry.** Future splat/radiance work is first-class appearance, while physical surface remains separately owned.
5. **No future item becomes executable merely because it appears in the roadmap.** Activation requires a complete contract and satisfied dependencies/reviews.

## V2L1.1 handoff decision

`V2L1.1 — SceneProject contract` is permitted to become the sole `ready` item only with a complete contract. Its authorized responsibility is deliberately minimal:

- define a distinct `SceneProjectId` identity domain;
- define an immutable `SceneProject` root carrying only that project identity in this work item;
- reuse the lexical/validation semantics already established for WRE opaque IDs without aliasing project identity to observation/source identity;
- require caller-supplied deterministic identity rather than hidden/random ID generation;
- require positive and negative unit tests;
- keep every unlisted manifest responsibility out of scope.

The following are explicitly not part of V2L1.1 and must not be represented by generic placeholder dictionaries/strings merely to make the object look complete: artifacts and dependency references, producer/model/checkpoint identity, provenance classes, artifact keys, persistence, source-media collections, coordinate frames, quality modes, user overrides, scene clusters, temporal groups, master-scene versions, runtime exports, warnings, routing, solver behavior or media processing.

Those responsibilities remain owned by later work items. This keeps the first product PR small enough to implement and review in one run.

## Decision

**V2L0 lot review: PASS.** The migration preserves the retained product baseline, preserves historical evidence, enforces V2 deny-by-default development governance, and has independent fast/native integration evidence after the V2 machine migration landed on `main`.

The candidate final handoff for PR #32 may mark V2L0.3 `done`, V2L0 review `passed`, `v2_migration_governance` `implemented`, and V2L1.1 as the sole `ready` item. PR #32 must not merge unless the final handoff head itself has green fast CI and native PyCOLMAP integration checks. No `SceneProject` implementation belongs in PR #32.
