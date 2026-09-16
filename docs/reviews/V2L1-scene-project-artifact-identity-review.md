# V2L1 SceneProject and artifact identity review

Date: 2026-09-16
Scope: lot `V2L1` — SceneProject and artifact identity
Review PR: #38
Verdict: **PASS**

## Review objective

Verify that V2L1 composes one minimal, solver-independent project/artifact identity substrate before WRE starts dependency-graph and cache work. The lot closes only if project identity, exact artifact references, producer/model/checkpoint/configuration identity, epistemic provenance class, deterministic computation identity and local metadata persistence compose without hidden mutable state, speculative manifest fields, dependency-graph behavior, cache semantics or regressions to the retained V1 baseline.

This review does not approve an artifact DAG, cache lookup, invalidation, materialization, adapter registry, scheduler or reconstruction algorithm.

## Reviewed implementation sequence

V2L1 was intentionally split into six independently reviewable work items:

1. PR #33 — `V2L1.1 SceneProject contract`: distinct immutable `SceneProjectId` and minimal `SceneProject` root only.
2. PR #34 — `V2L1.2 Artifact identity and exact input references`: `ArtifactId`, open validated `ArtifactKind`, and exact immutable `ArtifactRef`.
3. PR #35 — `V2L1.3 Producer model checkpoint configuration identity`: reuse retained `ProducerRef`; add typed model/checkpoint/configuration identities and `ArtifactProducerIdentity`.
4. PR #36 — `V2L1.4 Provenance class contract`: exactly `OBSERVED_RECONSTRUCTED`, `INFERRED`, and `GENERATED`, independent from retained run/source-observation lineage.
5. PR #37 — `V2L1.5 Canonical artifact key`: versioned deterministic computation identity from explicit output kind, ordered input-content fingerprints and producer identity.
6. PR #38 — `V2L1.6 Project and artifact metadata persistence`: minimal immutable `ArtifactMetadata` plus codec/SQLite round-trip by extending the retained persistence boundary.

No item was allowed to implement the next item's responsibility in advance.

## Contract composition audit

**PASS.** The six contracts now compose into a coherent artifact identity substrate:

- `SceneProjectId` establishes project identity without pretending the full future project manifest already exists.
- `ArtifactRef` identifies one exact artifact through typed `ArtifactId` plus declared `ArtifactKind`; filenames, paths and timestamps are not identity fallbacks.
- `ArtifactProducerIdentity` records exact producer implementation/version/revision, normalized-configuration digest and optional exact model/checkpoint identity while reusing retained `ProducerRef`.
- `ProvenanceClass` remains an epistemic label only; retained `DerivedArtifactProvenance` still owns run/source-observation lineage.
- `ArtifactKey` is a distinct, versioned SHA-256 computation identity derived from explicit typed key material. Logical project/artifact IDs, provenance class, paths, timestamps and mutable environment state do not contaminate the V1 key preimage.
- `ArtifactMetadata` joins project, artifact ref, artifact key, producer identity and provenance class as a minimal immutable persisted record without dependency edges or materialization fields.

The result is sufficient for later artifact lifecycle work without creating a generic metadata bag or prematurely encoding later capabilities.

## Identity and provenance invariant audit

**PASS.** V2L1 preserves the blueprint's separation of identity domains and provenance responsibilities:

- project, artifact, observation, source, camera and run identities remain distinct typed domains;
- byte/content identity and logical artifact identity remain distinct;
- computation identity and logical artifact identity remain distinct;
- provenance class is not inferred from solver, path, filename, artifact kind or route;
- checkpoint/model/configuration identities are explicit and fail closed;
- floating model version `latest` is rejected by the V2L1.3 contract;
- checkpoint-without-model is rejected;
- generated/inferred/reconstructed classes are not collapsed;
- existing V1 `ReconstructionRun` / `DerivedArtifactProvenance` semantics are retained rather than rewritten.

## Artifact-key audit

**PASS.** V2L1.5 locks a documented V1 canonical serialization and golden SHA-256 vector. The key material contains exactly the computation dimensions currently owned by V2L1: output kind, ordered input content fingerprints, producer implementation/version/revision, normalized-configuration digest and optional model/checkpoint identity. Input order and duplicates are preserved deliberately; the key layer performs no hidden sorting/deduplication.

The implementation performs no filesystem, registry, database, network, cache or environment lookup during key derivation. Hardware-sensitive identity is deliberately deferred to V2L3.2 rather than represented by a misleading placeholder.

## Persistence composition audit

**PASS.** PR #38 extends the retained V1 persistence architecture instead of replacing it:

- the same `SQLiteLocalStore` is retained;
- the same generic `wre_records` table, transaction semantics, canonical JSON and immutable conflict behavior are reused;
- `SceneProject` uses a dedicated `scene_project` record type keyed by `SceneProjectId`;
- `ArtifactMetadata` uses a dedicated `artifact_metadata` record type keyed by `ArtifactId`;
- the existing database and record schema versions are retained because the generic record table already supports the new types;
- identical writes remain idempotent and conflicting immutable identity reuse fails closed;
- artifact metadata storage neither requires nor creates a project record implicitly;
- V1 observation records and V2 project/artifact records coexist in the same database;
- codec decoding reconstructs the stored `ArtifactKey` exactly and does not regenerate it from partial metadata.

Focused tests cover reopen, missing records, idempotency, conflict rejection, malformed payloads, unknown provenance, native producer null model/checkpoint, no implicit project creation and V1/V2 coexistence.

## Negative-scope audit

**PASS.** The lot deliberately does **not** contain:

- artifact dependency edges or a dependency DAG;
- cycle detection or graph traversal;
- persisted dependency lookup or reverse indexes;
- cache lookup, reuse or invalidation;
- key-to-artifact resolution;
- artifact materialization path/URI/byte verification/corruption state;
- project-existence repair or cross-record mutation;
- full SceneProject manifest fields such as source collections, coordinate frames, quality modes, clusters, temporal groups, master versions or runtime exports;
- adapter/model shipping registries, hardware scheduling, quality policy or runtime behavior;
- any new reconstruction solver or replacement of retained V1 reconstruction code.

Those responsibilities remain assigned to later work items.

## CI and regression evidence

**PASS.** Every completed V2L1 item was validated on an exact handoff head before merge. The final heads recorded by their PRs include:

- V2L1.1 / PR #33: fast-ci #297, native PyCOLMAP #133, CodeQL #242 — PASS.
- V2L1.2 / PR #34: fast-ci #303, native PyCOLMAP #137, CodeQL #248 — PASS.
- V2L1.3 / PR #35: fast-ci #311, native PyCOLMAP #141, CodeQL #256 — PASS.
- V2L1.4 / PR #36: fast-ci #316, native PyCOLMAP #145, CodeQL #261 — PASS.
- V2L1.5 / PR #37: fast-ci #322, native PyCOLMAP #149, CodeQL #267 — PASS.

For V2L1.6, implementation head `0999063420f00eeb2c60c7b3f5e3cb9c9cf80f17` passes:

- fast-ci #325 / `35035440861`: **PASS** — repository validator, Ruff lint/format, Pyright, full pytest and actionlint;
- CodeQL #270 / `35035440829`: **PASS**.

The final review/state handoff commit in PR #38 changes governance metadata and review evidence in addition to the already-green implementation. It must itself pass the triggered final-head fast CI, native PyCOLMAP integration and CodeQL lanes before merge. Those PR checks are the merge-gate evidence for the exact final metadata state and avoid a self-referential evidence commit.

## Reuse and compatibility conclusion

**PASS.** V2L1 follows the compatibility-first rule. It introduces new V2 domain vocabulary only where a new identity responsibility actually exists, reuses retained `ProducerRef`, `Sha256Digest`, canonical JSON and `SQLiteLocalStore`, and leaves retained observation/ingestion/reconstruction/retrieval semantics in place. No restart-from-zero or duplicate persistence backend is justified by this lot.

## V2L2.1 handoff decision

`V2L2.1 — Artifact dependency DAG contract` may become the sole `ready` item only with a complete deny-by-default contract.

Its authorized responsibility is limited to an immutable in-memory directed dependency graph over exact `ArtifactRef` values:

- define an immutable direct edge meaning “artifact depends on dependency”;
- require every edge endpoint to be a declared graph node;
- reject self-dependencies;
- reject directed cycles fail closed;
- reject one logical `ArtifactId` appearing with conflicting `ArtifactKind` declarations inside one graph;
- preserve disconnected acyclic components and direct-edge semantics without inventing persistence/cache behavior;
- use explicit immutable collection semantics with focused positive/negative tests.

V2L2.1 must not persist dependencies, perform database dependency lookup, resolve cache entries, plan invalidation, materialize artifacts, mutate metadata, infer dependencies from `ArtifactKeyMaterial`, inspect files/paths, schedule jobs, or implement V2L2.2+ behavior.

## Decision

**V2L1 lot review: PASS.** V2L1 establishes the minimal immutable project/artifact identity and local metadata persistence substrate required by the production blueprint while preserving V1 compatibility and leaving dependency/cache lifecycle responsibilities to V2L2.

The candidate final handoff for PR #38 may mark V2L1.6 `done`, V2L1 review `passed`, `scene_project_artifacts` `implemented`, and V2L2.1 as the sole `ready` item. PR #38 must not merge unless the final handoff head itself is green on all triggered required lanes. No V2L2 product implementation belongs in PR #38.
