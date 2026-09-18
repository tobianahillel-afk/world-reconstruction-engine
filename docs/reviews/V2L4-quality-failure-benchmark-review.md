# V2L4 / V2M0 — Quality, failure, benchmark and production-substrate review

**Review status:** PASS  
**Reviewed:** 2026-09-18  
**Lot:** `V2L4 — Quality failure and benchmark records`  
**Milestone:** `V2M0 — Production substrate ready`

## Review objective

Verify two composition boundaries together without turning the review into a status-count exercise.

First, V2L4 must provide one coherent solver-independent language for stable failures, typed multidimensional measurements, explicit quality decisions, reproducible benchmark evidence and deterministic fail-closed policy evaluation. Unknown, missing, contradictory or unmapped input must never silently become success, and quality decisions must remain declarations rather than hidden routing actions.

Second, because V2L4 is the final lot of V2M0, the milestone review must verify the declared end-to-end milestone capability: WRE has a production substrate that can identify projects and artifacts, retain exact producer/provenance identity, represent artifact dependencies and cache/materialization state, register approved adapters/dependencies, record reproducible benchmark evidence and evaluate quality/failure outcomes deterministically. V2M0 does **not** claim arbitrary-media understanding, route selection or scene reconstruction; those begin in later milestones.

## V2L4 implementation sequence

V2L4 was intentionally split into six bounded work items:

1. PR #59 — `V2L4.1 Stable failure taxonomy`: exact closed cross-adapter `FailureCategory` vocabulary.
2. PR #60 — `V2L4.2 Multidimensional metric vector`: typed metric semantics, finite scalar observations and exact evaluator/input provenance.
3. PR #61 — `V2L4.3 Quality decision contract`: exact `PASS / ACCEPT_WITH_WARNINGS / RETRY / ESCALATE / UNRESOLVED` vocabulary.
4. PR #62 — `V2L4.4 Benchmark record schema`: immutable fixture/producer/config/hardware/quality-mode/metric evidence, plus the single canonical `QualityMode` vocabulary.
5. PR #63 — `V2L4.5 Deterministic quality policy evaluator`: immutable caller-selected policy and pure fail-closed evaluation.
6. PR #64 — `V2L4.6 Unknown metric and failure negative tests`: cross-contract negative/precedence regressions and this composition review.

No item was allowed to execute routing, retry, fallback, scheduling or default promotion.

## V2L4 composition findings

**PASS.**

- `FailureCategory` is stable cross-adapter meaning. Adapter-local raw `failure_signals` remain separate and are not parsed or auto-mapped by V2L4.
- `MetricVector` is typed measurement evidence. It preserves exact finite supplied values and evaluator/input provenance without normalization, weighting or one-score collapse.
- `QualityDecision` is a decision vocabulary only. `RETRY` and `ESCALATE` do not execute an action.
- `QualityMode` is a product/compute mode and remains distinct from provenance, failure and decision vocabularies.
- `BenchmarkRecord` is descriptive reproducibility evidence only. It carries no threshold, policy, route, rank, winner or promoted default.
- `QualityPolicy` is explicit, immutable and versioned. It admits exact metric names and failure mappings rather than discovering mutable global policy.
- Missing required metrics, unknown input metrics, direction contradictions and unmapped failures return `UNRESOLVED` instead of being ignored.
- V2L4.6 exhaustively verifies every current `FailureCategory` as unmapped under an empty policy, exercises multiple valid-but-unknown `MetricName` values and proves unresolved evidence dominates simultaneous mapped `RETRY` / `ESCALATE` conditions.
- Optional absent metrics and explicitly admitted informational metrics remain valid; the negative suite does not manufacture failure for otherwise valid input.
- Exported audit reasons reject impossible states: a direction-mismatch reason cannot carry equal directions and a threshold-violation reason cannot carry a non-violating value.
- The lot introduces no metric computation, raw failure parser, benchmark runner, profiler, routing graph, scheduler, persistence rewrite or external dependency.

## V2M0 end-to-end substrate findings

**PASS.** The five V2M0 lots compose into the milestone capability `Production substrate ready`:

- **V2L0 — governance/migration:** V2 authority, deny-by-default executable work items, retained V1 donor mapping and review gates are machine validated.
- **V2L1 — project/artifact identity:** distinct project/artifact identities, exact producer/model/checkpoint/config identity, provenance class, canonical artifact computation identity and local metadata persistence exist without speculative future manifest fields.
- **V2L2 — artifact DAG/cache:** immutable dependency relationships, exact dependency persistence, canonical-key lookup, dependency invalidation and materialization verification provide an explicit reusable artifact lifecycle without a hidden mutable cache manager.
- **V2L3 — adapter/model/dependency registry:** capabilities, hardware/runtime identity, shipping/license/dependency metadata and retained baseline registrations are explicit and fail closed.
- **V2L4 — quality/failure/benchmark:** failures, measurements, decisions, benchmark evidence and deterministic policy evaluation are stable cross-adapter contracts.

Together these lots support the substrate-level lifecycle:

```text
SceneProject / Observation evidence
        -> exact Artifact identity + producer/provenance identity
        -> immutable dependency DAG / cache lookup / materialization verification
        -> registered adapter + dependency/hardware identity
        -> typed failure + metric evidence
        -> reproducible BenchmarkRecord
        -> explicit deterministic QualityDecision
```

Each arrow is represented by explicit typed contracts or retained reviewed infrastructure; none requires filenames, mutable `latest` aliases, hidden global state or log-string inference.

This milestone intentionally stops before `MediaProfile`, retrieval, clustering, temporal grouping and routing. Therefore passing V2M0 does not claim that WRE already understands arbitrary media or reconstructs a scene. It proves the production substrate needed to implement those capabilities without later redesigning identity, artifact lifecycle, adapter registration or quality semantics.

## Cross-cutting invariant audit

**PASS.**

- Raw/derived evidence separation and exact provenance remain intact.
- Logical artifact identity, content/computation identity and materialization status remain distinct.
- Unknown or unsupported required conditions fail closed.
- Routing still chooses work rather than truth; no route graph exists in V2M0 quality code.
- Vendor/profiler execution technology remains outside stable contracts.
- Existing retained ingestion, persistence, retrieval and native COLMAP donor behavior is not rewritten by V2L4.
- The performance integration plan remains correctly deferred: V2L4.4 can reference performance evidence, while actual profilers/execution profiles remain owned by later representative workloads.

## Evidence

Exact merged V2L4 work-item heads:

- V2L4.1 / PR #59 / `7217b425465cbb02a63bc1b8ca301e51751eb88a`: fast-ci #454, native COLMAP #221, CodeQL #392 — **PASS**.
- V2L4.2 / PR #60 / `8934fc16eb954f2a4ffaa59360762c4bcf21c1b5`: fast-ci #462, native COLMAP #224, CodeQL #400 — **PASS**.
- V2L4.3 / PR #61 / `4a0fcfe0efdebc6fbd1fe0357126db2ab8bedc9e`: fast-ci #468, native COLMAP #227, CodeQL #406 — **PASS**.
- V2L4.4 / PR #62 / `fe6083342a501495da3d1b322f39e877bfd50702`: fast-ci #482, native COLMAP #232, CodeQL #420 — **PASS**.
- V2L4.5 / PR #63 / `2c4baa7bc57f99b9a65d276cf2f06708b137be92`: fast-ci #495, native COLMAP #238, CodeQL #433 — **PASS**. Two P2 findings on impossible audit-reason states were fixed with focused negative regressions before merge and both review threads were resolved.
- V2L4.6 negative-regression implementation / PR #64 / `0008e5cc9255b3d679dacabc21de4090f0074aec`: fast-ci #498 and CodeQL #436 — **PASS**.

V2M0 also inherits the accepted lot-composition evidence from:

- `docs/reviews/V2L0-v2-migration-compatibility-review.md`;
- `docs/reviews/V2L1-scene-project-artifact-identity-review.md`;
- `docs/reviews/V2L2-artifact-dag-cache-review.md`;
- `docs/reviews/V2L3-adapter-model-registry-review.md`;
- this V2L4 review.

The final PR #64 handoff head changes review/state/registry metadata after the already-green negative-regression implementation. It must pass fast CI, the triggered native PyCOLMAP/COLMAP integration lane and CodeQL on that exact final head before merge. Those exact final run identifiers belong in PR #64 merge evidence rather than in a self-referential review commit.

## V2L5.1 handoff decision

After this PASS, `V2L5.1 — MediaProfile contract` may become the sole `ready` item with a complete deny-by-default contract.

Its first responsibility is deliberately minimal: establish one immutable profile boundary for one existing `ObservationId` and exact canonical `ArtifactRef` evidence links. V2L5.1 does not compute blur/exposure, duplicate/diversity, camera/input-class, dynamic/long-sequence/sparse-coverage signals, perform frame selection, decode media, route work or introduce a generic signal dictionary.

The existing immutable observation/media-ingestion donor contracts remain the source evidence boundary; later V2L5 items add specific evidence contracts rather than stuffing speculative fields into V2L5.1.

## Decision

**V2L4 lot review: PASS.**

**V2M0 milestone review: PASS.** The declared milestone capability `Production substrate ready` is achieved as an end-to-end substrate composition, while arbitrary-media understanding and organization remain explicitly unclaimed future work in V2M1.

PR #64 must not merge unless its final handoff head is green on all triggered required lanes and has no unresolved review finding. No `MediaProfile` product implementation belongs in PR #64.
