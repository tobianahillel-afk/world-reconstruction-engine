# V2L2 — Artifact DAG and cache lot review

## Review status

PASS — 2026-09-16

V2L2 is accepted after the closing integration fixture passed fast CI and CodeQL and the retained native PyCOLMAP lane passed again on the review head.

## Scope reviewed

V2L2 establishes the minimal reusable artifact-DAG/cache substrate without introducing a mutable cache manager or scheduler:

- `V2L2.1` — immutable exact `ArtifactRef` dependency DAG with fail-closed cycle/endpoint validation;
- `V2L2.2` — immutable transactional persistence of direct dependency claims and exact direct lookup;
- `V2L2.3` — exact `ArtifactKey` candidate discovery only, preserving multiple matches and deterministic ordering;
- `V2L2.4` — pure dependency-scoped invalidation planning over changed roots plus downstream dependents;
- `V2L2.5` — immutable local materialization metadata plus read-only size/SHA-256 verification with `VERIFIED` / `MISSING` / `CORRUPTED` outcomes;
- `V2L2.6` — integration regression fixture composing the public V2L2.1–V2L2.5 contracts.

## Composition evidence

The V2L2.6 fixture proves the boundaries that matter for later production orchestration:

1. Exact `ArtifactKey` equality discovers candidates but does not itself establish reuse eligibility.
2. Multiple exact-key candidates remain visible; the fixture does not rank or auto-select a winner.
3. The fixture's explicit reuse precondition additionally requires exact persisted materialization metadata and a current `VERIFIED` local byte check.
4. Absent manifests, missing files and corrupted files do not silently fall back to filename, timestamp, project or other heuristics.
5. A changed canonical computation identity produces a different `ArtifactKey`, so stale candidates under the old key are not discovered.
6. Bytes corrupted after a successful verification are detected on the next check while persisted metadata/expectations remain unchanged.
7. Invalidation of a changed DAG node includes exactly that node and downstream direct/transitive dependents, excluding upstream dependencies and disconnected artifacts.
8. Invalidation planning is non-destructive: persisted records and local files are unchanged.
9. Reopening `SQLiteLocalStore` preserves candidate, materialization and dependency expectations deterministically.

## Fail-closed / non-goals audit

The lot intentionally does **not** add:

- a production cache-manager or cache-hit resolver;
- candidate ranking, recency preference or automatic winner selection;
- eviction, deletion, garbage collection, quarantine or repair;
- recomputation/retry execution, job ordering or scheduling;
- remote/distributed/object-store cache behavior;
- V2L3 adapter/model/shipping registry semantics;
- V2L4 stable failure taxonomy, metrics, benchmark or quality-policy semantics.

No V2L2.6 product implementation was added under `src/`; the closing fixture is test-only and composes existing public contracts.

## Evidence

### Fixture implementation head

Head: `26cec967848bafb82cef169234f4eac5b2eb27ba`

- fast-ci #368 (`35077140554`) — PASS: repository validator, Ruff lint, Ruff format, Pyright, full pytest, actionlint.
- CodeQL #313 (`35077140556`) — PASS.

### Review evidence head

Head: `e47f65caaa254c2e22b07c308fff3f6909919687`

- fast-ci #369 (`35077565143`) — PASS.
- native PyCOLMAP integration #178 (`35077565192`) — PASS in the exact pinned external PyCOLMAP 4.2.0 environment and retained L3/L4 retrieval references.
- CodeQL #314 (`35077565152`) — PASS.

### Retained product baseline

The immediately preceding V2L2.5 final product head `49c2f3312a675cc35bc9dbd32edd02a31e68471b` also passed native PyCOLMAP integration #176 (`35076402356`) before PR #43 merged. V2L2.6 itself adds no product source behavior.

## Decision

**PASS.** V2L2.1 through V2L2.6 compose correctly under the tested invariants. The lot may close, `artifact_dag_cache` may become `implemented`, and V2L3.1 may become the sole next `ready` work item once its full deny-by-default executable contract is present. This review does not authorize V2L3 implementation inside the V2L2 closing PR.
