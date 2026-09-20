# V2L8 — Scene identity and clustering review

**Review status:** PASS  
**Reviewed:** 2026-09-20  
**Lot:** `V2L8 — Scene identity and clustering`

## Review objective

Verify that V2L8.1 through V2L8.5 compose into one auditable fail-closed scene-identity boundary between proposal-only retrieval and later temporal/geometry processing.

The lot must distinguish candidate relationships from accepted scene relationships, adapt retained geometric-verification evidence conservatively, construct deterministic connected scene clusters only from explicit support, preserve unresolved and contradictory evidence without inventing a partition, protect repeated/symmetric look-alike cases from weak-similarity false merges and expose a scale-friendly one-pass batch boundary without importing large-collection infrastructure that belongs to V2L37.

## V2L8 implementation sequence

V2L8 was intentionally split into five bounded work items:

1. PR #84 — `V2L8.1 SceneCluster and scene relation hypothesis contract`: typed pairwise relationship disposition, exact evidence refs and minimal scene-cluster membership representation.
2. PR #85 — `V2L8.2 Local match and geometric verification evidence adapter`: pure adaptation of already-produced retained COLMAP two-view geometry evidence into conservative relationship hypotheses.
3. PR #86 — `V2L8.3 Deterministic verified scene clustering`: support-only connectivity, deterministic cluster identity/evidence and fail-closed contradiction handling.
4. PR #87 — `V2L8.4 Repeated symmetric look alike fail closed fixture`: deterministic synthetic regression proving visual retrieval similarity plus ambiguous geometry does not fuse distinct repeated/symmetric scenes.
5. PR #88 — `V2L8.5 Scalable scene clustering boundary`: immutable relationship batches plus one-pass globally ordered batch consumption with exact semantic equality to V2L8.3.

No V2L8 item performs temporal synchronization, geometry reconstruction, route selection, distributed graph execution or large-collection ANN/submap mapping.

## Composition findings

**PASS.**

- `PairCandidate` remains proposal-only input from V2L7. V2L8 never promotes retrieval similarity directly to same-scene truth.
- `SceneRelationDisposition` keeps `SUPPORTED`, `CONTRADICTED` and `UNRESOLVED` distinct. Missing evidence remains absent rather than becoming an implicit negative.
- `SceneRelationHypothesis` keeps canonical observation endpoints and exact evidence refs without generic confidence, score, rank, GPS, timestamp or route fields.
- `SceneCluster` contains canonical observation membership plus exact support evidence. Singleton clusters are explicit and do not need fabricated edge evidence.
- The retained COLMAP adapter is evidence-only. It receives supplied `PairCandidate` values and an already-produced `ColmapGeometricVerificationResult`; it does not execute feature extraction, raw matching, RANSAC or PyCOLMAP.
- Positive-inlier `CALIBRATED`, `CALIBRATED_RIG`, `UNCALIBRATED`, `PLANAR`, `PANORAMIC` and `PLANAR_OR_PANORAMIC` retained configurations may produce `SUPPORTED`. Absent geometry, zero inliers, `UNDEFINED`, `DEGENERATE`, `WATERMARK`, `MULTIPLE` and unsupported future configurations remain `UNRESOLVED`.
- Ordinary geometric-verification failure never becomes `CONTRADICTED`. Negative scene truth requires separately explicit contradictory evidence.
- V2L8.3 creates connectivity only from `SUPPORTED` relationships. `UNRESOLVED` and `CONTRADICTED` relationships never merge observations or contribute cluster support evidence.
- If explicit contradiction lies inside a component connected by support evidence, clustering raises `SceneClusteringConflictError` rather than dropping evidence or choosing an arbitrary graph cut.
- Scene cluster IDs derive only from canonical membership through the exact SHA-256 rule retained by V2L8.3 and V2L8.5.
- Multi-observation cluster evidence is the exact canonical unique union of support evidence internal to that component.
- The repeated/symmetric fixture models two distinct benchmark-only façade scenes. The cross-scene look-alike pair carries both classical and learned retrieval proposal evidence but retained `DEGENERATE` geometry with positive activity, and therefore remains `UNRESOLVED`.
- That fixture produces exactly two expected benchmark groups, zero false merges and no unresolved look-alike evidence in cluster support evidence.
- A separate fixture control with an explicit false supported bridge and a distinct contradiction raises `SceneClusteringConflictError`, proving contradiction is not silently overwritten by transitive support.
- Benchmark-only physical-scene grouping stays fixture-local and is never passed to production relationship/clustering code as truth or provenance.
- V2L8.5 introduces `SceneRelationBatch` only as an execution boundary. Each batch is immutable and canonical; the full stream is strictly canonical across batch boundaries.
- The batch path consumes an `Iterable` once, does not require rewinding/indexing and retains working relationship state only for union-find/component membership, contradiction endpoint pairs, unique support evidence and the previous pair key.
- One-batch, one-edge-per-batch and mixed valid partitions produce exactly the same `SceneCluster` values, IDs and evidence as the monolithic V2L8.3 path.
- The monolithic path now delegates to the same semantic core, preventing a separate clustering truth definition.
- Cross-batch duplicates, decreasing pair order and foreign endpoints fail closed.
- No filesystem/database spill, sharding, distributed union-find, ANN/HNSW/IVF/PQ, submap reconstruction or resource scheduler is introduced. Those remain V2L37/V2L46 responsibilities.
- No new third-party dependency, model, checkpoint, network path or subprocess is introduced by V2L8.3 through V2L8.5.

## Cross-cutting invariant audit

**PASS.**

- Observation/media truth remains immutable.
- Retrieval remains a work proposal rather than truth.
- Geometric ambiguity remains `UNRESOLVED`.
- Explicit contradiction remains first-class and cannot be silently overridden by transitive support.
- Repeated/symmetric visual similarity alone cannot fuse physical scenes.
- Evidence remains exact and typed; no generic metadata bag is added to scene identity contracts.
- Solver-private COLMAP match counts, matrices, paths and environment details remain outside the stable scene-relation API.
- Cluster identity is deterministic and independent of batch partitioning.
- Scaling work changes execution shape, not epistemic standards.
- No quality mode, route, geometry product, historical epoch or generated content is inferred by scene identity.
- Large-collection ANN and submap processing remain deferred to V2L37 rather than leaking into the V2L8 contract.

## Review findings resolved during the lot

**PASS after correction.**

- PR #84 required no semantic correction after the domain contract implementation; its lifecycle handoff independently passed all triggered lanes.
- PR #85 required one Ruff line-length correction, formatting-only normalization and one quoted YAML scalar containing `scene:`; no adapter semantics changed.
- PR #86 required formatting-only adjustments in the new clustering tests and one quoted YAML lifecycle scalar; the support/contradiction semantics remained unchanged.
- PR #87 required one Ruff B009 cleanup and formatting-only normalization in the synthetic fixture test; no fixture expectation or production truth contract changed.
- PR #88 initial implementation required only Ruff formatting in the clustering module and focused test before validator, Ruff, Pyright and full pytest all passed.
- Automated code-review quota was not relied on for lot closure; final diffs were reviewed manually against the executable deny-by-default work-item contracts.
- No unresolved PR review thread or known semantic regression remains at the V2L8 review boundary.

## Evidence

Exact retained work-item evidence:

- V2L8.1 / PR #84 / final head `1c47dfec1716dcb59a4eaf5380c235edc40bf2f5`: fast-ci #650, native PyCOLMAP #344, CodeQL #588 — **PASS**. Domain-only implementation head `35ca73f83baeca78537382e4a6a49dc10a6c807a` also passed fast-ci #649 and CodeQL #587.
- V2L8.2 / PR #85 / final head `c01b5af8c88b0079dba55cfee32a9d84012d4c1b`: fast-ci #656, native PyCOLMAP #347, CodeQL #594 — **PASS**.
- V2L8.3 / PR #86 / final head `e7fa26cb35c917803cf99e2e17103895f9894428`: fast-ci #662, native PyCOLMAP #350, CodeQL #600 — **PASS**. Functional implementation head `ddcb558e3ad3d40efa77157d1d523fcb05d74ae2` passed fast-ci #660 and CodeQL #598.
- V2L8.4 / PR #87 / final head `c8fdd24bdb739361e4426d9475c4838c36c0d179`: fast-ci #668, native PyCOLMAP #352, CodeQL #606 — **PASS**. Functional fixture head `b238e62a5be1d0ba95e6f87bb651b72bd7d3cb59` passed fast-ci #667 and CodeQL #605.
- V2L8.5 / PR #88 / reviewed implementation head `309329bb8084e7d795e87448028a5b3485a52e93`: fast-ci #671 and CodeQL #609 — **PASS**.

The final PR #88 lifecycle-handoff head changes review/state/registry metadata after the already-green reviewed implementation. It must independently pass every triggered exact-head lane before merge.

## V2L9.1 handoff decision

After this PASS, `V2L9.1 — TemporalGroup and SyncHypothesis contracts` may become the sole `ready` item.

V2L9.1 is intentionally representation-only. It defines a synchronization-context group tied to an existing `SceneClusterId` and a canonical pairwise relative-time hypothesis with explicit `SUPPORTED`, `CONTRADICTED` or `UNRESOLVED` state, exact evidence refs and a normalized microsecond offset only when supported.

The offset convention is relative event-time alignment between canonical endpoints, not an absolute capture timestamp, historical epoch or publication date. V2L9.1 does not parse EXIF/timecode, execute audio correlation, infer visual event alignment, build groups from evidence, fuse clocks into one timeline or touch long-term historical chronology. Those remain later V2L9/V2L31 responsibilities.

## Decision

**V2L8 lot review: PASS.**

Scene identity now composes proposal separation, conservative verification evidence, deterministic fail-closed clustering, repeated/symmetric false-merge protection and a one-pass scale-friendly execution boundary without weakening truth semantics or importing later large-collection infrastructure. V2L9 may begin with temporal representation contracts only after PR #88 final exact-head gates pass.
