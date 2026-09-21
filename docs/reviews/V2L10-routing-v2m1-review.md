# V2L10 / V2M1 — Routing foundation and arbitrary-media organization review

**Review status:** PASS  
**Reviewed:** 2026-09-21  
**Lot:** `V2L10 — Route graph, quality modes and bounded escalation`  
**Milestone:** `V2M1 — Arbitrary media understood and organized`

## Review objective

Verify that V2L10.1 through V2L10.6 compose into a deterministic, fail-closed and auditable routing foundation without turning routing into reconstruction truth, hidden solver selection or execution.

The same review performs the V2M1 milestone composition audit across accepted V2L5 through V2L10 evidence. V2M1 must demonstrate that arbitrary media can be profiled, selectively reused, proposed for pairwise retrieval, organized into scene identity and event-time evidence, and represented for auditable routing while preserving provenance and uncertainty. It must not claim camera/depth/geometry reconstruction, physical surfaces, photorealistic appearance, dynamic 4D, historical chronology or runtime compilation.

## V2L10 implementation sequence

V2L10 was intentionally split into six bounded work items:

1. PR #94 — `V2L10.1 Adopt canonical QualityMode in routing contracts`: exact reuse of the canonical V2L4.4 `QualityMode` object plus a one-field immutable route-quality request.
2. PR #95 — `V2L10.2 Route node and route graph contract`: typed node identity, semantic capability requirements and immutable acyclic graph structure without adapter selection or execution.
3. PR #96 — `V2L10.3 Deterministic router input contract`: canonical media-profile, resource-budget, existing-artifact and prior-failure input snapshot.
4. PR #97 — `V2L10.4 Bounded allowlisted fallback escalation policy`: explicit exact-match triggers, hard transition bound and pure one-step resolution.
5. PR #99 — `V2L10.5 Auditable route decision artifact`: caller-supplied decision identity, exact input/graph/policy binding and typed evidence-backed reasons.
6. PR #100 — `V2L10.6 Router determinism fail closed tests`: cross-contract fixture proving equality, stable wire tokens, canonical ordering and fail-closed invalid composition.

No V2L10 item chooses a route from competing candidates, resolves concrete adapters/models, executes route nodes or fallback transitions, schedules resources, mutates evidence thresholds or persists decisions.

## V2L10 composition findings

**PASS.**

- `wre.routing.QualityMode` is the exact canonical `wre.domain.QualityMode` type; routing does not define a second enum, alias table or coercion vocabulary.
- `PREVIEW`, `FAST`, `QUALITY` and `MASTER` remain compute/product intent. They do not alter provenance classes, evidence truth or quality-gate acceptance semantics.
- `RouteGraph` is structural only. Nodes require semantic `AdapterCapabilityName` values; the graph contains no concrete adapter, model, checkpoint, score, budget, execution result or route winner.
- Route nodes and edges require canonical immutable ordering; missing endpoints, duplicates, self edges and directed cycles fail closed.
- Disconnected nodes, multiple roots/leaves and branch/merge structures are representable without implying scheduling or execution order beyond the declared DAG dependencies.
- `RouterInputSnapshot` retains the exact canonical `RouteQualityRequest`, media profiles, explicit resource budget, existing artifact references and stable prior failure categories.
- Router input performs no hardware discovery, resource estimation, registry lookup, adapter resolution, graph construction, ranking or route selection.
- `RouteResourceBudget` is a caller-supplied bound, not a scheduler. CPU/RAM/GPU/VRAM/scratch constraints are validated without querying hardware or predicting runtime.
- `FallbackPolicy` is an explicit graph-bound allowlist. A trigger contains exactly one stable `FailureCategory` or non-success `QualityDecision`.
- Positive `PASS` and `ACCEPT_WITH_WARNINGS` outcomes cannot trigger fallback.
- Retry, escalation and unresolved behavior are explicit and bounded. Missing rules or exhausted transition bounds return `UNRESOLVED`; there is no wildcard, recursive try-everything loop or silent threshold weakening.
- `RouteDecisionArtifact` records an already-chosen route structure; it does not choose, score or rank one.
- Decision reasons are typed open semantic codes with exact input-evidence refs. Foreign evidence cannot silently extend decision provenance.
- A recorded fallback policy must target exactly the recorded graph, preventing a decision artifact from binding unrelated escalation semantics.
- Decision records contain no free-form explanation, timestamp, environment snapshot, random seed, score, rank, probability, concrete adapter/model/checkpoint identity or execution result.
- No routing contract mutates reconstruction provenance or promotes a routing reason into observed/reconstructed truth.

## V2L10 deterministic fixture findings

**PASS.**

The V2L10.6 fixture rebuilds one explicit canonical routing composition twice and obtains equal `RouterInputSnapshot`, `RouteGraph`, `FallbackPolicy`, reason tuple and `RouteDecisionArtifact` values.

Stable wire-token regressions cover quality mode, prior failures, node/edge IDs, fallback actions, reason codes and the `routing.route_decision` artifact kind.

Non-canonical permutations fail closed instead of being silently normalized for:

- media profiles;
- existing artifacts;
- prior failures;
- route nodes;
- route edges;
- fallback rules;
- decision reasons;
- reason evidence.

Cross-contract invalid states also fail closed:

- foreign reason evidence;
- wrong route-decision artifact kind;
- fallback policy bound to a different graph;
- directed route cycle;
- non-canonical quality-mode input;
- inconsistent resource budget.

Exact allowlisted fallback resolution is repeatable. Missing rules and transition-bound exhaustion return the same explicit `UNRESOLVED` result with no target.

The fixture imports no new route-selection, execution, persistence or geometry responsibility.

## V2M1 end-to-end composition audit

**PASS.**

Accepted V2L5 through V2L10 evidence composes into the intended V2M1 foundation:

### Media understanding — V2L5

- immutable `MediaProfile` values link observations to exact derived evidence;
- deterministic image-quality and visual-similarity measurements remain evidence, not hidden labels or route truth;
- specialist capture hints preserve observed-versus-candidate strength;
- missing evidence remains absent.

### Adaptive reuse — V2L6

- policy, quality/diversity selection, density bounds and selected-frame materialization remain separate;
- exact parent-video provenance and microsecond frame timing are preserved;
- canonical decoded-image pyramid artifacts permit reuse without repeating equivalent decode/resize work;
- selection/decode work does not assert scene identity.

### Pair proposal — V2L7

- sequential, GPS, classical retrieval and experimental learned retrieval produce `PairCandidate` proposals only;
- retrieval union/dedup preserves source evidence and recall measurement without defining scene truth or a default winner.

### Scene identity — V2L8

- accepted same-scene connectivity requires explicit supported scene-relation evidence;
- ambiguous/degenerate repeated or symmetric look-alikes remain unresolved;
- explicit contradiction fails closed rather than being overwritten by transitive support;
- batch scaling preserves exact clustering semantics.

### Event-time synchronization evidence — V2L9

- capture-time, same-video frame timing, bounded audio correlation and visual-event evidence share one explicit relative-offset convention;
- ambiguity, weakness and source disagreement remain separate evidence-scoped states;
- disagreement does not manufacture contradiction;
- no global timeline, historical epoch or multi-source fusion is invented.

### Auditable routing foundation — V2L10

- canonical quality intent, structural capability DAG, deterministic inputs, bounded fallback and typed decision records compose without selecting or executing a solver;
- routing consumes evidence without rewriting it;
- canonical invalid states fail closed.

The V2M1 pipeline therefore supports the product boundary **media understood and organized for later reconstruction**. It has not crossed into static reconstruction truth.

## Explicit V2M1 non-capabilities

V2M1 does **not** yet provide:

- canonical `CameraSolution`;
- `DepthField` or dense/sparse canonical depth;
- `PointMap` or canonical `GeometrySolution`;
- physical `SurfaceModel`;
- photometric calibration or photorealistic appearance;
- dynamic 4D reconstruction or persistent dynamic entities;
- long-term historical `TemporalState` / `ChangeEvent` reconstruction;
- `MasterScene` or target-specific runtime compilation;
- a production route-selection heuristic, scheduler or adapter executor.

Those remain later milestones and lots. Passing V2M1 must not be interpreted as reconstructed-scene completion.

## Cross-cutting invariant audit

**PASS.**

- Observation/media source truth remains immutable.
- Derived evidence remains distinct from truth and preserves exact provenance.
- Unknown/ambiguous evidence is not guessed.
- Retrieval proposals do not define scene identity.
- Scene identity does not infer geometry.
- Event-time synchronization does not become historical chronology.
- Routing chooses no truth and weakens no evidence standard.
- Quality modes do not redefine provenance.
- Fallback is explicit, allowlisted and bounded.
- Decision reasons cannot reference provenance that was not already present in router inputs.
- No hidden fallback, unbounded retry, generic metadata bag or mutable registry lookup enters deterministic routing contracts.
- No V2M2 geometry implementation starts inside V2M1 closure.

## Review findings resolved during V2L10

**PASS after correction.**

- PR #94 required no semantic correction after the minimal routing-quality contract; lifecycle handoff passed all triggered lanes.
- PR #95 route-graph implementation remained structural and acyclic; lifecycle handoff passed all triggered lanes.
- PR #96 required Ruff formatting only in the router-input implementation; no input semantics changed.
- PR #97 required Ruff formatting only in fallback implementation/tests; no trigger/action/bound semantics changed.
- PR #99 final lifecycle validation required quoting one YAML acceptance scalar containing `:`; no route-decision semantics changed.
- PR #100 fixture development required Ruff formatting/spacing only. No production routing contract was changed by V2L10.6.
- Automated code-review availability is not treated as semantic evidence; exact-head CI plus manual deny-by-default diff review govern closure.
- No unresolved review thread or known semantic regression remains at the V2L10/V2M1 review boundary.

## Evidence

Exact V2L10 evidence:

- V2L10.1 / PR #94 / final head `8c4d7986cd794923a7d2fc1ac2ffcea290ee536b`: fast-ci #702, CodeQL #640, native PyCOLMAP #367 — **PASS**.
- V2L10.2 / PR #95 / final head `5e2256e51987bd078362c4824d9988b81b3e853d`: fast-ci #705, CodeQL #643, native PyCOLMAP #369 — **PASS**.
- V2L10.3 / PR #96 / final head `b3ac92b8afc876a755fecf788da709483fbb8913`: fast-ci #709, CodeQL #647, native PyCOLMAP #371 — **PASS**.
- V2L10.4 / PR #97 / final head `92fe6e8c14c104a608c9dcf2a4d4675e465f881f`: fast-ci #716, CodeQL #655, native PyCOLMAP #373 — **PASS**.
- V2L10.5 / PR #99 / final head `a1c5e32269edd38dab1b22bffd12aaf25ad31b30`: fast-ci #722, CodeQL #661, native PyCOLMAP #376 — **PASS**.
- V2L10.6 / PR #100 / fixture head `66ece08d7db6c3f7cb284792de5acd061b943a4a`: fast-ci #726, CodeQL #665 — **PASS**.

The final PR #100 lifecycle/review head changes review/state/roadmap metadata after this already-green fixture evidence. It must independently pass every triggered exact-head lane before merge.

Accepted lot reviews V2L5, V2L6, V2L7, V2L8 and V2L9 provide the retained milestone composition evidence referenced above.

## V2L11.1 handoff decision

After this PASS, `V2L11.1 — Canonical CameraSolution contract` may become the sole `ready` item and V2M2 may begin.

V2L11.1 is deliberately contract-only. It introduces one canonical per-observation camera solution in an explicit `LocalFrameId`, with solver-independent projection-model identity, image dimensions, intrinsic parameters, **camera-from-local-frame** rigid extrinsics, optional exact uncertainty-artifact hooks and canonical `MetricVector` measurements.

V2L11.1 must not:

- infer metric scale or world/geographic placement;
- reuse the word “world” for an unanchored local reconstruction frame;
- assert physical `CameraId` identity from a solver calibration;
- create depth, points, `GeometrySolution` or surfaces;
- convert legacy `SparseReconstructionEstimate` values yet;
- execute COLMAP or another solver.

## Decision

**V2L10 lot review: PASS.**  
**V2M1 milestone review: PASS.**

WRE now has an audited arbitrary-media understanding/organization foundation with deterministic routing contracts, while reconstruction geometry remains explicitly unimplemented. V2M2 may begin with the canonical camera contract only after PR #100 final exact-head gates pass.
