# Agent development protocol

This repository is designed to be resumed by an AI coding agent with no conversation history.

## Mandatory startup sequence

1. Read `docs/00_START_HERE.md`.
2. Read the canonical blueprint documents referenced there: `docs/01_PRODUCT.md`, `docs/02_ARCHITECTURE.md`, `docs/03_PIPELINE.md`, `docs/14_TECHNOLOGY_SELECTION.md`, `docs/15_PRODUCTION_RUNTIME.md`, `docs/23_ENGINEERING_EXECUTION.md`, `docs/24_SYSTEM_INVARIANTS.md`, `docs/25_V2_ROADMAP.md`, and `docs/26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`.
3. Read `PROJECT_STATE.yaml`.
4. Read the roadmap index `registry/work-items.yaml`, follow its `work_item_files`, and locate the exact `active_work_item` in the corresponding milestone file under `registry/work-items/`.
5. Verify the active item has a complete executable contract: objective, allowed scope, out-of-scope, dependencies, read-before, reuse, acceptance and tests.
6. Read only that item's `read_before` files plus directly relevant source/tests.
7. Inspect declared reusable dependencies before implementing anything new.
8. Run the baseline checks declared by the active item.
9. Implement only the active work item.
10. Run its acceptance tests.
11. Review the diff against objective, allowed scope, acceptance criteria, out-of-scope, architecture, system invariants and regression risk.
12. Update project state and registries before merge.
13. Do not start the next work item in the same PR unless the registry explicitly says so.

If the blueprint advances beyond the machine-readable roadmap, follow the transition rule in `docs/00_START_HERE.md`: do not opportunistically reinterpret old work items. A dedicated roadmap/state migration must reconcile the implementation sequence.

`docs/26_V2_PRODUCT_ARCHITECTURE_FREEZE.md` is authoritative for the V2 product/architecture freeze and legacy-v1 policy. V1 is an implementation donor and source of regression evidence, not a compatibility target. If a legacy interface or sequencing assumption conflicts with a frozen v2 contract, the v2 contract wins.

## Scope is deny-by-default

A work item is an allowlist. Implement only behavior required by its objective, explicit `allowed_scope`, acceptance criteria, necessary supporting tests/docs, and already-accepted invariants that it must touch. The absence of an explicit prohibition is not permission.

- Do not pull adjacent features forward because they are convenient.
- Do not add speculative extension points, options or generic abstractions for hypothetical future callers.
- Do not widen accepted input/state space unless the active contract requires it.
- Do not silently add fallback behavior that changes semantics.
- If discovered work is outside scope, record it for a future item instead of bundling it.

Prefer positive-domain logic: encode the valid states directly with explicit types/enums/invariants rather than accepting arbitrary combinations and rejecting them later.

## Work-item activation and sizing

Planned future work may remain concise. Before any item becomes `ready`, `in_progress`, `in_review`, `blocked` or `done`, it must have the complete executable contract enforced by the repository validator.

A work item must be small enough for one agent to understand, implement, test and review in one development run. It should normally contain one primary responsibility, one independently testable output contract and at most one major external integration decision.

The v2 roadmap policy limits lots to at most six work items. Split before coding when an item combines independent failure domains such as contract + several adapters, candidate generation + truth acceptance, geometry + appearance, master representation + runtime optimization, short-event dynamics + long-term chronology, or implementation + a benchmark/default-promotion program that does not yet exist.

Prefer more small lots over deep administrative nesting. One coherent work item should normally equal one PR.

## Core product rules

- **WRE is an independent visual spatiotemporal reconstruction engine.** Do not redefine it as a MONDE subsystem or a single fixed photogrammetry/splat pipeline.
- **V2 is authoritative. V1 is a legacy implementation donor, not a compatibility target.** Reuse legacy code only when it cleanly satisfies the owning v2 contract; never distort v2 architecture to preserve a v1 API, data model, sequencing assumption or unfinished roadmap.
- **Reuse before implementation.** Do not reimplement a maintained, suitable foundational algorithm merely to own the code.
- External solvers and learned models remain behind explicit adapters with stable WRE contracts.
- Do not implement custom SIFT, SfM, bundle adjustment, RANSAC, ICP, MVS, Gaussian rasterization, video codecs or equivalent foundational systems unless an accepted ADR/work item shows a measured reason that existing implementations are inadequate.
- **Separate responsibilities.** Observation, correspondences, camera/depth geometry, explicit surface, photorealistic appearance, materials, environment, dynamics, chronology and runtime assets are different data/products even when one method can emit several of them.
- A visually convincing render is not automatically accurate collision/measurement geometry.
- **Preserve provenance class.** Reconstructed/observed, inferred and generated content must remain distinguishable.
- Generated completion may improve REALISTIC/CINEMATIC outputs but must not silently become reconstructed reality.
- **Route by data profile.** Sparse photos, large unordered collections, short dynamic videos, long streams, 360 capture, drone footage, multi-camera events and historical media may use different methods.
- PREVIEW/FAST/QUALITY/MASTER may spend different compute and use different specialist methods; their outputs must retain route/model/configuration/quality metadata.
- Quality must be measured per relevant dimension: geometry, cameras, rendering, temporal consistency, latency/resources and runtime performance are not collapsed into one misleading metric.
- Failed quality gates may retry, escalate, use a registered specialist fallback or remain unresolved.
- **Algorithms are replaceable.** New research enters through the technology/benchmark registry and adapter contracts, not by rewriting core domain semantics around the latest paper.
- Learned methods may be primary reconstruction engines when the owning work item permits them and their outputs satisfy the relevant product quality contract; they are not restricted to candidate proposal only.
- Classical geometric verification/refinement remains an important tool and may be required by route/quality policy, but the architecture does not assume every learned output is merely a proposal.
- Keep long-term chronological state distinct from short continuous dynamic-event time.
- Do not force century-scale scene history into one smooth deformation field.
- Dynamic objects must not contaminate stable scene geometry merely because they appear in source images.
- Sky/environment should be represented as environment when appropriate, not fake nearby solid geometry.
- Master representation, exchange format and runtime representation are distinct concerns.
- Photorealistic splats/radiance do not remove the need for explicit collision/navigation geometry when the runtime requires physics.
- Artifact caching, resumability, model/version/license metadata, benchmark evidence, human review, LOD/compression and runtime compilation are first-class product capabilities.
- A completed work item must have objective acceptance evidence and relevant tests.
- Avoid unrelated refactors inside feature PRs.

## Fail-closed behavior

Unknown, ambiguous or unsupported evidence must remain explicit instead of being coerced into a valid-looking state.

Examples include unsupported CRS, unknown timezone, unsupported camera model, weak scene identity, insufficient geometric support, uncertain historical dating and unapproved checkpoints. Prefer `UNKNOWN`, `UNRESOLVED`, a separate hypothesis, a hole or an explicit route failure over invented precision.

Fallbacks are allowlisted. Do not implement “try everything” behavior, silently lower correctness thresholds or switch quality/provenance classes because the preferred route failed.

## Contract-first sequencing

For a new capability family, normally implement in this order:

1. solver-independent contract;
2. codec/persistence support if needed;
3. smallest baseline/compatibility adapter;
4. focused fixtures/tests;
5. alternative specialist adapter(s);
6. benchmark under the same contract;
7. default/router promotion;
8. expensive optimization.

Do not build routing policy around incomparable solver-private outputs.

## Artifact and reproducibility rules

- Every expensive derived artifact must be traceable to its inputs, producer adapter/model/checkpoint, normalized configuration and producer version.
- Prefer content-addressed cache keys for deterministic/reusable work.
- A manual correction is versioned input, not hidden mutable state.
- A new model/checkpoint must have explicit registry metadata including source/version, license notes, hardware support and shipping status before becoming a default.
- Do not silently download floating `latest` checkpoints in reproducible paths.
- Heavy jobs should expose checkpoint/resume semantics when the underlying tool supports them.
- Coordinate-system conversions must be explicit, tested and centralized; never scatter undocumented axis/sign conventions.
- Color/exposure transforms must be explicit enough that incompatible camera pipelines are not accidentally treated as identical radiance observations.
- Legacy components may be reused, generalized or temporarily wrapped when they cleanly satisfy a v2 need. They may also be replaced or removed once validated v2 coverage and migration evidence exist; backward compatibility with v1 is not a product requirement.

## Reviews

Three review levels are required:

- **PR review:** every work item. Check diff, allowlisted scope, acceptance criteria, tests, interfaces, reuse, security, reproducibility and system invariants.
- **Lot review:** after the final item of a lot. Verify all promised capabilities exist and that architecture responsibilities were not accidentally conflated.
- **Milestone review:** end-to-end validation before advancing the milestone.

Review gates are blocking state transitions. Unless a dedicated review work item exists, the PR that completes the final work item in a lot performs the lot review before `PROJECT_STATE.yaml` advances to a later lot. At a milestone boundary, that same handoff must also complete the milestone review before the next milestone becomes active. A passed review must record evidence in `registry/reviews.yaml`. The repository validator enforces these transitions.

A lot cannot be marked complete until its lot review passes. A milestone cannot be marked complete until its milestone review passes.

## Tests and regression discipline

- Prefer fast deterministic tests on every PR.
- Add a regression test whenever fixing a bug that can reasonably recur.
- Tests must exercise invalid/forbidden states as well as happy paths for new contracts.
- Geometry comparisons use explicit tolerances; never rely on accidental exact floating-point equality.
- Fast CI validates contracts, schemas, adapters with lightweight fixtures and repository state.
- Heavy GPU/real-scene benchmarks belong in dedicated workflows/benchmark infrastructure unless the active item specifically requires them.
- Performance checks are informative by default unless a calibrated benchmark gate explicitly says otherwise.
- Router benchmarks must test quality/failure behavior as well as compute saved.
- Appearance benchmarks should include held-out views; geometry benchmarks should use independent geometry/camera references where available.
- Dynamic benchmarks should measure temporal consistency/tracking/object persistence, not only per-frame image quality.
- Historical-change tests must distinguish physical change from registration error, occlusion and uncertain dating.
- Runtime benchmarks should include FPS, memory, loading/streaming and LOD behavior for representative target devices.
- Run `uv run python scripts/validate_repo.py` before merge whenever repository state, roadmap, components, reviews or agent instructions change.

## Git discipline

- Work from a short-lived branch.
- Use a PR even for agent-only development after repository bootstrap.
- Keep CI green before merge.
- Prefer squash merge so `main` remains a readable sequence of work items.
- Never force-push `main`.
- Update `PROJECT_STATE.yaml` as part of the PR that changes implementation state.
- Final handoff metadata must make the next permitted change unambiguous without conversation history.
