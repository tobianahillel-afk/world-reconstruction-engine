# Agent development protocol

This repository is designed to be resumed by an AI coding agent with no conversation history.

## Mandatory startup sequence

1. Read `docs/00_START_HERE.md`.
2. Read the canonical blueprint documents referenced there: `docs/01_PRODUCT.md`, `docs/02_ARCHITECTURE.md`, `docs/03_PIPELINE.md`, `docs/14_TECHNOLOGY_SELECTION.md`, and `docs/15_PRODUCTION_RUNTIME.md`.
3. Read `PROJECT_STATE.yaml`.
4. Read the active item in `registry/work-items.yaml`.
5. Read only that item's `read_before` files plus directly relevant source/tests.
6. Inspect declared reusable dependencies before implementing anything new.
7. Run the baseline checks declared by the active item.
8. Implement only the active work item.
9. Run its acceptance tests.
10. Review the diff against objective, acceptance criteria, out-of-scope, architecture and regression risk.
11. Update project state and registries before merge.
12. Do not start the next work item in the same PR unless the registry explicitly says so.

If the blueprint has advanced beyond the machine-readable roadmap, follow the transition rule in `docs/00_START_HERE.md`: do not opportunistically reinterpret old work items. A dedicated roadmap/state migration must reconcile the implementation sequence.

## Core product rules

- **WRE is an independent visual spatiotemporal reconstruction engine.** Do not redefine it as a MONDE subsystem or a single fixed photogrammetry/splat pipeline.
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
- Failed quality gates may retry, escalate, use a specialist fallback or remain unresolved.
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
- One coherent work item should normally equal one PR.
- Avoid unrelated refactors inside feature PRs.
- Any discovered work outside scope must be recorded, not opportunistically bundled.

## Artifact and reproducibility rules

- Every expensive derived artifact must be traceable to its inputs, producer adapter/model/checkpoint, normalized configuration and producer version.
- Prefer content-addressed cache keys for deterministic/reusable work.
- A manual correction is versioned input, not hidden mutable state.
- A new model/checkpoint must have explicit registry metadata including source/version, license notes, hardware support and shipping status before becoming a default.
- Do not silently download floating `latest` checkpoints in reproducible paths.
- Heavy jobs should expose checkpoint/resume semantics when the underlying tool supports them.
- Coordinate-system conversions must be explicit, tested and centralized; never scatter undocumented axis/sign conventions.
- Color/exposure transforms must be explicit enough that incompatible camera pipelines are not accidentally treated as identical radiance observations.

## Reviews

Three review levels are required:

- **PR review:** every work item. Check diff, acceptance criteria, tests, interfaces, reuse, security, reproducibility and scope.
- **Lot review:** after the final item of a lot. Verify all promised capabilities exist and that architecture responsibilities were not accidentally conflated.
- **Milestone review:** end-to-end validation before advancing the milestone.

Review gates are blocking state transitions. Unless a dedicated review work item exists, the PR that completes the final work item in a lot performs the lot review before `PROJECT_STATE.yaml` advances to a later lot. At a milestone boundary, that same handoff must also complete the milestone review before the next milestone becomes active. A passed review must record evidence in `registry/reviews.yaml`. The repository validator enforces these transitions.

A lot cannot be marked complete until its lot review passes. A milestone cannot be marked complete until its milestone review passes.

## Tests and regression discipline

- Prefer fast deterministic tests on every PR.
- Add a regression test whenever fixing a bug that can reasonably recur.
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
