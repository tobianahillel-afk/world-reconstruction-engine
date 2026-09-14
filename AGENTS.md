# Agent development protocol

This repository is designed to be resumed by an AI coding agent with no conversation history.

## Mandatory startup sequence

1. Read `docs/00_START_HERE.md`.
2. Read `PROJECT_STATE.yaml`.
3. Read the active item in `registry/work-items.yaml`.
4. Read only that item's `read_before` files plus directly relevant source/tests.
5. Inspect declared reusable dependencies before implementing anything new.
6. Run the baseline checks declared by the active item.
7. Implement only the active work item.
8. Run its acceptance tests.
9. Review the diff against objective, acceptance criteria, out-of-scope, architecture and regression risk.
10. Update project state and registries before merge.
11. Do not start the next work item in the same PR unless the registry explicitly says so.

## Core rules

- **Reuse before implementation.** Do not reimplement a maintained, suitable algorithm merely to own the code.
- Do not implement custom SIFT, SfM, bundle adjustment, RANSAC, ICP, pose-graph optimization, M3C2/change measurement or equivalent foundational solvers unless an accepted ADR explicitly authorizes it.
- Learned methods may propose candidates only when explicitly allowed by architecture; authoritative acceptance must remain backed by measurable geometric evidence.
- Raw observations, derived evidence and estimated geometry are different data classes and must not be silently conflated.
- Never force a placement or merge when evidence is insufficient. `UNKNOWN` / `UNRESOLVED` is valid.
- Fragment and entity merges must be reversible and retain provenance.
- Preserve competing hypotheses when architecture requires them; do not silently delete alternatives.
- **Adaptive routing changes cost, not truth criteria.** A FAST route may perform less work than STANDARD/ESCALATED, but it must satisfy the same acceptance/evidence requirements before WRE accepts a result.
- Routing decisions must be deterministic/auditable enough to explain route, inputs, configuration/thresholds, reasons and escalation outcome.
- Absolute anchors (GPS, GCPs, known landmarks, reference fragments) enter as provenance-bearing constraints with uncertainty; they do not destructively rewrite local geometry or bypass residual checks.
- Historical geometric disagreement is not automatically an outlier. When independently supported across time, preserve time-valid geometry states/change hypotheses rather than overwriting earlier geometry.
- Temporal transitions/state changes must retain source evidence and remain reversible/auditable.
- Photorealistic learned/neural rendering products may be used for visualization only when architecture permits them; they are not canonical WRE world geometry.
- A completed work item must have objective acceptance evidence and relevant tests.
- One coherent work item should normally equal one PR.
- Avoid unrelated refactors inside feature PRs.
- Any discovered work outside scope must be recorded, not opportunistically bundled.

## Reviews

Three review levels are required:

- **PR review:** every work item. Check diff, acceptance criteria, tests, interfaces, reuse, security and scope.
- **Lot review:** after the final item of a lot. Verify all promised capabilities exist, are tested, documented and not accidentally omitted.
- **Milestone review:** end-to-end validation before advancing the milestone.

Review gates are blocking state transitions. Unless a dedicated review work item exists, the PR that completes the final work item in a lot performs the lot review before `PROJECT_STATE.yaml` advances to a later lot. At a milestone boundary, that same handoff must also complete the milestone review before the next milestone becomes active. A passed review must record evidence in `registry/reviews.yaml`. The repository validator enforces these transitions.

A lot cannot be marked complete until its lot review passes. A milestone cannot be marked complete until its milestone review passes.

## Tests and regression discipline

- Prefer fast deterministic tests on every PR.
- Add a regression test whenever fixing a bug that can reasonably recur.
- Geometry comparisons use explicit tolerances; never rely on accidental exact floating-point equality.
- Performance checks are informative by default unless a stable benchmark gate explicitly says otherwise.
- Heavy reconstruction/GPU tests belong outside the fast PR lane unless the active item specifically requires them.
- Router benchmarks must test quality/non-regression as well as compute saved; faster routing may not hide increased false acceptance.
- Temporal-change tests must distinguish true physical change from registration error, occlusion and contradictory dating.
- Run `uv run python scripts/validate_repo.py` before merge whenever repository state, roadmap, components, reviews or agent instructions change.

## Git discipline

- Work from a short-lived branch.
- Use a PR even for agent-only development after repository bootstrap.
- Keep CI green before merge.
- Prefer squash merge so `main` remains a readable sequence of work items.
- Never force-push `main`.
- Update `PROJECT_STATE.yaml` as part of the PR that changes state.
