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
- Do not implement custom SIFT, SfM, bundle adjustment, RANSAC, ICP, pose-graph optimization or equivalent foundational solvers unless an accepted ADR explicitly authorizes it.
- Learned methods may propose candidates only when explicitly allowed by architecture; authoritative acceptance must remain backed by measurable geometric evidence.
- Raw observations, derived evidence and estimated geometry are different data classes and must not be silently conflated.
- Never force a placement or merge when evidence is insufficient. `UNKNOWN` / `UNRESOLVED` is valid.
- Fragment and entity merges must be reversible and retain provenance.
- Preserve competing hypotheses when architecture requires them; do not silently delete alternatives.
- A completed work item must have objective acceptance evidence and relevant tests.
- One coherent work item should normally equal one PR.
- Avoid unrelated refactors inside feature PRs.
- Any discovered work outside scope must be recorded, not opportunistically bundled.

## Reviews

Three review levels are required:

- **PR review:** every work item. Check diff, acceptance criteria, tests, interfaces, reuse, security and scope.
- **Lot review:** after the final item of a lot. Verify all promised capabilities exist, are tested, documented and not accidentally omitted.
- **Milestone review:** end-to-end validation before advancing the milestone.

A lot cannot be marked complete until its lot review passes. A milestone cannot be marked complete until its milestone review passes.

## Tests and regression discipline

- Prefer fast deterministic tests on every PR.
- Add a regression test whenever fixing a bug that can reasonably recur.
- Geometry comparisons use explicit tolerances; never rely on accidental exact floating-point equality.
- Performance checks are informative by default unless a stable benchmark gate explicitly says otherwise.
- Heavy reconstruction/GPU tests belong outside the fast PR lane unless the active item specifically requires them.

## Git discipline

- Work from a short-lived branch.
- Use a PR even for agent-only development after repository bootstrap.
- Keep CI green before merge.
- Prefer squash merge so `main` remains a readable sequence of work items.
- Never force-push `main`.
- Update `PROJECT_STATE.yaml` as part of the PR that changes state.
