# L0 / M0 bootstrap review

Date: 2026-09-14
Reviewed main commit: `9d944fd2899a7ea1f76a0117b4be1b8fbfb3e9e7`
Scope: lot `L0` — Bootstrap and development system; milestone `M0` — Repo ready
Verdict: **PASS**

## Review objective

Verify that the repository bootstrap is complete, fast, auditable and resumable before the first product-domain implementation begins. This review covers all work items L0.1 through L0.5 and acts as both the required L0 lot review and the M0 milestone review because M0 contains only L0.

## L0.1 — Repository bootstrap and agent protocol

**PASS.** A new agent has an explicit entry point in `docs/00_START_HERE.md`, a mandatory execution protocol in `AGENTS.md`, canonical machine-readable state in `PROJECT_STATE.yaml`, a machine-readable work graph in `registry/work-items.yaml`, component/review/dependency registries, and a standard PR evidence contract.

Evidence:

- PR #1 — `[L0.1] Bootstrap repository and agent-resume protocol`
- `docs/00_START_HERE.md`
- `AGENTS.md`
- `PROJECT_STATE.yaml`
- `registry/work-items.yaml`
- `registry/reviews.yaml`

The repository explicitly states that repository state is authoritative over stale conversation context, which makes a fresh-agent resume possible without relying on chat history.

## L0.2 — Python tooling and fast CI

**PASS.** The project has a small Python 3.12 development toolchain with uv, Ruff, Pyright, pytest and actionlint. Fast CI uses a single latency-sensitive job, cancels obsolete runs, uses read-only repository permissions, does not persist checkout credentials and excludes native geometry/GPU work.

Evidence:

- PR #2 — `[L0.2] Add Python tooling and fast CI`
- `pyproject.toml`
- `.github/workflows/ci.yml`
- main fast-ci run `34882936749` on commit `9d944fd2899a7ea1f76a0117b4be1b8fbfb3e9e7`: passed

## L0.3 — Repository state validator

**PASS.** Repository metadata is executable policy rather than documentation only. The validator checks active work, status/dependency coherence, cycles, `read_before` paths, state/done synchronization, component paths, PR-template sections, review coverage and blocking lot/milestone transitions. Negative tests deliberately corrupt repository metadata and verify rejection.

Evidence:

- PR #3 — `[L0.3] Add repository state validator`
- `scripts/validate_repo.py`
- `src/wre/repo_validation.py`
- `tests/test_repo_validation.py`
- validator is executed before lint/type/tests in fast CI

A regression discovered during later handoffs was also removed: the active-dependency negative test now derives the current work item and its dependency dynamically instead of hard-coding an L0 item ID.

## L0.4 — Dependency and supply-chain baseline

**PASS with documented platform-setting limitations.** Development dependencies are locked with `uv.lock`, uv itself is pinned, CI installs in frozen mode, GitHub Actions are SHA-pinned, Dependabot is configured for native uv and GitHub Actions updates, and CodeQL runs outside the fast lane.

Evidence:

- PR #4 — `[L0.4] Add dependency and supply-chain baseline`
- `uv.lock`
- `.github/dependabot.yml`
- `.github/workflows/codeql.yml`
- `.github/workflows/dependency-review.yml`
- `docs/08_SECURITY.md`
- `registry/dependencies.yaml`

Known limitation: GitHub currently returns HTTP 404 for the Dependency Graph SBOM endpoint, so strict Dependency Review cannot execute until Dependency Graph is enabled in repository settings. The workflow performs an explicit capability preflight and warns/skips only that unavailable step; it does not claim that strict review is active. Branch rules, secret scanning and push protection are likewise documented as settings requiring manual verification because the current connector does not expose safe write actions for them. These limitations do not violate the L0.4 acceptance criterion, which explicitly scopes automation to repository permissions/capabilities available to the agent.

## L0.5 — Fixture and regression harness skeleton

**PASS.** The repository has versioned fixture/result contracts, deterministic fixture conventions, explicit metric comparators/tolerances, stable canonical serialization and a tiny synthetic fast-lane fixture. Fixture input paths are constrained for both POSIX and Windows path separators.

Evidence:

- PR #5 — `[L0.5] Add deterministic fixture and regression harness`
- `docs/09_FIXTURES.md`
- `registry/schemas/fixture.schema.json`
- `registry/schemas/regression-result.schema.json`
- `src/wre/regression.py`
- `tests/fixtures/synthetic/tiny-smoke/`
- `tests/test_regression_harness.py`
- final PR fast-ci: passed
- final PR CodeQL: passed

The fixture is intentionally a harness smoke test and makes no reconstruction-quality claim.

## Resume-path audit

**PASS.** The canonical chain is coherent:

`docs/00_START_HERE.md` → `AGENTS.md` → `PROJECT_STATE.yaml` → active `registry/work-items.yaml` item → its `read_before` files → relevant source/tests/dependency registry.

The validator additionally prevents a future state handoff from silently diverging from that chain.

## Scope audit

**PASS.** At the reviewed main commit, `src/wre/` contains only:

- `__init__.py`
- `repo_validation.py`
- `regression.py`

No SIFT implementation, SfM solver, bundle adjustment, RANSAC, ICP, fragment merge engine, world graph, media ingestion model or other product-domain reconstruction code has been prematurely implemented. Native geometry candidates remain registry entries only.

## CI and regression posture

**PASS for M0.** The fast lane is intentionally small and deterministic. The heavier full/regression lane is not yet configured because no native reconstruction engine exists; this is explicitly deferred to the owning roadmap lots. Correctness fixtures can already be represented by the L0.5 harness.

## Residual limitations carried into M1

These are recorded constraints, not hidden blockers:

- GitHub Dependency Graph is not currently enabled, so strict Dependency Review is not active.
- Branch protection/rulesets, secret scanning and push protection require manual GitHub-settings verification.
- Final WRE project license remains undecided; no native reconstruction candidate has been promoted to a mandatory dependency.
- Heavy/full geometry regression CI is intentionally deferred until geometry engines exist.

## Decision

The L0 lot review **passes**. The M0 milestone review **passes**. The bootstrap is sufficiently coherent, tested, resumable and scoped to begin M1.

The next permitted work item is `L1.1 — Observation models`. This transition does not authorize later L1 items, geometry engines or unrelated implementation. Normal one-work-item-per-PR discipline remains in force.
