# Start here

This is the mandatory entry point for a new development session.

## Canonical product context

Before interpreting the implementation roadmap, read the current product/architecture blueprint in this order:

1. [`01_PRODUCT.md`](01_PRODUCT.md) — what WRE is and the user outcome it targets.
2. [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) — responsibilities, representations, artifact graph, router and quality architecture.
3. [`03_PIPELINE.md`](03_PIPELINE.md) — adaptive reconstruction routes by data profile.
4. [`14_TECHNOLOGY_SELECTION.md`](14_TECHNOLOGY_SELECTION.md) — current candidate technologies and benchmark policy.
5. [`27_RESEARCH_CANDIDATE_COVERAGE.md`](27_RESEARCH_CANDIDATE_COVERAGE.md) — living map from current research families/candidates to the frozen WRE contracts and roadmap lots; candidate names are intentionally not frozen.
6. [`15_PRODUCTION_RUNTIME.md`](15_PRODUCTION_RUNTIME.md) — professional orchestration, caching, scheduling, runtime compilation and distribution.
7. [`23_ENGINEERING_EXECUTION.md`](23_ENGINEERING_EXECUTION.md) — how blueprint capabilities are split into one-run work items and implemented safely.
8. [`24_SYSTEM_INVARIANTS.md`](24_SYSTEM_INVARIANTS.md) — cross-cutting fail-closed product invariants every subsystem must preserve.
9. [`25_V2_ROADMAP.md`](25_V2_ROADMAP.md) — executable capability sequence for WRE v2.0.
10. [`26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`](26_V2_PRODUCT_ARCHITECTURE_FREEZE.md) — frozen v2 product/architecture authority, scientific replacement rule and legacy-v1 donor policy.

These documents define the current target product. WRE is an independent visual spatiotemporal reconstruction engine; it is not defined as a MONDE subsystem and is not a single COLMAP-to-splat pipeline. The v2 product/architecture is authoritative over legacy v1 structure; v1 is only an implementation donor and regression/reference source where useful.

The product/architecture documents are frozen under `26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`; the technology/candidate landscape is intentionally living. A new paper normally changes `14_TECHNOLOGY_SELECTION.md`, `27_RESEARCH_CANDIDATE_COVERAGE.md`, benchmark evidence or an adapter default—not the frozen architecture.

## Resume procedure

1. Read the canonical product context above.
2. Read `../AGENTS.md`.
3. Read `../PROJECT_STATE.yaml` and note `active_work_item`.
4. Read the v2 roadmap index `../registry/work-items.yaml`.
5. Follow its `work_item_files` list and find the exact active work-item ID in the corresponding milestone file under `../registry/work-items/`.
6. Check its dependencies are complete and its executable contract is fully specified.
7. Read only its `read_before` files plus directly relevant source/tests and reuse/dependency registries.
8. Run the item's baseline checks if configured.
9. Implement the item without expanding scope.
10. Run acceptance checks and relevant regression tests.
11. Review the diff against the item contract and system invariants.
12. Update state/registries in the same PR.

If repository state and conversation history disagree, the repository is authoritative unless the user explicitly changes the plan.

## Deny-by-default implementation rule

The active work item is an allowlist. An agent may implement only the objective, explicit `allowed_scope`, acceptance criteria, the minimum supporting code/tests/docs they require, and already-accepted invariants necessarily touched by the change. Missing `out_of_scope` wording is not permission to add adjacent functionality.

If the requested behavior does not fit the active item, record it as future work or perform an explicit roadmap/state change first. Do not opportunistically bundle it.

## Work-item activation rule

Planned future work may stay concise so the roadmap can survive changing research. **A work item cannot become `ready`, `in_progress`, `in_review`, `blocked` or `done` without a complete executable contract** containing at least:

- objective;
- allowed scope;
- out-of-scope list;
- dependencies;
- read-before files;
- reuse decision;
- acceptance criteria;
- tests/evidence.

The repository validator enforces this. The PR handing off to the next item must expand that item's contract before marking it `ready`.

For any work item that integrates or promotes a solver/model, activation also requires a candidate refresh against `14_TECHNOLOGY_SELECTION.md`, `27_RESEARCH_CANDIDATE_COVERAGE.md`, the current adapter/model registry and current evidence. Do not blindly implement the method that happened to be fashionable when the roadmap was written.

## Work-item sizing rule

Every implementation work item must be small enough for one agent to understand, implement, test and review in one development run. The v2 roadmap policy currently limits lots to at most six work items. If one item contains several independent solver integrations, responsibilities or failure domains, split it before coding. Prefer additional small lots over deep administrative nesting.

## Blueprint-versus-roadmap transition rule

The frozen v2 product/architecture can change only through an explicit blueprint-change decision as defined in `26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`. When an accepted blueprint revision advances beyond the machine-readable implementation roadmap:

- do not reinterpret an old work item as permission to implement newly described future capabilities;
- do not erase already completed work merely because the target architecture changed;
- create a dedicated roadmap/state migration that maps retained work, deprecated assumptions, new lots and new acceptance gates;
- classify existing implementation code by whether it cleanly serves the new v2 contract; legacy-v1 compatibility is never itself a reason to retain a design;
- after migration, `PROJECT_STATE.yaml` and the split v2 work-item registry again become the authoritative implementation sequence.

Replacing a solver/model behind an existing frozen contract is normal adapter evolution and does not require a blueprint change.

## Product in one paragraph

WRE receives arbitrary real-world photos and videos, organizes which observations belong together in space and time, chooses appropriate specialist reconstruction routes, recovers geometry/surfaces/appearance/dynamics/long-term temporal states, evaluates quality with explicit gates, and compiles high-fidelity master reconstructions into efficient interactive scenes. The target experience is to walk, fly, use a virtual drone camera, render new supported viewpoints, replay dynamic events and scrub historical scene states with a result that feels like being in the real place. Physical geometry, photorealistic appearance, materials, environment, temporal state and runtime representation remain distinct but composable layers.

## Development principles

- Reuse mature external methods before custom implementation.
- Keep core contracts solver-independent.
- Encode valid states directly; fail closed on ambiguity or unsupported input.
- Treat geometry, surface, appearance, materials, environment, dynamics and runtime formats as separate responsibilities.
- Route according to data profile, quality mode, hardware budget and quality-gate outcomes.
- Preserve reconstructed/inferred/generated provenance classes.
- Keep PREVIEW/FAST/QUALITY/MASTER products measurable and reproducible.
- Treat artifact caching, resumability, benchmark registries, human review, compression and streaming as product capabilities, not afterthoughts.
- Prefer diagnosable unresolved output over a polished but structurally wrong reconstruction.
- Treat legacy v1 as an optional implementation donor, never as a compatibility target. Reuse only what cleanly satisfies v2; replace/remove legacy code when validated v2 coverage exists.
