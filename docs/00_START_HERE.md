# Start here

This is the mandatory entry point for a new development session.

## Canonical product context

Before interpreting the implementation roadmap, read the current product/architecture blueprint in this order:

1. [`01_PRODUCT.md`](01_PRODUCT.md) — what WRE is and the user outcome it targets.
2. [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) — responsibilities, representations, artifact graph, router and quality architecture.
3. [`03_PIPELINE.md`](03_PIPELINE.md) — adaptive reconstruction routes by data profile.
4. [`14_TECHNOLOGY_SELECTION.md`](14_TECHNOLOGY_SELECTION.md) — current candidate technologies and benchmark policy.
5. [`15_PRODUCTION_RUNTIME.md`](15_PRODUCTION_RUNTIME.md) — professional orchestration, caching, scheduling, runtime compilation and distribution.
6. [`23_ENGINEERING_EXECUTION.md`](23_ENGINEERING_EXECUTION.md) — how blueprint capabilities are split into one-run work items and implemented safely.
7. [`24_SYSTEM_INVARIANTS.md`](24_SYSTEM_INVARIANTS.md) — cross-cutting fail-closed product invariants every subsystem must preserve.

These documents define the current target product. WRE is an independent visual spatiotemporal reconstruction engine; it is not defined as a MONDE subsystem and is not a single COLMAP-to-splat pipeline.

## Resume procedure

1. Read the canonical product context above.
2. Read `../AGENTS.md`.
3. Read `../PROJECT_STATE.yaml` and note `active_work_item`.
4. Find that exact ID in `../registry/work-items.yaml`.
5. Check its dependencies are complete.
6. Read its `read_before` files.
7. Inspect only relevant implementation/tests plus the reuse/dependency registries.
8. Run the item's baseline checks if configured.
9. Implement the item without expanding scope.
10. Run acceptance checks and relevant regression tests.
11. Review the diff against the item contract and system invariants.
12. Update state/registries in the same PR.

If repository state and conversation history disagree, the repository is authoritative unless the user explicitly changes the plan.

## Deny-by-default implementation rule

The active work item is an allowlist. An agent may implement only the objective, explicit acceptance criteria, the minimum supporting code/tests/docs they require, and already-accepted invariants necessarily touched by the change. Missing `out_of_scope` wording is not permission to add adjacent functionality.

If the requested behavior does not fit the active item, record it as future work or perform an explicit roadmap/state change first. Do not opportunistically bundle it.

## Work-item sizing rule

Every implementation work item must be small enough for one agent to understand, implement, test and review in one development run. If one item contains several independent solver integrations, responsibilities or failure domains, split it before coding. Prefer additional small lots over deep administrative nesting.

## Blueprint-versus-roadmap transition rule

The product/architecture blueprint can evolve ahead of the machine-readable implementation roadmap. When that happens:

- do not reinterpret an old work item as permission to implement newly described future capabilities;
- do not erase already completed work merely because the target architecture changed;
- create a dedicated roadmap/state migration that maps retained work, deprecated assumptions, new lots and new acceptance gates;
- classify existing code as retained, generalized, compatibility-wrapped, deprecated or removable before rewriting it;
- after migration, `PROJECT_STATE.yaml` and `registry/work-items.yaml` again become the authoritative implementation sequence.

Until such a migration is merged, the current active work item describes existing implementation state, while the canonical blueprint documents describe the target product architecture.

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
- Preserve tested existing components through compatibility/generalization unless replacement evidence justifies deletion.
