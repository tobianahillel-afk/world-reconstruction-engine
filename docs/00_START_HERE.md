# Start here

This is the mandatory entry point for a new development session.

## Canonical product context

Before interpreting the implementation roadmap, read the current product/architecture blueprint in this order:

1. [`01_PRODUCT.md`](01_PRODUCT.md) — what WRE is and the user outcome it targets.
2. [`02_ARCHITECTURE.md`](02_ARCHITECTURE.md) — responsibilities, representations, artifact graph, router and quality architecture.
3. [`03_PIPELINE.md`](03_PIPELINE.md) — adaptive reconstruction routes by data profile.
4. [`14_TECHNOLOGY_SELECTION.md`](14_TECHNOLOGY_SELECTION.md) — current candidate technologies and benchmark policy.
5. [`15_PRODUCTION_RUNTIME.md`](15_PRODUCTION_RUNTIME.md) — professional orchestration, caching, scheduling, runtime compilation and distribution.

These documents define the current target product. WRE is an independent visual spatiotemporal reconstruction engine; it is not defined as a MONDE subsystem and is not a single COLMAP-to-splat pipeline.

## Resume procedure

1. Read the canonical product context above.
2. Read `../AGENTS.md`.
3. Read `../PROJECT_STATE.yaml` and note `active_work_item`.
4. Find that exact ID in `../registry/work-items.yaml`.
5. Check its dependencies are complete.
6. Read its `read_before` files.
7. Inspect only relevant implementation/tests plus the reuse registry.
8. Run the item's baseline checks if configured.
9. Implement the item without expanding scope.
10. Run acceptance checks and relevant regression tests.
11. Review the diff.
12. Update state/registries in the same PR.

If repository state and conversation history disagree, the repository is authoritative unless the user explicitly changes the plan.

## Blueprint-versus-roadmap transition rule

The product/architecture blueprint can evolve ahead of the machine-readable implementation roadmap. When that happens:

- do not reinterpret an old work item as permission to implement newly described future capabilities;
- do not erase already completed work merely because the target architecture changed;
- create a dedicated roadmap/state migration that maps retained work, deprecated assumptions, new lots and new acceptance gates;
- after migration, `PROJECT_STATE.yaml` and `registry/work-items.yaml` again become the authoritative implementation sequence.

Until such a migration is merged, the current active work item describes existing implementation state, while `01_PRODUCT.md` through `15_PRODUCTION_RUNTIME.md` describe the target product architecture.

## Product in one paragraph

WRE receives arbitrary real-world photos and videos, organizes which observations belong together in space and time, chooses appropriate specialist reconstruction routes, recovers geometry/surfaces/appearance/dynamics/long-term temporal states, evaluates quality with explicit gates, and compiles high-fidelity master reconstructions into efficient interactive scenes. The target experience is to walk, fly, use a virtual drone camera, render new supported viewpoints, replay dynamic events and scrub historical scene states with a result that feels like being in the real place. Physical geometry, photorealistic appearance, materials, environment, temporal state and runtime representation remain distinct but composable layers.

## Development principles

- Reuse mature external methods before custom implementation.
- Keep core contracts solver-independent.
- Treat geometry, surface, appearance, materials, environment, dynamics and runtime formats as separate responsibilities.
- Route according to data profile, quality mode, hardware budget and quality-gate outcomes.
- Preserve reconstructed/inferred/generated provenance classes.
- Keep PREVIEW/FAST/QUALITY/MASTER products measurable and reproducible.
- Treat artifact caching, resumability, benchmark registries, human review, compression and streaming as product capabilities, not afterthoughts.
- Prefer diagnosable unresolved output over a polished but structurally wrong reconstruction.