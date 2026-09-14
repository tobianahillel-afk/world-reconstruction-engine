# Start here

This is the mandatory entry point for a new development session.

## Resume procedure

1. Read `../AGENTS.md`.
2. Read `../PROJECT_STATE.yaml` and note `active_work_item`.
3. Find that exact ID in `../registry/work-items.yaml`.
4. Check its dependencies are complete.
5. Read its `read_before` files.
6. Inspect only relevant implementation/tests plus the reuse registry.
7. Run the item's baseline checks if configured.
8. Implement the item without expanding scope.
9. Run acceptance checks and relevant regression tests.
10. Review the diff.
11. Update state/registries in the same PR.

If repository state and conversation history disagree, the repository is authoritative unless the user explicitly changes the plan.

## Product in one paragraph

WRE reconstructs local spatial fragments from media, incrementally places new observations into existing fragments when geometric evidence supports it, keeps disconnected/unknown fragments when it does not, and later merges or absolutely anchors fragments when independent evidence creates a reliable bridge. A valid fragment may remain in a local frame until GPS, ground-control points, known landmarks or a georeferenced reference fragment support world placement. When independently supported geometry differs across time, WRE preserves time-valid geometry states/change hypotheses instead of overwriting history or treating every disagreement as an outlier. The engine remains usable independently and exposes clean observation/evidence/geometry/temporal contracts for future MONDE integration.

## Development principle

Adapters and orchestration are WRE's default. Proven external solvers remain behind explicit interfaces. WRE's novel value is the evidence graph, deterministic compute routing, uncertainty/hypothesis handling, fragment lifecycle, absolute-anchor constraints, temporal state/history semantics, validation, incremental world graph and robust composition of existing geometry engines.
