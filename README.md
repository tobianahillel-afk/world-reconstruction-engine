# World Reconstruction Engine (WRE)

WRE is an independent visual spatiotemporal reconstruction engine for turning arbitrary real-world photos and videos into high-fidelity, freely navigable 3D/4D scenes.

The product is designed to organize unordered media, recover spatial and temporal relationships, reconstruct geometry and surfaces, build photorealistic appearance, represent dynamic motion and long-term historical states, and compile efficient runtime scenes for walking, flying, drone-style cameras, timeline exploration, XR or rendered video.

WRE is not defined by one algorithm or representation. It combines interchangeable specialist solvers behind stable adapters and selects routes according to the input data, requested quality and hardware budget. Explicit surface geometry, photorealistic splats/radiance, materials, sky/environment, dynamics, chronology and runtime LOD are separate responsibilities that can be combined without being conflated.

The engine is independent of MONDE. Future external integration is optional and does not define the product or its roadmap.

## Canonical blueprint and execution context

Start with [`docs/00_START_HERE.md`](docs/00_START_HERE.md), then follow its canonical reading order. The core documents include:

1. [`docs/01_PRODUCT.md`](docs/01_PRODUCT.md) — canonical product definition;
2. [`docs/02_ARCHITECTURE.md`](docs/02_ARCHITECTURE.md) — system layers, domain contracts and router architecture;
3. [`docs/03_PIPELINE.md`](docs/03_PIPELINE.md) — adaptive route graph for different data profiles;
4. [`docs/14_TECHNOLOGY_SELECTION.md`](docs/14_TECHNOLOGY_SELECTION.md) — technology and benchmark landscape;
5. [`docs/15_PRODUCTION_RUNTIME.md`](docs/15_PRODUCTION_RUNTIME.md) — artifact DAG, scheduling, runtime compilation, LOD and professional production behavior;
6. [`docs/27_RESEARCH_CANDIDATE_COVERAGE.md`](docs/27_RESEARCH_CANDIDATE_COVERAGE.md) — living, non-frozen research-candidate coverage map;
7. [`docs/28_PERFORMANCE_OPTIMIZATION_PLAYBOOK.md`](docs/28_PERFORMANCE_OPTIMIZATION_PLAYBOOK.md) — living performance/compression/execution strategy playbook;
8. [`docs/29_PERFORMANCE_INTEGRATION_MAP.md`](docs/29_PERFORMANCE_INTEGRATION_MAP.md) — living ownership and activation map for advanced performance work;
9. [`docs/23_ENGINEERING_EXECUTION.md`](docs/23_ENGINEERING_EXECUTION.md) — deny-by-default, one-run work-item and integration protocol;
10. [`docs/24_SYSTEM_INVARIANTS.md`](docs/24_SYSTEM_INVARIANTS.md) — cross-cutting product invariants;
11. [`docs/25_V2_ROADMAP.md`](docs/25_V2_ROADMAP.md) — executable V2 capability sequence;
12. [`docs/26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`](docs/26_V2_PRODUCT_ARCHITECTURE_FREEZE.md) — frozen V2 product/architecture authority and V1 donor-only policy.

The **V2 product/architecture is authoritative and frozen unless an explicit blueprint-change migration is approved**. Scientific solvers/models and execution/performance candidates are intentionally not frozen: before an integration or promotion work item activates, the candidate/dependency landscape is refreshed and relevant candidates/execution profiles are compared under stable WRE contracts.

V1 is not a compatibility target and does not need to be finished. Legacy code is only an implementation donor and regression/reference source when it cleanly satisfies a V2 contract.

Canonical implementation state is in [`PROJECT_STATE.yaml`](PROJECT_STATE.yaml) and the split machine roadmap rooted at [`registry/work-items.yaml`](registry/work-items.yaml). Planned future items may remain concise, but an item cannot become active until its complete deny-by-default contract is written and its scope passes the one-run complexity gate.

## Core principles

- **Reuse before implementation.** Mature solvers, renderers, trackers and codecs remain behind adapters.
- **Separate representations by responsibility.** Geometry, physical surface, appearance, materials, environment, dynamics, history and runtime assets are not one interchangeable object.
- **Route by data profile.** A sparse photo set, thousand-image Internet collection, short 4D event, long phone video, drone capture and century-scale chronology should not be forced through one chain.
- **Measure quality per dimension.** Rendering quality does not substitute for geometry quality; temporal consistency is evaluated separately again.
- **Preserve provenance classes.** Reconstructed, inferred and generated content remain distinguishable.
- **Build production infrastructure as part of the product.** Artifact caching, resumability, model/benchmark registries, quality gates, human review, compression and runtime compilation are first-class concerns.
- **Make algorithms replaceable.** New research normally changes the candidate shortlist, benchmark evidence, execution profile or default adapter rather than the frozen product architecture.
- **Refresh before integration.** The method named in an old roadmap/conversation is never assumed to remain the best; exact code/checkpoint/license/hardware and current alternatives are rechecked immediately before integration.
- **Measure before specializing.** Hardware/vendor-specific acceleration is introduced only under the owning capability after a reference path and representative profiling/quality evidence exist.
- **Keep work run-sized.** One work item should normally be one coherent PR a coding agent can implement, test and review in one development run; split independent integrations or failure domains before activation.

## Current development bootstrap

WRE currently uses Python 3.12 and `uv` for the lightweight development environment.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Fast pull-request checks intentionally exclude heavyweight GPU/reconstruction benchmarks. Native/external integrations and heavy quality benchmarks use dedicated lanes when the owning work item requires them.
