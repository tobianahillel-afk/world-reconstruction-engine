# World Reconstruction Engine (WRE)

WRE is an independent visual spatiotemporal reconstruction engine for turning arbitrary real-world photos and videos into high-fidelity, freely navigable 3D/4D scenes.

The product is designed to organize unordered media, recover spatial and temporal relationships, reconstruct geometry and surfaces, build photorealistic appearance, represent dynamic motion and long-term historical states, and compile efficient runtime scenes for walking, flying, drone-style cameras, timeline exploration, XR or rendered video.

WRE is not defined by one algorithm or representation. It combines interchangeable specialist solvers behind stable adapters and selects routes according to the input data, requested quality and hardware budget. Explicit surface geometry, photorealistic splats/radiance, materials, sky/environment, dynamics, chronology and runtime LOD are separate responsibilities that can be combined without being conflated.

The engine is independent of MONDE. Future external integration is optional and does not define the product or its roadmap.

## Canonical blueprint

Start with:

1. [`docs/00_START_HERE.md`](docs/00_START_HERE.md)
2. [`docs/01_PRODUCT.md`](docs/01_PRODUCT.md) — canonical product definition
3. [`docs/02_ARCHITECTURE.md`](docs/02_ARCHITECTURE.md) — system layers, domain contracts and router architecture
4. [`docs/03_PIPELINE.md`](docs/03_PIPELINE.md) — adaptive route graph for different data profiles
5. [`docs/14_TECHNOLOGY_SELECTION.md`](docs/14_TECHNOLOGY_SELECTION.md) — current candidate technology/benchmark landscape
6. [`docs/15_PRODUCTION_RUNTIME.md`](docs/15_PRODUCTION_RUNTIME.md) — artifact DAG, scheduling, runtime compilation, LOD and professional production behavior

Canonical implementation state currently remains in [`PROJECT_STATE.yaml`](PROJECT_STATE.yaml) and [`registry/work-items.yaml`](registry/work-items.yaml). The product/architecture blueprint above is the target definition; a dedicated roadmap migration must reconcile the existing implementation sequence with it before old roadmap assumptions are treated as final product scope.

## Core principles

- **Reuse before implementation.** Mature solvers, renderers, trackers and codecs remain behind adapters.
- **Separate representations by responsibility.** Geometry, physical surface, appearance, materials, environment, dynamics, history and runtime assets are not one interchangeable object.
- **Route by data profile.** A sparse photo set, thousand-image Internet collection, short 4D event, long phone video, drone capture and century-scale chronology should not be forced through one chain.
- **Measure quality per dimension.** Rendering quality does not substitute for geometry quality; temporal consistency is evaluated separately again.
- **Preserve provenance classes.** Reconstructed, inferred and generated content remain distinguishable.
- **Build production infrastructure as part of the product.** Artifact caching, resumability, model/benchmark registries, quality gates, human review, compression and runtime compilation are first-class concerns.
- **Make algorithms replaceable.** New research enters the benchmark registry and can replace a default without redesigning the core.

## Current development bootstrap

WRE currently uses Python 3.12 and `uv` for the lightweight development environment.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Fast pull-request checks intentionally exclude heavyweight GPU/reconstruction benchmarks.