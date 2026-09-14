# World Reconstruction Engine (WRE)

WRE is a deterministic, evidence-driven engine for reconstructing and incrementally connecting places from images, video and spatial metadata, while preserving uncertainty and time-valid geometry when the physical world changes.

The project composes proven geometry/reconstruction libraries instead of reimplementing them. Observations, derived evidence and estimated geometry remain distinct; uncertain placements, fragment merges, absolute anchors and temporal changes may remain unresolved rather than being guessed. Local fragments may exist without Earth coordinates and acquire absolute placement later from GPS, ground-control points, known landmarks or validated reference fragments.

## Start here

**Agents and contributors must begin with [`docs/00_START_HERE.md`](docs/00_START_HERE.md).**

Canonical project state lives in [`PROJECT_STATE.yaml`](PROJECT_STATE.yaml). Planned work and acceptance criteria live in [`registry/work-items.yaml`](registry/work-items.yaml).

No reconstruction implementation should be started before the bootstrap and development-quality gates in L0 are complete.

## Development bootstrap

WRE uses Python 3.12 and `uv` for the lightweight development environment.

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Pull requests run the same fast checks automatically. Heavy reconstruction/GPU tests are intentionally excluded from the fast lane.
