# World Reconstruction Engine (WRE)

WRE is a deterministic, evidence-driven engine for reconstructing and incrementally connecting places from images, video and spatial metadata.

The project composes proven geometry/reconstruction libraries instead of reimplementing them. Derived geometry remains auditable: observations, evidence and estimated geometry are kept distinct; uncertain placements and fragment merges may remain unresolved rather than being guessed.

## Start here

**Agents and contributors must begin with [`docs/00_START_HERE.md`](docs/00_START_HERE.md).**

Canonical project state lives in [`PROJECT_STATE.yaml`](PROJECT_STATE.yaml). Planned work and acceptance criteria live in [`registry/work-items.yaml`](registry/work-items.yaml).

No reconstruction implementation should be started before the bootstrap and development-quality gates in L0 are complete.
