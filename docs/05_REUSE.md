# Reuse and dependency policy

## Decision rule

Before implementing an algorithm:

1. Search `registry/dependencies.yaml`.
2. Inspect maintained upstream implementations and current APIs.
3. Check license, platform support, maintenance and reproducibility.
4. Prefer an adapter around a proven implementation.
5. If custom implementation is still justified, write an ADR before coding it.

## Current candidate stack

- **COLMAP** — point features/matching, two-view geometry, incremental/global/hierarchical SfM, bundle adjustment, MVS.
- **LIMAP** — structural point+line/plane/vanishing-point/wireframe reconstruction.
- **GTSAM** — world factor graph and incremental optimization.
- **TEASER++** — robust 3D/Sim3 fragment registration where applicable.
- **Open3D** — ICP/GICP refinement and geometry utilities.
- **OpenMVG / AliceVision / OpenSfM** — selective validation/adapters when they provide a distinct capability.
- **FFmpeg** — video demux/decode/keyframe support.
- **PROJ / GeographicLib** — coordinate/geodesy utilities when needed.

No candidate becomes a core dependency until its lot decides exact version, license status and integration mode.

## Development dependencies

Development tooling is reproducible through the committed `uv.lock`. CI must use frozen sync rather than silently resolving a new environment. Dependabot updates the native `uv` ecosystem and GitHub Actions on a grouped weekly cadence so maintenance remains visible without generating excessive PR noise.

Supply-chain controls and settings that are deliberately outside the fast lane are documented in [`08_SECURITY.md`](08_SECURITY.md).

## Licensing

The repository's final project license has not yet been selected. Until it is, do not make a copyleft dependency mandatory/core without an explicit architecture/license decision. Optional subprocess adapters must still be reviewed for distribution implications.
