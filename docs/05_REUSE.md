# Reuse and dependency policy

## Decision rule

Before implementing an algorithm:

1. Search `registry/dependencies.yaml`.
2. Inspect maintained upstream implementations and current APIs.
3. Check license, model/weight license where applicable, platform support, maintenance and reproducibility.
4. Prefer an adapter around a proven implementation.
5. If custom implementation is still justified, write an ADR before coding it.

## Current candidate stack

### Canonical geometry / reconstruction

- **COLMAP** — point features/matching, two-view geometry, incremental/global/hierarchical SfM, bundle adjustment, MVS, sequential/spatial/vocabulary/transitive candidate workflows and incremental image registration.
- **LIMAP** — structural point+line/plane/vanishing-point/wireframe reconstruction.
- **GTSAM** — world factor graph and incremental optimization.
- **TEASER++** — robust 3D/Sim3 fragment registration where applicable.
- **Open3D** — ICP/GICP refinement and geometry utilities.
- **OpenMVG / AliceVision / OpenSfM** — selective validation/adapters when they provide a distinct capability; OpenSfM patterns are particularly relevant to GPS/GCP/checkpoint handling.

### Media / coordinates

- **FFmpeg** — video demux/decode and media probing; keyframe policy remains WRE orchestration.
- **PROJ / GeographicLib** — coordinate-reference-system and geodesy utilities when needed.

### Robust world-graph / video specialists

- **Kimera-RPGO** — optional robust pose-graph consistency/outlier rejection; it complements rather than replaces GTSAM's primary world graph.
- **Basalt** — optional visual/visual-inertial odometry and trajectory path for video/IMU work.
- **Kalibr** — optional multi-camera, camera-IMU and rolling-shutter calibration.

### Difficult visual matching / localization

Prefer capabilities already available through the chosen COLMAP version before adding another stack. When a genuinely difficult Internet/archive-image case requires more, candidates include:

- **hloc (Hierarchical-Localization)** — optional retrieval/localization/mapping orchestration;
- **LightGlue** — optional adaptive learned matching for candidate correspondences.

These learned tools propose evidence; they do not bypass WRE geometric verification. Their framework/code licenses do not automatically cover every bundled extractor/model weight. Any concrete configuration must receive a per-model license review before becoming mandatory.

### Temporal / 4D change

- **py4dgeo** — planned L15 adapter for multi-epoch 3D/4D point-cloud change measurement (including M3C2-family workflows). WRE consumes its measurements as derived evidence and owns temporal hypotheses/state validity.

Do not reimplement mature 3D/4D change-measurement algorithms merely to own the code.

### Optional visualization

- **Nerfstudio** — optional photorealistic NeRF/Gaussian-splat-style visualization/view synthesis. It may consume WRE/COLMAP geometry, but its render representation is never canonical WRE world geometry.

## Reuse matrix

| Capability | Preferred existing project | WRE responsibility |
|---|---|---|
| point features / SfM / BA / MVS | COLMAP | adapter, provenance, orchestration, evidence policy |
| structural points+lines/planes/VP | LIMAP | adapter and solver-independent import |
| video decode/probe | FFmpeg | deterministic ingest/keyframe orchestration |
| candidate retrieval | COLMAP mechanisms + WRE graph/time/GPS signals | deterministic candidate/routing policy |
| difficult learned proposals | COLMAP integrated options first; optional hloc/LightGlue | proposal only, geometric verification required |
| robust Sim3 fragment registration | TEASER++ | candidate generation, evidence, acceptance/reversibility |
| ICP/GICP refinement | Open3D | orchestration and acceptance evidence |
| world factor graph | GTSAM | factor contracts, provenance, uncertainty |
| robust graph outlier rejection | optional Kimera-RPGO | audit/adapter, not hidden authority |
| GPS/GCP/checkpoints | GTSAM/OpenSfM patterns + PROJ/GeographicLib | solver-independent anchor constraints |
| VIO/trajectory | optional Basalt | adapter and evidence import |
| camera/IMU/rolling-shutter calibration | optional Kalibr | adapter and provenance |
| multi-epoch change measurement | py4dgeo | temporal evidence import, hypotheses/states |
| photorealistic visualization | optional Nerfstudio | display/export only; never canonical geometry |

No candidate becomes a core dependency until its owning work item decides exact version, license status, integration mode and acceptance tests.

## Development dependencies

Development tooling is reproducible through the committed `uv.lock`. CI must use frozen sync rather than silently resolving a new environment. Dependabot updates the native `uv` ecosystem and GitHub Actions on a grouped weekly cadence so maintenance remains visible without generating excessive PR noise.

Supply-chain controls and settings that are deliberately outside the fast lane are documented in [`08_SECURITY.md`](08_SECURITY.md).

## Licensing

The repository's final project license has not yet been selected. Until it is, do not make a copyleft dependency mandatory/core without an explicit architecture/license decision. Optional subprocess adapters must still be reviewed for distribution implications.

For learned computer-vision components, review code, model weights and any upstream feature extractor separately. A framework being Apache/BSD/MIT does not prove every bundled checkpoint is safe for the intended distribution.
