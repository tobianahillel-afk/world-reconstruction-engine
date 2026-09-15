# Production, orchestration and runtime blueprint

## Why this document exists

Research code demonstrates algorithms. A production reconstruction engine must also schedule work, reuse artifacts, recover from failure, compare alternatives, expose review tools, compile runtime assets and remain reproducible months later.

This document defines the non-algorithmic systems required for WRE to behave like a professional platform rather than a collection of scripts.

## Project manifest

Every reconstruction lives inside a `SceneProject` manifest containing at least:

- immutable project identifier;
- source-media references and hashes;
- project-level coordinate-frame conventions;
- requested quality mode;
- user overrides;
- active scene clusters and temporal groups;
- artifact graph roots;
- master-scene versions;
- runtime exports;
- warnings/unresolved states.

The manifest is append/version oriented. Manual actions must not become undocumented hidden state.

## Artifact graph and cache

Every expensive stage produces an artifact with a stable key derived from:

- exact input artifact hashes;
- adapter identity and version;
- model/checkpoint hash where applicable;
- normalized configuration;
- producer software version;
- optional hardware-sensitive fields when output can materially differ.

Artifacts are content-addressed where practical. Identical work is reused across reruns and quality modes.

Example:

```text
MediaAsset
  -> FrameSelection
  -> SceneGraph
  -> TrackSet
  -> CameraSolution
  -> DepthSolution
  -> SurfaceModel
  -> AppearanceModel
  -> MasterScene
  -> RuntimeScene[web]
  -> RuntimeScene[xr]
```

## Job model

A pipeline node becomes one or more jobs with explicit resource declarations:

- CPU cores;
- GPU class/count;
- minimum/target VRAM;
- RAM;
- scratch storage;
- expected output storage;
- estimated duration class;
- network/download requirements;
- checkpoint/resume capability.

The scheduler chooses execution resources from declarations and available hardware instead of allowing each adapter to assume a workstation layout.

## Resource estimator

Before launching heavy work, the engine should estimate:

- frame/image count after selection;
- expected feature/match count;
- pair graph size;
- GPU memory envelope;
- disk/scratch requirement;
- likely runtime class;
- whether chunking or downscaling is required.

The estimator may be empirical and calibration-driven. Its goal is to avoid predictable out-of-memory or disk failures and to choose sensible route variants.

## Checkpoint and resume

Long jobs should expose resumability whenever the underlying engine permits it. The orchestration layer records:

- checkpoint path/hash;
- last completed substage;
- adapter version/configuration;
- reason for interruption;
- compatibility rules for resume.

A retry must not silently resume from an incompatible checkpoint after a model/configuration change.

## Model and dependency registry

Every learned or native dependency used in production has registry metadata:

- canonical name;
- source repository/project;
- exact version/commit;
- checkpoint identifier/hash;
- direct license;
- known transitive-license notes;
- supported hardware/runtime;
- reproducibility notes;
- responsible adapter;
- benchmark status;
- shipping status: approved / experimental / benchmark-only / blocked.

This registry prevents “best paper” from automatically becoming “ship it”.

## Benchmark registry

Benchmark results are stored, not copied into prose only.

A benchmark record should include:

- fixture/data profile;
- exact adapter/model/configuration;
- quality mode;
- geometry metrics;
- rendering metrics;
- temporal metrics when applicable;
- latency;
- VRAM/RAM;
- storage size;
- failures/warnings;
- environment/hardware;
- comparison baseline.

Default selection policy should be data driven from these records.

## Failure taxonomy

Adapters should map raw failures to stable categories such as:

- unsupported input;
- dependency unavailable;
- insufficient overlap;
- camera ambiguity;
- calibration failure;
- geometric inconsistency;
- dynamic contamination;
- depth inconsistency;
- memory exhaustion;
- timeout;
- corrupted media;
- checkpoint incompatibility;
- quality-gate failure;
- unknown internal error.

The router uses failure categories to choose fallbacks instead of parsing arbitrary logs.

## Quality-gate service

Quality evaluation is a first-class service used by routes.

It receives artifact(s) plus a metric policy and returns:

- metric vector;
- thresholds/policy revision;
- `PASS`, `ACCEPT_WITH_WARNINGS`, `RETRY`, `ESCALATE`, or `UNRESOLVED`;
- human-readable reasons;
- machine-readable escalation hints.

This service makes quality behavior consistent across adapters.

## Human review UI

Professional workflows need correction tools. The assisted/expert UI should eventually support:

- media/frame rejection;
- scene-cluster split/merge;
- visual pair inspection;
- camera/track inspection;
- mask correction;
- reconstruction-region crop;
- surface cleanup;
- temporal grouping correction;
- change-event approval/rejection;
- generated-region visibility/provenance inspection;
- comparison of competing solver outputs;
- route and parameter overrides.

Every accepted manual edit becomes a versioned input artifact.

## Scene versioning

A `MasterScene` version references immutable component artifacts. New information creates a new version rather than destructively rewriting the previous master.

Version reasons include:

- additional media;
- new temporal epoch;
- improved solver/model;
- corrected human review;
- higher quality mode;
- updated materials/appearance;
- bug fix requiring recomputation.

This allows A/B comparison and reproducibility.

## Coordinate conventions

A dedicated convention layer prevents common reconstruction bugs.

It defines and tests:

- handedness;
- axis directions/up axis;
- world/local frames;
- camera-to-world versus world-to-camera transforms;
- image coordinate conventions;
- pixel-center convention;
- normalized-device/render conventions;
- metric scale status;
- conversions for OpenCV/COLMAP/OpenGL/glTF/game engines.

Conversions are explicit functions with regression fixtures, never scattered sign flips.

## Color and photometric pipeline

Input media may contain sRGB, HDR, RAW-derived, phone-computational or unknown color processing. The system should track:

- source color profile when known;
- decode transfer function;
- exposure/white-balance estimates;
- clipping/saturation masks;
- photometric normalization artifacts;
- working color space;
- output/display transform.

Appearance training should not accidentally treat incompatible camera pipelines as identical radiance observations.

## Camera/calibration subsystem

Camera models are explicit and extensible:

- pinhole;
- radial/tangential distortion;
- fisheye;
- equirectangular/360;
- rolling shutter;
- variable focal length/zoom;
- multi-camera rigs;
- optional IMU priors.

Media profiling selects candidate models; model selection remains an auditable artifact.

## Master representation versus exchange/runtime formats

The product distinguishes:

1. **Master representation** — maximum retained information and internal links.
2. **Exchange format** — interoperable files for external tools.
3. **Runtime format** — aggressively optimized for a target device.

Examples of exchange/runtime families can include glTF/GLB, PLY, standard camera files, SPZ/SOG-class splat formats and engine-specific packages. No single external format is expected to preserve every internal artifact.

## Runtime compiler

The runtime compiler receives a `MasterScene` and a target profile.

Target profile fields include:

- platform: web / desktop / mobile / XR / Unreal / Unity / offline;
- memory budget;
- GPU budget;
- display resolution/FPS target;
- maximum download size;
- desired temporal range;
- collision/physics requirements;
- quality/latency preference.

The compiler can produce:

- simplified mesh;
- collision mesh/voxels/navmesh;
- compressed splat hierarchy;
- texture/material atlases;
- environment maps;
- spatial chunk tree;
- temporal chunk tree;
- LOD metadata;
- occlusion/culling structures;
- streaming manifest;
- camera/timeline metadata.

## LOD strategy

LOD is representation specific but coordinated at scene level.

### Geometry LOD

- mesh decimation;
- geometry impostors where appropriate;
- collision simplification separate from visual simplification.

### Appearance LOD

- hierarchical splat selection;
- appearance resolution/SH reductions;
- texture mipmaps/atlas resolution.

### Temporal LOD

- lower sample rate for distant/low-importance dynamic elements;
- compressed key states for long chronology;
- selective loading of historical epochs.

The runtime should obey a global visible-memory budget rather than loading the entire master scene.

## Streaming

Scene chunks should be prioritized by:

- camera distance/frustum;
- predicted movement;
- screen-space contribution;
- requested time/epoch;
- object importance;
- current memory pressure.

Prefetching can follow camera velocity and timeline scrubbing direction.

## Physics and navigation

Photorealistic splats/radiance fields do not automatically provide usable physics. Runtime scenes may therefore include separate:

- collision geometry;
- walkable surfaces;
- navmesh;
- interaction proxies;
- dynamic collision bounds.

These assets link back to surface geometry and can be lower resolution than the visual representation.

## Viewer modes

The reference viewer/runtime should eventually support:

- orbit inspection;
- first-person walk;
- free-fly/drone camera;
- saved cinematic paths;
- timeline scrubbing;
- epoch switching;
- dynamic playback/pause/frame-step;
- provenance/confidence visualization for expert mode;
- strict/realistic/cinematic completion profile;
- optional XR.

## Automated publication/export

A successful run may compile:

- standalone interactive scene;
- web-hostable scene bundle;
- game-engine package;
- glTF/mesh/material package;
- splat package;
- rendered camera-path video;
- reconstruction report with metrics;
- reproducibility manifest.

## Security and untrusted media

Media and external model/tool execution are untrusted inputs. Production design should isolate parsers/native tools where practical, enforce file-size/resource limits and avoid shell-string construction from user-controlled paths.

Remote model/checkpoint downloads should be pinned and verified.

## CI versus heavy evaluation

Fast CI validates contracts, deterministic fixtures, adapters with lightweight mocks/fixtures, state schemas and regression invariants.

Heavy GPU/real-scene benchmarks run in dedicated workflows or benchmark infrastructure and publish retained results. Shipping-default changes require appropriate benchmark evidence rather than only unit tests.

## Professional-system success criteria

A production-quality WRE should make these operations routine:

- resume a project months later and understand every artifact;
- add new media without rebuilding unrelated work;
- swap one solver for a better one;
- compare two complete reconstruction routes objectively;
- reproduce an old result from pinned inputs/configuration;
- recover from interruption;
- inspect why a route failed;
- manually correct a bad automatic decision without forking the pipeline;
- compile the same master reconstruction for different runtime targets;
- upgrade runtime compression/LOD without rerunning geometry reconstruction.

These capabilities are part of the product, not infrastructure left for after the reconstruction algorithms are finished.