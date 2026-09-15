# Architecture blueprint

## Architectural objective

WRE is a modular reconstruction platform, not a fixed sequence of algorithms. Its architecture separates responsibilities, representations and quality criteria so the best available specialist can be selected for each dataset without coupling the product to one research method.

The organizing rule is:

> stable contracts and artifacts in the core; replaceable solvers and models behind adapters.

The core owns orchestration, artifact lifecycle, routing, metrics, provenance classes, quality gates, temporal organization, runtime compilation and user-visible project state. Specialist libraries own their algorithms.

## System layers

WRE is divided into six conceptual layers.

### 1. Observation layer

Stores original media and directly extracted facts:

- photos and videos;
- decoded frames;
- EXIF, timestamps and available calibration metadata;
- audio tracks for synchronization;
- source hashes and provenance;
- image-quality measurements.

Observation data is immutable. Derived artifacts never silently replace original evidence.

### 2. Organization and evidence layer

Builds relationships between observations:

- visual retrieval and same-scene hypotheses;
- local feature matches;
- dense tracks and optical flow;
- temporal synchronization hypotheses;
- scene/epoch clustering;
- dynamic/static masks;
- camera and depth priors;
- specialist data-profile classification.

This layer answers “what likely belongs together?” before committing to one reconstruction.

### 3. Geometry layer

Represents where things are:

- camera solutions and calibration;
- point maps and sparse landmarks;
- depth fields;
- static spatial geometry;
- local/global transforms;
- dense reconstructions;
- confidence and residual metrics.

Geometry may come from feed-forward models, SfM/MVS, SLAM, learned matching, hybrid global solvers or consensus between methods. Core contracts must not depend on solver-private structures.

### 4. Scene representation layer

Stores complementary world representations rather than collapsing everything into one asset:

- explicit surface: mesh, SDF, surfels or equivalent;
- photorealistic appearance: Gaussian/radiance/neural representation;
- physical materials: albedo, roughness, metallic, normals when recovered;
- environment: sky, horizon and distant illumination;
- persistent dynamic objects;
- motion/deformation fields;
- temporal states and change events;
- uncertainty fields;
- optional generative completion.

Each representation has a declared use. For example, splats may drive appearance while a mesh drives collision and measurement.

### 5. Production/orchestration layer

Turns research components into a reliable product:

- project manifests;
- artifact DAG and content-addressed cache;
- adapter/model registry;
- routing and fallback graph;
- resource estimation and scheduling;
- checkpoint/resume;
- benchmark registry;
- quality gates;
- failure classification;
- scene versioning;
- human-review state;
- reproducibility manifests.

### 6. Runtime layer

Compiles master scenes into device-oriented assets:

- spatial chunks;
- temporal chunks;
- LOD hierarchies;
- compressed splats/textures/meshes;
- collision geometry/navmesh;
- streaming metadata;
- camera paths and timeline controls;
- web/game/XR/offline-render packages.

The runtime does not need to carry every master artifact.

## Canonical domain contracts

The exact implementation may evolve, but the architecture requires equivalents of the following concepts.

### Project and media

- `SceneProject` — user/project boundary and retained configuration.
- `MediaAsset` — immutable source image/video.
- `FrameObservation` — selected video frame with timing and source link.
- `MediaProfile` — quality, camera class, 360/drone/blur/dynamic indicators.

### Organization

- `SceneCluster` — observations hypothesized to depict one connected scene.
- `TemporalGroup` — same event, epoch or temporally related observations.
- `PairCandidate` — proposed relationship between observations.
- `CorrespondenceSet` — sparse or dense matches/tracks with producer metadata.
- `SyncHypothesis` — offset/clock relationship between videos.

### Geometry

- `CameraSolution` — intrinsics, extrinsics, coordinate frame, uncertainty and metrics.
- `DepthField` — per-pixel depth plus confidence/validity.
- `PointMap` — dense or sparse 3D point predictions.
- `GeometrySolution` — one coherent reconstruction hypothesis.
- `SurfaceModel` — explicit physical surface.

### Appearance and environment

- `AppearanceModel` — photorealistic radiance/splat representation.
- `MaterialModel` — optional physical material decomposition.
- `EnvironmentModel` — sky/environment lighting representation.

### Dynamics and history

- `DynamicEntity` — persistent moving/deforming object.
- `Trajectory` — object/camera path through time.
- `MotionField` — continuous or sampled motion/deformation.
- `TemporalState` — long-term scene state valid for an interval or epoch.
- `ChangeEvent` — supported structural/semantic transition between states.

### Confidence and generation

- `ConfidenceField` — spatial/temporal quality or uncertainty.
- `CompletionArtifact` — generated or inferred visual completion with explicit provenance class.

### Runtime

- `MasterScene` — highest-fidelity retained scene assembly.
- `RuntimeScene` — compiled target-specific representation.
- `LODChunk` — spatial/temporal streamable unit.

## Provenance classes

Every artifact or visible element that can be confused with reality must declare one of three top-level provenance classes:

- `OBSERVED_RECONSTRUCTED` — directly supported by observations plus reconstruction;
- `INFERRED` — model-derived estimate with confidence/uncertainty;
- `GENERATED` — synthesis used for visual completion or cinematic output.

This classification is orthogonal to quality. A beautiful generated view is still generated.

## Artifact DAG

All expensive work is represented as a directed acyclic graph of immutable/versioned artifacts.

```text
MediaAsset
  -> FrameSet
  -> MediaProfile
  -> RetrievalIndex / SceneGraph
  -> CorrespondenceArtifacts
  -> CameraSolution / DepthSolution
  -> GeometrySolution
  -> SurfaceModel
  -> AppearanceModel / MaterialModel / EnvironmentModel
  -> TemporalSolution
  -> MasterScene
  -> RuntimeScene
```

An artifact key should include content hashes, configuration, adapter/model/checkpoint identity and producer version. Identical keys are reusable. Heavy stages should expose checkpoints where the underlying engine permits resume.

## Adapter architecture

External engines are integrated through capability-oriented adapters rather than through product-wide dependencies on their data structures.

Examples of adapter families:

- retrieval adapters;
- feature/matcher adapters;
- dense tracker adapters;
- camera/depth foundation-model adapters;
- SfM/SLAM/global mapper adapters;
- MVS/depth-fusion adapters;
- static/dynamic surface adapters;
- Gaussian/radiance adapters;
- 4D reconstruction adapters;
- relighting/material adapters;
- temporal-change adapters;
- compression/LOD/runtime adapters.

An adapter declares:

- supported input contract;
- produced output contract;
- hardware requirements;
- license/model provenance metadata;
- deterministic/reproducibility guarantees where applicable;
- metrics it can expose;
- failure classes;
- resumability semantics.

## Router architecture

Routing is based on data profile, requested quality, hardware budget, existing artifacts and quality-gate outcomes.

The router can distinguish scenarios such as:

- unordered static photo collections;
- sparse-view static scenes;
- very large Internet collections;
- short monocular dynamic video;
- long streaming video;
- multi-video synchronized events;
- 360/equirectangular capture;
- drone capture;
- historical multi-epoch media;
- high-blur/HDR/fisheye/rolling-shutter specialist cases.

A route is a graph, not necessarily a simple chain. Competing methods can run in parallel for high-quality modes and be compared or fused.

## Quality modes and routing policy

- `PREVIEW` prioritizes latency and dataset diagnosis.
- `FAST` targets useful interactive output with bounded resource use.
- `QUALITY` increases refinement, specialist use and holdout validation.
- `MASTER` may run multiple solvers, dense/surface refinement, advanced appearance, temporal checks and expensive quality gates.

The quality mode is not a truth label. Generated completion remains generated in all modes.

## Automatic quality gates

Each major stage should publish metrics and a gate decision.

### Camera/geometry gates

Candidate metrics:

- registered-view ratio;
- reprojection error;
- pose consistency;
- track length/distribution;
- baseline/coverage distribution;
- cycle consistency;
- depth multi-view consistency;
- scale/drift metrics where meaningful.

### Surface gates

Candidate metrics:

- Chamfer/point-to-surface error on fixtures;
- normal consistency;
- watertightness where required;
- hole/coverage measures;
- collision usability.

### Appearance gates

Candidate metrics:

- holdout PSNR/SSIM/LPIPS;
- perceptual quality;
- floater/ghosting indicators;
- disagreement with explicit geometry;
- temporal flicker for dynamic scenes.

### Dynamic/temporal gates

Candidate metrics:

- dense tracking error;
- trajectory consistency;
- temporal geometry consistency;
- object persistence through occlusion;
- change-vs-registration confidence;
- state-boundary uncertainty.

Gate outcomes can be `PASS`, `ACCEPT_WITH_WARNINGS`, `RETRY`, `ESCALATE` or `UNRESOLVED`.

## Static versus dynamic decomposition

The architecture distinguishes:

- stable background/structure;
- semi-static elements such as vegetation, scaffolding and signs;
- dynamic rigid objects;
- articulated/deformable objects;
- environment/sky;
- transient distractors that should not contaminate static reconstruction.

This classification can be uncertain and revised as more observations arrive.

## Temporal architecture

### Event-time 4D

Short/continuous events use synchronized time, tracks, motion/deformation and persistent object identities. Output may include dynamic surfaces and photorealistic dynamic appearance.

### Chronological time

Long-term changes use `TemporalState` plus `ChangeEvent`. States may overlap in uncertainty or have approximate validity intervals. Structural changes are not forced into smooth deformation.

### Mixed time

A historical state may itself contain a short dynamic event. Temporal representation is therefore hierarchical rather than one global scalar animation curve.

## Geometry versus appearance

WRE explicitly supports dual representation:

```text
MasterScene
  |- Physical representation
  |    |- depth / point maps
  |    |- surface / mesh / SDF
  |    `- collision / navigation
  |
  `- Visual representation
       |- Gaussian/radiance appearance
       |- materials
       `- environment / lighting
```

The renderer can combine these layers while the evaluator measures them separately.

## Human-in-the-loop architecture

The project state supports three interaction levels:

- `AUTO` — router and quality gates decide automatically;
- `ASSISTED` — user can correct clusters, masks, cameras, temporal assignments and regions;
- `EXPERT` — artifact graph, adapter selection and specialist parameters are inspectable/overrideable.

Manual edits become versioned artifacts rather than destructive hidden state.

## Runtime compiler

A dedicated compiler converts `MasterScene` to target-specific `RuntimeScene` assets. It may perform:

- mesh simplification and collision generation;
- texture/material baking;
- splat compression;
- spatial chunking;
- temporal chunking/compression;
- LOD generation;
- occlusion/culling metadata;
- navmesh generation;
- platform/device budgets;
- packaging for web, desktop, game engines or XR.

Master quality and runtime performance are therefore independent optimization targets.

## Technology selection policy

The architecture never hard-codes “the best model” into the product definition. Current candidates live in a technology landscape/benchmark registry. A new method can become preferred when it wins the relevant internal benchmark and passes license, reproducibility and integration criteria.

A candidate is evaluated per responsibility rather than by one overall leaderboard score.

## Boundary with MONDE

WRE must remain independently useful. If a future MONDE integration exists, it consumes explicit scene/observation/temporal outputs through optional contracts. MONDE-specific identity, belief-state or planetary world-model semantics do not define WRE's internal product architecture.