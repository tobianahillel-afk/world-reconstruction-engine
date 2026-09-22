# Technology selection and benchmark blueprint

## Purpose

This document captures the current technology landscape without turning transient 2026 research winners into permanent architecture. The product core depends on capability contracts; candidates are benchmarked behind adapters.

A candidate can be:

- **baseline** — stable reference that every replacement must beat or complement;
- **primary candidate** — preferred technology to evaluate for a responsibility;
- **specialist** — invoked only for a data profile/failure class;
- **research watch** — promising but not sufficiently released, licensed, mature or reproducible;
- **runtime/tooling** — production infrastructure rather than a reconstruction solver.

No entry is automatically approved for shipping. Version, checkpoint, license, transitive dependencies, hardware support and reproducibility are reviewed by the owning integration item.

## Evaluation dimensions

Every serious candidate is scored separately on relevant dimensions:

- geometric fidelity;
- camera/pose accuracy;
- visual fidelity;
- temporal consistency;
- robustness to in-the-wild inputs;
- sparse-view robustness;
- dynamic-scene robustness;
- scalability in number of images;
- scalability in video duration;
- inference/training latency;
- peak VRAM/RAM;
- storage/output size;
- streaming/runtime performance;
- implementation maturity;
- reproducibility;
- integration complexity;
- license/product compatibility.

A single weighted score can help route selection, but never replaces the dimension vector.

## Recommended benchmark modes

The internal benchmark suite should include at least:

1. controlled static object/room with known cameras/geometry;
2. outdoor building with repeated structure;
3. sparse-view outdoor scene;
4. Internet/tourist collection with illumination and distractors;
5. very large unordered photo collection;
6. short dynamic monocular video;
7. long handheld/walking video;
8. multiple unsynchronized videos of one event;
9. 360/equirectangular sequence;
10. drone/aerial sequence;
11. strong blur/rolling-shutter/HDR specialist cases;
12. historical cross-epoch scene with real structural change;
13. reflective/material-sensitive scene;
14. runtime large-scene LOD/streaming fixture.

Holdout views and independent geometry references should be used where available.

## Media ingest and decoding

### Baseline / production choices

- FFmpeg — mature decode/transcode/audio extraction.
- PyAV or equivalent binding — programmatic media access.
- image libraries with explicit color/EXIF handling.

Do not build custom codecs.

### Accelerated still-image candidates

For large photo collections where representative profiling shows CPU decode/resize or host transfer is material, evaluate accelerated backends behind the same canonical decoded-image contract:

- NVIDIA nvImageCodec / nvJPEG-class batch or device-resident decode on compatible NVIDIA systems;
- CPU/reference decode remains the portability and semantic comparison path;
- future accelerated backends from other vendors may enter through the same execution-profile boundary.

Do not couple `FrameObservation` or canonical decoded-image identity to one vendor. Benchmark orientation/color behavior, pixel equivalence, CPU/GPU utilization, transfer overhead and end-to-end downstream reuse rather than decode throughput alone.

## Frame quality and keyframing

Capabilities required:

- blur/sharpness scoring;
- exposure/clipping assessment;
- duplicate/near-duplicate detection;
- visual diversity;
- motion/baseline estimation;
- adaptive keyframe density.

Start with deterministic metrics plus learned features where they add measurable value. Quality selection must preserve source timing and frame provenance.

## Visual retrieval / same-scene discovery

### Candidates

- SALAD and modern visual-place-retrieval embeddings;
- DINO-family features for generic semantic/visual similarity;
- classical vocabulary-tree retrieval as deterministic baseline;
- hloc-style retrieval + local verification compositions.

### First V2 learned candidate

V2L7.5 evaluates SelaVPR++ Base with GeM and the standard floating-point global
descriptor path at exact upstream revision
`56bd921cbd3d53e9c5f91d0aafff147f95fb362a`. The candidate consumes verified
canonical decoded-image pyramids, uses an explicit local content-identified checkpoint,
and remains an optional external ML environment rather than a core dependency. The first
reference execution is CPU-only and exact-search-only; hashing, reranking, ANN backends,
geo-time inference and local/geometric verification remain separate responsibilities. The
runtime also verifies the exact reviewed Python-file set and SHA-256 identities for the
executable `model/` subtree before import, and rejects unreviewed bytecode/native-module
shadows so a dirty checkout cannot silently execute under the reviewed revision identity.

This is an experimental benchmark candidate, not a default. V2L7.6 must compare its
retrieval evidence with the sequential, GPS and classical vocabulary-tree sources before
any promotion. Direct upstream code/model licensing is recorded as MIT while transitive
environment and redistribution review remains pending in the dependency/model registry.

Retrieval proposes relationships. Repeated façades/symmetric scenes require local/geometric disambiguation before cluster fusion.

## Sparse/local matching

### Baselines

- SIFT / COLMAP matching.

### Primary modern candidates

- LightGlue with ALIKED/DISK/SuperPoint-family features as applicable;
- MASt3R-family matching/3D matching;
- RoMa/LoFTR-class dense or semi-dense matchers for difficult view changes.

### Disambiguation

- Doppelgangers++-class repeated-structure disambiguation.

The benchmark must measure false matches/false scene merges, not only recall.

## Dense tracking and motion evidence

### Primary candidates

- AllTracker — dense long-range tracking;
- CoTracker3 — robust point tracking baseline;
- modern continuous optical-flow approaches such as Optical Flow Matching when implementation maturity permits.

Dense tracks are shared infrastructure for camera solve, synchronization, dynamic decomposition and 4D reconstruction.

## Multi-video synchronization

Use an evidence hierarchy:

1. reliable timecode/metadata if present;
2. audio correlation;
3. visual event alignment;
4. temporal prototype/event alignment;
5. moving humans/objects as calibration signals where applicable.

Synchronization produces `SyncHypothesis` artifacts with confidence and residuals.

## Fast feed-forward geometry

### High-priority candidates

- Depth Anything 3 (DA3) family;
- VGGT-Ω;
- Pi3/Pi3X family;
- MapAnything-class general 3D foundation models;
- CUT3R for persistent/stateful reconstruction.

### Long/streaming candidates

- LongStream;
- ZipMap;
- Scal3R;
- SLAM3R/other dense real-time reconstruction systems.

These models are excellent for proposals, previews, priors and sometimes complete geometry solutions. Quality gates decide whether a precision/global solve is still required.

### First V2 feed-forward reference path

V2L13.2/V2L13.3 use **Depth Anything 3 DA3-BASE** only as the first bounded
PREVIEW integration reference, not as a benchmark winner or default route. The canonical
normalizer preserves the reviewed OpenCV world-to-camera pose direction, keeps DA3 relative
depth at unresolved local scale and emits no PointMap until a separate reviewed unprojection
semantic exists.

The V2L13.3 reference execution is deliberately conservative: exact source revision
`3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`, exact local DA3-BASE safetensors
checkpoint identity, verified `media.decoded_image_pyramid` inputs, CPU float32,
single-worker deterministic preprocessing and no network/download fallback. The runner uses
the bounded DA3-BASE config/model/input/output modules rather than the top-level export
facade. DA3's native depth confidence is `exp(x)+1`, not a calibrated probability, so this
reference path does not silently coerce it into canonical `[0,1]` confidence.

Source/checkpoint direct terms are recorded as Apache-2.0 while the optional external
runtime's transitive redistribution review remains pending. CUDA/autocast/compiled execution
and performance promotion are explicitly deferred to V2L13.6; camera/depth quality metrics
and comparison against the classical baseline remain V2L13.4/V2L13.5 responsibilities.

## Precision/global mapping

### Baselines

- COLMAP incremental/global/hierarchical mapping as supported by pinned versions.

### High-priority hybrid candidate

- GLUEMAP family: learned local geometry plus global geometric alignment/refinement.

### Supporting ecosystems

- hloc-style hierarchical localization;
- classical/global bundle adjustment and pose-graph tools behind adapters.

The precision layer owns global consistency, not the appearance renderer.

## Large unordered collections

### Candidates to benchmark

- VGG-T³-class linear/scalable feed-forward mapping;
- Scal3R;
- ZipMap;
- GLUEMAP with scalable retrieval/submaps;
- hierarchical/submap approaches built on mature SfM.

For retrieval/index execution at very large scale, compare exact CPU, CPU ANN, quantized ANN, GPU ANN and hybrid routes where relevant. The product contract is retrieval/index semantics and recall; no one CPU/GPU library is frozen.

License constraints may keep some methods benchmark-only until a compatible alternative is available.

## Sparse-view reconstruction

### Candidates

- DepthSplat;
- Sparse2DGS;
- GIFSplat;
- Free360 for extremely sparse unbounded/360-oriented scenarios;
- SPARS3R-class geometry-prior alignment.

Sparse-view systems must report uncertainty/coverage; generative priors cannot silently become measured geometry.

## Dense depth / MVS

### Baselines

- COLMAP PatchMatch and mature MVS/fusion components;
- OpenMVS-like mature external systems when license/product policy allows.

### Learned priors

- DA3/other metric depth and multi-view priors can initialize, regularize or fill difficult regions, subject to consistency checks.

## Static surface reconstruction

### Baselines

- depth fusion + Poisson/Delaunay/TSDF/SDF/mesh pipelines as appropriate.

### Hybrid research candidates

- 2DGS-derived geometry;
- OMeGa — joint explicit mesh and Gaussian optimization;
- SurfaceSplat — coupled surface/splat refinement;
- MeshSplatting-class hybrid representations.

The shipping surface should be chosen by geometry/runtime benchmark, not by render PSNR alone.

## Photorealistic static appearance

### Runtime/training foundation

- gsplat as a preferred low-level Gaussian rasterization/training backend when compatible.
- Nerfstudio/Splatfacto as a reusable research/production integration reference.

### In-the-wild appearance

- WildGaussians-class robust appearance models;
- robust Gaussian methods handling exposure/occluders/distractors.

### View-dependent effects

- higher-dimensional/view-dependent Gaussian/radiance methods such as 7DGS-class approaches where the use case justifies complexity.

### Motion/scale consistency references

- Mip-Splatting-class antialiasing and multi-scale consistency;
- StopThePop-class view-consistent sorting/rasterization;
- later motion-stable or XR-specific splat renderers that beat them under the same camera-path quality/runtime contract.

Appearance output is evaluated on held-out views, camera paths/scale changes where relevant, and against geometry artifacts.

## Dynamic/static decomposition

### Candidates

- DeGauss-class distractor/dynamic-static decomposition;
- segmentation + motion + depth fusion;
- SAM2/video segmentation and modern panoptic video segmentation as supporting tools.

The engine should distinguish static background, semi-static content, dynamic rigid objects, deformable objects and removable distractors.

## Fast 4D preview

### High-priority research candidates

- MoVieS;
- MoRe;
- D4RT-style unified dynamic reconstruction;
- UFO-4D-class sparse input methods when available.

The purpose is near-immediate 4D viability/preview, not automatic promotion to master truth.

## Quality 4D reconstruction

### High-priority candidates

- MotionScale;
- Shape of Motion;
- MoSca;
- ProDyG for online dynamic mapping;
- MOSAIC-GS and comparable long/dynamic systems as benchmarks.

### Persistent object reconstruction

- 4D Primitive-Mâché-class object-permanence pipelines.

### Uncertainty-aware 4D

- GP-4DGS-class probabilistic dynamic models — research watch until mature release.

## Dynamic surface

### Candidates

- 4DSurf;
- dynamic mesh/SDF pipelines;
- surface extraction from dynamic radiance representations when independently validated.

A dynamic splat render and a dynamic physical surface are separate products.

## Long dynamic sequences

### Candidates

- LongStream for streaming geometry;
- MotionScale/ProDyG-class dynamic reconstruction;
- ClipGStream/long-sequence systems where applicable;
- SLAMFormer-∞-class long-trajectory research for future evaluation.

Chunking, drift monitoring and loop/global refinement are required at product level.

## 360 / omnidirectional media

### Candidates

- PFGS360-class pose-free omnidirectional reconstruction;
- Free360 for sparse unbounded 360 view synthesis;
- native spherical/equirectangular camera handling in the ingest/calibration layer.

## Drone / aerial media

### Specialist candidates

- AeroGS-class pose-free dynamic UAV reconstruction;
- standard aerial SfM/MVS when the scene is mostly static and capture geometry supports it.

## Blur, rolling shutter, HDR and low light

Specialist routes should be adapter-driven:

- Deblur4DGS/MSCD-GS/BARD-GS-class blur-aware methods;
- rolling-shutter camera models/refinement;
- PhysHDR-GS/Mono4DGS-HDR-class HDR methods;
- low-light/dark-scene reconstruction specialists.

These methods are not default dependencies.

## Environment, lighting and inverse rendering

### Outdoor relighting/environment

- GaRe-class relightable outdoor Gaussian reconstruction;
- sky/environment-separated methods such as R3GW-class approaches.

### Materials

- MatSpray-class material fusion;
- physically-aware inverse-rendering methods such as IR-HGP-class approaches for difficult illumination/reflection cases.

Outputs should map to explicit material/environment contracts when possible.

## Historical chronology and evolving scenes

### Conceptual/algorithmic candidates

- Neural Scene Chronology — foundational model for viewpoint/illumination/time-controlled Internet-photo chronology;
- Cross-Temporal 3DGS — transfer between richly and sparsely observed epochs;
- LTGS — incremental long-term Gaussian scene updates;
- GaME — evolving-scene mapping and stale-content removal;
- Changes in Real Time — online multi-view change detection.

WRE should combine explicit `TemporalState`/`ChangeEvent` semantics with the best representation method rather than binding chronology to one renderer.

## LOD, compression and streaming

### Candidates / references

- LoD-of-Gaussians and modern hierarchical/multiscale Gaussian streaming;
- SuperSplat streaming/LOD concepts;
- SPZ/SOG-class compact splat formats;
- PCGS-class progressive splat compression;
- HAC++-class contextual entropy coding/adaptive quantization;
- vector-quantized Gaussian compression families;
- 4DGS-1K/ReCon-GS/HPC-class dynamic compression research;
- mesh simplification, texture atlases and glTF/GLB for explicit geometry pipelines.

Benchmark rate-distortion, decode/init cost, time-to-first-useful-view, random/spatial access and renderer compatibility rather than compression ratio alone. Master formats and runtime formats remain separate.

## Viewer / runtime

### Web

- SuperSplat/PlayCanvas ecosystem is a strong reference and candidate for splat rendering, walk/fly/orbit, skybox and collision workflows.

### Game/XR

- Unreal/Unity integration through explicit mesh + splat runtime assets;
- evaluate mature Gaussian plugins such as YaGS-class integrations rather than writing a renderer first.

The runtime must support separate collision/physics geometry even when splats drive visual appearance.

## Execution profiles and profiling tooling

These are runtime/tooling candidates, not new reconstruction contracts.

### Learned inference/training execution

- PyTorch eager/reference execution;
- `torch.compile`, CUDA Graphs and current compiler/runtime acceleration where applicable;
- TensorRT/Torch-TensorRT-class optimized execution for stable supported models/shapes;
- mixed precision or stronger quantization only with explicit quality evidence.

Execution profiles are compared under the same WRE input/output contract. Engine build/cold-start cost, fallback/unsupported operators, hardware/runtime compatibility and numerical/quality behavior are part of the benchmark.

### Profiling evidence

- solver-independent stage timers and CUDA events;
- NVTX-class ranges and Nsight-class tracing on compatible NVIDIA systems;
- equivalent vendor/platform profiling tools elsewhere.

The benchmark record should retain or reference evidence; WRE does not make one profiler a core dependency.

### Storage and locality acceleration

- conventional buffered/direct I/O as portable references;
- memory mapping/zero-copy where safe;
- topology-aware device/storage placement on heterogeneous systems;
- GPUDirect Storage/cuFile-class paths only on compatible systems where profiling shows I/O is material.

Direct-storage acceleration is not a default merely because the hardware supports it.

### Custom kernels

Triton/custom CUDA fused kernels are last-mile candidates only after profiling demonstrates a durable hotspot and maintained framework/library implementations are inadequate. Require a reference path and end-to-end benefit.

The development ownership/timing for these families is in `29_PERFORMANCE_INTEGRATION_MAP.md`.

## Professional external tool interoperability

WRE should be able to import/export or coexist with results from mature systems where useful:

- RealityCapture/RealityScan;
- Metashape;
- COLMAP;
- AliceVision/Meshroom;
- Nerfstudio;
- DCC tools such as Blender/Maya/ZBrush;
- game-engine and XR asset formats.

Interoperability is a feature, not a threat to the core architecture.

## Internal selection process

For each capability:

1. define representative fixtures and target metrics;
2. establish a stable baseline;
3. integrate candidate behind the same contract;
4. record exact version/checkpoint/license/hardware;
5. run PREVIEW/FAST/QUALITY-relevant benchmarks;
6. inspect failure classes, not only aggregate metrics;
7. choose a default per data profile;
8. retain meaningful fallback(s);
9. periodically rerun the benchmark when new methods appear.

For performance-sensitive candidates, first identify the actual end-to-end bottleneck and compare cold-start/steady-state, resource and quality effects. Do not promote a backend from an isolated microbenchmark.

## Replacement rule

A method can replace the current default when it is measurably better for the relevant profile after accounting for quality, latency, resources, reliability, licensing and integration cost. The architecture should make such replacement routine rather than disruptive.

## Research-watch principle

The technology landscape evolves faster than the product architecture. New papers should first enter the benchmark registry, not the core domain model. Architecture changes are justified only when a new research family introduces a genuinely new responsibility or representation that existing contracts cannot express.
