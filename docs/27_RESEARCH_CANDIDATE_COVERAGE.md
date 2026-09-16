# WRE v2.0 research candidate coverage map

Status: **LIVING / NOT FROZEN**.

This document maps the current research landscape to the frozen WRE v2.0 product architecture. It is intentionally allowed to change as new methods appear. Candidate names, benchmark priority and default recommendations are not part of the architecture freeze.

The goal is to answer two questions continuously:

1. does a newly important research family fit an existing WRE responsibility and contract?
2. which currently known candidates should be evaluated when the owning work item activates?

If a new paper fits an existing responsibility, it becomes an adapter/benchmark candidate. It does not justify redesigning the engine. A blueprint change is justified only when a genuinely new product responsibility or representation cannot be expressed by the frozen contracts.

## Current architecture conclusion

The reviewed 2025–2026 research landscape does **not** currently require a new top-level WRE architecture layer.

The major families identified so far fit the existing frozen responsibilities:

- media/profile evidence;
- retrieval and scene identity;
- temporal evidence/synchronization;
- camera/depth/geometry solutions;
- explicit surface;
- appearance/radiance;
- materials/environment;
- dynamic entities/motion/4D;
- long-term temporal states/change;
- master-scene assembly;
- runtime compilation/LOD/streaming;
- benchmark/routing/quality infrastructure.

This is an important validation of the architecture: new methods can normally change a candidate shortlist or default adapter without changing the core product.

## Candidate families that must remain visible

The list below supplements `14_TECHNOLOGY_SELECTION.md`. Inclusion here means **candidate to evaluate**, not shipping approval.

### Place / geo-time retrieval evidence

Relevant candidates/families:

- TIGER-class joint place/time retrieval;
- GT-Loc-class when/where evidence;
- Pinpoint-class cross-source worldwide retrieval/reranking;
- GeoCLIP-class coarse geolocation evidence;
- SALAD/modern visual-place retrieval;
- hloc/local geometric verification compositions.

WRE use:

- contribute evidence to `PairCandidate`, `SceneCluster`, temporal/date hypotheses and archival organization;
- never make global geolocation mandatory;
- never promote a learned place/time prediction directly to exact scene identity or capture timestamp.

Primary roadmap homes: `V2L7`, `V2L8`, `V2L31`.

### Long-tail Internet-photo robustness

Relevant candidates/families:

- MegaDepth-X-style data/training improvements for low-overlap Internet collections;
- Doppelgangers++-class repeated-structure disambiguation;
- GLUEMAP-style hybrid learned geometry plus global refinement;
- modern local matching such as LightGlue/ALIKED/DISK, MASt3R-family matching and RoMa/LoFTR-class difficult-view matching.

WRE use:

- improve retrieval/matching/geometry robustness without changing canonical geometry contracts;
- measure false scene merges and failure on long-tail inputs, not only average recall.

Primary roadmap homes: `V2L7`, `V2L8`, `V2L14`, `V2L37`, `V2L41`.

### Fast feed-forward camera/depth/geometry

High-priority current candidates/families:

- Depth Anything 3 (DA3) family;
- VGGT-Ω;
- Pi3/Pi3X;
- MapAnything-class universal geometry;
- CUT3R;
- Uni3R-class feed-forward reconstruction;
- TokenSplat-class pose-free feed-forward reconstruction.

WRE use:

- PREVIEW/FAST geometry, priors or full `GeometrySolution` when quality evidence supports it;
- never assume a feed-forward output is only a proposal, and never assume it is authoritative merely because it is fast.

Primary roadmap homes: `V2L13`, `V2L14`, `V2L23`.

### Stateful / streaming / very-large-scene geometry

High-priority candidates/families:

- LongStream;
- ZipMap;
- Scal3R;
- VGG-T³-class scalable unordered reconstruction;
- CUT3R/stateful reconstruction;
- Anchor3R/Online3R-class incremental or online scene-state approaches;
- SLAM3R and SLAMFormer-∞-class long-sequence systems.

WRE use:

- bounded-memory long-sequence geometry;
- scalable large-collection processing;
- scene state or hidden memory remains solver-private unless materialized as canonical WRE artifacts;
- project truth remains artifact-based, not an opaque neural state alone.

Primary roadmap homes: `V2L13`, `V2L28`, `V2L37`.

### Crowd-sourced / multi-reconstruction merge

Relevant candidates/families:

- MR.ScaleMaster-class Sim(3) merge of independently reconstructed monocular videos/maps;
- mature pose-graph/submap/global-alignment systems;
- hierarchical reconstruction and submap merge approaches.

WRE use:

- merge/compare independently produced `GeometrySolution` or submap artifacts;
- preserve source solution provenance and disagreement metrics;
- do not hide scale/registration uncertainty.

Primary roadmap homes: `V2L14`, `V2L29`, `V2L37`.

### Sparse-view reconstruction

High-priority candidates/families:

- DepthSplat;
- Sparse2DGS;
- GIFSplat;
- SPARS3R-class prior alignment;
- Free360 for sparse/unbounded 360-oriented conditions.

WRE use:

- specialist route only when coverage/profile supports it;
- preserve uncertainty and distinguish generative prior contribution from reconstructed support.

Primary roadmap homes: `V2L13`, `V2L36`, `V2L38`, `V2L43` when generation is involved.

### Dense tracking and dynamic evidence

High-priority candidates/families:

- AllTracker;
- CoTracker3;
- Optical Flow Matching-class continuous transport/flow;
- MegaSaM-class dynamic camera/depth support where useful;
- SAM2/video segmentation, VideoCUPS and VidEoMT-class scene/object decomposition as supporting evidence.

WRE use:

- create dense motion/tracking evidence shared by synchronization, decomposition and 4D routes;
- segmentation labels are evidence, not automatically persistent world identity.

Primary roadmap homes: `V2L24`, `V2L29`.

### Fast 4D preview

High-priority candidates/families:

- MoVieS;
- MoRe;
- D4RT-class unified dynamic reconstruction;
- UFO-4D-class sparse dynamic reconstruction.

WRE use:

- immediate viability/preview and fast dynamic hypothesis;
- normalization into WRE camera/depth/motion artifacts before downstream use.

Primary roadmap home: `V2L25`.

### Quality / online / persistent 4D

High-priority candidates/families:

- MotionScale;
- Shape of Motion;
- MoSca;
- MOSAIC-GS-class dynamic reconstruction;
- ProDyG for online dynamic reconstruction;
- 4D Primitive-Mâché for object permanence;
- GP-4DGS-class uncertainty-aware dynamic reconstruction as research watch;
- 4DSurf and dynamic mesh/SDF systems for physical dynamic surface.

WRE use:

- dynamic geometry/appearance quality routes;
- persistent `DynamicEntity` identity through occlusion where supported;
- explicit uncertainty for inferred motion or unobserved intervals;
- keep dynamic visual appearance separate from dynamic physical surface.

Primary roadmap homes: `V2L26`, `V2L27`, `V2L28`.

### Dynamic rendering / temporal appearance

Relevant candidates/families:

- RetimeGS-class continuous-time dynamic rendering;
- SpacetimeGS/4D-GS-class real-time dynamic appearance;
- 4DGS-1K-class runtime compression/performance research.

WRE use:

- appearance/runtime candidates after dynamic geometry/motion contracts exist;
- do not treat rendering quality alone as proof of correct motion/geometry.

Primary roadmap homes: `V2L26`, `V2L30`, runtime/compression portions of `V2L21` and `V2L49`.

### 360 / omnidirectional

High-priority candidates/families:

- PFGS360-class pose-free 360 reconstruction;
- Free360-class sparse/unbounded 360 synthesis;
- native spherical/equirectangular calibration paths.

Primary roadmap home: `V2L38`.

### Drone / aerial

Relevant candidates/families:

- AeroGS-class dynamic UAV reconstruction;
- mature aerial SfM/MVS for predominantly static capture.

Primary roadmap home: `V2L39`.

### Static surface and geometry/appearance coupling

High-priority candidates/families:

- depth fusion + mature mesh/SDF pipelines;
- 2DGS-derived geometry;
- OMeGa joint mesh+splat optimization;
- SurfaceSplat coupled surface/radiance refinement;
- MeshSplatting-class connected-mesh appearance systems.

WRE use:

- evaluate surface by geometry/physics metrics, not render PSNR alone;
- preserve separate `SurfaceModel` and `AppearanceModel` responsibilities even when one method optimizes both jointly.

Primary roadmap homes: `V2L16`, `V2L18`.

### Photorealistic static appearance

High-priority candidates/families:

- gsplat as low-level backend when approved;
- WildGaussians-class in-the-wild appearance;
- Nerfstudio/Splatfacto as integration reference;
- 7DGS-class view-dependent appearance where complexity is justified.

Primary roadmap home: `V2L18`.

### Materials, lighting and HDR

High-priority candidates/families:

- GaRe-class outdoor relighting;
- MatSpray-class material fusion;
- IR-HGP-class physically-aware inverse rendering;
- PhysHDR-GS-class HDR reconstruction;
- R3GW/related environment-separated methods.

WRE use:

- map recovered outputs into explicit `MaterialModel` / `EnvironmentModel` contracts;
- preserve measured versus inferred provenance.

Primary roadmap homes: `V2L19`, `V2L40`, `V2L42`.

### Historical chronology and evolving scenes

High-priority candidates/families:

- Neural Scene Chronology;
- Cross-Temporal 3DGS;
- LTGS;
- GaME;
- Changes in Real Time-class multi-view change detection.

WRE use:

- explicit `TemporalState` and `ChangeEvent` semantics remain authoritative;
- representation methods may assist transfer/update but never collapse decades of change into one mandatory smooth deformation.

Primary roadmap homes: `V2L31`–`V2L35`.

### LOD / streaming / compression / viewer

Relevant candidates/references:

- LoD-of-Gaussians and hierarchical Gaussian streaming;
- SuperSplat/PlayCanvas streaming, viewer and collision separation;
- SPZ/SOG-class compact splat formats;
- 4DGS-1K/ReCon-GS/HPC-class compression research;
- glTF/GLB and mature mesh optimization for explicit geometry;
- YaGS-class Unreal integration as a production reference.

WRE use:

- compile from `MasterScene` into target-specific `RuntimeScene` artifacts;
- keep collision/physics geometry independent from photorealistic appearance;
- benchmark device memory, loading, streaming and FPS rather than only offline file size.

Primary roadmap homes: `V2L21`, `V2L22`, `V2L30`, `V2L45`, `V2L49`.

## Architecture coverage matrix

| Research/problem family | Existing frozen WRE responsibility | Main roadmap homes | Architecture change needed now? |
| --- | --- | --- | --- |
| place/geo-time retrieval | relationship/date evidence | V2L7, V2L8, V2L31 | No |
| long-tail Internet photos | retrieval/matching/geometry robustness | V2L7, V2L8, V2L14, V2L37 | No |
| feed-forward geometry | camera/depth/geometry adapter | V2L13, V2L14 | No |
| stateful/streaming geometry | geometry adapter + artifact DAG + chunking | V2L28, V2L37 | No |
| crowd-sourced reconstruction merge | competing geometry/submap alignment | V2L14, V2L29, V2L37 | No |
| sparse views | specialist route + coverage confidence | V2L36 | No |
| dense tracking | motion evidence | V2L24 | No |
| fast 4D | dynamic preview | V2L25 | No |
| quality/persistent 4D | dynamic solution/entities/surface | V2L26, V2L27 | No |
| 360 | native camera + specialist route | V2L38 | No |
| drone | aerial route | V2L39 | No |
| blur/HDR/rolling shutter/low light | specialist profile + route | V2L40 | No |
| repeated/reflection/water difficulty | specialist matching/material route | V2L41, V2L42 | No |
| mesh+splat/SDF hybrids | separate surface + appearance contracts | V2L16, V2L18 | No |
| PBR/relighting | materials/environment | V2L19, V2L42 | No |
| historical/evolving scenes | TemporalState + ChangeEvent | V2L31–V2L35 | No |
| LOD/compression/streaming | runtime compiler | V2L21, V2L30, V2L49 | No |
| generative completion | explicit generated provenance | V2L43 | No |

## Candidate refresh rule

Before any work item integrates or promotes a solver/model, its activation PR must:

1. re-read `14_TECHNOLOGY_SELECTION.md` and this document;
2. check the current adapter/model/dependency registries;
3. refresh the external candidate landscape if the field has materially moved;
4. identify at minimum a stable baseline, the strongest currently viable primary candidate(s), specialist alternatives and research-watch methods where relevant;
5. review exact version/checkpoint, license, shipping constraints, hardware and reproducibility;
6. benchmark candidates under the same stable WRE contract and representative fixture;
7. record per-dimension metrics and failure classes rather than only one aggregate score;
8. promote a default only through the owning benchmark/quality policy.

A roadmap item such as “integrate first approved feed-forward geometry candidate” deliberately does **not** mean a candidate named in 2026 is permanently chosen.

## What should trigger a blueprint revision

A new method should **not** trigger architecture work merely because it is much better or combines several existing outputs.

A blueprint-change proposal is warranted only if the method exposes a necessary product responsibility that cannot be represented through the current WRE contracts without distortion. Examples would be a genuinely new form of user-visible truth/provenance, a fundamentally different temporal regime, or a runtime/product representation that cannot be expressed by the existing MasterScene/RuntimeScene boundaries.

So far, the reviewed candidate families do not cross that threshold.
