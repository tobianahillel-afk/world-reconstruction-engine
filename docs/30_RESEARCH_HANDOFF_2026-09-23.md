# Research handoff — 2026-09-23 frontier 3D/4D review

> **Status:** advisory research handoff, not a roadmap override.
>
> **Critical current-state rule:** do **not** widen the active item. As of this note, `PROJECT_STATE.yaml` is on **V2M2 / V2L14.3**. Complete the exact COLMAP/PyCOLMAP/Ceres bundle-adjustment baseline required by that item's executable contract before touching V2L14.4 or any future integration named below.

## Why this file exists

A broad review of the 2025–2026 reconstruction literature was compared against the current WRE v2 product, architecture, roadmap and implemented state.

The main conclusion is positive:

**WRE v2 does not need a new product architecture to absorb the strongest recent research.** The frozen design already separates the right responsibilities: observations, evidence, camera/depth geometry, explicit physical surface, photorealistic appearance, materials/environment, dynamics, long-term chronology, uncertainty/provenance, master assembly and target-specific runtime compilation.

The important work is therefore **candidate refresh, fair benchmarking and adapter integration**, not architecture churn.

WRE should continue to aim for **best-in-class integrated-system quality**, not make the false claim that one internal algorithm will beat every specialist on every specialist benchmark.

A specialist may remain better on one narrow task. WRE's advantage should come from:
- accepting heterogeneous real-world media;
- routing by data profile and quality mode;
- comparing competing solutions under common contracts;
- combining learned and classical geometry when evidence supports it;
- preserving uncertainty/provenance;
- keeping physical surface separate from photorealistic appearance;
- adding materials, environment, dynamics and chronology when useful;
- compiling efficient runtime assets without degrading master truth.

## Do not undo the current architecture

The following current v2 choices were strongly reinforced by the research review and should be preserved unless a future explicit blueprint-change proposal demonstrates a real missing product responsibility:

1. **Stable WRE contracts, replaceable specialist solvers.**
2. **`GeometrySolution`, `SurfaceModel`, `AppearanceModel`, `MaterialModel`, `EnvironmentModel`, `DynamicEntity`, `Trajectory`, `TemporalState`, `ConfidenceField`, `MasterScene` and `RuntimeScene` remain distinct responsibilities.**
3. **Observed/reconstructed, inferred and generated content remain distinguishable.**
4. **Do not make Gaussian splats the physical world model.** They are an excellent appearance/runtime representation, but measurement/collision/navigation may require mesh/SDF/surfels or another explicit surface.
5. **Do not make a learned hidden state the sole project truth.** Persist canonical artifacts and lineage.
6. **Do not replace all classical geometry with a neural model.** Strong 2026 systems increasingly combine learned dense geometry with bundle adjustment, factor graphs, Sim(3), SfM/SLAM or explicit spatial backends.
7. **Do not force all data profiles through one pipeline.** Sparse Internet photos, drone capture, long phone video, short dynamic scenes, transparent/water scenes and historical media need different specialists.
8. **Do not confuse photorealism with physical/geometric accuracy.** Measure them separately.
9. **Do not promote a paper because of one published metric.** Reproduce relevant behavior under WRE contracts and representative fixtures.
10. **Keep the candidate landscape living.** The method named today can be obsolete by the time its work item activates.

## Immediate relevance to the current V2M2 geometry work

The current repo has already implemented a bounded DA3-BASE feed-forward geometry path, reproducibility checks, quality metrics and comparison against the classical path, then introduced common competing-solution/refinement boundaries.

That direction is correct.

### Candidate families to add to the next V2M2 geometry refresh

The current research documents already include DA3, VGGT-Ω, Pi3/Pi3X, MapAnything, CUT3R, GLUEMAP, VGG-T3-class scaling and multiple classical/hybrid routes. The following additional families deserve explicit review before later V2L14 promotion/integration decisions:

#### AMB3R family
Potential WRE role:
- metric-scale feed-forward geometry;
- learned frontend plus explicit 3D backend;
- VO/SfM-style variants;
- comparison against DA3/VGGT-Ω/classical refinement under the canonical `GeometrySolution` contract.

Why it matters:
- it represents the broader trend toward coupling foundation-model predictions with an explicit spatial/backend representation rather than trusting unconstrained point maps alone.

#### Geometry-grounded transformer / GGPT-class hybrids
Potential WRE role:
- learned dense geometry explicitly grounded by sparse/classical geometry;
- reference architecture for V2L14 hybrid/global refinement thinking.

Why it matters:
- it closely matches WRE's intended pattern: neural dense information plus trusted geometric constraints instead of an either/or choice between AI and SfM.

#### UniSim-SLAM / modern neural + factor-graph hybrids
Potential WRE role:
- V2L14/V2L28 research candidate;
- learned local reconstruction combined with global Sim(3)/factor-graph consistency.

Why it matters:
- reinforces that a high-end system should preserve both learned perception and mature global optimization.

**Important:** none of these names changes V2L14.3. Finish the exact classical BA baseline first.

## PREVIEW / FAST efficiency candidates

### Deja View-class iterative/recurrent compact models
Potential home:
- V2L13-family future candidate refresh;
- execution-profile benchmarking.

Potential benefit:
- much smaller iterative model families may deliver an attractive quality/compute tradeoff for PREVIEW/FAST;
- iteration count can potentially become an execution budget rather than a new product semantic.

Guardrails:
- verify exact code/weights/license;
- benchmark against the same canonical outputs;
- do not encode a model-specific recurrent state into WRE core semantics.

### VGGT-Prime-class adaptive execution
Potential home:
- execution profile, not a new geometry type.

Potential benefit:
- attention/head/token compute reduction for existing foundation-style reconstruction.

Guardrail:
- treat as an implementation/performance candidate behind an existing adapter; do not create a new core `GeometrySolution` type because a model runs faster.

## Long-stream / bounded-memory geometry

V2L28 is already the correct roadmap home for long-sequence state, chunking, drift and loop/global refinement.

Refresh its candidate list with:

### LingBot-Map-class persistent streaming reconstruction
Evaluate for:
- bounded-memory long videos;
- persistent geometric context;
- long continuous trajectories.

### RegVGGT-class token-memory compression
Evaluate for:
- retaining only high-value historical tokens/context;
- reducing sequence memory cost without changing WRE scene semantics.

### FILT3R-class uncertainty-aware recurrent update
Evaluate for:
- memory update rules that explicitly account for uncertainty/confidence;
- preventing destructive overwrites in long streams.

### UniSim-SLAM / neural + factor-graph long-sequence systems
Evaluate for:
- local learned geometry plus global drift correction;
- loop/global consistency.

Product-level requirement remains:
- hidden neural memory is solver-private;
- canonical geometry/submaps, transforms, uncertainty and lineage must be materialized into WRE artifacts;
- long-stream quality must include drift, loop closure, boundary artifacts, memory and runtime cost.

## Dense tracking, persistent identity and long 4D sequences

### Point4D-class long-horizon 3D tracking
Potential homes:
- V2L24 dense motion evidence;
- V2L27 `DynamicEntity` / `Trajectory`;
- V2L28 long-sequence dynamics.

Why it matters:
- persistent 3D tracking across chunks/occlusions is closer to WRE's object-permanence goal than short-window 2D tracks alone.

Evaluate:
- identity continuity;
- reappearance after occlusion;
- 3D trajectory error;
- drift over long horizons;
- compute/memory.

### OmniX-class feed-forward 4D
Potential homes:
- V2L25 fast 4D preview;
- V2L26 quality 4D candidate comparison.

Evaluate beside:
- D4RT-class systems;
- MoVieS;
- MotionScale;
- Shape of Motion/MoSca/other current candidates.

Do not infer that one method should own geometry, tracking, appearance and physical surface simply because it emits all of them. Normalize outputs into WRE's separate contracts.

## Uncertainty and active capture

WRE already has `ConfidenceField` and expert provenance/confidence visualization. Keep that.

Add explicit candidate review for:

### GAVIS-class directional/anisotropic visibility uncertainty
Potential homes:
- appearance/geometry confidence;
- expert viewer;
- future assisted capture guidance.

Why it matters:
- uncertainty can depend on viewing direction, not only spatial location;
- this can support "which viewpoint should I capture next?" workflows.

### VarSplat/probabilistic Gaussian-family uncertainty
Potential homes:
- appearance confidence;
- sparse/streaming specialist evaluation.

A future **next-best-view / capture guidance** feature can likely be built from existing confidence + ASSISTED-mode responsibilities; do not create a new architecture layer unless later evidence shows the current contracts are insufficient.

## Static physical surface

V2L16 already owns explicit surface and is the right place to compare physical geometry methods.

Add:

### Gaussian Wrapping / blobs-to-spokes-class surface extraction
Potential role:
- convert strong Gaussian/dense visual reconstruction into watertight explicit surface;
- preserve thin structures better than naive mesh extraction.

Compare against:
- depth fusion;
- Poisson/Delaunay/TSDF/SDF baselines;
- OMeGa;
- SurfaceSplat;
- MeshSplatting-class hybrids.

Success metric:
- geometric/surface accuracy, completeness, watertightness where needed, thin-structure retention, collision/navigation suitability and resource cost — **not render PSNR alone**.

## Photometric normalization for heterogeneous media

This is especially important because WRE must accept arbitrary phone/tablet/drone/Internet/video sources with inconsistent exposure, tone mapping and metadata.

### P2GS-class physical photometric calibration
Potential homes:
- V2L17 photometric normalization;
- V2L18 appearance;
- V2L40 HDR/special conditions.

Evaluate whether these methods can help disentangle:
- scene radiance;
- per-camera exposure;
- tone mapping/camera response;
- actual scene change.

Do not allow camera processing differences to masquerade as physical scene/material differences.

## Physically faithful materials and relighting

The current roadmap already has the correct responsibilities in V2L19 and V2L42.

Add or re-evaluate the following at activation:

### DiffReg-PBIR-class physics-based inverse rendering
Potential role:
- joint geometry/material/lighting optimization;
- diffusion/learned priors as regularizers rather than as truth;
- production-ready PBR-style assets when evidence supports them.

### GHPT-class hybrid Gaussian + mesh + path-traced transport
Potential role:
- high-fidelity relighting/runtime research;
- Gaussian appearance combined with explicit mesh visibility/ray tracing.

### Stochastic Gaussian ray-tracing / Stoch3DGS-class systems
Potential role:
- differentiable ray tracing;
- shadow/relighting/non-pinhole or physically richer rendering;
- possible MASTER/offline or high-end runtime backend.

### MatSpray / IR-HGP / GAINS families
Already conceptually covered; keep them in the candidate pool and compare under one `MaterialModel` / `EnvironmentModel` contract.

Key rule:
- learned priors may help infer materials, but inferred material values are not direct observation;
- retain provenance and uncertainty.

## HDR and camera-response-aware reconstruction

In addition to current PhysHDR-GS-class coverage, review:

- P2GS-class exposure/tone/radiance separation;
- InstantHDR-class feed-forward HDR reconstruction where mature;
- other current HDR Gaussian/radiance methods.

For heterogeneous Internet/phone/drone media, test robustness to:
- unknown camera response;
- clipping/saturation;
- auto exposure;
- white-balance shifts;
- HDR/LDR mixing.

## Reflection, transparency, glass and water

V2L41 already owns difficult reflective/translucent/water inputs. Make the specialist shortlist more explicit when that lot activates.

### RT-Splatting-class reflection/transmission reconstruction
Potential role:
- scenes containing glass/transmission where normal opaque Gaussian assumptions fail.

### RefracGS-class refraction/water reconstruction
Potential role:
- refractive media/surfaces;
- water-specialist route.

### Opti-NeuS / transparent-object neural-surface families
Potential role:
- transparent/opaque layered objects when applicable.

### Polarization-aware reconstruction
Potential future candidates:
- PolarGuide-GSDR-class methods;
- differentiable polarized path tracing (DPPT-class);
- polarization-aware inverse rendering.

Important product rule:
- RGB remains the universal minimum input;
- specialist sensors such as polarization/LiDAR/events can enrich a route when available, but they must not become mandatory for normal WRE operation.

## Long 4D runtime / compression

V2L30 and runtime/compiler work already own the right responsibilities.

Add:

### Layered 4D-Rotor / L4DRotorGS-class temporal compression
Evaluate for:
- long dynamic scenes;
- temporal buckets/lifetimes;
- rate-distortion;
- memory residency;
- temporal random access;
- runtime FPS;
- time-to-first-useful-frame.

Do not promote by compression ratio alone.

Also retain current 4DGS-1K/ReCon-GS/HPC-class references and compare under the same runtime contract.

## Urban / aerial / city-scale reconstruction

### Urban-GS-class aerial + street-level fusion
Potential homes:
- V2L37 scalable mapping;
- V2L39 aerial/drone;
- V2L18 appearance.

Why it matters:
- aerial and ground observations have radically different scale/coverage characteristics;
- city-scale systems need multi-scale densification, pruning, global/local decomposition and strong runtime LOD.

### GeoGS/geospatial Gaussian-class sparse satellite reconstruction
Research watch:
- potentially relevant if WRE later expands beyond ordinary photo/video capture into satellite/geospatial imagery;
- do not expand current product scope solely because the paper exists.

## Event cameras, LiDAR and other sensors

The current product requirement remains arbitrary photos/videos as the universal baseline.

However, keep future adapter space for:
- LiDAR-assisted geometry/SLAM;
- IMU/GNSS priors;
- event cameras for high-speed/low-light motion;
- thermal/multispectral/polarization where available.

Do not make these mandatory and do not distort the image/video-first product to chase sensor-specific research. Treat them as optional observation/specialist evidence.

## Learned global-light-transport renderers

### RenderFormer-V2-class neural transport
Potential role:
- runtime/research renderer for complex global illumination;
- possibly high-throughput preview/production rendering once mature.

Guardrail:
- learned light transport is not automatically physically authoritative;
- preserve a conventional PBR/path-traced reference where physical fidelity matters.

## World models / generated world memory

WorldCrafter/OctWorld/programmatic-world-model-class systems are interesting research references for:
- persistent memory;
- long-horizon scene consistency;
- generative exploration.

They must **not** become authoritative reconstructed reality.

Any generated completion must enter through the existing `CompletionArtifact` / GENERATED provenance boundary and remain distinguishable from observed/reconstructed and inferred content.

## Recommended candidate-refresh priorities by milestone

This is guidance for future activation reviews, **not permission to start these lots early**.

### V2M2 — geometry
Priority review:
1. AMB3R family;
2. GGPT/geometry-grounded hybrids;
3. UniSim-SLAM / modern learned + factor-graph hybrids;
4. current VGGT-Ω/DA3/MapAnything/Pi3 families;
5. execution-only candidates such as Deja View/VGGT-Prime where relevant.

### V2M3 — static photorealism, surfaces, materials, runtime
Priority review:
1. Gaussian Wrapping-class surface extraction;
2. P2GS-class photometric consistency;
3. DiffReg-PBIR-class inverse rendering;
4. GHPT/Stoch3DGS-class physically richer rendering;
5. existing MeshSplatting/MatSpray/IR-HGP/PhysHDR families.

### V2M4 — dynamic 4D
Priority review:
1. Point4D-class long tracking;
2. OmniX-class feed-forward 4D;
3. D4RT/MoVieS/MotionScale/4DSurf current families;
4. LingBot-Map/RegVGGT/FILT3R/UniSim for long sequence state;
5. L4DRotorGS-class runtime/compression.

### V2M5 — chronology
Keep:
- LTGS;
- GaME;
- Cross-Temporal 3DGS;
- Neural Scene Chronology;
- explicit `TemporalState` / `ChangeEvent` semantics remain authoritative.

### V2M6 — difficult/specialist inputs
Priority review:
1. Urban-GS-class aerial+street;
2. RT-Splatting-class reflection/transmission;
3. RefracGS-class water/refraction;
4. polarization-aware inverse rendering if sensors/data justify it;
5. current blur/HDR/low-light/360/aerial specialists.

## Benchmark policy reinforced by the research review

For every candidate refresh, compare at least the dimensions relevant to the owning WRE capability.

### Geometry
- camera pose;
- intrinsics/calibration when applicable;
- depth/point-map accuracy;
- metric scale;
- global consistency/drift;
- surface accuracy/completeness;
- failure rate.

### Appearance
- held-out views;
- perceptual quality;
- camera-path temporal stability;
- zoom/scale stability;
- view-dependent effects;
- exposure/white-balance robustness.

### Dynamics
- trajectory accuracy;
- persistent identity;
- occlusion/reappearance;
- motion/deformation accuracy;
- temporal consistency;
- long-horizon drift.

### Materials/lighting
- relighting consistency;
- albedo/roughness/metallic/normal quality where ground truth exists;
- disentanglement of lighting vs material;
- HDR behavior;
- reflection/transmission/refraction specialists where relevant.

### Production/runtime
- wall-clock;
- cold start;
- throughput;
- RAM/VRAM;
- storage;
- checkpoint/resume;
- rate-distortion;
- decode/init;
- first-useful-view latency;
- streaming/random access;
- FPS;
- platform/driver/compiler constraints;
- license and redistribution constraints.

Never collapse these into one "best" score.

## Universal input requirement remains correct

WRE must continue to accept heterogeneous media such as:
- phone/tablet/drone/action-camera photos and video;
- Internet/archival photos;
- images with complete, partial, wrong or absent metadata;
- mixed focal lengths/resolutions/camera pipelines;
- LDR/HDR and varying exposure;
- 360/fisheye/rolling-shutter/blurred sources through specialist routes.

Metadata is useful evidence, not a mandatory dependency.

Keep original media immutable. Derived normalized images/frames, inferred calibration, estimated dates/poses and confidence must remain separate artifacts with provenance.

## The desired final WRE behavior

The ideal integrated system is not "one perfect model." It should behave more like:

```text
heterogeneous real-world media
        |
        v
organization / evidence / calibration
        |
        +--> competing learned geometry
        +--> classical SfM/SLAM/global geometry
        +--> specialist sparse/large/360/aerial routes
        |
        v
canonical competing GeometrySolution artifacts
        |
        v
comparison / refinement / quality gates
        |
        +--> explicit physical SurfaceModel
        +--> photorealistic AppearanceModel
        +--> MaterialModel + EnvironmentModel
        +--> DynamicEntity / Trajectory / MotionField
        +--> TemporalState / ChangeEvent
        +--> ConfidenceField
        |
        v
MasterScene
        |
        v
target-specific RuntimeScene / LOD / compression / streaming
```

When a specialist is clearly best for one data profile, WRE should use it rather than reimplement it. When two methods provide complementary evidence, WRE may combine/refine them behind stable contracts. When evidence is insufficient, WRE should fail closed or preserve a hole instead of inventing precision.

## Non-negotiable warning for future agents

**Do not turn this research handoff into opportunistic scope expansion.**

At every activation:
1. read the current work-item contract;
2. refresh the candidate landscape;
3. re-check exact code/checkpoint/version/license/hardware availability;
4. verify current primary sources — paper names and repositories can change after this note;
5. benchmark relevant viable candidates under one WRE contract;
6. integrate only the candidate(s) permitted by that work item;
7. preserve the existing frozen architecture unless a genuine unrepresentable product responsibility is discovered.

The research landscape moves faster than the roadmap. That is intentional. The architecture should stay stable while the specialists keep improving.
