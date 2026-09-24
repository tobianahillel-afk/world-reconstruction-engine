# Product blueprint

## Product thesis

World Reconstruction Engine (WRE) is an independent visual spatiotemporal reconstruction engine. It transforms arbitrary real-world photos and videos into high-fidelity, freely navigable 3D/4D scenes whose geometry, appearance, motion and long-term temporal states can be refined as new observations arrive.

The target experience is not a point cloud, a single mesh, a Gaussian Splat, a rendered video or a world-map database. Those are representations or outputs. The product goal is a scene in which a user can feel present in the observed place or event: walk at ground level, look in every direction, fly or use a drone-like camera, stop or scrub time, revisit different historical states, and render new viewpoints when the observations support them.

WRE is autonomous from MONDE. Future integration with a larger world model may exist through optional interfaces, but it is not the product definition, a requirement for reconstruction, or the organizing principle of this repository.

## Inputs

The engine accepts real-world media with widely varying quality and structure:

- unordered photographs from one or many cameras;
- videos from phones, action cameras, drones or 360 cameras;
- multiple unsynchronized videos of the same event;
- Internet or archival photo collections with changing illumination, weather, people and cameras;
- media collected over seconds, years or decades;
- controlled high-quality capture when available.

GPS, camera poses, calibration, synchronized clocks and exact timestamps are useful priors but are not mandatory. The engine must profile the available observations before deciding how to reconstruct them.

## Primary user outcomes

A successful project may produce one or more of the following:

1. **Static immersive place reconstruction** — photorealistic and freely navigable.
2. **Dynamic 4D replay** — an event reconstructed through time with a virtual camera independent of the original cameras.
3. **Historical chronology** — multiple supported states of one place that can be explored with a time control.
4. **Production asset** — explicit surface geometry, appearance and materials suitable for downstream DCC/game/XR workflows.
5. **Fast preview** — an immediate approximation used to validate coverage before expensive processing.
6. **Master reconstruction** — the highest-quality retained scene representation from which runtime assets are compiled.

The product must support incremental enrichment: a project can begin with a few observations and later gain new views, videos, epochs or higher-quality captures without gratuitously recomputing every prior artifact.

## CPU portability baseline

WRE must remain usable on a supported CPU-only installation. Every mandatory product capability and milestone path must retain at least one supported CPU route, even when that route is slower or lower-throughput than an accelerated specialist.

GPU, CUDA, NPU or other accelerator-specific adapters may improve latency, throughput or quality and may be selected when compatible hardware is available. They are optional execution specialists, not prerequisites for the core product to function or for the mandatory roadmap to advance.

If no reviewed candidate for an optional specialist can run on the available hardware, the technology gate may explicitly defer that candidate without fabricating support or blocking a CPU-capable baseline. Quality/resource differences must remain visible in benchmark and routing evidence.

## Quality ambition

WRE is designed to be best-in-class at the integrated-system level. It does not assume one internal model can outperform every specialist research system on every benchmark. Instead it combines, routes and evaluates specialist methods so that the complete product can optimize the real trade space:

- visual fidelity;
- geometric fidelity;
- temporal consistency;
- robustness to in-the-wild media;
- processing latency;
- GPU/RAM/storage cost;
- scalability in views, duration and scene size;
- sparse-view performance;
- dynamic-scene performance;
- historical reconstruction quality;
- runtime performance;
- maintainability and replaceability of algorithms.

No single scalar score replaces the per-dimension metrics. A visually excellent method may have weak physical geometry; a precise mapper may be unsuitable for interactive appearance. WRE preserves those distinctions.

## Product quality modes

The same project can be processed at different budgets:

- **PREVIEW** — fastest useful reconstruction for coverage and viability inspection.
- **FAST** — interactive, good-quality result with bounded compute.
- **QUALITY** — publication/production-oriented reconstruction with stronger refinement and validation.
- **MASTER** — maximize supported fidelity using the available compute, specialist routes and cross-checks.

The router may select different algorithms at each level. Quality modes change compute and representation richness; they must not silently relabel synthetic completion as observation.

## Core representation principle

A scene is not one 3D object. WRE maintains complementary representations with explicit responsibilities:

- **observations** — original photos, video frames and source metadata;
- **camera solutions** — poses, intrinsics and calibration hypotheses;
- **correspondence evidence** — matches, tracks and cross-view relationships;
- **geometry** — depth, point maps and spatial structure;
- **surface** — mesh/SDF/surfels or equivalent physical surface used for measurement, collisions and navigation;
- **appearance** — radiance/splat/neural representations optimized for photorealistic rendering;
- **materials** — optional albedo, normal, roughness, metallic and related physical properties;
- **environment** — sky, horizon and distant illumination rather than fake nearby geometry;
- **dynamic entities and motion** — persistent objects, trajectories and deformation/motion fields;
- **temporal states** — long-term scene states and supported changes between epochs;
- **confidence** — spatial/temporal uncertainty and quality estimates;
- **runtime assets** — compressed, tiled and level-of-detail representations compiled for a target device;
- **synthetic completion** — optional generated content, always distinguishable from measured/reconstructed content.

A beautiful appearance representation must not automatically become collision geometry, measurement geometry or historical truth.

## Time has two distinct meanings

WRE separates at least two temporal regimes.

### Continuous event time

For short dynamic sequences, time describes motion: people, vehicles, articulated objects, camera motion and changing geometry. The product target is free-viewpoint 4D replay with temporal consistency and object persistence through occlusion where evidence permits.

### Long-term chronology

For observations separated by weeks, years or decades, time describes scene states and structural change. Buildings may be built or demolished, trees grow, scaffolding appears, façades change and streets are redesigned. These changes should be represented as states/events rather than forcing century-scale history into one continuous deformation field.

A project may contain both regimes.

## Realism target

The desired visual result is that the first impression is not “I am looking at a reconstruction”, but “I feel as if I am there”. Achieving that can require more than texture mapping:

- photometrically consistent appearance across cameras;
- fine geometric detail;
- view-dependent effects when supported;
- separated sky/environment representation;
- handling of illumination and exposure variation;
- material properties and relighting where useful;
- dynamic/static decomposition;
- high-quality anti-aliasing, LOD and streaming at runtime.

The product should exploit the best reusable technology for each layer rather than reimplementing mature algorithms for ownership's sake.

## Truth, inference and generation

WRE distinguishes three user-visible provenance classes:

1. **OBSERVED / RECONSTRUCTED** — supported by source observations and reconstruction evidence.
2. **INFERRED** — derived by geometric/temporal models with explicit uncertainty.
3. **GENERATED** — synthesized to improve visual continuity where observations do not support a direct reconstruction.

Generated completion can be valuable for cinematic or immersive experiences, but it must never be silently presented as measured reality.

Suggested experience profiles:

- **STRICT** — only supported reconstruction; holes are allowed.
- **REALISTIC** — conservative visual completion is allowed and marked internally.
- **CINEMATIC** — wider generative freedom for content creation.

## Automatic organization of media

The engine must be able to receive a disorganized set of media and build structure before reconstruction:

- detect image/video quality, duplicates and useful frames;
- identify observations likely to depict the same scene;
- geometrically verify ambiguous visual similarity;
- group or relate temporal observations;
- synchronize overlapping event videos where possible;
- detect specialist input classes such as 360, drone, rolling-shutter or strongly blurred footage.

Similarity alone must not silently fuse repeated façades, symmetric structures or unrelated look-alike places.

## Adaptive reconstruction instead of one mandatory pipeline

WRE is a router over interchangeable specialists. Different data profiles require different methods: a handful of photos, thousands of tourist images, a 30-minute phone walk, a 360 video, a multi-camera event and a century of archival imagery are not the same problem.

The router chooses a route based on data profile, requested quality, hardware budget, previous artifacts and quality-gate results. Cheap methods can provide previews; stronger or specialist methods can refine or replace them. Failure can trigger escalation rather than forcing a plausible-looking result.

External models and solvers live behind explicit adapters and stable domain contracts. A future method should be benchmarkable and replaceable without redesigning the product.

## Professional production behavior

WRE is a production engine, not a folder of research scripts. Every stage produces versioned artifacts connected in an artifact DAG. Each artifact records, as applicable:

- content-addressed inputs;
- producing adapter/model/checkpoint;
- parameters and software version;
- hardware/runtime context when relevant;
- quality metrics and confidence;
- dependency artifacts;
- output hashes and formats.

Unchanged artifacts are reusable. Interrupted heavy jobs should be resumable. New algorithms should be added through adapters and compared against retained benchmark fixtures.

## Runtime and distribution

The highest-quality master representation is not necessarily a runtime representation. WRE compiles a master scene into target-specific assets using combinations of:

- mesh/SDF/collision geometry;
- splat/radiance appearance;
- texture/material assets;
- spatial and temporal chunks;
- LOD hierarchies;
- compression;
- device-aware streaming.

Targets may include web, desktop, DCC tools, game engines, XR and offline rendered video.

## Human control

Automation is the default, not a prohibition on expert correction. The product should support three levels of interaction:

- **AUTO** — end-to-end automatic route and quality gates;
- **ASSISTED** — review and correct masks, clusters, cameras, changes or regions;
- **EXPERT** — inspect the artifact graph, override routes and tune specialist parameters.

A professional tool must make failure diagnosable instead of hiding it behind a final render.

## Non-goals

WRE is not defined as:

- a planetary database or MONDE subsystem;
- a clone of one photogrammetry product;
- a Gaussian-Splat-only tool;
- a single end-to-end neural model;
- a requirement to geolocate every scene on Earth;
- a system that invents unobserved reality by default;
- a promise that one internal algorithm is universally best.

## Canonical success statement

Given arbitrary real-world photos and videos, WRE should organize the observations, recover their spatial and temporal relationships, reconstruct the best-supported 3D/4D scene its evidence permits, produce both physical and photorealistic representations, preserve meaningful uncertainty and temporal states, compile efficient runtime assets, and let the user walk, fly, render or scrub time through the result.

The long-term product ambition is to be the strongest integrated system for turning ordinary real-world media into photorealistic, temporal, freely navigable 3D/4D scenes.