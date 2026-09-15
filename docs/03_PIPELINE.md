# Adaptive reconstruction pipeline

WRE does not have one mandatory reconstruction chain. It has a common preparation path followed by scenario-specific route graphs and shared quality gates.

## Common preparation

```text
raw photos / videos
  -> immutable media ingest + hashing
  -> decode / frame extraction
  -> metadata / audio / camera hints
  -> quality analysis
       blur / exposure / resolution / duplicates / frame diversity
  -> media profiling
       still / video / 360 / drone / dynamic / long sequence / specialist conditions
  -> retrieval embeddings and candidate relationships
  -> scene graph / scene clustering
  -> temporal grouping hypotheses
  -> geometric relationship verification where needed
  -> route selection
```

The common path must not assume all media belongs to one scene.

## Router inputs

The router considers at least:

- number and type of observations;
- overlap and visual diversity;
- estimated scene connectivity;
- static/dynamic content;
- video duration and number of cameras;
- 360/drone/fisheye/rolling-shutter indicators;
- blur, HDR/exposure and image quality;
- timestamp/synchronization availability;
- historical versus continuous-event time;
- existing project artifacts;
- requested PREVIEW/FAST/QUALITY/MASTER mode;
- available GPU/CPU/RAM/storage budget;
- previous quality-gate failures.

The router returns an auditable route graph and escalation conditions.

## Route A — unordered static photo collection

Typical input: tourist photos, personal photo sets, archival images from one approximate epoch.

```text
scene cluster
  -> retrieval / candidate graph
  -> local feature matching + geometric verification
  -> fast camera/depth proposal
  -> precision/global camera solve
  -> geometry quality gate
  -> dense depth / fusion
  -> explicit surface
  -> photorealistic appearance
  -> optional material/environment recovery
  -> master-scene assembly
```

For QUALITY/MASTER, competing camera/depth solutions may be compared or refined rather than selecting the first successful solver.

## Route B — sparse-view static scene

Typical input: only a handful of usable views.

```text
sparse observations
  -> foundation-model geometry/depth prior
  -> sparse-view specialist
  -> camera/depth refinement
  -> uncertainty/coverage map
  -> surface if support is adequate
  -> appearance reconstruction
  -> optional conservative completion
```

Low coverage must remain visible in confidence/coverage artifacts even when a generative view looks plausible.

## Route C — very large photo collection

Typical input: hundreds or thousands of unordered Internet/tourist photos.

```text
large media set
  -> scalable retrieval index
  -> visual clustering / duplicate suppression
  -> scalable/stateful geometry proposal
  -> component/submap reconstruction
  -> global alignment/refinement
  -> long-tail hard-case escalation
  -> static/dynamic distractor decomposition
  -> epoch separation if dates differ materially
  -> dense/surface/appearance per chunk
  -> chunk merge + LOD compile
```

The engine must avoid all-pairs work when scalable retrieval/graph methods can reduce the candidate set without unacceptable recall loss.

## Route D — short monocular dynamic video

Typical input: seconds of phone/action-camera video with moving objects.

```text
video
  -> keyframes + dense tracks
  -> fast 4D preview
  -> camera/depth/motion proposal
  -> static/dynamic decomposition
  -> quality 4D reconstruction
  -> persistent-object reconstruction
  -> optional dynamic surface
  -> dynamic appearance
  -> temporal quality gate
  -> free-viewpoint runtime asset
```

A fast feed-forward result can be shown immediately while a stronger optimization route continues only when requested by the selected quality mode.

## Route E — long monocular video

Typical input: long walk, drive or scan sequence.

```text
long video
  -> adaptive keyframing
  -> streaming/stateful geometry
  -> drift monitoring
  -> local chunk reconstruction
  -> loop/overlap detection
  -> global/chunk refinement
  -> static/dynamic separation
  -> per-chunk dense/surface/appearance
  -> spatial LOD compile
```

Memory use must scale with an active window/state rather than requiring every frame to remain fully resident.

## Route F — multiple videos of the same event

```text
video set
  -> metadata/timecode hypotheses
  -> audio synchronization
  -> visual-event synchronization
  -> human/object motion synchronization when useful
  -> per-camera calibration/trajectory
  -> cross-camera track graph
  -> joint or merged 4D reconstruction
  -> persistent object/motion solution
  -> dynamic surface + appearance
  -> free-viewpoint replay
```

Synchronization is an estimated artifact with quality metrics, not an implicit assumption.

## Route G — 360 / equirectangular capture

```text
360 media
  -> native omnidirectional calibration/profile
  -> omnidirectional camera/depth solve
  -> 360-specific reconstruction
  -> environment separation
  -> surface + appearance
```

Do not force all omnidirectional media through naive perspective conversion when a native route is better.

## Route H — drone / aerial dynamic capture

```text
UAV media
  -> aerial motion/profile detection
  -> camera trajectory / scale handling
  -> static background + moving-object separation
  -> aerial geometry refinement
  -> dense/surface/appearance
  -> optional event-time 4D
```

Large altitude/scale variation is treated explicitly.

## Route I — long-term chronology

Typical input: the same place photographed over months, years or decades.

```text
media over time
  -> scene identity relation
  -> timestamp/date evidence
  -> epoch/state clustering
  -> per-epoch camera/geometry solution
  -> cross-epoch registration
  -> change detection
  -> change confidence gate
  -> TemporalState / ChangeEvent graph
  -> cross-temporal geometry/appearance transfer where valid
  -> timeline runtime compilation
```

Long-term change is not represented as arbitrary smooth motion by default. Structural events may be discrete.

## Specialist preprocessing/escalation routes

The data profiler may add specialist stages for cases such as:

- strong motion blur;
- alternating exposure/HDR;
- low light;
- fisheye or unusual optics;
- rolling shutter;
- reflective/translucent surfaces;
- water;
- weak texture;
- repeated/symmetric architecture;
- very sparse overlap;
- very long sequences;
- severe dynamic distractors.

These are route extensions, not reasons to make the default pipeline permanently heavy.

## Geometry path

Regardless of route, geometric products remain explicit:

```text
CameraSolution
  + DepthField / PointMap
  -> GeometrySolution
  -> dense fusion/refinement
  -> SurfaceModel
```

The exact solver may differ. Geometry quality is evaluated independently from appearance.

## Appearance path

```text
GeometrySolution / SurfaceModel
  + source images
  + exposure/appearance normalization
  -> AppearanceModel
  -> optional material decomposition
  -> EnvironmentModel
  -> photorealistic render quality gate
```

Appearance can use splat/radiance representations without replacing the explicit physical surface.

## Dynamic path

```text
camera/depth solution
  + dense tracks
  + segmentation/decomposition
  -> DynamicEntity candidates
  -> Trajectory / MotionField
  -> persistent 4D representation
  -> optional dynamic SurfaceModel
  -> dynamic AppearanceModel
```

Occluded entities may persist when the chosen model supports object permanence; predicted intervals retain uncertainty.

## Historical path

```text
same-place observations
  -> temporal evidence
  -> TemporalState candidates
  -> cross-state alignment
  -> change detection
  -> ChangeEvent
  -> state-specific geometry/appearance
```

A later state never silently overwrites an earlier retained state.

## Quality gates and escalation

Every route contains explicit gates. Typical flow:

```text
stage
  -> metrics
  -> PASS
     or ACCEPT_WITH_WARNINGS
     or RETRY with adjusted configuration
     or ESCALATE to stronger/specialist solver
     or UNRESOLVED
```

Examples:

- poor registered-view ratio -> broaden retrieval/matching or try alternate geometry model;
- depth inconsistency -> alternate depth prior/MVS or retain holes;
- weak dynamic tracking -> stronger tracker or reduce temporal span;
- appearance ghosting -> improve masks/poses/exposure model before more splat densification;
- suspected historical change -> cross-epoch alignment check before accepting a change event.

## Preview versus final work

A fast stage can feed later stages without becoming authoritative by accident.

Example:

```text
PREVIEW geometry
  -> user/automatic viability decision
  -> reuse as prior/initialization
  -> QUALITY precision solve
  -> MASTER multi-solver refinement
```

Artifacts record which quality mode and route produced them.

## Runtime compilation

After master-scene assembly:

```text
MasterScene
  -> target profile
       web / desktop / game / XR / offline
  -> spatial chunking
  -> temporal chunking
  -> mesh simplification/collision
  -> splat/texture compression
  -> LOD hierarchy
  -> streaming manifest
  -> RuntimeScene
```

Runtime compilation is allowed to simplify representation while preserving a link to the master artifacts.

## Human review loop

At configurable gates, assisted/expert workflows may:

- split/merge scene clusters;
- reject bad frames;
- correct masks;
- add/remove correspondence constraints;
- select a preferred camera solution;
- edit reconstruction regions;
- correct temporal grouping;
- approve/reject detected changes;
- mark generated completion boundaries.

Manual decisions are versioned inputs to downstream artifacts.

## Failure philosophy

The engine should diagnose failure rather than produce a polished but structurally wrong scene. It is valid to return:

- insufficient coverage;
- disconnected scene components;
- unresolved camera ambiguity;
- uncertain temporal ordering;
- unsupported surface regions;
- dynamic regions not reconstructable at requested quality.

The product can still offer an optional generated visualization, but its provenance class remains distinct from supported reconstruction.