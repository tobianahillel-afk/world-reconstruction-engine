# Reuse and dependency policy

## Decision rule

Before implementing an algorithm or integrating a model:

1. identify the WRE capability contract being satisfied;
2. search `registry/dependencies.yaml` and `docs/14_TECHNOLOGY_SELECTION.md`;
3. inspect maintained upstream implementations and current APIs;
4. check code license, model/checkpoint license, transitive/native components, platform support, maintenance and reproducibility;
5. prefer an adapter around a proven implementation;
6. benchmark candidates behind the same WRE contract;
7. if custom foundational implementation is still justified, write an ADR before coding it.

A research paper being newer or better on one benchmark is not by itself a reason to replace a stable default. Promotion is data-profile-specific and must account for quality, reliability, resources, integration cost and licensing.

## Ownership boundary

WRE should own:

- stable domain contracts;
- artifact identity/provenance;
- orchestration and routing;
- normalization across solvers/models;
- quality metrics and gates;
- failure taxonomy and fallback policy;
- benchmark fixtures/results;
- project/master/runtime state;
- compatibility and export/import boundaries.

WRE should normally reuse external implementations for mature foundational algorithms such as feature extraction, matching, SfM/BA, MVS, tracking, rasterization, codecs, robust registration and specialized learned reconstruction.

## Candidate roles

Dependency status and algorithm role are separate concepts. A technology may be:

- `baseline` — stable reference/compatibility path;
- `primary_candidate` — current preferred method to evaluate for a capability/profile;
- `specialist` — route used only for specific data/failure conditions;
- `research_watch` — promising but not yet ready/licensed/reproducible;
- `runtime_tooling` — production/runtime infrastructure;
- `benchmark_only` — useful for comparison but not approved for shipping.

No candidate becomes a default merely by appearing in documentation.

## Current reusable families

### Classical/precision geometry baseline

- **COLMAP** — point features/matching, two-view geometry, incremental/global/hierarchical SfM, bundle adjustment, MVS and several candidate-pairing mechanisms.
- **LIMAP** — structural point+line/plane/vanishing-point/wireframe reconstruction when benchmark evidence justifies it.
- **OpenMVG / AliceVision / OpenSfM** — selective alternative geometry/validation paths where they provide distinct value.
- **GTSAM / TEASER++ / Open3D** — factor-graph, robust registration and refinement capabilities when those contracts become active.

COLMAP remains a retained baseline/compatibility adapter in v2; it is not the entire architecture.

### Media / coordinates

- **FFmpeg** — mature video demux/decode/audio extraction; WRE owns selection/profile/orchestration policy.
- **PROJ / GeographicLib** — coordinate-reference-system and geodesy utilities when needed.

### Retrieval / matching

Potential reusable families include:

- COLMAP sequential/spatial/vocabulary mechanisms;
- SALAD and modern visual-place-retrieval embeddings;
- DINO-family global/semantic descriptors;
- hloc-style retrieval/local verification;
- LightGlue with appropriate feature extractors;
- MASt3R-family matching/3D correspondence methods;
- RoMa/LoFTR-class difficult-view matchers.

**Important v2 rule:** retrieval similarity is proposal evidence, but a learned matching or geometry model is not globally restricted to “proposal only.” Its authority depends on the owning capability contract and quality policy. Scene-identity fusion still requires the evidence prescribed by the scene-organization contract, and repeated/symmetric structures require explicit disambiguation.

### Feed-forward / stateful geometry

Candidates include DA3-family systems, VGGT-Ω, Pi3/Pi3X, MapAnything-class systems, CUT3R, LongStream, ZipMap, Scal3R, SLAM3R and future equivalents.

These may serve as previews, priors, specialist paths **or primary `GeometrySolution` producers** when the relevant benchmark/quality contract supports that use. The core architecture must not assume every learned geometry result is merely a proposal to COLMAP.

### Surface / appearance

- mature MVS/fusion/mesh systems for explicit physical surface;
- gsplat and compatible Gaussian backends for appearance;
- Nerfstudio/Splatfacto as reusable appearance integration/reference tooling;
- OMeGa/SurfaceSplat/MeshSplatting-class hybrids as benchmark candidates;
- robust in-the-wild appearance systems where released/licensed.

**Appearance is a first-class WRE representation.** A NeRF/Gaussian system may produce a canonical `AppearanceModel` for a master scene when its contract/quality criteria are satisfied. It does **not** become collision/measurement geometry automatically.

### Dynamic / 4D

Candidates include dense trackers such as AllTracker/CoTracker-family methods; fast 4D preview methods such as MoVieS/MoRe/D4RT-class systems; quality dynamic methods such as MotionScale/Shape-of-Motion/MoSca/ProDyG/MOSAIC-GS-class systems; and persistent-object/dynamic-surface research families.

WRE owns the stable dynamic contracts, temporal quality policy, provenance, orchestration and runtime composition. It does not hard-code one paper as “the 4D architecture.”

### Historical chronology

- py4dgeo for mature multi-epoch point-cloud change measurement where applicable;
- Neural Scene Chronology/Cross-Temporal-3DGS/LTGS/GaME/real-time change-detection families as conceptual or implementation candidates depending on release/license maturity.

WRE owns `TemporalState` / `ChangeEvent` semantics and long-term chronology regardless of which renderer/change detector is used.

### Runtime / LOD / viewers

- SuperSplat/PlayCanvas as strong web-runtime/viewer references and potential integrations;
- glTF/GLB and mature mesh tooling for explicit geometry exchange;
- SPZ/SOG-class compact splat formats where compatible;
- Unreal/Unity integrations through explicit runtime contracts.

Do not write a custom renderer, codec or compression format before measuring that maintained alternatives are inadequate for the target contract.

## Development dependencies

Development tooling is reproducible through the committed `uv.lock`. CI must use frozen sync rather than silently resolving a new environment. Dependabot updates the native `uv` ecosystem and GitHub Actions on a grouped weekly cadence so maintenance remains visible without generating excessive PR noise.

Supply-chain controls and settings that are deliberately outside the fast lane are documented in [`08_SECURITY.md`](08_SECURITY.md).

## Learned-model registry requirements

Before a learned model/checkpoint can become a production default, its registry entry must include at least:

- canonical source repository/project;
- exact code version/commit;
- checkpoint/model identity and hash when available;
- code and checkpoint licenses reviewed separately;
- supported hardware/runtime;
- required preprocessing/postprocessing;
- artifact contract produced;
- reproducibility/determinism notes;
- benchmark status by relevant data profile;
- shipping status.

Floating `latest` checkpoints are forbidden in reproducible paths.

## Licensing

The repository's final project license has not yet been selected. Until it is, do not make a copyleft dependency mandatory/core without an explicit architecture/license decision. Optional subprocess adapters must still be reviewed for distribution implications.

For learned components, review code, model weights, upstream extractors, datasets and bundled assets separately. A framework being Apache/BSD/MIT does not prove every checkpoint or dependency is safe for the intended distribution.

## Replacement rule

A maintained candidate can replace a current default for a particular profile/quality mode only after:

1. integration behind the same WRE contract;
2. exact version/license/reproducibility review;
3. representative benchmark evidence;
4. failure-class inspection, not aggregate score only;
5. no regression in required provenance/invariant behavior;
6. routing/registry update with an explicit fallback.

A replacement should normally require changing adapter/registry/router policy, not redesigning core scene semantics.
