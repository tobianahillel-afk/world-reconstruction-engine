# V2L15 — Dense depth / MVS / fusion review

**Review status:** PASS  
**Reviewed:** 2026-09-28  
**Lot:** `V2L15 — Dense depth / MVS / fusion`  
**Milestone:** `V2M2 — Reliable static physical reconstruction` remains in progress

## Review objective

Verify that V2L15.1 through V2L15.6 form one coherent solver-independent dense-depth evidence layer over the canonical V2 geometry contracts, while preserving exact camera/depth semantics, immutable provenance, explicit holes, unresolved source scale/frame state and a mandatory CPU-capable product path.

Passing this lot means WRE can represent dense depth evidence, retain an experimental COLMAP PatchMatch donor boundary, express multi-artifact fusion and learned-prior boundaries, measure support/holes/confidence descriptively, and compare retained canonical depth evidence against one trusted controlled reference without collapsing the evidence into a universal score or backend decision.

Passing V2L15 does **not** mean that a production dense-depth backend has been selected, that the optional CUDA PatchMatch donor has executed successfully on real GPU hardware, that a learned model has been chosen/executed, that depth has been fused into a physical surface, or that collision/measurement/navigation suitability has been established.

## V2L15 implementation sequence

1. PR #152 — `V2L15.1 Dense depth artifact contract`: pure immutable `DenseDepthArtifact` over one exact `GeometrySolutionCandidate`, canonical `DepthField` linkage and complete source ancestry.
2. V2L15.2 — optional COLMAP MVS / PatchMatch donor work:
   - PR #154 qualified the exact official `pycolmap-cuda12==4.2.0` accelerator dependency closure without execution;
   - PR #155 added the experimental `ColmapPatchMatchDenseDepthAdapter` path and canonical camera-z normalization boundary;
   - later focused preflight PRs exercised the exact official wheel, real `DepthMap` binary I/O and production normalization on CPU;
   - PR #164 recorded the unavailable-accelerator policy and marked real CUDA PatchMatch execution **deferred / unverified**, not passed.
3. PR #166 — `V2L15.3 Depth consistency fusion contract`: pure multi-`DenseDepthArtifact` fusion boundary with exact shared source geometry/provenance and no fusion algorithm.
4. PR #167 — `V2L15.4 Learned depth prior adapter boundary`: pure model-independent prior boundary; no learned model/checkpoint/runtime is selected or executed.
5. PR #168 — `V2L15.5 Dense depth coverage / holes / confidence`: deterministic read-only support reporting from existing validity/confidence evidence.
6. PR #171 — `V2L15.6 Controlled dense depth benchmark and lot review`: deterministic CPU-only controlled benchmark composing V2L15.5 support evidence with trusted-reference depth error evidence.

No V2L15 work item creates a `SurfaceModel`, mesh, TSDF/SDF, collision/navigation/measurement asset, appearance/material/environment representation, dynamic 4D state, chronology, `MasterScene` or `RuntimeScene`.

## Dense depth artifact findings

**PASS.**

V2L15.1 defines one `DenseDepthArtifact` as immutable canonical dense-depth evidence derived from one exact complete source `GeometrySolutionCandidate`.

The contract preserves:

- exact artifact identity and kind `geometry.dense_depth`;
- exact source geometry hypothesis;
- non-empty canonical unique `DepthField` values;
- exact source camera / observation / image-dimension linkage;
- caller-supplied depth-value convention, values, validity and optional confidence;
- partial camera coverage;
- producer/configuration identity;
- complete source-artifact ancestry;
- unresolved local frame / scale semantics from the source geometry.

It does not execute MVS, convert depth conventions, invent metric scale, fuse evidence, fill holes or create a surface.

## COLMAP PatchMatch donor findings

**PASS for the retained experimental contract/integration boundary; real CUDA execution remains deferred and unverified.**

The V2L15.2 donor uses a separate exact accelerator dependency identity around official `pycolmap-cuda12==4.2.0`. CPU-verifiable integration evidence demonstrates:

- exact official wheel/artifact identity;
- sparse-model/image-workspace preparation;
- real PyCOLMAP `DepthMap` binary I/O;
- production WRE dense-depth normalization against the real binding;
- camera/observation/dimension linkage;
- camera-z depth semantics;
- finite positive depth retained and non-positive values represented as unsupported canonical zero-depth pixels;
- partial camera coverage;
- no confidence synthesis;
- no stereo-fusion or physical-surface claim.

The real `pycolmap.patch_match_stereo` CUDA execution lane remains unavailable on the current trusted runner set. Under the repository's unavailable-accelerator rule this proof is explicitly **deferred / unverified**. It is not rewritten as success, and no shipping/default/performance claim may rely on it.

The mandatory V2 roadmap remains CPU-capable and does not require this optional accelerator donor to execute before later pure contracts can proceed.

## Fusion boundary findings

**PASS.**

V2L15.3 defines a solver-independent multi-candidate depth-fusion boundary rather than a fusion algorithm.

It requires:

- at least two canonical `DenseDepthArtifact` inputs;
- exact shared complete source geometry;
- compatible overlapping camera/observation/dimension/depth-convention semantics;
- new output evidence identities;
- exact retained input/transitive ancestry.

It permits partial or disjoint camera support and does not perform depth conversion, PatchMatch, stereo fusion, point generation, TSDF/SDF construction or surface creation.

## Learned prior boundary findings

**PASS — boundary only.**

V2L15.4 defines how a future learned dense-depth prior can enter WRE without making a model choice part of the core contract.

The boundary preserves exact source geometry, explicit supporting evidence, declared depth-value convention, canonical `DenseDepthArtifact` output and full ancestry.

No Torch/CUDA runtime, learned model, checkpoint, download, execution, calibration, scale promotion or automatic truth/default promotion exists in this lot. The learned prior remains **model-independent contract work only**.

## Coverage / holes / confidence findings

**PASS.**

V2L15.5 evaluates one existing `DenseDepthArtifact` without mutating it.

The evaluator reports separately:

- source-camera support ratio;
- emitted valid-pixel ratio;
- emitted hole-pixel ratio;
- confidence-bearing field ratio;
- mean confidence over valid pixels only when caller-supplied confidence exists.

Holes are exactly `DepthField.validity == False`; missing cameras are not fabricated as pixel holes. Confidence remains informational and is never treated as calibrated probability, correctness or truth.

## V2L15.6 controlled CPU benchmark

**PASS.**

The retained benchmark uses one checked-in synthetic controlled fixture with:

- one exact canonical three-camera source `GeometrySolutionCandidate`;
- one trusted fully-valid camera-z `DenseDepthArtifact` reference;
- three descriptive synthetic candidate artifacts;
- no solver/model/checkpoint execution;
- no GPU/accelerator dependency.

Fixture identity:

- fixture id: `fixture.dense-depth-controlled.v1`;
- fixture SHA-256: `7356108b43fe0be5b1c206afa5f14d35262ba5c7d8b51fb093a969dc3dad3f1a`.

Retained candidate evidence:

| Candidate | Camera support | Valid-pixel ratio | Hole ratio | Reference-valid coverage | Median absolute depth error | Median absolute relative depth error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense:candidate-full` | 1.0 | 1.0 | 0.0 | 1.0 | 0.5 | 0.25 |
| `dense:candidate-partial` | 0.6666666667 | 0.5 | 0.5 | 0.3333333333 | 0.5 | 0.2083333333 |
| `dense:candidate-zero-overlap` | 0.3333333333 | 0.0 | 1.0 | 0.0 | omitted | omitted |

The partial candidate demonstrates that missing cameras and existing holes reduce trusted-reference coverage without fabricating new candidate hole samples.

The zero-overlap candidate demonstrates that reference error metrics are omitted when no jointly valid candidate/reference pixels exist; zero or infinity is not fabricated.

The benchmark retains every V2L15.5 support observation and its original evaluator/input-artifact provenance unchanged. Reference-comparison observations use the deterministic V2L15.6 evaluator and the canonical candidate/reference `ArtifactRef` pair.

Confidence is never compared against the trusted reference and never changes reference error support.

Every candidate emits one canonical `BenchmarkRecord` with:

- the shared fixture identity;
- the shared CPU hardware identity;
- the candidate's exact producer identity;
- `QualityMode.QUALITY`;
- one canonical multi-dimensional `MetricVector`;
- no comparison-baseline/winner semantics.

The retained JSON explicitly records:

- complete support and reference-comparison metric vectors;
- exact fixture/reference/candidate/producer identities;
- omitted reference metrics for the zero-overlap case;
- `v2l15_2_real_cuda_patchmatch_execution = deferred_unverified`;
- `learned_depth_prior = model_independent_boundary_only`;
- descriptive-only evidence.

Its schema contains no winner, rank, overall score, preferred/default route, threshold, retry, fallback, `QualityDecision` or shipping-promotion field.

## Retained benchmark evidence

PR #171 implementation head:

- head: `c9ef5b72d1f06656675b2df773a20cf887f1d20a`;
- dense-depth-controlled-benchmark #12 — **PASS**;
- fast-ci #1211 — **PASS**;
- CodeQL #1040 — **PASS**;
- dependency-review #308 — **PASS**;
- colmap-integration #787 — **PASS**;
- colmap-mvs-integration #74 — **PASS** with the real accelerator-only PatchMatch execution still correctly deferred/skipped;
- da3-reference #220 — **PASS**;
- da3-execution-profile #179 — **PASS**;
- feed-forward-classical-comparison #196 — **PASS**.

Retained Actions artifact:

- name: `v2l15-6-controlled-dense-depth-benchmark`;
- artifact id: `10968259769`;
- GitHub artifact digest: `sha256:71ab48e760f2b2131a86089553caff152d8ddf31ef3a1a5390a899d9bb83cf8b`;
- retained JSON byte length: `43485`;
- retained JSON SHA-256: `45c11ed2b50868f5e0a59c7926c0db80b493fe452c77e4157dfe2f2a4d7f908a`.

## Cross-cutting invariant audit

**PASS.**

- Raw/source geometry and dense-depth evidence remain immutable.
- Derived benchmark evidence does not overwrite candidate/reference evidence.
- Depth-value conventions are never converted to manufacture comparability.
- Local frame and scale status are not rewritten or promoted.
- Missing cameras and unsupported pixels remain explicit missing/invalid support.
- Confidence remains optional descriptive evidence.
- Fusion and learned-prior contracts do not silently execute algorithms/models.
- Metric vectors remain multi-dimensional and are not collapsed into one score.
- No backend is selected as universal winner/default.
- The mandatory product path remains CPU-capable.
- Deferred accelerator evidence remains explicitly unverified.
- No `SurfaceModel`, collision, measurement or navigation suitability is created from depth evidence.
- Appearance and later scene/runtime responsibilities remain separate.

## Explicit V2L15 non-capabilities

After V2L15, WRE still does **not** yet provide:

- an explicit physical `SurfaceModel`;
- a baseline depth-fusion-to-surface implementation;
- collision/measurement/navigation suitability evidence;
- unsupported-region surface semantics;
- a surface benchmark;
- a selected universal dense-depth backend;
- real validated CUDA PatchMatch execution on a trusted supported accelerator runner;
- a concrete learned dense-depth runtime/model;
- automatic hole filling/inpainting;
- photorealistic appearance/material/environment reconstruction;
- dynamic 4D or chronology;
- `MasterScene` / `RuntimeScene` compilation.

These remain V2L16 and later responsibilities.

## V2L16.1 handoff decision

After this PASS, `V2L16.1 — SurfaceModel contract and intended use metadata` may become the sole `ready` item.

V2L16.1 must establish an immutable solver-independent explicit physical-surface metadata contract while keeping **intended use** distinct from **proven suitability**.

The first contract should preserve:

- exact surface artifact identity/provenance;
- exact source `GeometrySolutionCandidate` lineage;
- explicit local-frame and scale-status semantics;
- open representation identity such as mesh/SDF/surfels without hard-coding one solver;
- explicit canonical intended-use metadata for collision, navigation and/or measurement;
- complete canonical source-artifact ancestry.

Declaring an intended use must not itself prove collision, navigation or measurement suitability. Suitability metrics remain owned by V2L16.3.

V2L16.1 must not execute meshing/fusion/TSDF/SDF/Poisson/Open3D or another surface solver, generate/fill unsupported physical regions, create runtime collision/nav assets, infer metric scale, add suitability scores, or implement V2L16.2 or later behavior.

## Decision

**V2L15 lot review: PASS.**

WRE now has coherent dense-depth evidence, an optional experimental classical donor boundary, model-independent fusion/prior contracts, explicit support diagnostics and one retained controlled CPU benchmark.

The benchmark is descriptive and synthetic. No production dense-depth backend is promoted. Real CUDA PatchMatch execution remains deferred/unverified. V2M2 remains in progress and may advance only to the pure V2L16.1 `SurfaceModel` contract after PR #171's final lifecycle head passes all triggered exact-head gates.
