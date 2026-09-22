# V2L13.5 — Preview versus classical camera comparison

**Evidence status:** PASS for the bounded V2L13.5 comparison  
**Reviewed:** 2026-09-22  
**Lot:** `V2L13 — Feed-forward geometry` remains in progress  
**Fixture:** `colmap-l3-end-to-end`

## Objective

Compare the exact experimental `da3.base_preview` route and the exact current
`colmap.incremental_precision_geometry` classical baseline on the same deterministic rendered
observations and the same trusted reference cameras, using only the directly comparable
frame/scale-invariant camera-pose metrics accepted in V2L13.4.

This evidence is descriptive. It defines **no winner**, **no default** route, no quality threshold,
no shipping-status promotion and no claim of universal solver superiority.

## Controlled evidence path

The comparison lane uses the existing versioned `colmap-l3-end-to-end` synthetic scene:

```text
one deterministic 12-view rendered PGM scene
  -> one shared ObservationId / source-byte set
  -> trusted reference cameras from the versioned fixture
       -> exact DA3-BASE CPU reference execution
       -> exact PyCOLMAP 4.2.0 SIFT/matching/verification/incremental mapping
  -> canonical CameraSolution tuples
  -> one shared projection-independent camera-pose evaluator
```

The fixture uses 12 reference observations, and both routes matched all 12 by exact
`ObservationId`.

The retained comparison artifact from combined workflow run #6 was produced from PR #116 head
`fe20184980c3fe290eec5724ee8dcfa23503bc19`.

## Exact route identities

### Preview route

- adapter: `da3.base_preview`
- model: `depth_anything_3.da3_base`
- source revision: `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`
- checkpoint: `depth-anything/DA3-BASE`
- checkpoint SHA-256:
  `e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5`
- reference execution: Python 3.12.12, CPU float32, exact V2L13.3 package closure
- configuration SHA-256:
  `dab5e2b186d080ff3a1e097a2faa309f3747fdadcec59af86e8ee81020ddc9ab`

### Classical route

- adapter: `colmap.incremental_precision_geometry`
- PyCOLMAP: `4.2.0`
- observed COLMAP build:
  `Commit be5e291 on 2026-08-31 without GPU support`
- feature configuration SHA-256:
  `ea75755939340ce9e91f07ad96980c37063de23dfb07ba0d1ae0739c5ea6fdb9`
- matching configuration SHA-256:
  `5d26c599fed8a0a9f6408086008614fa067120fa4a273f7571760365e9433d2c`
- geometric-verification configuration SHA-256:
  `92923dabeb8ac3d8994fe96eeb40a58714914f2bfd452d060a5dfaa17df8b3cf`
- incremental-mapper configuration SHA-256:
  `6be1580855b39f528eae9723dc16adddecd8da3241da324a10b32a663a95bc6a`
- projection family in this fixture: `SIMPLE_RADIAL`

The classical outcome contained exactly one canonical model. V2L13.5 rejects zero-model or
disconnected-multiple-model outcomes as unresolved comparison evidence; it does not select the
largest component, merge local frames or discard alternatives.

## Directly comparable camera-pose evidence

The same four metric descriptors are evaluated against the same trusted reference-camera tuple.

| Metric | DA3-BASE preview | COLMAP incremental |
| --- | ---: | ---: |
| `geometry.camera.observation_coverage_ratio` | 1.0 | 1.0 |
| `geometry.camera.relative_rotation_error_deg_median` | 1.4394171023° | 0.4603724911° |
| `geometry.camera.translation_pair_coverage_ratio` | 1.0 | 1.0 |
| `geometry.camera.relative_translation_direction_error_deg_median` | 12.2050836330° | 1.8912699527° |

On this controlled fixture, the two reported angular-error values are numerically lower for the
classical route. That is a factual statement about this one retained synthetic run only. It is not
converted into an overall score, rank, preferred route, default-selection rule or product claim.

The relative metrics compare pairwise camera relationships only. They do not compare absolute
translation vectors, camera centers, scale, `LocalFrameId` values or raw coordinates across the two
independent local frames.

## Projection-family limitation

DA3 emits canonical `pinhole` cameras for this bounded preview path. The classical route emits
COLMAP `SIMPLE_RADIAL` cameras.

Therefore V2L13.5 does **not** coerce `SIMPLE_RADIAL` to pinhole, drop distortion, or manufacture
symmetric focal/principal-point metrics. The V2L13.4 pinhole-only intrinsic metrics remain valid
route-specific DA3/reference evidence but are **not directly comparable projection families** in
this controlled DA3-versus-COLMAP record.

## Immutability and fail-closed findings

- the source observations and trusted reference-camera membership are shared;
- the COLMAP geometric-verification database remains byte-identical across incremental mapping;
- the DA3 checkpoint remains byte-identical across execution;
- the reviewed DA3 source checkout remains clean after execution;
- camera quality evaluation does not mutate the canonical DA3 geometry or classical camera values;
- a classical `NO_MODEL` or `DISCONNECTED_MODELS` outcome is an unresolved comparison, not a
  reason to select, rank, merge or align a model;
- unavailable intrinsic comparability remains explicit rather than being filled with zero, NaN,
  infinity or a fabricated penalty.

## Scope limitation

`colmap-l3-end-to-end` is a deterministic synthetic integration fixture and **not a representative natural-image quality benchmark**.

Accordingly this evidence does not establish:

- that either route is generally more accurate;
- a production PREVIEW default;
- a route threshold, retry or fallback rule;
- a shipping-status change;
- depth, point-map, surface, appearance or photometric superiority;
- timing, memory, throughput or hardware-efficiency conclusions;
- a generic competing-`GeometrySolution` comparison/consensus contract.

Broader representative quality evidence remains later benchmark work. Generic competing geometry
comparison belongs to V2L14. Learned execution performance/profiling belongs to V2L13.6.

## V2L13.5 decision

The bounded comparison objective is satisfied when the exact-head combined lane and ordinary CI
are green with this retained review.

The result is deliberately **no winner** and **no default**. V2L13.5 records comparable camera-pose
evidence and explicit projection limitations only.
