# Testing and review strategy

WRE uses layered evidence so routine development stays fast while changes to reconstruction quality remain objectively measurable.

## Test lanes

### Fast PR lane

Target: seconds to a few minutes.

Required categories as applicable:

- frozen dependency/lock consistency;
- Ruff lint/format;
- static type checks;
- domain/unit tests;
- small deterministic integration tests;
- repository/state/registry validation;
- GitHub Actions linting;
- mocked adapter failure/contract tests.

Fast CI should prove repository coherence and local correctness without requiring every GPU/model/native reconstruction stack.

### Native integration lane

Used for exact external environments such as COLMAP/FFmpeg/native libraries. It should prove that WRE's assumptions still match the pinned upstream API/behavior with real execution.

Native integration tests must not replace fast contract tests. They complement them.

### Regression / benchmark lane

Runs manually, on schedule and at milestone-sensitive/default-selection changes. It covers representative real/synthetic datasets, solver/model comparisons and retained metric vectors.

### Heavy production/runtime lane

Used for GPU-heavy reconstruction, long sequences, 4D scenes, large collections and representative runtime/device testing. Results are retained artifacts, not transient console output only.

## Fixture harness

Fixture layout, determinism rules, metric comparators and result conventions are defined in [`09_FIXTURES.md`](09_FIXTURES.md). Machine-readable contracts live under `registry/schemas/`.

Tiny deterministic fixtures belong in the fast lane. Captured/large/GPU-heavy datasets remain outside ordinary PR CI unless an owning item explicitly requires them.

## Required negative evidence

New contracts must test invalid/forbidden states, not only happy paths. Examples:

- ambiguous timezone remains ambiguous;
- unsupported CRS does not become WGS84;
- repeated-facade similarity does not force a scene merge;
- unsupported camera model fails/escales explicitly;
- weak geometry does not become collision surface;
- generated content cannot be decoded/persisted as reconstructed provenance;
- incompatible checkpoint/configuration cannot resume;
- later temporal states do not overwrite earlier states;
- runtime compilation cannot mutate master-scene identity.

## Benchmark dimensions

Benchmarks retain a vector of metrics rather than one universal score.

### Camera / geometry

- registered-view ratio;
- pose error/consistency;
- reprojection error;
- track support;
- depth multi-view consistency;
- independent point/surface error;
- scale/drift where meaningful;
- coverage/holes.

### Surface

- point-to-surface/Chamfer-type error where reference exists;
- normal consistency;
- coverage/holes;
- watertightness only where required;
- collision/navigation usability.

### Appearance

- held-out PSNR/SSIM/LPIPS or successor metrics;
- perceptual/ghosting/floater diagnostics;
- consistency with explicit geometry;
- robustness across exposure/illumination/view dependence.

### Dynamic / 4D

- dense tracking error;
- temporal geometry consistency;
- trajectory consistency;
- object persistence through occlusion;
- temporal flicker;
- free-viewpoint held-out quality;
- resource/latency behavior over sequence duration.

### Historical chronology

- cross-epoch alignment error;
- unchanged-region stability;
- true-change recall/precision where reference exists;
- false-change rejection under misregistration/occlusion;
- state-boundary/date uncertainty behavior.

### Runtime

- FPS/frame time;
- GPU/CPU memory;
- initial load/time-to-first-frame;
- streaming bandwidth/download size;
- chunk/LOD transition behavior;
- temporal scrubbing latency;
- collision/navigation correctness where applicable.

## Quality-mode testing

`PREVIEW`, `FAST`, `QUALITY` and `MASTER` may use different routes and budgets. Tests should verify:

- each mode reports its producing route/model/configuration;
- lower-cost modes do not change provenance semantics;
- escalation happens only through registered conditions;
- route termination is bounded;
- a faster mode may return less detail or unresolved output instead of lowering correctness requirements for the artifact it claims to produce;
- MASTER may run competing methods but must preserve comparison/provenance evidence.

## Adapter testing

Every external adapter should have:

1. pure contract/config validation tests;
2. failure-taxonomy mapping tests;
3. deterministic artifact/provenance identity tests;
4. mocked/substitute execution tests where useful;
5. real upstream integration coverage in the appropriate lane;
6. at least one regression fixture for a previously observed upstream/WRE incompatibility when applicable.

## Foundational fixture families

The v2 program should eventually cover at least:

1. controlled static room/object with known cameras/geometry;
2. outdoor building with repeated structure;
3. sparse-view scene;
4. Internet/tourist collection with distractors and illumination change;
5. large unordered collection;
6. short dynamic monocular video;
7. long handheld/walking sequence;
8. multiple unsynchronized videos of one event;
9. 360/equirectangular sequence;
10. drone/aerial sequence;
11. blur/rolling-shutter/HDR/low-light specialist cases;
12. historical unchanged and real-change epochs;
13. reflective/material-sensitive scene;
14. runtime large-scene LOD/streaming fixture;
15. generated-completion provenance fixture proving reconstructed/inferred/generated regions stay distinguishable.

## Review gates

Every PR receives a focused review. Every lot receives a capability review before advancement. Every milestone receives an end-to-end user-capability review.

Correctness and invariant regressions block merge. Performance regressions are initially reported unless a stable calibrated budget is explicitly adopted. Default-model/router changes require retained benchmark evidence appropriate to the profile they affect.

A lot review verifies composition and missing functionality, not merely that all item statuses are `done`. A milestone review should exercise the milestone's user-facing vertical slice whenever practical.
