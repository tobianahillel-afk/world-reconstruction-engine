# V2L17 — Color and photometric pipeline review

**Review status:** PASS  
**Reviewed:** 2026-10-08  
**Lot:** `V2L17 — Color and photometric pipeline`  
**Milestone:** `V2M3 — Photorealistic master scene and interactive runtime`

## Review objective

Verify that V2L17.1 through V2L17.5 form one coherent, provenance-first photometric evidence layer from exact source metadata through a bounded decoded-color convention, explicit caller-supplied normalization, descriptive endpoint/over-range diagnostics and compatibility evidence.

Passing V2L17 means WRE can carry one explicitly supported sRGB RGB8 observation through a deterministic linear-sRGB reference path while keeping source declarations, capture hints, decoded-pixel interpretation, photometric normalization and diagnostic validity evidence separate and immutable.

Passing V2L17 does **not** create an `AppearanceModel`, estimate scene radiance truth, provide universal color management or authorize automatic exposure/white-balance/color repair.

## V2L17 implementation sequence

1. PR #181 — `V2L17.1 Source photometry metadata contract`: exact observation-bound source color/exposure/white-balance evidence with explicit absent, unknown, invalid and resolved states.
2. PR #185 — `V2L17.2 Explicit decoded-to-linear color convention reference`, with follow-up fix PR #187: one bounded sRGB RGB8 to linear-sRGB binary64 reference path with exact decoded artifact/materialization lineage.
3. PR #186 — `V2L17.3 Explicit photometric normalization artifact`: caller-supplied EV and RGB gains only; capture metadata never synthesizes normalization factors.
4. PR #189 — `V2L17.4 Photometric validity evidence`: separate source-low endpoint, source-high endpoint and normalized-above-one masks plus fail-closed compatibility assessment.
5. PR #190 — `V2L17.5 Deterministic photometric pipeline fixture`: one CPU-only retained fixture composing the accepted V2L17.1–V2L17.4 contracts over ready, unresolved, rejected, foreign and unsupported cases.

No V2L17 item creates static appearance, materials, environment, geometry routing, scene selection or a runtime representation.

## V2L17.1 source photometry findings

**PASS.**

`SourcePhotometryMetadata` binds interpreted photometric evidence to one exact `ObservationId` and immutable `ObservationMetadata.raw_entries`.

Source color, exposure and white-balance families retain independent interpretation states:

- `absent`;
- `unknown`;
- `invalid`;
- `resolved`.

Resolved values require exact retained source evidence. Unknown or invalid evidence remains explicit and cannot expose a valid-looking interpreted value.

Capture ISO, shutter time, aperture, exposure compensation, white-balance mode and color temperature are descriptive provenance. They do not become exposure-normalization or RGB-gain factors.

## V2L17.2 decoded color convention findings

**PASS as one bounded reference route, not universal color management.**

The accepted route requires:

- exact decoded-image pyramid artifact and materialization identity;
- exact observation lineage;
- RGB8 packed decoded pixels;
- source-pixel orientation;
- independently declared sRGB full-range RGB8 encoding;
- a complete resolved source-color declaration matching sRGB / BT.709 D65 / sRGB transfer / identity matrix / full range;
- linear-sRGB binary64 working and output conventions;
- the reviewed standard-library reference backend.

The reference applies only the IEC 61966-2-1 inverse sRGB transfer and retains a distinct immutable converted buffer with source, content and derived identities.

Absent/unknown color evidence remains unresolved. Contradictory, unsupported or profile/ICC-dependent evidence is rejected rather than guessed.

The route does not decode arbitrary image files and is not an ICC engine, camera-response estimator or chromatic-adaptation system.

## V2L17.3 explicit normalization findings

**PASS.**

Photometric normalization is a separate immutable stage over one exact ready V2L17.2 result.

A ready normalization requires both:

- explicit caller-supplied exposure adjustment in EV;
- explicit caller-supplied positive finite RGB white-balance gains.

Missing factors remain `unresolved`. Non-finite, zero, negative or unrepresentable factors are `rejected`.

The reference applies only:

1. linear-light `2^EV` scale;
2. caller-supplied per-channel RGB gains.

Values above 1.0 are preserved. No clipping or rescaling is performed.

Source ISO, shutter, aperture, exposure compensation, WB mode and color temperature never synthesize hidden defaults, unity gains or an inferred EV.

## V2L17.4 validity and compatibility findings

**PASS as descriptive evidence only.**

V2L17.4 keeps three masks separate:

- `source_low_endpoint_candidate` — exact decoded RGB8 channel value 0;
- `source_high_endpoint_candidate` — exact decoded RGB8 channel value 255;
- `normalized_above_one` — normalized linear value strictly greater than 1.0.

Source endpoint masks are **candidates**, not proof of sensor clipping. They do not prove lost information, recovered highlights or invalid pixels.

The normalized over-range mask is strictly `value > 1.0`; exact `1.0` is not flagged.

`PhotometricCompatibilityAssessment` retains compatible, unresolved and incompatible evidence with exact reasons and identity. Foreign observations/plans, rejected evidence and unsupported conventions remain explicit. The assessment does not drop, repair, reroute, rank or select observations.

## V2L17.5 controlled end-to-end fixture

**PASS.**

The checked-in canonical fixture has SHA-256:

`a532be3128f254bb9aa212cce6d2de48b14fc6ea2d5ca6a9aa58500b1e6a5308`

It contains five canonically ordered cases:

| Case | Color | Normalization | Compatibility | Purpose |
| --- | --- | --- | --- | --- |
| `ready-endpoints-overrange` | ready | ready | compatible | endpoint candidates, exact 1.0 and >1.0 evidence |
| `unresolved-missing-factors` | ready | unresolved | unresolved | descriptive metadata cannot synthesize factors |
| `rejected-explicit-factors` | ready | rejected | incompatible | invalid caller-supplied gain fails closed |
| `foreign-normalization` | ready | ready | incompatible | cross-observation normalization remains foreign |
| `unsupported-color` | rejected | not constructed | incompatible | unsupported source color remains explicit |

The ready case uses RGB8:

`[0, 255, 255, 64, 255, 128]`

with explicit EV `1.0` and gains `[1.0, 0.5, 2.0]`.

Its normalized values include:

- exact `1.0` channels that remain unflagged as over-range;
- one `4.0` channel that remains preserved and is flagged by `normalized_above_one`;
- no clipping, repair or recovery.

Its masks are:

- low endpoint candidate: `[1, 0, 0, 0, 0, 0]`;
- high endpoint candidate: `[0, 1, 1, 0, 1, 0]`;
- normalized above one: `[0, 0, 1, 0, 0, 0]`.

The fixture runner:

- constructs only accepted WRE V2L17.1–V2L17.4 values;
- validates canonical checked-in JSON;
- reads no source image payload;
- performs no arbitrary media decoding;
- uses no NumPy, Pillow, OpenCV, colour-science, LittleCMS, torch, model, solver, subprocess, network or accelerator;
- emits deterministic canonical JSON only;
- retains source identities, color/normalization statuses and identities, all three mask values/content/identities, and compatibility status/reasons/identity;
- contains no repair, recovered-color/highlight, inferred factor, winner, score, rank, route, default, `QualityDecision`, `AppearanceModel` or shipping-promotion field.

Repeated execution is byte-identical. Tests also prove changed source bytes, raw evidence, explicit factors and mask contents move the corresponding downstream identities without mutating upstream objects.

### Retained fixture evidence

Implementation head `7e21cafbc52813b168c3b2ec2f326a0c68e04e17` produced retained controlled evidence:

- workflow: `photometric-pipeline-controlled`;
- run id: `37830122442`;
- artifact id: `11572639409`;
- artifact name: `v2l17-5-photometric-pipeline-controlled`;
- artifact digest: `sha256:0ada11007d230c61ebeaef67881c1cdeebf62d9a43de2de03549ce6d31cc3edb`;
- artifact size: 3,616 bytes;
- dedicated generation, nine fixture regressions and artifact upload all passed;
- CodeQL `37830122425` passed;
- dependency-review `37830122419` passed.

The final lifecycle/review head must independently pass full repository validation, fast CI, CodeQL and the dedicated controlled workflow before merge.

## Separation-of-responsibility audit

**PASS.**

Across V2L17:

- raw capture metadata remains evidence, not normalization policy;
- decoded pixel encoding is separate from source metadata declarations;
- decoded-to-working color conversion is separate from exposure/WB normalization;
- explicit normalization is separate from validity diagnostics;
- endpoint diagnostics are separate from proven sensor clipping;
- normalized over-range is separate from validity/repair policy;
- compatibility is evidence, not routing or selection;
- all derived stages preserve exact upstream lineage and immutable identities;
- unsupported or incomplete evidence fails closed;
- physical `SurfaceModel` remains independent from photometric evidence;
- static `AppearanceModel`, materials and environment remain later responsibilities.

## Explicit V2L17 non-capabilities

V2L17 does **not** provide:

- universal ICC/profile interpretation;
- arbitrary camera-color calibration or camera-response estimation;
- scene-radiance truth;
- chromatic adaptation;
- automatic exposure estimation;
- automatic white balance;
- normalization-factor inference from EXIF/video metadata or neighboring observations;
- HDR merge;
- tone mapping;
- highlight recovery;
- clipping or saturation repair;
- inpainting;
- automatic rescaling of values above one;
- proof that RGB8 0 or 255 means sensor clipping;
- proof that an over-range normalized value is invalid;
- observation ranking, dropping, routing or fallback;
- `QualityDecision`;
- `AppearanceModel`;
- Gaussian/radiance reconstruction;
- materials or environment recovery;
- geometry modification;
- safety, measurement or collision decisions;
- a universal default for unknown/unsupported cameras.

The bounded sRGB CPU reference is one explicitly supported path and is not promoted into a universal camera pipeline.

## V2L18.1 handoff decision

After this PASS, only `V2L18.1 — Canonical AppearanceModel contract` may become ready.

V2L18.1 must define a pure solver-independent immutable metadata/provenance envelope for one static photorealistic appearance artifact, distinct from `SurfaceModel`, `MaterialModel` and `EnvironmentModel`.

The contract may bind appearance to one exact source `GeometrySolutionCandidate`, an optional exact physical `SurfaceModel` support artifact when applicable, the exact source observations/photometric evidence used, one open representation token, exact local-frame/scale semantics, producer identity and canonical source-artifact ancestry.

It must not:

- implement a Gaussian/radiance solver;
- treat appearance as collision/measurement/navigation geometry;
- infer or resolve geometry scale;
- create materials/environment semantics;
- add routing/default/quality policy;
- render or compile runtime assets;
- promote generated completion into observed/reconstructed appearance.

## Decision

**V2L17 lot review: PASS.**

WRE now has an explicit, fail-closed color/photometric evidence substrate suitable for later appearance reconstruction without conflating capture metadata, decoded color, normalization, clipping candidates or compatibility.

The final PR #190 lifecycle/review head must independently pass repository validation, the dedicated photometric controlled workflow and all applicable exact-head checks before merge. No V2L18 implementation is included in this review commit.
