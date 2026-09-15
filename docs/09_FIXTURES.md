# Fixture and regression conventions

WRE fixtures are versioned evidence bundles used to test deterministic behavior and quality contracts without hiding scenario assumptions in test code.

## Fixture layout

Each fixture lives in its own directory under `tests/fixtures/` and contains exactly one `fixture.json` metadata file plus the input assets it declares.

```text
tests/fixtures/
  synthetic/
    <fixture-id>/
      fixture.json
      <declared input files>
  captured/
    <fixture-id>/
      fixture.json
      <declared input files or external references>
```

The canonical metadata contract is `registry/schemas/fixture.schema.json`. Runtime loading is implemented by `wre.regression.load_fixture` and deliberately performs the checks needed by the fast lane without forcing a heavy schema runtime dependency.

Required metadata includes a schema version, stable fixture id, fixture kind/profile, intended lane, determinism flag, optional integer seed, declared inputs, metric expectations and provenance. Input paths must be relative and may not escape the fixture directory.

## Determinism

Fast fixtures must be deterministic. If a fixture or runner uses randomness, its seed must be explicit and every consuming layer must use it. Correctness results must not depend on wall-clock time, unordered filesystem traversal or ambient machine state.

Benchmark fixtures for nondeterministic GPU/model systems may report distributions/repeated runs where necessary, but exact model/checkpoint/configuration/hardware identity must be retained.

## Metrics and tolerances

Simple deterministic expectations may use:

- `eq`: absolute difference from target within `abs_tolerance`;
- `lte`: observed value at most `target + abs_tolerance`;
- `gte`: observed value at least `target - abs_tolerance`.

Floating-point tolerances are explicit. Rich benchmark records may carry metric vectors beyond the bootstrap regression schema; the owning work item must version any expanded schema before use.

## Lanes

- `fast`: tiny deterministic fixtures in ordinary PR CI;
- `native`: exact external-environment integration fixtures;
- `regression`: broader correctness suites;
- `benchmark`: comparative quality/resource evaluation;
- `full`: heavy end-to-end GPU/native/runtime workflows.

A fixture belongs in the fastest lane that can execute reliably without making routine development slow or flaky.

## Scenario metadata

Fixtures should identify the data profile they are intended to represent rather than relying only on filenames. Relevant profile fields may include:

- unordered static photos;
- sparse view;
- large Internet collection;
- short dynamic video;
- long streaming video;
- multi-video event;
- 360/equirectangular;
- drone/aerial;
- historical/multi-epoch;
- repeated/symmetric architecture;
- blur/rolling shutter/HDR/low light;
- reflective/water/material-sensitive;
- runtime/LOD/streaming.

Not every fixture needs every field; the schema should allow only fields required by the owning benchmark contract.

## Quality modes

A benchmark can declare applicable `PREVIEW`, `FAST`, `QUALITY` and/or `MASTER` modes. Expected behavior is mode-specific:

- PREVIEW may accept lower detail/coverage but must preserve provenance and failure semantics;
- FAST may bound resource use;
- QUALITY may require stronger held-out validation/refinement;
- MASTER may run competing solvers/specialists and retain comparison evidence.

A lower-cost mode must not satisfy a test by silently changing an artifact's provenance or semantic claim.

## Planned fixture families

Important families include:

- **scene organization:** true same-scene pairs, unrelated look-alikes and repeated façades;
- **routing:** data profiles whose expected selected route/escalation is known;
- **geometry:** known cameras/depth/surface plus disconnected/low-overlap failure cases;
- **surface:** collision/navigation suitability and unsupported-region holes;
- **appearance:** held-out views, illumination/exposure variation, ghosting/floater cases;
- **dynamic:** known motion, static/dynamic contamination, object occlusion/persistence;
- **multi-video synchronization:** known clock/audio/visual offsets plus ambiguous cases;
- **long sequence:** drift/loop/chunk behavior;
- **absolute anchoring:** valid/conflicting GPS/GCP/reference constraints;
- **historical unchanged:** independent epochs with no physical change;
- **historical true change:** controlled structural change;
- **historical false apparent change:** misregistration/occlusion/uncertainty;
- **historical contradiction:** incompatible dating/evidence;
- **generated completion:** strict separation of reconstructed/inferred/generated regions;
- **runtime:** LOD/streaming/memory/FPS/collision/timeline behavior.

## Current fixtures

`tests/fixtures/synthetic/tiny-smoke/` is intentionally trivial. It exercises the bootstrap fixture loader/comparators and is not a reconstruction quality claim.

`tests/fixtures/synthetic/colmap-l3-end-to-end/` is the retained native COLMAP baseline fixture. It renders deterministic synthetic observations and exercises the existing COLMAP feature extraction through estimated-geometry import as one chain. In v2 it remains valuable as a compatibility/baseline fixture; it is not the universal reconstruction benchmark.

## Fixture integrity

Fixtures are test evidence and must not silently mutate to fit new implementations. A material fixture change requires review of whether expectations, baseline results and benchmark comparability must be versioned.

Do not weaken a production invariant or acceptance threshold solely to make a fixture pass. Fix the fixture when it is invalid; fix the implementation when the fixture exposes a real regression.
