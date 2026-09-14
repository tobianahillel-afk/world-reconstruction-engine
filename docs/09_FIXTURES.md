# Fixture and regression conventions

WRE fixtures are small, versioned evidence bundles used to test deterministic behavior without hiding assumptions in test code.

## Fixture layout

Each fixture lives in its own directory under `tests/fixtures/` and contains exactly one `fixture.json` metadata file plus the input assets it declares.

```text
tests/fixtures/
  synthetic/
    <fixture-id>/
      fixture.json
      <declared input files>
```

The canonical metadata contract is `registry/schemas/fixture.schema.json`. Runtime loading is implemented by `wre.regression.load_fixture` and deliberately performs the checks needed by the fast lane without introducing a JSON Schema dependency.

Required metadata includes a schema version, stable fixture id, fixture kind, intended lane, determinism flag, optional integer seed, declared input files, metric expectations and provenance. Input paths must be relative and may not escape the fixture directory.

## Determinism

Fast fixtures must be deterministic. If a fixture or runner uses randomness, its seed must be explicit and all code consuming it must use that seed. Fixture results must not depend on wall-clock time, unordered filesystem traversal or ambient machine state.

The canonical regression result does **not** require timestamps. Operational envelopes may add timing or execution metadata later, but correctness comparisons must remain reproducible.

## Metrics and tolerances

Each expected metric uses one of three comparators:

- `eq`: absolute difference from the target must be within `abs_tolerance`;
- `lte`: observed value must be at most `target + abs_tolerance`;
- `gte`: observed value must be at least `target - abs_tolerance`.

Tolerances are explicit. Floating-point behavior must never rely on accidental exact equality.

The canonical result contract is `registry/schemas/regression-result.schema.json`. Results contain the observed metrics and an ordered check record explaining why each expectation passed or failed.

## Lanes

- `fast`: tiny deterministic fixtures that run in ordinary PR CI.
- `regression`: broader correctness suites, generally scheduled or manually requested.
- `full`: heavy native/geometry workflows such as future COLMAP/LIMAP fixtures.

A fixture belongs in the fastest lane that can run reliably without making routine development slow or flaky.

## Planned advanced fixture families

Future owning lots should add explicit fixture metadata rather than hiding scenario semantics in test names. Important families include:

- **routing equivalence:** the same accepted/rejected/unresolved truth outcome under FAST/STANDARD/ESCALATED where each route has gathered sufficient evidence, plus expected escalation when a cheaper route cannot;
- **absolute anchoring:** correct GPS/GCP/reference-fragment placement, scale resolution where applicable, and deliberately conflicting anchors that must not be silently averaged;
- **temporal unchanged:** independently reconstructed epochs describing unchanged geometry;
- **temporal true change:** known synthetic or controlled physical change with an interval/point that can be checked;
- **temporal false apparent change:** misregistration, occlusion or reconstruction uncertainty that must not be promoted to a physical change;
- **temporal contradiction:** incompatible dating or competing epoch/alignment evidence that should remain unresolved.

Heavy captured datasets for these cases belong in regression/full lanes. Small synthetic invariants may be represented in the fast lane when they remain deterministic and cheap.

## Current bootstrap fixture

`tests/fixtures/synthetic/tiny-smoke/` is intentionally trivial. It contains four hand-authored points and exercises the fixture loader, all three comparators, tolerance handling, deterministic result serialization and failure reporting. It is a harness test, not a reconstruction algorithm or quality claim.
