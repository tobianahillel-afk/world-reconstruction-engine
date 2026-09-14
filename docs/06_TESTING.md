# Testing and review strategy

## Lanes

### Fast PR lane

Target: seconds to a few minutes. Intended checks once L0 tooling is installed:

- lock/dependency consistency
- Ruff lint/format
- type checks
- unit tests
- small integration tests
- registry/state validation
- GitHub Actions linting

### Full/regression lane

Runs manually, on schedule and at milestone-sensitive changes. It will eventually include real COLMAP/LIMAP fixtures, fragment merge tests, geometry regression datasets and performance reports.

## Fixture harness

Fixture layout, determinism rules, metric comparators and result conventions are defined in [`09_FIXTURES.md`](09_FIXTURES.md). The machine-readable contracts live at:

- `registry/schemas/fixture.schema.json`
- `registry/schemas/regression-result.schema.json`

Tiny deterministic synthetic fixtures belong in the fast lane. Heavy captured/reconstruction fixtures remain outside ordinary PR CI until their owning lots introduce them deliberately.

## Foundational geometry fixtures

Planned early fixtures:

1. one small place with known/consistent reconstruction
2. two unrelated places -> exactly two fragments, no merge
3. two initially disconnected sets plus bridge observations -> validated merge
4. deliberately similar but distinct facades -> false merge rejection
5. known camera/geometry synthetic fixture for numeric error measurement

## Review gates

Every PR receives a focused review. Every lot receives a lot review before advancing. Every milestone receives an end-to-end review.

Correctness regressions may block merging. Performance regressions are initially reported rather than blocked unless a stable benchmark threshold is explicitly adopted.
