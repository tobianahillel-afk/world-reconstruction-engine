# L3.7 — COLMAP end-to-end fixture

L3.7 closes the implementation portion of the L3 COLMAP baseline with one deterministic, image-based, native PyCOLMAP fixture. It exists to prove that the independently tested L3.2 through L3.6 contracts compose coherently without bypassing the image feature and matching stages.

## Scope

The fixture exercises this exact chain with `pycolmap==4.2.0` in the dedicated COLMAP integration lane:

```text
versioned synthetic scene specification
  -> deterministic PGM image rendering
  -> L3.2 feature extraction
  -> L3.3 raw pair matching
  -> L3.4 geometric verification
  -> L3.5 incremental sparse reconstruction
  -> L3.6 solver-native reconstruction import
  -> fixture metric evaluation
```

It does not create a `SpatialFragment`, merge reconstructions, establish metric/world scale, georeference the model, add covariance, or begin L4 candidate retrieval.

## Versioned fixture

The canonical fixture lives at:

```text
tests/fixtures/synthetic/colmap-l3-end-to-end/
  fixture.json
  scene.json
```

`scene.json` stores the deterministic scene parameters rather than generated image bytes. The test renders the images from that versioned specification so the repository remains small while the fixture inputs and assumptions remain reviewable.

The scene uses:

- 12 grayscale PGM views at 640 x 480;
- a fixed seed;
- multiple unique high-contrast texture patches;
- multiple depths so the observations contain parallax rather than one flat translated texture;
- small known camera-center offsets;
- a nominal focal length of 768 px, equal to the L3.2 default focal prior `1.2 * image_width`.

The synthetic rendering is deliberately simple. It is intended to exercise WRE/COLMAP integration deterministically, not to benchmark photogrammetric accuracy on natural imagery.

## Evidence and immutability checks

The fixture uses the public WRE L3 APIs rather than direct solver shortcuts. In particular, it does not start from a pre-populated synthetic COLMAP reconstruction database.

Between stages it checks that parent evidence remains immutable:

- L3.3 does not mutate the L3.2 feature database;
- L3.4 does not mutate the L3.3 raw-match database;
- L3.5 does not mutate the L3.4 verified database;
- L3.6 does not mutate the solver-native L3.5 model files.

Observation identity continues to come from WRE `ObservationId` values and the persisted L3.2 image-name mapping. The final imported geometry remains local estimated geometry with `LocalScaleStatus.UNRESOLVED`.

## Fixture metrics

`fixture.json` declares explicit regression expectations for:

- observation count;
- exhaustive attempted-pair count;
- raw matched-pair count;
- geometrically verified-pair count;
- reconstructed model count;
- registered-image count;
- imported model count;
- imported observation count;
- imported 3D point count;
- unresolved-scale model count.

Exact invariants use `eq`. Solver-derived support counts use conservative `gte` thresholds so the fixture detects loss of reconstruction support without pretending that every internal feature/point count is a stable cross-platform API contract.

## Execution lane

The native fixture is intentionally `full` and runs only when `WRE_COLMAP_INTEGRATION=1`, which is set by `.github/workflows/colmap-integration.yml`. Ordinary fast CI still loads and validates the fixture metadata through the non-native test without installing PyCOLMAP.

The integration environment is pinned to the approved L3 environment (`pycolmap==4.2.0`, `numpy==2.5.3`) and retains the same one-thread/CPU/fixed-seed WRE baseline configurations used by L3.2-L3.5.

## Acceptance meaning

A passing L3.7 fixture demonstrates that the L3 contracts compose into one native sparse reconstruction/import path and that the declared evidence/provenance boundaries survive the composition.

It is not evidence that WRE is ready for arbitrary real-world reconstruction, false-match rejection at L5 quality, fragment acceptance at L6, absolute placement, dense reconstruction, or production-scale operation. Those remain owned by later lots.
