# COLMAP geometric verification

L3.4 consumes the immutable raw-match database produced by L3.3 and asks the approved COLMAP 4.2.0 environment to estimate two-view geometry. The output is **derived geometric evidence**. It is not an accepted fragment relation, camera trajectory, or world reconstruction.

## Reuse boundary

WRE uses the official PyCOLMAP 4.2.0 `geometric_verification(...)` entry point for already matched image pairs. That binding uses:

- `GeometricVerifierOptions` for verifier execution;
- `ExistingMatchedPairingOptions` to enumerate existing raw-match rows;
- `TwoViewGeometryOptions` and COLMAP's own robust estimators for F/E/H classification, inlier selection and optional relative pose.

WRE does not implement RANSAC, F/E/H estimation, pose decomposition or a binary decoder for COLMAP's geometry database fields. The verified `TwoViewGeometry` objects are read back through the official PyCOLMAP Database API.

The exact external environment remains the L3.1 baseline: `pycolmap==4.2.0` with the reviewed COLMAP 4.2.0 build assumptions. No new runtime dependency is added by L3.4.

## Immutable parent artifact

The L3.3 matched database has a recorded SHA-256 and byte length. L3.4 verifies those bytes before any solver call, copies the verified parent to a fresh target database, rechecks the copy, and runs COLMAP only on that copy.

The L3.3 artifact is never mutated in place. An existing L3.4 target is rejected. Any exception removes the partial L3.4 database and leaves the parent untouched.

## Deterministic baseline configuration

The baseline intentionally fixes all exposed stochastic/concurrency controls:

- verifier threads: `1`;
- existing-pair batch size: `1000`;
- RANSAC seed: `0`;
- RANSAC threads: `1`;
- RANSAC max error: `4.0` pixels;
- RANSAC confidence: `0.999`;
- RANSAC minimum trials: `100`;
- RANSAC maximum trials: `10000`;
- RANSAC minimum inlier ratio: `0.25`;
- dynamic-trial multiplier: `3.0`;
- minimum accepted COLMAP inliers: `15`;
- two-view minimum inlier ratio: `0.0`;
- E/F ratio threshold: `0.95`;
- H ratio threshold: `0.8`;
- watermark detection enabled with the COLMAP baseline thresholds;
- Sampson refinement enabled;
- relative-pose computation requested;
- rig verification disabled;
- existing relative poses are not trusted;
- forced-H mode disabled;
- DEGENSAC disabled;
- recursive multiple-model estimation disabled.

The last two exclusions are architectural, not missing features. L5 owns stronger evidence geometry, including competing F/E/H hypotheses and DEGENSAC/AC-RANSAC policy. L3.4 is the simple COLMAP baseline and must not pre-empt that later evidence layer.

The full configuration is serialized canonically and SHA-256-addressed through the producing `ReconstructionRun`.

## Evidence retained per pair

For every raw-match pair, L3.4 records an immutable WRE summary containing:

- canonical source `ObservationId` pair and image names;
- raw L3.3 match count;
- COLMAP `TwoViewGeometryConfiguration` classification;
- the exact COLMAP inlier feature-index pairs;
- optional F, E and H matrices;
- optional `cam2_from_cam1` as a finite 3x4 rigid-transform matrix;
- optional median triangulation angle;
- whether COLMAP produced estimated camera intrinsics on either side.

The complete verified COLMAP database is also content-addressed, so solver-native information not promoted into the summary remains reproducibly available as part of the artifact.

## Canonical orientation

COLMAP pair IDs are ordered by COLMAP image IDs, whereas WRE pair identity is canonicalized by `ObservationId`. Those two orderings are not guaranteed to agree.

L3.4 therefore maps COLMAP image IDs back through the L3.3 image-name/observation evidence. When the COLMAP orientation is opposite WRE's canonical observation order, WRE calls COLMAP's own `TwoViewGeometry.invert()` before exporting F/E/H/inlier/pose evidence. This prevents a subtle but serious orientation error where a matrix or relative pose could be attached to the reversed WRE pair.

## Epistemic boundary

A non-degenerate COLMAP two-view result is still solver-derived evidence, not final truth. L3.4 does **not** assert that:

- the pair belongs to one accepted `SpatialFragment`;
- the relative pose is globally consistent;
- one COLMAP model dominates all competing hypotheses;
- the relation survives triplet/cycle checks;
- the geometry is metric or world-anchored;
- the pair is safe to merge into canonical world state.

In particular, L3.4 does not run mapping, triangulation tracks, bundle adjustment, fragment construction or world-graph optimization. L3.5 next owns incremental reconstruction as another solver stage. L5 later owns WRE's richer evidence-geometry model and consistency policies.

## Integrity checks

After COLMAP returns, WRE requires:

- exactly one two-view geometry record for every raw-match pair;
- no geometry pair outside L3.3 raw-match membership;
- unchanged raw match counts between L3.3 and L3.4;
- unique canonical WRE pair identity;
- known COLMAP configuration enum values;
- finite matrix/pose values;
- non-negative unique inlier feature indices;
- inlier count not exceeding the raw-match count.

Any mismatch fails the stage and removes the partial output database rather than normalizing contradictory evidence.

## Testing

Fast tests use a small fake PyCOLMAP boundary to verify configuration, immutable-parent behavior, cleanup, canonical pair orientation, exact option wiring and membership guards without making PyCOLMAP a core Python dependency.

The dedicated COLMAP integration lane installs exact `pycolmap==4.2.0` and `numpy==2.5.3`, performs real L3.2 feature extraction, real L3.3 raw matching and real L3.4 geometric verification on deterministic synthetic image data, while checking that the L3.3 parent database remains byte-identical.
