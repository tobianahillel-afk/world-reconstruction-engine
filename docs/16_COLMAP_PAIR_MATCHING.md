# COLMAP pair matching

L3.3 consumes the audited feature database produced by L3.2 and adds raw descriptor correspondences only. It deliberately stops before any fundamental/essential/homography estimation, inlier classification, pose inference or reconstruction.

## Reuse and upstream boundary

WRE reuses official PyCOLMAP/COLMAP 4.2.0. It does not implement SIFT matching. PyCOLMAP exposes `match_exhaustive(...)` together with `FeatureMatchingOptions.skip_geometric_verification`. Upstream documents this flag as skipping the geometric-verification stage and forwarding raw matches unchanged. WRE relies on that explicit supported path rather than reimplementing a descriptor matcher or trying to delete accepted geometry after the fact.

The approved dependency model from L3.1/L3.2 remains unchanged: PyCOLMAP is an exact external solver environment, lazily loaded by solver-backed operations and exercised by the dedicated COLMAP integration lane.

## Immutable parent artifact

The L3.2 database has its own recorded SHA-256 and byte length. L3.3 must not mutate that artifact in place because doing so would invalidate its recorded identity and blur feature extraction with matching.

Before matching, WRE therefore:

1. resolves the L3.2 feature database;
2. hashes it and requires exact equality with the recorded L3.2 SHA-256 and byte length;
3. creates a fresh output database by byte-for-byte copying the verified parent;
4. verifies the copied bytes before the solver is allowed to mutate the copy.

Failures remove the partial L3.3 output while leaving the L3.2 parent untouched. An existing target database is rejected rather than silently reused.

## Deterministic baseline

The first L3 matching baseline is intentionally conservative and explicit:

- exhaustive pairing over the current finite feature database;
- SIFT brute-force matcher;
- CPU device;
- exactly one matching thread;
- fixed random seed;
- `max_num_matches = 32768`;
- Lowe ratio threshold `0.8`;
- maximum descriptor distance `0.7`;
- cross-check enabled;
- CPU brute-force matching enabled;
- guided matching disabled;
- rig verification disabled;
- geometric verification explicitly skipped;
- exhaustive pairing block size `50`.

These values are part of a canonical WRE configuration document whose SHA-256 is retained with the result. Smarter candidate-generation policies are intentionally not introduced here: L4 owns sequence/GPS/visual/graph candidate retrieval and strategy routing.

## Output evidence

The L3.3 result records:

- the producing `ReconstructionRun` and canonical source observation provenance;
- the exact compatible PyCOLMAP/COLMAP environment identity;
- the matching configuration SHA-256;
- the parent L3.2 feature-database SHA-256;
- the new matched-database path, SHA-256 and exact byte length;
- the exhaustive attempted-pair count;
- canonical WRE observation pairs for rows present in COLMAP's raw `matches` table and their match counts.

COLMAP pair IDs remain solver-local. WRE uses the official PyCOLMAP `pair_id_to_image_pair` conversion and the audited L3.2 image-name mapping to recover WRE observation identity.

A pair having raw descriptor correspondences is evidence only. It is **not** evidence that the pair depicts the same rigid scene, belongs to one fragment, or supports a valid camera pose.

## Hard geometric boundary

After matching, WRE opens the resulting database read-only and requires the `two_view_geometries` table to remain empty. Any geometric contamination is treated as an L3.3 failure and the partial output is removed.

L3.3 must not:

- estimate F, E or H models;
- run RANSAC/DEGENSAC or any two-view geometric verifier;
- classify geometric inliers;
- run guided matching;
- estimate relative or absolute pose;
- build tracks;
- triangulate points;
- run a mapper or bundle adjustment;
- create or mutate `SpatialFragment` geometry;
- import a COLMAP reconstruction.

L3.4 owns geometric verification. Later lots add stronger competing-model evidence and consistency checks; raw L3.3 matches never bypass those acceptance stages.

## Testing

Fast tests use a small fake PyCOLMAP boundary to verify configuration, parent immutability, output cleanup, provenance and the no-geometry invariant without making PyCOLMAP a mandatory core dependency.

The dedicated COLMAP integration lane installs exact `pycolmap==4.2.0` and `numpy==2.5.3`, extracts real SIFT features from a deterministic synthetic image pair, runs real exhaustive CPU matching with geometric verification disabled, verifies that raw matches exist, verifies that `two_view_geometries` remains empty, and confirms that the original L3.2 feature database is byte-identical after L3.3 completes.
