# COLMAP incremental reconstruction

L3.5 consumes the immutable verified COLMAP database produced by L3.4 and runs the approved COLMAP 4.2.0 incremental sparse-SfM pipeline. Its output is a **solver-native reconstruction artifact**. It is not yet imported WRE geometry.

## Reuse boundary

WRE reuses the official PyCOLMAP 4.2.0 `incremental_mapping(...)` entry point and `IncrementalPipelineOptions`.

COLMAP owns:

- incremental image registration;
- initial-pair selection and initialization;
- triangulation;
- local/global bundle adjustment;
- creation of one or more sparse reconstruction models;
- serialization of native reconstruction files.

WRE does not implement those algorithms in L3.5. WRE owns orchestration, provenance, deterministic configuration, source/database integrity, output ownership, and solver-artifact auditing.

The exact approved external environment remains `pycolmap==4.2.0` / COLMAP 4.2.0. L3.5 adds no new core runtime dependency.

## Input contract

The parent input is the L3.4 verified database, identified by its recorded SHA-256 and byte length. L3.5 verifies both before any mapper work and verifies the database again after mapping.

The database is treated as immutable. L3.5 never intentionally modifies it. If the bytes change during the stage, the run fails rather than accepting the result.

L3.5 also receives the image-like observations whose membership exactly matches L3.4 provenance. Every source file is re-hashed against its persisted observation asset before it is exposed to COLMAP.

COLMAP image names are solver identity and must match the database exactly. WRE therefore stages verified source bytes into a private temporary image directory using the exact existing COLMAP image names. Unsafe paths, duplicate names, membership mismatches, or changed source bytes fail explicitly.

## Output ownership

The requested reconstruction output directory must not already exist.

WRE atomically claims it with a fresh directory creation before invoking COLMAP. This prevents a concurrent invocation from being overwritten or later removed by this invocation's cleanup logic.

If L3.5 fails after claiming the directory, it removes only the directory it owns. A pre-existing output is rejected and left untouched.

## Deterministic baseline

The L3.5 baseline uses:

- `min_num_matches = 15`;
- `ignore_watermarks = false`;
- `multiple_models = true`;
- `max_num_models = 50`;
- `max_model_overlap = 20`;
- `min_model_size = 10`;
- `init_num_trials = 200`;
- structure-less registration fallback enabled;
- structure-less-only mode disabled;
- point-color extraction disabled;
- reconstruction threads = `1`;
- random seed = `0`;
- focal-length refinement enabled;
- principal-point refinement disabled;
- extra-parameter refinement enabled;
- rig-sensor refinement enabled;
- bundle-adjustment GPU disabled;
- absolute position priors disabled;
- `load_all_images = false`;
- no runtime limit by default (`-1`).

The top-level `num_threads` and `random_seed` controls are passed through COLMAP's incremental pipeline. In COLMAP 4.2.0 these settings propagate into the mapper/triangulation path, while the single-thread setting also reaches Ceres bundle-adjustment solver options.

The full WRE-owned baseline configuration is canonically serialized and SHA-256-addressed through its producing `ReconstructionRun`.

## Multiple models and unresolved outcomes

`multiple_models` stays enabled deliberately. Disconnected or insufficiently connected evidence must not be forced into a single reconstruction merely to increase coverage.

COLMAP can therefore return zero, one, or multiple native sparse models.

A zero-model result is a valid explicit **unresolved solver outcome**. L3.5 records an empty model tuple and `has_reconstruction = false`; it does not fabricate poses, points, fragments, or a fallback world placement.

## Solver-native artifact

For every model returned by PyCOLMAP, L3.5 retains only minimal audit metadata at the WRE boundary:

- COLMAP model index;
- relative model directory;
- number of registered images;
- number of 3D points;
- every solver-native file path under the model directory;
- SHA-256 and byte length of every file.

The native files remain the authoritative L3.5 reconstruction product.

L3.5 does **not** import or reinterpret:

- camera intrinsics;
- registered camera/image poses;
- 3D point coordinates;
- tracks;
- reprojection errors;
- local-frame semantics;
- `SpatialFragment` membership.

Those conversions belong to L3.6, where the importer can define explicit WRE provenance and geometry contracts without conflating solver output with canonical state.

## Output audit

After `incremental_mapping(...)` returns, WRE requires:

- a mapping from non-negative integer model indices to reconstruction objects;
- output-directory membership exactly matching those model indices;
- every model directory to be a real directory, not a symlink;
- each returned reconstruction to pass COLMAP's own `is_valid()` check;
- non-negative integer registered-image and 3D-point counts;
- every retained solver-native entry to be a regular file;
- no symbolic links in model output;
- canonical, unique file ordering;
- the L3.4 database to remain byte-identical to its pre-run identity.

Contradictory or malformed output fails the stage. WRE does not silently normalize it.

## Epistemic boundary

A successful COLMAP sparse reconstruction is still **estimated solver geometry**, not accepted world truth.

L3.5 does not assert that:

- a returned model is a canonical `SpatialFragment`;
- separate models should be merged;
- reconstructed scale is metric;
- coordinates are Earth/world aligned;
- camera poses have WRE acceptance semantics;
- geometry survives later competing-model, cycle, fragment, merge, anchor, uncertainty, or temporal checks.

This keeps the repository's observation / derived-evidence / estimated-geometry separation intact.

## Testing

Fast tests use a strict fake PyCOLMAP boundary to verify:

- exact option wiring;
- deterministic single-thread/seed policy;
- parent-database immutability checks;
- source-byte validation and image-name membership;
- output ownership and cleanup;
- zero-model unresolved behavior;
- model/output membership auditing;
- solver-file hashing;
- public API export.

The dedicated COLMAP integration lane installs exact `pycolmap==4.2.0` and exercises the real incremental mapper on COLMAP's own synthetic-dataset mechanism. This verifies the real API/environment and solver-native output contract without pretending to replace the larger geometry-quality fixture planned for L3.7.

## Next boundary

L3.6 will import the solver-native reconstruction into explicit WRE estimated-geometry models. L3.5 must not pre-empt that importer by embedding COLMAP camera/point objects or silently promoting solver geometry into canonical fragments.
