# COLMAP feature extraction

L3.2 is the first WRE work item that invokes a reconstruction-engine operation. Its scope is deliberately narrow: extract local image features with the approved PyCOLMAP 4.2.0 environment and preserve the result as derived solver evidence. Pair matching, geometric verification, camera-pose estimation and reconstruction remain later work items.

## Dependency boundary

PyCOLMAP remains an **exact external solver environment**, not a mandatory import-time dependency of the core WRE package. L3.1 already pins the supported binding and COLMAP identity to 4.2.0 and exercises the official wheel in a dedicated integration lane. L3.2 keeps that architecture: production extraction lazily loads the approved solver, validates its identity first, and fails explicitly when it is unavailable or incompatible.

This is an intentional runtime decision rather than an omitted lock step. The ordinary WRE runtime stays usable for ingestion, persistence and evidence inspection without installing a native reconstruction engine. The dedicated COLMAP integration environment pins the concrete wheel inputs used to execute solver-backed tests. Any future packaging/bundling mode requires a fresh dependency review.

## Upstream operation

PyCOLMAP 4.2.0 exposes `extract_features(database_path, image_path, image_names, camera_mode, reader_options, extraction_options, device, cancellation_token)`. The operation creates/populates the COLMAP SQLite database with image/camera records plus keypoints and descriptors.

WRE uses this API rather than implementing SIFT. The L3.2 baseline uses:

- SIFT features;
- CPU device explicitly, not AUTO/GPU;
- one extraction thread;
- `CameraMode.PER_IMAGE` so COLMAP does not silently share one camera identity between distinct WRE observations;
- an explicit canonical list of staged image names;
- no pair matching or two-view geometry operation.

COLMAP image/camera rows created as a side effect of feature extraction are **solver-local scaffolding**. They are not promoted into WRE `Camera`, calibration, pose or geometry facts by L3.2.

## Input snapshot

Feature extraction operates on immutable WRE image-like observations (`ImageObservation` or `VideoFrameObservation`) plus local source paths supplied explicitly by the caller.

Before the solver runs, WRE copies each source into a temporary extraction snapshot with a deterministic generated name, hashes the copied bytes and verifies SHA-256 plus byte length against the persisted observation asset. The solver therefore consumes a verified snapshot rather than an unchecked mutable path.

Input observations are canonicalized by `ObservationId`; duplicate observation IDs are rejected. A source path may contain arbitrary host naming, but generated COLMAP image names are WRE-controlled and carry no source-path semantics.

## Configuration identity

The WRE feature-extraction configuration is immutable and serialized canonically before invocation. Its SHA-256 digest is retained in the result. The exact PyCOLMAP/COLMAP version is a separate part of provenance, so a configuration digest is never interpreted without its producer environment.

The baseline intentionally exposes only WRE-owned knobs needed for a deterministic first extractor. It does not mirror every upstream option prematurely. Later changes to the supported configuration surface must be explicit and test-covered.

## Output evidence

The extraction result records:

- the compatible PyCOLMAP/COLMAP environment identity;
- canonical source observation IDs;
- the canonical configuration SHA-256;
- the produced COLMAP database path, SHA-256 and exact byte length;
- the deterministic COLMAP image name associated with each observation;
- per-image keypoint/descriptor row counts read back from the database;
- confirmation that the `matches` and `two_view_geometries` tables remain empty.

The database is a solver artifact containing derived feature evidence. Its content hash identifies the concrete artifact produced by one run; L3.2 does **not** claim that SQLite bytes or floating-point feature values are universally bit-identical across operating systems or CPU implementations.

## Epistemic boundary

Keypoints/descriptors are derived evidence. They do not establish that two observations depict the same place, do not define tracks, and do not estimate geometry.

L3.2 must not:

- call a matching API;
- populate `matches`;
- populate `two_view_geometries`;
- accept camera intrinsics inferred by COLMAP as WRE truth;
- estimate F/E/H models;
- run a mapper or bundle adjustment;
- create camera poses, 3D points or `SpatialFragment` geometry;
- import a COLMAP reconstruction.

L3.3 owns pair matching. L3.4 owns geometric verification. L3.5 owns incremental reconstruction, and L3.6 owns reconstruction import.
