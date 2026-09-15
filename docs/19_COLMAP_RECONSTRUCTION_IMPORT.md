# L3.6 — COLMAP reconstruction import

L3.6 converts audited solver-native sparse reconstruction output from L3.5 into explicit WRE **estimated geometry**. It does not accept a `SpatialFragment`, infer world placement, resolve metric scale, or perform any L3.7 end-to-end acceptance fixture.

## Inputs

The importer consumes three provenance-bearing inputs:

1. the L3.5 `ColmapIncrementalReconstructionResult`, including content-addressed native model files;
2. the L3.2 `ColmapFeatureExtractionResult`, used only for the persisted `ObservationId <-> COLMAP image name` mapping;
3. a dedicated L3.6 `ReconstructionRun` whose inputs match the same raw-observation membership.

The L3.2 name mapping is required because COLMAP native sparse models know solver image names/IDs, not WRE `ObservationId` values. L3.6 never guesses observation identity from file-system position or a rewritten filename.

## Native artifact integrity

Before PyCOLMAP reads a model, L3.6 rechecks the L3.5 publication boundary:

- the model directory must be a real directory rather than a symlink;
- file membership must exactly match the L3.5 `ColmapSparseModelArtifact` manifest;
- every listed file must still match its recorded SHA-256 and byte length;
- symlinks and non-regular file entries are rejected.

The importer then reuses official `pycolmap.Reconstruction(path)` from the exact PyCOLMAP/COLMAP environment recorded by L3.5. WRE does not parse `cameras.bin`, `images.bin`, `points3D.bin`, rig files, or other native reconstruction files itself.

## Imported estimated geometry

Each native COLMAP sub-model becomes one independent `SparseReconstructionEstimate` with its own deterministic `LocalFrameId`. The imported model contains:

- producer-labelled camera calibration estimates;
- one `CameraPoseEstimate` per registered observation;
- sparse 3D points with reprojection error when available;
- point tracks mapped back to WRE observations and feature indices;
- per-object `DerivedArtifactProvenance`;
- an explicit local scale status.

Camera poses are stored as **camera-from-local-frame** rigid transforms. The COLMAP method is named `cam_from_world()`, but in WRE the source coordinate system is only the solver's local reconstruction frame. L3.6 therefore does not call it a world frame.

Camera calibration IDs are estimate IDs, not physical `CameraId` identities from source metadata. A COLMAP camera object may be shared by several registered observations, so its calibration provenance records those observations without asserting that WRE has identified one physical camera device.

## Scale and world placement

L3.5 runs the local monocular sparse baseline without absolute position priors. Therefore every imported model is marked:

`LocalScaleStatus.UNRESOLVED`

This is deliberate. A numerically stable COLMAP coordinate system is not evidence of metric scale, Earth alignment, geographic placement, or compatibility with another local reconstruction. Later anchoring/merging work must add explicit transforms and evidence.

## Multiple models and unresolved outcomes

`multiple_models=True` remains meaningful across the import boundary. L3.6 imports each native model separately and never forces disconnected or competing sub-models together.

If L3.5 returned zero models, L3.6 returns zero imported models. It does not invoke PyCOLMAP merely to manufacture an empty geometry object, and it does not create fallback poses or points.

## Explicit non-goals

L3.6 does **not**:

- create or accept `SpatialFragment` membership;
- merge solver sub-models;
- choose which competing reconstruction is true;
- infer metric scale;
- georeference a local model;
- create global/world coordinates;
- estimate covariance or confidence beyond the imported solver measurements;
- perform L3.7's end-to-end reconstruction fixture;
- modify the solver-native L3.5 artifact.

Fragment creation and lifecycle remain owned by L6. World anchoring remains owned by L9. Uncertainty expansion remains owned by L10. L3.7 only validates the complete L3 baseline after this importer exists.

## Deterministic identity

For each L3.5 model, L3.6 hashes a canonical manifest containing the model index and the path/hash/byte-length identity of every published native file. That digest derives deterministic WRE IDs for:

- the sparse reconstruction estimate;
- its local frame;
- camera calibration estimates;
- imported 3D points.

The identities therefore follow the exact native model bytes instead of incidental temporary paths.

## Tests

Fast tests cover domain invariants, observation-name mapping, tamper detection, environment mismatch and zero-model handling. The dedicated COLMAP integration lane also asks real `pycolmap==4.2.0` to write and reread a synthetic sparse reconstruction, then verifies that L3.6 imports the same registered-image and 3D-point counts while retaining unresolved local scale.
