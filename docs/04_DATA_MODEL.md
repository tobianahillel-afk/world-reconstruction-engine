# Initial data model

The initial model is intentionally small. Add fields/types only when a work item requires them.

Planned core concepts:

- `Observation`
- `ImageObservation`
- `VideoFrameObservation`
- `Camera`
- `FeatureSet`
- `PairCandidate`
- `GeometricMatch`
- `Track`
- `CameraPoseEstimate`
- `SpatialFragment`
- `FragmentPlacement`
- `MergeHypothesis`
- `SpatialConstraint`
- `ReconstructionRun`

Every derived object must be able to identify its source observations/evidence and producing run/version. IDs from different domains must not be casually interchangeable.
