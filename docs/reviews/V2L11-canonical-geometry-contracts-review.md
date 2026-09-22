# V2L11 — Canonical camera/depth/geometry contracts review

**Review status:** PASS  
**Reviewed:** 2026-09-22  
**Lot:** `V2L11 — Canonical camera/depth/geometry contracts`  
**Milestone:** `V2M2 — Reliable static physical reconstruction` remains in progress

## Review objective

Verify that V2L11.1 through V2L11.6 compose into one solver-independent canonical static-geometry foundation with explicit local coordinate frames, camera-from-local-frame extrinsics, depth/point products, coherent geometry assembly, truthful local-scale semantics, a bounded one-way legacy donor bridge, and a deterministic coordinate-convention regression fixture.

Passing this lot must not be interpreted as a completed static reconstruction system. V2L11 defines and composes canonical geometry values only. It does not yet provide the V2L12 classical precision solver adapter, feed-forward geometry, dense reconstruction, global alignment, scale solving, world anchoring, physical surface generation, appearance reconstruction, runtime compilation, dynamic 4D, or historical chronology.

## V2L11 implementation sequence

V2L11 was intentionally split into six bounded work items:

1. PR #101 — `V2L11.1 Canonical CameraSolution contract`: immutable per-observation camera calibration/pose values in an explicit `LocalFrameId`, with open projection-model identity, exact image dimensions/intrinsics, camera-from-local-frame rigid extrinsics, uncertainty hooks and metrics.
2. PR #102 — `V2L11.2 DepthField validity confidence contract`: immutable per-observation depth raster semantics bound to one canonical `CameraSolutionId`, with explicit depth-value convention, validity, optional confidence and metrics.
3. PR #103 — `V2L11.3 PointMap contract`: immutable ordered local-frame dense-or-sparse 3D point predictions with explicit source-observation support, optional confidence and metrics.
4. PR #104 — `V2L11.4 GeometrySolution assembly local scale contract`: immutable coherent local-frame geometry hypothesis referencing canonical cameras/depth/points with explicit `UNRESOLVED` or `METRIC` local-scale status.
5. PR #105 — `V2L11.5 One way legacy SparseReconstructionEstimate conversion`: deterministic in-memory donor bridge into canonical `CameraSolution`, `PointMap`, and `GeometrySolution` values without permanent reverse compatibility.
6. PR #106 — `V2L11.6 Coordinate convention regression fixture and lot review`: deterministic asymmetric coordinate fixture locking the accepted camera-from-local-frame convention and lot composition boundary.

No V2L11 item executes a reconstruction solver, aligns multiple local frames, estimates metric scale, anchors geometry to Earth/world coordinates, constructs a physical surface, or chooses between competing geometry hypotheses.

## Canonical contract composition findings

**PASS.**

### CameraSolution

- `CameraSolution` is per observation and solver-independent.
- It carries explicit `ObservationId`, `LocalFrameId`, projection-model identity, image dimensions, finite intrinsic parameters, rigid rotation/translation, uncertainty artifact references and metrics.
- Its extrinsic convention is exactly:

  `x_camera = rotation_matrix * x_local + translation_xyz`

- The local frame makes no metric, Earth, geographic, reference-fragment or cross-frame compatibility claim.
- A solver calibration is not promoted into physical `CameraId`, device identity or rig identity.
- Camera value validation does not solve, invert, normalize, align or anchor the supplied pose.

### DepthField

- `DepthField` is bound explicitly to one `ObservationId` and one canonical `CameraSolutionId`.
- It carries image dimensions, an open depth-value convention, row-major depth samples, validity, optional normalized confidence and metrics.
- It does not duplicate camera extrinsics/intrinsics or own `LocalFrameId`; the exact camera solution reference owns those semantics.
- A depth field is not automatically a point map, fused geometry, physical surface or metric/world claim.

### PointMap

- `PointMap` carries one explicit `LocalFrameId`, canonical source-observation support, an ordered sequence of local 3D positions, optional normalized confidence and metrics.
- Point order is retained exactly as supplied after validation.
- The contract performs no hidden point generation, sorting, deduplication, fusion, scale resolution, topology inference or coordinate-frame conversion.
- It does not imply a mesh, collision surface, measurement surface, geographic location or fragment placement.

### GeometrySolution

- `GeometrySolution` assembles one coherent local-frame geometry hypothesis from canonical camera, depth and point product identities.
- It owns an explicit `GeometryScaleStatus` with exactly `UNRESOLVED` and `METRIC`.
- `UNRESOLVED` is a valid geometry state rather than a hidden failure.
- `METRIC` means local coordinate distances are meter-valued; it does not define origin, orientation, Earth alignment, geographic placement or compatibility with another local frame.
- The contract contains no scale factor because scale conversion/rescaling is outside this value contract.
- A `GeometrySolution` is still geometry evidence/hypothesis, not an explicit physical `SurfaceModel`.

## Legacy donor bridge findings

**PASS.**

V2L11.5 provides one intentionally narrow migration bridge from retained legacy `SparseReconstructionEstimate` values into the V2 contracts.

The bridge:

- accepts one already validated in-memory legacy estimate;
- emits one canonical `CameraSolution` per legacy camera pose;
- preserves exact observation identity, local frame, camera-from-local-frame rotation/translation, image dimensions and intrinsic parameters;
- lowercases legacy projection-model tokens only, relying on the canonical token validator to fail closed on unsupported syntax;
- emits one canonical `PointMap` from the legacy canonical point order and exact model observation membership, including an explicit empty point map for a zero-point estimate;
- maps legacy `UNRESOLVED` and `METRIC` scale status exactly to V2 `GeometryScaleStatus`;
- emits one `GeometrySolution` referencing the converted camera set and point map;
- performs no coordinate rescaling, scale inference, world/Earth placement, fragment acceptance, geometry optimization, depth synthesis, persistence, native solver read or external execution.

Legacy tracks, feature indices, reprojection errors, prior-focal flags and per-object provenance remain source-only evidence because no corresponding canonical target field owns those semantics. They are not silently repackaged into metrics or generic metadata.

The bridge is one-way by design. V2 does not acquire a permanent requirement to serialize back into legacy estimated-geometry types.

## Coordinate-convention fixture findings

**PASS.**

V2L11.6 uses a non-identity proper rotation, nonzero translation and two non-symmetric local points. The fixture is intentionally asymmetric so identity-like or accidental symmetric behavior cannot mask a convention error.

For the representative local point `(2.0, 7.0, -1.0)`, with the fixture rotation and translation, the canonical camera coordinate is verified by explicit scalar arithmetic as:

`(-4.0, 0.0, 4.0)`

The fixture proves that representative wrong interpretations do not equal the canonical result:

- transposed rotation;
- inverse-pose style interpretation;
- translation sign flip;
- identity-like translation without the declared rotation.

The same exact `LocalFrameId` is threaded through `CameraSolution`, `PointMap` and `GeometrySolution`. No contract renames that frame into world, Earth, geographic, fragment or another implicit coordinate domain.

The fixture also proves:

- `PointMap.positions_xyz` remains exactly the supplied local coordinate sequence in supplied order;
- switching only `GeometryScaleStatus.UNRESOLVED` to `METRIC` does not mutate camera extrinsics, points, frame identity or create a transform/anchor;
- an equivalent retained legacy estimate converts through V2L11.5 with the same rotation, translation, points, local frame and scale mapping;
- the converted camera obeys the same explicit `R * x + t` equation as a directly constructed canonical camera;
- no NumPy, Torch, Open3D, PyCOLMAP, filesystem, database, subprocess, network or coordinate-alignment dependency is introduced by the fixture.

No V2L11 production contract required correction after this composition fixture.

## Cross-cutting invariant audit

**PASS.**

- Raw observation identity remains distinct from physical camera identity.
- Local coordinate identity remains distinct from world/Earth/geographic identity.
- A local frame does not imply metric scale.
- Metric local scale does not imply absolute origin or orientation.
- Camera pose, depth, point maps, assembled geometry and physical surface remain separate products.
- Estimated geometry does not become accepted scene truth merely by construction.
- Coordinate-system conversions remain explicit responsibilities rather than hidden value-contract behavior.
- Absolute anchors do not enter V2L11.
- Unknown/unresolved scale remains explicit rather than guessed.
- Unsupported projection syntax fails closed rather than being coerced through a vendor alias table.
- No generic metadata bag is used to smuggle unsupported evidence into canonical geometry contracts.
- No V2L12 solver normalization or execution begins inside V2L11 closure.

## Explicit V2L11 non-capabilities

After V2L11, WRE still does **not** yet provide:

- a canonical production adapter descriptor for the current pinned COLMAP route;
- V2 artifact/cache normalization for classical feature, matching and verification stages;
- a V2 mapper that executes COLMAP and emits canonical camera/geometry products;
- automatic disconnected/zero-model handling for the new classical route;
- feed-forward camera/depth geometry routes;
- hybrid or consensus geometry;
- global scale solving;
- cross-fragment/local-frame alignment;
- Earth/ECEF/ENU/geographic/GPS/GCP anchoring;
- dense point fusion or dense physical geometry;
- `SurfaceModel`, mesh, SDF, collision, navigation or measurement surface;
- photorealistic appearance, materials or environment reconstruction;
- dynamic 4D or long-term historical reconstruction;
- `MasterScene` or target runtime compilation.

Those responsibilities remain later V2M2/V2M3/V2M4/V2M5 work.

## Review findings resolved during V2L11

**PASS after development corrections.**

- V2L11.1 through V2L11.4 closed with their exact canonical contract boundaries and no remaining semantic regression.
- V2L11.5 development required mechanical Ruff wrapping/format corrections and removal of accidentally literal newline escape sequences written during API editing. The final implementation and lifecycle heads were green; no final semantic contract changed as a result of those mechanical fixes.
- V2L11.6 fixture development required Ruff formatting only. No production geometry contract changed.
- The V2L11.6 coordinate fixture did not reveal a camera-pose inversion, transpose, translation-sign, local-frame or scale-semantics defect.
- Automated code-review availability is not treated as semantic evidence. Exact-head CI plus manual deny-by-default diff review govern closure.
- No unresolved review thread or known semantic regression remains at the V2L11 review boundary.

## Evidence

Accepted V2L11 implementation evidence:

- V2L11.1 / PR #101 / final head `03c2729f0df50ce00abf83ac3304a898dddba441`: fast-ci #736, CodeQL #675, native PyCOLMAP #383 — **PASS**.
- V2L11.2 / PR #102 / final head `63f85ea9c7f60ec87e081b73f851ea8d18a9530a`: fast-ci #743, CodeQL #682, native PyCOLMAP #387 — **PASS**.
- V2L11.3 / PR #103 / final head `a7aa7ae558cecf3325dd051836691407a13aa1ae`: fast-ci #749, CodeQL #688, native PyCOLMAP #391 — **PASS**.
- V2L11.4 / PR #104 / final head `6ce418121568102727ffefb7dff2e0d61226cdfa`: fast-ci #756, CodeQL #695, native PyCOLMAP #396 — **PASS**.
- V2L11.5 / PR #105 / final head `0fe663e3a54fea5116b703c69406e07ecd188f69`: fast-ci #765, CodeQL #704, native PyCOLMAP #405 — **PASS**.
- V2L11.6 / PR #106 / fixture-only head `a92b3fae13d35875fec61bc3033550bd5cd6d9a0`: fast-ci #768, CodeQL #707 — **PASS**.

The final PR #106 lifecycle/review head changes review/state/work-item metadata after this already-green fixture evidence. It must independently pass every triggered exact-head lane before merge.

## V2L12.1 handoff decision

After this PASS, `V2L12.1 — Current COLMAP adapter descriptor and normalization` may become the sole `ready` item.

V2L12.1 may describe and normalize the current pinned classical COLMAP capability behind V2 artifact contracts. It must preserve the V2L11 geometry semantics established here rather than reintroducing solver-private coordinate or geometry representations as core truth.

V2L12.1 must not:

- change the camera-from-local-frame convention;
- treat solver-local coordinates as world/Earth coordinates;
- infer physical `CameraId` from COLMAP camera/calibration identity;
- infer metric scale or absolute placement from an unanchored sparse model;
- merge or align unrelated local frames;
- introduce physical `SurfaceModel` or appearance responsibility;
- bypass canonical V2 artifact/provenance identities.

## Decision

**V2L11 lot review: PASS.**

WRE now has an audited solver-independent canonical camera/depth/point/geometry contract layer with explicit local-frame and local-scale semantics, a bounded one-way legacy donor bridge, and a deterministic regression that locks the camera-from-local-frame convention. V2M2 remains in progress and may continue with V2L12.1 only after PR #106 final exact-head gates pass.
