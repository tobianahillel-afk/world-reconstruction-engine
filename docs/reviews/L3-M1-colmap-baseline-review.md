# L3 COLMAP baseline and M1 milestone review

Date: 2026-09-15
Scope: lot `L3` — COLMAP baseline; milestone `M1` — Reconstruct a place
Implementation head reviewed: `a844f4869b617f66e69e591e6596262ce54de4b3`
Verdict: **PASS**

## Review objective

Verify that L3 composes deterministic media observations from L1/L2 into solver-derived local sparse geometry through a pinned COLMAP/PyCOLMAP baseline, while preserving WRE's observation/evidence/estimated-geometry separation, provenance, parent immutability and unresolved local scale. For M1, verify that the implemented L1-L3 contracts now form one coherent, tested path from media observations to an imported local reconstruction before candidate retrieval, fragments or world placement begin.

## Preconditions from L1 and L2

**PASS.** The L1 core-domain review and L2 media-ingestion review are already passed and remain authoritative for their scopes. L3 consumes typed observations, reconstruction runs/provenance and deterministic image-like inputs without redefining raw observation semantics. L2 continues to own media ingestion and raw/semantic metadata interpretation; L3 does not reinterpret publication time, GPS or source identity.

## L3.1 — COLMAP environment adapter

**PASS.** PR #19 establishes the approved external solver boundary around exact `pycolmap==4.2.0`. The adapter records PyCOLMAP/COLMAP/Ceres/build/CUDA identity and rejects incompatible environments. PyCOLMAP remains an external integration dependency rather than an ordinary WRE runtime dependency, with a dedicated native CI lane.

## L3.2 — Feature extraction

**PASS.** PR #20 reuses native PyCOLMAP CPU SIFT through a deterministic WRE adapter. Inputs are content-addressed observations, image naming is deterministic, the configuration is digestible/auditable, and the produced COLMAP database remains derived evidence rather than accepted geometry.

## L3.3 — Raw pair matching

**PASS.** PR #21 reuses exhaustive PyCOLMAP matching with one thread/fixed seed and explicitly skips geometric verification. The implementation discovered that COLMAP may still materialize default `TwoViewGeometry` rows while verification is skipped; WRE therefore accepts only semantically empty `UNDEFINED` placeholders at this boundary and rejects non-empty verified geometry. Raw correspondences remain distinct derived evidence.

## L3.4 — Two-view geometric verification

**PASS.** PR #22 reuses PyCOLMAP's existing-match geometric verifier and persists auditable two-view evidence including inliers, model matrices/classification and relative-pose outputs when available. Orientation is canonicalized with COLMAP's own inversion semantics. The output remains solver-derived evidence: it is not promoted directly to a WRE world pose, fragment or accepted state.

## L3.5 — Incremental sparse reconstruction

**PASS.** PR #23 reuses native PyCOLMAP incremental mapping from the verified database with deterministic CPU/one-thread/fixed-seed settings. Disconnected solver sub-models are allowed rather than forced together. Native sparse-model files are audited/content-addressed as solver output, but no WRE `SpatialFragment` acceptance occurs here.

## L3.6 — Reconstruction importer

**PASS.** PR #24 reads audited native sparse models through official `pycolmap.Reconstruction(path)` instead of reimplementing COLMAP binary parsing. Before read, L3.5 file membership, SHA-256 and byte length are rechecked. Registered solver image names are resolved through the persisted L3.2 mapping back to WRE `ObservationId` values.

The importer creates explicit solver-independent estimated geometry for camera calibration, camera-from-local-frame pose, sparse points and tracks. COLMAP's solver-local frame is not renamed into an absolute world claim: the imported model receives a WRE `LocalFrameId` and `LocalScaleStatus.UNRESOLVED`. Multiple native models remain separate and zero-model output remains explicit. No fragment, merge, georeference or metric-scale claim is made.

## L3.7 — Native end-to-end fixture

**PASS.** PR #25 adds the first complete image-based composition fixture for L3. A versioned `scene.json` deterministically renders 12 PGM observations with unique textured patches at multiple depths. The test then invokes the public WRE L3.2-L3.6 APIs in sequence:

```text
rendered observations
  -> feature extraction
  -> exhaustive raw matching
  -> geometric verification
  -> incremental sparse reconstruction
  -> native-model import
  -> WRE regression expectations
```

The fixture does **not** start from a pre-populated synthetic COLMAP reconstruction/database, so it covers the actual feature and matching boundaries. It additionally hashes parent artefacts to verify that matching does not mutate the feature database, verification does not mutate the raw-match database, mapping does not mutate the verified database, and import does not mutate the native sparse-model files.

The first native run was intentionally treated as evidence rather than worked around. With a total synthetic camera baseline of only `0.36` at scene depths `4.5`-`7.5`, COLMAP built a connected 12-image/66-pair correspondence graph but reported `No good initial image pair found` and produced no sparse model. The fixture scene was corrected by increasing the camera baseline to `1.5`, creating sufficient parallax. No L3 verifier/mapper threshold or acceptance rule was weakened. The corrected scene passed the exact same WRE configurations.

The accepted fixture expectations require 12 observations, all 66 exhaustive attempted pairs, at least 10 raw matched pairs, at least 10 geometrically verified pairs, exactly one reconstruction, at least 10 registered images, exactly one imported model, at least 10 imported observations, at least 20 imported 3D points and exactly one unresolved-scale imported model.

## Native integration and CI evidence

Implementation head `a844f4869b617f66e69e591e6596262ce54de4b3`:

- fast-ci run `34978343545` / run #236: **PASS**;
- repository metadata validator: **PASS**;
- Ruff lint and format: **PASS**;
- Pyright: **PASS**;
- pytest fast suite: **PASS**;
- actionlint: **PASS**;
- exact native COLMAP integration run `34978343630` / run #86: **PASS**, `60 passed` under `pycolmap==4.2.0` and `numpy==2.5.3`;
- CodeQL run `34978343694` / run #201: **PASS**;
- dependency-review run `34978343746` / run #85: **PASS** under the repository's existing dependency-graph capability policy.

The previous L3.7 native run `34978155936` / run #85 is retained as useful negative evidence: all earlier L3 integration tests passed, but the first end-to-end scene failed at mapper initialization because the synthetic geometry did not provide a good initial pair. The subsequent scene-only parallax fix produced the passing run above.

## Reuse and supply-chain audit

**PASS.** L3 reuses COLMAP/PyCOLMAP for foundational photogrammetric operations rather than reimplementing SIFT, matching, two-view robust estimation, incremental SfM or native sparse-model parsing. The approved integration remains pinned to PyCOLMAP 4.2.0, and the dedicated lane verifies the exact external environment. WRE owns orchestration, provenance, immutable evidence boundaries, solver-independent estimated-geometry contracts and acceptance semantics.

No new native dependency or model-weight dependency is introduced by L3.7.

## Epistemic and scope audit

**PASS.** The L3 implementation preserves the project rules:

- raw observations remain distinct from derived features/matches/two-view evidence;
- solver reconstruction and imported camera/point geometry remain estimates, not world truth;
- observation identity is not inferred from camera/image similarity;
- parent solver evidence is not silently mutated between pipeline stages;
- disconnected solver models are not automatically merged;
- local scale remains explicitly unresolved;
- no GPS/GCP/landmark anchor is invented or consumed as an absolute constraint in L3;
- no `SpatialFragment` is created or accepted by the COLMAP baseline;
- no L4 retrieval/routing, L5 competing-geometry policy, L6 fragment lifecycle or later-world machinery is pulled forward.

## Residual limitations

These are explicit later-lot scope, not blockers for L3/M1:

- the L3.7 fixture is synthetic and validates integration/composition, not natural-image reconstruction accuracy;
- L3 uses exhaustive pairing as a baseline and does not yet implement scalable sequential/GPS/visual/graph candidate retrieval;
- robust competing F/E/H evidence policy, DEGENSAC/AC-RANSAC and triplet/cycle consistency remain owned by L5;
- solver sub-models are estimated geometry and are not yet accepted into the fragment lifecycle owned by L6;
- metric/world scale, GPS/GCP/reference-fragment anchoring and global factor-graph optimization remain later work;
- uncertainty/covariance and competing placement hypotheses remain later work;
- dense, temporal/4D and production-scale validation remain later milestones.

## M1 milestone decision

**M1 milestone review: PASS.** L1, L2 and L3 now form a coherent, deterministic and provenance-preserving path from typed media observations to a native COLMAP sparse reconstruction imported as explicit local WRE estimated geometry. The milestone demonstrates the repository's first complete reconstruction path while deliberately retaining unresolved scale and refusing premature fragment/world acceptance.

This is a controlled baseline milestone, not a claim that arbitrary images can already be placed into a global world model. The next milestone exists precisely to add candidate retrieval, stronger evidence geometry and the fragment engine.

## Decision

**L3 lot review: PASS. M1 milestone review: PASS.** L3.1 through L3.7 meet their intended baseline boundary and the full image-based fixture passes without weakening solver/evidence thresholds. After the canonical state handoff and final CI on that exact handoff head, the next permitted implementation item is `L4.1` — sequential pairing, under milestone `M2` / lot `L4`. No L4 implementation belongs in PR #25.
