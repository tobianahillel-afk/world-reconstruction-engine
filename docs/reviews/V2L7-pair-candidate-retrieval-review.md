# V2L7 — Pair candidate retrieval review

**Review status:** PASS  
**Reviewed:** 2026-09-20  
**Lot:** `V2L7 — Pair candidate retrieval`

## Review objective

Verify that V2L7.1 through V2L7.6 compose into one auditable proposal-only relationship discovery boundary for arbitrary media without turning sequence proximity, GPS proximity, visual retrieval similarity or benchmark fixture labels into physical scene identity.

The lot must establish one canonical `PairCandidate` contract, adapt the retained sequential and GPS donors without changing their source semantics, add one deterministic classical visual retrieval baseline and one bounded modern learned visual-place candidate, then benchmark source-specific and union candidate recall while preserving every source evidence reference. It must stop before local matching, geometric verification, `SceneCluster`, scene truth, routing, geometry and scalable ANN infrastructure.

## V2L7 implementation sequence

V2L7 was intentionally split into six bounded work items:

1. PR #77 — `V2L7.1 Canonical PairCandidate contract`: canonical unordered observation endpoints, explicit source-family evidence and deterministic source/evidence union.
2. PR #78 — `V2L7.2 Sequential pairing donor adapter`: pure adaptation of already-produced retained sequential pairs into `PairCandidate`.
3. PR #79 — `V2L7.3 GPS pairing donor adapter`: pure adaptation of already-produced retained GPS pairs into `PairCandidate`.
4. PR #80 — `V2L7.4 Classical vocabulary retrieval baseline`: exact PyCOLMAP 4.2 vocabulary-tree proposal path over a verified feature database and explicit local vocabulary artifact.
5. PR #81 — `V2L7.5 Modern visual retrieval first candidate`: bounded experimental SelaVPR++ Base + GeM exact-search candidate with pinned source/checkpoint/runtime identity; PR #83 subsequently remediated three valid late post-merge review findings without adding capability.
6. PR #82 — `V2L7.6 Retrieval union/dedup recall benchmark`: benchmark-only relevant-pair fixture truth, source/union recall/count metrics, canonical union exclusively through `merge_pair_candidates` and existing `BenchmarkRecord` reuse.

No V2L7 item establishes same-scene truth, local correspondence truth, cluster membership, route choice or geometry.

## Composition findings

**PASS.**

- `PairCandidate` means only “worth considering together”. It contains two canonical distinct `ObservationId` endpoints and one or more typed source contributions with exact `ArtifactRef` evidence.
- Source-specific scores, distances, ranks, model/checkpoint details, GPS values and sequence indices do not leak into the canonical pair contract.
- `merge_pair_candidates` is the only candidate-union semantic. Duplicate endpoints combine source families and exact evidence refs deterministically without priority, weighting or winner selection.
- Sequential adaptation never executes the donor generator and preserves donor-specific sequence semantics only through referenced evidence.
- GPS adaptation never executes the donor generator and preserves coordinates, eligibility, datum, radius and distance semantics only through referenced evidence.
- The classical vocabulary baseline reuses the verified COLMAP feature database and PyCOLMAP 4.2 `VocabTreePairGenerator`, verifies exact source/vocabulary bytes before execution and operates on a temporary database copy so the retained source artifact is not mutated.
- Classical retrieval produces candidate relationships only. Raw local matching, two-view geometry and geometric verification are intentionally absent.
- SelaVPR++ remains an experimental benchmark candidate. Its exact upstream revision, official checkpoint digest, CPU reference runtime versions, deterministic preprocessing/search configuration and source-tree integrity checks remain explicit.
- The learned candidate uses only verified canonical decoded-image pyramid level-zero RGB8/source-pixel materializations and retains source-specific cosine evidence outside `PairCandidate`.
- The post-merge PR #83 remediation now rejects unsupported Python before Torch/model execution, rejects a non-`media.decoded_image_pyramid` artifact identity before byte consumption and exactly renormalizes descriptors already accepted within the documented norm tolerance.
- No learned candidate is approved or promoted merely because it is modern or may achieve higher recall. Direct SelaVPR++ code/model licensing is recorded as MIT while transitive environment/redistribution review remains pending.
- V2L7.6 defines benchmark-only `RetrievalRelevantPair` labels. Those labels are fixture truth for evaluation only and are never copied into production scene identity.
- Source-specific recall and union recall compare canonical endpoint identity only. False-positive proposals increase candidate count without becoming negative or positive scene truth.
- The union benchmark concatenates only supplied source candidate sets and delegates evidence-preserving deduplication exclusively to `merge_pair_candidates`.
- Optional incremental recall is relative to an explicitly named comparison source and cannot declare a winner or default.
- Existing `MetricVector`, `MetricProvenance`, `BenchmarkFixtureIdentity`, `BenchmarkRecord`, `QualityMode` and `HardwareRuntimeIdentity` contracts are reused rather than duplicated.
- Benchmark execution is bounded and in-memory. It does not execute PyCOLMAP, Torch, FFmpeg, model assets, filesystem discovery, network access, ANN, matching, verification, clustering, routing or geometry.
- Repeated façades, symmetric structures and visually similar but distinct places remain unresolved by retrieval alone. V2L8 must require verification/disambiguation evidence before cluster fusion.
- No ANN/HNSW/GPU/distributed index is introduced. Large-collection retrieval/index execution remains owned by V2L37.
- No V2L8 implementation is started by the V2L7 lifecycle closure.

## Source-specific limitations retained explicitly

- **Sequential:** assumes useful locality in source/frame order; proximity is not scene identity.
- **GPS:** depends on available and resolved coordinate evidence; geographic proximity is not sufficient scene identity.
- **Classical vocabulary:** depends on the retained feature database and explicit vocabulary quality/coverage; retrieval similarity still requires later verification.
- **SelaVPR++:** experimental, CPU exact-search reference only, with full transitive environment/redistribution review still pending; learned place similarity is not a truth source.
- **Union benchmark:** measures proposal coverage/recall on explicit fixture labels, not precision of scene identity and not production route quality.
- **Repeated/symmetric look-alikes:** deliberately remain a V2L8 verification/disambiguation responsibility.

## Cross-cutting invariant audit

**PASS.**

- Original media/observation truth remains immutable and separate from derived retrieval evidence.
- Missing scene identity is never fabricated from sequence, GPS or visual similarity.
- Routing is not involved and retrieval never becomes a hidden quality decision.
- No source priority or automatic learned-model promotion exists.
- Exact source/model/checkpoint/configuration/runtime identity remains outside stable solver-independent pair semantics.
- Artifact evidence remains exact and typed; filenames, timestamps and mutable paths are not used as pair identity.
- Benchmark labels stay benchmark-local and are not promoted to observed/reconstructed provenance.
- Invalid external source/checkpoint/runtime/materialization states fail closed.
- V2L7 does not perform matching, geometric verification, clustering, temporal grouping, geometry reconstruction or runtime compilation.
- The next lot begins with representation contracts only; it does not inherit retrieval similarity as accepted scene truth.

## Review findings resolved during the lot

**PASS after correction.**

- PR #80 resolved three review findings around native PyCOLMAP vocabulary retrieval normalization/hardening before lifecycle handoff.
- PR #81 passed its original exact-head gates, but three valid P2 review findings were posted approximately four minutes after merge. PR #83 reopened the V2L7.5 lifecycle explicitly and corrected all three:
  1. unsupported Python is now rejected before Torch/model/checkpoint inference execution;
  2. decoded-image artifact kind is verified as `media.decoded_image_pyramid` before generic materialization verification/byte consumption;
  3. descriptors accepted within the documented L2 norm tolerance are renormalized exactly before cosine search.
- PR #83 final head `2ab9346bf4c95aa3cc3c570683a40b731226eede` passed fast-ci #643, native PyCOLMAP #338 and CodeQL #581; the three late PR #81 threads were linked to the remediation and resolved after merge.
- PR #82 initial head was blocked by five mechanical Ruff lint findings. After those were fixed, `ruff format --check` requested formatting-only changes in the two new benchmark files. No benchmark semantic change was required by those mechanical findings.
- PR #82 reviewed implementation head `397398380bac017b95975271b83f73914a2f82a2` passes repository validation, Ruff lint/format, Pyright, full pytest, actionlint, native PyCOLMAP and CodeQL.
- Automated Codex review quota was exhausted during PR #83. The remediated diff and V2L7.6 implementation were therefore also reviewed manually against the executable work-item acceptance criteria and deny-by-default scope.
- No unresolved semantic defect is known at the V2L7 lot-review boundary.

## Evidence

Exact retained work-item evidence:

- V2L7.1 / PR #77 / final head `7efa2ebf2382876a836ada8bbf5ee4e09242ab2e`: fast-ci #584, native COLMAP #279, CodeQL #522 — **PASS**.
- V2L7.2 / PR #78 / final head `25fce1396886ccdb1d35b831a9d064d64e8bf7bd`: fast-ci #590, native COLMAP #285, CodeQL #528 — **PASS**.
- V2L7.3 / PR #79 / final head `1769a8d12c835d9cf2860c843231813df422f7a0`: fast-ci #597, native COLMAP #292, CodeQL #535 — **PASS**.
- V2L7.4 / PR #80 / final head `092b8a3dca449c113a7e3ef2ac3674d0efc77eec`: fast-ci #612, native COLMAP #307, CodeQL #550, dependency-review #132 — **PASS**.
- V2L7.5 / PR #81 / original final head `050b027b7ffddf3b9f1c20f75167b1ac6ff2eaea`: fast-ci #639, native COLMAP #334, FFmpeg 8 #33, CodeQL #577 — **PASS**, followed by the separately audited PR #83 remediation described above.
- V2L7.5 remediation / PR #83 / final head `2ab9346bf4c95aa3cc3c570683a40b731226eede`: fast-ci #643, native PyCOLMAP #338, CodeQL #581 — **PASS**.
- V2L7.6 / PR #82 / reviewed implementation head `397398380bac017b95975271b83f73914a2f82a2`: fast-ci #646, native PyCOLMAP #341, CodeQL #584 — **PASS**.

The final PR #82 lifecycle-handoff head changes review/state/registry metadata and activates only V2L8.1 after the already-green reviewed implementation. It must independently pass every triggered exact-head lane before merge.

## V2L8.1 handoff decision

After this PASS, `V2L8.1 — SceneCluster and scene relation hypothesis contract` may become the sole `ready` item.

V2L8.1 is intentionally representation-only. It defines typed `SUPPORTED`, `CONTRADICTED` and `UNRESOLVED` pairwise relationship state with exact evidence references plus minimal `SceneClusterId` / `SceneCluster` membership contracts. It does not execute retrieval, matching, geometric verification, cluster construction, transitive closure, repeated-structure disambiguation or scalable graph processing.

A `PairCandidate` does not automatically become a supported scene relationship. V2L8.2 must provide explicit local/geometric verification evidence; V2L8.3 owns deterministic cluster construction from verified relationships.

## Decision

**V2L7 lot review: PASS.**

Pair candidate retrieval now composes deterministic donor adaptation, classical and learned proposal sources, evidence-preserving canonical union and benchmark-only recall measurement while preserving the central invariant that retrieval proposes work but does not establish physical scene identity. V2L8 may begin with scene-relationship and cluster representation contracts only after PR #82 final exact-head gates pass.
