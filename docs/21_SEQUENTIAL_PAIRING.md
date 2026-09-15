# L4.1 — Sequential candidate pairing

L4.1 adds WRE's first cheap candidate-retrieval strategy. It proposes observation pairs from an **explicitly supplied sequence order**. It does not compute descriptors, match features, verify geometry, rank candidates, select a reconstruction route, create a fragment or accept any spatial claim.

## Epistemic boundary

A sequential candidate means only:

> these two observations are close enough in the supplied sequence to be worth considering together.

It is not evidence that the observations depict the same place, that their features match, that an essential/fundamental/homography model exists, or that either observation can be placed in a reconstruction. Those later claims remain owned by matching and geometric-evidence stages.

The output therefore records only:

- the exact source-observation membership through `DerivedArtifactProvenance`;
- the explicit ordered observation sequence used by this retrieval run;
- the canonical unordered observation pair;
- the two sequence indices and their positive distance;
- the canonical configuration hash;
- the reviewed COLMAP sequential-policy reference version.

## Reuse decision

WRE reuses the neighbor policy of COLMAP/PyCOLMAP **4.2.0** `SequentialPairGenerator` rather than inventing a different sequential-spacing scheme.

For ordinary non-rig sequence pairing, upstream COLMAP 4.2.0 uses:

- `overlap > 0`;
- linear mode: offsets `1, 2, ..., overlap`;
- quadratic mode: offsets `1, 2, 4, 8, ...` for `overlap` powers of two.

The upstream PyCOLMAP binding exposes `SequentialPairGenerator`, but constructing it requires a COLMAP database. WRE candidate retrieval occurs before feature extraction/matching in the target pipeline, so requiring a solver database merely to enumerate cheap sequence neighbors would invert that dependency and make the cheapest retrieval strategy depend on the native reconstruction environment.

L4.1 therefore implements only the trivial WRE orchestration needed to apply the reviewed upstream offsets to WRE observation identities. It does **not** reimplement feature matching, geometric verification or another foundational vision algorithm.

Reference reviewed at tag `4.2.0`:

- `src/colmap/controllers/pairing.h` — `SequentialPairingOptions` / `SequentialPairGenerator`;
- `src/colmap/controllers/pairing.cc` — ordered-image and neighbor-generation semantics;
- `src/pycolmap/pipeline/match_features.cc` — PyCOLMAP options/generator binding;
- `src/colmap/controllers/pairing_test.cc` — linear and quadratic reference cases.

## Explicit sequence order instead of filename order

COLMAP's standalone sequential generator orders database images lexicographically by image name. WRE deliberately does **not** treat filename ordering as temporal/sequence truth.

The caller must provide `ordered_observation_ids` explicitly. The `ReconstructionRun` still stores the same membership canonically for provenance, while the retrieval result retains the supplied order separately. L4.1 never sorts that sequence by observation id, asset URI, filename, reception time or capture time.

This is important for Internet/archive imagery and for metadata with uncertain time semantics. A filename can be an implementation label without being evidence of chronology.

For video-derived observations, callers may use already-established frame ordering evidence from ingestion. L4.1 itself does not infer a sequence from timestamps or frame names.

## Configuration

`SequentialPairingConfig` schema version 1 exposes:

- `overlap` — positive integer, default `10` as in COLMAP 4.2.0;
- `quadratic_overlap` — default `true` as in COLMAP 4.2.0;
- `expand_rig_images` — fixed to `false` in L4.1;
- `loop_detection` — fixed to `false` in L4.1.

Rig expansion needs explicit rig/frame semantics and is not smuggled into ordinary sequence proximity. Vocabulary-tree loop detection belongs to later L4 retrieval work and must not be pulled into L4.1.

## Determinism and canonical pair identity

For identical explicit sequence membership/order and configuration, candidate generation is deterministic.

Pairs use canonical ascending observation IDs so later retrieval strategies can deduplicate the same unordered pair. Because canonical ID order can differ from sequence order, each candidate also stores the sequence index associated with each canonical ID plus `sequence_distance`.

A single-observation sequence is valid and produces zero pairs. WRE does not force a candidate merely to obtain downstream work.

## Out of scope

L4.1 does not implement:

- GPS/spatial pairing (`L4.2`);
- vocabulary-tree retrieval (`L4.3`);
- graph-neighbour candidate generation (`L4.4`);
- multi-signal candidate ranking (`L4.5`);
- FAST/STANDARD/ESCALATED route selection (`L4.6`);
- SIFT matching or geometric verification;
- fragment creation/placement/merge;
- any acceptance threshold for spatial truth.

Candidate generation changes which pairs may be attempted later. It does not change what evidence is required to accept a geometric result.
