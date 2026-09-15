# L4.2 — GPS candidate pairing

L4.2 adds a second cheap candidate-retrieval strategy. It proposes observation pairs from **explicit, resolved WGS84 GPS evidence**. It does not match features, verify geometry, place a camera, create/anchor a fragment, or accept that two observations depict the same place.

## Epistemic boundary

A GPS candidate means only:

> these two observations have compatible explicit WGS84 position metadata and are close enough under the configured retrieval policy to be worth attempting together later.

It is not proof that the images overlap visually, that either GPS reading is accurate, or that either observation belongs at the metadata coordinate in accepted world geometry. Matching and geometric verification remain downstream. Absolute GPS anchoring remains owned by L9, where uncertainty, residuals and competing constraints are modeled explicitly.

The L4.2 result records:

- exact source-observation membership through `DerivedArtifactProvenance`;
- an eligibility decision for every supplied metadata interpretation;
- canonical unordered candidate-pair identities;
- the WGS84-ECEF chord distance used by retrieval;
- canonical configuration identity;
- the reviewed COLMAP spatial-policy reference version.

## Input eligibility is deliberately strict

L4.2 consumes the immutable `ObservationMetadataInterpretation` produced by L2.4. An observation is spatially eligible only when:

1. `GpsMetadataInterpretation.status == resolved`;
2. latitude and longitude therefore satisfy the L2.4 finite/range invariants;
3. `GPSMapDatum` is explicitly present; and
4. the datum normalizes lexically to `WGS84` (for example `WGS-84` or `WGS 84`).

A resolved coordinate with no datum is **not** silently assumed to be WGS84. A coordinate naming another datum is also not transformed or treated as WGS84. Those observations remain in provenance and receive explicit `map_datum_missing` or `map_datum_unsupported` eligibility states, but they cannot generate L4.2 candidates.

Likewise, L2.4 `absent`, `incomplete` and `invalid` GPS interpretations remain explicit non-eligible states. L4.2 never reconstructs partial coordinates or repairs invalid metadata.

## Reuse decision

The candidate policy reuses the reviewed COLMAP/PyCOLMAP **4.2.0** `SpatialPairGenerator` design:

- a maximum spatial radius;
- a maximum number of nearest neighbors considered per query observation;
- no minimum-neighbor forcing in WRE (`min_num_neighbors = 0`).

The defaults match the useful safe subset of COLMAP 4.2.0: `max_distance = 100 m` and `max_num_neighbors = 50`. WRE fixes the minimum-neighbor count to zero because a nonzero minimum can deliberately produce neighbors beyond the radius; correctness-first retrieval must not manufacture a geographically distant candidate merely to fill a quota.

COLMAP's native spatial generator requires a COLMAP database/pose-prior cache. As with L4.1, making cheap pre-feature retrieval depend on a solver database would invert WRE's target pipeline. Production L4.2 therefore remains solver-independent and applies the reviewed radius/KNN policy directly to WRE metadata interpretations.

Reference reviewed at tag `4.2.0`:

- `src/colmap/controllers/pairing.h` — `SpatialPairingOptions` and `SpatialPairGenerator`;
- `src/colmap/controllers/pairing.cc` — radius/KNN behavior and position-prior conversion;
- `src/colmap/geometry/gps.cc` — ellipsoid-to-ECEF conversion;
- `src/pycolmap/geometry/gps.cc` — exact `GPSTransform` / ellipsoid Python binding.

## Explicit WGS84 conversion

WRE converts eligible latitude/longitude to Earth-Centered Earth-Fixed coordinates at altitude zero using the WGS84 constants and the same closed-form conversion used by COLMAP's `GPSTransform`. Candidate distance is ordinary Euclidean chord distance between those ECEF coordinates.

Altitude is intentionally ignored because the L2.4 contract currently exposes latitude/longitude and datum, not a validated altitude. This corresponds to the safe `ignore_z` spatial-retrieval use case and avoids inventing height evidence.

A subtle upstream detail is handled explicitly: COLMAP 4.2.0's `SpatialPairGenerator` constructs `GPSTransform` with its default GRS80 ellipsoid even when a pose prior is marked WGS84. WRE does **not** inherit that implicit default. Because L4.2 admits only metadata explicitly identified as WGS84, it selects the WGS84 ellipsoid explicitly. The dedicated native CI test compares WRE candidate distance against `pycolmap.GPSTransform(GPSTransformEllipsoid.WGS84)` from exact PyCOLMAP 4.2.0.

The ECEF chord metric is a retrieval heuristic, not a geodesic-survey assertion and not an L9 world constraint. It naturally handles the antimeridian and polar longitude behavior without naive longitude subtraction.

## Deterministic neighbor selection

Eligible observations are processed in canonical `ObservationId` order. For each query observation, WRE:

1. computes distance to every other eligible observation;
2. orders neighbors by `(distance, ObservationId)` for deterministic tie-breaking;
3. considers at most `max_num_neighbors` neighbors;
4. stops once distance exceeds `max_distance_m`;
5. canonicalizes and deduplicates unordered observation pairs.

Output pair ordering is canonical by observation IDs, not by distance. L4.2 therefore does not pretend to perform the multi-signal candidate ranking owned by L4.5.

The current correctness-first implementation is pairwise in the number of eligible GPS observations. L4.2 does not introduce a separate spatial-index dependency solely for early optimization. L4.7 benchmarks retrieval; if scale requires an index, a mature implementation can replace the search backend while preserving these observable semantics.

## Configuration

`GpsPairingConfig` schema version 1 exposes:

- `max_num_neighbors` — positive integer, default `50`;
- `max_distance_m` — finite positive distance in metres, default `100.0`.

Its canonical configuration document also records:

- fixed `min_num_neighbors = 0`;
- distance model `wgs84_ecef_chord_altitude_zero`;
- datum policy `explicit_wgs84_only`.

That makes the safety policy part of configuration identity instead of an undocumented implementation assumption.

## Out of scope

L4.2 does not implement:

- CRS transformation for non-WGS84 metadata;
- assuming WGS84 when `GPSMapDatum` is absent;
- altitude/vertical-datum interpretation;
- GPS accuracy/error-radius modeling or authoritative anchoring;
- visual vocabulary retrieval (`L4.3`);
- graph-neighbour candidates (`L4.4`);
- multi-signal ranking (`L4.5`);
- FAST/STANDARD/ESCALATED routing (`L4.6`);
- feature matching or geometric verification;
- fragment placement, merge or world-graph factors.

Those capabilities remain in their owning work items. GPS metadata can cheaply propose work here; it cannot bypass the later evidence standard.
