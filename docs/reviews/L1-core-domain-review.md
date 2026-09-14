# L1 core-domain review

Date: 2026-09-14
Reviewed implementation head: `796c65659d6349cde98cfc4ea7049dc0760f6df6`
Scope: lot `L1` — Core domain
Verdict: **PASS**

## Review objective

Verify that L1 supplies the complete solver-independent core promised by the roadmap — observations, cameras/metadata, local fragments, run provenance and local persistence — with deterministic invariants, appropriate tests and no accidental implementation of later ingestion/reconstruction lots.

This is the required L1 lot review performed by the final L1 work item PR, as required by `AGENTS.md`. It is **not** an M1 milestone review: M1 also contains L2 Media ingestion and L3 COLMAP baseline, which remain incomplete.

## L1.1 — Observation models

**PASS.** Raw source media has an explicit domain representation independent of any reconstruction solver.

Evidence:

- PR #7 — `[L1.1] Add core observation models`
- `src/wre/domain/observations.py`
- `tests/test_observations.py`
- `docs/04_DATA_MODEL.md`

The model provides typed observation/source identities, content-addressed media references, source references, timezone-aware receipt/capture instants, still-image observations and video-frame observations with deterministic frame coordinates. It does not silently classify features, matches or poses as raw observations.

## L1.2 — Camera and metadata models

**PASS.** Camera identity and source metadata are representable without conflating descriptive metadata with evidence that two observations came from the same physical device.

Evidence:

- PR #8 — `[L1.2] Add camera and metadata models`
- `src/wre/domain/cameras.py`
- `tests/test_cameras.py`
- `docs/04_DATA_MODEL.md`

`CameraId` is explicit rather than derived automatically from manufacturer/model/serial fields. Raw metadata remains namespaced and uninterpreted. EXIF/container parsing, geographic interpretation and calibration remain outside L1.

## L1.3 — SpatialFragment model

**PASS.** L1 can name one local spatial component and its member observations without pretending that the component is already placed in the world.

Evidence:

- PR #9 — `[L1.3] Add SpatialFragment domain model`
- `src/wre/domain/fragments.py`
- `tests/test_fragments.py`
- `docs/04_DATA_MODEL.md`

Fragment membership is non-empty, duplicate-free, immutable and canonically ordered. `LocalFrameId` identifies a local frame but makes no claim about metric scale, Earth alignment, georeferencing or compatibility with another fragment frame. No lifecycle or merge behavior has been pulled forward from later lots.

## L1.4 — ReconstructionRun and provenance

**PASS.** Derived computations can identify their raw inputs and producing implementation/version without making execution provenance equivalent to truth.

Evidence:

- PR #10 — `[L1.4] Add ReconstructionRun and provenance models`
- `src/wre/domain/runs.py`
- `tests/test_runs.py`
- `docs/04_DATA_MODEL.md`

Runs have typed identity, explicit producer/version/revision, canonical raw inputs, timezone-aware timing and an optional configuration digest. `DerivedArtifactProvenance` links an artifact to its producing run and raw source observations. It does not assert correctness, acceptance or promotion into another epistemic layer.

## L1.5 — Local persistence layer

**PASS.** The full L1 domain can be persisted locally using an explicit, deterministic and versioned boundary without introducing an ORM or runtime dependency.

Evidence:

- PR #11 — `[L1.5] Add local persistence layer`
- `src/wre/persistence/codec.py`
- `src/wre/persistence/sqlite_store.py`
- `tests/test_persistence.py`
- `docs/10_PERSISTENCE.md`
- implementation fast-ci run `34889364989`: passed on `796c65659d6349cde98cfc4ea7049dc0760f6df6`
- implementation CodeQL run `34889364874`: passed on the same head

SQLite provides local transactions and durability; canonical JSON provides explicit record serialization. Database and record schema versions are checked. Identical writes are idempotent, while reuse of the same stable identity with different content fails rather than silently overwriting the earlier record. Tests cover image/video, camera/metadata, fragments, runs, reopening, canonical payloads, conflicts, incompatible schemas and corrupted payloads.

## Epistemic-boundary audit

**PASS.** L1 preserves the intended separation between raw/source-facing records and later derived/estimated geometry:

- observations contain source media identity and source/time context, not feature or pose results;
- metadata values can remain raw/uninterpreted when semantics are unresolved;
- fragments are local components, not accepted global placements;
- provenance records how a result was produced, not whether it is true;
- persistence stores those objects without changing their epistemic class.

No L1 type forces a geographic placement or merge when evidence is absent.

## Determinism and identity audit

**PASS.** Typed IDs prevent casual cross-domain interchange. Set-like memberships used by fragments, runs and derived provenance are duplicate-free and canonically ordered. Known instants are timezone-aware. Persistence normalizes serialized instants to UTC and uses canonical JSON. Same-ID conflicting persisted content is rejected explicitly.

## Dependency and reuse audit

**PASS.** L1 introduces no runtime dependency. Domain modeling uses Python dataclasses and standard-library types; local persistence reuses Python's maintained SQLite binding rather than adding an ORM or custom storage engine. `pyproject.toml` still declares `dependencies = []`.

No foundational geometry algorithm has been reimplemented in L1.

## Test and CI audit

**PASS.** Each L1 work item has focused unit tests:

- `tests/test_observations.py`
- `tests/test_cameras.py`
- `tests/test_fragments.py`
- `tests/test_runs.py`
- `tests/test_persistence.py`

The L1.5 implementation head passes repository validation, Ruff lint/format, Pyright, pytest, actionlint and CodeQL. The final review/state handoff head must pass the same gates before merge; exact final-head runs are recorded in PR #11 acceptance evidence.

## Scope audit

**PASS.** L1 does not implement work owned by later lots. In particular, it does not contain:

- media discovery, copying or decoding;
- duplicate-detection workflow;
- EXIF/XMP parsing or GPS/time normalization;
- camera calibration/intrinsics inference;
- SIFT or other feature extraction;
- image-pair matching or geometric verification;
- COLMAP/LIMAP adapters or solver-private models;
- camera poses, point clouds or dense reconstruction;
- fragment lifecycle, global placement or merge transforms;
- world factor-graph optimization.

Those remain available to their owning work items beginning with L2.1.

## Residual limitations carried into L2

These are intentional boundaries rather than hidden failures:

- persistence schema migrations are not implemented; unsupported versions fail explicitly;
- there is no mutable/update policy for stable record identities; conflicting content requires an explicit future policy rather than last-write-wins;
- raw metadata has representation but no extraction/normalization logic yet;
- no reconstruction output exists yet, so provenance and fragment models are exercised with synthetic domain objects only;
- M1 remains pending until L2 and L3 are completed and reviewed.

## Decision

The L1 lot review **passes**. The core domain is complete enough, tested enough and sufficiently isolated from solver/ingestion concerns to begin L2.

This decision does **not** complete milestone M1. The next permitted work item after the final PR #11 handoff is `L2.1 — Image ingestion`, and only that item becomes active. Normal one-work-item-per-PR discipline remains in force.
