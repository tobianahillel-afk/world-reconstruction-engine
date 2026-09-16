# V2L3 — Adapter, model and shipping registries review

**Review status:** PASS  
**Reviewed:** 2026-09-16

## Scope

V2L3.1 through V2L3.6 are reviewed together. The lot establishes typed adapter capabilities, exact hardware/runtime identity participation in artifact keys, the closed adapter/model registry schema, adapter-local failure/metric/resume declarations, fail-closed registry validation, and registration of the retained COLMAP/PyCOLMAP, ExifRead and FFmpeg baselines.

## Findings

- `AdapterCapabilityDescriptor` remains solver-independent and contains only explicit typed input/output artifact kinds.
- Existing model/checkpoint identity is reused; `HardwareRuntimeIdentity` is explicit and optional, and absent hardware identity preserves prior artifact-key bytes.
- The registry schema remains closed and versioned with no hidden defaults.
- Adapter-local `failure_signals` and `metric_names` do not define the stable WRE failure taxonomy or metric semantics owned by V2L4.
- Registry validation is fail-closed, canonical and non-mutating; floating identities, unknown dependencies and invalid license/shipping combinations are rejected.
- The populated registry contains exactly `colmap.sparse_sfm`, `exifread.raw_exif` and `ffmpeg.keyframe_extraction` in canonical order.
- Each production entry references exactly its retained approved dependency record and adds no model/checkpoint.
- No pending research candidate is promoted and no retained runtime integration is rewritten.
- COLMAP remains the exact retained PyCOLMAP/COLMAP 4.2.0 environment behind the existing WRE sparse-import boundary; the registry does not claim future V2L12 canonical geometry outputs.
- The two V2L3.3 tests that required the initial registry to stay empty were updated at V2L3.6 to test enduring envelope and dependency-separation invariants; the schema and validator were not relaxed.

## Evidence

- V2L3.1 / PR #45 / `00c7748d794126a3981ba51f79891b0791d3af8e`: fast-ci #380, PyCOLMAP #187, CodeQL #325 — PASS.
- V2L3.2 / PR #46 / `1b31a667bc33f1d278e0669af75c62db0868dc5d`: fast-ci #386, PyCOLMAP #191, CodeQL #331 — PASS.
- V2L3.3 / PR #47 / `54bf9a4f6b669ffe8cb4ddcf866f799107fca8d1`: fast-ci #394, PyCOLMAP #196, CodeQL #339 — PASS.
- V2L3.4 / PR #48 / `05125a0d6eec43e055cf09116cb3c76575ac4a59`: fast-ci #399, PyCOLMAP #200, CodeQL #344 — PASS.
- V2L3.5 / PR #49 / `934f5b5778790d52fd634396708bd899b4607ebc`: fast-ci #410, PyCOLMAP #205, CodeQL #355 — PASS.
- V2L3.6 implementation / `d84b39ea2d0fd05340fcc945d8c97d97b113606c`: fast-ci #414 and CodeQL #359 — PASS.

The final PR #50 handoff head must pass fast-ci, native PyCOLMAP 4.2.0 integration and CodeQL on the same SHA before merge; those final run identifiers are kept in PR evidence to avoid self-referential review commits.

## Decision

**PASS.** V2L3 is complete without routing/default-selection, benchmark-policy, V2L4 quality semantics, new model downloads or V2L12 canonical geometry behavior. The next product item is `V2L4.1 — Stable failure taxonomy`.
