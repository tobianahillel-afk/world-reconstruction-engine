# COLMAP environment adapter

L3 introduces the first reconstruction-engine integration. L3.1 is deliberately limited to proving and describing a compatible COLMAP environment; it does not extract features, match images, verify geometry or run a mapper.

## Reference integration

WRE's L3 reference integration is the **official PyCOLMAP 4.2.0 binding** published by the COLMAP project. The upstream release is `colmap/colmap` tag `4.2.0`; PyCOLMAP reports its binding version through `pycolmap.__version__`, the underlying COLMAP version through `pycolmap.COLMAP_version`, build provenance through `pycolmap.COLMAP_build`, Ceres identity through `pycolmap.__ceres_version__`, and build/runtime GPU capability through the upstream `has_cuda` attribute.

The adapter requires the exact binding version `4.2.0` and exact COLMAP version string `COLMAP 4.2.0`. It preserves the upstream build and Ceres strings verbatim rather than trying to infer private build semantics from them.

COLMAP itself is BSD-3-Clause. Upstream explicitly notes that its dependencies are separately licensed and can affect a resulting build. L3.1 therefore approves the official PyCOLMAP 4.2.0 binding as the reference environment but does not claim that every possible downstream/private COLMAP build shares identical redistribution obligations.

## Why PyCOLMAP instead of the Ubuntu COLMAP package

The reference WRE Python range is Python 3.12–3.13. Upstream PyCOLMAP 4.2.0 publishes official manylinux/macOS/Windows wheels for those Python versions and exposes the same version/build identity needed by the adapter. Ubuntu 24.04's distribution COLMAP package is older than the WRE 4.2 baseline, so it is not the reference environment for L3.

L3.1 keeps PyCOLMAP as an **external reconstruction environment requirement** rather than adding it to WRE's runtime lockfile. No L3.1 production path invokes feature extraction yet. A dedicated integration workflow installs exactly `pycolmap==4.2.0` and proves the adapter against the real official wheel. L3.2, which first owns actual feature extraction, must decide whether to promote PyCOLMAP into the normal runtime dependency/lock contract before using it in the feature path.

This separation keeps the normal fast lane small and avoids changing `pyproject.toml`/`uv.lock` merely to inspect an external solver environment.

## Adapter contract

`inspect_colmap_environment` must:

1. import `pycolmap` lazily so the WRE package remains importable when the external solver environment is absent;
2. fail with an explicit WRE environment error when PyCOLMAP cannot be imported;
3. require exact PyCOLMAP version `4.2.0`;
4. require exact underlying `COLMAP 4.2.0` identity;
5. require non-empty build and Ceres version strings;
6. require the upstream GPU capability field to be a boolean;
7. return an immutable solver-independent identity record containing only environment/provenance facts;
8. perform no feature extraction, database creation, matching, geometry verification or reconstruction.

The environment identity is descriptive evidence about which solver build is available. It is not a reconstruction result and does not imply that any future solver output is accepted as geometric truth.

## Testing

Fast CI uses injected/fake module objects to exercise missing, malformed, wrong-version and valid environment identities without making the entire repository depend on a heavy solver wheel.

A separate path-scoped COLMAP integration workflow installs exact `pycolmap==4.2.0` and runs the real environment smoke test. That workflow is the L3.1 compatibility evidence for the official upstream wheel.

## Explicit L3.1 boundary

Not implemented by L3.1:

- SIFT or learned feature extraction;
- COLMAP database population;
- image import/camera-model policy;
- pair generation or matching;
- two-view geometry verification;
- incremental/global/hierarchical mapping;
- bundle adjustment;
- reconstruction import into WRE domain models;
- GPU route selection or performance policy.

Those behaviors remain in their owning L3 and later work items. L3.2 is the next permitted implementation item after L3.1 is reviewed and merged.
