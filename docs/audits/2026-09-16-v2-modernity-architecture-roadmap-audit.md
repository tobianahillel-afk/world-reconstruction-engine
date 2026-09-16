# WRE v2 modernity, architecture and roadmap audit — 2026-09-16

Status: **AUDITED WITH FOLLOW-UPS**

This audit checks whether WRE v2 is architecturally complete, whether its technology-selection process can keep choosing modern specialists, whether currently pinned tooling is reasonably current, and whether the implementation roadmap is sized for one-agent development runs.

This document is evidence, not a frozen technology list. The product/architecture freeze remains `docs/26_V2_PRODUCT_ARCHITECTURE_FREEZE.md`; the living candidate map remains `docs/27_RESEARCH_CANDIDATE_COVERAGE.md`.

## Executive verdict

### Architecture coverage — PASS

No missing top-level product responsibility was found in the reviewed 2025–2026 research landscape. Current and newly reviewed methods fit the frozen WRE responsibilities for observation/evidence, geometry, surface, appearance, materials/environment, dynamic/time, MasterScene/runtime, and quality/benchmark/orchestration.

The architecture should **not** be redesigned whenever a stronger paper appears. A stronger method normally becomes an adapter/benchmark candidate under an existing contract.

### “Always use the best option” policy — PASS AFTER HARDENING

It is impossible to truthfully freeze one universally “best” algorithm because quality depends on data profile, geometry/rendering objective, latency, hardware, license, checkpoint availability, reproducibility and maturity. The correct guarantee is procedural:

1. stable solver-independent WRE contract;
2. current candidate refresh immediately before an integration/promotion item activates;
3. exact dependency/model/checkpoint/license/hardware review;
4. comparison under the same WRE contract and representative fixtures;
5. per-dimension metrics and failure classes;
6. default promotion only through benchmark/quality policy.

The audit hardens this as an explicit pre-activation rule rather than relying only on prose.

### Work-item/lot sizing — PASS AFTER SPLITS

All lots remain at or below six work items. The existing one-PR-per-work-item and full-contract-before-ready rules are sound. Several future concise stubs still combined independent failure domains and were split during this audit before they became active.

The engineering protocol now adds an explicit one-run complexity gate. Wall-clock time cannot be guaranteed because model downloads, native builds, GPU jobs and CI queues vary, but the reasoning/code/review scope must fit one coherent development run. If not, the item is split before activation.

## Architecture audit

The six-layer architecture remains appropriate:

1. immutable observations;
2. organization/evidence;
3. geometry;
4. complementary scene representations;
5. production/orchestration;
6. target-specific runtime compilation.

Important separations remain correct and modern:

- geometry is independent from photorealistic appearance;
- physical surface is independent from splat/radiance appearance;
- materials and environment have explicit ownership;
- event-time 4D is distinct from long-term chronology;
- MasterScene is distinct from RuntimeScene;
- routing chooses work, not truth;
- generated/inferred/reconstructed provenance remains explicit;
- external research methods remain behind adapters;
- Artifact DAG, cache, quality gates, failure taxonomy, benchmarks, scheduling, checkpoint/resume and human review are first-class product capabilities.

No reviewed 2026 candidate required a seventh fundamental architecture layer.

## Current installed/pinned stack audit

### COLMAP / PyCOLMAP 4.2.0 — CURRENT / STRONG BASELINE

The retained environment is not an obsolete photogrammetry fossil. COLMAP 4.2.0 is a 2026 release and includes a substantially modernized classical/hybrid baseline. Relevant upstream evolution includes global mapping/GLOMAP capabilities, learned-feature/matching support such as LoMa, native equirectangular support introduced in recent releases, and accelerated bundle-adjustment work such as Caspar in the recent release line.

Action: keep it as a classical precision baseline, but V2L12 must exercise the current viable 4.2 capabilities rather than mechanically preserving the old V1 route. Feed-forward/hybrid candidates are compared separately under V2L13/V2L14.

### NumPy 2.5.3 — CURRENT

The native COLMAP integration lane pins a current NumPy 2.5.x environment with PyCOLMAP 4.2.0. No modernization action is currently required beyond normal dependency review.

### ExifRead 3.5.1 — CURRENT

The pinned ExifRead release is current and suitable for raw EXIF extraction. It remains intentionally narrow: raw metadata extraction does not imply semantic time/GPS/camera truth.

### FFmpeg 6.1.1 Ubuntu Noble package — REVIEW REQUIRED, NOT BLIND UPGRADE

Upstream FFmpeg has moved to the 8.x release line, while WRE pins the Ubuntu 24.04/Noble 6.1.1 package for exact reproducibility and a reviewed subprocess-only licensing posture.

This is a real freshness gap, but replacing the package merely because the version number is larger would be poor engineering. Before media/frame-selection ownership expands, perform a focused FFmpeg freshness review covering:

- whether 8.x adds material WRE capabilities/performance/security relevant to decode, audio or frame extraction;
- availability of a reproducible supported package/container across target installations;
- GPL/LGPL build composition and redistribution implications;
- deterministic behavior and regression fixtures;
- migration cost versus retained Noble package stability.

Until that review, 6.1.1 is a reproducible baseline, not a claim to be the newest/best FFmpeg.

### uv 0.12.13 — NEAR CURRENT

The repository pin is only a small patch distance behind the observed 0.12.x release line. Dependabot already checks Python/GitHub Actions dependencies weekly. Update through normal CI rather than a special architecture change.

### Python >=3.12,<3.14 — COMPATIBILITY REVIEW LATER

Python 3.14 is available and PyCOLMAP 4.2 provides modern wheels, but WRE currently targets 3.12/3.13. The raw-EXIF dependency's documented support and the full native/tool stack should be checked before widening the supported runtime. A newer interpreter alone is not a product improvement.

### Project license — UNRESOLVED SHIPPING GOVERNANCE

`registry/dependencies.yaml` still records `project_license: undecided`.

This does not block contract development, but it **must be resolved before commercial shipping/default decisions that depend on model/checkpoint redistribution**. Candidate code licenses, model/checkpoint licenses and training-dataset terms are separate concerns.

## Research landscape audit

The existing technology blueprint already covers many strong 2025–2026 families: DA3, VGGT-Ω, Pi3/Pi3X, MapAnything/CUT3R, LongStream, ZipMap, Scal3R, VGG-T³, GLUEMAP, LightGlue/modern matchers, AllTracker/CoTracker3, MoVieS/MoRe/D4RT, MotionScale/Shape of Motion/MoSca/ProDyG, 4D Primitive-Mâché, PFGS360, OMeGa/SurfaceSplat/MeshSplatting, WildGaussians/7DGS, GaRe/MatSpray/IR-HGP, historical chronology candidates and modern LOD/compression systems.

The audit also identified additional 2026 candidates/families that should remain visible during future refreshes:

- **S2D** — sparse-to-dense reconstruction with minimal input;
- **HeroGS** — sparse-view Gaussian reconstruction;
- **SparseSplat** — compact feed-forward Gaussian maps;
- **SGS-Intrinsic** — sparse-view inverse rendering/material-light disentanglement;
- **DGGT** — feed-forward dynamic-driving 4D reconstruction family;
- **V-DPM** — dynamic point-map/4D video reconstruction family;
- **SV-GS** — sparse-view 4D Gaussian reconstruction;
- **CAGS** — confidence-guided sparse-view high-resolution Gaussian reconstruction;
- compact/adaptive feed-forward Gaussian families such as **C3G**, Z-order Gaussian transformers and off-grid/adaptive-density primitive prediction;
- stateful/SLAM families such as LoGeR/VGGT-SLAM/MASt3R-SLAM where their actual release, license and benchmark evidence justify evaluation.

These names are **candidates**, not new defaults and not frozen architecture.

### License/shipping caveat

A benchmark winner can be unusable as a shipping default. For example, some state-of-the-art research repositories/checkpoints use non-commercial/research-only terms even when nearby code is permissive. Every concrete candidate therefore requires a fresh code + checkpoint + dataset/redistribution review before promotion.

The candidate-selection system must optimize for the strongest **shippable and reproducible** method for a given profile, not the largest headline benchmark number.

## Roadmap sizing audit

### Structural rule

The machine roadmap enforces at most six work items per lot. Every current lot satisfies that limit after the audit splits.

### Future items split by this audit

The following concise planned items were too broad because they combined independently failing specialist/product domains:

- V2L40: HDR and low-light integration are now separate work items;
- V2L42: material-fusion and inverse-rendering integrations are now separate work items;
- V2L44: camera/mask corrections are separated from reconstruction-region/temporal corrections;
- V2L45: DCC packaging is separated from game-engine packaging;
- V2L51: final validation is explicitly validation-only and cannot hide new product implementation.

### Activation rule

A planned stub may remain concise. Before it becomes `ready`, it must be expanded into a complete deny-by-default contract and pass the one-run complexity gate.

Split before activation when an item would require:

- multiple independent external model integrations;
- multiple independently shippable responsibilities;
- unrelated UI/export/runtime surfaces;
- dependency approval + adapter + benchmark/default promotion that cannot be reviewed coherently together;
- new product implementation inside a final validation item.

Heavy research integrations may be split further into dependency/checkpoint approval, adapter/normalization, real integration fixture, and benchmark/default promotion when needed.

## CI and supply-chain audit

Current repository practices are strong:

- `uv` frozen lockfile install;
- Ubuntu 24.04 CI;
- Ruff lint/format;
- Pyright;
- pytest;
- actionlint;
- CodeQL lane;
- dependency-review workflow;
- weekly Dependabot for uv and GitHub Actions;
- GitHub Actions pinned by full commit SHA;
- exact PyCOLMAP integration lane.

The audit does not recommend replacing these simply for novelty. Modernity means maintained, reproducible and fit-for-purpose, not churn.

## Changes made by this audit

- replaced the last compatibility-first engineering language with V2-authoritative legacy-donor migration;
- added an explicit candidate-freshness gate before solver/model integration or promotion;
- added an explicit pre-activation one-run complexity gate;
- encoded those policies in the machine roadmap;
- modernized future classical-geometry wording around current COLMAP capabilities rather than a frozen V1 route;
- split the oversized specialist/human-review/export work-item stubs listed above;
- made V2L51 validation-only;
- documented missing/current 2026 candidate families and dependency freshness findings.

## Open follow-ups

1. **FFmpeg freshness review** before media/video processing becomes a larger production dependency.
2. **Project-license decision** before learned-model/checkpoint shipping approvals become business-critical.
3. **Python 3.14 support review** when there is a concrete deployment/performance benefit and all core dependencies support it.
4. Continue candidate refresh at every external solver/model integration item.
5. Use V2L49 continuous benchmark/default refresh to make replacement routine after the initial product is assembled.

## Final audit conclusion

WRE cannot honestly promise that a static list of algorithms is eternally “the best.” It can—and after this audit is explicitly designed to—make the strongest current choice reproducibly for each responsibility and data profile, while keeping old baselines only when they still provide measurable value.

The architecture is modern and sufficiently general for the reviewed 2026 landscape. The primary remaining risk is not an architectural gap; it is failing to refresh candidates/licensing/versions when future integration items activate. The hardened activation and benchmark rules are intended to prevent that failure mode.
