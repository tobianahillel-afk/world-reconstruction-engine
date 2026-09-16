# WRE v2 performance, compression and execution audit — 2026-09-16

Status: **ACTIONABLE / ARCHITECTURE UNCHANGED**.

This audit asks whether WRE is missing practical techniques used by modern reconstruction, ML and large-scene runtimes that improve speed, memory, storage, bandwidth or interactive realism without changing the frozen v2 architecture.

The answer is: **the architecture already has the right homes, but several implementation strategies were too implicit.** This audit promotes them into the living performance playbook and roadmap so future adapters/runtimes benchmark them rather than rediscovering them late.

## Executive priority

### P0 — high-value work avoidance / reuse

1. **Reusable decoded-image and multiresolution-pyramid artifacts**
   - Gap found: downstream retrieval/matching/geometry adapters could otherwise decode and resize equivalent images independently.
   - Action: new planned `V2L6.6` before retrieval.

2. **Explicit cross-mode warm starts**
   - PREVIEW/FAST outputs may initialize QUALITY/MASTER when the downstream adapter explicitly consumes them.
   - This must be an artifact dependency, never a hidden promotion of weak output.

3. **Quality-gate early exit**
   - Do not run a more expensive route when the requested quality contract already passes unless that mode explicitly requires comparison/consensus.

4. **Residual/uncertainty-driven refinement**
   - Spend expensive resolution/iterations/geometry only in regions/components where measured error or uncertainty justifies it.

These normally save more compute than low-level kernel tuning because they remove entire jobs or reduce the work domain.

### P1 — large GPU / appearance / runtime gains

5. **Asynchronous CPU decode → H2D → GPU compute → writeback pipeline**
   - bounded queues;
   - pinned buffers when useful;
   - overlap transfer and compute;
   - avoid GPU→CPU round trips when the next consumer remains on GPU.

6. **Mixed precision as a benchmarked execution profile**
   - FP16/BF16/TF32 where stable;
   - preserve higher precision in sensitive operations;
   - no blanket FP8/INT8 MASTER policy without quality evidence.

7. **Compiled execution / CUDA graphs / autotuning where suitable**
   - compare eager and compiled cold-start + steady-state performance;
   - avoid shape-driven recompilation with bounded buckets;
   - cache only against compatible hardware/runtime/model identity.

8. **Shape bucketing, bounded batching and persistent model workers**
   - avoid reloading checkpoints/reinitializing CUDA for each small job;
   - use measured VRAM headroom, not fixed universal batch sizes.

9. **gsplat efficiency controls when gsplat is selected**
   - packed representation;
   - sparse gradients where optimizer-compatible;
   - projected-radius clipping for visually negligible far splats;
   - explicit finite Gaussian/densification/pruning budgets;
   - distributed execution only when its coordination cost is justified.

### P1 — runtime delivery and compression

10. **Do not ship raw PLY as the assumed final splat runtime format**
    - benchmark SPZ-class compact interchange/delivery;
    - SOG for compact non-streamed scenes;
    - Streamed SOG or equivalent spatial multi-LOD format for very large scenes.

11. **Scene-wide primitive budget**
    - cap visible splats/geometry according to device/performance mode;
    - allocate detail across the scene rather than per asset independently.

12. **Error-driven LOD, not distance-only forever**
    - screen-space/projected error can spend primitives where they improve the image most;
    - retain distance mode for low-memory targets when appropriate.

13. **Coarse-first progressive loading**
    - first useful frame before full high-detail download;
    - predicted prefetch and memory-pressure eviction.

14. **Meshoptimizer/glTF delivery path**
    - vertex-cache/fetch optimization;
    - measured simplification;
    - `EXT_meshopt_compression` where supported;
    - KTX2/BasisU textures;
    - GPU instancing for genuinely repeated entities.

### P1 — media pipeline

15. **FFmpeg 8 compatibility**
    - separate PR #56 experimentally installs exact Ubuntu 26.04 `ffmpeg=7:8.0.1-3ubuntu2`.
    - the actual version-agnostic WRE extraction contract passes on FFmpeg 8: synthetic FFV1 creation, probing, deterministic timestamps/frame indices, PNG extraction, persistence and idempotent rerun.
    - this proves technical compatibility; primary-baseline migration still needs a packaging/stability decision because GitHub `ubuntu-26.04` is currently a preview runner.

16. **Optional hardware video decode**
    - benchmark NVDEC/FFmpeg hardware acceleration on NVIDIA profiles and equivalent backends elsewhere;
    - keep a reference software path;
    - include transfer cost and timestamp/pixel equivalence in evaluation.

17. **Demand-driven still-image processing**
    - libvips-class tiled/demand-driven decode/resample is a candidate for huge images and long preprocess pipelines;
    - benchmark before adoption rather than materializing full-resolution intermediates by default.

### P2 — target-specific refinements

18. Dynamic resolution and target-FPS-driven runtime budgets.
19. XR/foveated detail budgets where target platform support exists.
20. Chunk-level random/range access, per-chunk compression and lazy materialization for huge artifacts.
21. Memory mapping / zero-copy only where it demonstrably reduces copies without weakening lifetime/integrity guarantees.
22. Hardware-specific kernel/profile tuning after higher-level work avoidance is exhausted.

## What WRE already did intelligently

The audit did **not** find a naïve monolithic pipeline. Existing decisions already provide major efficiency foundations:

- content-addressed Artifact DAG;
- verified materialization and reuse;
- dependency-scoped invalidation;
- PREVIEW / FAST / QUALITY / MASTER budgets;
- adaptive routing/fallback rather than one fixed pipeline;
- separate MasterScene and target RuntimeScene;
- planned spatial/temporal chunks, LOD, culling, compression and prefetch;
- planned resource scheduler and checkpoint/resume;
- adapter/model/hardware identity and reproducibility;
- continuous benchmark/default refresh.

Those choices are more important than a single fast kernel because they prevent repeated work at system scale.

## Concrete roadmap changes from this audit

- add `V2L6.6` reusable decoded-image/multiresolution-pyramid artifact cache;
- make V2L18 benchmark quality **and** execution/resource efficiency;
- make V2L21 explicitly own target formats, global primitive budgets, error-aware LOD, progressive first frame and full runtime budgets;
- make V2L22 measure time-to-first-frame as well as FPS/memory;
- make V2L23 compare artifact reuse/warm-start/early-exit across quality modes;
- make V2L45 evaluate mature glTF/mesh/texture compression paths;
- make V2L46 model execution profiles and mitigation alternatives such as chunking/batching/precision/concurrency;
- make V2L49 compare and promote execution profiles separately from model defaults;
- add machine policy `activation_requires_performance_review: true`.

## Anti-patterns explicitly rejected

- upgrading every dependency solely because its version number is larger;
- always using GPU decode even when transfer/setup dominates;
- always using mixed precision regardless of geometry stability;
- treating compile/JIT time as free;
- setting one global batch size for every GPU/input shape;
- rendering every Gaussian because it exists;
- using raw PLY/raw textures as a production runtime by default;
- building full-detail assets before presenting a first useful frame;
- optimizing microbenchmarks while end-to-end WRE becomes slower;
- trading physical geometry correctness for render FPS without changing the claimed quality/product.

## Acceptance principle for future optimization work

Every substantial optimization should report at least the dimensions it materially touches:

- end-to-end and stage latency;
- cold-start versus steady-state time;
- CPU/GPU utilization;
- peak/steady VRAM and RAM;
- I/O / transfer bytes;
- artifact/package/download size;
- time to first useful result/frame;
- runtime frame time/FPS;
- geometry/camera/render/temporal metrics relevant to the owning contract.

Performance work is successful only when the measured gain is worth its complexity and does not silently invalidate the quality/provenance claims that remain advertised.

## Final verdict

WRE was already architecturally prepared for high performance, but it left several modern techniques as generic future words such as “compression”, “LOD” or “resource-aware scheduling”. This audit turns the most important of those into concrete, testable strategies without freezing one vendor/library or contaminating the stable architecture.

The highest-value direction is not “turn on every optimization”. It is a hierarchy: **reuse → skip → coarse-to-fine → focus on uncertainty → overlap/parallelize → accelerate kernels → compress for the target runtime**.
