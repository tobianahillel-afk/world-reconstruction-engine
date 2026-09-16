# WRE v2 performance, compression and efficiency playbook

Status: **LIVING / NOT FROZEN**.

This document records implementation strategies that can materially reduce latency, VRAM/RAM, storage, bandwidth or runtime cost without changing the frozen WRE product architecture. It complements the stable contracts in the core; exact libraries, kernels, formats and thresholds remain benchmark-driven.

The detailed ownership and activation timing for these strategies is mapped in `29_PERFORMANCE_INTEGRATION_MAP.md`.

The governing principle is:

> **Avoid unnecessary work first; accelerate the remaining work second. Never buy speed by silently weakening the claimed quality or provenance class.**

An optimization is not a default merely because it is fashionable. It must be measured against the same WRE quality contract and representative data profile.

## 1. Optimization order

Prefer optimizations in this order:

1. skip work whose artifact already exists and is VERIFIED;
2. avoid processing irrelevant media/frames/pairs/regions;
3. reuse lower-cost intermediate artifacts when they are valid inputs to later routes;
4. reduce resolution/detail only where the selected quality mode permits it;
5. chunk and parallelize independent work;
6. overlap I/O, transfer and compute;
7. use lower precision/compiled kernels only after quality equivalence is measured;
8. compress/quantize runtime outputs according to target-device error budgets.

A faster implementation that recomputes avoidable stages is not considered optimized.

## 2. Reusable media decode and image pyramids

Repeated JPEG/video decode and repeated resize operations can become a hidden tax when retrieval, matching, geometry, depth and learned models each request their own resolutions.

WRE should materialize reusable derived image/frame artifacts when the same deterministic decode/preprocess result is consumed by multiple downstream stages. Their artifact identity must include at least source bytes, decoder identity/version, requested pixel/color convention and resolution/preprocess configuration.

Recommended behavior:

- keep immutable original media as source truth;
- cache decoded frames only when reuse justifies storage;
- support reusable multiresolution image pyramids rather than decoding/resizing the original independently in every adapter;
- preserve exact links back to original observations;
- do not mix later photometric normalization into a geometry/retrieval decode artifact unless that transformation is explicit in the artifact identity.

For very large still images, demand-driven/tiled libraries such as libvips are candidates because they avoid materializing unnecessary full-resolution intermediates and can fuse resize/processing pipelines. Benchmark against simpler decoders; do not mandate libvips without evidence.

## 3. Video hardware decode as an optional acceleration route

Canonical media semantics must not depend on one GPU vendor, but high-volume video ingest may optionally use hardware decode when it produces contract-equivalent observations.

Candidate routes include FFmpeg hardware acceleration such as NVDEC on supported NVIDIA systems and equivalent platform backends elsewhere.

Rules:

- retain a software/reference decode path;
- hardware decode is selected by capability/resource policy, not assumed;
- compare timestamp/frame-selection semantics and decoded pixel behavior against the reference path;
- avoid GPU→CPU round trips when the next consumer is already on the GPU;
- when frames must return to CPU/persistence, include transfer cost in the benchmark;
- never make hardware decode a source of hidden nondeterministic frame identity.

## 4. Asynchronous CPU / transfer / GPU pipeline

For learned adapters, input preparation should not serially alternate between CPU decode, H2D copy, inference and artifact writeback when these stages can overlap.

Candidate execution shape:

```text
CPU decode/preprocess N+1
        ||
pinned-memory async H2D N
        ||
GPU compute N-1
        ||
CPU artifact encode/write N-2
```

Useful mechanisms may include bounded producer/consumer queues, pinned host buffers, non-blocking transfers and multiple CUDA streams where the adapter and hardware support them.

The scheduler/resource estimator must cap queue depth and pinned memory; unbounded prefetch is a memory leak disguised as optimization.

## 5. Shape bucketing and batching

Learned models frequently run faster when compatible images/views are processed together, but naïve batching can waste memory through padding or increase latency.

Adapters may benchmark:

- grouping inputs by compatible resolution/aspect/camera profile;
- bounded dynamic batches based on measured VRAM headroom;
- micro-batching for large scenes;
- persistent workers/model instances to avoid repeated checkpoint load and CUDA initialization.

Batch policy is part of normalized execution configuration when it can alter bytes or numerical behavior.

## 6. Mixed precision and tensor-core execution

FP16/BF16/TF32-class execution can materially accelerate learned geometry, tracking, appearance and inverse-rendering workloads on suitable GPUs.

Policy:

- benchmark FP32/reference versus candidate lower-precision modes on the same fixture;
- measure geometry/camera/temporal quality, not only render PSNR;
- use BF16 where its numerical range materially improves stability;
- allow sensitive reductions/optimizers to remain in higher precision;
- FP8/INT8 or stronger quantization requires separate acceptance evidence and must never be silently enabled for MASTER output;
- precision choice participates in configuration/hardware reproducibility identity when material.

## 7. Compiled execution and CUDA graphs

For repeated model inference/training shapes, adapters may benchmark current compiler/runtime acceleration such as `torch.compile`, kernel autotuning, CUDA graphs or ahead-of-time compiled artifacts.

These can reduce Python/kernel-launch overhead but can also increase compilation latency, memory use or shape-specialization complexity.

WRE should therefore:

- record eager versus compiled benchmark evidence;
- reuse compiled artifacts/caches only when compatible with model, configuration and hardware/runtime identity;
- prefer shape buckets that avoid pathological recompilation;
- include cold-start and steady-state latency separately;
- fall back to the validated eager path when compilation is unsupported or regresses quality/stability.

## 8. Coarse-to-fine reconstruction

Expensive high-resolution processing should normally be earned by evidence.

Applicable patterns:

- establish cameras/global geometry at lower resolution;
- refine only after global consistency is plausible;
- increase image/depth/render resolution progressively;
- allocate denser geometry or Gaussians to regions with high projected error/detail;
- keep sky/distant background/environment from consuming the same detail budget as close physical geometry.

This pattern is especially important for very large scenes, sparse-view reconstruction and MASTER routes.

## 9. Residual / uncertainty-driven refinement

Uniformly spending MASTER compute everywhere is wasteful.

Quality metrics should be usable spatially or by component when possible. Expensive work can focus on:

- high reprojection residuals;
- low depth consistency;
- uncertain camera regions;
- holes/weak surface support;
- high appearance error on held-out views;
- dynamic boundaries/occlusions;
- temporal disagreement/change ambiguity.

The unresolved low-confidence region remains explicit if refinement still cannot support it.

## 10. Cross-mode warm starts

`PREVIEW`, `FAST`, `QUALITY` and `MASTER` should not automatically mean four independent reconstructions.

When a later adapter can validly consume an earlier artifact, route composition should allow explicit warm starts such as:

- feed-forward cameras/depth initializing global refinement;
- FAST surface/depth seeding QUALITY fusion;
- lower-resolution appearance initializing higher-resolution training;
- previous MasterScene version initializing incremental updates.

The lower-mode artifact must appear as an explicit input/provenance dependency. A warm start never silently promotes PREVIEW quality to MASTER truth.

## 11. Quality-gate early exit

The cheapest valid route should be allowed to stop once it already satisfies the requested quality contract.

Examples:

- do not run classical/global refinement merely because it exists when FAST geometry already passes the FAST policy;
- do not run multiple expensive specialists when one accepted route has enough confidence and the requested mode does not require consensus;
- do run comparison/refinement when MASTER policy explicitly requires it.

This is one of the highest-value optimizations because it removes complete jobs rather than making them slightly faster.

## 12. Chunking, submaps and parallel execution

Large collections and long videos should expose independent chunks/submaps/windows that can be processed concurrently subject to resource limits.

Good boundaries preserve:

- overlap necessary for alignment;
- deterministic chunk identity;
- resumability;
- local cache reuse;
- explicit merge/refinement artifacts.

Parallelism is bounded by CPU/GPU/RAM/disk and downstream merge cost; maximum concurrency is not automatically maximum throughput.

## 13. Gaussian training/rasterization efficiency

When gsplat or an equivalent backend is selected, benchmark backend-specific efficiency features rather than using default parameters blindly.

Candidate controls include:

- packed sparse representations for large scenes;
- sparse gradients when compatible with the optimizer;
- projected-radius clipping for Gaussians too small to contribute visibly;
- antialiased rasterization where quality requires it;
- lower spherical-harmonic degree for PREVIEW/FAST or distant LODs;
- bounded Gaussian population / explicit densification and pruning strategy;
- distributed rasterization/training where the scene justifies multi-GPU cost.

Each speed/memory setting must be evaluated together with visual and geometry disagreement metrics.

## 14. Finite representation budgets

Optimization/training should not be allowed to grow representations without a resource objective.

Track at least:

- Gaussian/primitive count;
- mesh triangle/vertex count;
- texture pixels/bytes;
- spatial chunks;
- temporal samples;
- runtime-visible memory.

Densification, pruning, merging, simplification and quantization should target explicit quality-versus-size curves rather than arbitrary fixed thresholds.

## 15. Splat runtime formats

PLY is useful as interchange/debug data but should not be assumed to be the final delivery format.

Current runtime candidates include:

- **SPZ-class** compact splat storage for portable compressed packages;
- **SOG** for compact non-streamed scenes;
- **Streamed SOG** or equivalent spatial multi-LOD bundles for very large scenes.

Runtime-format choice is target-specific. Preserve a richer master/interchange representation when needed so changing runtime compression never requires geometry reconstruction.

## 16. Global splat budget and error-driven LOD

Large-scene runtimes should use a scene-wide visible primitive budget rather than independently rendering every asset at its preferred detail.

The runtime compiler/viewer should be able to choose LOD according to:

- global splat/memory budget;
- camera/frustum;
- projected screen size;
- per-LOD visual error where available;
- target FPS/resolution;
- device class;
- object/region importance;
- temporal importance.

Distance-only LOD is a baseline. Error-per-cost allocation is preferable when its extra metadata/residency cost is acceptable.

## 17. Coarse-first progressive loading

Time-to-first-frame is a separate product metric from final image quality.

For streamed scenes:

1. load the coarsest spatial/appearance representation first;
2. present an immediately navigable scene;
3. stream higher LODs by camera/time prediction and error contribution;
4. evict low-value detail under memory pressure.

A multi-gigabyte MasterScene should never require full download before the first useful view when the target runtime supports streaming.

## 18. Visibility and tiny-primitive culling

Runtime and training should avoid work that cannot affect the image materially.

Benchmark:

- frustum culling;
- spatial-tree chunk culling;
- occlusion strategies where representation permits them;
- projected-radius/tiny-splat rejection;
- far-plane/environment separation;
- lower SH/detail for distant primitives.

Culling thresholds belong to a target-quality profile, not a hidden renderer constant.

## 19. Mesh and texture runtime optimization

For explicit mesh/material outputs, evaluate mature glTF optimization rather than shipping naïve meshes/textures.

Candidate techniques/tools include:

- meshoptimizer/gltfpack-style vertex-cache and vertex-fetch optimization;
- geometry quantization and `EXT_meshopt_compression` where target runtimes support it;
- mesh simplification with measured geometric/silhouette error;
- KTX2/Basis Universal texture compression and mipmaps;
- atlas generation when it reduces draw calls without destroying material semantics;
- GPU instancing for genuinely repeated objects/components;
- separate visual mesh and collision/navmesh simplification.

## 20. Dynamic resolution and XR/foveated budgets

For interactive/XR targets, runtime quality can adapt to measured load without changing MasterScene truth.

Possible target policies:

- dynamic render resolution;
- target-FPS-driven splat/LOD budget adjustment;
- lower detail outside the focal region for XR/foveated rendering where platform support exists;
- temporal LOD for distant/low-importance dynamics.

These are runtime compilation/view policies and must not alter reconstructed provenance.

## 21. Artifact storage and chunk-level access

Very large derived artifacts should not require loading one monolithic blob merely to access a small region/time range.

Where the representation supports it, prefer:

- independently verifiable chunks;
- deterministic manifests;
- per-chunk compression;
- random/range access;
- lazy materialization;
- memory mapping or zero-copy reads when safe and useful.

Artifact integrity remains end-to-end explicit; compression is a storage/runtime representation concern rather than a reason to weaken content identity.

## 22. Performance telemetry is evidence

Every significant optimization must retain enough evidence to answer:

- what wall-clock stage became faster?
- what CPU/GPU utilization changed?
- peak and steady VRAM/RAM?
- bytes read/written/transferred?
- artifact/runtime size?
- time to first useful result?
- final FPS / frame time?
- did any geometry/render/temporal metric regress?

Do not merge a complex optimization because a microbenchmark improved while end-to-end WRE got slower.

## 23. Optimization ownership in the roadmap

Primary homes:

- media decode/pyramid reuse: `V2L6` plus later resource hardening;
- candidate-specific inference/training optimization: owning adapter lots (`V2L13`, `V2L18`, `V2L24`–`V2L28`, specialist lots);
- coarse-to-fine / residual refinement: geometry/appearance/dynamic owning lots plus quality gates;
- global runtime budgets, progressive streaming, splat/mesh compression: `V2L21`, `V2L22`, `V2L30`;
- resource-aware batching/precision/chunking/scheduling: `V2L46`;
- checkpoint/compiled-state compatibility where appropriate: `V2L47`/`V2L50`;
- continuous performance regression/default refresh: `V2L49`/`V2L51`.

## 24. What is deliberately not globally mandated

WRE must **not** globally mandate:

- one precision mode;
- one compiler mode;
- one GPU vendor;
- hardware decoding;
- one splat compression format;
- one Gaussian budget;
- one LOD error threshold;
- one batch size;
- one densification strategy.

The best setting changes with hardware, data, requested quality and runtime target. The stable requirement is that alternatives are measurable, reproducible and selected explicitly.

## 25. Standardized causal profiling and trace evidence

Wall-clock totals are necessary but not sufficient to optimize a heterogeneous media/GPU pipeline. Once representative GPU workloads exist, WRE should support retained causal evidence that distinguishes decode, preprocess, host-to-device transfer, model execution, postprocess, device-to-host transfer, compression and artifact writeback.

Candidate mechanisms include:

- lightweight stage timers and CUDA events;
- NVTX-class ranges for cross-library trace correlation;
- Nsight-class CPU/GPU timeline profilers on NVIDIA systems;
- equivalent platform/vendor profilers elsewhere.

The profiler is not a core dependency. The core/benchmark responsibility is to retain enough solver-independent evidence or references to explain why an execution profile improved or regressed end-to-end behavior.

Profile representative workloads rather than toy shapes. Always distinguish cold-start/compile/checkpoint-load cost from steady-state throughput where the distinction matters.

## 26. Accelerated still-image decode as an execution profile

Large photo collections can become CPU/decode/resize bound before learned retrieval or geometry begins. On compatible hardware, candidates such as nvImageCodec/nvJPEG-class decoding may reduce host work and can expose device-resident output to downstream GPU consumers.

Rules:

- `V2L6.6` first defines canonical decoded/pyramid artifact semantics and a portable reference path;
- accelerated still-image decode is an execution-profile candidate, never the observation contract itself;
- benchmark end-to-end behavior including orientation, color/pixel equivalence, batch shape, CPU utilization, VRAM, H2D transfer and downstream reuse;
- small images or small batches may not benefit, so do not globally enable GPU decoding;
- retain a portable/reference route and allow future non-NVIDIA accelerated backends behind the same semantics.

## 27. Learned execution profiles beyond eager/compile

Stable learned adapters may benchmark ahead-of-time inference/runtime systems such as TensorRT/Torch-TensorRT-class execution when the model, operators, shapes and target hardware make them viable.

Treat these as execution profiles of the same adapter contract, not new WRE product types.

Compare at least when applicable:

- reference/eager behavior;
- compiled behavior;
- precision mode;
- engine-build/cold-start cost;
- steady-state latency/throughput;
- RAM/VRAM;
- unsupported-op/fallback behavior;
- output quality and failure classes.

Do not require TensorRT globally. A profile is eligible only after representative evidence shows a useful end-to-end tradeoff and its exact runtime/engine compatibility can be reproduced.

## 28. Artifact residency, tiers and eviction

Content-addressed artifact identity does not imply that every materialized byte should remain on fast local storage forever.

Future resource policy should distinguish **logical artifact history** from **local materialization residency**. Evicting eligible recomputable bytes must not erase artifact identity, producer/input provenance, DAG relationships or the fact that the artifact previously existed.

Retention/eviction decisions may consider:

- irreplaceability/source status;
- recomputation cost and expected duration;
- downstream fan-out/value;
- artifact size;
- recent/frequent access;
- quality level and user retention policy;
- whether an artifact is a cheap preview/temp cache or an expensive MASTER intermediate;
- whether a cache is hardware/compiler-specific and safe to regenerate.

Blind LRU is not automatically appropriate. Re-materialized bytes must still satisfy normal verification/compatibility rules.

## 29. Motion- and scale-stable appearance/runtime quality

Still-view PSNR/SSIM/LPIPS can miss severe interactive defects. Gaussian/radiance representations should be evaluated on representative camera motion and scale changes where applicable.

Candidate quality dimensions include:

- popping during camera rotation/translation;
- shimmer/aliasing under zoom, focal-length or distance changes;
- unstable blend/sort behavior;
- LOD-transition visibility;
- temporal consistency along a saved camera path;
- stereo/XR discomfort indicators when an XR target is relevant.

Mip-Splatting-class antialiasing/scale-consistency methods and StopThePop-class view-consistent sorting are examples to refresh when the owning appearance/viewer work activates. The quality dimension is the product requirement; the named implementation is replaceable.

## 30. Progressive, vector-quantized and entropy-coded splat compression

Compact runtime delivery should evaluate more than fixed one-shot formats.

Candidate research families include:

- progressive bitstreams that improve quantity/quality as more bytes arrive, such as PCGS-class approaches;
- contextual entropy coding and adaptive quantization such as HAC++-class approaches;
- vector/codebook quantization of Gaussian attributes;
- combinations of pruning/masking, quantization and entropy coding.

Measure **rate-distortion and runtime cost**, not compression ratio alone:

- bytes versus visual quality;
- decode/initialization time;
- time-to-first-useful-view;
- random/spatial access suitability;
- progressive refinement behavior;
- device memory after decode;
- compatibility with the selected renderer/LOD hierarchy.

The rich `MasterScene` representation remains separate so a future better runtime compressor can be adopted without reconstructing geometry.

## 31. Hierarchical and multiscale Gaussian LOD

Distance-only LOD is a useful baseline but not the final strategy for large photorealistic scenes.

Candidate representations may provide:

- hierarchical Gaussian clusters;
- multiscale primitives trained/constructed for different screen frequencies;
- progressive-compression levels that double as LOD;
- error bounds or expected visual contribution per level/chunk.

The runtime compiler should expose enough metadata to allocate a global visible-cost budget by screen-space error/utility rather than letting every object independently choose maximum detail.

## 32. ANN execution backends for huge collections

Very large unordered collections should treat the ANN/index implementation as a replaceable execution backend.

Depending on collection size and hardware, benchmark relevant combinations of:

- exact CPU search;
- CPU ANN;
- quantized/compressed CPU ANN;
- GPU ANN;
- hybrid indexing/query strategies.

Measure candidate recall, index build time, query latency/throughput, RAM, VRAM, index size and transfer overhead. GPU ANN is not required for small collections merely because a GPU is present.

## 33. Topology-aware scheduling and direct storage-to-GPU paths

On multi-GPU/multi-NUMA/high-throughput systems, resource class alone can be insufficient. A later scheduler may benefit from empirical locality information such as:

- storage-to-device path;
- NUMA affinity;
- PCIe/NVLink/peer connectivity;
- measured H2D/D2H or peer bandwidth;
- GPU memory pressure and peer-access capability.

A single-GPU workstation may resolve all of this trivially; do not burden the deterministic scheduler baseline with premature topology logic.

When storage I/O is a measured bottleneck and the target system supports it, GPUDirect Storage/cuFile-class direct storage-to-GPU paths are candidate execution profiles. Keep a conventional path and include topology/setup constraints in the benchmark; direct I/O is not useful merely because the API exists.

## 34. Custom Triton/CUDA kernels are last-mile work

Do not plan custom kernels by default. First use mature libraries, vectorized framework operations, batching, compilation and known backend controls.

A custom Triton/CUDA fused kernel becomes appropriate only when representative profiling shows a durable hotspot that materially limits WRE and existing maintained implementations cannot satisfy the same contract efficiently.

Any such work belongs to the owning adapter/runtime implementation and must have:

- a reference implementation;
- numerical/quality equivalence tests;
- hardware/runtime compatibility evidence;
- end-to-end benchmark improvement, not only kernel microbenchmark improvement;
- a maintained fallback when the custom path is unsupported.

## 35. Activation timing summary

The intended sequence is deliberately staged:

1. `V2L4.4` — benchmark/performance evidence vocabulary, not profiler integration;
2. `V2L6.6` — canonical reusable decode/pyramid artifacts and portable reference path;
3. `V2L13.6` — first representative learned GPU profiling/execution-profile benchmark;
4. `V2L18`/`V2L21`/`V2L22` — motion-stable appearance, advanced compression and hierarchical LOD;
5. `V2L37` — CPU/GPU/quantized ANN comparison at large-collection scale;
6. `V2L46` — locality-aware estimation plus materialization residency/tiering/eviction and optional direct-I/O evaluation;
7. `V2L49` — continuous execution-profile comparison/promotion;
8. custom kernels only after all earlier stages leave a measured hotspot.

See `29_PERFORMANCE_INTEGRATION_MAP.md` for the full responsibility matrix and review checklist.

## Final principle

The product should become faster over time by replacing expensive work with reusable artifacts, better routing, bounded representations and target-aware execution—not by accumulating opaque performance flags.

A new optimization is successful when it improves a measured WRE dimension while preserving the contract dimensions it still claims to satisfy.
