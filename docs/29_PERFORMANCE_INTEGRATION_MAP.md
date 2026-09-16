# WRE v2 performance integration map

Status: **LIVING / NOT FROZEN**.

## Purpose

This document maps advanced performance, compression and execution ideas into the frozen WRE v2 architecture and executable roadmap. It complements `14_TECHNOLOGY_SELECTION.md`, `27_RESEARCH_CANDIDATE_COVERAGE.md` and `28_PERFORMANCE_OPTIMIZATION_PLAYBOOK.md`.

The mapping exists to prevent two opposite mistakes:

1. implementing hardware/vendor optimizations before WRE can measure whether they matter;
2. postponing all performance design until the end, when artifact, runtime or scheduling contracts would already make efficient execution difficult.

The architectural rule remains unchanged:

> stable WRE semantics in contracts and artifacts; replaceable algorithms, execution profiles and hardware-specific accelerations behind adapters, runtime compilation and scheduling policy.

No optimization in this document creates a new top-level architecture layer. The reviewed ideas fit the existing observation, organization/evidence, production/orchestration and runtime responsibilities.

## Integration principles

### Measure before specializing

A hardware-specific optimization must be justified by representative WRE profiling or benchmark evidence. A microbenchmark is insufficient if end-to-end wall clock, quality, reliability or resource use regresses.

### Execution profile is not product semantics

An adapter may eventually expose several validated execution profiles, for example reference/eager execution, compiled execution, lower precision, TensorRT/Torch-TensorRT-class execution or a future backend. All profiles must preserve the same owning WRE input/output contract and declared provenance/quality semantics.

Execution-profile identity becomes reproducibility-relevant when it can materially change output bytes, numerical behavior, resource requirements or failure behavior.

### Logical artifact identity is separate from local residency

Artifact metadata, identity, provenance and DAG relationships remain retained according to product semantics even when recomputable local materialization bytes are evicted under a future storage-pressure policy. Cache eviction must not masquerade as deletion of project history.

### Keep vendor names outside stable contracts

Core types must not become `NvJpegArtifact`, `TensorRTGeometrySolution`, `GPUDirectScene`, `StopThePopAppearance` or equivalents. Concrete technologies belong in living candidate docs, registries and adapter/runtime implementations.

### Optimize only after correctness is measurable

The owning capability must first have a stable contract and enough metrics to detect semantic or quality regression. Custom kernels are last-mile work, not foundational architecture.

## Placement matrix

| Optimization family | Architectural owner | Primary roadmap home | Activation point | Status |
| --- | --- | --- | --- | --- |
| standardized stage profiling / traces | benchmark + adapter execution | `V2L4.4`, first real GPU use in `V2L13.6`, continuous in `V2L49` | benchmark record can reference evidence; profiler integration waits for a representative GPU adapter | required capability |
| canonical decoded images / pyramids | observation + artifact reuse | `V2L6.6` | before downstream retrieval/geometry repeatedly decode the same media | required capability |
| GPU still-image decode | adapter/media execution profile | `V2L6.6` contract, evaluated from `V2L13.6` onward | only when profiling shows decode/resize/transfer is material | conditional execution profile |
| compiled / TensorRT-class learned execution | adapter execution profile | `V2L13.6`, repeated by owning learned lots, promoted through `V2L49` | after one stable learned adapter and reference measurements exist | conditional execution profile |
| motion/scale-stable splat rendering | appearance metrics + viewer runtime | `V2L18.4`, `V2L18.6`, `V2L22.6` | with first serious Gaussian/radiance appearance path | required quality dimension; candidate implementation replaceable |
| progressive / VQ / entropy splat compression | runtime compiler | `V2L21.3` | only after canonical `AppearanceModel` exists | required runtime capability; format replaceable |
| hierarchical / multiscale / progressive LOD | runtime compiler | `V2L21.4`, exercised in `V2L22.6` | with first target-specific `RuntimeScene` | required runtime capability |
| CPU/GPU ANN and quantized retrieval | organization/evidence at scale | `V2L37.1`, `V2L37.5` | only for very large collections | backend replaceable |
| topology-aware resource placement | production scheduler | `V2L46.3`, `V2L46.6` | when heterogeneous/multi-GPU/multi-NUMA hardware makes locality material | required scheduler modelling at hardening stage |
| artifact residency/tiering/eviction | artifact materialization + resource policy | `V2L46.5` | when derived artifacts become large enough for storage pressure to matter | required production capability |
| GPUDirect Storage / direct storage-to-GPU paths | hardware-specific execution/storage profile | `V2L46.6`, later `V2L49` comparison | only on compatible systems and only if I/O is a measured bottleneck | conditional hardware profile |
| custom Triton/CUDA fused kernels | owning adapter private implementation | no dedicated roadmap item | only after profiling isolates a durable hotspot and mature backends are insufficient | last-mile optional optimization |

## Roadmap sequencing

### V2M0 — measurement vocabulary first

`V2L4.4` owns the benchmark record vocabulary. It should be able to retain or reference performance evidence such as wall-clock stage timings, cold versus steady-state behavior, CPU/GPU utilization, peak RAM/VRAM, bytes transferred/read/written and trace artifacts when available.

`V2L4.4` must **not** integrate Nsight, NVTX or another profiler. It defines a solver-independent record that later workloads can populate.

### V2M1 — canonical media artifacts before accelerated media backends

`V2L6.6` owns canonical reusable decoded-image and multiresolution-pyramid artifacts. Their identity must include the material decode/preprocess semantics needed for reproducibility, not the accidental filename or one vendor backend.

A simple/reference path is sufficient to close the lot. GPU decoding is evaluated later as an execution profile only when it can produce contract-equivalent observations and profiling shows useful end-to-end benefit.

### V2M2 — first representative learned GPU execution benchmark

`V2L13` is the first natural home for an end-to-end learned execution-profile benchmark because a real feed-forward geometry adapter, quality metrics and controlled reference already exist by then.

`V2L13.6` should compare the validated reference execution with only relevant candidate profiles. Candidate examples may include eager versus compiled execution, mixed precision and TensorRT/Torch-TensorRT-class paths. It also owns stage-level profiling evidence for decode/preprocess, transfer, model execution, postprocess and artifact writeback.

The lot review moves to `V2L13.6`; `V2L13.5` remains the geometry quality comparison rather than becoming a mixed geometry/performance review item.

### V2M3 — appearance quality in motion and target-specific runtime efficiency

`V2L18.4` must measure more than still-image PSNR/SSIM/LPIPS. Where the selected representation is susceptible, appearance evaluation should include camera-path motion consistency, scale/zoom consistency, flicker/shimmering, popping and relevant geometry disagreement.

`V2L18.6` compares quality, resource efficiency and relevant execution profiles without hard-coding one rasterizer.

`V2L21.3` owns target-specific appearance packaging/compression. Compact splat formats, vector/codebook quantization, entropy models and progressive bitstreams are candidate implementation families. A single work item must not integrate several independently failing heavy formats; activation must split if necessary.

`V2L21.4` owns the scene-level spatial hierarchy and error/resource-driven LOD manifest. Distance-only LOD remains a baseline, while hierarchical, multiscale and progressive-compressed representations are candidate implementations behind the same runtime responsibility.

`V2L22.6` exercises these choices on actual camera motion and scale changes as well as load time, time-to-first-frame, FPS/frame time and memory.

### V2M6 — large-collection retrieval backend selection

`V2L37.1` defines scalable ANN/quantized retrieval-index semantics without freezing a CPU or GPU library.

`V2L37.5` compares relevant backends using candidate recall, index-build time, query latency/throughput, RAM, VRAM and index size. GPU ANN is not required for small collections and is not a product semantic.

### V2M7 — storage pressure, locality and hardware-specific acceleration

`V2L46.3` extends empirical resource/execution estimation to material locality when relevant: device class, transfer path, storage locality, NUMA placement, peer connectivity and measured transfer-bandwidth class. A single-GPU workstation may resolve these dimensions trivially.

`V2L46.5` owns materialization residency/tiering/eviction. Logical artifact identity and provenance survive eviction of recomputable bytes. Retention policy should be driven by recoverability, recomputation cost, downstream value, size, access pattern and user/policy retention requirements rather than blind LRU alone.

`V2L46.6` performs the final resource/locality/throughput/quality benchmark. Hardware-specific paths such as GPUDirect Storage belong here only when the target hardware is compatible and representative traces show I/O is significant.

`V2L49` later turns execution-profile comparisons into a continuous promotion mechanism. A faster profile cannot become a default if it violates quality, provenance, reproducibility, licensing or target-device constraints.

## Current candidate examples

These names are **examples to refresh at activation**, not frozen winners.

### GPU still-image decode

- NVIDIA nvImageCodec / nvJPEG-class accelerated image decode on compatible NVIDIA systems;
- CPU/reference image libraries remain required for portability and comparison;
- future non-NVIDIA accelerated backends can enter through the same execution-profile boundary.

### Learned execution

- PyTorch eager/reference execution;
- `torch.compile` / CUDA Graphs where applicable;
- TensorRT / Torch-TensorRT-class ahead-of-time optimized execution for stable models/shapes;
- future compiler/runtime backends.

### Gaussian quality, compression and LOD

Candidate/reference families include:

- Mip-Splatting-class antialiasing / scale consistency;
- StopThePop-class view-consistent sorting/rasterization;
- PCGS-class progressive compression;
- HAC++-class contextual entropy coding and adaptive quantization;
- vector-quantized Gaussian compression research;
- LoD-of-Gaussians / hierarchical and multiscale Gaussian representations;
- SPZ/SOG/Streamed-SOG-class runtime formats where appropriate.

### Storage/locality

- conventional buffered or direct file I/O as portable references;
- memory mapping / zero-copy where safe and useful;
- GPUDirect Storage / cuFile-class paths on compatible systems when justified;
- topology-aware scheduling based on measured device/storage relationships.

## Profiling policy

Use representative end-to-end workloads. At minimum, a serious optimization evaluation should identify when applicable:

- total wall clock;
- time to first useful result;
- cold-start versus steady-state cost;
- stage timings;
- CPU utilization;
- GPU utilization;
- peak and steady RAM/VRAM;
- host-to-device and device-to-host transfer volume/time;
- storage bytes/read/write bandwidth;
- artifact/runtime size;
- output quality and failure-class changes.

Standardized trace annotations such as NVTX-class ranges and vendor/system profilers such as Nsight are implementation candidates, not core dependencies. The product requirement is retained causal evidence sufficient to explain why an optimization helped or hurt.

## Cache residency policy

Future residency policy must distinguish at least conceptually:

- immutable source/irreplaceable evidence;
- expensive high-quality derived artifacts;
- reusable intermediate artifacts;
- cheap recomputable previews/temporary data;
- hardware/compiler-specific caches.

Eviction removes eligible **materialization bytes**, not canonical artifact identity or provenance history. Re-materialization must still verify bytes against the artifact identity and current compatibility rules.

## Runtime quality in motion

A photorealistic runtime cannot be evaluated only on isolated held-out screenshots. Representative camera paths should include translation, rotation, zoom/focal-scale change and distance changes likely to expose aliasing, popping, shimmer, unstable sort order or LOD transitions.

Performance and visual consistency are measured together. An optimization that raises FPS by introducing visible temporal instability is not an automatic improvement.

## What must not be developed early

Do not pre-build:

- a mandatory NVIDIA-only media pipeline;
- a global TensorRT dependency before a stable learned adapter exists;
- GPUDirect Storage integration on a workload that is not I/O-bound;
- custom Triton/CUDA kernels without profiling evidence;
- several competing compression formats inside one run-sized work item;
- topology-aware complexity in the deterministic local scheduler baseline;
- automatic eviction before artifact materialization identity/verification and resource ownership are mature.

## Review checklist

For every future optimization PR, ask:

1. Does it live under the correct frozen responsibility?
2. Does it preserve the owning WRE contract and provenance class?
3. Is the technology an adapter/execution/runtime candidate rather than a core type?
4. Was the current candidate landscape refreshed at activation?
5. Is there a stable reference path?
6. Was the optimization motivated by representative profiling or a clear runtime target?
7. Are cold-start, steady-state and end-to-end effects distinguished where relevant?
8. Are quality and failure regressions measured alongside latency/resources?
9. Does cache cleanup preserve logical artifact history?
10. Is the work still one-run sized, with at most one major external integration decision?
11. If hardware-specific, is there an explicit compatibility/fallback story?
12. Can a future stronger implementation replace it without rewriting core WRE semantics?

## Architecture conclusion

All reviewed optimization families fit the frozen WRE v2 architecture. They require better measurement, execution profiles, runtime compilation and resource policy—not a new architecture layer.

The intended development order is therefore:

```text
benchmark vocabulary
  -> canonical reusable media artifacts
  -> representative learned GPU profiling/execution profiles
  -> appearance motion/scale metrics
  -> runtime compression + hierarchical LOD
  -> large-collection ANN backend comparison
  -> resource locality + residency/eviction
  -> continuous execution-profile promotion
  -> custom kernels only for measured remaining hotspots
```

This keeps WRE aggressively optimizable while preserving the central v2 rule: freeze product semantics; continuously replace and improve the specialists and execution strategies.
