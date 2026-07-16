# CUDA vs Vulkan for nvofg — a grounded technical analysis

This answers the four questions posed for the project (A: overall, B: per-stage,
C: Tensor Cores, D: CNN back-ends). It is grounded in what we verified on the reference
machine (RTX 5070 Laptop, driver 610.43.03): `VK_NV_optical_flow`,
`VK_KHR_cooperative_matrix` (rev 2), `VK_NV_cooperative_matrix2`, and CUDA 13.3 all
present.

**Bottom line up front:** nvofg's core should be **Vulkan-only**. The single feature
that historically forced CUDA into a Vulkan pipeline — access to the OFA — is now a
Vulkan extension (ADR 0001). Every remaining stage is either trivially a Vulkan compute
shader or (for the future learned interpolator) reachable via Vulkan cooperative-matrix.
CUDA/TensorRT become worth reconsidering **only** if a future model is large enough that
TensorRT's kernel autotuning beats hand-written Vulkan cooperative-matrix kernels by a
margin that justifies a CUDA interop boundary — a decision we defer, with a concrete
trigger, to question D.

---

## A) Vulkan-only vs CUDA overall

The relevant axes, each judged for a **real-time, per-frame, in-engine** library whose
inputs and outputs are already `VkImage`s the app owns.

### Memory transfers
- **Vulkan-only:** zero transfers. The OFA reads the app's `VkImageView`s and writes a
  flow `VkImage`; refine/warp/composite are compute passes over those same images. No
  copy ever leaves the device, and nothing crosses an API boundary.
- **CUDA:** to touch those same images CUDA must **import** them via
  `VK_KHR_external_memory_fd` → `cudaExternalMemory`. That is not a copy, but it forces
  every shared resource to be allocated as exportable, adds per-resource import
  bookkeeping, and constrains tiling/layout. Any resource *not* pre-exported needs a real
  copy. Net: CUDA can reach zero-copy too, but only by paying an interop tax on every
  shared surface.
- **Verdict:** Vulkan-only wins decisively — it is zero-copy *by construction*, CUDA only
  *by careful arrangement*.

### Synchronization
- **Vulkan-only:** one primitive — timeline semaphores — orders OFA (queue family 5) →
  compute → present. Barriers are the same `VkMemoryBarrier2` the app already uses.
- **CUDA:** must import each Vulkan timeline semaphore as a `cudaExternalSemaphore` and
  interleave `cudaWaitExternalSemaphores` / `cudaSignalExternalSemaphores` with CUDA
  stream ordering. This is a *second* scheduler (CUDA streams) bolted onto Vulkan queues;
  every hand-off is a place to deadlock or stall, and it is exactly the class of bug that
  bit RustMineClient's present pacer before (design.md §9).
- **Verdict:** Vulkan-only wins — one scheduler, one primitive.

### Latency
- FG's whole value proposition is latency-sensitive: it *holds* frame N. Each API
  boundary crossing (semaphore import wait, stream sync) adds scheduling jitter measured
  in tens of µs and, worse, variance. A single-API pipeline has no boundary to cross.
- **Verdict:** Vulkan-only wins on both mean and (more importantly) tail latency.

### Maintainability
- **Vulkan-only:** one build system, one shading language (GLSL/Slang → SPIR-V), one set
  of debug tools (RenderDoc, Nsight Graphics, validation layers). The app integrates via
  the C ABI and never sees CUDA.
- **CUDA:** adds nvcc, a CUDA toolkit build dependency for *every* downstream integrator,
  a second debugger (Nsight Compute/Systems), and the external-memory/-semaphore interop
  layer as permanent surface area. A "forbids unsafe / minimal-deps" engine (RMC's stated
  posture) would reject a hard CUDA dependency.
- **Verdict:** Vulkan-only wins strongly.

### Portability
- **Vulkan-only:** the compute stages (prep/refine/warp/composite) run on **any** Vulkan
  GPU. Only the OFA execute is NVIDIA-specific, and that is already isolated behind a
  tier system (design.md §8: Tier B shader flow for AMD/Intel). The library stays
  vendor-neutral except for the one hardware call.
- **CUDA:** NVIDIA-only, full stop. Adopting CUDA anywhere in the core forecloses the
  Tier-B vendor-neutral story permanently.
- **Verdict:** Vulkan-only wins — and this one is strategic, not just tactical.

### Performance
- For the OFA itself: identical — same silicon, and via the extension the *only*
  difference from the SDK is the submit path (a recorded command vs a driver enqueue),
  which is if anything leaner.
- For prep/refine/warp: these are memory-bandwidth-bound image ops. A competent Vulkan
  compute shader saturates bandwidth just as a CUDA kernel does; there is no compute
  headroom for CUDA to unlock. CUDA's raw advantages (warp intrinsics, `__shared__`
  tuning, cooperative groups) matter for compute-bound, reduction-heavy, or
  irregular-control-flow kernels — none of which describe a warp/blend pass.
- The *one* place CUDA's ecosystem has a genuine, measurable edge is **large GEMM-heavy
  neural nets via TensorRT** — see questions C and D.
- **Verdict:** parity for everything nvofg does today; CUDA's edge is confined to a
  future, large-model scenario.

### Recommendation (A)
**Vulkan-only for the entire core.** It is zero-copy by construction, single-scheduler,
lower tail-latency, far more maintainable, and keeps the vendor-neutral fallback alive —
at no performance cost for any stage nvofg currently has. CUDA is reconsidered only under
the narrow, explicit trigger in question D, and even then only *behind* the modular
interpolator boundary so the core stays Vulkan-only.

---

## B) Which stages would actually benefit from CUDA?

Evaluated per stage, not pauschal. "Benefit" = a real, measurable win that outweighs the
interop tax from question A.

| Stage | Nature | Vulkan compute adequate? | Would CUDA help? |
|---|---|---|---|
| **1. Format prep** (color→R8/NV12, resample to grid) | bandwidth-bound image op | Yes, fully | No. Nothing to gain; pure copy/convert. |
| **2. Optical flow** (OFA execute) | fixed-function silicon | **Extension = same HW** | No. CUDA path drives the *same* OFA; the extension reaches it without leaving Vulkan. |
| **3. Flow refine** (upsample, dilate/median, cost→confidence, fwd/bwd reconcile, MV fuse) | stencil/local filters over images | Yes, fully | Marginal at best. Median/dilate are small local kernels; CUDA `__shared__` tiling could shave a little, but it is bandwidth-bound and not hot enough to justify interop. |
| **4a. Warp** (forward+backward, occlusion-aware) | gather/scatter over images | Yes | No. Scatter warp needs atomics either way; Vulkan image atomics / `imageStore` suffice. |
| **4b. Hole-fill / inpaint** (edge-aware, push-pull pyramid) | multi-res image pyramid | Yes | Marginal. Push-pull is a mip pyramid — both APIs express it equally; no CUDA-only trick. |
| **4c. Blend / anti-ghost clamp** | per-pixel weighted combine | Yes | No. |
| **5. UI composite** | graphics/blit | Yes (graphics) | No. |
| **6. (future) CNN interpolator** | conv + small GEMM | Yes, via cooperative-matrix | **Possibly** — see C/D. TensorRT autotuning can beat hand kernels for larger nets. |
| **7. (future) Transformer interpolator** | attention = large GEMM/softmax | cooperative-matrix feasible | **Most likely CUDA/TensorRT-favoring** of all stages, if a model this heavy is ever used real-time (doubtful at FG latency budgets). |

**Reading of the table:** stages 1–5 — i.e. *all of v1 and the entire classical
pipeline* — see **no** benefit from CUDA. Only the hypothetical learned interpolators
(6, 7) could, and only above a size threshold (question D). Because those live behind the
modular interpolator interface, a CUDA/TensorRT back-end could be added later as *one
pluggable implementation* without touching the Vulkan core.

---

## C) Tensor Cores

### Can Vulkan use Tensor Cores directly?
**Yes.** Confirmed present on the reference GPU:
- `VK_KHR_cooperative_matrix` (rev 2) — the cross-vendor, standardized path;
- `VK_NV_cooperative_matrix` and `VK_NV_cooperative_matrix2` — NVIDIA extensions, the
  latter adding flexible dimensions and reduced setup overhead.

Cooperative-matrix types in a SPIR-V compute shader (`OpTypeCooperativeMatrixKHR`,
`OpCooperativeMatrixMulAddKHR`) compile to Tensor-Core MMA instructions (HMMA/IMMA) on
NVIDIA hardware. So Vulkan compute reaches the Tensor Cores **without CUDA**, in-band with
the rest of the pipeline (same command buffer, same semaphores, zero interop). Supported
component types and tile shapes are enumerated at runtime via
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` (fp16/bf16/int8 accumulate to
fp16/fp32/int32, typical 16×16×16 tiles).

### If one wanted more than cooperative-matrix gives
Options, ranked:

| Option | Perf | Portability | Future-proofing | Notes |
|---|---|---|---|---|
| **`VK_KHR_cooperative_matrix`** | High — direct MMA, hand-schedulable | **Best** — cross-vendor KHR (NVIDIA, and increasingly others) | **Best** — standardized, tracks new hardware via runtime property query | In-band, zero interop. **Recommended default.** |
| `VK_NV_cooperative_matrix2` | Highest hand-written — flexible dims, less overhead | NVIDIA-only | Good on NVIDIA | Use as an *optimization* behind a runtime check, not the baseline. |
| **CUDA (WMMA / CUTLASS)** | Highest with effort — full CUTLASS tuning | NVIDIA-only | Good | Pays the full interop tax (A). Only if a kernel is GEMM-dominated *and* cooperative-matrix leaves measurable performance on the table. |
| **TensorRT** | Highest for whole large nets — autotuned, fused | NVIDIA-only | Good; opaque engine blobs are version-sensitive | Best when the workload is a *whole* sizeable network, not a kernel. See D. |

### Recommendation (C)
Use **`VK_KHR_cooperative_matrix` as the default Tensor-Core path**, with
`VK_NV_cooperative_matrix2` as an optional NVIDIA fast-path selected at runtime. This
keeps Tensor-Core acceleration **in-band and vendor-portable**, consistent with the
Vulkan-only core (A). Reserve CUDA/TensorRT for the specific large-model case (D), behind
the interpolator interface. Rationale weighting follows the project's priority order
(quality > stability > API > performance): cooperative-matrix keeps stability and API
cleanliness while still delivering Tensor-Core throughput.

---

## D) CNN / learned-interpolator back-ends

If nvofg ever adds a learned interpolator (the modular design explicitly allows
`CNNInterpolator` / `TransformerInterpolator`), the choice is Vulkan compute
(cooperative-matrix) vs CUDA vs TensorRT.

### The three options
- **Vulkan compute + cooperative-matrix.** In-band, zero interop, vendor-portable,
  single toolchain. You hand-write (or generate) conv/GEMM kernels using
  cooperative-matrix. Best for **small** nets where kernel launch/interop overhead would
  dominate and where hand-fusion is tractable.
- **CUDA (cuDNN / CUTLASS).** Mature conv/GEMM primitives, but you own the graph
  execution and pay the external-memory/-semaphore interop tax per frame.
- **TensorRT.** Ingests an ONNX graph, autotunes and fuses kernels for the specific GPU,
  and typically **wins on end-to-end latency for whole networks above a few MMACs** —
  because it optimizes across layers (fusion, precision, kernel selection) in ways
  hand-written kernels rarely match. Cost: a heavyweight NVIDIA dependency, per-GPU engine
  build/caching, opaque version-sensitive blobs, and the full interop tax.

### When does CUDA/TensorRT actually pay off?
The interop tax (A) is a roughly fixed per-frame cost (semaphore import waits + stream
sync + any layout constraints), on the order of tens of µs plus variance. TensorRT's
advantage grows with arithmetic intensity. So the break-even is where **autotuned
whole-graph execution saves more than the interop tax costs**. Practical rule of thumb:

- **Tiny nets** (a few conv layers, "reactive/occlusion refinement" scale, ≲ ~1–2 GFLOP
  per frame): **Vulkan cooperative-matrix.** Interop tax would dominate; keep it in-band.
- **Medium nets** (a real interpolation CNN, e.g. a small U-Net, ~2–20 GFLOP/frame):
  **measure.** cooperative-matrix is likely competitive and far cleaner; adopt TensorRT
  only if profiling shows it clears the interop tax by a comfortable margin (say ≥ 1.3×
  end-to-end).
- **Large nets** (heavy CNN or any transformer at these resolutions, ≳ 20 GFLOP/frame):
  **TensorRT** is the pragmatic choice *if* it can even fit the frame budget — at 120 fps
  the whole FG budget is ~8 ms and the interpolator gets a fraction of that, which by
  itself argues against large models for real-time FG.

### Recommendation (D)
- **v1 and the classical warp interpolator:** no neural net, so **Vulkan-only**, question
  closed.
- **First learned interpolator, if any:** start with **Vulkan cooperative-matrix** — it
  keeps the whole pipeline single-API and vendor-portable, which matters more than a
  possible small speedup (priority order again).
- **Escalate to TensorRT only on evidence:** a specific model, profiled on target
  hardware, where autotuned whole-graph execution beats the cooperative-matrix
  implementation by a clear margin *after* accounting for interop. Because the
  interpolator is a pluggable module, this can be a self-contained
  `TensorRTInterpolator` back-end that never contaminates the core.

---

## Consequences for the codebase
1. Core stays **Vulkan-only**; no CUDA in the build graph, no CUDA dependency for
   integrators.
2. The **modular interpolator** boundary (design req. #8) is also the CUDA/TensorRT
   containment boundary: any future NVIDIA-only learned back-end lives entirely inside one
   interpolator implementation and is selectable at runtime, leaving Tier-B portability
   intact for everyone else.
3. Tensor-Core work, when it comes, uses **`VK_KHR_cooperative_matrix`** first.
