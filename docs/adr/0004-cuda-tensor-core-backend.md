# ADR 0004 — Optional CUDA Tensor-Core backend

- **Status:** Accepted (foundation proven; interpolator integration pending a model)
- **Date:** 2026-07-16
- **Deciders:** project owner (explicit direction) + implementation
- **Relates to:** `docs/analysis/cuda-vs-vulkan.md` §C/§D; ADR 0002 §8 (modular interpolator).

## Context

The CUDA-vs-Vulkan analysis recommended **`VK_KHR_cooperative_matrix`** as the default
Tensor-Core path and confined **CUDA/TensorRT** to a pluggable interpolator back-end used
only above a model-size threshold. The project owner then explicitly directed:
*"build in the usage of Tensor Cores via CUDA."*

This ADR reconciles that direction with the analysis: we build the CUDA Tensor-Core path
**now**, but as an **optional backend behind the modular-interpolator boundary**, so the
Vulkan core stays vendor-neutral and CUDA-free by default — exactly where §D said CUDA
belongs. The difference from the analysis is *timing* (build the proven foundation now,
at the owner's request) not *placement* (still behind the boundary, still optional).

## Decision

1. **Add an optional CUDA Tensor-Core backend**, gated by the CMake option
   `NVOFG_ENABLE_CUDA` (**OFF by default**). When off, nothing about the build, the C
   ABI, or the Vulkan-only core changes; there is no CUDA dependency.
2. It lives **behind the `NVOFG_INTERP_CNN` interpolator boundary** (ADR 0002 §8). The
   core pipeline (prep → flow → refine) and the classical `WarpInterpolator` never touch
   CUDA. A future `CudaTensorInterpolator` is *one* pluggable implementation.
3. The integration contract is **CUDA↔Vulkan interop**, proven on the reference GPU:
   - **External memory** (`VK_KHR_external_memory_fd`, `OPAQUE_FD`): Vulkan-allocated
     pipeline buffers are imported into CUDA (`cudaImportExternalMemory` →
     `cudaExternalMemoryGetMappedBuffer`) — zero-copy, no host round-trip.
   - **External timeline semaphore** (`VK_KHR_external_semaphore_fd`, `OPAQUE_FD`):
     Vulkan's timeline semaphore is imported into CUDA
     (`cudaImportExternalSemaphore`), so the existing §4 sync model extends across the
     API boundary. Per-frame: **Vulkan produces → signals T=n → CUDA waits n, runs the
     Tensor-Core kernel on the shared memory, signals n+1 → Vulkan consumes at n+1.**
   - **Tensor Cores** via CUDA **WMMA** (`nvcuda::wmma`, fp16×fp16→fp32), the compute
     primitive a learned interpolator's conv/GEMM layers map onto.

## Evidence (both PASS on RTX 5070 Laptop, sm_120, CUDA 13.3)

- `src/spike/cuda_tensor.cu` — WMMA 64×64×64 GEMM, max abs error vs CPU = 0.000
  → Tensor Cores execute correctly via CUDA.
- `src/spike/cuda_vk_interop.cu` — Vulkan-allocated exportable buffer + timeline
  semaphore imported into CUDA; Vulkan signals 1 → CUDA WMMA GEMM on the **Vulkan**
  memory → signals 2 → Vulkan waits 2; result exact (0.000).
  → the full production interop pattern works end-to-end.

Built via `cmake -DNVOFG_ENABLE_CUDA=ON` (targets `nvofg_cuda_tensor`,
`nvofg_cuda_vk_interop`).

## Consequences

### Positive
- The hardest, riskiest part of a CUDA interpolator (zero-copy interop + cross-API sync +
  Tensor-Core execution) is **de-risked and reproducible** — the M0-equivalent proof for
  this path.
- The Vulkan-only, vendor-neutral core and default build are **unchanged**; CUDA is
  strictly opt-in and contained.
- `VK_KHR_cooperative_matrix` remains available as the *in-band, portable* Tensor-Core
  path for a Vulkan-native learned interpolator; CUDA is the *high-tuning, NVIDIA-only*
  alternative. Both are legitimate implementations of the same interpolator interface;
  choice is per-build/runtime, not baked into the core.

### Negative / open
- **NVIDIA-only** when enabled, and adds a CUDA toolkit build dependency for that build —
  accepted because it is optional and isolated.
- **Interop tax** per frame (semaphore import waits + stream sync), as the analysis noted;
  justified only for a model large enough to amortize it — the same size-threshold rule
  from §D still governs *whether* to use this backend at runtime.
- **No trained model yet.** This ADR establishes the *infrastructure and proven path*, not
  a learned network. Wiring a real `CudaTensorInterpolator` needs (a) a model + weights and
  (b) the conv/GEMM kernels; both are future work built on this foundation.

## Alternatives considered
- **Cooperative-matrix only (analysis default).** Still the recommended path for a *small*
  in-band learned interpolator and for non-NVIDIA GPUs; retained. CUDA is added alongside,
  not instead, for the large-model / maximum-tuning case.
- **TensorRT instead of hand-written CUDA.** Better for a *whole* large ONNX graph
  (analysis §D); can be a further backend behind the same interpolator boundary reusing
  this exact interop. Not needed to prove the path.
