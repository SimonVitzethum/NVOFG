# Plan A — native learned Frame Generation (execution plan)

The shippable path to close the quality gap to DLSS-G **natively and legally**: an own learned
interpolator/extrapolator that loads **no NVIDIA binary**, runs on-device, and drops into the
`NVOFG_INTERP_CNN` backend that already exists in the API. Path B stays private research and serves
only as an **offline quality reference**.

## What already exists (so this is execution, not green-field)

- **Classical native FG is done and best-in-class** (design.md, RenderFX ROADMAP): OFA
  (`VK_NV_optical_flow`) flow + the `prep → OFA → refine → warp → composite` pipeline, with UI /
  reactive / material masks, bidirectional/occlusion, HDR, quality tiers, debug views, and a pacer.
  It ships today; **the learned model layers on top of it, it is not a prerequisite.**
- **The design is done — design.md §21**: architecture (SoftSplat forward-warp + gated-conv fusion,
  ~0.26–1.0M params, ExtraNet/ExtraSS/GFFE lineage), the extrapolation-first latency decision, the
  10–50× size lever (we supply MV+depth+OFA flow, so the net doesn't learn flow), the data plan,
  losses, and the <2–3 ms @1080p budget. **This doc executes §21; it does not re-design it.**
- **The runtime boundary is built and proven — ADR 0004**: the `NVOFG_INTERP_CNN` interpolator
  boundary; an optional CUDA Tensor-Core backend gated by `NVOFG_ENABLE_CUDA` (OFF by default);
  CUDA↔Vulkan interop (external memory + timeline semaphores) and Tensor-Core WMMA both proven by the
  `src/spike/cuda_tensor.cu` and `src/spike/cuda_vk_interop.cu` spikes. `VK_KHR_cooperative_matrix`
  is the vendor-neutral default path.
- **The API hook exists**: `NvofgInterpolator::NVOFG_INTERP_CNN` is reserved in `include/nvofg.h`;
  selecting it must Just Work once a backend is wired.

**The gap = (1) the trained model, (2) its on-device inference backend, (3) the data + training +
validation pipeline.** Everything else is in place.

## Track 1 — buildable NOW (no training server): make trained weights drop-in

Each milestone builds, passes a headless test, is Vulkan-validation-clean, and is committed
separately (RenderFX ROADMAP discipline). Priority: correctness > image quality > perf.

- **A1. `CudaTensorInterpolator` scaffold** behind `NVOFG_INTERP_CNN` / `NVOFG_ENABLE_CUDA`.
  Load a weights file, run the fusion-net forward pass (WMMA/coopmat) over the already-computed
  inputs (warped color(s), fwd/bwd flow, occlusion/confidence, depth, MV, disocclusion + UI/reactive
  masks), **residual-add the warped RGB**, write the registered output. **Start with an identity /
  passthrough "model"** (emits the classical warp) so the entire path — registration, interop, sync,
  output — is exercised and headless-verified (0 VUIDs) *before* any weights exist. Real weights then
  drop in with no plumbing changes. Mirror a **coopmat path** (vendor-neutral) next to the CUDA one.
- **A2. Data-capture harness.** Offline tool: render/capture sequences at **2× target fps** so every
  other frame is GT, dumping per-frame color + MV + depth + UI/reactive masks → training triplets.
  Coordinate the capture format with the RMC/Minecraft consumer (other agent). Target: tens of
  thousands of triplets across varied motion (pans, fast entities, transparency).
- **A3. Model + training pipeline (PyTorch).** Implement §21.4's SoftSplat forward-warp + gated-conv
  fusion net and §21.6's losses (Charbonnier + LPIPS + census/warp + light GAN + temporal; UI/reactive
  masked out, disocclusion up-weighted). Data loaders: Vimeo-90K/X4K bootstrap + rendered fine-tune.
  **Validate end-to-end at tiny scale** (overfit a few clips) to prove the pipeline before any big run.
- **A4. Validation harness (gates every quality claim).** Golden metrics vs the GT in-between frame:
  PSNR/SSIM/**LPIPS**/VMAF + a temporal-stability metric. Compares warp-only vs learned vs (privately)
  the Path B reference on a fixed clip set. Reused unchanged by all of Track 2.
- **A5. Weight export/import.** Trained model → the fp16 layout the `CudaTensorInterpolator` /
  coopmat path loads (WMMA/coopmat tile layout, versioned header).

## Track 2 — training (gated on the RTX 5070/5080 server, ~10²–10³ GPU-h per §21.7)

- **B1.** Vimeo-90K/X4K pretraining of the RGB synthesis net.
- **B2.** Rendered-data fine-tune with real MV/depth/masks (the A2 captures).
- **B3.** Ship **extrapolation as the primary mode** (predict next frame from past only → **no added
  latency**, no flip-metering dependence — §21.2); add the **interpolation mode** (bidirectional OFA
  flow, cleaner disocclusion) as the quality/offline fallback.
- **B4.** fp16 + `VK_KHR_cooperative_matrix` (Tensor Cores) to hit **<2–3 ms @1080p**; optional
  TensorRT backend above the size threshold (ADR 0004).
- **B5.** Quality iteration against the A4 harness + the Path B reference; close the disocclusion /
  ghosting / shading-correction cases where classical warp fails.

## Sequencing — the immediate next steps (all Track 1, no server needed)

1. **A1 scaffold first** — identity model through `NVOFG_INTERP_CNN`, headless-verified. This makes
   the whole rest drop-in and is the single highest-leverage build step.
2. **A4 harness in parallel** — nothing about quality is claimed without it.
3. **A3 model + training pipeline** — validated at tiny scale so the server run is turnkey.
4. **A2 data capture** — coordinate format with the RMC agent.
5. When the server lands → Track 2.

**Note on pacing.** Because the **primary path is extrapolation (no held-back frame → no added
latency)**, `VK_NV_low_latency2` pacing is **only** needed for the secondary *interpolation* mode —
so it is a later, lower-priority item, not the critical path. (This supersedes the earlier
"pacing first" note.)

## Definition of "ready"

- **Track-1-ready:** `NVOFG_INTERP_CNN` selectable end-to-end with an identity model, headless-clean;
  training pipeline validated at small scale; harness live. Ships nothing user-visible yet but makes
  the model turnkey.
- **v1-ready:** extrapolation model trained, <2–3 ms @1080p, **visibly better than classical warp**
  on the harness, no NVIDIA code, no clearance needed — the shippable product.
