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
  **Identity test is a BIT-EXACT assertion:** `output == classical_warp` exactly, not "looks the
  same" — if the fp16/coopmat path or a colorspace rounding shifts the passthrough off the pure warp,
  the "identity" isn't one and every later quality delta is measured against a shifted baseline. Pin
  the tolerance to 0 (or the warp's own fp16 output, computed the same way).
- **A2. Data-capture harness — the real bottleneck; do it BEFORE A3.** At ~1M params neither VRAM nor
  compute limits us — **data does.** Offline tool: render/capture at **2× target fps** so every other
  frame is GT, dumping per-frame color + MV + depth + UI/reactive masks → triplets. **A2 ships with
  its own correctness gate FIRST: a sub-pixel alignment round-trip** — warp the captured N-1/N+1 to
  phase 0.5 with the *known* flow and confirm it lands on the capture-GT frame N to sub-pixel, with
  identical jitter state, MV convention, and HUD exclusion. A silent sub-pixel offset between
  capture-GT and model-input poisons the loss (the net learns to compensate the offset, not to
  interpolate) and only shows up as bad generalisation. **The alignment test must pass before
  generating tens of thousands of triplets** — else they are tens of thousands of subtly-wrong ones.
  (Same discipline as catching a silent error class before the expensive run.) Coordinate the format
  with the RMC/Minecraft consumer (other agent). Aim: tens of thousands across pans, fast entities,
  transparency.
- **A3. Model + training pipeline (PyTorch) — BOTH modes at tiny scale.** Implement §21.4's SoftSplat
  forward-warp + gated-conv fusion net and §21.6's losses (Charbonnier + LPIPS + census/warp + light
  GAN + temporal; UI/reactive masked out, disocclusion up-weighted). **Phase-conditioned** (random
  t∈(0,1) per sample) so one net serves any multiplier (see model doc). **Overfit BOTH
  interpolation AND extrapolation at tiny scale** and record the quality gap between them *before*
  committing 10²–10³ GPU-h — extrapolation is the *harder* learning task (below), so the first server
  run must not be the moment you discover the quality is inverted from the plan. Vimeo/X4K bootstrap +
  rendered fine-tune loaders.
- **A4. Validation harness — stratified by difficulty regime, both modes (gates every quality
  claim).** Golden metrics vs GT: PSNR/SSIM/**LPIPS**/VMAF + temporal-stability. **Stratify the clip
  set into `easy` / `disocclusion-heavy` / `shading-change` regimes and report per-regime**, because
  our classical warp is already best-in-class: on easy clips it is near-perfect and the learned model
  can only tie, so an *averaged* metric is warp-dominated and hides the win (and risks discarding a
  real gain). The learned model's value lives *only* in the hard regimes — measure it there. Report
  interpolation AND extrapolation separately. Reused unchanged by all of Track 2.
- **A5. Weight export/import.** Trained model → the fp16 layout the `CudaTensorInterpolator` /
  coopmat path loads (WMMA/coopmat tile layout, versioned header).

## Hard invariants (protect §21's shippability)

- **Path B is a READ-ONLY metric in A4, never a training tensor.** The Path B reference output may be
  *measured against* (a benchmark target) but must **never** reach A3/B1/B2 as a target, distillation
  signal, or pseudo-GT — no Path-B tensor enters the training graph, ever. This is the line that keeps
  §21 legally shippable; A4/B5 is exactly where "I measure against DLSS" could silently become "I
  train on DLSS" if someone later takes a shortcut. Enforce it as a repo firewall.
- **Latency choice ≠ quality choice.** Extrapolation is chosen for *latency* (no held-back frame),
  **despite** being the *harder, lower-quality* learning task — it must guess newly-revealed
  disocclusion, direction changes, acceleration, fade-ins, for which no signal exists in past-only
  inputs. Interpolation has both surrounding frames (the OFA bidir flow nearly solves it; the rest is
  disocclusion). Treat mode as a deliberate trade-off axis, not "primary = the good one."

## Track 2 — training (on the RTX 5080 `ki-pc-fisch-101`, ~10²–10³ GPU-h per §21.7)

- **B1.** Vimeo-90K/X4K pretraining of the RGB synthesis net.
- **B2.** Rendered-data fine-tune with real MV/depth/masks (the A2 captures).
- **B3.** Train **both modes** from the shared backbone: **extrapolation = the low-latency primary**
  (past-only → no held-back frame, no flip-metering — §21.2), **interpolation = the quality path**
  (bidirectional OFA flow, cleaner disocclusion, for latency-insensitive/offline use). The plan ships
  extrapolation primary for latency, *knowing* it trades quality — hence both are measured (A4) from
  the start.
- **B4.** fp16 + `VK_KHR_cooperative_matrix` (Tensor Cores) to hit **<2–3 ms @1080p**; optional
  TensorRT backend above the size threshold (ADR 0004).
- **B5.** Quality iteration against the A4 **per-regime** metrics + the Path B reference (read-only);
  close the disocclusion / ghosting / shading-correction cases where classical warp fails.

## Sequencing — the immediate next steps (all Track 1, no server needed)

1. **A1 scaffold first** — identity model through `NVOFG_INTERP_CNN`, bit-exact-verified. Highest
   leverage: makes everything downstream drop-in.
2. **A2 with its alignment test** — the data bottleneck; the alignment gate must pass before mass
   capture, or the server run trains on subtly-misaligned data.
3. **A4 harness, stratified + both modes** — nothing about quality is claimed without it, and it must
   measure the right thing (per-regime, per-mode).
4. **A3 model + pipeline, both modes tiny-scale** — validated (and the extrap-vs-interp gap known)
   before the expensive run.
5. When the server run starts → Track 2.

**Note on pacing.** Extrapolation-primary holds back no frame → `VK_NV_low_latency2` pacing is needed
**only** for the interpolation mode, so it is later/lower-priority, not the critical path. (Supersedes
the earlier "pacing first" note.)

## Definition of "ready"

- **Track-1-ready:** `NVOFG_INTERP_CNN` selectable end-to-end with a **bit-exact** identity model,
  headless-clean; A2 alignment test green; stratified both-mode harness live; pipeline validated at
  tiny scale with the extrap-vs-interp gap measured. Ships nothing user-visible yet — makes the model
  turnkey and the server run honest.
- **v1-ready:** trained model, <2–3 ms @1080p, **wins in the hard regimes** (disocclusion / ghosting /
  shading) on the per-regime harness and at least ties on easy — measured for both modes, no NVIDIA
  code, no clearance needed. The shippable product.
