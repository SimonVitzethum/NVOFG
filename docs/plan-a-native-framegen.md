# Plan 1 — Own native Frame Generation (the shippable path)

The product path. Loads **no NVIDIA binary**, uses **our/open weights**, is legally clean to ship
(no §69e limits, no Authenticode bypass, no NVIDIA clearance needed). Path B stays private interop
research and, crucially, serves here as an **offline quality reference** to train/ablate against.

## The head start (why "bald ready" is realistic)

nvofg **already has a working, legally-clean native frame generator**: `VK_NV_optical_flow` (the OFA)
produces motion vectors and the Slang warp shader (`src/shaders/warp.slang`) synthesises the
intermediate frame. That is Phase 0 — done. "Ready soon" is **productionising that classical path**;
the learned model is a quality upgrade layered on top, not a prerequisite for a first shippable
release.

## Phased plan

### Phase 0 — classical native FG  ✅ have it
OFA optical flow + Slang warp interpolation. Engine-agnostic, native, ships legally.

### Phase 1 — productionise the classical path  → **the near-term MVP (weeks, no server needed)**
Turn the working warp into a shippable feature. No ML, no external compute — pure engineering on
what already runs:
1. **Quality passes** — the visible-artefact list, in priority order (image quality > stability):
   - Disocclusion / occlusion handling (holes at object edges) — bidirectional warp + hole-fill.
   - Motion-vector cleanup (OFA flow → confidence mask, reject/blend low-confidence regions).
   - UI/HUD exclusion (don't warp overlays — a mask input or a late composite).
   - Ghosting/edge fringing reduction; clamp to plausible colour neighbourhood.
2. **Pacing** — `VK_NV_low_latency2` (Reflex) present metering so the generated frame is scheduled
   between real frames (without pacing the extra frame adds latency instead of smoothness). This is
   ordinary native Vulkan and **overlaps directly with the Path B present stage** — do it once, use
   it for both.
3. **API / integration** — clean RenderFX hook (prev+curr color, optional MVs/depth from the app;
   fall back to OFA-only when the app provides nothing), on/off, 1 generated frame first.
4. **Validation harness** — a fixed set of rendered sequences + metrics (PSNR/SSIM/LPIPS/VMAF vs the
   true intermediate frame, plus a warp-consistency metric) so quality is measured, not eyeballed.
   This harness is reused unchanged in Phase 2/3.

**Exit of Phase 1 = a shippable native FG.** Not DLSS-G quality yet, but real, legal, and running.

### Phase 2 — the learned interpolator/extrapolator  → **needs the 5080 server (~100–200 h)**
Layer a small learned model on top of the warp (predict a residual/refinement, not the whole frame
— cheaper, more stable, and the OFA warp already gives a strong prior):
- **Architecture:** compact CNN (or tiny transformer) over `VK_KHR_cooperative_matrix` (Tensor
  cores), **extrapolation-first** (ExtraNet / ExtraSS / GFFE lineage — extrapolate the next frame from
  history, lower latency than interpolation which must hold a frame). ~0.4–1M params, budgeted to fit
  the per-frame time budget at target resolutions.
- **Inputs:** warped candidate (from Phase 0), OFA flow + confidence, prev/curr color, optional
  depth + app motion vectors. **Output:** residual correction + a blend/hole mask.
- **Data:** rendered sequences with ground-truth intermediate frames + MVs + depth (our own renders;
  optionally augment with open datasets). **Path B provides an additional reference target** — a
  legitimate interoperability use (compare our output to NVIDIA's on identical inputs, offline).
- **Loss:** perceptual (LPIPS) + L1 + warp/temporal-consistency + edge/disocclusion-weighted terms.
- **Training:** the 5080 server, ~100–200 h; iterate architecture against the Phase-1 harness.
- **Inference:** coopmat path in RenderFX; runs as a refinement pass after the warp.

### Phase 3 — quality parity push
Use Path B's output (private, offline) as the quality target for ablations; close the gap on the
hardest cases (fast motion, thin structures, transparency, UI). Scale to 2–3 generated frames if the
pacing + quality budget allows.

## Sequencing vs Path B (do these now, in this order)

1. **Phase 1.2 pacing (`VK_NV_low_latency2`) first** — it is needed by *both* Plan 1 and Path B, is
   pure native Vulkan, and unblocks a real end-to-end FG loop immediately.
2. **Phase 1.1 quality passes** — turns the warp into something shippable; each pass is measurable on
   the harness.
3. **Phase 1.4 harness** in parallel — it gates everything downstream and is cheap to stand up.
4. When the **5080 server** lands → Phase 2 training, with Path B as the offline reference target.

Path B's functional grind (Init → CreateFeature → EvaluateFeature) continues in parallel *as research
only*; its value to the product is (a) the pacing work shared with Phase 1.2 and (b) the reference
frames for Phase 2/3 training — not a shipped dependency.

## Definition of "ready"

- **MVP-ready (Phase 1):** native FG on by a flag, paced, artefact-managed on the common cases,
  measured on the harness, integrated in RenderFX. Ships without any NVIDIA code and without a
  trained model.
- **Quality-ready (Phase 2+):** learned refinement closes most of the gap to DLSS-G on the harness
  metrics.
