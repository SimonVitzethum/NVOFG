# ADR 0002 — Frame-generation pipeline architecture

- **Status:** Accepted
- **Date:** 2026-07-16
- **Scope:** how the stages between "two color frames in" and "one interpolated frame
  out" compose — the modular interpolator, confidence weighting, bidirectional flow,
  and motion-vector / depth fusion (project requirements #1–#8).
- **Builds on:** ADR 0001 (OFA via `VK_NV_optical_flow`), `docs/hardware-capabilities.md`.

## Context

`design.md` §3 gives a five-box pipeline. The project brief adds hard requirements the
original sketch under-specifies: bidirectional flow (#1), *continuous* confidence from
the cost map rather than a threshold (#2), motion-vector fusion (#3), depth fusion (#4),
material IDs (#5), a reactive mask (#6), a mandatory UI mask (#7), and a **pluggable
interpolator** so classical warping can later be swapped for a learned model (#8).

The M0 probe established the hardware levers we build on: bidirectional flow, a cost map,
and **external hints are all supported**, output flow is `R16G16_SFIXED5_NV` (S10.5), and
the OFA runs on a dedicated queue family.

## Decision

### Overall shape

```
 registered: prevColor currColor [depth] [motion] [uiMask] [reactive] [matID]  → output
     │
 (compute, app queue)          (OFA, OF queue)            (compute, app queue)
     ▼                              ▼                            ▼
 ┌────────┐  luma+hint  ┌────────────────────┐  fwd/bwd  ┌──────────────┐  refined ┌──────────────┐
 │  PREP  │────────────▶│  OFA EXECUTE (bidir)│─flow+cost▶│    REFINE    │─────────▶│ INTERPOLATOR │
 └────────┘             └────────────────────┘           └──────────────┘   flow   │  (pluggable) │
                              ▲ hint = app MVs                 ▲ depth/MV/cost →     └──────────────┘
                                                               confidence & masks         │
                                                                                          ▼
                                                                                  ┌──────────────┐
                                                                                  │ UI COMPOSITE │
                                                                                  └──────────────┘
```

The **flow/refine machinery is fixed**; the **interpolator is an interface**. That
boundary (requirement #8) is also the CUDA/TensorRT containment boundary from the
CUDA-vs-Vulkan analysis: a future learned back-end is one `Interpolator` implementation
and never touches prep/execute/refine.

### 1. Two-queue execution & sync
- PREP, REFINE, INTERPOLATE, COMPOSITE are compute/graphics recorded into the **app's
  command buffer** on the app's queue.
- OFA EXECUTE is recorded onto the **optical-flow queue** and submitted by nvofg. A
  **timeline semaphore** orders app-queue-prep → OFA-execute → app-queue-refine. The
  flow/cost images use queue-family ownership transfers (or `CONCURRENT` on first cut).
- The app waits on the returned `NvofgFrameSync` before presenting. No `vkQueueWaitIdle`
  on the hot path (the M0 spike's `WaitIdle` is spike-only).

### 2. Bidirectional flow (req. #1)
- The session is created with `BOTH_DIRECTIONS` so one execute yields **forward**
  (N-1→N) and **backward** (N→N-1) flow. Cost is enabled (`ENABLE_COST`).
- REFINE runs a **forward–backward consistency check**: a pixel whose forward vector,
  followed by the backward vector at its destination, does not return near the origin is
  marked **occluded** (moving object edge) — the warp fills it from the opposite side.

### 3. Confidence, not threshold (req. #2)
- The OFA cost map is converted to a **continuous confidence** `c ∈ [0,1]`, not a binary
  reject. Chosen mapping (documented so it can be tuned/replaced):

  `c = exp(-cost / σ)`, with σ a per-quality constant (cost is small=good on NVIDIA OFA).

  An `exp` falloff is smooth, monotonic, and never hard-cuts, which the brief wants
  (less ghosting, stabler transitions). Forward–backward disagreement and, when present,
  depth/MV agreement multiply into `c`. `c` is then used as the **blend weight** between
  the warped sides and as the hole/inpaint weight — a bad region is *down-weighted*, not
  discarded.

### 4. Motion-vector fusion — in two places (req. #3)
Because the OFA accepts hints (`hintSupported=1`), MV fusion is done **twice, adaptively**:
- **Pre (hint):** app MVs are converted to S10.5 pixel space and fed as the OFA **hint**,
  steering the hardware search — strictly better initial flow, especially on low-texture
  or repetitive regions where pure OFA aliases.
- **Post (refine):** where OFA confidence `c` is low, REFINE blends toward the app MV:
  `F = lerp(F_ofa, F_mv, 1 - c)`. Neither source is fully replaced; the blend is
  confidence-weighted. MVs also give a disocclusion prior via reprojection.

### 5. Depth fusion (req. #4)
- When depth is registered, REFINE reprojects prev/curr depth into the halfway frame
  using `reproj` and derives a **disocclusion mask** from depth discontinuities and
  front-most-surface selection — telling the interpolator which side (prev or curr) is
  the *visible* source for a hole.

### 6. Material / object IDs (req. #5) & reactive mask (req. #6)
- Optional `material_id` and `reactive` maps are registered and passed through to the
  interpolator. v1 uses `reactive` to **suppress interpolation** (dup nearest real pixel)
  for alpha-tested/particle/water pixels where flow is meaningless — the FSR-style idea.
  Material IDs are plumbed but only inform special-casing later (glass/water/vegetation).

### 7. UI mask is mandatory (req. #7)
- Where `ui_mask` marks UI, the output is taken **from the nearest real frame, never
  interpolated**. Applied as the final compositing step (or the app draws UI post-FG).
  HUD/crosshair/text therefore never ghost.

### 8. The `Interpolator` interface (req. #8)
```cpp
struct InterpolatorInputs {   // all GPU images + the refined flow/confidence/masks
    VkImageView prevColor, currColor;
    VkImageView flowFwd, flowBwd, confidence, occlusion, disocclusion;
    VkImageView uiMask, reactive, materialId;  // nullable
    float phase;                               // 0.5 for 2x
};
class Interpolator {
public:
    virtual ~Interpolator() = default;
    virtual void record(VkCommandBuffer cmd, const InterpolatorInputs&, VkImageView out) = 0;
};
```
- **v1: `WarpInterpolator`** — occlusion-aware two-sided backward warp + confidence blend
  + edge-aware hole fill, all Slang compute. `CNNInterpolator` / `TransformerInterpolator`
  are later implementations of the same interface (Tensor Cores via
  `VK_KHR_cooperative_matrix`, per the analysis).

### Shading language & packaging
- Shaders are written in **Slang** (compiled to SPIR-V with `slangc`), chosen for modern
  language features and maintainability. `slangc` is located by CMake via an overridable
  path so nvofg stays standalone.
- Compiled SPIR-V is **embedded** into the library (generated C arrays) so the shipped
  `.so` is self-contained — no runtime shader-file lookup.

### M1 scope (this milestone) vs later
To honour "iterate, no big-bang", M1 implements the **skeleton + forward-flow warp**:
PREP(luma) → OFA forward flow (grid 4×4, cost on) → REFINE(upsample, decode S10.5,
cost→confidence, light median) → `WarpInterpolator` (two-sided backward warp + confidence
blend + simple hole fill) → output. Bidirectional consistency, depth/MV/reactive/UI paths
are wired as no-ops/flags now and filled in M2, exactly where this ADR places them.

## Consequences
- The pluggable boundary keeps the core Vulkan-only and portable while allowing NVIDIA-
  only learned back-ends later (analysis §D).
- Confidence-as-weight and two-place MV fusion are baked into REFINE's data model from the
  start, so M2 adds inputs, not restructuring.
- Two-queue timeline sync is designed in from M1 (not retrofitted), matching the §4 sync
  contract in the C ABI.

## Alternatives considered
- **Forward (scatter) warp as primary.** Rejected for v1: scatter needs atomics and
  careful z-resolution; two-sided *backward* warp with a confidence blend gives comparable
  quality with simpler, more portable compute. Revisit if disocclusion quality demands it.
- **Threshold-based cost rejection** (design.md §3 wording). Rejected per requirement #2
  in favour of continuous confidence; a threshold is the `σ→0` degenerate case.
- **Runtime shader files.** Rejected in favour of embedding for a self-contained `.so`.
