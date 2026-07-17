# nvofg — native Vulkan frame generation on NVIDIA Optical Flow

A small, **engine-agnostic** frame-generation library for **native Linux Vulkan**
applications, built directly on NVIDIA's hardware Optical Flow accelerator
(`libnvidia-opticalflow.so`, the `NvOFVk` API). It generates interpolated frames
*between* two rendered frames to multiply presented framerate — the same class of
feature as DLSS-G / FSR 3 FG, but:

- **native** — no Wine, no Proton, no Windows PE bridge;
- **no paid dependency** — no `Lossless.dll` (unlike lsfg-vk);
- **hardware-accelerated** — uses the GPU's dedicated OFA block, not a shader
  optical-flow approximation;
- **reusable** — a C ABI + Rust crate any Vulkan engine can adopt, not tied to
  RustMineClient.

> This is *our own* interpolation using NVIDIA's optical-flow **hardware**. It is
> **not** DLSS-G's learned model — NVIDIA gates that behind a Windows-only PE
> (`nvngx_dlssg.dll`). What is native and usable is the OFA silicon and its
> `NvOFVk` Vulkan API, which is exactly what this framework drives.

---

## 1. Goals & non-goals

### Goals
- **G1.** Given two consecutive final color frames (`prev`, `curr`) as `VkImage`s,
  produce one or more interpolated frames on the GPU with **zero CPU readback**.
- **G2.** Optional but strongly recommended **auxiliary inputs** — depth and
  application motion vectors — to disambiguate disocclusion and reject bad flow.
  The library must work *without* them (pure optical flow) and *better* with them.
- **G3.** **UI exclusion** — the app can pass a mask (or draw the HUD after FG) so
  interpolation never ghosts crosshair/text. This is the structural advantage over
  layer-based FG (lsfg-vk) which cannot see the HUD.
- **G4.** A clean **C ABI** (`nvofg.h`) + an idiomatic **Rust wrapper crate**
  (`nvofg`), so any engine — ours or a third party — integrates in ~50 lines.
- **G5.** **GPU-driven, ~0 CPU/frame.** All warp/blend/composite work is Vulkan
  compute/graphics the app records into its own command buffers. No hidden threads,
  no per-frame allocations after warm-up.
- **G6.** **Graceful absence.** If OFA is missing (non-NVIDIA GPU, old driver,
  disabled), `nvofg_create` returns `UNSUPPORTED` and the app runs normally at 1×.

### Non-goals
- Not a present-pacing layer. The library **produces frames**; *pacing/scheduling*
  onto the swapchain is a thin optional helper (§7), not the core.
- Not DLSS-G. No learned interpolation model; quality target is "clean 2× at ≥60 fps
  base," competitive with FSR 3 FG, not identical to DLSS-G.
- Not an upscaler. Composes *downstream* of DLSS/DLAA/FSR/native rendering; it takes
  whatever final frames it's given.
- No D3D12/Windows path in v1 (the API supports it; out of scope here).

---

## 2. Background: what the driver actually exposes

Verified on this machine (RTX 5070 Laptop, driver 610.43.03):

- `libnvidia-opticalflow.so.610.43.03` exports:
  - `NvOFGetMaxSupportedApiVersion`
  - `NvOFAPICreateInstanceVk`  ← **the Vulkan entry point**
  - `NvOFAPICreateInstanceCuda`
- Error strings confirm the Vulkan structs we must fill:
  - `NV_OF_REGISTER_RESOURCE_PARAMS_VK` (`.image`, `.format`, `.hOFGpuBuffer`)
  - `NV_OF_EXECUTE_INPUT_PARAMS_Vulkan` (`.numFencePoints ≥ 1`, fence points)
  - `NV_OF_EXECUTE_OUTPUT_PARAMS_*` (output flow buffer, optional cost/global-flow)
  - `NV_OF_INIT_PARAMS` (grid size, mode, `predDirection`, hints, output cost,
    global flow)
- `VK_NVX_binary_import` (rev 2) + `VK_NVX_image_view_handle` (rev 4) are present —
  relevant only if we ever import CUDA kernels; **not needed** for the pure-`NvOFVk`
  path, which is self-contained through the `.so`.

The `NvOFVk` instance is created **against the app's existing `VkInstance` /
`VkPhysicalDevice` / `VkDevice`** — it shares the app's device, no separate context.
Resources are registered `VkImage`s; execution is enqueued on a `VkQueue` and
synchronised with **timeline/binary semaphores via fence points**. This is what makes
a zero-copy, GPU-driven integration possible.

> **Header source.** The struct layouts above come from NVIDIA's **Optical Flow SDK**
> (`nvOpticalFlowCommon.h`, `nvOpticalFlowVulkan.h`). v1 vendors those headers (BSD-ish
> NVIDIA SDK license — verify before redistribution) under `third_party/nvof/`. The
> runtime `.so` is the user's installed driver; we `dlopen` it, we do **not** ship it.

---

## 3. The pipeline (per generated frame)

```
        frame N-1 (color)          frame N (color)
             │                          │
             │   optional: depthN, mvN (app motion vectors), uiMaskN
             ▼                          ▼
  ┌───────────────────────────────────────────────┐
  │ 1. FORMAT PREP (compute)                        │  color → NV12/ABGR8 as OFA wants,
  │    resample to OFA grid, register once          │  at flow grid res (1/1,1/2,1/4)
  └───────────────────────────────────────────────┘
             │
             ▼
  ┌───────────────────────────────────────────────┐
  │ 2. NvOFExecute  (OFA hardware)                  │  → flow field F (R16G16 per grid cell),
  │    input=N-1, reference=N, forward (+ optional  │    optional cost buffer C,
  │    backward for bidir), fence-point synced      │    optional global flow
  └───────────────────────────────────────────────┘
             │  F (fwd), F' (bwd), cost
             ▼
  ┌───────────────────────────────────────────────┐
  │ 3. FLOW REFINE (compute)                        │  upsample flow to full res,
  │    - dilate/median to kill speckle              │  scale by phase t (0.5 for 2×),
  │    - reject high-cost cells (use cost buffer)   │  reconcile fwd/bwd for occlusion,
  │    - fuse app MVs where flow is unreliable      │  build disocclusion mask from depth
  └───────────────────────────────────────────────┘
             │  refined halfway flow + occlusion mask
             ▼
  ┌───────────────────────────────────────────────┐
  │ 4. WARP + HOLE-FILL + BLEND (compute/graphics)  │  forward+backward warp N-1,N to N-0.5,
  │    - two-sided warp, occlusion-aware blend      │  fill holes from the non-occluded side,
  │    - edge-aware inpaint on remaining holes       │  clamp to src neighborhood (anti-ghost)
  └───────────────────────────────────────────────┘
             │  interpolated color N-0.5
             ▼
  ┌───────────────────────────────────────────────┐
  │ 5. UI COMPOSITE (graphics)                      │  re-draw HUD / apply uiMask so the
  │    (skip if app draws UI post-FG itself)        │  crosshair & text never interpolate
  └───────────────────────────────────────────────┘
             │
             ▼   interpolated frame ready → app presents it before frame N
```

Steps 1,3,4,5 are **our** Vulkan compute/graphics (portable, could even run on
non-NVIDIA if step 2 is swapped for a shader flow — see §8). Step 2 is the OFA
hardware call. For **2× FG** we synthesize one middle frame (phase t=0.5). For 3×/4×
we synthesize at t∈{1/3,2/3} or {1/4,2/4,3/4} from the *same* flow field — cheap,
one `NvOFExecute` feeds all intermediate phases.

### Latency model
FG inherently holds frame N to interpolate N-0.5 *before* N is shown:

```
render:   … N-1 ─────────── N ───────────── N+1 …
present:  … N-1 ─▶ [N-0.5] ─▶ N ─▶ [N+0.5] ─▶ N+1 …
                    ↑ generated from N-1 & N; so N waits one extra half-frame
```

So base fps should be capped to `displayHz / multiplier` and the app should keep
VSync off (the app or the §7 pacer owns final timing). Added latency ≈ one base
frame; there is **no Reflex** on native Linux — document it.

---

## 4. Public API (C ABI — `nvofg.h`)

Opaque handle, explicit resource registration, app-recorded commands. All functions
are `nvofg_*`, return `NvofgResult`, never allocate on the hot path.

```c
typedef struct NvofgContext NvofgContext;   // opaque

typedef enum {
    NVOFG_OK = 0,
    NVOFG_UNSUPPORTED,        // no OFA / driver too old / non-NVIDIA  → caller runs 1×
    NVOFG_INVALID_ARGUMENT,
    NVOFG_DEVICE_LOST,
    NVOFG_OUT_OF_MEMORY,
    NVOFG_INTERNAL,
} NvofgResult;

typedef enum { NVOFG_QUALITY_PERF, NVOFG_QUALITY_BALANCED, NVOFG_QUALITY_HIGH } NvofgQuality;

// ---- creation ----------------------------------------------------------------
typedef struct {
    VkInstance        instance;
    VkPhysicalDevice  physical_device;
    VkDevice          device;
    VkQueue           queue;             // queue nvofg submits OFA work on (may be app's)
    uint32_t          queue_family_index;
    PFN_vkGetInstanceProcAddr gipa;      // so nvofg resolves VK fns against app's loader
    uint32_t          width, height;     // full present resolution
    NvofgQuality      quality;
    uint32_t          flags;             // NVOFG_FLAG_USE_DEPTH | _USE_MOTION | _BIDIRECTIONAL | _HDR
} NvofgCreateInfo;

NvofgResult nvofg_create(const NvofgCreateInfo* info, NvofgContext** out);
void        nvofg_destroy(NvofgContext*);

// Query without committing (for a settings menu): fills caps, returns UNSUPPORTED if none.
NvofgResult nvofg_query_support(VkPhysicalDevice, NvofgCaps* out);

// ---- resource registration (once, at swapchain create / resize) --------------
// Register the app's persistent images. nvofg calls NvOFRegisterResourceVk under the hood.
typedef struct {
    VkImage      image;
    VkImageView  view;
    VkFormat     format;
    uint32_t     width, height;
} NvofgImageDesc;

NvofgResult nvofg_register_color (NvofgContext*, const NvofgImageDesc* prev, const NvofgImageDesc* curr);
NvofgResult nvofg_register_aux   (NvofgContext*, const NvofgImageDesc* depth /*nullable*/,
                                                 const NvofgImageDesc* motion /*nullable*/,
                                                 const NvofgImageDesc* ui_mask /*nullable*/);
NvofgResult nvofg_register_output(NvofgContext*, const NvofgImageDesc* interpolated /* storage image */);
void        nvofg_unregister_all (NvofgContext*);   // before destroying/resizing images

// ---- per-frame recording (hot path, records into the app's cmd buffer) --------
typedef struct {
    VkCommandBuffer cmd;              // app's cmd buffer, in recording state
    float           phase;           // 0..1 position of the generated frame (0.5 = 2×)
    // matrices to reproject app MVs / depth into halfway space (optional, when aux present)
    float           reproj[16];      // prevVP_unjittered · inverse(currVP_unjittered), row-major
    float           near_plane, far_plane;
    uint32_t        reset;           // 1 on camera cut / teleport → skip interpolation, dup frame
} NvofgGenerateInfo;

// Records: format-prep → (submit OFA execute, returns its semaphore) → refine → warp → composite.
// The OFA execute is submitted internally on `queue` and synchronised to `cmd` via the
// returned timeline value; app waits on it in its present submit. Zero CPU stall.
NvofgResult nvofg_record_generate(NvofgContext*, const NvofgGenerateInfo*, NvofgFrameSync* out_sync);

// ---- introspection ----------------------------------------------------------
NvofgResult nvofg_caps(NvofgContext*, NvofgCaps* out);   // grid sizes, max res, bidir support
const char* nvofg_last_error(NvofgContext*);             // human-readable (wraps NvOFGetLastError)
```

### Threading / sync contract
- `nvofg_record_generate` **only records** into the passed `cmd`, except the single
  `NvOFExecute` which the driver enqueues on `queue`; nvofg returns a
  `NvofgFrameSync{ VkSemaphore, uint64_t value }` (timeline) the app adds to its
  present submit's wait list. No `vkQueueWaitIdle`, no fences on the hot path.
- All barriers between prep/flow/warp stages are recorded by nvofg into `cmd`.
- The app owns image lifetime; nvofg holds only registered handles + its own internal
  scratch (flow field, cost, pyramid) allocated once at `create`/`register`.

---

## 5. Rust wrapper (`nvofg` crate)

Thin, safe, `ash`-based. Mirrors the C ABI but RAII and `Result`.

```rust
pub struct FrameGen { /* owns *mut NvofgContext */ }

pub struct CreateInfo<'a> {
    pub device:   &'a ash::Device,
    pub instance: &'a ash::Instance,
    pub physical_device: vk::PhysicalDevice,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub extent: vk::Extent2D,
    pub quality: Quality,
    pub use_depth: bool,
    pub use_motion: bool,
    pub bidirectional: bool,
}

impl FrameGen {
    /// Returns Ok(None) if the GPU/driver has no usable OFA — caller runs at 1×.
    pub fn new(info: &CreateInfo) -> Result<Option<FrameGen>, Error>;

    pub fn register_color(&mut self, prev: &ImageDesc, curr: &ImageDesc) -> Result<(), Error>;
    pub fn register_aux(&mut self, depth: Option<&ImageDesc>, motion: Option<&ImageDesc>,
                        ui_mask: Option<&ImageDesc>) -> Result<(), Error>;
    pub fn register_output(&mut self, interpolated: &ImageDesc) -> Result<(), Error>;

    /// Records generation into `cmd`; returns the timeline semaphore + value to wait on.
    pub fn record_generate(&mut self, cmd: vk::CommandBuffer, gen: &GenerateInfo)
        -> Result<FrameSync, Error>;
}
```

Design notes:
- `new` returning `Option` bakes G6 (graceful absence) into the type system — the
  caller literally can't forget the fallback.
- No `unsafe` in the public surface; the FFI + `dlopen` live in an internal `sys`
  module (`nvofg-sys`). Matches an engine that "forbids unsafe" in app code.
- Feature flag `raw-handles` to interop with engines not using `ash` (wgpu-hal, vulkano)
  by accepting raw `u64` Vulkan handles.

---

## 6. Integration recipe (any engine, ~50 lines)

```
1. At device create:
   - add instance/device extensions nvofg asks for (query via nvofg_required_extensions).
   - FrameGen::new(...) → store Option<FrameGen>.
2. At swapchain create / resize:
   - allocate: prev_color, curr_color (copies of final frame), interpolated (storage).
   - register_color / register_aux(depth?, motion?, ui_mask?) / register_output.
3. Each rendered frame:
   a. render world+post+UPSCALE as usual → final frame in `curr_color`.
   b. if FrameGen present and !first_frame and !camera_cut:
        record_generate(cmd, GenerateInfo{ phase:0.5, reproj, reset }) → sync.
        present `interpolated` (wait on sync), THEN present `curr_color`.
      else: present `curr_color` only.
   c. swap prev_color ↔ curr_color (ping-pong; no copy if you render into the ping side).
4. Cap base fps to displayHz/multiplier, VSync off (or use the §7 pacer).
```

For **RustMineClient** specifically: it already produces the two hardest-to-get
inputs — **motion vectors + depth + a `reproj` matrix** (built for the DLSS path,
see `upscale.rs`) — so it can register aux immediately and get the depth/MV-assisted
quality tier for free. The HUD is drawn into the scene, so RMC uses the `ui_mask`
path (or moves HUD post-FG) to avoid crosshair ghosting.

---

## 7. Optional present pacer (`nvofg_pacer`, separate module)

Producing frames is core; *scheduling* them smoothly is a documented helper, not
required. The pacer:
- caps the base submit rate to `displayHz / multiplier`;
- interleaves `[gen] real [gen] real …` onto the swapchain with even spacing using a
  timeline-semaphore release schedule (no CPU spin — target-time waits via
  `VK_KHR_present_wait` when available, else a monotonic sleep on a dedicated thread);
- exposes `nvofg_pacer_tune(multiplier, flow_scale, quality)` for a settings menu.

Kept separate so engines with their own frame scheduler ignore it.

---

## 8. Portability / fallback tiers

| Tier | Flow source | Needs | Quality |
|---|---|---|---|
| **A. OFA (native NVIDIA)** | `NvOFVk` hardware | NVIDIA driver ≥ 5xx, Turing+ | best, HW-accelerated |
| **B. Shader flow (portable)** | compute optical-flow (pyramidal Lucas-Kanade / Farnebäck) | any Vulkan GPU | lower, more artifacts, no extra deps |
| **C. MV-only warp** | app motion vectors only, no optical flow | app provides MVs+depth | ok for slow motion; ghosts on shading changes |

v1 ships **Tier A + Tier C** (C is nearly free given the warp stage already exists).
Tier B is a stretch goal that makes the library vendor-neutral (AMD/Intel) — same
API, `NVOFG_QUALITY_*` selects, `nvofg_query_support` reports which tiers a GPU can do.
This keeps the framework honest to its "anyone can use it" mandate.

---

## 9. Risks & mitigations

- **Present-pacing DEVICE_LOST** (bit us before in RMC): keep pacing *out* of the core;
  the core only records into the app's cmd buffer. The optional pacer is isolated and
  off by default.
- **OFA grid coarseness.** OFA outputs flow on a 1/1, 1/2, or 1/4 grid — fine motion
  (thin lines, text) aliases. Mitigate with cost-buffer rejection + app-MV fusion +
  UI exclusion; document text/HUD as worst case (same caveat as all FG).
- **Disocclusion holes.** Bidirectional flow (`predDirection=bidirectional`) + depth
  gives a second warp side to fill from; edge-aware inpaint for the residue.
- **Camera cuts.** `reset=1` → skip interpolation, duplicate the real frame (never
  interpolate across a cut).
- **Driver/API version drift.** Gate on `NvOFGetMaxSupportedApiVersion`; pin tested
  ranges; degrade to Tier C if the Vk entry rejects our version.
- **Header licensing.** NvOF SDK headers are NVIDIA-licensed — vendor with the license
  file, confirm redistribution terms, or generate our own struct definitions from the
  documented ABI if terms disallow shipping the headers.
- **HDR / color space.** `NVOFG_FLAG_HDR` path keeps warp/blend in linear/scRGB; OFA
  runs on a luma-ish derived image, not the HDR values directly.

---

## 10. Repository layout & milestones

```
nvofg/
├─ design.md                 (this file)
├─ include/nvofg.h           C ABI
├─ src/                       C++ core (Vulkan, dlopen libnvidia-opticalflow)
│  ├─ context.cpp            create/destroy, OFA instance, caps
│  ├─ nvof_vk.cpp            NvOFVk register/execute wrappers
│  ├─ prep.comp / refine.comp / warp.comp / composite.comp   (GLSL/Slang)
│  └─ pacer.cpp              (optional module)
├─ third_party/nvof/         vendored NvOF SDK headers (+ LICENSE)
├─ crates/
│  ├─ nvofg-sys/             raw FFI bindings to nvofg.h
│  └─ nvofg/                 safe ash-based wrapper
├─ examples/
│  ├─ triangle_2x/           minimal ash app proving 2× FG on a spinning scene
│  └─ headless_flow/         dump OFA flow field to PNG (debug)
└─ tests/                    golden-image + flow-sanity tests
```

**Milestones**
- **M0 — spike:** `dlopen` the `.so`, `NvOFGetMaxSupportedApiVersion`,
  `NvOFAPICreateInstanceVk` against an ash device, register two images, run one
  `NvOFExecute`, dump the flow field to PNG. *Proves the native OFA path end-to-end.*
- **M1 — 2× core:** prep → execute → refine → warp → blend → one interpolated image,
  no aux. `examples/triangle_2x` visibly doubles a spinning quad's smoothness.
- **M2 — aux quality:** depth + app-MV fusion, bidirectional flow, disocclusion mask,
  camera-cut reset. Golden-image tests.
- **M3 — API polish:** C ABI + Rust crate finalized, `query_support`, graceful
  `UNSUPPORTED`, resize handling, docs.
- **M4 — integrate in RMC:** wire behind the existing Video → Frame Generation toggle,
  reuse RMC's MV/depth/reproj. Independent of the library repo.
- **M5 (stretch) — pacer + Tier B** shader flow for vendor-neutrality.

---

## 11. Why this is worth building

- It's the **only native, no-Wine, no-paid-DLL, hardware-accelerated** frame-gen path
  on Linux — a gap nothing currently fills (lsfg-vk needs `Lossless.dll`; DLSS-G/FSR3
  FG are Windows/Proton-gated; OptiScaler targets Proton games, not native binaries).
- Built as a **standalone framework**, it's useful to any native-Linux Vulkan engine,
  not just RMC — which is exactly the mandate for this folder.
- The hard, differentiating inputs (motion vectors, depth, reproj matrix, UI mask) are
  things a real renderer already has — so integrators get the *good* quality tier, not
  just blind optical flow.

## 12. Sources / evidence
- On-system driver: `/usr/lib/libnvidia-opticalflow.so.610.43.03` exports
  `NvOFAPICreateInstanceVk`, `NvOFGetMaxSupportedApiVersion`; error strings confirm
  `NV_OF_REGISTER_RESOURCE_PARAMS_VK`, `NV_OF_EXECUTE_INPUT_PARAMS_Vulkan` (fence
  points), init params (grid/predDirection/cost/global-flow).
- NVIDIA **Optical Flow SDK** (headers `nvOpticalFlowCommon.h`, `nvOpticalFlowVulkan.h`)
  — the authoritative struct/ABI reference.
- Prior investigation: `../.RustMineClient/docs/16-ngx-framegen-bridge-feasibility.md`
  (why OFA is the one feasible native route) and `docs/15-lsfg-vk-framegen-plan.md`
  (the layer-based alternative this replaces for NVIDIA users).
```

---

## 13. Long-term vision — "RenderFX" and the case for a focused core

> **Status:** vision / design-guardrail, **not** a committed roadmap item. This section
> evaluates a proposal to grow nvofg into a modular Vulkan rendering-effects framework
> ("RenderFX") with a shared Core (resource/sync/descriptor management + common inputs:
> motion vectors, depth, jitter, camera matrices, exposure, resolution, color) and
> optional plugin modules (Frame Generation, Upscaling, Anti-Aliasing, Ray
> Reconstruction) over vendor backends (NVIDIA/AMD/Intel/Generic), behind one API. Per
> the request, **no current API or implementation is changed by this section** — it only
> records the analysis and the guardrails that keep such an evolution *possible* without
> breaking today's users.

### 13.1 The proposal, stated fairly
Almost every modern temporal technique consumes the same inputs (MVs, depth, jitter,
matrices, exposure, color, resolution). The proposal is to manage those **once** in a
vendor-neutral Core and expose each technique as an optional module sharing that Core, so
an engine tags its resources one time and toggles DLSS/DLAA/RR/FG/XeSS/FSR as plugins.
The Core must not force any proprietary dependency (NGX/FidelityFX/XeSS); vendor tech
lives only in optional backends.

### 13.2 This is validated prior art — and that cuts both ways
The design is essentially what **NVIDIA Streamline** (`sl.*`: tag resources once, enable
DLSS/DLSS-G/Reflex as plugins) and the **AMD FidelityFX SDK** (a common backend +
effect modules) already do. So the shape is proven and the input-sharing win is real.
But it also means the idea's hard parts are *already solved by shipping SDKs on Windows*,
and that the differentiated, unfilled gap is specifically a **native-Linux, vendor-neutral**
version — which is a far larger mission than "frame generation on the OFA."

### 13.3 Critical assessment

**Would it improve reusability?** Yes, for the *input contract*. A small, stable,
vendor-neutral description of MV/depth/jitter/matrices/exposure/color/resolution is
genuinely reusable across FG, upscaling, and AA, and removes duplicate plumbing for
integrators. This part of the vision is sound and worth designing toward now.

**Would it improve maintainability? Mixed — and mostly *worse* if done as one library.**
- A shared Core only helps if the effects truly share it. They diverge more than the
  pitch suggests: FG needs *two* finished frames + the OFA + a middle-frame present
  cadence; upscaling needs *one* frame + jitter *history* + a fixed output scale; AA/TAA
  needs history but not FG's occlusion machinery; RR is denoiser-shaped. A Core general
  enough to serve all becomes a leaky abstraction that is *harder* to maintain than
  focused libraries — the classic framework-vs-library tradeoff.
- Scope explosion is the real risk. nvofg's entire value today is being the **only**
  native, no-Wine, no-paid-DLL, hardware-accelerated Linux FG. Turning it into a
  rendering-effects framework dilutes that focus and could mean never shipping an
  *excellent* FG because effort went into a framework.

**Would the module menu even exist natively?** Largely no, and this is decisive. On
native Linux the "good" upscalers/RR are as gated as DLSS-G: DLSS/DLAA/RR are NGX
(the quality models are Windows PEs), XeSS's best path is Intel-GPU/oneAPI, and only
**FSR** is truly open and cross-vendor. So a Linux-native RenderFX would realistically
offer *FSR + its own OFA frame-gen + generic shader fallbacks* — a fraction of the
advertised DLSS/DLAA/RR menu. The unified-API ergonomics would be real; the module set
would be modest. Honest scoping matters here (same caveat as §1's non-goals).

**API-stability risk.** Designing the grand `enable_dlss/enable_rr/execute` surface now,
before each effect's real needs are understood, is premature abstraction. The *inputs*
subset is well understood and safe to stabilize; the *per-effect orchestration* surface
is not, and freezing it early would create exactly the API breakage the request wants to
avoid.

**Naming.** `nvofg` literally means *NVIDIA Optical Flow*. A vendor-neutral umbrella
wants a neutral identity. That nvofg is honestly NVIDIA-OFA-specific is a feature, and it
argues for the umbrella being a *different* thing that composes nvofg, not nvofg renamed.

### 13.4 Recommendation — compose, don't absorb

**Keep nvofg a focused, best-in-class native-Linux frame-generation library.** Do **not**
grow it into RenderFX. Instead, if the umbrella is pursued, build it as a **separate
composition layer** (a "Streamline for native Linux") that *depends on* nvofg as its FG
module and on FSR/XeSS/etc. as sibling modules. This is the Unix-tools model — small
sharp libraries plus a thin composition layer — and it is objectively better here than a
monolith-core-absorbs-everything because it: preserves nvofg's focus and shippability;
lets each module keep the divergent internals it actually needs; avoids a premature
over-general Core; and sidesteps the naming/scope tension. It also matches the boundary
this project already drew (ADR 0002's pluggable interpolator; the CUDA-vs-Vulkan
analysis's rule that vendor tech lives *behind* a boundary, never in the core).

**What that means we do — and don't — do now (guardrails, zero API change):**
- **Do** keep the public input structs (`NvofgImageDesc`, `NvofgAuxDesc`, `reproj`,
  near/far, resolution, quality) clean, minimal, and free of FG-internal coupling, so a
  future shared "core-inputs" contract can adopt the *same* description without a break.
  Treat these as the seed of a reusable input vocabulary.
- **Do** keep the Core dependency-free of any vendor SDK beyond the already-isolated OFA
  backend (ADR 0001), so nvofg could later be *one backend module* under an umbrella
  without carrying proprietary weight into that umbrella's core.
- **Do** keep the modular-interpolator seam (ADR 0002 §8) — it is the template for how an
  umbrella would host effect modules.
- **Don't** introduce an `enable_*/execute` mega-API, a shared descriptor/resource
  manager spanning hypothetical effects, or any speculative Core abstraction now. Those
  are the parts most likely to be wrong before real second/third modules exist, and they
  are exactly what the request says must not destabilize current users.
- **When a real second module appears** (most plausibly an FSR-style upscaler or a Tier-B
  shader optical-flow, §8), *that* is the moment to extract the shared input/sync/
  descriptor utilities into a genuine `core` crate — refactored from two working modules,
  not designed in the abstract. Two concrete users is the right trigger; one is not.

### 13.5 Bottom line
The vision is directionally right about *input sharing* and *plugin modularity*, and
wrong about *where it should live*. Ship nvofg as a focused FG library; keep its inputs
and boundaries clean enough to be reused; and let a **separate** native-Linux composition
project (if it ever materializes) adopt nvofg as its FG backend. That maximizes both the
near-term value (a great FG that actually ships) and the long-term option value (a clean
seam to build the umbrella later) with **no** risk to current users.

---

## 14. Linux-first mandate

The primary target is **native Linux Vulkan**, and the architecture is designed
Linux-first, not Windows-first. Concretely this means, as standing constraints on every
future decision:

- **No platform assumption beyond a Vulkan 1.3 loader + a POSIX-ish host.** No Win32,
  no D3D, no PE bridges, no Wine/Proton in any code path.
- **Vulkan is the substrate.** Interop with other APIs (CUDA, §13/ADR 0004) is optional,
  internal, and imported *into* Vulkan (external memory/semaphore fd) — never a Windows
  handle type, never a requirement.
- **Windows is a possible *consequence*, never a driver of design.** If a feature falls
  out naturally cross-platform, fine; but Linux support is never traded for it.

This is already how nvofg is built (ADR 0001: `VK_NV_optical_flow`, no dlopen of Windows
DLLs; interop via `OPAQUE_FD`, ADR 0004). §14 records it as an explicit, permanent
principle so later modules can't quietly regress it.

## 15. Extensibility guardrails for a future modular framework

§13 recommends nvofg stay a focused FG library and that any "RenderFX" umbrella be a
**separate** project composing it. This section makes that seam concrete: the small,
zero-cost things to get right *now* so the umbrella is possible **without API breaks**,
and — critically — the parts of the proposed design that are **wrong as stated** and
should be replaced with something objectively better.

### 15.1 Shared Frame Context — adopt as an *input vocabulary*, not a subsystem
The proposal (one `FrameContext` holding color/depth/MVs/material/reactive/exposure/
jitter/reproj, later roughness/albedo/normals/GBuffer; produced once, reused by every
technique) is **sound and worth designing toward** — it is precisely what Streamline and
the FidelityFX SDK do, and it removes the duplicate-plumbing tax when a second technique
appears. But two critical caveats:

- **The Core must not *own* or *produce* these resources — the renderer does.** The
  framework only needs a stable *description* of them. nvofg's public inputs
  (`NvofgImageDesc`, `NvofgAuxDesc`, `reproj`, near/far) are already a subset of this
  vocabulary. The guardrail is to keep those structs a clean, minimal, FG-agnostic
  *view* onto app resources — which they already are.
- **ABI extensibility is the real requirement.** "Add roughness/albedo/normals later
  without breaking users" is an ABI-versioning problem, not a grand-design problem. The
  concrete mechanism: give the input structs a **Vulkan-style extensibility contract** —
  a `sType`/`pNext` chain *or* a leading `struct_size`/`version` field — so new fields
  append without changing existing layouts, plus a **capabilities query** so callers
  discover what a given build/GPU consumes. This is the single most important thing to do
  now; it costs nothing and is the actual guarantee of "no future API breaks."

**Verdict:** adopt the Frame Context as a versioned, renderer-owned input vocabulary that
nvofg's inputs already prefigure. Do **not** build a resource manager for it now.

### 15.2 Render graph — reject the framework-owned graph; adopt a *stage-ordering contract*
This is where the proposal is **objectively wrong as stated**, and a better architecture
exists. A framework that owns a render graph (resource lifetime, barrier scheduling,
aliasing, execution) is a large engine subsystem — and **real engines already have one**
(RustMineClient has `render_graph.rs`). Two render graphs fighting over the same command
buffers is a recipe for redundant barriers, ownership conflicts, and exactly the
present/sync fragility §9 warns about.

The industry-proven answer is **Streamline's model, not FidelityFX's framework**: the
composition layer owns **no** graph. It defines only:
1. a **logical stage order** — `Renderer → FrameContext → Upscaling → RayReconstruction →
   FrameGeneration → Present` — as a *contract*, and
2. **stage plugins** that record into command buffers the **engine's** graph schedules.

So "each stage selects a backend" is right; "the framework is a render graph" is wrong.
The framework is a **set of composable stage modules + an ordering contract**; the engine
remains the scheduler. This is strictly better on maintainability (no duplicate graph),
integration (drops into any engine's existing graph), and the §9 safety mandate (the
framework never owns queue submission it can DEVICE_LOST on). nvofg already embodies this:
it records into the app's command buffers and only owns its isolated OFA submit.

**Verdict:** logical stage order + pluggable per-stage backends: **yes.** A framework that
executes a render graph: **no.** Adopt the Streamline-style thin-layer model.

### 15.3 Algorithm ⟂ Execution Backend — adopt, with a caveat about the sparse matrix
Separating the *algorithm* (Warp / CNN / Transformer) from the *execution backend*
(Vulkan Compute / CUDA / TensorRT / CPU) is a genuinely good orthogonal-axis design and is
**already latent** in the codebase: ADR 0002 §8's `Interpolator` interface is the
algorithm axis; ADR 0004's optional CUDA path is a backend axis. Refinements:

- **The matrix is sparse, not full.** "Warp on TensorRT" is meaningless — Warp is not a
  network. The backend axis is meaningful only for the **learned** algorithms (CNN /
  Transformer), which can run on Vulkan-cooperative-matrix, CUDA, or TensorRT. Classical
  Warp is Vulkan-compute by nature. So the interface is: *algorithm* is the public
  selector; *backend* is an implementation detail of a learned algorithm, chosen by
  build/runtime capability — not an independent public knob for every algorithm.
- **A CPU backend is worth it — as a golden reference.** A slow, exact CPU implementation
  of each learned algorithm is invaluable for deterministic tests and debugging
  (compare GPU output against it), which fits nvofg's "test everything" mandate. This is
  the strongest reason to keep the backend axis explicit.

**Verdict:** adopt the split for learned interpolators; keep it an implementation detail
behind the algorithm interface, not a full public matrix; add a CPU reference backend for
tests.

### 15.4 Hardware-independent public API — already the principle; reaffirm
The public API must never expose CUDA / NGX / OFA specifics; backend selection is
internal. This is **already the design**: ADR 0001 keeps the core Vulkan-only; the Tier
A (OFA) / Tier B (shader) ladder is exactly the "VK_NV_optical_flow → future AMD → future
Intel → shader fallback" abstraction the proposal asks for, already implemented and
selected internally (`NVOFG_FLAG_FORCE_SHADER_FLOW` + auto-fallback). ADR 0004 keeps CUDA
optional and behind the interpolator boundary. The guardrail: **never let a vendor type
leak into `nvofg.h`** (no `cudaX`, no `NVSDK_NGX_*`, no `NvOF*` in the public header —
already true). Reaffirmed as permanent.

## 16. DLSS / DLAA / Ray Reconstruction on native Linux — Option A vs B

The request asks whether DLSS-class techniques should use **(A)** NVIDIA's official Linux
interfaces (NGX) or **(B)** the "unified RenderFX" approach of driving underlying native
capabilities directly, the way nvofg drives the OFA for frame generation. This needs a
rigorous answer because it rests on a technical asymmetry the framing hides.

### 16.1 The decisive asymmetry: reusable hardware block vs. proprietary model
nvofg can bypass DLSS Frame Generation because FG decomposes into **(i) a general-purpose
hardware block** (the OFA, a standalone optical-flow engine usable via a standard Vulkan
extension) **plus (ii) a viable non-proprietary algorithm** (classical warp; a learned
one later). The *value* of DLSS-G is partly a trained network, but a *usable, lower-tier*
FG exists **without** it.

**DLSS Super Resolution, DLAA, and Ray Reconstruction have no such decomposition.** Their
value **is** the trained neural network. There is no "underlying native upscaling block"
analogous to the OFA — the only underlying capability is the **Tensor Cores**, which are
just matrix-multiply units. Running them buys you nothing unless you already possess a
trained SR/denoiser model. Therefore "Option B for DLSS SR/RR" reduces to **"train your
own DLSS,"** which is an enormous ML research program, would not match DLSS quality, and —
if it meant extracting or reusing NVIDIA's weights — is legally untenable. Option B, as a
*native reimplementation of DLSS*, **collapses** for SR/DLAA/RR. It only ever worked for
FG, and that case is already built.

### 16.2 What is actually available natively
- **DLSS SR / DLAA / RR via NGX** *are* reachable on Linux — the driver ships the NGX
  runtime and the feature `.so` snippets; the app links the NGX app-side lib (RMC already
  does this). This is **Option A, and it works on Linux today.**
- **DLSS Frame Generation** is the Windows-gated one (`nvngx_dlssg.dll` PE) — the gap
  nvofg fills.
- **Open, vendor-neutral models** — FSR (fully open), XeSS (Intel, open-ish) — are the
  only "native, non-proprietary-model" upscalers/AA that actually exist. They are the
  legitimate Option-B-*spirit* backends: vendor-neutral, no NVIDIA dependency — but they
  are *other vendors' models*, not a native reimplementation of DLSS.

### 16.3 Axis-by-axis
| Axis | A: NGX (NVIDIA official) | B: "native reimplementation of DLSS" | Open backends (FSR/XeSS) |
|---|---|---|---|
| Image quality (SR/RR) | **best** (the real model) | far worse (your model) | between |
| Maintainability | low (NVIDIA maintains the model) | **very high** (own a model) | moderate |
| Performance/latency | optimized | unknown, likely worse | good |
| Linux-first | works (NGX on Linux) | native but hollow | fully native/open |
| Architectural cleanliness | clean *behind a stage interface* | n/a | clean behind same interface |
| Portability | NVIDIA-only | NVIDIA-only (Tensor Cores) | **cross-vendor** |
| Future-proofing | tracks NVIDIA | fragile | good |
| Future-driver compat | **best** (official) | fragile / RE-risk | n/a |
| Complexity | moderate | **highest** | moderate |
| Licensing/legal | clean (NGX SDK license; blobs from driver) | **untenable** if it reuses NVIDIA weights | clean (open) |

### 16.4 Recommendation — hybrid, and it is *not* an A-vs-B choice
The correct architecture **unifies the interface, not the implementation strategy**:

1. **Unify at the stage level** (§15.2): `Upscaling` and `RayReconstruction` are stages
   with pluggable backends, consuming the shared Frame Context.
2. **Per stage, offer multiple backends:**
   - **NGX (Option A)** as the *NVIDIA* backend for SR/DLAA/RR — the only path to DLSS
     quality, official, legal, Linux-supported.
   - **Open models (FSR/XeSS) + a built-in temporal** as the *vendor-neutral* backends —
     this is the RenderFX philosophy done correctly (pluggable, portable) **without**
     pretending to reimplement DLSS.
   - **native-HW-direct (Option B *spirit*)** *only where the FG asymmetry holds* — i.e.
     **frame generation** (OFA + warp = nvofg), already built.
3. **Do not attempt to natively reimplement DLSS SR/RR.** It is the one part of B that is
   infeasible and legally fraught; the asymmetry (§16.1) is why.

This hybrid is objectively superior to a pure A or a pure B: it delivers DLSS quality
where only NVIDIA can (via official NGX), vendor neutrality via open backends, and the
unique native FG nvofg already provides — all behind **one** stage interface. It is also
Linux-first (every backend runs natively) and future-proof: **if NVIDIA ever ships native
Linux DLSS-G, it is simply another `FrameGeneration` backend** next to nvofg, and any
future RR is another `RayReconstruction` backend — no API change, which is the stated
goal. That drop-in property is the entire payoff of §15.2's stage design.

### 16.5 Ray Reconstruction as a first-class stage — reserve, don't build
Making RR a first-class stage (inputs: MVs, depth, roughness, diffuse/specular albedo,
normals — all in the Frame Context) is architecturally right: **reserve the stage and the
Frame-Context fields.** But be honest that a *non-DLSS* RR is a large neural-denoiser
research effort with the same model-vs-hardware asymmetry as §16.1 — so in practice the RR
stage's only near-term real backend is **NGX RR (Option A)**, with open/none as the other
slots. Reserving the stage costs nothing and avoids a future API break; implying a native
RR is near would be dishonest.

## 17. Net verdict on the long-term vision

- **Linux-first, hardware-independent public API, shared input vocabulary, pluggable
  per-stage backends, algorithm⟂backend split** — all **sound**, mostly **already latent**
  in nvofg, and worth *preparing for now* via cheap guardrails (ABI-versioned input
  structs + capability queries; keep vendor types out of the public header; a CPU
  reference backend for tests).
- **A framework-owned render graph** — **rejected** in favour of a Streamline-style
  stage-ordering contract with engine-owned scheduling (§15.2). Objectively better on
  maintainability, integration, and safety.
- **"Option B: reimplement DLSS natively"** — **rejected** for SR/DLAA/RR due to the
  hardware-block-vs-model asymmetry (§16.1); the hybrid (NGX + open backends + native FG,
  one interface) is objectively superior.
- **Where it lives** — still **a separate RenderFX project** that composes nvofg as its FG
  backend (§13). nvofg stays a focused, shippable FG library; the framework is the thin
  composition layer.

None of the above changes the current implementation. The only *actionable now, zero-risk*
items are the ABI-extensibility guardrails in §15.1/§15.4 — and even those can wait until
the first real second consumer exists, since the public structs are already clean subsets
of the Frame Context vocabulary.

---

## 18. Capability discovery & backend dispatch (RenderFX) — refinements

§13–§17 settle the shape: a **separate**, Linux-first RenderFX that owns no render graph
(§15.2 stage-plugin model), composes nvofg as one Frame-Generation backend (§13), uses NGX
for DLSS SR/DLAA/RR (§16), and exposes **one API with interchangeable backends**. This
section records the follow-up asks — a Vulkan-style capability query and an automatic
**backend dispatcher** — and refines two points where the literal proposal should be
improved. The governing principle, elevated here as primary:

> **Unify the API, never the implementation.** The application programs against stages and
> a Frame Context; which backend executes a stage is a runtime detail it need not know —
> but *may* inspect and override.

### 18.1 Capability discovery — yes, but *inspectable policy + override*, not opaque magic
A Vulkan-style capability query is the right model, and nvofg already embodies the
module-level version of it: `nvofg_query_support` / `NvofgCaps` are **cheap, side-effect-
free**, and report not a bare bool but *tiers and features* (OFA vs shader, bidir, cost,
hint, grid, max res). RenderFX's `RenderFXCapabilities` should be the **aggregation** of
each module's own such query — RenderFX is an *aggregator*, not a reimplementation of every
vendor probe. This keeps modules independent and RenderFX thin (fractal capabilities).

The **refinement**: the request's `enable_dlss()/enable_rr()` → "RenderFX automatically
chooses the best backend" should not become an *opaque* auto-selector. That is not the
Vulkan model — Vulkan **enumerates and lets you pick with enough information to pick well**;
it does not silently choose your physical device. Pure auto-selection removes legitimate
application control:
- an app may prefer **FSR even on NVIDIA** for cross-machine visual consistency;
- an app may **avoid a proprietary blob** for licensing or reproducibility;
- an app may weight **latency over quality** (or vice-versa), or pin a backend for A/B
  testing and golden images.

So the better contract is **capability query → app states a *policy/preference* →
RenderFX returns a *recommended, fully-inspectable selection* → app may override/pin**:

```text
caps        = renderfx_query(frame_context_available, device)   // side-effect-free
preference  = { quality|balanced|perf, allow_proprietary, vendor_pin?, latency_budget }
selection   = renderfx_resolve(caps, preference)   // deterministic, inspectable, overridable
```

`selection` is data the app can read (which backend per stage, and *why*), not hidden
state. Auto-selection is then just "resolve with the default policy" — convenient, but
never a black box. This is strictly more capable than the literal proposal and still gives
the one-liner ergonomics for apps that don't care.

Each capability entry must carry more than "supported": a **quality/tier class**, a rough
**cost/latency** hint, and — tying back to §15.1 — the **Frame Context inputs the backend
requires**, so the app can produce *only* what the chosen selection needs.

### 18.2 Backend dispatch is a *set-resolution* problem, not per-stage independent picks
This is the one place the naive dispatcher (`Capability Discovery → Backend Selection →
Execution`, picking each stage's local best) is **subtly wrong**, and the correction
matters. Stages are **not independent** — real cross-stage constraints and bundles exist:
- NVIDIA's **RR requires DLSS SR** (RR replaces the SR upscale; you can't run NGX RR under
  an FSR upscaler);
- **FSR frame generation pairs with FSR upscaling**; a hypothetical **DLSS-G bundles SR+FG**;
- backends share history/jitter conventions that must agree across the temporal stages.

A dispatcher that greedily picks each stage's best in isolation can therefore produce an
**invalid combination** (e.g. FSR-upscale + NGX-RR). The correct model is that the
dispatcher **resolves a compatible *set* across all active stages**, respecting declared
bundles/constraints, and **reports conflicts** rather than silently mixing incompatible
backends. Concretely, each backend declares its cross-stage requirements as part of its
capability record; `renderfx_resolve` performs constraint satisfaction over the active
stages, not `argmax` per stage.

This is more work than the linear pipeline the proposal sketches, but it is the difference
between a dispatcher that *works* and one that produces subtly broken frames. Document it
as a first-class requirement of the dispatcher, not an afterthought.

### 18.3 "No vendor APIs in the public interface" vs. naming backends
A capability enum will contain identifiers like `DLSS_SR`, `FSR`, `nvofg`, `XeSS`. That is
**not** a violation of the hardware-independence rule (§15.4): a vendor name as a *backend
identifier* is fine; what must never leak is a vendor *API surface* (`NVSDK_NGX_*`,
`cudaX`, `NvOF*` types) into the RenderFX/nvofg public headers. The app selects
`FrameGeneration = nvofg` by an opaque handle/enum and drives it through the uniform stage
interface; it never touches nvofg's, NGX's, or CUDA's own types. This distinction is what
lets §16's hybrid (NGX + open + native) sit behind one clean API.

### 18.4 Net
- **Capability discovery:** adopt, as an aggregation of each module's existing cheap query
  (nvofg already provides its own); expose **policy + inspectable recommended selection +
  override**, not an opaque auto-picker.
- **Backend dispatch:** adopt, but as **cross-stage set resolution** with declared
  constraints/bundles and conflict reporting — never independent per-stage `argmax`.
- **Everything else** (unified-API-not-implementation, NGX-for-SR/RR, nvofg-stays-
  specialized-and-composed, Linux-first, no vendor types in the public surface) is
  affirmed as already concluded in §13–§17.

As before: **no implementation change now.** These are guardrails so that when the RenderFX
project is eventually built, its capability/dispatch layer is right the first time and
nvofg slots in as a backend unchanged.

---

## 19. NGX (DLSS SR/DLAA/RR) init on Linux — the `/usr/share/nvidia/ngx` write trap

**Relevant to RenderFX's NGX backend (§16), not to nvofg's OFA core** (nvofg uses only
`VK_NV_optical_flow` and needs none of this). Documented here because it blocks any NGX
feature init on a stock Linux install and the fix is non-obvious.

### Symptom
`NVSDK_NGX_VULKAN_Init*` fails with **`0xBAD00005`** (`NVSDK_NGX_Result_FAIL_InvalidParameter`).
The real cause is that NGX tries to create/write its model + CUBIN cache under the
**root-owned** `/usr/share/nvidia/ngx/` and gets *permission denied* (it needs root to
create that directory). Observed on driver **610**.

### What does NOT work (driver 610)
None of the commonly-cited redirects steer NGX off `/usr/share/nvidia/ngx` on this driver:
- `nvidia-ngx-conf.json` with `ngx_models_path`
- `__NGX_DISABLE_UPDATER`
- `NGX_CUBIN_DISABLE_RESOURCE_CACHE`

Don't rely on them; they silently no-op and init still fails.

### The fix — pass a WRITABLE application-data path to NGX init
`NVSDK_NGX_VULKAN_Init_with_ProjectID(...)` takes an **`InApplicationDataPath`**
(`wchar_t*`, UTF-32/4-byte on Linux). Point it at a directory the process **can write**
(a per-user cache dir, or a writable dir next to the DLSS feature snippets). NGX then
writes its logs **and its cubin/model cache into that directory**, so it never touches
`/usr/share/nvidia/ngx` and init succeeds — no root, no system-dir writes.

Do it in **both** places NGX reads a path from (belt and braces):
1. the legacy `InApplicationDataPath` argument, and
2. the modern `NVSDK_NGX_FeatureCommonInfo.PathListInfo` (the snippet/search path list).

Reference implementation (RustMineClient, `crates/renderer/src/ngx.rs::init`) — the exact
call that made DLSS SR/DLAA/RR init succeed on driver 610 despite the root-owned system
dir:

```c
// app_data_path = a WRITABLE dir (RMC uses its vendored vendor/dlss/... dir; a
// per-user ~/.cache/<app>/ngx works equally). UTF-32, null-terminated on Linux.
const wchar_t* app_data_path = L"/writable/dir";           // must be writable!
NVSDK_NGX_FeatureCommonInfo fci = {0};
const wchar_t* paths[1] = { app_data_path };
fci.PathListInfo.Path   = paths;                            // (2) modern path list
fci.PathListInfo.Length = 1;
fci.LoggingInfo = /* your callback */;

NVSDK_NGX_VULKAN_Init_with_ProjectID(
    project_id, NVSDK_NGX_ENGINE_TYPE_CUSTOM, engine_ver,
    app_data_path,                                          // (1) legacy data path
    instance, physicalDevice, device, gipa, gdpa,
    &fci, NVSDK_NGX_Version_API);
```

Key points:
- The directory must **exist and be writable** before `Init` (create it, e.g.
  `~/.cache/<app>/ngx`, `mkdir -p`).
- Pass real `vkGetInstanceProcAddr` / `vkGetDeviceProcAddr` (a null `gipa`/`gdpa` is a
  *separate* `0xBAD00005` cause — don't conflate the two).
- On Linux `wchar_t` is **4 bytes**: encode the path as UTF-32, null-terminated
  (`chars().map(|c| c as u32)` + trailing `0`), not UTF-16.
- Unrelated but adjacent: the DLSS **feature snippets** (`libnvidia-ngx-dlss.so`,
  `libnvidia-ngx-dlssd.so`) must be findable on the path list, else `SuperSampling.Available`
  probes false even though `Init` succeeded.

### For RenderFX
When RenderFX wires up its NGX backend (§16), it must create a writable per-user NGX cache
dir and feed it as above. Treat "writable ApplicationDataPath" as a hard precondition of
NGX init on Linux, surfaced through the same graceful-degradation path as a missing OFA
(§G6): if the dir can't be made writable, report the NGX/DLSS backend unavailable and fall
back (nvofg FG / native rendering) rather than failing the whole pipeline.

## 20. Path B — running the real Windows DLSS Frame Generation (`nvngx_dlssg.dll`) natively

**Motivation.** DLSS Frame Generation (DLSS-G) — the trained NN NVIDIA ships on Windows — has
**no native Linux runtime**. The driver (610.x) ships native ELF snippets for DLSS Super
Resolution (`libnvidia-ngx-dlss.so`) and Ray Reconstruction (`libnvidia-ngx-dlssd.so`) but
**not** for FG; DLSS-G exists only as a **Windows PE** at `/usr/lib/nvidia/wine/nvngx_dlssg.dll`.
Probed on an RTX 5070 / driver 610, the native Linux NGX host returns
`FrameGeneration.Available = 0`, `FeatureInitResult = 0xBAD00004` (`FAIL_FeatureNotSupported`),
`needsDriver = 0` — the native NGX host **deliberately does not expose FG**. "Path B" asks:
can a *native* Linux Vulkan process load and drive that Windows PE in-process, **without**
running the whole app under Proton/Wine?

### 20.1 Verdict — infeasible in practice (documented, not pursued)

Not strictly impossible, but effort is **XL** and the endpoint is *re-hosting most of Proton's
graphics stack in-process*, unsupported and legally exposed. The PE loader is the easy part and
misleads on total cost. **Single biggest blocker:** `nvngx_dlssg.dll` is not a self-contained
compute kernel — it is a Streamline/NGX component that expects to sit behind a **D3D12 device +
DXGI/Vulkan swapchain whose `vkQueuePresentKHR`/`vkAcquireNextImageKHR` it intercepts**, and it
**hard-requires NVIDIA Reflex** (`VK_NV_low_latency2`) present-metering to pace generated frames.
Satisfying that natively means standing up, in-process, exactly the Windows-PE shims Proton
provides — a Win32 surface, a D3D12 impl (= vkd3d-proton), NVIDIA's Windows NGX host
(`nvngx.dll`/`_nvngx.dll`), and dxvk-nvapi's Reflex plumbing. That is "build a single-DLL Wine",
which defeats the no-Wine mandate in spirit.

### 20.2 How DLSS-G runs on Linux today (the Proton chain)

PE = Windows PE running in Wine's address space; native = ELF `.so` in the NVIDIA Linux driver:

1. **Game `.exe` (PE)** — Wine's PE loader. FG integrated via **NVIDIA Streamline** (`sl.dlss_g`)
   or directly via NvAPI + NGX SDK.
2. **dxvk-nvapi (`nvapi64.dll`, PE)** — does *not* implement FG; forwards D3D12 DLSS/FG calls
   into vkd3d-proton and provides Reflex entry points (D3D12 via vkd3d-proton ≥ 2.12, or the
   native Vulkan layer `libdxvk_nvapi_vkreflex_layer.so` over `VK_NV_low_latency2`, R550+).
3. **NVIDIA NGX host (`nvngx.dll` + `_nvngx.dll`, PE)** — NVIDIA's Windows NGX runtime at
   `/usr/lib/nvidia/wine/`; performs the feature handshake and loads the FG snippet.
4. **`nvngx_dlssg.dll` (PE)** — the actual FG implementation; drives the OFA via
   `VK_NV_optical_flow` + compute, and manages generated-frame present via the hooked swapchain.
5. **vkd3d-proton** — provides the D3D12 device/queue/resources the FG code binds to → Vulkan.
6. **Native driver core** — `libnvidia-ngx.so`, the Vulkan driver implementing
   `VK_NV_optical_flow` and `VK_NV_low_latency2`, and kernel present-metering for Reflex.

**Structural fact:** the only native-ELF pieces are the driver, `libnvidia-ngx.so`, the Reflex
Vulkan layer, and vkd3d-proton's translation. **DLSS-G proper — the NGX host and the FG snippet —
is Windows PE.** (For DLSS *SR* the driver ships a native ELF snippet, which is why SR/RR work
natively and FG does not.)

### 20.3 What Path B would require (staged; difficulty S/M/L/XL)

| Stage | Work | Reuse | Diff | Risk |
|---|---|---|---|---|
| **S1** In-process PE loader (no Wine process) | map PE, relocations, TLS/`DllMain`, resolve imports vs shims; load 2 PEs (host + snippet) | taviso/`loadlibrary` proves the mechanism (x86-64→x86-64, no FEX/box64) | **M** | low–mod |
| **S2** Win32 shim surface | stub/emulate kernel32/UCRT/ntdll imports (heap, TLS, threads, events, module discovery); correct pthread/futex mapping (FG is multithreaded, syncs via binary semaphores); NGX host↔snippet version/signature checks | none clean (else link Wine) | **L** | moderate |
| **S3** D3D12/DXGI surface | give the DLL an `ID3D12Device`/`ID3D12CommandQueue`/`IDXGISwapChain` it can bind + hook; even the Vulkan FG path routes device/resources through the Streamline swapchain wrapper | vkd3d-proton — but it's authored as Win32 PE DLLs, not a native lib with a stable COM surface | **XL** | **very high** |
| **S4** Reflex / present-metering | implement the full Reflex marker protocol (`eReflexMarkerPresentStart/End`) over `VK_NV_low_latency2` so FG can meter real vs. generated frames | native `VK_NV_low_latency2`; dxvk-nvapi shows the pattern | **L** | high |
| **S5** NGX host + FG snippet handshake | make the **Windows** `nvngx.dll` host init and return success for `FrameGeneration` (the native host returns `0xBAD00004`); undocumented, versioned host↔snippet interface | the NVIDIA Windows PEs (PE only) | **XL** | **very high** |
| **S6** Feed it our Vulkan frame + backbuffer | hand FG color/depth/MV + a backbuffer it can present; cede present control to a DLL that wants to own the swapchain | our nvofg inputs | **L** | high |

**Cumulative:** S1 is small; the project is dominated by S3 + S5 (both XL — essentially "re-host
Proton's D3D12/NGX stack in-process"). You would ship vkd3d-proton + a Win32 shim + NVIDIA's
Windows NGX PEs inside a "native" process — a bespoke mini-Wine.

### 20.4 Blockers, ranked

1. **D3D12 device + swapchain-interception contract (S3)** — reproducing it without Wine ≈
   reproducing vkd3d-proton + Wine COM. *Biggest blocker.*
2. **NGX host exposes FG only via the Windows PE (S5)** — native host returns `0xBAD00004`; the
   only host that loads the snippet is the Windows `nvngx.dll`, which itself needs S2/S3.
3. **Mandatory Reflex present-metering (S4)** — FG won't pace without correct markers; timing is
   fragile even inside the fully-supported Proton path.
4. **Win32 shim volume (S2)** — individually easy, collectively converges on "be Wine".
5. **Legal exposure (§20.6).**

### 20.5 Sanctioned / native alternatives & roadmap signals

- **No native Linux DLSS-G today.** Streamline `sl.dlss_g` is Windows-only, shipped as prebuilt
  signed DLLs (not buildable from source); the DLSS-G guide lists Windows-only requirements.
- `nvpro-samples/vk_streamline` shows Reflex + DLSS SR + DLSS-G in Vulkan, but is Windows-oriented
  and inherits Streamline's Windows-only FG plugin; the sole Linux question on its tracker got no
  NVIDIA answer.
- **Native primitives DO exist on Linux:** `VK_NV_optical_flow` (OFA, Ampere+) and
  `VK_NV_low_latency2`/Reflex (R550+). So a *future native ELF FG snippet* is technically
  plausible — but there is **no public signal** NVIDIA plans one.
- **NVK/Mesa** gained experimental DLSS **SR** by importing CuBIN kernels via
  `VK_NVX_binary_import` — evidence that native snippet execution (not PE loading) is the
  sanctioned direction; it does **not** provide an FG route.

**Roadmap read:** the credible native path is "NVIDIA ships a native ELF DLSS-G snippet exposed by
the native NGX host", not "we load the Windows PE". Track driver release notes for a
`libnvidia-ngx-dlssg.so` and for `FrameGeneration` no longer returning `0xBAD00004`; file a
native-Linux DLSS-G request with NVIDIA.

### 20.6 Legal note

`nvngx_dlssg.dll` is governed by NVIDIA's RTX/DLSS SDK EULA + driver license: licensed for use on
NVIDIA GPUs, **may not be distributed stand-alone / without substantial value-add**, with
pre-release notification obligations. The file at `/usr/lib/nvidia/wine/` is distributed **for the
driver's Wine/Proton runtime**; loading it from a bespoke non-Proton in-process loader is a use
NVIDIA does not contemplate or document, and redistributing it with the app is very likely outside
the grant (cf. NVIDIA's strictness over NGX DLL redistribution, e.g. the Blender DLSS friction).
**Gray-to-red area — any shipping Path-B product needs explicit written clearance from NVIDIA.**
*Not legal advice.*

### 20.7 Decision

**Path B is documented and rejected for production.** It is XL, unsupported, undocumented, and
legally exposed, and it contradicts the native-no-Wine mandate in spirit even if the app's own
frame loop stays ELF. Frame generation on Linux stays **native** — nvofg (OFA + warp today),
with the quality gap to DLSS-G closed by a **native learned interpolator we train ourselves**
(§21, the `NVOFG_INTERP_CNN` path over `VK_KHR_cooperative_matrix`), not by hosting NVIDIA's
Windows DLL. Revisit only if NVIDIA ships a native ELF FG snippet.

*If empirical certainty on S1/S2 is ever wanted:* `objdump -p /usr/lib/nvidia/wine/nvngx_dlssg.dll`
(or `winedump`) to enumerate the real import table, and test whether the Windows `nvngx.dll` host
enforces a signature/driver check that rejects a non-Wine environment. Neither changes the S3/S5
verdict.

## 21. Training our own frame-generation model — what it would take

Path B (§20) is a dead end; the native way to close the quality gap to DLSS-G is to **train
our own learned interpolator** and run it on-device (`NVOFG_INTERP_CNN` over
`VK_KHR_cooperative_matrix`). The decisive insight: **we already have the expensive inputs**
(OFA optical flow via `VK_NV_optical_flow`, engine motion vectors, depth, reactive/UI masks),
so our model does **not** have to *estimate* motion — which is exactly what makes pure-RGB
video-frame-interpolation (VFI) nets big and slow. That collapses the model into the tiny,
real-time "rendering-aware" regime.

### 21.1 Honest verdict

An independent team can realistically reach **"very good — visibly better than classical warp,
close to DLSS-G on most scenes"**, *not* a guaranteed pixel-match of DLSS-G. The hard part
(dense motion) is done by the OFA in hardware; our network only does **synthesis + disocclusion
inpainting + ghosting/shading correction**, which is a small, well-understood net (0.4–1.1M
params in the literature). Matching DLSS-G's last-10% polish (proprietary training at scale, and
DLSS 4's transformer) is the genuinely hard part — but 80–95% of the perceived quality is
reachable natively and legally.

### 21.2 Interpolation vs extrapolation — the latency decision

| | Interpolation (DLSS 3 FG, FSR 3 FG) | Extrapolation (ExtraNet, GFFE, Intel ExtraSS) |
|---|---|---|
| Uses | prev **and** curr frame → in-between | only **past** frames → predict future |
| Latency | **+~1 source-frame (~10 ms @60fps)**; must hold the newest frame back; needs Reflex/Anti-Lag to claw back | **no added latency** (nothing held back) |
| Disocclusion | easier — the future frame usually *reveals* what was hidden | harder — must hallucinate newly-revealed regions |
| Quality | cleaner | artifact-prone in disoccluded/shadow/reflection regions |

DLSS-G itself is **interpolation** — but it only survives the latency hit because NVIDIA pairs it
with **hardware flip-metering + Reflex** pacing, which we do **not** have on Linux. Tellingly, the
entire *game* frame-generation research lineage — **ExtraNet** (NVIDIA/UCSB), **ExtraSS** (Intel),
**GFFE** (Intel) — is **extrapolation**, chosen specifically to avoid the hold-back latency. So the
recommendation is a **shared backbone with both modes**, shipping **extrapolation as the primary
game path**:
- **v1 = extrapolation** (predict next frame from past frames + MV/OFA): **no added latency**, no
  dependence on flip-metering, and it is exactly the ExtraNet/ExtraSS/GFFE design point that hits
  the real-time budget. The cost is harder disocclusion (no future frame) — mitigated by our depth +
  occlusion masks + G-buffer-guided hole-filling.
- **interpolation mode (quality/offline fallback)** reuses nvofg's *existing* bidirectional OFA flow
  (prev↔curr) and gives cleaner disocclusion; pair with `VK_NV_low_latency2` (Reflex, native on
  Linux R550+) for pacing. Good for cutscenes / latency-insensitive use.

### 21.3 Why engine inputs make the model 10–50× smaller (the key lever)

Pure-RGB VFI must learn optical flow, which dominates its size and cost. Rendering-aware nets get
motion **analytically** from MVs/OFA and shrink dramatically. Measured references:

| Model | Class | Params | Inference | Notes |
|---|---|---|---|---|
| RIFE | RGB VFI (efficient end) | 9.8M | ~31 ms @720p | IFNet flow estimator is the bulk |
| FILM | RGB VFI, large motion | ~tens of M | 0.39 s @720p | feature pyramid + flow + fusion U-Net |
| DAIN / CAIN | RGB VFI | 24–43M | 40–436 ms | |
| (standalone flow) RAFT / PWC-Net | flow only | 5.3M / 8.8M | — | this is the cost we *avoid* |
| **ExtraNet** | game extrapolation | **~1.1M** | 30.7 ms → **17.2 ms (TensorRT)** @1080p | G-buffer + reliable MVs, gated convs |
| **STSSNet / STSS** | game SS+extrapolation | **0.4M (417K)** | **4.4 ms** @1080p (1.8 ms TensorRT) | MV+depth, no learned flow |
| GFFE | G-buffer-free extrapolation | (n/r) | 6.6 ms @1080p (RTX 4070Ti S) | heuristic motion, not a heavy flow net |
| PatchEX | patch extrapolation | (n/r) | ~2.14 ms @1080p | |

*Sources:* RIFE ([arXiv 2011.06294](https://arxiv.org/abs/2011.06294)), FILM ([arXiv 2202.04901](https://arxiv.org/abs/2202.04901)),
STSS ([arXiv 2312.10890](https://arxiv.org/html/2312.10890v1)), ExtraNet
([SIGGRAPH Asia 2021](https://dl.acm.org/doi/10.1145/3478513.3480531), [code](https://github.com/fuxihao66/ExtraNet)),
GFFE ([arXiv 2406.18551](https://arxiv.org/abs/2406.18551)), ExtraSS
([SIGGRAPH Asia 2023](https://dl.acm.org/doi/fullHtml/10.1145/3610548.3618224)), PatchEX ([arXiv 2407.17501](https://arxiv.org/pdf/2407.17501)).

**We supply MV + depth + OFA flow + masks → we live in the 0.4–1.1M-param, 2–20 ms regime, not the
10–40M-param VFI regime.** That is the whole reason this is tractable natively.

### 21.4 Recommended architecture

A small **rendering-aware forward-warp + fusion network** (SoftSplat / ExtraNet / ExtraSS lineage) —
**~0.26–1.0M params** (ExtraNet is **~262K params** computed from its released weights):
- **Warp stage (no learned flow):** forward-warp prev frame(s) to the target phase via **softmax
  splatting** (Niklaus & Liu, CVPR 2020, [arXiv 2003.05534](https://arxiv.org/abs/2003.05534)) using
  our *given* OFA flow / MVs, resolving many-to-one collisions with **depth** as the splat weight.
  This is the key departure from RIFE/IFRNet, which spend most parameters *estimating* flow.
- **Inputs to the fusion net:** the warped color(s), forward+backward flow, occlusion/confidence
  (already produced by nvofg's REFINE stage), depth, engine motion vectors, the **OFA-vs-MV
  disagreement** map, a **disocclusion mask** (from fwd/bwd MV consistency), and the **reactive + UI
  masks** (never synthesize HUD/particles from geometry motion).
- **Backbone:** lightweight encoder–decoder with **gated convolutions** (ExtraNet: 11 lightweight
  gated-conv encoder layers + 7 conv decoder layers, plus a weight-shared **history encoder** for
  moving shadows/reflections MVs miss); skip connections; a **residual add** of the warped RGB
  stabilises training; output = the generated color that corrects ghosting and fills disocclusion
  holes the classical warp leaves.
- **Why both OFA flow and engine MVs (like DLSS-G):** MVs describe only *geometry* motion; the OFA
  captures *pixel* motion MVs miss — particles, reflections, shadows, lighting. DLSS-G's autoencoder
  fuses exactly these four (prev, curr, OFA flow, MV+depth); we mirror that fusion.
  ([NVIDIA DLSS 3](https://www.nvidia.com/en-us/geforce/news/dlss3-ai-powered-neural-graphics-innovations/))
- This is a **drop-in for `NVOFG_INTERP_CNN`**: it replaces the WARP stage's hand-tuned blend with a
  learned one; everything upstream (OFA execute, refine, masks) is unchanged.

### 21.5 Training data plan

- **Rendering-aware GT (primary):** render engine sequences at **2× the target frame rate** so every
  other frame is the ground-truth "in-between"; capture the **aux buffers per frame** (color, MV,
  depth, HUD/UI mask, reactive mask). For this project's consumer (RMC/Minecraft, handled by the
  other agent) that means an offline capture harness dumping color+MV+depth+mask sequences. Aim for
  tens of thousands of frame-triplets across varied motion (camera pans, fast entities, transparency).
- **Public VFI pretraining (bootstrap):** **Vimeo-90K** (~73k septuplets), **X4K1000FPS** (4K, extreme
  motion), **ATD-12K** (animation) — pretrain the synthesis net, then fine-tune on rendered data with
  real MV/depth. Public data lacks aux buffers, so it only warm-starts the RGB synthesis.
- Resolution: train at 540p–1080p crops; the net is resolution-independent (fully convolutional).

### 21.6 Losses

Charbonnier/L1 reconstruction **+ LPIPS/VGG perceptual** (sharpness) **+ census/warping loss**
(flow consistency) **+ light adversarial (GAN)** for micro-detail **+ temporal-stability loss**
(penalize flicker across consecutive generated frames). UI/reactive regions are **masked out** of
all losses — the model must not learn to interpolate HUD. Disocclusion regions get up-weighted
(that is where classical warp fails and where the perceived quality lives).

### 21.7 Training compute & real-time inference budget

- **Training:** models this small (0.4–1.1M params) train on a **single RTX 5070** in **days, not
  weeks** — order **10²–10³ GPU-hours** for a competitive v1 (RIFE-class nets were trained at
  single-GPU/small-cluster scale). No large cluster required for v1.
- **Inference target: <2–3 ms @1080p.** STSSNet already hits 4.4 ms naïve / **1.8 ms with TensorRT**
  at 1080p on an RTX 3090; on Blackwell (RTX 5070) with fp16 and **`VK_KHR_cooperative_matrix`**
  (Tensor Cores — our `src/spike/cuda_tensor.cu` / `cuda_vk_interop.cu` WMMA spikes prove the path),
  <2 ms is realistic. **Quantization headroom is large:** QW-Net (NVIDIA, SIGGRAPH Asia 2020) runs
  **~95% of a reconstruction net's compute in INT4** with quality above TAA — so INT8/INT4 coopmat
  paths are on the table for v2. ([QW-Net](https://dl.acm.org/doi/10.1145/3414685.3417786))

### 21.8 Staged roadmap

- **v0 — data + harness** (~2–4 weeks): a UE5/Unity capture pipeline rendering at 2× fps for GT
  intermediate frames + aux buffers (color pre/post-UI, MV, depth, UI/reactive masks); the offline
  PyTorch training harness; softmax-splat + disocclusion-mask reference ops. Deliverable: ~20k+ GT
  tuples across 5–10 scenes.
- **v1 — "beat the classical warp"** (~2–3 person-months, 1–2 eng): ~0.26–1M-param splat+fusion net,
  **extrapolation-first** (no latency). Pretrain on Vimeo-90K; fine-tune on rendered MV/depth/mask
  data. Losses: Charbonnier + census + VGG. Export fp16, run on `VK_KHR_cooperative_matrix`; target
  **<2–3 ms @1080p** (anchor: ExtraNet ~262K params, 8 ms@720p / 15.8 ms@1080p naïve TensorRT-fp16 on
  an RTX 3090; ExtraSS "Ours-ESS" **4.1 ms** at 540p→1080p; Blackwell + coopmat is faster).
  **Success bar:** visibly fewer disocclusion holes + less ghosting than the current warp, stable UI.
  High-confidence outcome (low bar, strong inputs). Effort: **L–XL** (data harness + training +
  coopmat inference kernel — the real work item).
- **v2 — "DLSS-G-adjacent"** (~9–18 person-months, 2–3 eng): add temporal-stability loss over
  sequences, FILM Gram + light GAN for sharpness, OFA-vs-MV handling for transparencies/shadows,
  **INT8/INT4 quantization** (QW-Net runs **~95% of a reconstruction net in INT4** above TAA quality —
  [SIGGRAPH Asia 2020](https://dl.acm.org/doi/10.1145/3414685.3417786)), per-title fine-tuning, and an
  **interpolation mode** with `VK_NV_low_latency2` pacing. **Goal: FSR3-competitive, DLSS-G-adjacent
  on most scenes** — honest residual gaps in fast disocclusion, transparencies/particles, and
  end-to-end latency (no flip-metering hardware). Effort: **XL**.

### 21.9 Bottom line

Because the OFA already provides the motion the big VFI nets spend their parameters estimating, a
**native, legal, on-device learned interpolator in the 0.4–1M-param / <3 ms class is the right way
to approach DLSS-G quality on Linux** — not hosting NVIDIA's Windows DLL. v1 is a concrete,
single-GPU-trainable milestone that slots into the reserved `NVOFG_INTERP_CNN` interpolator with no
public API change. Uncertainties to keep honest: DLSS-G's and QW-Net's exact params/ms are
unpublished (figures above are from comparable open academic nets), and matching DLSS 4's
transformer-grade polish is a stretch goal, not a v1 promise.
