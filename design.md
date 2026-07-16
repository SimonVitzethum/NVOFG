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
