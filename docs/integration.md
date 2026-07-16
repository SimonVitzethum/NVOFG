# Integrating nvofg

A minimal, current guide for wiring nvofg into a native Linux Vulkan engine. Mirrors
`design.md` §6 and reflects the AUTOMATIC-mode sync model (ADR 0003).

## Device requirements

At **device creation** the app must enable, in addition to whatever it already uses:

| Requirement | Why |
|---|---|
| Device extension `VK_NV_optical_flow` | the OFA backend. Get the list from `nvofg_required_device_extensions()`. |
| Feature `VkPhysicalDeviceOpticalFlowFeaturesNV.opticalFlow` | enables the OFA. |
| Feature `timelineSemaphore` (`VkPhysicalDeviceVulkan12Features` or core 1.3) | nvofg's cross-stage sync semaphore. |
| Feature `shaderStorageImageWriteWithoutFormat` | the warp stage writes the (arbitrary-format) output storage image. |
| One queue from the **optical-flow queue family** | the OFA executes there. Discover it with `nvofg_optical_flow_queue_family()`. |

nvofg calls `nvofg_query_support()` / `nvofg_create()` and returns `NVOFG_UNSUPPORTED`
if the OFA is absent, so the graceful 1× fallback (G6) is a single branch.

## Registered images

Allocate and register, at swapchain create / resize:

- **prev / curr color** — copies of the final frame nvofg reads. Create with at least
  `SAMPLED_BIT` (nvofg samples them). Register via `nvofg_register_color()`.
- **output** — a storage image nvofg writes the interpolated frame into. Create with
  `STORAGE_BIT` (+ whatever you need to present/copy it). Register via
  `nvofg_register_output()`.
- **aux (optional)** — depth / motion / ui_mask / reactive / material_id via
  `nvofg_register_aux()` (consumed from M2 on).

## Per frame

```c
NvofgGenerateInfo gi = {0};
gi.phase = 0.5f;                       // 2x
gi.prev_layout = /* layout prev color is in */;
gi.curr_layout = /* layout curr color is in */;
gi.input_timeline = colorsReadyTimeline;  // signalled when prev/curr finished rendering
gi.input_value    = colorsReadyValue;     // (VK_NULL_HANDLE handle => already ordered)

NvofgFrameSync sync;
if (nvofg_record_generate(ctx, &gi, &sync) == NVOFG_OK) {
    // present the interpolated image, waiting on (sync.semaphore, sync.value),
    // then present curr color.
}
```

nvofg owns all internal command buffers, layout transitions, and queue submits in
AUTOMATIC mode. The app only signals *colors ready* and waits on the returned timeline
point before presenting the interpolated frame. See `examples/headless_interp.cpp` for a
complete, self-contained driver.

## Resize

On swapchain recreation, call `nvofg_resize(ctx, w, h)` (or `FrameGen::resize`). It waits
for GPU idle, tears down the size-dependent resources, and clears all registrations —
then re-register color/aux/output with the new images before the next generate.

## Rust

```rust
use ash::vk;
use nvofg::*;

let of_family = optical_flow_queue_family(instance, pd, gipa).expect("no OFA");
// ... create the device with VK_NV_optical_flow, opticalFlow + timelineSemaphore +
// shaderStorageImageWriteWithoutFormat features, and a queue from `of_family` ...

let ci = CreateInfo {
    instance, physical_device: pd, device, queue, queue_family_index,
    of_queue, of_queue_family_index: of_family,
    get_instance_proc_addr: gipa,
    extent: vk::Extent2D { width, height },
    quality: Quality::High, interpolator: Interpolator::Warp, mode: Mode::Automatic,
    flags: flags::BIDIRECTIONAL | flags::USE_MOTION | flags::USE_UI_MASK,
};
let Some(mut fg) = FrameGen::new(&ci)? else { /* run at 1x */ return Ok(()); };
fg.register_color(&prev, &curr)?;
fg.register_aux(&Aux { ui_mask: Some(ui), motion: Some(mv), ..Default::default() })?;
fg.register_output(&out)?;

let sync = fg.record_generate(&GenerateInfo {
    phase: 0.5,
    input: Some(FrameSync { semaphore: colors_ready, value: colors_ready_val }),
    ..Default::default()
})?;
// present `out` waiting on (sync.semaphore, sync.value), then present curr.
```

Build the C++ library first (`cmake -S . -B build && cmake --build build`); the crates
link it via `NVOFG_LIB_DIR` (default `<repo>/build`). See
`crates/nvofg/examples/support.rs`.

## Frame pacing
Cap base fps to `displayHz / multiplier` and keep VSync off. Added latency ≈ one base
frame; there is no Reflex on native Linux.

The optional **pacer** (`nvofg_pacer.h`, design §7) evenly spaces the `[gen] real …`
sequence so it doesn't present in bursts. It touches no Vulkan — call it right before
each `vkQueuePresentKHR` (real or generated):

```c
NvofgPacerConfig pc = { .display_hz = 120.0, .multiplier = 2 };
NvofgPacer* pacer = nvofg_pacer_create(&pc);   // cap base fps to nvofg_pacer_base_fps_cap()
// per presented frame:
nvofg_pacer_wait_slot(pacer);   // blocks until the next evenly-spaced display slot
vkQueuePresentKHR(...);
// on a hitch / resize / vsync change: nvofg_pacer_reset(pacer);
```

Or, with `VK_KHR_present_wait`, read `nvofg_pacer_next_slot_time()` and pass it as the
present target instead of sleeping.

## Portability tiers
- **Tier A (default, NVIDIA):** hardware OFA via `VK_NV_optical_flow`.
- **Tier B (any Vulkan GPU):** portable block-match shader flow — lower quality, no OFA.
  Set `NVOFG_FLAG_FORCE_SHADER_FLOW`, or nvofg auto-selects it when no OFA is present.
- **Optional CUDA Tensor-Core backend** (`NVOFG_ENABLE_CUDA`, behind the CNN
  interpolator boundary): see ADR 0004.
