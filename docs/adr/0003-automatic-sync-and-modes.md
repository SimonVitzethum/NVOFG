# ADR 0003 — Automatic sync model & integration modes

- **Status:** Accepted
- **Date:** 2026-07-16
- **Amends:** `design.md` §4 threading/sync contract; ADR 0002 §1.

## Context

`design.md` frames the hot path as "nvofg only records into the app's single
`cmd`, except the one OFA execute the driver enqueues on `queue`." Implementation
revealed this cannot hold as written:

- `vkCmdOpticalFlowExecuteNV` must be recorded onto an **optical-flow-capable queue
  family** (index 5 on the reference GPU). It is invalid to record it into a
  graphics/compute command buffer, so the flow stage inherently needs its own queue
  submission and cross-queue timeline synchronisation.
- Consequently nvofg must also know **when the color images are render-complete**
  before the prep pass reads them — information the original `GenerateInfo` did not
  carry.

The project owner's directive: nvofg must stay a **drop-in library**, not a
framework. Engines must not be forced to select queues, own timeline semaphores,
know layouts, or place barriers. Priorities (quality > stability > API > perf)
justify extra internal submits in exchange for an encapsulated, hard-to-misuse API.

## Decision

### Integration modes (`NvofgMode`)
- **`NVOFG_MODE_AUTOMATIC` (default, v1):** nvofg fully owns the pipeline. It selects
  the compute and optical-flow queues, allocates and records its own command
  buffers, performs all layout transitions and queue-family ownership transfers, and
  submits the internal work itself — timeline-chained:

  1. wait on the app's **input** timeline point (colors ready),
  2. prep (luma) on the compute queue,
  3. OFA execute (+ flow/cost copy) on the optical-flow queue,
  4. refine → interpolate → composite on the compute queue,
  5. signal an **output** timeline point.

  `nvofg_record_generate` returns that output point in `NvofgFrameSync`; the app
  waits on it before presenting. The call submits but never blocks the CPU.

- **`NVOFG_MODE_EXTERNAL_COMMANDS` (reserved):** a future expert mode letting an
  engine pass its own command buffer / drive sync, for studios that want to fold the
  compute stages into their own submits. Declared now so adding it later is
  additive, not a breaking change. Not implemented in v1.

### API additions (`NvofgGenerateInfo`)
```c
VkSemaphore   input_timeline;   // app signals this when colors are ready
uint64_t      input_value;      // ... at this value (VK_NULL_HANDLE handle => skip wait)
VkImageLayout prev_layout;      // layout prev color is in when ready
VkImageLayout curr_layout;      // layout curr color is in when ready
VkCommandBuffer cmd;            // EXTERNAL_COMMANDS only (reserved); ignored otherwise
```
The old always-present `cmd` field is demoted to expert-mode-only. `NvofgCreateInfo`
gains a `mode` field (defaults to AUTOMATIC when zero-initialised).

## Consequences
- **Integration is ~"tell me colors are ready, wait on my result."** No engine-side
  queue/semaphore/layout/barrier knowledge required — the stated mandate.
- **Cost:** up to three internal submits per generated frame (compute / OF / compute).
  Acceptable under the priority order; a later optimisation can coalesce where a
  single queue family covers multiple stages, and EXTERNAL_COMMANDS can eliminate
  submits entirely for engines that want that.
- The output-only latency/pacing behaviour (design.md §3) is unchanged.

## Alternatives considered
- **App records prep, nvofg owns OFA+post.** Fewer nvofg submits, but pushes a
  nvofg-provided prep pass and a timeline signal into every integrator — more surface
  area, exactly what the library-not-framework mandate rejects. Preserved conceptually
  as the future EXTERNAL_COMMANDS mode for engines that opt in.
- **Keep the single-cmd model.** Impossible for the cross-queue OFA execute; would
  require a hidden internal submit anyway with none of the encapsulation benefit.
