# ADR 0001 — Drive the OFA through `VK_NV_optical_flow`, not the NvOF SDK

- **Status:** Accepted
- **Date:** 2026-07-16
- **Deciders:** project owner + implementation
- **Supersedes:** the `dlopen(libnvidia-opticalflow.so)` + vendored NvOF SDK approach
  described in `design.md` §2, §4, §9, §10.

## Context

`design.md` specifies reaching the NVIDIA Optical Flow Accelerator (OFA) by
`dlopen`-ing `libnvidia-opticalflow.so` and calling `NvOFAPICreateInstanceVk`, with
the `NV_OF_*_VK` structs taken from NVIDIA's **Optical Flow SDK** headers
(`nvOpticalFlowCommon.h`, `nvOpticalFlowVulkan.h`), vendored under `third_party/nvof/`.
That design carried three explicit risks (`design.md` §9): header licensing, ABI /
version drift, and the general fragility of a hand-bound private `.so` ABI.

During M0 bring-up we verified that the **same OFA silicon is exposed as a standard,
registered Vulkan extension — `VK_NV_optical_flow`** — on the target machine
(RTX 5070 Laptop, driver 610.43.03). The extension is part of `vulkan_core.h`; no
vendored headers are required. The M0 probe (`src/spike/probe.cpp`) creates a
`VkDevice` with the extension, enables the `opticalFlow` feature, finds a dedicated
optical-flow queue family, and enumerates the OFA image formats — using **only** the
Vulkan loader.

## Decision

**nvofg drives the OFA exclusively through `VK_NV_optical_flow`.** The library does
**not** `dlopen` `libnvidia-opticalflow.so` and does **not** use the proprietary NvOF
SDK API or headers.

Concretely nvofg uses:

| Purpose | API |
|---|---|
| Capability query | `vkGetPhysicalDeviceFeatures2` / `…Properties2` with `VkPhysicalDeviceOpticalFlowFeaturesNV` / `…PropertiesNV` |
| Supported formats | `vkGetPhysicalDeviceOpticalFlowImageFormatsNV` |
| Session lifetime | `vkCreateOpticalFlowSessionNV` / `vkDestroyOpticalFlowSessionNV` |
| Resource binding | `vkBindOpticalFlowSessionImageNV` |
| Execute | `vkCmdOpticalFlowExecuteNV` (recorded into a command buffer) |

A key structural consequence: with the SDK, "execute" was a driver-side enqueue onto a
`VkQueue` synchronised via fence points. With the extension, **execute is a normal
command-buffer command** (`vkCmdOpticalFlowExecuteNV`) recorded onto an
optical-flow-capable queue. This fits nvofg's "app records into a command buffer"
contract far more naturally and removes the need for the library to own a hidden submit.

## Consequences

### Positive
- **Eliminates all three §9 risks at once:** no NVIDIA headers to license/redistribute,
  no private ABI to drift against, no `dlopen`/symbol-resolution fragility. The only
  runtime dependency is the Vulkan loader + the user's driver.
- **In-band with the app's device.** Sessions, bindings, and execute all use the app's
  `VkDevice`, `VkImageView`s, and command buffers — genuinely zero-copy and GPU-driven.
- **Cleaner sync.** Execute is a recorded command; cross-queue ordering uses ordinary
  timeline semaphores (§4 sync contract) instead of SDK fence points.
- **Simpler tree.** `third_party/nvof/` is not needed; `nvofg-sys` does not bind a
  private ABI. Capability enums come straight from `vulkan_core.h`.
- **Honest licensing.** The library ships no NVIDIA proprietary code or headers, so the
  MIT/Apache-2.0 dual license is unencumbered.

### Negative / risks
- **Extension availability.** A driver could ship the `.so` but not the extension. On
  modern drivers (≥ 5xx) the extension is present; older drivers are out of scope for
  v1. `nvofg_query_support` reports absence cleanly (→ `NVOFG_UNSUPPORTED`, app runs 1×),
  which is the same graceful-absence path we already require (G6).
- **Feature-parity uncertainty.** The SDK historically exposed knobs (ROI lists,
  performance presets, hint semantics) that we must confirm are all reachable through
  the extension's `VkOpticalFlowSessionCreateInfoNV` / `VkOpticalFlowExecuteInfoNV`.
  Initial inspection shows parity for the features we need — grid size, hints, cost,
  bidirectional flow, global flow, ROIs (`maxNumRegionsOfInterest=32` on this GPU).
  **Per project directive:** if a needed capability turns out to be SDK-only, that gap
  will be documented in detail here with justification before any exception is made. As
  of M0 no such gap is known.

### Neutral
- The public C ABI (`nvofg.h`) is unchanged by this decision — it never exposed the SDK.
  Only the internal implementation strategy changes relative to `design.md`.

## Alternatives considered

1. **dlopen + NvOF SDK (design.md original).** Rejected: carries the licensing/ABI/
   fragility risks above with no compensating benefit now that the extension is present.
2. **Extension-first with a dlopen SDK fallback.** Rejected for v1: doubles the backend
   surface and reintroduces the SDK licensing question for a fallback that current
   drivers don't need. May be reconsidered only if a concrete, needed SDK-only feature
   gap is found and documented.

## Evidence
- `src/spike/probe.cpp` output on RTX 5070 Laptop / driver 610.43.03:
  `opticalFlow=true`, `hintSupported=1`, `costSupported=1`,
  `bidirectionalFlowSupported=1`, `globalFlowSupported=1`, output grids 1×1/2×2/4×4,
  resolution up to 8192×8192, dedicated optical-flow queue family.
- See `docs/hardware-capabilities.md` for the full capability dump and its implications.
