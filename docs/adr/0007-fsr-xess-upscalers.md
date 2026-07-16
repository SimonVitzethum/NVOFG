# ADR 0007 — FSR & XeSS upscaling backends (RenderFX)

- **Status:** Accepted — FSR **functional** (portable); XeSS **integrated + compile-verified**,
  gated on a native runtime Intel does not ship for Linux today.
- **Date:** 2026-07-16
- **Relates to:** design.md §16 (backend strategy), §18 (stages/dispatch); ADR 0006 (NGX);
  RenderFX ROADMAP.

## Context

After the official NGX backends (DLSS SR/DLAA/RR — ADR 0006), the next vendor upscalers are
**AMD FSR** and **Intel XeSS**. Both slot into the existing Upscaling stage
(`RFX_BACKEND_FSR`, `RFX_BACKEND_XESS`) with no new API surface — the capability table,
resolver, cross-stage family constraints and dispatch already model them. The question for
each is only *how it is implemented on native Linux* and *whether it can run here*.

## Decision

### FSR — a portable, vendor-neutral spatial upscaler (functional)

FSR is delivered as an **independent implementation of the AMD FSR1 / EASU approach** in
Slang (`src/shaders/upscale_fsr.slang`), not a link against AMD's SDK:

- Per destination pixel, reconstruct from a 4×4 source neighbourhood with an **anisotropic
  Lanczos-2** kernel whose lobe is stretched *along* the local edge (Sobel-estimated) and
  tightened *across* it — directional, edge-preserving reconstruction (the EASU idea).
- **EASU deringing:** clamp the result to the nearest-2×2 min/max, so the sharpening
  negative lobes never overshoot (no ringing). Verified on a ramp: stays monotonic + smooth.

**Why re-implement rather than vendor the SDK:** AMD's FidelityFX-SDK is a large,
Windows/DX-oriented CMake project; the public release zips are Windows binaries and the
FSR1 kernels are HLSL. A focused Slang EASU is *portable to any Vulkan GPU with no SDK*,
matches the project's Slang shader pipeline, and is fully testable here. This is honestly
labelled "FSR-style edge-adaptive spatial upscaler (EASU)", not AMD's bit-exact kernel.
FSR is spatial, so its `required_inputs` is **color only** (no motion/depth/jitter). RCAS
sharpening and an FSR2/3 *temporal* variant (motion+depth+jitter) are documented follow-ups.

### XeSS — official integration, gated on a native runtime (unsupported on Linux today)

XeSS is integrated against **Intel's official `xess_vk.h` headers** (vendored under
`renderfx/vendor/xess/`, git-ignored). `src/xess.cpp` implements the full path —
`xessVKCreateContext` at `rfx_create` (probes support → capability flag), lazy
`xessVKInit` per output resolution, and `xessVKExecute` from `rfx_record_upscaling`
(color+motion+depth+jitter, history reset) — behind `-DRENDERFX_XESS`, with inert stubs
otherwise (identical pattern to `ngx.cpp`).

**Environment reality:** Intel's public XeSS SDK (v3.0.1, the latest) ships a **Windows-only
runtime** — `libxess.dll` / `libxess.lib`; **no XeSS release contains a native-Linux
`libxess.so`** (verified: no ELF in the SDK, no Linux asset across all `intel/xess`
releases). On Linux, XeSS is otherwise reachable only through Proton (the Windows DLL under
Wine) or bundled inside the Intel Arc driver stack for Intel GPUs — neither is a standalone
library an app can link on this NVIDIA host. This project is **native-Linux with no
Wine/Proton**, so the Windows DLL is out of scope.

Therefore XeSS **reports `unsupported`** here and the resolver falls back — the same
graceful-degradation contract (G6) as NGX-when-absent. The integration is nonetheless kept
**honest and drop-in ready**:

- `src/xess.cpp` is **compile-verified against Intel's official headers** every build via
  the `renderfx_xess_apicheck` object target (real path, `-DRENDERFX_XESS`, no link) — so
  the code stays API-correct and does not bit-rot.
- It becomes functional with **no code change** wherever a native XeSS runtime exists
  (an Intel Arc Linux host, or if Intel ships a Linux `.so`): configure `-DRENDERFX_XESS
  -DRFX_XESS_LIB=<path/to/libxess.so>` and `rfx_query_capabilities` flips XeSS to
  `supported`.

## Consequences

- **Positive:** FSR gives every Vulkan GPU a real edge-adaptive upscaler now, with zero SDK
  dependency. XeSS is wired end-to-end and stays compile-correct against the vendor API, so
  enabling it later is a build-flag + runtime step, not an integration. No API surface was
  added for either — both are ordinary Upscaling-stage backends.
- **Negative / open:** XeSS is non-functional on native Linux until a native runtime ships —
  by necessity, not by design. When it runs, note XeSS wants inputs in
  `SHADER_READ_ONLY_OPTIMAL` and output in `GENERAL` (vs RenderFX's GENERAL-everywhere
  record contract) — a layout adaptation for the host that provides the runtime.

## Alternatives considered
- **Ship XeSS via the Windows DLL under Wine/Proton** — rejected: violates the native-Linux,
  no-Wine mandate (design.md §14).
- **Vendor AMD's FidelityFX-SDK and link FSR2/3** — rejected for now: heavy Windows/DX-
  oriented build; the portable Slang EASU delivers a working spatial upscaler immediately.
  A temporal FSR2/3 backend remains a future option behind the same `RFX_BACKEND_FSR`
  family (or a dedicated id) if a clean Linux/Vulkan path appears.
