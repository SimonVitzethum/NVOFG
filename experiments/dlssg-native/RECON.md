# Path B — native DLSS Frame Generation bridge (public log)

High-level, legal-safe log of the Path B interoperability experiment. **Concrete NVIDIA-internal
details** (function offsets, struct layouts, gate sequences, disassembly) are deliberately kept out
of this file and live in **`INTERNALS_PRIVATE.md`** (git-ignored, never published) — see the
Compliance section below for why.

## Goal

Make NVIDIA's on-disk Frame-Generation module (`nvngx_dlssg.dll`, shipped with the Linux driver for
its Wine/Proton runtime) run in a **native Linux process, no Wine/Proton**, driven by the NGX Vulkan
API, so our independently-created library (nvofg/RenderFX) can interoperate with it. NVIDIA binaries
are **loaded in place from the installed driver, never copied or redistributed** (enforced by
`.gitignore`: `*.dll`/`*.exe`).

## Method

- A native PE loader maps + relocates the driver's PE modules, resolves their Vulkan/CUDA imports to
  the native `libvulkan.so.1`/`libcuda.so.1` via a MS-x64 ↔ SysV ABI thunk, and provides a
  Win32/CRT shim so the modules initialise (`DllMain → TRUE`).
- **Proton is used only as an oracle/reference**, never shipped: running the same driver DLLs under
  GE-Proton lets us *measure* the correct behaviour (via `WINEDEBUG` channels, `DXVK_NVAPI_LOG_LEVEL`
  traces, and a small nvapi-dump probe) instead of guessing. The working method is **differential
  boundary diffing**: instrument the same boundary natively and under Proton, diff the inputs, fix
  the first divergent value, repeat. Most fixes turned out to be single values (an adapter LUID, an
  nvapi data field, a module-path lookup, an OS-version report).

## Milestones

- **Feasibility (Gate 3) GREEN.** The native Linux driver *does* report Frame Generation available
  when queried through the complete stack — the "no" was never predetermined.
- **Snippet loads + initialises natively** (`DllMain → TRUE`), driven by the native NGX Vulkan API.
  The load path is gated by a registry lookup, file checks, and an Authenticode verification of the
  genuine signed driver file.
- **Capability phase COMPLETE.** `GetFeatureRequirements(FrameGeneration)` returns, **byte-identical
  to the Proton oracle, `FeatureSupported = 0` (fully supported)**, with the same MinHWArch/MinOS.
  The native NGX host + snippet report DLSS Frame Generation as fully supported, entirely natively.
  Key measured fixes: a synthesised-but-consistent adapter LUID (the native driver reports the
  Vulkan `deviceLUIDValid=0`; DXVK synthesises one), an nvapi arch-data field, a self-module path
  lookup, and an OS-version report — each one value, each measured against the oracle.
- **Functional phase — in progress.** The full `Init` path is a *sequence* of small string/path/
  config shim fixes (each a field NGX expects Wine to have populated). One Init blocker cleared so
  far; the next is mapped. After Init come `CreateFeature` (model weights + CUDA↔Vulkan interop) and
  `EvaluateFeature` (needs a float/SSE-aware ABI thunk — the current thunk is INTEGER-class only),
  then present-pacing (`VK_NV_low_latency2`, which is ordinary native Vulkan). See
  `FUNCTIONAL_PHASE_PLAN.md` for the staged plan and the Path B vs own-model decision.

## Compliance posture (see also the project's legal assessment)

- The reverse engineering is **for interoperability** of an independently created program, which in
  Germany is permitted by **§ 69e UrhG**; the EULA's reverse-engineering ban is **void to that
  extent** under **§ 69g Abs. 2 UrhG**, and **§ 3 Abs. 1 Nr. 2 GeschGehG** permits disassembling a
  lawfully acquired product. The driver is licensed and loaded in place.
- **§ 69e Abs. 2 Nr. 2** forbids passing the obtained information to third parties beyond what
  interoperability requires. Therefore the concrete NVIDIA internals live in `INTERNALS_PRIVATE.md`,
  which is **git-ignored and must never be published, committed publicly, or redistributed**.
- Path B is retained as **private interoperability research**, not a distributed product. Shipping it
  as an end-user tool raises further issues (the §69e Abs. 2/3 purpose-and-exploitation limits, the
  Authenticode bypass, end-user licensing) and is out of scope. The shippable product path is the
  **own native model** (see `PLAN1_OWN_MODEL.md`), which loads no NVIDIA binary at all.

## Legal

`nvngx_dlssg.dll` is NVIDIA-licensed and shipped for the driver's Wine/Proton runtime. This
experiment loads the on-disk driver file in place; it does **not** copy or redistribute it. Any
shippable product built on Path B needs explicit NVIDIA clearance.
