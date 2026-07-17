# ADR 0008 — Path B: native in-process loader for DLSS Frame Generation (`nvngx_dlssg.dll`)

- **Status:** Accepted — **experiment underway, strongly positive.** S0–S2 (snippet loads + DllMain
  native) + S5(a) (host loads, API reachable) + **S5(b): `NVSDK_NGX_VULKAN_Init_ProjectID` runs
  natively end-to-end via ms_abi↔SysV Vulkan/CUDA thunks + a live VkDevice, returning a clean
  `0xBAD00002` FAIL_PlatformError — failing ONLY because nvapi64.dll isn't bridged**. The single
  remaining dependency to a successful Init is an **nvapi shim** (dxvk-nvapi role). Own trained model
  (§21) kept as the parallel fallback.
- **Date:** 2026-07-17
- **Relates to:** design.md §20 (Path B plan), §21 (own model); ADR 0006 (NGX).

## Context

DLSS Frame Generation has no native Linux runtime — it ships only as the Windows PE
`/usr/lib/nvidia/wine/nvngx_dlssg.dll`, and the native NGX host returns
`FrameGeneration = FAIL_FeatureNotSupported (0xBAD00004)`. §20 planned "Path B" (load that PE in a
native Linux process without Proton) and rated it *infeasible in practice*. The user chose to
**pursue B anyway**, because it **scales to future DLSS FG versions** for free (a newer snippet drops
in), with an own trained model (§21) as fallback. This ADR records the decision to run Path B as a
staged, empirically-gated experiment under `experiments/dlssg-native/`.

## Decision

Pursue Path B in stages, each gated by an empirical result, and **stop the moment a stage proves a
hard blocker** (rather than sinking XL effort blind). The own-model track (§21) stays alive in
parallel as the fallback. The Windows DLL is **never vendored/redistributed** — the experiment loads
the on-disk driver file in place (legal posture per §20.6; product use needs NVIDIA clearance).

## What the recon changed (the reason to continue)

The §20 plan assumed the snippet dragged in D3D12/DXGI/COM (→ "re-host Proton in-process", XL). The
S0/S1 recon **disproved that for the snippet**:

- `nvngx_dlssg.dll` imports **only Vulkan (22) + CUDA (26) + MSVC-CRT (KERNEL32 143) + 7 trivial
  Win32** (VERSION/ADVAPI32/USER32). **No D3D12, no DXGI, no COM** in the snippet.
- `pe_probe.c` maps + relocates the 9 MB PE natively (3770 relocations) and **resolves all 48
  Vulkan+CUDA imports against native `libvulkan.so.1` / `libcuda.so.1`**. The snippet is a
  CUDA+Vulkan compute module (NN on CUDA, shared with Vulkan via external memory/semaphores).
- The only remainder to *load* the snippet is **150 standard MSVC-CRT symbols + a PE-aware CRT/SEH
  init** — a bounded, reusable shim (Wine `ucrtbase`/`kernel32` or taviso/loadlibrary).

So the **snippet-loading** half is now demonstrably tractable.

**S2 update — it loads and runs.** `pe_load.c` executes the snippet's `DllMain(DLL_PROCESS_ATTACH)`
natively and it **returns TRUE** (3/3 runs). Required, beyond S1: executable section protections;
the **MS-x64 ABI boundary** (`ms_abi` shims); a **~40-stub MSVC-CRT shim** with `GetProcAddress`
returning callable stubs for the CRT's optional-API probes; and a **fake Windows TEB on `gs`** +
**PE-TLS** via `arch_prctl(ARCH_SET_GS)` (the Wine TEB mechanism — the `gs:[0x58]` fault was the last
blocker). No D3D12/DXGI/COM, no exotic dependency.

**The crux is now S5**: the snippet exports **0 named functions**, so the host↔snippet interface is
an **internal, undocumented NGX ABI** — either load the Windows host PEs (`_nvngx.dll` adds
crypto/COM/winsock, all CRT-shimmable) so `CreateFeature(FrameGeneration)` reaches the native
snippet, or reverse the entry ABI and drive it from a minimal custom host. That, plus S4 Reflex
present-metering, is where Path B still may die.

## Consequences

- **Positive:** the biggest assumed blocker (D3D12/COM in the snippet) is gone; S1 is a concrete,
  reproducible milestone; the effort is now honestly scoped to S2 (CRT shim) → S5 (host handshake).
- **Negative / open:** S5 is undecided and could still be fatal; S4 latency/pacing remains; the
  legal posture forbids shipping without NVIDIA clearance. If S5 blocks, we fall back to §21.
- **No product coupling yet:** this lives entirely under `experiments/dlssg-native/` and touches no
  nvofg/RenderFX shipping code.

## Alternatives
- **Own trained model (§21)** — kept as the parallel fallback; more practical to *ship*, does not
  scale to future DLSS versions for free, needs training compute (a 5080 server, ~100–200 h).
- **Proton/Wine** — rejected (native-Linux, no-Wine mandate).
