# Path B — native DLSS Frame Generation: reconnaissance & plan

Goal: run NVIDIA's real DLSS Frame Generation NN on **native Linux** by loading the Windows
snippet `nvngx_dlssg.dll` in-process, **without** running the app under Proton/Wine. Rationale
vs. training our own model (design.md §21): Path B **scales to future DLSS FG versions** for free
(newer `nvngx_dlssg.dll` drops in). See design.md §20 for the full plan; this is the working log.

The DLL is **not vendored** — it ships with the NVIDIA driver at
`/usr/lib/nvidia/wine/nvngx_dlssg.dll` (Windows PE32+, NVIDIA-licensed; do not redistribute).

## S0 — Reconnaissance (done)

Driver 610.43.03, files under `/usr/lib/nvidia/wine/`:

| File | Size | Role |
|---|---|---|
| `nvngx_dlssg.dll` | 9.29 MB | **the FG snippet** — the actual Frame Generation implementation |
| `nvngx.dll` | 0.48 MB | thin NGX loader-host (imports only KERNEL32) |
| `_nvngx.dll` | 1.99 MB | heavy NGX core host (crypto/COM/winsock/shell) |

### Import surface — the key finding

**`nvngx_dlssg.dll` imports only Vulkan + CUDA + MSVC-CRT. No D3D12, no DXGI, no COM.**

| Imported DLL | # funcs | Native equivalent |
|---|---|---|
| KERNEL32.dll | 143 | MSVC C-runtime surface (heap/TLS/SEH/locale/file/sync) — shim (S2) |
| **nvcuda.dll** | 26 | **native `libcuda.so.1`** — incl. Vulkan interop (`cuImportExternalMemory`, `cuImportExternalSemaphore`, `cuSignal/WaitExternalSemaphoresAsync`, `cuMemcpy2DAsync`) |
| **vulkan-1.dll** | 22 | **native `libvulkan.so.1`** |
| VERSION.dll | 3 | trivial stub (`GetFileVersionInfo*`, `VerQueryValueA`) |
| ADVAPI32.dll | 3 | trivial stub (`RegOpenKeyExW`/`RegQueryValueExW`/`RegCloseKey` — reads DRS settings) |
| USER32.dll | 1 | trivial stub (`GetWindowThreadProcessId`) |

So the FG snippet is effectively **a CUDA + Vulkan compute module wrapped in the MSVC C runtime**:
the NN runs on CUDA, sharing memory/semaphores with Vulkan via external-memory/semaphore interop.
The nasty surface (CRYPT32/bcrypt crypto, WS2_32 winsock, ole32 COM, SHELL32) lives **only in the
host `_nvngx.dll`** — not in the snippet.

## S1 — Native PE map + relocate + import-resolve (done — `pe_probe.c`)

`pe_probe.c` maps the PE image, applies base relocations, and resolves imports against the native
drivers. Result on this machine:

```
machine=0x8664 sections=9 imageBase=0x180000000 sizeOfImage=9474048 dirs=16
mapped @ <anon>  relocations applied=3770
native: libvulkan.so.1=ok  libcuda.so.1=ok
Vulkan (vulkan-1.dll):  resolved 22, missing 0
CUDA   (nvcuda.dll):    resolved 26, missing 0
Win32 CRT stubs needed: 150  (imports by ordinal: 0)
```

**Milestone reached:** the 9 MB PE maps + relocates cleanly in a native Linux process, and **all 48
driver-facing imports (Vulkan + CUDA) bind to the real native drivers**. The entire remaining
surface is **150 standard MSVC-CRT symbols** (no exotic APIs, no ordinals) plus a PE-aware CRT/SEH
init to run `DllMain`. The probe does **not** execute the DLL (safety) — it stops at "mapped +
imports wired".

## Remaining stages (design.md §20.3, re-scoped by the recon)

- **S2 — MSVC-CRT shim (M, was L).** Provide the 150 KERNEL32/ADVAPI32/USER32/VERSION symbols +
  SEH (`RtlUnwindEx`/`RtlLookupFunctionEntry`/`RtlVirtualUnwind`) + TLS + CRT startup, then call
  `DllMain`. Reuse options: **Wine's `ucrtbase`/`kernel32`** as libraries, or **taviso/loadlibrary**
  (already stubs most of this for AV DLLs), rather than hand-writing all 150.
- **S3 — D3D12/DXGI surface. RE-SCOPED: likely NOT needed for the snippet.** The snippet imports
  Vulkan, not D3D12/DXGI. The swapchain/present interception the DLSS-G *Vulkan* API describes is
  driven at the **host/Streamline** layer; the snippet itself takes Vulkan images + CUDA. To be
  confirmed once DllMain runs and we see what handshake it expects.
- **S5 — the real crux: the NGX host handshake.** The snippet is normally loaded and driven by the
  NGX host (`nvngx.dll`/`_nvngx.dll`, or the native `libnvidia-ngx.so`). The native host returns
  `FrameGeneration = FAIL_FeatureNotSupported (0xBAD00004)` and loads only ELF snippets. Options:
  (a) load the **Windows host PEs** too (drags in the crypto/COM/winsock surface of `_nvngx.dll`);
  (b) reverse the **host↔snippet entry interface** and drive the snippet with a **minimal custom
  host** (avoids `_nvngx.dll`, but the interface is undocumented/versioned). This is the make-or-
  break stage.
- **S4 — Reflex present-metering** and **S6 — feed our Vulkan frame/backbuffer**: unchanged.

## Next step

S2: stand up the CRT shim (evaluate Wine `ucrtbase`/`kernel32`-as-library vs taviso/loadlibrary),
get `DllMain` to return, and log the first NGX/host entry points the snippet looks for — which
directly probes the S5 handshake. Own-model (§21) stays the parallel fallback (5080 server, 100–200 h).

## Legal

`nvngx_dlssg.dll` is NVIDIA-licensed and shipped for the driver's Wine/Proton runtime. This
experiment loads the on-disk driver file in place; it does **not** copy or redistribute it. Any
shippable product built on Path B needs explicit NVIDIA clearance (design.md §20.6).
