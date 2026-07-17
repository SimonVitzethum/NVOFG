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

## S2 — run DllMain natively (DONE — `pe_load.c`)

`pe_load.c` extends S1 with everything needed to *execute* the DLL and reached a decisive result:
**`nvngx_dlssg.dll`'s `DllMain(DLL_PROCESS_ATTACH)` runs and returns `TRUE` — natively on Linux,
no Wine/Proton — reproducibly (3/3 runs, exit 0).** What it took:

1. **Executable section protections** (`mprotect` per section `Characteristics`).
2. **The Microsoft-x64 ABI boundary** — a PE calls its imports MS-x64 (rcx/rdx/r8/r9 + 32B shadow),
   not SysV. Every shim is `__attribute__((ms_abi))`; unknown imports get a generated ms_abi
   trampoline that self-identifies and returns 0. (This also means the native libvulkan/libcuda
   pointers from S1 will each need an ms_abi→SysV thunk when actually *called* in S6.)
3. **A minimal MSVC-CRT shim** (~40 real stubs: heap→malloc, TLS/FLS→arrays, sync→no-op,
   time→clock_gettime, module/handle, `WideCharToMultiByte`/`MultiByteToWideChar`, env). The CRT
   probes kernel32 via `GetProcAddress` for optional APIs — the shim returns callable stubs so the
   probes succeed.
4. **The Windows TEB on the `gs` segment** — the last blocker. The CRT reads `gs:[0x58]`
   (ThreadLocalStoragePointer), `gs:[0x30]` (TEB self), `gs:[0x60]` (PEB); Linux uses `fs` for glibc
   and leaves `gs` empty, so it faulted at `0x58`. Fix: allocate a fake TEB + PEB, set up **PE-TLS**
   (copy the image TLS template, store the slot index, hang it off the TLS array), and point `gs` at
   the TEB via `arch_prctl(ARCH_SET_GS)` — exactly the Wine TEB mechanism, ~40 lines. After that,
   DllMain returns 1.

**Conclusion:** loading + initialising the real DLSS-G snippet natively is **not the blocker** — it
works. No D3D12/DXGI/COM, no exotic dependency; just the bounded CRT surface + the TEB/gs setup.

## S5 — host ↔ snippet interface

`nvngx_dlssg.dll` exports **0 named functions** — the host drives the snippet through an internal,
undocumented NGX ABI, not exports. Two ways in: **(a)** load the Windows NGX host PEs too, or **(b)**
reverse the host↔snippet entry ABI. We pursued **(a)**.

### S5(a) — the Windows NGX host loads natively too (DONE — `s5_host.c`)

`s5_host.c` generalises the S2 loader into a **multi-PE loader** (module registry + real
`LoadLibrary`/`GetProcAddress` + export-table parsing) and loads the NGX core host. Result:

```
[loading] _nvngx.dll
[_nvngx.dll DllMain -> 1, exports@rva=0x156950]
== NGX Vulkan API exports resolved from the natively-loaded host ==
   NVSDK_NGX_VULKAN_Init                     FOUND
   NVSDK_NGX_VULKAN_Init_ProjectID           FOUND
   NVSDK_NGX_VULKAN_Init_Ext2                FOUND
   NVSDK_NGX_VULKAN_CreateFeature            FOUND
   NVSDK_NGX_VULKAN_CreateFeature1           FOUND
   NVSDK_NGX_VULKAN_EvaluateFeature          FOUND
   NVSDK_NGX_VULKAN_GetFeatureRequirements   FOUND
   NVSDK_NGX_VULKAN_GetCapabilityParameters  FOUND
```

So **both the FG snippet (S2) and the Windows NGX core host (`_nvngx.dll`) load + init natively
(DllMain → 1), and the host's full `NVSDK_NGX_VULKAN_*` API — including `CreateFeature` — is
reachable.** During the host's DllMain it probes Windows API-set DLLs (`api-ms-win-core-synch…`,
`…-fibers…`) via `LoadLibrary`; we return not-found and it falls back to kernel32 (DllMain still
returns 1). The host also exports the same API as the thin forwarder `nvngx.dll`.

### S5(b) — `NVSDK_NGX_VULKAN_Init_ProjectID` RUNS natively; only the nvapi bridge is missing (DONE)

`s5_host.c` now creates a live `VkInstance`/`VkDevice`, wires the **generic MS-x64→SysV ABI thunk**
(`ms2sysv_common`) for the 22 Vulkan + 26 CUDA imports (all ≤6 int/ptr args → register+2-stack
shuffle), passes **ms_abi `gipa`/`gdpa` wrappers**, and calls `NVSDK_NGX_VULKAN_Init_ProjectID` on
the natively-loaded host. Result:

```
[calling NVSDK_NGX_VULKAN_Init_ProjectID ...]
  ... registry reads, VerifyVersionInfo, GetSystemDirectoryW ...
  [LoadLibrary] msasn1/cryptnet/cryptbase/wldp/drvstore/devobj.dll  (tolerated, not found)
  [LoadLibrary] nvapi64.dll        -> not found
  [STUB] dyn:nvapi_QueryInterface  -> 0
  [LoadLibrary] vulkan-1.dll ; [STUB] dyn:vkGetInstanceProcAddr
  [NGX Init returned 0xBAD00002]           # FAIL_PlatformError
```

**Init runs to completion through NGX's real init logic and returns a clean, well-defined NGX error
— no crash, no ABI mismatch.** It fails at `0xBAD00002` (`FAIL_PlatformError`) purely because
**`nvapi64.dll` is not bridged**, so the host can't identify the GPU architecture. Everything else —
the PE loader, the MS-x64 CRT shim, the TEB/gs, the ms_abi↔SysV Vulkan/CUDA thunks, the ms_abi
gipa/gdpa — works end-to-end. This retires the "internal host↔snippet ABI" fear: the app talks to
the host via the **documented `NVSDK_NGX_VULKAN_*` API**, and the host loads the snippet itself.

**The single remaining dependency to a successful `Init` is an nvapi shim** (the dxvk-nvapi role):
implement enough of `nvapi_QueryInterface` + the `NvAPI_*` GPU-architecture queries, bridged to the
native driver (`libnvidia-ml.so` / `libcuda`), that the host classifies the RTX 5070 as Blackwell.
Then `CreateFeature(FrameGeneration)` loads the snippet, and S6 (feed our frame via the snippet's
CUDA-Vulkan external-memory/semaphore interop) + S4 (Reflex `VK_NV_low_latency2` pacing) follow.

### S5(c) — Gate 2 characterised: NGX arch-detection uses PRIVATE nvapi interfaces (the wall)

Instrumented `nvapi_QueryInterface` to log the interface IDs NGX resolves during Init. It queries,
after `NvAPI_Initialize` (0x0150E828), a set of **private/undocumented** interface IDs:

```
0xAD298D3F  0x0150E828(Initialize)  0x33C7358C  0x593E8644  0x0694D52E  0x375DBD6B  0xDA8466A0
```

These are **not** the documented public IDs (`EnumPhysicalGPUs` 0xE5AC921F, `GPU_GetArchInfo`
0xD8265D24, `GPU_GetPCIIdentifiers` 0x2DDFB66E) — so NGX uses internal nvapi entry points whose
function semantics + out-struct layouts are undocumented. Two hoped-for shortcuts were tested and
**do not apply to this build**:

- **NVML fallback — absent.** NGX's strings mention `nvngx_common_nvml`, so we bridged `nvml.dll` →
  native `libnvidia-ml.so.1` (public API, easy) and returned NULL from `nvapi_QueryInterface` to
  force the fallback. NGX **never loads `nvml.dll`** — it relies on nvapi alone for arch detection.
  (The nvml bridge is left in `s_GetProcAddress`, ready if a future NGX build uses it.)
- **dxvk-nvapi — pulls the DXVK stack.** jp7677/dxvk-nvapi's `nvapi64.dll` implements these private
  interfaces, but it imports **`d3d11.dll` / `d3d12.dll` / `dxgi.dll`** (it queries the GPU through
  DXVK/vkd3d, not Vulkan directly). Loading it natively would drag in the whole DXVK d3d stack —
  the "re-host Proton" wall relocated into the nvapi layer. Not a clean unlock.

**So Gate 2 requires one of:** (a) **reverse-engineer** the ~6 private nvapi interfaces NGX calls
(disassemble `nvngx`/`_nvngx` to learn each function's out-struct, implement against native
NVML/Vulkan) — deep, multi-session, uncertain; or (b) host the **DXVK d3d11/dxgi/d3d12 stack** to run
dxvk-nvapi — deeper. Until one is done, **Gate 3 (the FG-available go/no-go) cannot be reached**,
because the capability query needs a *successful* Init.

*(Note: with the CRT time/thread stubs returning live values, ~40% of runs crash intermittently at a
fixed 0xa8ec8148cb inside NGX — a data-dependent path off a time/seed stub; clean runs deterministically
reach `0xBAD00002`. Not worth determinising until Gate 2 is crossed.)*

### Honest status & decision point

Path B went from "infeasible in practice" to: **both NVIDIA PEs load natively, DllMain→1, the
documented `NVSDK_NGX_VULKAN_*` API drives the host, 48 foreign-ABI Vulkan/CUDA imports run through
real Init logic via one register-shuffle thunk with zero errors, and Init returns a clean, precisely
localised `PlatformError`** — blocked at exactly one gate: GPU-arch detection through private nvapi
interfaces. Crossing that gate to even *test* whether Linux NGX reports FG-available is an
RE-heavy (or DXVK-heavy) effort. The own model (§21, native, single-GPU-trainable) remains the
pragmatic parallel path and does not depend on any of this.

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

**S5** (the crux): load the NGX host PEs (`nvngx.dll` + `_nvngx.dll`) under the same loader (they
need the same CRT shim + a few more Win32 DLLs: CRYPT32/bcrypt/ole32/WS2_32 — all shimmable), get
`NVSDK_NGX_VULKAN_Init` + `NVSDK_NGX_VULKAN_CreateFeature(FrameGeneration)` to succeed through the
Windows host talking to the natively-loaded snippet; **or** trace the host↔snippet entry ABI and
drive the snippet directly. Then S6: implement the ms_abi→SysV thunks for the 22 Vulkan + 26 CUDA
imports and feed our frame via CUDA-Vulkan external-memory/semaphore interop; S4: Reflex pacing.

Own-model (§21) stays the parallel fallback (5080 server, 100–200 h) in case S5's internal ABI
proves intractable.

## Build & run

```
make            # builds pe_probe (S1) and, with: cc -O2 -o pe_load pe_load.c -ldl (S2)
./pe_probe /usr/lib/nvidia/wine/nvngx_dlssg.dll   # S1: map/relocate/resolve + gap report
./pe_load  /usr/lib/nvidia/wine/nvngx_dlssg.dll   # S2: run DllMain natively -> returns 1
```

## Legal

`nvngx_dlssg.dll` is NVIDIA-licensed and shipped for the driver's Wine/Proton runtime. This
experiment loads the on-disk driver file in place; it does **not** copy or redistribute it. Any
shippable product built on Path B needs explicit NVIDIA clearance (design.md §20.6).
