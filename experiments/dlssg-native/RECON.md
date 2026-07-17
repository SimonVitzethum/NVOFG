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

### S5(c) RE round — nvapi IDs mapped; blocker refined to the Vulkan arch path

RE'd the private IDs via the official **NVIDIA/nvapi** `nvapi_interface.h` + **dxvk-nvapi** source:

| ID | Function | Needed? |
|---|---|---|
| 0x0150E828 | `NvAPI_Initialize` | yes |
| 0x0694D52E | `NvAPI_DRS_CreateSession` | yes (reads DLSS driver-settings) |
| 0x375DBD6B | `NvAPI_DRS_LoadSettings` | yes |
| 0xDA8466A0 | `NvAPI_DRS_GetBaseProfile` | yes |
| 0x73BF8338 | `NvAPI_DRS_GetSetting` | yes (→ SETTING_NOT_FOUND ⇒ NGX uses defaults) |
| 0xAD298D3F, 0x33C7358C, 0x593E8644 | private/internal | **optional** — dxvk-nvapi doesn't implement them either, yet DLSS-G works under Proton |

Implemented `nvapi_QueryInterface` dispatch for `Initialize` + the DRS set (no-op successes, empty
settings) and NULL for the 3 optional private IDs — matching dxvk-nvapi. **Init still returns
`0xBAD00002`.** The trace then shows the host `LoadLibrary("vulkan-1.dll")` +
`GetProcAddress("vkGetInstanceProcAddr")` — so GPU-arch resolution is **Vulkan-side**, not via the
nvapi arch functions. Bridged that too (`vkGetInstanceProcAddr` → our ms_abi `gipa`; other `vk*` →
native libvulkan via ms2sysv). **Still `0xBAD00002`**, and the intermittent crash rate rises in this
deeper path.

**Refined blocker:** the nvapi surface NGX needs is now satisfied (Init + DRS); the residual
`PlatformError` lives in NGX's **Vulkan GPU-arch/platform resolution** — the host presumably creates
its own instance / queries device props (likely `VK_KHR_driver_properties` / vendor+device ID) and
isn't yet getting a classifiable Blackwell. Next concrete sub-step: trace exactly which
instance/device Vulkan calls the host's arch path makes and what property it can't satisfy. This is
deep, iterative, and the ultimate go/no-go (does Linux NGX report FG-available at all) is still
unknown — the own model (§21) remains the pragmatic parallel path.

### S5(c) deep RE — crossed PlatformError; NGX accepts Blackwell; wall = Windows adapter correlation

Iteratively RE'd the nvapi call chain NGX makes at Init (IDs mapped via the official
NVIDIA/nvapi headers), bridging each function to real values. Each bridged fn advances NGX:

```
Initialize + DRS_{CreateSession,LoadSettings,GetBaseProfile,GetSetting}
vkGetPhysicalDeviceProperties2 (interposed) -> NGX sees device=0x2D18 "RTX 5070", asks
    VkPhysicalDeviceIDProperties (LUIDValid=0 natively; we force a consistent fake LUID)
SYS_GetDriverAndBranchVersion(610.43)   ===> error 0xBAD00002 -> 0xBAD00001
EnumPhysicalGPUs(1) -> GPU_GetArchInfo -> report GB200/Blackwell   [NGX ACCEPTS the arch]
GetLogicalGPUFromPhysicalGPU + GPU_{PCIIdentifiers,FullName,GPUType,BusType} (real 5070)
GPU_GetLogicalGpuInfo -> fill pOSAdapterId (=our LUID) + physicalGpuCount=1 + handle
```

**Two milestones:** (1) **crossed the PlatformError gate** (0xBAD00002 → 0xBAD00001) once the
driver-version bridge was added; (2) **NGX accepts the Blackwell arch** we report via nvapi.
The fake nvapi handles must point to **real zeroed memory** (the Windows nvapi handles are
internally pointers — NGX dereferences them; a bare token like `0x600D2` crashes at the deref).

**Current wall:** right after `GPU_GetLogicalGpuInfo` fills the logical-GPU struct (OS-adapter
LUID + physical handles), NGX crashes at NULL inside its **Windows OS-adapter correlation** —
it processes the LUID/adapter data and dereferences state we can't synthesise (precise struct
offsets + ultimately a real DXGI/OS adapter to match the LUID against). This is the same
Windows-adapter-platform dependency identified at the top of S5, now reached from *deep inside*:
NGX gets the Blackwell 5070 fine but wants to correlate it to an OS/DXGI adapter, which has no
native-Linux equivalent. `0xBAD00001` (FeatureNotSupported) is the still-initialising state; NGX
crashes before returning a definitive Init result.

**Assessment:** Path B is proven through *the entire loader + MS-x64 ABI + CRT/TEB + nvapi
surface + GPU-arch acceptance* — an extraordinary distance. The residual is NGX's internal
Windows-adapter correlation: iterative precise-struct RE that converges on needing a real
Windows/DXGI adapter (the DXVK dependency). The go/no-go (does Linux NGX report FG-available)
remains unreached because Init crashes in that correlation before completing. The own model
(§21) stays the pragmatic parallel path.

### S5(c) crash-site RE — both pinpointed inside `_nvngx.dll`'s adapter enumeration

Enhanced the SIGSEGV handler to resolve the faulting RIP to `module+offset`. Both crashes are
in the NGX host `_nvngx.dll`, in its GPU/adapter-enumeration processing:

- **`_nvngx.dll+0xb7f9`** (after `GPU_GetLogicalGpuInfo`): iterates an internal pointer array —
  `rax=*(r14)`, `r8=&array[i]`, `mov (%r8),%rdx` faults. NGX walks an internal GPU/adapter list
  our synthesized nvapi enumeration didn't fully/correctly populate.
- **`_nvngx.dll+0xceee`** (the earlier, intermittent one): builds a ~40-byte struct on the stack
  (2×`movups` + `movsd`), then `movzbl (%rcx),%eax` with `rcx=NULL` and a length check `esi >= 19`
  — reading a **string identifier** (UUID / name / path, ≥19 chars) we return null or too short.

So beyond the arch acceptance, NGX processes a **fuller GPU/adapter representation** — an internal
adapter array + string identifiers — before it will complete Init. This is precise, disassembly-
driven RE of NGX internals, and it keeps converging on NGX's **Windows adapter model** (the LUID/
UUID/adapter-list correlation), whose ultimate backing is a real DXGI/OS adapter with no native-
Linux equivalent. Each fix advances NGX one instruction-region further; the tail is long and points
at the DXVK/DXGI dependency. `0xBAD00001` remains the still-initialising state; Init crashes before
a definitive FG-available result. Own model (§21) stays the pragmatic parallel path.

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

## 🟢 GATE 3 — ANSWERED GREEN: the native Linux driver reports FG available

Before sinking more sessions into the native adapter-layer rebuild, we answered the one question
the whole Path-B chain runs toward, with the cheapest *complete* oracle: run the **same** driver
`_nvngx.dll` under **GE-Proton** (full dxvk-nvapi + vkd3d + dxgi stack) against the **real Linux
driver + RTX 5070**, and read `NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration)`
(`ngxfg_probe.exe`, mingw; `run_proton.sh`). Result (`gate3_oracle_result.txt`):

```
FrameGeneration   result=0x00000001 (Success)  FeatureSupported=0x0 (SUPPORTED)  MinHWArch=0x190  MinOS=10.0.0
```

**`FeatureSupported == 0` ⇒ the native Linux driver reports DLSS Frame Generation as AVAILABLE on
the RTX 5070** (through the complete Windows stack). The "no" was **not** structurally predetermined —
Path B has a **proven payoff**. (SuperSampling/RayReconstruction returned `0xBAD00012` NotImplemented
via this specific discovery call — a quirk of the requirements path, not our concern; FG is the
target and it's Supported.)

**Consequence:** the native adapter-layer rebuild (the +0xceee string identifier, the +0xb7f9
adapter array, the LUID/OS-adapter correlation) is now **justified**, and Proton/Wine is the
complete **reference** for every adapter struct we synthesise — we port what Wine demonstrably feeds
NGX, instead of guessing. Gate 3 green flips the strategy from "should we keep digging?" to "finish
the native adapter layer, using Wine as the oracle for each struct."

## S5(d) — reference extraction from Proton + native nvapi bridge aligned

With Gate 3 green, extracted the **exact reference** by running `ngxfg_probe.exe` under GE-Proton with
`DXVK_NVAPI_LOG_LEVEL=trace` — the precise `NvAPI_*` sequence NGX makes for
`GetFeatureRequirements(FrameGeneration)`, with each call's args + return:

```
Initialize -> SYS_GetDriverAndBranchVersion -> EnumPhysicalGPUs -> DRS_CreateSession/LoadSettings/
GetBaseProfile -> DRS_GetSetting(0x10e41df2 -> "Setting not found") -> GPU_GetArchInfo ->
GetLogicalGPUFromPhysicalGPU -> GPU_GetLogicalGpuInfo (OK) -> DRS_FindApplicationByName ->
DRS_GetProfileInfo (OK) -> DRS_DestroySession -> DRS_GetSetting(... "not found") ...
```

Key reference facts (the derivation rules, per the user's "rule not value" discipline):
- **`GPU_GetLogicalGpuInfo` returns OK under Proton** — and dxvk-nvapi's version returns *Error* when
  `deviceLUIDValid==false`. So **under Proton the Vulkan `deviceLUID` is VALID** (DXVK's device enables
  it); natively it is **0** (measured). This is the one value Proton revealed: the native path must
  **synthesize a consistent LUID** (same bytes in `VkPhysicalDeviceIDProperties.deviceLUID` and in
  `GetLogicalGpuInfo`'s `pOSAdapterId`), because the driver won't provide one.
- **`DRS_GetSetting` returns "Setting not found"** = `NVAPI_SETTING_NOT_FOUND = -160` (an *NvAPI_Status*,
  not an NGX code). **Fixed a real bug**: our bridge returned `0xBAD00004` (an NGX result) which NGX
  misinterpreted as an nvapi status. Now `-160`.
- `DRS_GetProfileInfo` zeroes `profileName` (valid empty wstring). Implemented.

Bridged the full DRS set (`FindApplicationByName`, `GetProfileInfo`, `DestroySession`, correct
`GetSetting` status) matching dxvk-nvapi. Native `GetFeatureRequirements` now advances through the
entire nvapi sequence.

**Current native wall (S5d):** the crash moved out of nvapi entirely — into NGX's own
**path resolution**: the faulting site builds the string `"NVIDIA"` + … and the adjacent literal is
`"error: NGXGetPath failed"`. So NGX's **`NGXGetPath`** (locating its model/config files from the
ApplicationDataPath via the Win32 file APIs) **fails natively** (under Proton, Wine's filesystem makes
it succeed), and NGX then crashes building the error string. The next layer is a **minimal Windows
filesystem/path shim** (make `NGXGetPath` succeed, or the ApplicationDataPath resolve) — reference
again available from the Proton run. This is engineering with a known-green endpoint (Gate 3), not a
feasibility risk.

## S5e — path + adapter shims: native GetFeatureRequirements now RUNS (no crash)

The `NGXGetPath` crash was **`SHGetKnownFolderPath`** (SHELL32) returning null (my trap didn't write
`*out`) → null path string → `wcslen` crash. Implemented it (returns a valid path + `CoTaskMemFree`
stub). The crash then moved forward, revealing (via file-API + IAT logging) the real consumers of
that path and the adapter identity:

- **`SHGetKnownFolderPath`** → NGX builds `…\NVIDIA\NGX\models\config\versions\1\files\nvngx_deny_list.txt`
  and probes it via **`PathFileExistsW`** / `CreateFileW`. Absent is fine (empty deny list); stubbed.
- **`D3DKMTEnumAdapters2`** (win32u kernel display-adapter enumeration) — **this is where NGX gets the
  OS-adapter LUID to correlate with the GPU.** Implemented to report one adapter whose
  `AdapterLuid == g_luid` (our deviceUUID-derived LUID, matching the Vulkan `deviceLUID`). Under
  Proton this is Wine's win32u.

With these, **native `NVSDK_NGX_VULKAN_GetFeatureRequirements(FrameGeneration)` now RUNS to completion
without crashing** (it returned `0xBAD00012`/NotImplemented rather than Proton's Success — the
requirements path still degrades when the adapter correlation isn't 100%, but it no longer faults).

**Remaining:** the full `Init_ProjectID` path still crashes at its own earlier sites (`_nvngx.dll`
+0xb7f9 adapter-array iteration, +0xceee string) — it takes a heavier path than GetFeatureRequirements
and wants a more complete D3DKMT adapter model (likely `D3DKMTOpenAdapterFromLuid` +
`D3DKMTQueryAdapterInfo`, and the exact `D3DKMT_ADAPTERINFO` struct). This is the deep adapter-
correlation tail — grind each against the Proton reference (WINEDEBUG / dxvk trace). Multi-session,
but with a known-green endpoint (Gate 3). The own model (§21) stays the parallel path.

## S5e — the wall: NGX wants undocumented WDDM driver-private adapter info (D3DKMT)

Traced the adapter path past `D3DKMTEnumAdapters2`: NGX then calls **`D3DKMTQueryAdapterInfo`** with
**`Type = 48` (`KMTQAITYPE_KMD_DRIVER_VERSION` / registry family), buffer size 1072 bytes**, then
`D3DKMTCloseAdapter`. My stub zeroes the buffer + returns STATUS_SUCCESS, but NGX still returns
NotImplemented — it needs **real WDDM driver-private adapter data** (an undocumented 1072-byte
structure the Windows kernel display driver fills). Under Proton this is Wine's `win32u`
`D3DKMTQueryAdapterInfo`, which bridges to the native NVIDIA driver; there is no native-Linux D3DKMT.

This is the honest bottom of the Path-B rabbit hole: past the loader + MS-x64 ABI + CRT/TEB + nvapi
surface + path shim + adapter enumeration, NGX's Windows host reaches into the **WDDM kernel
display-driver model** (D3DKMT driver-private structures), which has no native-Linux equivalent and
whose formats are undocumented. Reproducing them faithfully (Type 48 and the further D3DKMT calls
CreateFeature will make) is genuine multi-week reverse-engineering — each struct measurable against
the Proton/Wine reference, but each a deep dive.

**Status of Path B (honest):**
- **Gate 3 = GREEN**: the native driver *does* report FG available (proven via the Proton oracle).
  The feasibility question is answered — no more machbarkeit risk.
- The native loader is extensive and correct as far as it goes: both NVIDIA PEs load + init, the NGX
  Vulkan API is driven natively, 48 foreign-ABI Vulkan/CUDA imports run through one register-shuffle
  thunk, the nvapi surface matches dxvk-nvapi, and **native `GetFeatureRequirements(FG)` runs to
  completion**.
- The remaining path (full `Init` crash tail → `CreateFeature` → `EvaluateFeature` float-ABI thunk →
  present/pacing) is gated on faithfully synthesizing the **WDDM D3DKMT driver-private adapter model**
  — a multi-week effort. Every piece has a Proton reference, so it is engineering, not research; but
  it is a long tail.
- The own model (§21) — native, legal, single-GPU-trainable, no Windows-adapter dependency — remains
  the pragmatic path to shippable native FG, and the 5080 server (≈24 h) makes it actionable.

## Next step (now justified)

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
