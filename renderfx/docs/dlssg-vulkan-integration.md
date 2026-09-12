# DLSS-G (Frame Generation) — Vulkan-Integration Spec

Technische Spec für den künftigen RenderFX-FG-Slot `dlss_fg`. Quellen: Streamline-2.14.1-DLSS-G-Guide
(`/tmp/dlls/fg310/ProgrammingGuideDLSS_G.md`) + vendored NGX-Helper
(`renderfx/vendor/ngx/include/nvsdk_ngx_helpers_dlssg_vk.h`,
`renderfx/vendor/ngx/include/nvsdk_ngx_params_dlssg.h`,
`renderfx/vendor/ngx/include/nvsdk_ngx_defs_dlssg.h`). Kein Marketing.

## 1. Vulkan-Voraussetzungen

**Extensions: keine hartcodierte Liste — Laufzeit-Query.** Die vendored Header enthalten keine
Extension-Strings. Vor `vkCreateInstance`/`vkCreateDevice` abfragen:

- `NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements` —
  `renderfx/vendor/ngx/include/nvsdk_ngx_vk.h:646`
- `NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements` —
  `renderfx/vendor/ngx/include/nvsdk_ngx_vk.h:699`
- Deprecated, aber von `renderfx/src/ngx_probe.cpp:41` noch genutzt:
  `NVSDK_NGX_VULKAN_RequiredExtensions` — `nvsdk_ngx_vk.h:114`.
  Die Probe hängt zusätzlich `VK_KHR_get_physical_device_properties2` an (`ngx_probe.cpp:43`).

**Treiberversion / API-Level (Guide):** DLSS-G betreibt Optical Flow unter Vulkan per Default im
Interop-Modus; nativer Modus braucht Treiber ≥ 527.64 (Windows) / ≥ 525.72 (Linux) und
`VK_API_VERSION_1_1` (empfohlen `VK_API_VERSION_1_3`) — `ProgrammingGuideDLSS_G.md:192`.
Die Probe erstellt die Instance mit `VK_API_VERSION_1_3` (`ngx_probe.cpp:45`).

**Queue:** Der DLSS-G-VK-Helper nennt keine Queue-Familie (Create/Evaluate nehmen nur
`VkCommandBuffer`, `nvsdk_ngx_helpers_dlssg_vk.h:46,63`). Repo-Konvention: Graphics-Queue
(`ngx_probe.cpp:58`, `VK_QUEUE_GRAPHICS_BIT`). Present-Sync läuft über die Present-Queue mit
binären Semaphoren (`ProgrammingGuideDLSS_G.md:906-939`, s. §2 Present).

**Bilder:** Der Helper übergibt `VkImageView`+`VkImage`+SubresourceRange+Format+W/H pro Ressource
(`nvsdk_ngx_defs_vk.h:53-60`, Typ-Tag `nvsdk_ngx_defs_vk.h:26-30`). Repo-Konvention aus
`renderfx/src/ngx.cpp:68-76` (`wrap()`): **GENERAL-Layout erwartet**; Outputs `readWrite=true`,
was laut `nvsdk_ngx_defs_vk.h:90-91` `VK_IMAGE_USAGE_STORAGE_BIT` auf dem Image verlangt.
Eingaben `read-only`, Ausgaben (`OutputInterpolated`, optional `OutputReal`) `readWrite`.

## 2. Lebenszyklus

### 2.1 NGX-Vulkan-Init

`NVSDK_NGX_VULKAN_Init_with_ProjectID` — `nvsdk_ngx_vk.h:258`. Repo-Muster `ngx.cpp:80-116`:

- ProjectID muss GUID-förmig sein, Engine-Typ CUSTOM (`ngx.cpp:98-101`):
  `"b1e7fa2c-9d34-4c1a-8b77-6f0a1e2d3c4b"`, `NVSDK_NGX_ENGINE_TYPE_CUSTOM`, `"1.0"`.
- **Beschreibbarer Data-Path ist Pflicht** (sonst Init-Fehler `0xBAD00005`): derselbe Pfad als
  `wchar_t*`-Legacy-Arg UND in `NVSDK_NGX_FeatureCommonInfo.PathListInfo` (`ngx.cpp:91-96`,
  `ngx_probe.cpp:74-84`). Heute: vendored `rel`-Dir via `RENDERFX_NGX_DATA_PATH`.
- Shutdown via `NVSDK_NGX_VULKAN_Shutdown1(device)` (`ngx.cpp:126`).

### 2.2 Capability: SuperSampling vs. FrameGeneration

- SR/DLAA: `SuperSampling.Available` (`ngx.cpp:110`); RR: `SuperSamplingDenoising.Available`
  (`ngx.cpp:111`). Abfrage über `NVSDK_NGX_VULKAN_GetCapabilityParameters` (`nvsdk_ngx_vk.h:422`).
- FG: `FrameGeneration.Available` / `FeatureInitResult` / `NeedsUpdatedDriver`
  (`nvsdk_ngx_defs_dlssg.h:48-52`; Query-Muster `ngx_probe.cpp:99-102`); daneben
  `FrameInterpolation.*` (`nvsdk_ngx_defs_dlssg.h:54-58`).
- **Nativer Linux-NGX-Stand:** `FrameGeneration.Available = 0`, `FeatureInitResult = 0xBAD00004`
  (`FAIL_FeatureNotSupported`) — `design.md:907`. Per-Adapter-Vorabcheck auch ohne Init möglich:
  `NVSDK_NGX_VULKAN_GetFeatureRequirements` (`nvsdk_ngx_vk.h:605`); nativer Path-B-Loader meldet
  dort `FeatureSupported = 0` (GREEN) — `experiments/dlssg-native/gate3_oracle_result.txt:6-8`,
  `experiments/dlssg-native/RECON.md:30-40`.

### 2.3 CreateFeature (FG)

`NGX_VK_CREATE_DLSSG` (`nvsdk_ngx_helpers_dlssg_vk.h:45-60`): setzt Node-Masken, `Width`/`Height`,
`DLSSG.BackbufferFormat` (`NativeBackbufferFormat`), dann
`NVSDK_NGX_VULKAN_CreateFeature(cmd, NVSDK_NGX_Feature_FrameGeneration, …)`.
Create-Struct `NVSDK_NGX_DLSSG_Create_Params` (`nvsdk_ngx_params_dlssg.h:31-39`): `Width`,
`Height`, `NativeBackbufferFormat`, `RenderWidth/RenderHeight`, `DynamicResolutionScaling`.
Optionale Create-Time-Parameter:

- `DLSSG.ResourceAlwaysProvidedFlags` / `ResourceNeverProvidedFlags`
  (`nvsdk_ngx_defs_dlssg.h:109-110`, Flags `nvsdk_ngx_defs_dlssg.h:32-46`) — VRAM-Optimierung.
- `DLSSG.UserInterfaceRecompositionEnabled` (`nvsdk_ngx_defs_dlssg.h:374`) — UIR-Modus; verlangt
  Hudless + UI/UIAlpha (s. §2.4). Endnutzer-Override via NVIDIA App möglich.
- MFG-Obergrenze als Capability: `DLSSG.MultiFrameCountMax` (`nvsdk_ngx_defs_dlssg.h:357`).

Lazy-Create auf erstem Record + Recreate bei Dimensionswechsel — Muster `ngx.cpp:143-160`.

### 2.4 EvaluateFeature (FG) — Parameter-Tabelle

`NGX_VK_EVALUATE_DLSSG` (`nvsdk_ngx_helpers_dlssg_vk.h:62-193`) packt in `NVSDK_NGX_Parameter`;
Opt-Struct `NVSDK_NGX_DLSSG_Opt_Eval_Params` (`nvsdk_ngx_params_dlssg.h:41-136`).
Evaluate läuft auf offenem, aufzeichnendem `VkCommandBuffer`
(`NVSDK_NGX_VULKAN_EvaluateFeature_C`, `nvsdk_ngx_vk.h:755`).

| NGX-Parameter (`nvsdk_ngx_defs_dlssg.h`) | Quelle im Helper | Pflicht | Bedeutung |
|---|---|---|---|
| `DLSSG.Backbuffer` (`:64`) | `:33,69` | ja | finale Farbe inkl. PostFX/UI |
| `DLSSG.MVecs` (`:65`) | `:35,70` | ja | dichte MVs inkl. Kamera; gleiche Anforderungen wie DLSS-SR (Guide `:267`) |
| `DLSSG.Depth` (`:66`) | `:34,71` | ja | gleiche Tiefe wie für MV-Erzeugung (Guide `:266`) |
| `DLSSG.HUDLess` (`:78`) | `:36,72` | opt (stark empfohlen) | Szene VOR UI, sonst identisch zum Backbuffer |
| `DLSSG.UI` (`:82`) / `DLSSG.UIAlpha` (`:84`) | `:37-38,73-74` | opt | UI-Farbe (premultipliziert, `Final = UI + (1-A)·Hudless`) bzw. nur Alpha (bevorzugt); nur eines nötig (`:85-86`) |
| `DLSSG.BidirectionalDistortionField` (`:75`) | `:39,75` | opt | nur bei starker PostFX-Verzerrung; dann MVs/Depth UNverzerrt, Farbe verzerrt |
| `DLSSG.OutputInterpolated` (`:91`) | `:40,76` | ja | generierter Frame, gleiches Format wie Backbuffer |
| `DLSSG.OutputReal` (`:93`) | `:41,77` | opt | Snippet darf Debug-Text auf Real-Frame rendern |
| `DLSSG.OutputDisableInterpolation` (`:123`) | `:42,78` | opt | ≥4-Byte-Buffer; Snippet schreibt `true` in Byte 0 wenn der interpolierte Frame verworfen gehört |
| `DLSSG.MultiFrameCount` (`:361`) | `:82` | MFG | zu generierende Zwischenframes: 1 = 2x, 2 = 3x, … (allg. n → (n+1)x). Max = `MultiFrameCountMax` (`:357`) bzw. SL `numFramesToGenerateMax` (Guide `:837-841`). Konkrete Modi zur Laufzeit prüfen — **keine feste 2x–6x-Zahl in Guide/Helper belegt** (Kostentabelle nennt 2x/4x, Guide `:58-68`) |
| `DLSSG.MultiFrameIndex` (`:363`) | `:83` | MFG | 1-basierter Index des gerade generierten Zwischenframes ≤ MultiFrameCount |
| `DLSSG.CameraViewToClip` … `PrevClipToClip` (`:128-136`) | `:85-89` | ja | 5× float4x4, OHNE TAA-Jitter (`nvsdk_ngx_params_dlssg.h:50`) |
| `DLSSG.JitterOffsetX/Y` (`:139-140`) | `:91-92` | ja | Clip-Space-Jitter |
| `DLSSG.MvecScaleX/Y` (`:144`) | `:94-95` | ja | MV-Normierung in [-1,1] (Guide `:661-662`) |
| Kamera: Pos/Up/Right/Fwd/Near/Far/FOV/Aspect (`:150-180`) | `:97-120` | möglichst alle | max. Forward-Kompatibilität |
| `DLSSG.ColorBuffersHDR` (`:183`) | `:121` | — | HDR-Monitor-Pfad; HDR10/BT.2100 + UINT10 (Guide `:746`); FP16/scRGB NICHT unterstützt (Guide `:750`) |
| `DLSSG.DepthInverted` (`:186`) | `:123` | — | Linearisierung: `1/(1-d)` bzw. `1/d` bei invertiert (`:227-228`) |
| `DLSSG.CameraMotionIncluded` (`:189`) | `:125` | — | Kamera in MVs enthalten? |
| `DLSSG.Reset` (`:194`) | `:127` | events | keine Frame-Kohärenz (Szenenwechsel) |
| `DLSSG.NotRenderingGameFrames` (`:200`) | `:130` | events | Menü/Pause/Cutscene — kein Game-Frame |
| `DLSSG.OrthoProjection` (`:203`) | `:132` | — | ortho an/aus |
| `DLSSG.MvecInvalidValue` / `MvecDilated` (`:216-217`) | `:134-135` | — | Invalid-Marker, Dilation-Status |
| `DLSSG.MenuDetectionEnabled` (`:220`) | `:137` | — | Vollbild-Menü-Autoerkennung (braucht UI-Buffer; manu
...[truncated 4474 chars]