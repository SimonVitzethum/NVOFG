// ngxfg_create_harness.c — DLSS-G CreateFeature Skelett (Wine-first, wie ngxfg_init_harness.c).
//
// Minimal Windows x64 .exe (mingw), Lauf unter GE-Proton gegen den echten Treiber.
// Ablauf: VkInstance + VkPhysicalDevice + REAL VkDevice (Interop-Exts, GFR-first),
// dann NVSDK_NGX_VULKAN_Init_ProjectID mit denselben Args wie s5_host nativ
// (UUID-a0b1/engine-CUSTOM/"1.0", dataPath, FeatureInfo-Variante), danach:
//   1) NVSDK_NGX_VULKAN_GetScratchBufferSize(FrameGeneration) abfragen + loggen,
//   2) NVSDK_NGX_VULKAN_CreateFeature(FrameGeneration) mit Width/Height/
//      BackbufferFormat/NodeMasks versuchen + Ergebnis + Handle loggen,
//   3) NVSDK_NGX_VULKAN_ReleaseFeature + NVSDK_NGX_VULKAN_Shutdown1.
// Jede Stufe via emit() nach stdout UND ngxfg_create_result.txt. KEIN Evaluate.
//
// Build (Rezept aus run_init_proton.sh uebernommen, NICHT ausfuehren — nur schreiben):
//   x86_64-w64-mingw32-gcc -O2 -o ngxfg_create_harness.exe ngxfg_create_harness.c \
//     -I<vulkan-headers> -I<ngx-headers>
// Run (analog run_init_proton.sh, DLLs neben die .exe legen):
//   ngxfg_create_harness.exe <width> <height> <format-uint> [variant] [cold] [sdkv]
//     width/height : z.B. 1920 1080
//     format-uint  : NativeBackbufferFormat, s. FORMAT-MAPPING unten (Default 4)
//     variant      : 0 = FeatureInfo NULL, 1 = &FeatureCommonInfo (Default 1)
//     cold         : 1 = GFR-Pre-Call skippen (Default 0)
//     sdkv         : z.B. 0x... (Default NVSDK_NGX_Version_API)
//
// ---------------------------------------------------------------------------
// FORMAT-MAPPING: NativeBackbufferFormat-uint -> VkFormat (Recherche 2026-09-12)
// ---------------------------------------------------------------------------
// Quellen: renderfx/vendor/ngx/include/nvsdk_ngx_defs.h,
//          nvsdk_ngx_defs_dlssg.h, nvsdk_ngx_helpers_dlssg_vk.h,
//          nvsdk_ngx_params_dlssg.h, /tmp/dlls/fg310/ProgrammingGuideDLSS_G.md.
//
// Befund: KEINE Quelle im Repo oder Guide enthaelt eine uint->VkFormat-Tabelle
// fuer DLSSG.BackbufferFormat. NativeBackbufferFormat ist ein plain
// `unsigned int` (nvsdk_ngx_params_dlssg.h:35) und wird im VK-Helper per
// SetUI("DLSSG.BackbufferFormat", ...) 1:1 durchgereicht
// (nvsdk_ngx_helpers_dlssg_vk.h:57); der D3D12-Helper macht dasselbe
// (nvsdk_ngx_helpers_dlssg.h:58). Welches Enum die uint fuellt, ist nirgends
// dokumentiert. NICHT geraten — alles Unbelegte steht als ??? unten.
//
// | uint | Kandidat A: NVSDK_NGX_Buffer_Format    | Kandidat B: VkFormat-Nummer | Status |
// |      | (nvsdk_ngx_defs.h:240-249, Enum BELEGT)| (Vulkan-Spec, nur d. Nummern|        |
// |      |                                       |  belegt, Zuordnung ???)      |        |
// |------|---------------------------------------|-------------------------------|--------|
// |  0   | Unknown                               | VK_FORMAT_UNDEFINED (0): ??? | ???    |
// |  1   | RGB8UI                                | ???                           | ???    |
// |  2   | RGB16F                                | ???                           | ???    |
// |  3   | RGB32F                                | ???                           | ???    |
// |  4   | RGBA8UI                               | VK_FORMAT_R8G8B8A8_UNORM (37): ???, auch B8G8R8A8_UNORM (44): ??? | ??? |
// |  5   | RGBA16F                               | VK_FORMAT_R16G16B16A16_SFLOAT (109): ??? | ??? |
// |  6   | RGBA32F                               | VK_FORMAT_R32G32B32A32_SFLOAT (110): ??? | ??? |
// | 7+   | (kein NGX-Enum-Wert mehr)             | z.B. A2B10G10R10_UNORM_PACK32 (64): ??? | ??? |
//
// Belegte Randfakten (kein Raten):
//  a) Enum-Werte 0..6 aus der C-Enum-Reihenfolge in nvsdk_ngx_defs.h:240-249
//     (Unknown=0 ... RGBA32F=6). Dass DLSSG.BackbufferFormat dieses Enum nutzt,
//     ist ANNAHME (???), gestuetzt nur darauf, dass es das einzige Format-Enum
//     in den NGX-Headern ist und DLSS-SR-Create gar kein Formatfeld hat.
//  b) Gegenhypothese: natives API-Enum (VkFormat-Zahl auf VK, DXGI_FORMAT auf
//     D3D — EIN Struct NVSDK_NGX_DLSSG_Create_Params fuer beide Pfade spricht
//     dafuer; Streamline-Guide:886 verlangt in sl::DLSSGOptions ebenfalls
//     "3D API-specific format enums"). Ebenfalls ???.
//  c) Guide:746+750: HDR => UINT10/RGB10 + HDR10/BT.2100; FP16/scRGB wird von
//     DLSS-G NICHT unterstuetzt. Guide: OutputInterpolated hat dasselbe Format
//     wie der Backbuffer (implizit: Create-Format muss zum Swapchain-Format passen).
//  d) Asymmetrie BELEGT: Der D3D12-Helper setzt zusaetzlich InternalWidth/
//     InternalHeight/DynamicResolution (nvsdk_ngx_helpers_dlssg.h:59-61), der
//     VK-Helper NICHT (nvsdk_ngx_helpers_dlssg_vk.h:45-60 setzt nur Node-Masken,
//     Width, Height, BackbufferFormat). Darum setzt dieses Skelett exakt die
//     5 VK-Parameter — kein RenderWidth/RenderHeight.
//  e) NGX_VK_ESTIMATE_VRAM_DLSSG (helpers_dlssg_vk.h:195-225) nimmt ebenfalls
//     uint32-Format-Enums (color/mvec/depth/hudless/ui) ohne Doku, welches Enum
//     — dieselbe offene Frage, eine Stufe spaeter (VRAM-Schaetzung).
//
// Erstes argv-Raster zum Durchprobieren (Hypothesen, NICHT belegt):
//   4 (RGBA8UI-Hypothese) und 37 (VkFormat-Hypothese R8G8B8A8_UNORM),
//   dann 44 (B8G8R8A8 — typisches Swapchain-Format), 109, 5.
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx_defs_dlssg.h"
#include "nvsdk_ngx.h"
#include <stdarg.h>

static FILE* g_rf;
static void emit(const char* f, ...) {
    va_list a; va_start(a, f); vprintf(f, a); va_end(a);
    if (g_rf) { va_start(a, f); vfprintf(g_rf, f, a); va_end(a); fflush(g_rf); }
}

typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_INIT_PID)(const char*, NVSDK_NGX_EngineType, const char*,
    const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
    PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
    const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_GFR)(const NVSDK_NGX_FeatureDiscoveryInfo*,
    unsigned* /*OutExtensionCount*/, void** /*OutExtensionProperties*/);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_GFR_VK)(VkInstance, VkPhysicalDevice,
    const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_GETCAPS)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_ALLOC)(NVSDK_NGX_Parameter**);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_DESTROYP)(NVSDK_NGX_Parameter*);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_SCRATCH)(NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, size_t*);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_CREATE)(VkCommandBuffer, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_RELEASE)(NVSDK_NGX_Handle*);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_SHUT1)(VkDevice);
typedef NVSDK_NGX_Result (NVSDK_CONV *PFN_SHUT)(void);
typedef void (NVSDK_CONV *PFN_SETUI)(NVSDK_NGX_Parameter*, const char*, unsigned int);

int main(int argc, char** argv) {
    // argv[1]=width argv[2]=height argv[3]=format-uint (Pflicht-Schema laut Auftrag);
    // [4]=variant [5]=cold [6]=sdkv (Init-Kontrollen aus ngxfg_init_harness.c).
    unsigned width  = (argc > 1) ? (unsigned)strtoul(argv[1], 0, 0) : 1920;
    unsigned height = (argc > 2) ? (unsigned)strtoul(argv[2], 0, 0) : 1080;
    unsigned fmt    = (argc > 3) ? (unsigned)strtoul(argv[3], 0, 0) : 4; // Hypothese: RGBA8UI (???)
    int variant = (argc > 4) ? atoi(argv[4]) : 1;
    int cold    = (argc > 5) ? atoi(argv[5]) : 0;
    unsigned sdkv = (argc > 6) ? (unsigned)strtoul(argv[6], 0, 0) : (unsigned)NVSDK_NGX_Version_API;
    g_rf = fopen("ngxfg_create_result.txt", "w");
    emit("create-harness: width=%u height=%u fmt=%u variant=%d cold=%d sdkv=0x%x\n",
        width, height, fmt, variant, cold, sdkv);
    emit("note: fmt-mapping UNBELEGT (s. Kommentar-Tabelle oben); kein Evaluate in diesem Skelett\n");

    // ---- Vulkan: Instance + NVIDIA-Device mit Interop-Exts (aus ngxfg_init_harness.c) ----
    HMODULE vk = LoadLibraryA("vulkan-1.dll");
    if (!vk) { emit("no vulkan-1.dll\n"); return 2; }
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(vk, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance vkCreateInstance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo app = {0}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    const char* ie[] = { "VK_KHR_get_physical_device_properties2",
        "VK_KHR_external_memory_capabilities", "VK_KHR_external_semaphore_capabilities" };
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 3; ici.ppEnabledExtensionNames = ie;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { emit("vkCreateInstance failed\n"); return 2; }
    PFN_vkEnumeratePhysicalDevices vkEnum = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties vkProps = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    PFN_vkGetPhysicalDeviceQueueFamilyProperties vkQ = (PFN_vkGetPhysicalDeviceQueueFamilyProperties)gipa(inst, "vkGetPhysicalDeviceQueueFamilyProperties");
    PFN_vkCreateDevice vkCreateDevice = (PFN_vkCreateDevice)gipa(inst, "vkCreateDevice");
    PFN_vkGetDeviceQueue vkGetDQ = (PFN_vkGetDeviceQueue)gipa(inst, "vkGetDeviceQueue");
    uint32_t n = 0; vkEnum(inst, &n, NULL);
    VkPhysicalDevice pds[8]; if (n > 8) n = 8; vkEnum(inst, &n, pds);
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; ++i) { VkPhysicalDeviceProperties p; vkProps(pds[i], &p);
        if (p.vendorID == 0x10DE && pd == VK_NULL_HANDLE) pd = pds[i]; }
    if (!pd) { emit("no NVIDIA vk device\n"); return 2; }
    uint32_t qn = 0; vkQ(pd, &qn, NULL);
    VkQueueFamilyProperties qp[16]; if (qn > 16) qn = 16; vkQ(pd, &qn, qp);
    uint32_t qfam = 0xFFFFFFFF;
    for (uint32_t i = 0; i < qn; ++i) if (qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfam = i; break; }
    if (qfam == 0xFFFFFFFF) { emit("no graphics queue\n"); return 2; }
    float pr = 1.0f;
    VkDeviceQueueCreateInfo qci = {0}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qfam; qci.queueCount = 1; qci.pQueuePriorities = &pr;
    PFN_vkEnumerateDeviceExtensionProperties vkDevExt =
        (PFN_vkEnumerateDeviceExtensionProperties)gipa(inst, "vkEnumerateDeviceExtensionProperties");
    static const char* want[] = {
        "VK_KHR_external_memory", "VK_KHR_external_memory_fd",
        "VK_KHR_external_semaphore", "VK_KHR_external_semaphore_fd",
        "VK_KHR_dedicated_allocation", "VK_KHR_get_memory_requirements2",
    };
    const char* have[8]; uint32_t nhave = 0;
    if (vkDevExt) {
        uint32_t ne = 0; vkDevExt(pd, NULL, &ne, NULL);
        VkExtensionProperties ep[256]; if (ne > 256) ne = 256; vkDevExt(pd, NULL, &ne, ep);
        for (int w = 0; w < 6; ++w) {
            for (uint32_t e = 0; e < ne; ++e)
                if (!strcmp(ep[e].extensionName, want[w])) { have[nhave++] = want[w]; break; }
        }
    }
    emit("device interop extensions enabled: %u/6\n", nhave);
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nhave; dci.ppEnabledExtensionNames = have;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { emit("vkCreateDevice failed\n"); return 2; }
    VkQueue queue; vkGetDQ(dev, qfam, 0, &queue);
    PFN_vkGetDeviceProcAddr gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    emit("vulkan ok: inst=%p pd=%p dev=%p qfam=%u\n", (void*)inst, (void*)pd, (void*)dev, qfam);

    // ---- Command-Pool + offener Command-Buffer (CreateFeature braucht recording cmdbuf) ----
    PFN_vkCreateCommandPool pCreatePool = (PFN_vkCreateCommandPool)gdpa(dev, "vkCreateCommandPool");
    if (!pCreatePool) pCreatePool = (PFN_vkCreateCommandPool)gipa(inst, "vkCreateCommandPool");
    PFN_vkAllocateCommandBuffers pAllocCb = (PFN_vkAllocateCommandBuffers)gdpa(dev, "vkAllocateCommandBuffers");
    if (!pAllocCb) pAllocCb = (PFN_vkAllocateCommandBuffers)gipa(inst, "vkAllocateCommandBuffers");
    PFN_vkBeginCommandBuffer pBegin = (PFN_vkBeginCommandBuffer)gdpa(dev, "vkBeginCommandBuffer");
    if (!pBegin) pBegin = (PFN_vkBeginCommandBuffer)gipa(inst, "vkBeginCommandBuffer");
    PFN_vkEndCommandBuffer pEnd = (PFN_vkEndCommandBuffer)gdpa(dev, "vkEndCommandBuffer");
    if (!pEnd) pEnd = (PFN_vkEndCommandBuffer)gipa(inst, "vkEndCommandBuffer");
    if (!pCreatePool || !pAllocCb || !pBegin || !pEnd) { emit("no cmdbuf fns\n"); return 2; }
    VkCommandPoolCreateInfo pci = {0}; pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pci.queueFamilyIndex = qfam;
    VkCommandPool pool = VK_NULL_HANDLE;
    if (pCreatePool(dev, &pci, NULL, &pool) != VK_SUCCESS) { emit("vkCreateCommandPool failed\n"); return 2; }
    VkCommandBufferAllocateInfo ai = {0}; ai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    ai.commandPool = pool; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (pAllocCb(dev, &ai, &cmd) != VK_SUCCESS) { emit("vkAllocateCommandBuffers failed\n"); return 2; }
    VkCommandBufferBeginInfo bi = {0}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (pBegin(cmd, &bi) != VK_SUCCESS) { emit("vkBeginCommandBuffer failed\n"); return 2; }
    emit("cmdbuf recording: %p\n", (void*)cmd);

    // ---- NGX laden: GFR-first, dann Init_ProjectID (aus ngxfg_init_harness.c) ----
    HMODULE ngx = LoadLibraryA("_nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("nvngx.dll");
    if (!ngx) { emit("no _nvngx.dll beside exe\n"); return 3; }
    PFN_GFR_VK GFR = (PFN_GFR_VK)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    if (!cold && GFR) {
        NVSDK_NGX_FeatureDiscoveryInfo di = {0};
        di.SDKVersion = (NVSDK_NGX_Version)sdkv; di.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
        di.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        di.Identifier.v.ApplicationId = 0x1337ULL;
        di.ApplicationDataPath = L"Z:\\tmp"; di.FeatureInfo = NULL;
        NVSDK_NGX_FeatureRequirement rq; memset(&rq, 0, sizeof(rq));
        NVSDK_NGX_Result rg = GFR(inst, pd, &di, &rq);
        emit("pre-GFR(FG) result=0x%08X supported=0x%X\n", (unsigned)rg, (unsigned)rq.FeatureSupported);
    } else {
        emit("pre-GFR skipped (cold=%d hasGFR=%d)\n", cold, GFR != NULL);
    }
    PFN_INIT_PID Init = (PFN_INIT_PID)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Init_ProjectID");
    emit("Init_ProjectID export: %p\n", (void*)Init);
    if (!Init) { emit("no Init_ProjectID export\n"); return 3; }
    static const wchar_t* paths[1] = { L"Z:\\tmp" };
    NVSDK_NGX_FeatureCommonInfo fci; memset(&fci, 0, sizeof(fci));
    fci.PathListInfo.Path = paths; fci.PathListInfo.Length = 1;
    const NVSDK_NGX_FeatureCommonInfo* pinfo = (variant == 0) ? NULL : &fci;
    NVSDK_NGX_Result r = Init("a0b1c2d3-1234-5678-9abc-def012345678",
        NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0", L"Z:\\tmp",
        inst, pd, dev, gipa, gdpa, pinfo, (NVSDK_NGX_Version)sdkv);
    emit("Init_ProjectID(variant=%d) returned 0x%08X (0x1=SUCCESS)\n", variant, (unsigned)r);
    if (r != NVSDK_NGX_Result_Success) { emit("VERDICT: init-failed\n"); return 1; }

    // ---- Exporte fuer Params/Scratch/Create/Release/Shutdown aufloesen ----
    PFN_GETCAPS pGetCaps = (PFN_GETCAPS)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    PFN_ALLOC pAlloc = (PFN_ALLOC)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_AllocateParameters");
    PFN_DESTROYP pDestroy = (PFN_DESTROYP)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_DestroyParameters");
    PFN_SCRATCH pScratch = (PFN_SCRATCH)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetScratchBufferSize");
    PFN_CREATE pCreate = (PFN_CREATE)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_CreateFeature");
    PFN_RELEASE pRelease = (PFN_RELEASE)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_ReleaseFeature");
    PFN_SHUT1 pShut1 = (PFN_SHUT1)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Shutdown1");
    PFN_SHUT pShut = (PFN_SHUT)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_Shutdown");
    PFN_SETUI pSetUI = (PFN_SETUI)(void*)GetProcAddress(ngx, "NVSDK_NGX_Parameter_SetUI");
    emit("exports: GetCaps=%p Alloc=%p Destroy=%p Scratch=%p Create=%p Release=%p Shut1=%p Shut=%p SetUI=%p\n",
        (void*)pGetCaps, (void*)pAlloc, (void*)pDestroy, (void*)pScratch,
        (void*)pCreate, (void*)pRelease, (void*)pShut1, (void*)pShut, (void*)pSetUI);
    if (!pScratch || !pCreate || !pRelease || !pSetUI || (!pGetCaps && !pAlloc)) {
        emit("missing exports for create path\n"); return 3;
    }

    // ---- Stufe 1: Parameter-Map (GetCapabilityParameters bevorzugt: enthaelt u.a.
    // den VRAM-Schaetz-Callback, s. NGX_VK_ESTIMATE_VRAM_DLSSG-Kommentar) ----
    NVSDK_NGX_Parameter* params = NULL;
    NVSDK_NGX_Result rp = NVSDK_NGX_Result_FAIL_OutOfDate;
    if (pGetCaps) { rp = pGetCaps(&params); emit("GetCapabilityParameters -> 0x%08X params=%p\n", (unsigned)rp, (void*)params); }
    if ((!params || rp != NVSDK_NGX_Result_Success) && pAlloc) {
        params = NULL;
        rp = pAlloc(&params); emit("AllocateParameters(fallback) -> 0x%08X params=%p\n", (unsigned)rp, (void*)params);
    }
    if (rp != NVSDK_NGX_Result_Success || !params) { emit("VERDICT: params-failed\n"); return 1; }

    // ---- Stufe 2: exakt die 5 VK-Create-Parameter aus NGX_VK_CREATE_DLSSG
    // (nvsdk_ngx_helpers_dlssg_vk.h:53-57): Node-Masken 1/1 (Single-GPU) ----
    pSetUI(params, NVSDK_NGX_Parameter_CreationNodeMask, 1);
    pSetUI(params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    pSetUI(params, NVSDK_NGX_Parameter_Width, width);
    pSetUI(params, NVSDK_NGX_Parameter_Height, height);
    pSetUI(params, NVSDK_NGX_DLSSG_Parameter_BackbufferFormat, fmt);
    emit("params set: CreationNodeMask=1 VisibilityNodeMask=1 Width=%u Height=%u DLSSG.BackbufferFormat=%u\n",
        width, height, fmt);

    // ---- Stufe 3: Scratch-Groesse abfragen + loggen ----
    size_t scratchBytes = 0;
    NVSDK_NGX_Result rs = pScratch(NVSDK_NGX_Feature_FrameGeneration, params, &scratchBytes);
    emit("GetScratchBufferSize(FG) -> 0x%08X bytes=%llu\n", (unsigned)rs, (unsigned long long)scratchBytes);

    // ---- Stufe 4: CreateFeature-Versuch auf offenem Command-Buffer ----
    NVSDK_NGX_Handle* fgHandle = NULL;
    NVSDK_NGX_Result rc = pCreate(cmd, NVSDK_NGX_Feature_FrameGeneration, params, &fgHandle);
    emit("CreateFeature(FG %ux%u fmt=%u) -> 0x%08X handle=%p\n", width, height, fmt, (unsigned)rc, (void*)fgHandle);
    pEnd(cmd);
    emit("cmdbuf ended\n");
    {   // best-effort submit + wait, damit Create-Work die GPU sieht (Fehler nur loggen)
        PFN_vkQueueSubmit pSubmit = (PFN_vkQueueSubmit)gdpa(dev, "vkQueueSubmit");
        if (!pSubmit) pSubmit = (PFN_vkQueueSubmit)gipa(inst, "vkQueueSubmit");
        PFN_vkDeviceWaitIdle pWait = (PFN_vkDeviceWaitIdle)gdpa(dev, "vkDeviceWaitIdle");
        if (!pWait) pWait = (PFN_vkDeviceWaitIdle)gipa(inst, "vkDeviceWaitIdle");
        if (pSubmit && pWait) {
            VkSubmitInfo si = {0}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
            VkResult vr = pSubmit(queue, 1, &si, VK_NULL_HANDLE);
            emit("vkQueueSubmit -> %d\n", (int)vr);
            if (vr == VK_SUCCESS) { vr = pWait(dev); emit("vkDeviceWaitIdle -> %d\n", (int)vr); }
        } else {
            emit("submit/wait skipped (no fns)\n");
        }
    }

    // ---- Stufe 5: ReleaseFeature (nur bei erzeugtem Handle) + Param-Destroy ----
    if (fgHandle && rc == NVSDK_NGX_Result_Success) {
        NVSDK_NGX_Result rr = pRelease(fgHandle);
        emit("ReleaseFeature(handle=%p) -> 0x%08X\n", (void*)fgHandle, (unsigned)rr);
        fgHandle = NULL;
    } else {
        emit("ReleaseFeature skipped (no handle)\n");
    }
    if (pDestroy && params) {
        NVSDK_NGX_Result rd = pDestroy(params);
        emit("DestroyParameters -> 0x%08X\n", (unsigned)rd);
        params = NULL;
    }

    // ---- Stufe 6: Shutdown ----
    if (pShut1) {
        NVSDK_NGX_Result rh = pShut1(dev);
        emit("Shutdown1(dev) -> 0x%08X\n", (unsigned)rh);
    } else if (pShut) {
        NVSDK_NGX_Result rh = pShut();
        emit("Shutdown() -> 0x%08X\n", (unsigned)rh);
    } else {
        emit("no Shutdown export\n");
    }

    if (rc == NVSDK_NGX_Result_Success) { emit("VERDICT: create-ok\n"); return 0; }
    emit("VERDICT: create-failed rc=0x%08X\n", (unsigned)rc);
    return 1;
}
