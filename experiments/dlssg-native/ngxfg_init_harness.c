// Wine-first Init harness (method: validate under Proton BEFORE fixing natively).
// Minimal Windows x64 .exe (mingw) run under GE-Proton against the real driver:
// VkInstance + VkPhysicalDevice + REAL VkDevice, then
// NVSDK_NGX_VULKAN_Init_ProjectID with the SAME argument values s5_host passes
// natively (UUID-a0b1/engine-CUSTOM/"1.0", dataPath, FeatureInfo variant), then
// NVSDK_NGX_VULKAN_GetCapabilityParameters. If this returns 0x1 under Proton,
// our native failures are purely loader-side; if it fails too, our args are wrong.
//
// Build:  x86_64-w64-mingw32-gcc -O2 -o ngxfg_init_harness.exe ngxfg_init_harness.c \
//           -I<vulkan-headers> -I<ngx-headers>   (see run_init_proton.sh)
// Run:    ./run_init_proton.sh [0|1]   (0 = FeatureInfo NULL, 1 = &FeatureCommonInfo)
#include <windows.h>
#include <stdio.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "nvsdk_ngx_defs.h"
#include "nvsdk_ngx.h"
#include <stdarg.h>
static FILE* g_rf;
static void emit(const char* f,...){ va_list a; va_start(a,f); vprintf(f,a); va_end(a); if(g_rf){ va_start(a,f); vfprintf(g_rf,f,a); va_end(a); fflush(g_rf);} }

typedef NVSDK_NGX_Result (*PFN_INIT_PID)(const char*, NVSDK_NGX_EngineType, const char*,
    const wchar_t*, VkInstance, VkPhysicalDevice, VkDevice,
    PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr,
    const NVSDK_NGX_FeatureCommonInfo*, NVSDK_NGX_Version);
typedef NVSDK_NGX_Result (*PFN_REQ)(const NVSDK_NGX_FeatureDiscoveryInfo*,
    unsigned* /*OutExtensionCount*/, void** /*OutExtensionProperties*/);

int main(int argc, char** argv) {
    int variant = (argc > 1) ? atoi(argv[1]) : 1;
    unsigned sdkv = (argc > 2) ? (unsigned)strtoul(argv[2], 0, 0) : (unsigned)NVSDK_NGX_Version_API;
    g_rf = fopen("ngxfg_init_result.txt", "w");
    emit("variant=%d sdkv=0x%x\n", variant, sdkv);
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
    // DLSS-G is a CUDA module sharing images with Vulkan: enable the interop
    // extensions NGX reports plus the external memory/semaphore FD set —
    // but only those the device actually enumerates (blind enabling fails).
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
    for (int w = 0; w < 6; ++w) {
        int ok = 0; for (uint32_t h = 0; h < nhave; ++h) if (have[h] == want[w]) ok = 1;
        emit("  [%c] %s\n", ok ? 'x' : ' ', want[w]);
    }
    VkDeviceCreateInfo dci = {0}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nhave; dci.ppEnabledExtensionNames = have;
    VkDevice dev; if (vkCreateDevice(pd, &dci, NULL, &dev) != VK_SUCCESS) { emit("vkCreateDevice failed\n"); return 2; }
    VkQueue queue; vkGetDQ(dev, qfam, 0, &queue);
    PFN_vkGetDeviceProcAddr gdpa = (PFN_vkGetDeviceProcAddr)gipa(inst, "vkGetDeviceProcAddr");
    emit("vulkan ok: inst=%p pd=%p dev=%p\n", (void*)inst, (void*)pd, (void*)dev);

    HMODULE ngx = LoadLibraryA("_nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("nvngx.dll");
    if (!ngx) { emit("no _nvngx.dll beside exe\n"); return 3; }
    // Mirror s5_host order: GFR(FrameGeneration) FIRST (populates NGX globals),
    // then Init. argv[3]=1 skips the GFR pre-call (cold-Init control).
    int cold = (argc > 3) ? atoi(argv[3]) : 0;
    typedef NVSDK_NGX_Result (*PFN_GFR)(VkInstance, VkPhysicalDevice,
        const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);
    PFN_GFR GFR = (PFN_GFR)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    if (!cold && GFR) {
        NVSDK_NGX_FeatureDiscoveryInfo di = {0};
        di.SDKVersion = (NVSDK_NGX_Version)sdkv; di.FeatureID = NVSDK_NGX_Feature_FrameGeneration;
        di.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        di.Identifier.v.ApplicationId = 0x1337ULL;
        di.ApplicationDataPath = L"Z:\\tmp"; di.FeatureInfo = NULL;
        NVSDK_NGX_FeatureRequirement rq; memset(&rq, 0, sizeof(rq));
        NVSDK_NGX_Result rg = GFR(inst, pd, &di, &rq);
        emit("pre-GFR(FG) result=0x%08X supported=0x%X\n", (unsigned)rg, (unsigned)rq.FeatureSupported);
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
    return (r == NVSDK_NGX_Result_Success) ? 0 : 1;
}
