// Gate 3 oracle: does the native Linux NVIDIA driver report DLSS Frame Generation as
// AVAILABLE when queried through the COMPLETE Windows stack (dxvk-nvapi + vkd3d + dxgi)
// under Proton? A Windows x64 .exe (mingw), run under GE-Proton against the real driver +
// RTX 5070. Uses NVSDK_NGX_VULKAN_GetFeatureRequirements(FeatureID), which needs only a
// VkInstance + VkPhysicalDevice (no device, no full Init, no adapter correlation) and
// returns FeatureSupported: 0 = supported; else a bitfield (DriverVersionUnsupported=2,
// AdapterUnsupported=4, OSVersionBelow=8, NotImplemented=16). This is the one number the
// whole Path-B chain has been running toward: if FG is NotImplemented here, the driver
// lacks the GPU-side FG path and no amount of native adapter-layer rebuild can heal it.
//
// Build:  x86_64-w64-mingw32-gcc -O2 -o ngxfg_probe.exe ngxfg_probe.c -I<vulkan-headers> -I<ngx-headers>
// Run:    under GE-Proton (see run_proton.sh), with nvngx.dll beside the exe.
#include <windows.h>
#include <stdio.h>
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>
#include "nvsdk_ngx_defs.h"
#include <stdarg.h>
static FILE* g_rf;
static void emit(const char* f,...){ va_list a; va_start(a,f); vprintf(f,a); va_end(a); if(g_rf){ va_start(a,f); vfprintf(g_rf,f,a); va_end(a); fflush(g_rf);} }

typedef NVSDK_NGX_Result (*PFN_GFR)(VkInstance, VkPhysicalDevice,
    const NVSDK_NGX_FeatureDiscoveryInfo*, NVSDK_NGX_FeatureRequirement*);

int main(void) {
    g_rf=fopen("ngxfg_result.txt","w");
    // --- Vulkan via the loader (DXVK under Proton) ---
    HMODULE vk = LoadLibraryA("vulkan-1.dll");
    if (!vk) { emit("no vulkan-1.dll\n"); return 2; }
    PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(vk, "vkGetInstanceProcAddr");
    PFN_vkCreateInstance vkCreateInstance = (PFN_vkCreateInstance)gipa(NULL, "vkCreateInstance");
    VkApplicationInfo app = {0}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    const char* ie[] = { "VK_KHR_get_physical_device_properties2" };
    VkInstanceCreateInfo ici = {0}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = ie;
    VkInstance inst; if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { emit("vkCreateInstance failed\n"); return 2; }
    PFN_vkEnumeratePhysicalDevices vkEnumeratePhysicalDevices = (PFN_vkEnumeratePhysicalDevices)gipa(inst, "vkEnumeratePhysicalDevices");
    PFN_vkGetPhysicalDeviceProperties vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)gipa(inst, "vkGetPhysicalDeviceProperties");
    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice pds[8]; if (n > 8) n = 8; vkEnumeratePhysicalDevices(inst, &n, pds);
    VkPhysicalDevice pd = VK_NULL_HANDLE;
    for (uint32_t i = 0; i < n; ++i) { VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(pds[i], &p);
        emit("  vk device: vendor=0x%04X device=0x%04X %s\n", p.vendorID, p.deviceID, p.deviceName);
        if (p.vendorID == 0x10DE && pd == VK_NULL_HANDLE) pd = pds[i]; }
    if (!pd) { emit("no NVIDIA vk device\n"); return 2; }

    // --- NGX host ---
    HMODULE ngx = LoadLibraryA("_nvngx.dll");
    if (!ngx) ngx = LoadLibraryA("nvngx.dll");
    emit("  nvngx loaded: %p\n",(void*)ngx);
    if (!ngx) { emit("no _nvngx.dll beside exe\n"); return 3; }
    PFN_GFR GFR = (PFN_GFR)(void*)GetProcAddress(ngx, "NVSDK_NGX_VULKAN_GetFeatureRequirements");
    emit("  GetFeatureRequirements export: %p\n",(void*)GFR);
    if (!GFR) { emit("no GetFeatureRequirements export\n"); return 3; }

    struct { const char* name; NVSDK_NGX_Feature id; } feats[] = {
        {"SuperSampling(DLSS)", (NVSDK_NGX_Feature)1},
        {"FrameGeneration",     (NVSDK_NGX_Feature)11},
        {"RayReconstruction",   (NVSDK_NGX_Feature)13},
    };
    for (int i = 0; i < 3; ++i) {
        NVSDK_NGX_FeatureDiscoveryInfo info = {0};
        info.SDKVersion = NVSDK_NGX_Version_API;
        info.FeatureID = feats[i].id;
        info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Application_Id;
        info.Identifier.v.ApplicationId = 0x0000000000001337ULL;
        info.ApplicationDataPath = L"Z:\\tmp";
        info.FeatureInfo = NULL;
        NVSDK_NGX_FeatureRequirement req; memset(&req, 0, sizeof(req));
        NVSDK_NGX_Result r = GFR(inst, pd, &info, &req);
        emit("  %-20s result=0x%08X FeatureSupported=0x%X MinHWArch=0x%X MinOS=%.32s\n",
               feats[i].name, (unsigned)r, (unsigned)req.FeatureSupported, req.MinHWArchitecture, req.MinOSVersion);
    }
    emit("VERDICT: FrameGeneration FeatureSupported==0 => Gate 3 GREEN (driver reports FG available);\n");
    emit("         ==16 NotImplemented => Gate 3 RED (driver lacks FG path); ==4 AdapterUnsupported; ==2 DriverVersion.\n");
    return 0;
}
