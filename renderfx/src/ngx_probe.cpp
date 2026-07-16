// NGX init/capability probe (de-risking, like nvofg's M0): does NVIDIA NGX initialise
// on this device+driver, and does it report DLSS Super Resolution / DLAA and DLSS Ray
// Reconstruction as available? Mirrors RMC's proven extern-"C" NGX FFI + the vendored
// feature blobs under vendor/ngx/lib/Linux_x86_64/rel.
//
// Built only with -DRENDERFX_NGX. Exit 0 = NGX initialised (prints availability);
// 2 = no NVIDIA device / NGX unavailable (the backend then reports unsupported).
#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
int NVSDK_NGX_VULKAN_RequiredExtensions(unsigned* ic, const char*** ie, unsigned* dc, const char*** de);
int NVSDK_NGX_VULKAN_Init_with_ProjectID(const char* proj, unsigned engineType, const char* engineVer,
                                         const uint32_t* appDataPath, VkInstance, VkPhysicalDevice, VkDevice,
                                         PFN_vkGetInstanceProcAddr gipa, PFN_vkGetDeviceProcAddr gdpa,
                                         const void* featInfo, unsigned sdkVer);
int NVSDK_NGX_VULKAN_GetCapabilityParameters(void** params);
int NVSDK_NGX_Parameter_GetI(void* p, const char* name, int* out);
int NVSDK_NGX_VULKAN_Shutdown1(VkDevice);
}

static bool ngxOk(int r) { return (r & 0xFFF00000u) != 0xBAD00000u; }  // NVSDK_NGX_SUCCEED

// wchar_t is 4 bytes on Linux; NGX wants a wchar_t* data path.
static std::vector<uint32_t> wpath(const char* s) {
    std::vector<uint32_t> w; for (; *s; ++s) w.push_back((uint32_t)(unsigned char)*s); w.push_back(0); return w;
}

int main() {
    const char* dataPath = RENDERFX_NGX_DATA_PATH;  // vendored rel dir (feature blobs)

    // NGX-required instance/device extensions (before device creation).
    unsigned ic = 0, dc = 0; const char** ie = nullptr; const char** de = nullptr;
    NVSDK_NGX_VULKAN_RequiredExtensions(&ic, &ie, &dc, &de);
    std::vector<const char*> instExts(ie, ie + ic);
    instExts.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

    VkApplicationInfo app{}; app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{}; ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO; ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)instExts.size(); ici.ppEnabledExtensionNames = instExts.data();
    VkInstance inst; if (vkCreateInstance(&ici, nullptr, &inst) != VK_SUCCESS) { std::printf("no instance\n"); return 2; }

    uint32_t n = 0; vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n); vkEnumeratePhysicalDevices(inst, &n, pds.data());
    VkPhysicalDevice pd = VK_NULL_HANDLE; uint32_t fam = 0;
    for (auto c : pds) {
        VkPhysicalDeviceProperties p{}; vkGetPhysicalDeviceProperties(c, &p);
        if (p.vendorID != 0x10DE) continue;   // NVIDIA
        uint32_t qn = 0; vkGetPhysicalDeviceQueueFamilyProperties(c, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> q(qn); vkGetPhysicalDeviceQueueFamilyProperties(c, &qn, q.data());
        for (uint32_t i = 0; i < qn; ++i) if (q[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { pd = c; fam = i; break; }
        if (pd) break;
    }
    if (!pd) { std::printf("SKIP: no NVIDIA device (NGX unavailable -> backend unsupported)\n"); return 2; }
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(pd, &props);
    std::printf("device: %s\n", props.deviceName);

    float prio = 1.0f; VkDeviceQueueCreateInfo qci{}; qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = fam; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    std::vector<const char*> devExts(de, de + dc);
    VkDeviceCreateInfo dci{}; dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)devExts.size(); dci.ppEnabledExtensionNames = devExts.data();
    VkDevice dev; if (vkCreateDevice(pd, &dci, nullptr, &dev) != VK_SUCCESS) { std::printf("no device\n"); return 2; }

    auto gdpa = (PFN_vkGetDeviceProcAddr)vkGetInstanceProcAddr(inst, "vkGetDeviceProcAddr");
    std::vector<uint32_t> wp = wpath(dataPath);
    int r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        "renderfx-probe", 0 /*ENGINE_TYPE_CUSTOM*/, "0.1", wp.data(),
        inst, pd, dev, vkGetInstanceProcAddr, gdpa, nullptr, 0x0000015 /*NVSDK_NGX_Version_API*/);
    std::printf("NGX init result: 0x%08X (%s)\n", (unsigned)r, ngxOk(r) ? "ok" : "FAILED");
    if (!ngxOk(r)) { std::printf("RESULT: NGX unavailable on this device/driver -> backends report unsupported\n"); vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr); return 2; }

    void* params = nullptr;
    if (ngxOk(NVSDK_NGX_VULKAN_GetCapabilityParameters(&params)) && params) {
        int sr = 0, rr = 0;
        NVSDK_NGX_Parameter_GetI(params, "SuperSampling.Available", &sr);         // DLSS SR / DLAA
        NVSDK_NGX_Parameter_GetI(params, "SuperSamplingDenoising.Available", &rr); // DLSS Ray Reconstruction
        std::printf("  DLSS SR / DLAA available : %d\n", sr);
        std::printf("  DLSS Ray Reconstruction  : %d\n", rr);
        std::printf("RESULT: NGX initialised (SR=%d RR=%d)\n", sr, rr);
    }

    NVSDK_NGX_VULKAN_Shutdown1(dev);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
