// M1 lifecycle test: query_support, create, caps, destroy against the real device,
// plus the graceful-absence path on a non-OFA device. Exits non-zero on failure.
#include "nvofg.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define REQUIRE(cond, msg)                                                       \
    do {                                                                        \
        if (!(cond)) { std::fprintf(stderr, "FAIL: %s\n", msg); return 1; }     \
        std::printf("ok: %s\n", msg);                                           \
    } while (0)

static bool hasOfExt(VkPhysicalDevice pd) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    for (auto& x : e)
        if (std::strcmp(x.extensionName, VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0) return true;
    return false;
}

int main() {
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    VkInstance instance;
    if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS) {
        std::fprintf(stderr, "SKIP: no Vulkan instance\n");
        return 0;  // no Vulkan at all -> not a nvofg failure
    }

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(instance, &n, pds.data());

    VkPhysicalDevice ofDev = VK_NULL_HANDLE, nonOfDev = VK_NULL_HANDLE;
    for (auto pd : pds) (hasOfExt(pd) ? ofDev : nonOfDev) = pd;

    // --- graceful absence: query_support on a non-OFA device returns UNSUPPORTED ---
    if (nonOfDev) {
        NvofgCaps caps{};
        NvofgResult r = nvofg_query_support(instance, nonOfDev, vkGetInstanceProcAddr, &caps);
        REQUIRE(r == NVOFG_UNSUPPORTED && caps.supported == 0,
                "query_support on non-OFA device -> UNSUPPORTED");
    }

    if (!ofDev) {
        std::printf("SKIP: no OFA device on this machine (graceful-absence path only)\n");
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    // --- query_support reports the real caps ---
    NvofgCaps caps{};
    REQUIRE(nvofg_query_support(instance, ofDev, vkGetInstanceProcAddr, &caps) == NVOFG_OK,
            "query_support on OFA device -> OK");
    REQUIRE(caps.supported == 1, "caps.supported == 1");
    REQUIRE(caps.max_width >= 256 && caps.max_height >= 256, "caps max resolution sane");
    REQUIRE(caps.min_grid_size >= 1 && caps.max_grid_size <= 8, "caps grid range sane");
    std::printf("  caps: bidir=%u cost=%u hint=%u global=%u grid=[%u..%u] max=%ux%u\n",
                caps.bidirectional, caps.cost_map, caps.hint_support, caps.global_flow,
                caps.min_grid_size, caps.max_grid_size, caps.max_width, caps.max_height);

    // --- create a device with the required extension + feature ---
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ofDev, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(ofDev, &qfCount, qfs.data());
    uint32_t gfxFamily = 0;
    for (uint32_t i = 0; i < qfCount; ++i)
        if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { gfxFamily = i; break; }

    uint32_t ofFamily = 0;
    REQUIRE(nvofg_optical_flow_queue_family(instance, ofDev, vkGetInstanceProcAddr, &ofFamily)
                == NVOFG_OK, "nvofg_optical_flow_queue_family -> OK");

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qcis[2]{};
    qcis[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qcis[0].queueFamilyIndex = gfxFamily;
    qcis[0].queueCount = 1;
    qcis[0].pQueuePriorities = &prio;
    qcis[1] = qcis[0];
    qcis[1].queueFamilyIndex = ofFamily;
    uint32_t qciCount = (gfxFamily == ofFamily) ? 1u : 2u;
    VkPhysicalDeviceOpticalFlowFeaturesNV of{};
    of.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
    of.opticalFlow = VK_TRUE;
    uint32_t reqCount = 0;
    const char* const* reqExts = nvofg_required_device_extensions(&reqCount);
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = &of;
    dci.queueCreateInfoCount = qciCount;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = reqCount;
    dci.ppEnabledExtensionNames = reqExts;
    VkDevice device;
    REQUIRE(vkCreateDevice(ofDev, &dci, nullptr, &device) == VK_SUCCESS,
            "vkCreateDevice with nvofg_required_device_extensions");
    VkQueue queue, ofQueue;
    vkGetDeviceQueue(device, gfxFamily, 0, &queue);
    vkGetDeviceQueue(device, ofFamily, 0, &ofQueue);

    NvofgCreateInfo info{};
    info.instance = instance;
    info.physical_device = ofDev;
    info.device = device;
    info.queue = queue;
    info.queue_family_index = gfxFamily;
    info.of_queue = ofQueue;
    info.of_queue_family_index = ofFamily;
    info.gipa = vkGetInstanceProcAddr;
    info.width = 1920;
    info.height = 1080;
    info.quality = NVOFG_QUALITY_BALANCED;
    info.interpolator = NVOFG_INTERP_WARP;
    info.flags = 0;

    NvofgContext* ctx = nullptr;
    REQUIRE(nvofg_create(&info, &ctx) == NVOFG_OK && ctx != nullptr, "nvofg_create -> OK");

    NvofgCaps caps2{};
    REQUIRE(nvofg_caps(ctx, &caps2) == NVOFG_OK && caps2.supported == 1, "nvofg_caps -> OK");
    REQUIRE(nvofg_last_error(ctx)[0] == '\0', "no error after successful create");

    nvofg_destroy(ctx);
    std::printf("ok: nvofg_destroy\n");

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    std::printf("ALL LIFECYCLE CHECKS PASSED\n");
    return 0;
}
