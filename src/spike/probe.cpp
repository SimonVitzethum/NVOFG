// nvofg M0 support probe.
//
// Proves the *native* OFA path on this machine end to end, up to (not including)
// a flow execute: create a VkInstance/VkDevice, find a physical device that
// exposes VK_NV_optical_flow, enable the feature, locate an optical-flow-capable
// queue, and enumerate the OFA-supported image formats per usage. Prints the
// hardware's optical-flow capabilities.
//
// This deliberately uses *only* the Vulkan loader — no dlopen of
// libnvidia-opticalflow.so, no NVIDIA SDK headers — to validate the design
// decision that VK_NV_optical_flow alone is sufficient (see ADR 0001).
//
// Exit code 0 = a usable OFA was found and queried; 2 = none found (the library
// would return NVOFG_UNSUPPORTED here and the app would run at 1x).

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

const char* vkres(VkResult r) {
    switch (r) {
        case VK_SUCCESS: return "VK_SUCCESS";
        case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
        case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
        case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        default: return "VK_ERROR_<other>";
    }
}

#define CHECK(expr)                                                              \
    do {                                                                        \
        VkResult _r = (expr);                                                   \
        if (_r != VK_SUCCESS) {                                                 \
            std::fprintf(stderr, "%s failed: %s\n", #expr, vkres(_r));          \
            return 1;                                                           \
        }                                                                       \
    } while (0)

bool has_ext(const std::vector<VkExtensionProperties>& exts, const char* name) {
    for (const auto& e : exts)
        if (std::strcmp(e.extensionName, name) == 0) return true;
    return false;
}

void print_grid_sizes(const char* label, VkOpticalFlowGridSizeFlagsNV g) {
    std::printf("    %s:", label);
    if (g & VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV) std::printf(" 1x1");
    if (g & VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV) std::printf(" 2x2");
    if (g & VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV) std::printf(" 4x4");
    if (g & VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV) std::printf(" 8x8");
    std::printf("\n");
}

void query_formats(PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV fn,
                   VkPhysicalDevice pd, const char* label,
                   VkOpticalFlowUsageFlagsNV usage) {
    VkOpticalFlowImageFormatInfoNV info{};
    info.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_INFO_NV;
    info.usage = usage;

    uint32_t count = 0;
    VkResult r = fn(pd, &info, &count, nullptr);
    if (r != VK_SUCCESS || count == 0) {
        std::printf("    %-10s: (none)\n", label);
        return;
    }
    std::vector<VkOpticalFlowImageFormatPropertiesNV> props(count);
    for (auto& p : props) p.sType = VK_STRUCTURE_TYPE_OPTICAL_FLOW_IMAGE_FORMAT_PROPERTIES_NV;
    fn(pd, &info, &count, props.data());

    std::printf("    %-10s:", label);
    for (uint32_t i = 0; i < count; ++i) std::printf(" fmt=%d", (int)props[i].format);
    std::printf("\n");
}

}  // namespace

int main() {
    // ---- instance --------------------------------------------------------
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "nvofg_probe";
    app.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;

    VkInstance instance = VK_NULL_HANDLE;
    CHECK(vkCreateInstance(&ici, nullptr, &instance));

    auto pGetFormats = (PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV)
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceOpticalFlowImageFormatsNV");

    uint32_t pd_count = 0;
    CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, nullptr));
    std::vector<VkPhysicalDevice> pds(pd_count);
    CHECK(vkEnumeratePhysicalDevices(instance, &pd_count, pds.data()));

    int usable = 0;

    for (VkPhysicalDevice pd : pds) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);

        uint32_t ext_count = 0;
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, nullptr);
        std::vector<VkExtensionProperties> exts(ext_count);
        vkEnumerateDeviceExtensionProperties(pd, nullptr, &ext_count, exts.data());

        bool of_ext = has_ext(exts, VK_NV_OPTICAL_FLOW_EXTENSION_NAME);
        std::printf("GPU: %s\n", props.deviceName);
        std::printf("  VK_NV_optical_flow: %s\n", of_ext ? "present" : "ABSENT");
        if (!of_ext) { std::printf("\n"); continue; }

        // ---- feature + properties chain ----------------------------------
        VkPhysicalDeviceOpticalFlowFeaturesNV of_feat{};
        of_feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
        VkPhysicalDeviceFeatures2 feat2{};
        feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        feat2.pNext = &of_feat;
        vkGetPhysicalDeviceFeatures2(pd, &feat2);

        VkPhysicalDeviceOpticalFlowPropertiesNV of_props{};
        of_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV;
        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &of_props;
        vkGetPhysicalDeviceProperties2(pd, &props2);

        std::printf("  opticalFlow feature: %s\n", of_feat.opticalFlow ? "true" : "false");
        std::printf("  hintSupported=%u costSupported=%u bidirectional=%u globalFlow=%u\n",
                    of_props.hintSupported, of_props.costSupported,
                    of_props.bidirectionalFlowSupported, of_props.globalFlowSupported);
        std::printf("  resolution: min %ux%u  max %ux%u  maxROIs=%u\n",
                    of_props.minWidth, of_props.minHeight, of_props.maxWidth,
                    of_props.maxHeight, of_props.maxNumRegionsOfInterest);
        print_grid_sizes("output grid sizes", of_props.supportedOutputGridSizes);
        print_grid_sizes("hint grid sizes  ", of_props.supportedHintGridSizes);

        // ---- optical-flow-capable queue family ---------------------------
        uint32_t qf_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qf_count);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qf_count, qfs.data());
        int of_family = -1;
        for (uint32_t i = 0; i < qf_count; ++i)
            if (qfs[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) { of_family = (int)i; break; }
        std::printf("  optical-flow queue family: %d\n", of_family);

        // ---- create a device with the extension + feature enabled --------
        if (of_feat.opticalFlow && of_family >= 0) {
            float prio = 1.0f;
            VkDeviceQueueCreateInfo qci{};
            qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            qci.queueFamilyIndex = (uint32_t)of_family;
            qci.queueCount = 1;
            qci.pQueuePriorities = &prio;

            const char* dev_exts[] = { VK_NV_OPTICAL_FLOW_EXTENSION_NAME };

            VkPhysicalDeviceOpticalFlowFeaturesNV enable_of{};
            enable_of.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
            enable_of.opticalFlow = VK_TRUE;
            VkPhysicalDeviceSynchronization2Features sync2{};
            sync2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES;
            sync2.synchronization2 = VK_TRUE;
            enable_of.pNext = &sync2;

            VkDeviceCreateInfo dci{};
            dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            dci.pNext = &enable_of;
            dci.queueCreateInfoCount = 1;
            dci.pQueueCreateInfos = &qci;
            dci.enabledExtensionCount = 1;
            dci.ppEnabledExtensionNames = dev_exts;

            VkDevice device = VK_NULL_HANDLE;
            VkResult dr = vkCreateDevice(pd, &dci, nullptr, &device);
            if (dr != VK_SUCCESS) {
                std::printf("  vkCreateDevice(optical flow): %s\n", vkres(dr));
            } else {
                std::printf("  device with VK_NV_optical_flow created OK\n");
                if (pGetFormats) {
                    std::printf("  OFA-supported image formats per usage:\n");
                    query_formats(pGetFormats, pd, "input",  VK_OPTICAL_FLOW_USAGE_INPUT_BIT_NV);
                    query_formats(pGetFormats, pd, "output", VK_OPTICAL_FLOW_USAGE_OUTPUT_BIT_NV);
                    query_formats(pGetFormats, pd, "hint",   VK_OPTICAL_FLOW_USAGE_HINT_BIT_NV);
                    query_formats(pGetFormats, pd, "cost",   VK_OPTICAL_FLOW_USAGE_COST_BIT_NV);
                }
                vkDestroyDevice(device, nullptr);
                usable = 1;
            }
        }
        std::printf("\n");
    }

    vkDestroyInstance(instance, nullptr);

    if (usable) {
        std::printf("RESULT: usable OFA found -> nvofg would run hardware frame generation.\n");
        return 0;
    }
    std::printf("RESULT: no usable OFA -> nvofg_create would return NVOFG_UNSUPPORTED (1x).\n");
    return 2;
}
