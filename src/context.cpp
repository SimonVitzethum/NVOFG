// nvofg — context lifecycle, capability query, diagnostics.
#include "internal.hpp"

#include <cstring>
#include <new>

namespace {

// Load an instance-level function via the app's loader.
template <class Fn>
Fn loadInstance(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, const char* name) {
    return reinterpret_cast<Fn>(gipa(inst, name));
}

// Fill NvofgCaps from a physical device using only gipa-resolved core 1.1 calls.
// Returns true if a usable OFA is present.
bool queryCaps(PFN_vkGetInstanceProcAddr gipa, VkInstance inst, VkPhysicalDevice pd,
               NvofgCaps* out) {
    *out = NvofgCaps{};

    auto getFeatures2 = loadInstance<PFN_vkGetPhysicalDeviceFeatures2>(
        gipa, inst, "vkGetPhysicalDeviceFeatures2");
    auto getProps2 = loadInstance<PFN_vkGetPhysicalDeviceProperties2>(
        gipa, inst, "vkGetPhysicalDeviceProperties2");
    auto enumExts = loadInstance<PFN_vkEnumerateDeviceExtensionProperties>(
        gipa, inst, "vkEnumerateDeviceExtensionProperties");
    if (!getFeatures2 || !getProps2 || !enumExts) return false;

    // Extension present?
    uint32_t extCount = 0;
    enumExts(pd, nullptr, &extCount, nullptr);
    bool hasExt = false;
    if (extCount) {
        VkExtensionProperties* exts = new VkExtensionProperties[extCount];
        enumExts(pd, nullptr, &extCount, exts);
        for (uint32_t i = 0; i < extCount; ++i)
            if (std::strcmp(exts[i].extensionName, VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0)
                hasExt = true;
        delete[] exts;
    }
    if (!hasExt) return false;

    VkPhysicalDeviceOpticalFlowFeaturesNV feat{};
    feat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV;
    VkPhysicalDeviceFeatures2 f2{};
    f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f2.pNext = &feat;
    getFeatures2(pd, &f2);
    if (!feat.opticalFlow) return false;

    VkPhysicalDeviceOpticalFlowPropertiesNV op{};
    op.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_PROPERTIES_NV;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &op;
    getProps2(pd, &p2);

    out->supported     = 1;
    out->bidirectional = op.bidirectionalFlowSupported;
    out->cost_map      = op.costSupported;
    out->global_flow   = op.globalFlowSupported;
    out->hint_support  = op.hintSupported;
    out->max_width     = op.maxWidth;
    out->max_height    = op.maxHeight;
    // Grid range from the supported-output-grid mask.
    VkOpticalFlowGridSizeFlagsNV g = op.supportedOutputGridSizes;
    out->min_grid_size = (g & VK_OPTICAL_FLOW_GRID_SIZE_1X1_BIT_NV) ? 1
                       : (g & VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV) ? 2
                       : (g & VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV) ? 4 : 8;
    out->max_grid_size = (g & VK_OPTICAL_FLOW_GRID_SIZE_8X8_BIT_NV) ? 8
                       : (g & VK_OPTICAL_FLOW_GRID_SIZE_4X4_BIT_NV) ? 4
                       : (g & VK_OPTICAL_FLOW_GRID_SIZE_2X2_BIT_NV) ? 2 : 1;
    return true;
}

}  // namespace

extern "C" {

const char* const* nvofg_required_device_extensions(uint32_t* out_count) {
    static const char* kExts[] = { VK_NV_OPTICAL_FLOW_EXTENSION_NAME };
    if (out_count) *out_count = 1;
    return kExts;
}

NvofgResult nvofg_query_support(VkInstance instance, VkPhysicalDevice pd,
                                PFN_vkGetInstanceProcAddr gipa, NvofgCaps* out) {
    if (!instance || !pd || !gipa || !out) return NVOFG_INVALID_ARGUMENT;
    return queryCaps(gipa, instance, pd, out) ? NVOFG_OK : NVOFG_UNSUPPORTED;
}

NvofgResult nvofg_optical_flow_queue_family(VkInstance instance, VkPhysicalDevice pd,
                                            PFN_vkGetInstanceProcAddr gipa, uint32_t* out_family) {
    if (!instance || !pd || !gipa || !out_family) return NVOFG_INVALID_ARGUMENT;
    auto getQF = loadInstance<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
        gipa, instance, "vkGetPhysicalDeviceQueueFamilyProperties");
    if (!getQF) return NVOFG_INTERNAL;
    uint32_t n = 0;
    getQF(pd, &n, nullptr);
    if (!n) return NVOFG_UNSUPPORTED;
    VkQueueFamilyProperties* qf = new VkQueueFamilyProperties[n];
    getQF(pd, &n, qf);
    NvofgResult res = NVOFG_UNSUPPORTED;
    for (uint32_t i = 0; i < n; ++i)
        if (qf[i].queueFlags & VK_QUEUE_OPTICAL_FLOW_BIT_NV) { *out_family = i; res = NVOFG_OK; break; }
    delete[] qf;
    return res;
}

NvofgResult nvofg_create(const NvofgCreateInfo* info, NvofgContext** out) {
    if (!info || !out || !info->device || !info->physical_device || !info->instance
        || !info->gipa || !info->width || !info->height)
        return NVOFG_INVALID_ARGUMENT;
    *out = nullptr;

    NvofgCaps caps{};
    if (!queryCaps(info->gipa, info->instance, info->physical_device, &caps))
        return NVOFG_UNSUPPORTED;

    auto* ctx = new (std::nothrow) NvofgContext();
    if (!ctx) return NVOFG_OUT_OF_MEMORY;

    ctx->instance       = info->instance;
    ctx->physicalDevice = info->physical_device;
    ctx->device         = info->device;
    ctx->queue          = info->queue;
    ctx->queueFamily    = info->queue_family_index;
    ctx->gipa           = info->gipa;
    ctx->width          = info->width;
    ctx->height         = info->height;
    ctx->quality        = info->quality;
    ctx->interpolator   = info->interpolator;
    ctx->mode           = info->mode;
    ctx->flags          = info->flags;
    ctx->caps           = caps;
    ctx->ofFamily       = info->of_queue_family_index;
    ctx->ofQueue        = info->of_queue;

    // Resolve device-level entrypoints via the app's loader.
    auto gdpa = loadInstance<PFN_vkGetDeviceProcAddr>(
        info->gipa, info->instance, "vkGetDeviceProcAddr");
    if (!gdpa) { delete ctx; return NVOFG_INTERNAL; }
    auto L = [&](const char* n) { return gdpa(info->device, n); };
    // The image-formats query is a physical-device function -> load at instance level.
    ctx->of.getImageFormats = (PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV)
        info->gipa(info->instance, "vkGetPhysicalDeviceOpticalFlowImageFormatsNV");
    ctx->of.createSession  = (PFN_vkCreateOpticalFlowSessionNV)  L("vkCreateOpticalFlowSessionNV");
    ctx->of.destroySession = (PFN_vkDestroyOpticalFlowSessionNV) L("vkDestroyOpticalFlowSessionNV");
    ctx->of.bindImage      = (PFN_vkBindOpticalFlowSessionImageNV)L("vkBindOpticalFlowSessionImageNV");
    ctx->of.cmdExecute     = (PFN_vkCmdOpticalFlowExecuteNV)     L("vkCmdOpticalFlowExecuteNV");
    if (!ctx->of.complete()) {
        ctx->setError("failed to resolve VK_NV_optical_flow entrypoints");
        delete ctx;
        return NVOFG_INTERNAL;
    }

    // Resolve the optical-flow queue if the app did not pass one explicitly.
    if (ctx->ofQueue == VK_NULL_HANDLE) {
        auto getQueue = (PFN_vkGetDeviceQueue) L("vkGetDeviceQueue");
        if (getQueue) getQueue(info->device, ctx->ofFamily, 0, &ctx->ofQueue);
    }
    if (ctx->ofQueue == VK_NULL_HANDLE) {
        ctx->setError("no optical-flow queue: pass of_queue or create one from "
                      "nvofg_optical_flow_queue_family()");
        delete ctx;
        return NVOFG_INVALID_ARGUMENT;
    }

    // Cross-stage timeline semaphore.
    auto createSem = (PFN_vkCreateSemaphore) L("vkCreateSemaphore");
    if (!createSem) { delete ctx; return NVOFG_INTERNAL; }
    VkSemaphoreTypeCreateInfo st{};
    st.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    st.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    st.initialValue = 0;
    VkSemaphoreCreateInfo sci{};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &st;
    if (createSem(info->device, &sci, nullptr, &ctx->timeline) != VK_SUCCESS) {
        ctx->setError("failed to create timeline semaphore");
        delete ctx;
        return NVOFG_INTERNAL;
    }

    *out = ctx;
    return NVOFG_OK;
}

void nvofg_destroy(NvofgContext* ctx) {
    if (!ctx) return;
    if (ctx->device) vkDeviceWaitIdle(ctx->device);
    if (ctx->pipelineReady) nvofg::destroyPipeline(ctx);
    if (ctx->timeline) vkDestroySemaphore(ctx->device, ctx->timeline, nullptr);
    delete ctx;
}

NvofgResult nvofg_caps(NvofgContext* ctx, NvofgCaps* out) {
    if (!ctx || !out) return NVOFG_INVALID_ARGUMENT;
    *out = ctx->caps;
    return NVOFG_OK;
}

const char* nvofg_last_error(NvofgContext* ctx) {
    return ctx ? ctx->lastError.c_str() : "";
}

NvofgResult nvofg_set_debug_view(NvofgContext* ctx, NvofgDebugView view,
                                 const NvofgImageDesc* /*target*/) {
    if (!ctx) return NVOFG_INVALID_ARGUMENT;
    ctx->debugView = view;
    return NVOFG_OK;
}

}  // extern "C"
