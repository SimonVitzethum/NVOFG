// nvofg internal definitions (not part of the public ABI).
#pragma once
#include <vulkan/vulkan.h>
#include "nvofg.h"

#include <string>

namespace nvofg {

// Optical-flow extension entrypoints, resolved via the app's loader (gipa).
struct OfDispatch {
    PFN_vkGetPhysicalDeviceOpticalFlowImageFormatsNV getImageFormats = nullptr;
    PFN_vkCreateOpticalFlowSessionNV  createSession  = nullptr;
    PFN_vkDestroyOpticalFlowSessionNV destroySession = nullptr;
    PFN_vkBindOpticalFlowSessionImageNV bindImage    = nullptr;
    PFN_vkCmdOpticalFlowExecuteNV     cmdExecute     = nullptr;
    bool complete() const {
        return getImageFormats && createSession && destroySession && bindImage && cmdExecute;
    }
};

}  // namespace nvofg

// The opaque context (C ABI forward-declares `struct NvofgContext`).
struct NvofgContext {
    // --- caller-provided handles / config ---
    VkInstance        instance = VK_NULL_HANDLE;
    VkPhysicalDevice  physicalDevice = VK_NULL_HANDLE;
    VkDevice          device = VK_NULL_HANDLE;
    VkQueue           queue = VK_NULL_HANDLE;     // app/compute queue nvofg submits on
    uint32_t          queueFamily = 0;
    PFN_vkGetInstanceProcAddr gipa = nullptr;
    uint32_t          width = 0, height = 0;
    NvofgQuality      quality = NVOFG_QUALITY_BALANCED;
    NvofgInterpolator interpolator = NVOFG_INTERP_WARP;
    NvofgMode         mode = NVOFG_MODE_AUTOMATIC;
    uint32_t          flags = 0;

    // --- optical-flow queue (dedicated family) ---
    VkQueue  ofQueue = VK_NULL_HANDLE;
    uint32_t ofFamily = 0;

    // --- resolved dispatch + capabilities ---
    nvofg::OfDispatch of;
    NvofgCaps         caps{};

    // --- cross-stage timeline semaphore (§4 sync contract) ---
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t    timelineValue = 0;

    // --- diagnostics ---
    std::string lastError;
    NvofgDebugView debugView = NVOFG_DEBUG_NONE;

    void setError(const char* msg) { lastError = msg ? msg : ""; }
};
