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

// An owned scratch image + its backing memory + default view.
struct Image {
    VkImage        image  = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view   = VK_NULL_HANDLE;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    uint32_t       w = 0, h = 0;
};

struct Buffer {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize   size   = 0;
};

// A registered (app-owned) image handle.
struct RegImage {
    VkImage     image  = VK_NULL_HANDLE;
    VkImageView view   = VK_NULL_HANDLE;
    VkFormat    format = VK_FORMAT_UNDEFINED;
    uint32_t    w = 0, h = 0;
};

// One compute stage: shader + layouts + pipeline.
struct Stage {
    VkShaderModule        module     = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout  = VK_NULL_HANDLE;
    VkPipelineLayout      pipeLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline   = VK_NULL_HANDLE;
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

    // --- flow grid config ---
    uint32_t gridSize = 4, gridW = 0, gridH = 0;

    // --- OFA session + scratch resources (built lazily once registered) ---
    VkOpticalFlowSessionNV session = VK_NULL_HANDLE;
    nvofg::Image  lumaPrev, lumaCurr;    // R8 OFA input/reference (prep writes)
    nvofg::Image  flowImg, costImg;      // OFA outputs (grid res)
    nvofg::Buffer flowBuf, costBuf;      // flow/cost copied out for shader reads
    nvofg::Image  refinedFlow, confidence;

    // --- registered app images ---
    nvofg::RegImage prevColor, currColor, output;
    bool haveColor = false, haveOutput = false;

    // --- optional aux (registered; consumed per flags) ---
    nvofg::RegImage depth, motion, uiMask, reactive, materialId;
    bool hasDepth = false, hasMotion = false, hasUiMask = false, hasReactive = false;

    // --- fallback 1x1 images bound where an aux input is absent ---
    nvofg::Image dummyR8, dummyRG16;

    // --- bidirectional (BOTH_DIRECTIONS) resources ---
    bool bidir = false;
    nvofg::Image  flowBwdImg, costBwdImg;
    nvofg::Buffer flowBwdBuf, costBwdBuf;
    nvofg::Image  refinedFlowBwd, occlusion;

    // --- motion-vector hint (ENABLE_HINT) ---
    bool useHint = false;
    nvofg::Image  hintImg;              // SFIXED5 at hint grid
    nvofg::Stage  hintStage;           // converts app MVs -> S10.5 hint
    VkDescriptorSet hintSet = VK_NULL_HANDLE;

    // --- debug visualisation ---
    nvofg::RegImage debugTarget;
    bool haveDebugTarget = false;
    nvofg::Stage  debugStage;
    VkDescriptorSet debugSet = VK_NULL_HANDLE;

    // --- compute stages ---
    nvofg::Stage prepStage, refineStage, warpStage;
    VkSampler        linearSampler = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet  prepPrevSet = VK_NULL_HANDLE, prepCurrSet = VK_NULL_HANDLE;
    VkDescriptorSet  refineSet = VK_NULL_HANDLE, warpSet = VK_NULL_HANDLE;
    VkCommandPool    computePool = VK_NULL_HANDLE, ofPool = VK_NULL_HANDLE;
    bool pipelineReady = false;

    // --- diagnostics ---
    std::string lastError;
    NvofgDebugView debugView = NVOFG_DEBUG_NONE;

    void setError(const char* msg) { lastError = msg ? msg : ""; }
};

namespace nvofg {
// Build the OFA session, scratch resources, and compute pipelines once colors +
// output are registered. Idempotent. Tears everything down on destroy.
NvofgResult ensurePipeline(NvofgContext* ctx);
void        destroyPipeline(NvofgContext* ctx);
}  // namespace nvofg
