#pragma once
#include "renderfx.h"
#include "nvofg.h"

namespace renderfx {
// Fill the capability table with per-build `supported` flags.
void buildCapabilities(bool nvofgOfa, bool nvofgShader, RfxCapabilities* out);

// A RenderFX-owned scratch image (e.g. the temporal-upscaler history ping-pong).
struct OwnedImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    uint32_t w = 0, h = 0;
};
}  // namespace renderfx

struct RfxContext {
    RfxCreateInfo info{};
    RfxCapabilities caps{};

    // The committed frame-generation backend (nvofg), created on rfx_commit.
    RfxBackendId fgBackend = RFX_BACKEND_NONE;
    NvofgContext* nvofg = nullptr;
    bool registered = false;
    uint64_t framesGenerated = 0;   // runtime statistics counter

    // The committed per-stage backends (dispatched by the record functions).
    RfxBackendId upscaleBackend = RFX_BACKEND_NONE;
    RfxBackendId rrBackend = RFX_BACKEND_NONE;

    // Native upscaling backend (built lazily, records into the app's command buffer).
    VkShaderModule        upSm = VK_NULL_HANDLE;
    VkDescriptorSetLayout upSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      upPipeLayout = VK_NULL_HANDLE;
    VkPipeline            upPipeline = VK_NULL_HANDLE;
    VkSampler             upSampler = VK_NULL_HANDLE;
    VkDescriptorPool      upPool = VK_NULL_HANDLE;
    VkDescriptorSet       upSet = VK_NULL_HANDLE;
    bool                  upReady = false;

    // Temporal (TAAU) backend: history ping-pong + dummy motion, owned by RenderFX.
    VkShaderModule        tSm = VK_NULL_HANDLE;
    VkDescriptorSetLayout tSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      tPipeLayout = VK_NULL_HANDLE;
    VkPipeline            tPipeline = VK_NULL_HANDLE;
    VkDescriptorPool      tPool = VK_NULL_HANDLE;
    VkDescriptorSet       tSet = VK_NULL_HANDLE;
    renderfx::OwnedImage  history[2];
    renderfx::OwnedImage  dummyMotion;
    bool                  tReady = false;
    uint32_t              frameParity = 0;
    bool                  historyValid = false;

    // RR/GBuffer debug visualisation pipeline (built lazily).
    VkShaderModule        dbgSm = VK_NULL_HANDLE;
    VkDescriptorSetLayout dbgSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      dbgPipeLayout = VK_NULL_HANDLE;
    VkPipeline            dbgPipeline = VK_NULL_HANDLE;
    VkDescriptorPool      dbgPool = VK_NULL_HANDLE;
    VkDescriptorSet       dbgSet = VK_NULL_HANDLE;
    renderfx::OwnedImage  dbgDummy;
    bool                  dbgReady = false;
};

namespace renderfx {
void destroyUpscale(RfxContext* ctx);
void destroyDebug(RfxContext* ctx);
}  // namespace renderfx
