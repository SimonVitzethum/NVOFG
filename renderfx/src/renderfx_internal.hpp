#pragma once
#include "renderfx.h"
#include "nvofg.h"

struct RfxContext;  // global fwd (full definition below)

namespace renderfx {
// Fill the capability table with per-build `supported` flags. NGX SR/RR and FSR/XeSS
// availability are probed at rfx_create; defaults keep older callers (tests) working.
void buildCapabilities(bool nvofgOfa, bool nvofgShader, RfxCapabilities* out,
                       bool ngxSr = false, bool ngxRr = false,
                       bool fsr = false, bool xess = false);

// --- Official NVIDIA NGX backend (DLSS SR/DLAA/Ray Reconstruction). Real when built
// with -DRENDERFX_NGX + the vendored SDK; otherwise inert stubs (ADR 0006, design §16). ---
// Init NGX on ctx's device (design.md §19 writable-path + UUID fix) and report which
// features this device+driver expose. Returns false if NGX is unavailable (then the
// backends stay unsupported and the pipeline falls back — graceful degradation G6).
bool ngxInit(RfxContext* ctx, bool* srAvail, bool* rrAvail);
void ngxShutdown(RfxContext* ctx);
// DLAA: native-res anti-aliasing (color+depth+motion+jitter). dst == render res.
RfxResult ngxRecordDLAA(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                        const RfxImageDesc* dst, uint32_t reset);
// Ray Reconstruction (DLSS-D): denoise + upscale consuming the GBuffer.
RfxResult ngxRecordRR(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                      const RfxImageDesc* output, uint32_t reset);

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

    // Official NGX backend state (opaque renderfx::NgxState*, owned by ngx.cpp). NGX
    // availability probed once at rfx_create; features created lazily on first record.
    void* ngx = nullptr;
    bool  ngxSr = false;   // DLSS Super Resolution + DLAA available on this device
    bool  ngxRr = false;   // DLSS Ray Reconstruction available on this device

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

    // FSR (EASU-style) spatial upscaler: src(sampled, via Load) + dst(storage), 16B push.
    VkShaderModule        fsrSm = VK_NULL_HANDLE;
    VkDescriptorSetLayout fsrSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      fsrPipeLayout = VK_NULL_HANDLE;
    VkPipeline            fsrPipeline = VK_NULL_HANDLE;
    VkDescriptorPool      fsrPool = VK_NULL_HANDLE;
    VkDescriptorSet       fsrSet = VK_NULL_HANDLE;
    bool                  fsrReady = false;

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
