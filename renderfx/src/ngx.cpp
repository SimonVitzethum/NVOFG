// Official NVIDIA NGX backend for RenderFX: DLAA (native-res AA, DLSS Super Resolution
// family) and DLSS Ray Reconstruction (DLSS-D). Uses the vendored NGX SDK's inline VK
// helpers (NGX_VULKAN_CREATE/EVALUATE_DLSS[D]_EXT) so param packing matches NVIDIA's
// reference exactly. Real only with -DRENDERFX_NGX + the vendored SDK; otherwise this
// file compiles to inert stubs so context.cpp stays branch-free (ADR 0006, design §16).
//
// NGX is the *only* correct DLAA/RR path (design §16.1: model-not-hardware asymmetry —
// no native reimplementation). Where NGX cannot init, the backends report unsupported
// and the resolver never selects them (graceful degradation G6).
#include "renderfx_internal.hpp"

#ifndef RENDERFX_NGX
// ---------------------------------------------------------------------------- stubs
namespace renderfx {
bool ngxInit(RfxContext*, bool* sr, bool* rr) { if (sr) *sr = false; if (rr) *rr = false; return false; }
void ngxShutdown(RfxContext*) {}
RfxResult ngxRecordDLAA(RfxContext*, VkCommandBuffer, const RfxFrameContext*, const RfxImageDesc*, uint32_t) { return RFX_UNSUPPORTED; }
RfxResult ngxRecordRR(RfxContext*, VkCommandBuffer, const RfxFrameContext*, const RfxImageDesc*, uint32_t) { return RFX_UNSUPPORTED; }
}  // namespace renderfx
#else
// ---------------------------------------------------------------------------- real
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssd_vk.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace renderfx {
namespace {

// Per-context NGX state. Handles are created lazily on first record (CreateFeature needs
// a command buffer) and recreated if the frame dimensions change.
struct NgxState {
    NVSDK_NGX_Parameter* caps = nullptr;      // capability params (feature availability)
    NVSDK_NGX_Parameter* dlaaParams = nullptr;// per-feature param blocks
    NVSDK_NGX_Parameter* rrParams = nullptr;
    NVSDK_NGX_Handle* dlaa = nullptr;
    NVSDK_NGX_Handle* rr = nullptr;
    uint32_t dlaaW = 0, dlaaH = 0;
    uint32_t rrW = 0, rrH = 0, rrTW = 0, rrTH = 0;
    bool initialised = false;
};

inline NgxState* state(RfxContext* ctx) { return static_cast<NgxState*>(ctx->ngx); }
inline bool ok(NVSDK_NGX_Result r) { return NVSDK_NGX_SUCCEED(r); }

// wchar_t is 4 bytes on Linux; NGX wants a wchar_t* application-data path.
std::vector<uint32_t> wpath(const char* s) {
    std::vector<uint32_t> w;
    for (; *s; ++s) w.push_back((uint32_t)(unsigned char)*s);
    w.push_back(0);
    return w;
}

VkImageAspectFlags aspectOf(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D32_SFLOAT:            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:    return VK_IMAGE_ASPECT_DEPTH_BIT;
        default:                               return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// Wrap an RfxImageDesc as an NGX Vulkan image-view resource (GENERAL layout expected).
NVSDK_NGX_Resource_VK wrap(const RfxImageDesc& d, bool readWrite) {
    VkImageSubresourceRange range{};
    range.aspectMask = aspectOf(d.format);
    range.baseMipLevel = 0; range.levelCount = 1;
    range.baseArrayLayer = 0; range.layerCount = 1;
    return NVSDK_NGX_Create_ImageView_Resource_VK(d.view, d.image, range, d.format,
                                                  d.width, d.height, readWrite);
}

}  // namespace

bool ngxInit(RfxContext* ctx, bool* srAvail, bool* rrAvail) {
    if (srAvail) *srAvail = false;
    if (rrAvail) *rrAvail = false;
    if (!ctx || !ctx->info.device || !ctx->info.instance) return false;

    auto gdpa = (PFN_vkGetDeviceProcAddr)ctx->info.gipa(ctx->info.instance, "vkGetDeviceProcAddr");

    // design.md §19: NGX defaults its cubin/model cache to the root-owned
    // /usr/share/nvidia/ngx and Init fails 0xBAD00005 unless we pass a WRITABLE path in
    // BOTH the legacy arg AND FeatureCommonInfo.PathListInfo. The vendored `rel` dir is
    // user-writable and holds the DLSS/RR snippets. The project id must be a valid UUID.
    std::vector<uint32_t> wp = wpath(RENDERFX_NGX_DATA_PATH);
    const wchar_t* pathW = reinterpret_cast<const wchar_t*>(wp.data());
    const wchar_t* paths[1] = { pathW };
    NVSDK_NGX_FeatureCommonInfo fci; std::memset(&fci, 0, sizeof(fci));
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        "b1e7fa2c-9d34-4c1a-8b77-6f0a1e2d3c4b", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
        pathW, ctx->info.instance, ctx->info.physical_device, ctx->info.device,
        ctx->info.gipa, gdpa, &fci, NVSDK_NGX_Version_API);
    if (!ok(r)) return false;

    auto* st = new NgxState();
    st->initialised = true;
    ctx->ngx = st;

    if (ok(NVSDK_NGX_VULKAN_GetCapabilityParameters(&st->caps)) && st->caps) {
        int sr = 0, rr2 = 0;
        NVSDK_NGX_Parameter_GetI(st->caps, NVSDK_NGX_Parameter_SuperSampling_Available, &sr);
        NVSDK_NGX_Parameter_GetI(st->caps, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &rr2);
        if (srAvail) *srAvail = sr != 0;
        if (rrAvail) *rrAvail = rr2 != 0;
    }
    return true;
}

void ngxShutdown(RfxContext* ctx) {
    if (!ctx || !ctx->ngx) return;
    auto* st = state(ctx);
    if (ctx->info.device) vkDeviceWaitIdle(ctx->info.device);
    if (st->dlaa) NVSDK_NGX_VULKAN_ReleaseFeature(st->dlaa);
    if (st->rr)   NVSDK_NGX_VULKAN_ReleaseFeature(st->rr);
    if (st->dlaaParams) NVSDK_NGX_VULKAN_DestroyParameters(st->dlaaParams);
    if (st->rrParams)   NVSDK_NGX_VULKAN_DestroyParameters(st->rrParams);
    if (st->initialised && ctx->info.device) NVSDK_NGX_VULKAN_Shutdown1(ctx->info.device);
    delete st;
    ctx->ngx = nullptr;
}

RfxResult ngxRecordDLAA(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                        const RfxImageDesc* dst, uint32_t reset) {
    if (!ctx || !ctx->ngx) return RFX_UNSUPPORTED;
    auto* st = state(ctx);
    // DLSS/DLAA require depth + motion; without them there is nothing correct to do.
    if (!(fc->provided_inputs & RFX_INPUT_DEPTH) || !(fc->provided_inputs & RFX_INPUT_MOTION))
        return RFX_INVALID_ARGUMENT;

    // Render == color dims; target == dst dims. Equal => DLAA (native AA); dst larger =>
    // DLSS Super Resolution. One code path, dimensions decide (design §16 symmetry).
    const uint32_t W = fc->color.width, H = fc->color.height;
    const uint32_t TW = dst->width, TH = dst->height;
    if (st->dlaa && (st->dlaaW != W || st->dlaaH != H)) {
        vkDeviceWaitIdle(ctx->info.device);
        NVSDK_NGX_VULKAN_ReleaseFeature(st->dlaa); st->dlaa = nullptr;
    }
    if (!st->dlaa) {
        if (!st->dlaaParams &&
            !ok(NVSDK_NGX_VULKAN_AllocateParameters(&st->dlaaParams))) return RFX_INTERNAL;
        NVSDK_NGX_DLSS_Create_Params cp; std::memset(&cp, 0, sizeof(cp));
        cp.Feature.InWidth = W;  cp.Feature.InHeight = H;
        cp.Feature.InTargetWidth = TW; cp.Feature.InTargetHeight = TH;
        cp.Feature.InPerfQualityValue = (W == TW && H == TH)
            ? NVSDK_NGX_PerfQuality_Value_DLAA : NVSDK_NGX_PerfQuality_Value_MaxQuality;
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                  NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        if (!ok(NGX_VULKAN_CREATE_DLSS_EXT(cmd, 1, 1, &st->dlaa, st->dlaaParams, &cp)))
            return RFX_INTERNAL;
        st->dlaaW = W; st->dlaaH = H;
    }

    NVSDK_NGX_Resource_VK color = wrap(fc->color, false);
    NVSDK_NGX_Resource_VK depth = wrap(fc->depth, false);
    NVSDK_NGX_Resource_VK motion = wrap(fc->motion, false);
    NVSDK_NGX_Resource_VK output = wrap(*dst, true);

    NVSDK_NGX_VK_DLSS_Eval_Params ep; std::memset(&ep, 0, sizeof(ep));
    ep.Feature.pInColor = &color;
    ep.Feature.pInOutput = &output;
    ep.pInDepth = &depth;
    ep.pInMotionVectors = &motion;
    ep.InJitterOffsetX = fc->jitter[0];
    ep.InJitterOffsetY = fc->jitter[1];
    ep.InRenderSubrectDimensions.Width = W;
    ep.InRenderSubrectDimensions.Height = H;
    ep.InReset = reset ? 1 : 0;
    ep.InMVScaleX = fc->mv_scale[0] != 0.0f ? fc->mv_scale[0] : 1.0f;
    ep.InMVScaleY = fc->mv_scale[1] != 0.0f ? fc->mv_scale[1] : 1.0f;
    if (fc->provided_inputs & RFX_INPUT_EXPOSURE) ep.InPreExposure = fc->exposure;

    return ok(NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, st->dlaa, st->dlaaParams, &ep))
               ? RFX_OK : RFX_INTERNAL;
}

RfxResult ngxRecordRR(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                      const RfxImageDesc* output, uint32_t reset) {
    if (!ctx || !ctx->ngx) return RFX_UNSUPPORTED;
    auto* st = state(ctx);
    // RR consumes the GBuffer; depth+motion+normals+roughness+diffuse+specular albedo.
    const uint32_t need = RFX_INPUT_DEPTH | RFX_INPUT_MOTION | RFX_INPUT_NORMALS |
                          RFX_INPUT_ROUGHNESS | RFX_INPUT_ALBEDO_DIFFUSE | RFX_INPUT_ALBEDO_SPECULAR;
    if ((fc->provided_inputs & need) != need) return RFX_INVALID_ARGUMENT;

    const uint32_t W = fc->color.width, H = fc->color.height;
    const uint32_t TW = output->width, TH = output->height;
    if (st->rr && (st->rrW != W || st->rrH != H || st->rrTW != TW || st->rrTH != TH)) {
        vkDeviceWaitIdle(ctx->info.device);
        NVSDK_NGX_VULKAN_ReleaseFeature(st->rr); st->rr = nullptr;
    }
    if (!st->rr) {
        if (!st->rrParams &&
            !ok(NVSDK_NGX_VULKAN_AllocateParameters(&st->rrParams))) return RFX_INTERNAL;
        NVSDK_NGX_DLSSD_Create_Params cp; std::memset(&cp, 0, sizeof(cp));
        cp.InWidth = W; cp.InHeight = H;
        cp.InTargetWidth = TW; cp.InTargetHeight = TH;
        cp.InPerfQualityValue = (W == TW && H == TH) ? NVSDK_NGX_PerfQuality_Value_DLAA
                                                     : NVSDK_NGX_PerfQuality_Value_MaxQuality;
        cp.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
        cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;   // dedicated roughness
        cp.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
        cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                  NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
        if (!ok(NGX_VULKAN_CREATE_DLSSD_EXT1(ctx->info.device, cmd, 1, 1, &st->rr,
                                             st->rrParams, &cp)))
            return RFX_INTERNAL;
        st->rrW = W; st->rrH = H; st->rrTW = TW; st->rrTH = TH;
    }

    NVSDK_NGX_Resource_VK color = wrap(fc->color, false);
    NVSDK_NGX_Resource_VK depth = wrap(fc->depth, false);
    NVSDK_NGX_Resource_VK motion = wrap(fc->motion, false);
    NVSDK_NGX_Resource_VK normals = wrap(fc->normals, false);
    NVSDK_NGX_Resource_VK roughness = wrap(fc->roughness, false);
    NVSDK_NGX_Resource_VK diffuse = wrap(fc->albedo_diffuse, false);
    NVSDK_NGX_Resource_VK specular = wrap(fc->albedo_specular, false);
    NVSDK_NGX_Resource_VK out = wrap(*output, true);
    // NGX consumes non-const matrix pointers; copy into locals.
    float w2v[16]; std::memcpy(w2v, fc->world_to_view, sizeof(w2v));
    float v2c[16]; std::memcpy(v2c, fc->view_to_clip, sizeof(v2c));

    NVSDK_NGX_VK_DLSSD_Eval_Params ep; std::memset(&ep, 0, sizeof(ep));
    ep.pInColor = &color;
    ep.pInOutput = &out;
    ep.pInDepth = &depth;
    ep.pInMotionVectors = &motion;
    ep.pInNormals = &normals;
    ep.pInRoughness = &roughness;
    ep.pInDiffuseAlbedo = &diffuse;
    ep.pInSpecularAlbedo = &specular;
    ep.pInWorldToViewMatrix = w2v;
    ep.pInViewToClipMatrix = v2c;
    ep.InJitterOffsetX = fc->jitter[0];
    ep.InJitterOffsetY = fc->jitter[1];
    ep.InRenderSubrectDimensions.Width = W;
    ep.InRenderSubrectDimensions.Height = H;
    ep.InReset = reset ? 1 : 0;
    ep.InMVScaleX = fc->mv_scale[0] != 0.0f ? fc->mv_scale[0] : 1.0f;
    ep.InMVScaleY = fc->mv_scale[1] != 0.0f ? fc->mv_scale[1] : 1.0f;
    if (fc->provided_inputs & RFX_INPUT_EXPOSURE) ep.InPreExposure = fc->exposure;

    return ok(NGX_VULKAN_EVALUATE_DLSSD_EXT(cmd, st->rr, st->rrParams, &ep))
               ? RFX_OK : RFX_INTERNAL;
}

}  // namespace renderfx
#endif  // RENDERFX_NGX
