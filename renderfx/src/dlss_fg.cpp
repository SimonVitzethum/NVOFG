// NGX DLSS-G (Frame Generation) backend for RenderFX, in the style of ngx.cpp.
//
// Contract: renderfx/docs/dlssg-vulkan-integration.md (§2 lifecycle, §2.3 create,
// §2.4 evaluate table). Uses the vendored NGX SDK's inline VK helpers
// (NGX_VK_CREATE_DLSSG / NGX_VK_EVALUATE_DLSSG) so param packing matches NVIDIA's
// reference exactly. Real only with -DRENDERFX_NGX + the vendored SDK; otherwise
// this file compiles to inert stubs so context.cpp stays branch-free (ADR 0006).
//
// Pure NGX path on purpose: the native Path-B loader (experiments/dlssg-native)
// lands later *behind* these same entry points (dlssFgInit/Shutdown/Record).
//
// WHAT IS MISSING UNTIL FUNCTIONAL (Init/Create/Evaluate sequence stands, but):
//  TODO(wiring-1): declare dlssFgInit/dlssFgShutdown/dlssFgRecord in
//      renderfx/src/renderfx_internal.hpp next to the ngx decls (:19-26) and add a
//      `void* dlssFg` state slot next to `void* ngx` (:60). NOT done here — this
//      file must not modify existing files, so it keeps its state in a file-local
//      registry keyed by RfxContext* instead (see below); delete the registry once
//      the slot exists.
//  TODO(wiring-2): probe in renderfx/src/context.cpp:22 (rfx_create, next to the
//      ngxInit call) and gate the RFX_BACKEND_DLSS_FG capability on the result;
//      shut down in renderfx/src/context.cpp:34 (rfx_destroy, next to ngxShutdown).
//      Coordinate the single NGX init with ngx.cpp:80 — NVSDK_NGX_VULKAN_Init may
//      only run once per process/device; dlssFgInit must become a no-op when
//      ctx->ngx is already initialised (or vice versa).
//  TODO(wiring-3): dispatch RFX_BACKEND_DLSS_FG in
//      renderfx/src/context.cpp:88 (rfx_record_frame_generation). Open point: that
//      API takes no VkCommandBuffer today, but NGX FG evaluate records into one —
//      the public FG record path needs cmd-buffer plumbing first.
//  TODO(wiring-4): renderfx/src/upscale.cpp:282 needs NO change for FG (FG is the
//      FRAME_GENERATION stage, not upscaling). The DLSS-upscaler pairing is a pure
//      resolver constraint already declared in backends.cpp:63-65
//      (requires_family_stage=UPSCALING, requires_family=DLSS).
//  TODO(loader): native Linux Path-B loader behind these entry points; until then
//      FrameGeneration.Available is 0 on native Linux (design.md) and Create fails.
#include "renderfx_internal.hpp"

#ifndef RENDERFX_NGX
// ---------------------------------------------------------------------------- stubs
namespace renderfx {
bool dlssFgInit(RfxContext*, bool* fgAvail) { if (fgAvail) *fgAvail = false; return false; }
void dlssFgShutdown(RfxContext*) {}
RfxResult dlssFgRecord(RfxContext*, VkCommandBuffer, const RfxFrameContext*,
                       const RfxImageDesc*, uint32_t) { return RFX_UNSUPPORTED; }
}  // namespace renderfx
#else
// ---------------------------------------------------------------------------- real
#include "nvsdk_ngx.h"
#include "nvsdk_ngx_helpers.h"
#include "nvsdk_ngx_helpers_vk.h"
#include "nvsdk_ngx_helpers_dlssg_vk.h"
#include "nvsdk_ngx_params_dlssg.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace renderfx {
namespace {

// Per-context FG state. Handles are created lazily on first record (CreateFeature
// needs a command buffer) and recreated if the frame dimensions change.
// W/H = backbuffer (output) dims, TW/TH = render (input) dims.
struct DlssFgState {
    NVSDK_NGX_Parameter* caps = nullptr;      // capability params (feature availability)
    NVSDK_NGX_Parameter* fgParams = nullptr;  // per-feature param block
    NVSDK_NGX_Handle* handle = nullptr;
    uint32_t W = 0, H = 0, TW = 0, TH = 0;
    bool initialised = false;
};

// TODO(wiring-1): replace with direct storage once RfxContext grows a `void* dlssFg`
// slot (renderfx_internal.hpp:60). Until then the registry keeps this file free of
// changes to existing files; entries are removed in dlssFgShutdown.
std::mutex& fgRegistryMutex() {
    static std::mutex m;
    return m;
}
std::unordered_map<RfxContext*, DlssFgState*>& fgRegistry() {
    static std::unordered_map<RfxContext*, DlssFgState*> r;
    return r;
}
DlssFgState* fgState(RfxContext* ctx) {
    std::lock_guard<std::mutex> lock(fgRegistryMutex());
    auto it = fgRegistry().find(ctx);
    return it != fgRegistry().end() ? it->second : nullptr;
}
inline bool fgOk(NVSDK_NGX_Result r) { return NVSDK_NGX_SUCCEED(r); }

// wchar_t is 4 bytes on Linux; NGX wants a wchar_t* application-data path.
// Same rule as ngx.cpp:wpath.
std::vector<uint32_t> fgWpath(const char* s) {
    std::vector<uint32_t> w;
    for (; *s; ++s) w.push_back((uint32_t)(unsigned char)*s);
    w.push_back(0);
    return w;
}

VkImageAspectFlags fgAspectOf(VkFormat f) {
    switch (f) {
        case VK_FORMAT_D16_UNORM:
        case VK_FORMAT_D32_SFLOAT:            return VK_IMAGE_ASPECT_DEPTH_BIT;
        case VK_FORMAT_D24_UNORM_S8_UINT:
        case VK_FORMAT_D32_SFLOAT_S8_UINT:    return VK_IMAGE_ASPECT_DEPTH_BIT;
        default:                               return VK_IMAGE_ASPECT_COLOR_BIT;
    }
}

// Wrap an RfxImageDesc as an NGX Vulkan image-view resource (GENERAL layout expected).
NVSDK_NGX_Resource_VK fgWrap(const RfxImageDesc& d, bool readWrite) {
    VkImageSubresourceRange range{};
    range.aspectMask = fgAspectOf(d.format);
    range.baseMipLevel = 0; range.levelCount = 1;
    range.baseArrayLayer = 0; range.layerCount = 1;
    return NVSDK_NGX_Create_ImageView_Resource_VK(d.view, d.image, range, d.format,
                                                  d.width, d.height, readWrite);
}

// TODO(loader): VkFormat -> NGX backbuffer-format mapping. NativeBackbufferFormat is
// a DXGI-style enum, NOT a VkFormat; passing it through raw would be wrong. Fill this
// in with the Path-B loader work (experiments/dlssg-native); until then Create::_Back-
// bufferFormat carries 0 and CreateFeature is expected to fail — pure NGX path only.
unsigned int fgBackbufferFormat(const RfxImageDesc& /*dst*/) { return 0; }

}  // namespace

bool dlssFgInit(RfxContext* ctx, bool* fgAvail) {
    if (fgAvail) *fgAvail = false;
    if (!ctx || !ctx->info.device || !ctx->info.instance) return false;
    if (fgState(ctx)) return fgState(ctx)->initialised;

    auto gdpa = (PFN_vkGetDeviceProcAddr)ctx->info.gipa(ctx->info.instance, "vkGetDeviceProcAddr");

    // Path rule copied from ngx.cpp:ngxInit (design.md §19): NGX defaults its
    // cubin/model cache to the root-owned /usr/share/nvidia/ngx and Init fails
    // 0xBAD00005 unless we pass a WRITABLE path in BOTH the legacy arg AND
    // FeatureCommonInfo.PathListInfo. The vendored `rel` dir is user-writable and
    // holds the DLSS-G snippets. The project id must be a valid UUID.
    // TODO(wiring-2): single-Init coordination with ngx.cpp:80 — see file header.
    std::vector<uint32_t> wp = fgWpath(RENDERFX_NGX_DATA_PATH);
    const wchar_t* pathW = reinterpret_cast<const wchar_t*>(wp.data());
    const wchar_t* paths[1] = { pathW };
    NVSDK_NGX_FeatureCommonInfo fci; std::memset(&fci, 0, sizeof(fci));
    fci.PathListInfo.Path = paths;
    fci.PathListInfo.Length = 1;

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        "b1e7fa2c-9d34-4c1a-8b77-6f0a1e2d3c4b", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
        pathW, ctx->info.instance, ctx->info.physical_device, ctx->info.device,
        ctx->info.gipa, gdpa, &fci, NVSDK_NGX_Version_API);
    if (!fgOk(r)) return false;

    auto* st = new DlssFgState();
    st->initialised = true;
    {
        std::lock_guard<std::mutex> lock(fgRegistryMutex());
        fgRegistry()[ctx] = st;
    }

    // Capability contract (integration doc §2.2): FrameGeneration.Available /
    // FeatureInitResult / NeedsUpdatedDriver. Native Linux reports Available = 0,
    // FeatureInitResult = 0xBAD00004 (FAIL_FeatureNotSupported) until Path-B lands.
    if (fgOk(NVSDK_NGX_VULKAN_GetCapabilityParameters(&st->caps)) && st->caps) {
        int fg = 0;
        NVSDK_NGX_Parameter_GetI(st->caps, NVSDK_NGX_Parameter_FrameGeneration_Available, &fg);
        if (fgAvail) *fgAvail = fg != 0;
    }
    return true;
}

void dlssFgShutdown(RfxContext* ctx) {
    if (!ctx) return;
    DlssFgState* st = fgState(ctx);
    if (!st) return;
    {
        std::lock_guard<std::mutex> lock(fgRegistryMutex());
        fgRegistry().erase(ctx);
    }
    if (ctx->info.device) vkDeviceWaitIdle(ctx->info.device);
    if (st->handle) NVSDK_NGX_VULKAN_ReleaseFeature(st->handle);
    if (st->fgParams) NVSDK_NGX_VULKAN_DestroyParameters(st->fgParams);
    if (st->initialised && ctx->info.device) NVSDK_NGX_VULKAN_Shutdown1(ctx->info.device);
    delete st;
}

RfxResult dlssFgRecord(RfxContext* ctx, VkCommandBuffer cmd, const RfxFrameContext* fc,
                       const RfxImageDesc* output, uint32_t reset) {
    if (!ctx || !fc || !output) return RFX_INVALID_ARGUMENT;
    DlssFgState* st = fgState(ctx);
    if (!st) return RFX_UNSUPPORTED;
    // FG interpolates from rendered color + depth + dense motion (incl. camera);
    // without them there is nothing correct to do (integration doc §2.4 table).
    if (!(fc->provided_inputs & RFX_INPUT_DEPTH) || !(fc->provided_inputs & RFX_INPUT_MOTION))
        return RFX_INVALID_ARGUMENT;
    if (!cmd) return RFX_INVALID_ARGUMENT;

    // Backbuffer (output) dims vs. render (input) dims; recreate on change
    // (pattern ngx.cpp:143-160).
    const uint32_t W = output->width, H = output->height;
    const uint32_t TW = fc->color.width, TH = fc->color.height;
    if (st->handle && (st->W != W || st->H != H || st->TW != TW || st->TH != TH)) {
        vkDeviceWaitIdle(ctx->info.device);
        NVSDK_NGX_VULKAN_ReleaseFeature(st->handle); st->handle = nullptr;
    }
    if (!st->handle) {
        if (!st->fgParams &&
            !fgOk(NVSDK_NGX_VULKAN_AllocateParameters(&st->fgParams))) return RFX_INTERNAL;
        NVSDK_NGX_DLSSG_Create_Params cp; std::memset(&cp, 0, sizeof(cp));
        cp.Width = W; cp.Height = H;
        cp.NativeBackbufferFormat = fgBackbufferFormat(*output);  // TODO(loader): 0 for now
        cp.RenderWidth = TW; cp.RenderHeight = TH;
        cp.DynamicResolutionScaling = false;
        // Create-Time (integration doc §2.3): the VK helper packs NodeMasks +
        // Width/Height + BackbufferFormat; DLSSG.Width/Height are set explicitly.
        // TODO(loader): DLSSG.InternalWidth/Height (= RenderWidth/Height, cf. the
        // D3D12 helper) + ResourceAlways/NeverProvidedFlags + UIR create flag.
        if (!fgOk(NGX_VK_CREATE_DLSSG(cmd, 1, 1, &st->handle, st->fgParams, &cp)))
            return RFX_INTERNAL;
        NVSDK_NGX_Parameter_SetUI(st->fgParams, NVSDK_NGX_DLSSG_Parameter_Width, W);
        NVSDK_NGX_Parameter_SetUI(st->fgParams, NVSDK_NGX_DLSSG_Parameter_Height, H);
        st->W = W; st->H = H; st->TW = TW; st->TH = TH;
    }

    NVSDK_NGX_Resource_VK backbuffer = fgWrap(fc->color, false);
    NVSDK_NGX_Resource_VK mvecs = fgWrap(fc->motion, false);
    NVSDK_NGX_Resource_VK depth = fgWrap(fc->depth, false);
    NVSDK_NGX_Resource_VK out = fgWrap(*output, true);

    NVSDK_NGX_VK_DLSSG_Eval_Params ep; std::memset(&ep, 0, sizeof(ep));
    ep.pBackbuffer = &backbuffer;
    ep.pMVecs = &mvecs;
    ep.pDepth = &depth;
    // TODO(ui): HUDLess/UI/UIAlpha need app-provided pre-UI + UI buffers; the Frame
    // Context has no UI inputs yet (reactive/material_id are NOT UI). Passing nullptr
    // today; enabling UIR also requires DLSSG.UserInterfaceRecompositionEnabled at
    // create time (integration doc §2.3/§2.4).
    ep.pHudless = nullptr;
    ep.pUI = nullptr;
    ep.pUIAlpha = nullptr;
    ep.pBidirectionalDistortionField = nullptr;
    ep.pOutputInterpFrame = &out;
    ep.pOutputRealFrame = nullptr;
    ep.pOutputDisableInterpolation = nullptr;

    NVSDK_NGX_DLSSG_Opt_Eval_Params opt; std::memset(&opt, 0, sizeof(opt));
    // TODO(mfg): MultiFrameCount>1 / MultiFrameIndex>1 need capability
    // DLSSG.MultiFrameCountMax + one Evaluate per index; default is 1x (2x total).
    opt.multiFrameCount = 1;
    opt.multiFrameIndex = 1;
    // Frame Context -> NGX mapping (integration doc §2.4). RfxFrameContext carries
    // no camera pos/up/right/fwd vectors, no FOV, and no clip-space helper matrices,
    // so only jitter / mv-scale / reset / near-far / aspect are wired; the rest stays
    // zero until the Frame Context grows FG camera inputs.
    std::memcpy(opt.cameraViewToClip, fc->view_to_clip, sizeof(opt.cameraViewToClip));
    // TODO(fg-cam): clipToCameraView (inverse), clipToLensClip (identity unless lens
    // distortion), clipToPrevClip/prevClipToClip (from fc->reproj), camera pos/up/
    // right/fwd, FOV, cameraMotionIncluded, orthoProjection, depthInverted.
    opt.jitterOffset[0] = fc->jitter[0];
    opt.jitterOffset[1] = fc->jitter[1];
    opt.mvecScale[0] = fc->mv_scale[0] != 0.0f ? fc->mv_scale[0] : 1.0f;
    opt.mvecScale[1] = fc->mv_scale[1] != 0.0f ? fc->mv_scale[1] : 1.0f;
    opt.cameraNear = fc->near_plane;
    opt.cameraFar = fc->far_plane;
    opt.cameraAspectRatio = H != 0 ? (float)W / (float)H : 1.0f;
    opt.reset = reset ? true : false;

    return fgOk(NGX_VK_EVALUATE_DLSSG(cmd, st->handle, st->fgParams, &ep, &opt))
               ? RFX_OK : RFX_INTERNAL;
}

}  // namespace renderfx
#endif  // RENDERFX_NGX
