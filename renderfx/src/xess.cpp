// Intel XeSS backend for RenderFX: temporal super-resolution upscaler, dispatched from
// the Upscaling stage (RFX_BACKEND_XESS). Real only with -DRENDERFX_XESS + a linkable
// XeSS runtime; otherwise inert stubs (like ngx.cpp) so context.cpp stays branch-free.
//
// ENVIRONMENT REALITY (ADR 0007): Intel's public XeSS SDK ships a *Windows-only* runtime
// (libxess.dll) — no native-Linux libxess.so exists in any XeSS release. This project is
// native-Linux with no Wine/Proton, so the Windows DLL is out of scope. The integration
// below is written against Intel's official xess_vk.h headers (vendored) and is API-
// complete + compile-verified; it becomes functional wherever a native XeSS runtime is
// present (an Intel Arc Linux host, or if Intel ships a Linux .so). Until then XeSS reports
// unsupported and the resolver falls back — the graceful-degradation contract (G6).
#include "renderfx_internal.hpp"

#ifndef RENDERFX_XESS
// ---------------------------------------------------------------------------- stubs
namespace renderfx {
bool xessInit(RfxContext*, bool* avail) { if (avail) *avail = false; return false; }
void xessShutdown(RfxContext*) {}
RfxResult xessRecordUpscale(RfxContext*, VkCommandBuffer, const RfxFrameContext*, const RfxImageDesc*, uint32_t) { return RFX_UNSUPPORTED; }
}  // namespace renderfx
#else
// ---------------------------------------------------------------------------- real
#include "xess/xess_vk.h"

#include <cstring>

namespace renderfx {
namespace {

struct XessState {
    xess_context_handle_t ctx = nullptr;
    uint32_t initW = 0, initH = 0;   // output (target) resolution the context was init'd for
    bool inited = false;
};

inline XessState* state(RfxContext* c) { return static_cast<XessState*>(c->xess); }
inline bool ok(xess_result_t r) { return r == XESS_RESULT_SUCCESS; }

xess_vk_image_view_info wrap(const RfxImageDesc& d) {
    xess_vk_image_view_info v{};
    v.imageView = d.view; v.image = d.image; v.format = d.format;
    v.width = d.width; v.height = d.height;
    v.subresourceRange = VkImageSubresourceRange{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return v;
}

// Ensure the XeSS context is initialised for a given output resolution (re-init on change).
bool ensureInit(RfxContext* c, uint32_t outW, uint32_t outH) {
    auto* st = state(c);
    if (st->inited && st->initW == outW && st->initH == outH) return true;
    if (st->inited) { vkDeviceWaitIdle(c->info.device); st->inited = false; }
    xess_vk_init_params_t ip{};
    ip.outputResolution = {outW, outH};
    ip.qualitySetting = XESS_QUALITY_SETTING_QUALITY;
    ip.initFlags = XESS_INIT_FLAG_NONE;   // render-res (low-res) motion vectors + depth
    if (!ok(xessVKInit(st->ctx, &ip))) return false;
    st->initW = outW; st->initH = outH; st->inited = true;
    return true;
}

}  // namespace

bool xessInit(RfxContext* c, bool* avail) {
    if (avail) *avail = false;
    if (!c || !c->info.device || !c->info.instance) return false;
    auto* st = new XessState();
    // Context creation probes device support; failure => XeSS unavailable here.
    if (!ok(xessVKCreateContext(c->info.instance, c->info.physical_device, c->info.device, &st->ctx))) {
        delete st; return false;
    }
    c->xess = st;
    if (avail) *avail = true;   // supported; per-resolution init happens lazily on first record
    return true;
}

void xessShutdown(RfxContext* c) {
    if (!c || !c->xess) return;
    auto* st = state(c);
    if (c->info.device) vkDeviceWaitIdle(c->info.device);
    if (st->ctx) xessDestroyContext(st->ctx);
    delete st;
    c->xess = nullptr;
}

RfxResult xessRecordUpscale(RfxContext* c, VkCommandBuffer cmd, const RfxFrameContext* fc,
                            const RfxImageDesc* dst, uint32_t reset) {
    if (!c || !c->xess) return RFX_UNSUPPORTED;
    // XeSS is temporal: needs motion + depth (low-res MV path).
    if (!(fc->provided_inputs & RFX_INPUT_MOTION) || !(fc->provided_inputs & RFX_INPUT_DEPTH))
        return RFX_INVALID_ARGUMENT;
    if (!ensureInit(c, dst->width, dst->height)) return RFX_INTERNAL;

    xess_vk_execute_params_t ep{};
    ep.colorTexture    = wrap(fc->color);
    ep.velocityTexture = wrap(fc->motion);
    ep.depthTexture    = wrap(fc->depth);
    ep.outputTexture   = wrap(*dst);
    ep.jitterOffsetX = fc->jitter[0];
    ep.jitterOffsetY = fc->jitter[1];
    ep.exposureScale = 1.0f;
    ep.resetHistory  = reset ? 1u : 0u;
    ep.inputWidth  = fc->color.width;
    ep.inputHeight = fc->color.height;
    return ok(xessVKExecute(state(c)->ctx, cmd, &ep)) ? RFX_OK : RFX_INTERNAL;
}

}  // namespace renderfx
#endif  // RENDERFX_XESS
