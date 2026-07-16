// RenderFX context — aggregates backend capabilities and drives execution by
// *composing* focused libraries. Today the only functional backend is nvofg (frame
// generation); RenderFX never reimplements it, it calls nvofg's C ABI.
#include "renderfx_internal.hpp"

#include <cstring>
#include <new>

static NvofgImageDesc toNvofg(const RfxImageDesc& d) {
    return NvofgImageDesc{d.image, d.view, d.format, d.width, d.height};
}

extern "C" {

RfxResult rfx_create(const RfxCreateInfo* info, RfxContext** out) {
    if (!info || !out || !info->device || !info->instance || !info->gipa) return RFX_INVALID_ARGUMENT;
    auto* ctx = new (std::nothrow) RfxContext();
    if (!ctx) return RFX_INTERNAL;
    ctx->info = *info;
    *out = ctx;
    return RFX_OK;
}

void rfx_destroy(RfxContext* ctx) {
    if (!ctx) return;
    if (ctx->info.device) vkDeviceWaitIdle(ctx->info.device);
    renderfx::destroyUpscale(ctx);
    if (ctx->nvofg) nvofg_destroy(ctx->nvofg);
    delete ctx;
}

RfxResult rfx_query_capabilities(RfxContext* ctx, RfxCapabilities* outCaps) {
    if (!ctx || !outCaps) return RFX_INVALID_ARGUMENT;
    // Aggregate each backend's own cheap probe (design.md §18.1). nvofg exposes its
    // own capability query; RenderFX does not reimplement it.
    NvofgCaps nc{};
    bool ofa = nvofg_query_support(ctx->info.instance, ctx->info.physical_device,
                                   ctx->info.gipa, &nc) == NVOFG_OK && nc.supported;
    // Portable shader FG works on any Vulkan GPU.
    renderfx::buildCapabilities(ofa, /*shader*/ true, outCaps);
    ctx->caps = *outCaps;
    return RFX_OK;
}

RfxResult rfx_commit(RfxContext* ctx, const RfxSelection* sel) {
    if (!ctx || !sel || !sel->valid) return RFX_INVALID_ARGUMENT;
    RfxBackendId fg = sel->backend[RFX_STAGE_FRAME_GENERATION];
    ctx->fgBackend = fg;

    // Only the frame-generation stage has a functional backend today. Upscaling/RR
    // selections are recorded but their execution backends are reserved (§16).
    if (fg == RFX_BACKEND_NVOFG || fg == RFX_BACKEND_SHADER_FG) {
        if (ctx->nvofg) { nvofg_destroy(ctx->nvofg); ctx->nvofg = nullptr; ctx->registered = false; }
        NvofgCreateInfo ci{};
        ci.instance = ctx->info.instance;
        ci.physical_device = ctx->info.physical_device;
        ci.device = ctx->info.device;
        ci.queue = ctx->info.queue;
        ci.queue_family_index = ctx->info.queue_family_index;
        ci.of_queue = ctx->info.of_queue;
        ci.of_queue_family_index = ctx->info.of_queue_family_index;
        ci.gipa = ctx->info.gipa;
        ci.width = ctx->info.width;
        ci.height = ctx->info.height;
        ci.quality = NVOFG_QUALITY_HIGH;
        ci.interpolator = NVOFG_INTERP_WARP;
        ci.mode = NVOFG_MODE_AUTOMATIC;
        ci.flags = NVOFG_FLAG_USE_DEPTH | NVOFG_FLAG_USE_MOTION | NVOFG_FLAG_BIDIRECTIONAL;
        if (fg == RFX_BACKEND_SHADER_FG) ci.flags |= NVOFG_FLAG_FORCE_SHADER_FLOW;
        NvofgResult r = nvofg_create(&ci, &ctx->nvofg);
        if (r != NVOFG_OK) return (r == NVOFG_UNSUPPORTED) ? RFX_UNSUPPORTED : RFX_INTERNAL;
    }
    return RFX_OK;
}

RfxResult rfx_record_frame_generation(RfxContext* ctx, const RfxFrameContext* fc,
                                      const RfxImageDesc* prev_color,
                                      const RfxImageDesc* output, float phase,
                                      VkImageLayout color_layout,
                                      const RfxFrameSync* input_ready,
                                      RfxFrameSync* out_sync) {
    if (!ctx || !fc || !prev_color || !output || !out_sync) return RFX_INVALID_ARGUMENT;
    if (!ctx->nvofg) return RFX_NOT_REGISTERED;

    if (!ctx->registered) {
        NvofgImageDesc prev = toNvofg(*prev_color);
        NvofgImageDesc curr = toNvofg(fc->color);
        NvofgImageDesc out = toNvofg(*output);
        if (nvofg_register_color(ctx->nvofg, &prev, &curr) != NVOFG_OK) return RFX_INTERNAL;
        NvofgAuxDesc aux{};
        NvofgImageDesc depth, motion;
        if (fc->provided_inputs & RFX_INPUT_DEPTH)  { depth = toNvofg(fc->depth);   aux.depth = &depth; }
        if (fc->provided_inputs & RFX_INPUT_MOTION) { motion = toNvofg(fc->motion); aux.motion = &motion; }
        nvofg_register_aux(ctx->nvofg, &aux);
        if (nvofg_register_output(ctx->nvofg, &out) != NVOFG_OK) return RFX_INTERNAL;
        ctx->registered = true;
    }

    NvofgGenerateInfo gi{};
    gi.phase = phase;
    std::memcpy(gi.reproj, fc->reproj, sizeof(gi.reproj));
    gi.near_plane = fc->near_plane;
    gi.far_plane = fc->far_plane;
    gi.prev_layout = color_layout;
    gi.curr_layout = color_layout;
    if (input_ready) { gi.input_timeline = input_ready->semaphore; gi.input_value = input_ready->value; }

    NvofgFrameSync sync{};
    NvofgResult r = nvofg_record_generate(ctx->nvofg, &gi, &sync);
    if (r != NVOFG_OK) return RFX_INTERNAL;
    out_sync->semaphore = sync.semaphore;
    out_sync->value = sync.value;
    ctx->framesGenerated++;
    return RFX_OK;
}

uint64_t rfx_query_stage_features(RfxContext* ctx, RfxStage stage) {
    if (!ctx) return 0;
    return rfx_stage_features(&ctx->caps, stage);
}

RfxResult rfx_get_statistics(RfxContext* ctx, RfxStatistics* out) {
    if (!ctx || !out) return RFX_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    out->schema_version = RFX_CAPABILITY_SCHEMA_VERSION;
    out->frames_generated = ctx->framesGenerated;
    // GPU-time fields stay 0 (unknown): RenderFX owns no command buffers, so timings
    // must come from backends exposing timestamps (RFX_FEATURE_STATISTICS). nvofg does
    // not yet expose per-stage GPU timing — reserved for a future backend revision.
    return RFX_OK;
}

}  // extern "C"
