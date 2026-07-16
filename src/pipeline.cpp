// nvofg — resource registration and per-frame recording.
//
// M1 status: the prep/execute/refine/warp pipeline is being wired incrementally.
// These entrypoints currently report NVOFG_INTERNAL until the corresponding
// stage lands; the context lifecycle and capability query (context.cpp) are live.
#include "internal.hpp"

extern "C" {

NvofgResult nvofg_register_color(NvofgContext* ctx, const NvofgImageDesc* prev,
                                 const NvofgImageDesc* curr) {
    if (!ctx || !prev || !curr) return NVOFG_INVALID_ARGUMENT;
    ctx->setError("nvofg_register_color: not yet implemented (M1 in progress)");
    return NVOFG_INTERNAL;
}

NvofgResult nvofg_register_aux(NvofgContext* ctx, const NvofgAuxDesc* aux) {
    if (!ctx || !aux) return NVOFG_INVALID_ARGUMENT;
    ctx->setError("nvofg_register_aux: not yet implemented (M1 in progress)");
    return NVOFG_INTERNAL;
}

NvofgResult nvofg_register_output(NvofgContext* ctx, const NvofgImageDesc* interpolated) {
    if (!ctx || !interpolated) return NVOFG_INVALID_ARGUMENT;
    ctx->setError("nvofg_register_output: not yet implemented (M1 in progress)");
    return NVOFG_INTERNAL;
}

void nvofg_unregister_all(NvofgContext* /*ctx*/) {}

NvofgResult nvofg_record_generate(NvofgContext* ctx, const NvofgGenerateInfo* info,
                                  NvofgFrameSync* out_sync) {
    if (!ctx || !info || !out_sync) return NVOFG_INVALID_ARGUMENT;
    ctx->setError("nvofg_record_generate: not yet implemented (M1 in progress)");
    return NVOFG_NOT_REGISTERED;
}

}  // extern "C"
