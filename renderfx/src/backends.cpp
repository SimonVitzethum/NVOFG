// RenderFX backend registry + string/serialization/inspection helpers. The static
// table declares each backend's stage, family, determinism, inputs, and cross-stage
// constraints (design.md §18.2); `supported` is filled per device+build by probing.
// Only nvofg / shader-FG / native are functional today (design.md §16).
#include "renderfx.h"
#include "renderfx_internal.hpp"

#include <cstdio>
#include <cstring>

namespace {

// Fields: id, stage, family, supported(0->set later), proprietary, deterministic,
//         quality_tier, cost_hint, required_inputs, requires_family_stage,
//         requires_family, name, note.
const RfxBackendCaps kTable[] = {
    // ---- Upscaling ----
    {RFX_BACKEND_NATIVE, RFX_STAGE_UPSCALING, RFX_FAMILY_GENERIC, 0, 0, 1, 20, 5,
     RFX_INPUT_COLOR, RFX_STAGE_COUNT, RFX_FAMILY_GENERIC,
     "Native (bilinear)", "built-in vendor-neutral upscale"},
    {RFX_BACKEND_TEMPORAL, RFX_STAGE_UPSCALING, RFX_FAMILY_GENERIC, 0, 0, 1, 50, 15,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER | RFX_INPUT_REPROJ,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "Temporal", "built-in temporal upscaler (reserved)"},
    {RFX_BACKEND_DLSS_SR, RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, 0, 1, 0, 90, 30,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "DLSS SR", "NGX Super Resolution (reserved)"},
    {RFX_BACKEND_DLAA, RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, 0, 1, 0, 92, 32,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "DLAA", "NGX DLAA (reserved)"},
    {RFX_BACKEND_FSR, RFX_STAGE_UPSCALING, RFX_FAMILY_FSR, 0, 0, 1, 70, 20,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "FSR", "AMD FidelityFX SR (reserved)"},
    {RFX_BACKEND_XESS, RFX_STAGE_UPSCALING, RFX_FAMILY_XESS, 0, 0, 1, 75, 22,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "XeSS", "Intel XeSS (reserved)"},

    // ---- Ray Reconstruction (requires a DLSS upscaler in NVIDIA's stack) ----
    {RFX_BACKEND_DLSS_RR, RFX_STAGE_RAY_RECONSTRUCTION, RFX_FAMILY_DLSS, 0, 1, 0, 90, 45,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION | RFX_INPUT_ROUGHNESS |
         RFX_INPUT_ALBEDO_DIFFUSE | RFX_INPUT_ALBEDO_SPECULAR | RFX_INPUT_NORMALS,
     RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, "DLSS RR", "NGX Ray Reconstruction (reserved; needs DLSS SR)"},

    // ---- Frame Generation ----
    {RFX_BACKEND_NVOFG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_NVOFG, 0, 0, 1, 60, 30,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION | RFX_INPUT_REPROJ,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "nvofg (OFA)", "native OFA frame generation"},
    {RFX_BACKEND_SHADER_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_GENERIC, 0, 0, 1, 40, 35,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "nvofg (shader)", "portable shader frame generation (Tier B)"},
    {RFX_BACKEND_DLSS_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_DLSS, 0, 1, 0, 85, 35,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION,
     RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, "DLSS FG", "NGX DLSS-G (Windows-gated; reserved)"},
    {RFX_BACKEND_FSR_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_FSR, 0, 0, 1, 70, 30,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION,
     RFX_STAGE_UPSCALING, RFX_FAMILY_FSR, "FSR FG", "FSR frame generation (reserved; pairs with FSR)"},
};
constexpr uint32_t kTableCount = sizeof(kTable) / sizeof(kTable[0]);

}  // namespace

namespace renderfx {

void buildCapabilities(bool nvofgOfa, bool nvofgShader, RfxCapabilities* out) {
    std::memset(out, 0, sizeof(*out));
    out->count = kTableCount;
    for (uint32_t i = 0; i < kTableCount; ++i) {
        out->backends[i] = kTable[i];
        RfxBackendCaps& b = out->backends[i];
        switch (b.id) {
            case RFX_BACKEND_NATIVE:    b.supported = 1; break;
            case RFX_BACKEND_NVOFG:     b.supported = nvofgOfa ? 1 : 0;
                                        if (!nvofgOfa) b.note = "no OFA on this device"; break;
            case RFX_BACKEND_SHADER_FG: b.supported = nvofgShader ? 1 : 0; break;
            default:                    b.supported = 0;   // reserved (needs SDK)
        }
    }
}

}  // namespace renderfx

extern "C" {

const char* rfx_backend_name(RfxBackendId id) {
    switch (id) {
        case RFX_BACKEND_NONE: return "none";
        case RFX_BACKEND_NATIVE: return "native";
        case RFX_BACKEND_TEMPORAL: return "temporal";
        case RFX_BACKEND_DLSS_SR: return "dlss_sr";
        case RFX_BACKEND_DLAA: return "dlaa";
        case RFX_BACKEND_DLSS_RR: return "dlss_rr";
        case RFX_BACKEND_DLSS_FG: return "dlss_fg";
        case RFX_BACKEND_FSR: return "fsr";
        case RFX_BACKEND_FSR_FG: return "fsr_fg";
        case RFX_BACKEND_XESS: return "xess";
        case RFX_BACKEND_NVOFG: return "nvofg";
        case RFX_BACKEND_SHADER_FG: return "nvofg_shader";
        default: return "unknown";
    }
}

RfxBackendId rfx_backend_from_name(const char* name) {
    if (!name) return RFX_BACKEND_NONE;
    for (int i = 0; i < RFX_BACKEND_COUNT; ++i)
        if (std::strcmp(name, rfx_backend_name((RfxBackendId)i)) == 0) return (RfxBackendId)i;
    return RFX_BACKEND_NONE;
}

const char* rfx_stage_name(RfxStage s) {
    switch (s) {
        case RFX_STAGE_UPSCALING: return "Upscaling";
        case RFX_STAGE_RAY_RECONSTRUCTION: return "RayReconstruction";
        case RFX_STAGE_FRAME_GENERATION: return "FrameGeneration";
        default: return "?";
    }
}

const char* rfx_reason_text(RfxReason r) {
    switch (r) {
        case RFX_REASON_SUPPORTED: return "supported";
        case RFX_REASON_HIGHEST_QUALITY: return "highest quality among candidates";
        case RFX_REASON_LOWEST_COST: return "lowest cost (perf/battery)";
        case RFX_REASON_PROPRIETARY_ALLOWED: return "proprietary allowed";
        case RFX_REASON_OPEN_SOURCE: return "open / vendor-neutral";
        case RFX_REASON_VENDOR_PINNED: return "matched vendor preference";
        case RFX_REASON_PREFERRED_NATIVE: return "preferred native (debug)";
        case RFX_REASON_DETERMINISTIC: return "deterministic";
        case RFX_REASON_CONSTRAINT_SATISFIED: return "compatible with selected upscaler";
        case RFX_REASON_ALTERNATIVE_UNAVAILABLE: return "a higher option was unavailable";
        case RFX_REASON_OVERRIDDEN: return "pinned by application";
        case RFX_REASON_DISABLED_NO_COMPATIBLE: return "disabled: no compatible backend";
        case RFX_REASON_STAGE_NOT_REQUESTED: return "stage not requested";
        default: return "?";
    }
}

uint32_t rfx_format_pipeline(const RfxSelection* sel, char* buf, uint32_t cap) {
    if (!buf || cap == 0) return 0;
    uint32_t n = 0;
    auto put = [&](const char* fmt, auto... args) {
        if (n < cap) n += (uint32_t)std::snprintf(buf + n, cap - n, fmt, args...);
    };
    if (!sel || !sel->valid) { put("RenderFX: no valid pipeline\n"); return n < cap ? n : cap - 1; }
    put("RenderFX pipeline (Frame Context ->):\n");
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        const RfxStageExplanation& e = sel->explanation[s];
        put("  %-18s: %-14s [", rfx_stage_name((RfxStage)s), rfx_backend_name(sel->backend[s]));
        for (uint32_t i = 0; i < e.reason_count; ++i)
            put("%s%s", i ? ", " : "", rfx_reason_text(e.reasons[i]));
        put("]\n");
    }
    put("  -> Present\n");
    return n < cap ? n : cap - 1;
}

}  // extern "C"
