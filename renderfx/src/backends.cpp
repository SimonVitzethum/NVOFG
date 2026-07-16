// RenderFX backend registry. The static table declares every backend's stage, family,
// inputs, and cross-stage constraints (design.md §18.2); `supported` is filled per
// device+build by capability probing. Only nvofg / shader-FG / native are functional
// today — the DLSS/FSR/XeSS entries are honest reserved slots (need external SDKs, §16).
#include "renderfx.h"
#include "renderfx_internal.hpp"

#include <cstring>

namespace {

// note: `supported` is set later by buildCapabilities().
const RfxBackendCaps kTable[] = {
    // ---- Upscaling ----
    {RFX_BACKEND_NATIVE, RFX_STAGE_UPSCALING, RFX_FAMILY_GENERIC, 0, 0, 20, 5,
     RFX_INPUT_COLOR, RFX_STAGE_COUNT, RFX_FAMILY_GENERIC,
     "Native (bilinear)", "built-in vendor-neutral upscale"},
    {RFX_BACKEND_TEMPORAL, RFX_STAGE_UPSCALING, RFX_FAMILY_GENERIC, 0, 0, 50, 15,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER | RFX_INPUT_REPROJ,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "Temporal", "built-in temporal upscaler (reserved)"},
    {RFX_BACKEND_DLSS_SR, RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, 0, 1, 90, 30,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "DLSS SR", "NGX Super Resolution (reserved)"},
    {RFX_BACKEND_DLAA, RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, 0, 1, 92, 32,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "DLAA", "NGX DLAA (reserved)"},
    {RFX_BACKEND_FSR, RFX_STAGE_UPSCALING, RFX_FAMILY_FSR, 0, 0, 70, 20,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "FSR", "AMD FidelityFX SR (reserved)"},
    {RFX_BACKEND_XESS, RFX_STAGE_UPSCALING, RFX_FAMILY_XESS, 0, 0, 75, 22,
     RFX_INPUT_COLOR | RFX_INPUT_MOTION | RFX_INPUT_DEPTH | RFX_INPUT_JITTER,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "XeSS", "Intel XeSS (reserved)"},

    // ---- Ray Reconstruction (requires a DLSS upscaler in NVIDIA's stack) ----
    {RFX_BACKEND_DLSS_RR, RFX_STAGE_RAY_RECONSTRUCTION, RFX_FAMILY_DLSS, 0, 1, 90, 45,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION | RFX_INPUT_ROUGHNESS |
         RFX_INPUT_ALBEDO_DIFFUSE | RFX_INPUT_ALBEDO_SPECULAR | RFX_INPUT_NORMALS,
     RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, "DLSS RR", "NGX Ray Reconstruction (reserved; needs DLSS SR)"},

    // ---- Frame Generation ----
    {RFX_BACKEND_NVOFG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_NVOFG, 0, 0, 60, 30,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION | RFX_INPUT_REPROJ,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "nvofg (OFA)", "native OFA frame generation"},
    {RFX_BACKEND_SHADER_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_GENERIC, 0, 0, 40, 35,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION,
     RFX_STAGE_COUNT, RFX_FAMILY_GENERIC, "nvofg (shader)", "portable shader frame generation (Tier B)"},
    {RFX_BACKEND_DLSS_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_DLSS, 0, 1, 85, 35,
     RFX_INPUT_COLOR | RFX_INPUT_DEPTH | RFX_INPUT_MOTION,
     RFX_STAGE_UPSCALING, RFX_FAMILY_DLSS, "DLSS FG", "NGX DLSS-G (Windows-gated; reserved)"},
    {RFX_BACKEND_FSR_FG, RFX_STAGE_FRAME_GENERATION, RFX_FAMILY_FSR, 0, 0, 70, 30,
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
        // Only the functional backends report supported on this build.
        switch (b.id) {
            case RFX_BACKEND_NATIVE:    b.supported = 1; break;               // trivial bilinear
            case RFX_BACKEND_NVOFG:     b.supported = nvofgOfa ? 1 : 0;
                                        if (!nvofgOfa) b.note = "no OFA on this device"; break;
            case RFX_BACKEND_SHADER_FG: b.supported = nvofgShader ? 1 : 0; break;  // any Vulkan GPU
            default:                    b.supported = 0;                      // reserved (needs SDK)
        }
    }
}

}  // namespace renderfx

extern "C" const char* rfx_backend_name(RfxBackendId id) {
    switch (id) {
        case RFX_BACKEND_NONE: return "None";
        case RFX_BACKEND_NATIVE: return "Native";
        case RFX_BACKEND_TEMPORAL: return "Temporal";
        case RFX_BACKEND_DLSS_SR: return "DLSS SR";
        case RFX_BACKEND_DLAA: return "DLAA";
        case RFX_BACKEND_DLSS_RR: return "DLSS RR";
        case RFX_BACKEND_DLSS_FG: return "DLSS FG";
        case RFX_BACKEND_FSR: return "FSR";
        case RFX_BACKEND_FSR_FG: return "FSR FG";
        case RFX_BACKEND_XESS: return "XeSS";
        case RFX_BACKEND_NVOFG: return "nvofg";
        case RFX_BACKEND_SHADER_FG: return "nvofg-shader";
        default: return "?";
    }
}

extern "C" const char* rfx_stage_name(RfxStage s) {
    switch (s) {
        case RFX_STAGE_UPSCALING: return "Upscaling";
        case RFX_STAGE_RAY_RECONSTRUCTION: return "RayReconstruction";
        case RFX_STAGE_FRAME_GENERATION: return "FrameGeneration";
        default: return "?";
    }
}
