#pragma once
#include "renderfx.h"
#include "nvofg.h"

namespace renderfx {
// Fill the capability table with per-build `supported` flags.
void buildCapabilities(bool nvofgOfa, bool nvofgShader, RfxCapabilities* out);
}  // namespace renderfx

struct RfxContext {
    RfxCreateInfo info{};
    RfxCapabilities caps{};

    // The committed frame-generation backend (nvofg), created on rfx_commit.
    RfxBackendId fgBackend = RFX_BACKEND_NONE;
    NvofgContext* nvofg = nullptr;
    bool registered = false;
    uint64_t framesGenerated = 0;   // runtime statistics counter
};
