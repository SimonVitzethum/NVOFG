// RenderFX resolver tests — pure, headless (no GPU). Exercises policy selection,
// overrides, and the cross-stage set-resolution constraints (design.md §18.2).
#include "renderfx.h"

#include <cstdio>
#include <cstring>

namespace renderfx { void buildCapabilities(bool, bool, RfxCapabilities*); }

#define REQUIRE(c, m)                                                            \
    do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); return 1; }          \
         std::printf("ok: %s\n", m); } while (0)

static void setSupported(RfxCapabilities& c, RfxBackendId id, uint32_t v) {
    for (uint32_t i = 0; i < c.count; ++i) if (c.backends[i].id == id) c.backends[i].supported = v;
}

static RfxPreference basePref() {
    RfxPreference p{};
    p.quality = RFX_QUALITY_QUALITY;
    p.allow_proprietary = 1;
    p.vendor_pin = RFX_FAMILY_GENERIC;
    p.stages_enabled = 0;
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) p.override_backend[s] = RFX_BACKEND_COUNT;
    return p;
}

int main() {
    // A) FG only, OFA present -> nvofg wins over shader (higher tier).
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        RfxPreference p = basePref(); p.allow_proprietary = 0;
        p.stages_enabled = 1u << RFX_STAGE_FRAME_GENERATION;
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_OK && s.valid, "resolve OK (FG only)");
        REQUIRE(s.backend[RFX_STAGE_FRAME_GENERATION] == RFX_BACKEND_NVOFG, "FG -> nvofg (OFA)");
        REQUIRE((s.required_inputs & RFX_INPUT_MOTION) && (s.required_inputs & RFX_INPUT_DEPTH),
                "required inputs include motion+depth");
    }

    // B) No OFA -> portable shader FG.
    {
        RfxCapabilities c; renderfx::buildCapabilities(false, true, &c);
        RfxPreference p = basePref();
        p.stages_enabled = 1u << RFX_STAGE_FRAME_GENERATION;
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_OK, "resolve OK (no OFA)");
        REQUIRE(s.backend[RFX_STAGE_FRAME_GENERATION] == RFX_BACKEND_SHADER_FG, "FG -> shader (no OFA)");
    }

    // C) Cross-stage: DLSS RR selected only alongside a DLSS upscaler (constraint holds).
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        setSupported(c, RFX_BACKEND_DLSS_SR, 1);
        setSupported(c, RFX_BACKEND_DLSS_RR, 1);
        RfxPreference p = basePref();
        p.stages_enabled = (1u << RFX_STAGE_UPSCALING) | (1u << RFX_STAGE_RAY_RECONSTRUCTION)
                         | (1u << RFX_STAGE_FRAME_GENERATION);
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_OK && s.valid, "resolve OK (SR+RR+FG)");
        REQUIRE(s.backend[RFX_STAGE_UPSCALING] == RFX_BACKEND_DLSS_SR, "Upscaling -> DLSS SR");
        REQUIRE(s.backend[RFX_STAGE_RAY_RECONSTRUCTION] == RFX_BACKEND_DLSS_RR,
                "RR -> DLSS RR (DLSS upscaler present)");
        REQUIRE(s.backend[RFX_STAGE_FRAME_GENERATION] == RFX_BACKEND_NVOFG, "FG -> nvofg");
    }

    // D) DLSS RR CANNOT be selected without a DLSS upscaler (proprietary disallowed).
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        setSupported(c, RFX_BACKEND_DLSS_SR, 1);
        setSupported(c, RFX_BACKEND_DLSS_RR, 1);
        RfxPreference p = basePref(); p.allow_proprietary = 0;   // excludes DLSS family
        p.stages_enabled = (1u << RFX_STAGE_UPSCALING) | (1u << RFX_STAGE_RAY_RECONSTRUCTION);
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_OK, "resolve OK (no proprietary)");
        REQUIRE(s.backend[RFX_STAGE_UPSCALING] == RFX_BACKEND_NATIVE, "Upscaling -> Native (no DLSS)");
        REQUIRE(s.backend[RFX_STAGE_RAY_RECONSTRUCTION] == RFX_BACKEND_NONE,
                "RR -> None (constraint: needs a DLSS upscaler)");
    }

    // E) Strict override of an unsupported backend -> no valid selection (reported, not silent).
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        RfxPreference p = basePref();
        p.stages_enabled = 1u << RFX_STAGE_FRAME_GENERATION;
        p.override_backend[RFX_STAGE_FRAME_GENERATION] = RFX_BACKEND_DLSS_FG;  // unsupported here
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_NO_VALID_SELECTION && !s.valid,
                "override of unsupported backend -> NO_VALID_SELECTION");
    }

    // F) Vendor pin + bundle: FSR FG pairs with FSR upscaling.
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        setSupported(c, RFX_BACKEND_FSR, 1);
        setSupported(c, RFX_BACKEND_FSR_FG, 1);
        RfxPreference p = basePref(); p.vendor_pin = RFX_FAMILY_FSR;
        p.stages_enabled = (1u << RFX_STAGE_UPSCALING) | (1u << RFX_STAGE_FRAME_GENERATION);
        RfxSelection s;
        REQUIRE(rfx_resolve(&c, &p, &s) == RFX_OK && s.valid, "resolve OK (FSR pin)");
        REQUIRE(s.backend[RFX_STAGE_UPSCALING] == RFX_BACKEND_FSR, "Upscaling -> FSR (pinned)");
        REQUIRE(s.backend[RFX_STAGE_FRAME_GENERATION] == RFX_BACKEND_FSR_FG,
                "FG -> FSR FG (bundle with FSR upscaler)");
    }

    // G) Explanations: FG=nvofg is supported; DLSS FG exists but is unavailable here.
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        RfxPreference p = basePref();
        p.stages_enabled = 1u << RFX_STAGE_FRAME_GENERATION;
        RfxSelection s; rfx_resolve(&c, &p, &s);
        const RfxStageExplanation& e = s.explanation[RFX_STAGE_FRAME_GENERATION];
        bool hasSupported = false, hasAltUnavail = false;
        for (uint32_t i = 0; i < e.reason_count; ++i) {
            if (e.reasons[i] == RFX_REASON_SUPPORTED) hasSupported = true;
            if (e.reasons[i] == RFX_REASON_ALTERNATIVE_UNAVAILABLE) hasAltUnavail = true;
        }
        REQUIRE(e.backend == RFX_BACKEND_NVOFG && hasSupported, "explanation: nvofg supported");
        REQUIRE(hasAltUnavail, "explanation: alternative unavailable (DLSS FG higher tier, absent)");
    }

    // H) Intent 'open_source_only' excludes proprietary DLSS even if supported.
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        setSupported(c, RFX_BACKEND_DLSS_SR, 1);
        RfxPreference p = basePref(); p.open_source_only = 1;
        p.stages_enabled = 1u << RFX_STAGE_UPSCALING;
        RfxSelection s; rfx_resolve(&c, &p, &s);
        REQUIRE(s.backend[RFX_STAGE_UPSCALING] == RFX_BACKEND_NATIVE,
                "open_source_only -> Native (DLSS excluded)");
    }

    // I) Intent 'deterministic' excludes non-deterministic (neural) backends.
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        setSupported(c, RFX_BACKEND_DLSS_SR, 1);
        RfxPreference p = basePref(); p.deterministic = 1;
        p.stages_enabled = 1u << RFX_STAGE_UPSCALING;
        RfxSelection s; rfx_resolve(&c, &p, &s);
        REQUIRE(s.backend[RFX_STAGE_UPSCALING] != RFX_BACKEND_DLSS_SR,
                "deterministic -> excludes DLSS (non-deterministic)");
    }

    // J) Policy derivation is explicit and inspectable.
    {
        RfxPreference p = basePref(); p.quality = RFX_QUALITY_PERF; p.power_saver = 1; p.open_source_only = 1;
        RfxPolicy pol;
        REQUIRE(rfx_derive_policy(&p, &pol) == RFX_OK, "derive_policy OK");
        REQUIRE(pol.exclude_proprietary == 1, "policy: open_source_only -> exclude_proprietary");
        REQUIRE(pol.cost_weight >= 4, "policy: perf+power_saver -> high cost weight");
    }

    // K) Stable serializable identifiers: name<->id round-trips for every backend.
    {
        bool ok = true;
        for (int i = 0; i < RFX_BACKEND_COUNT; ++i)
            if (rfx_backend_from_name(rfx_backend_name((RfxBackendId)i)) != (RfxBackendId)i) ok = false;
        REQUIRE(ok, "backend name<->id round-trips (stable serialization)");
        REQUIRE(RFX_BACKEND_NVOFG == 10, "stable numeric id (nvofg == 10)");
    }

    // L) Pipeline inspection: format the resolved pipeline.
    {
        RfxCapabilities c; renderfx::buildCapabilities(true, true, &c);
        RfxPreference p = basePref();
        p.stages_enabled = 1u << RFX_STAGE_FRAME_GENERATION;
        RfxSelection s; rfx_resolve(&c, &p, &s);
        char buf[512];
        uint32_t n = rfx_format_pipeline(&s, buf, sizeof(buf));
        REQUIRE(n > 0 && std::strstr(buf, "nvofg") && std::strstr(buf, "Present"),
                "format_pipeline renders the resolved pipeline");
        std::printf("---\n%s---\n", buf);
    }

    std::printf("ALL RENDERFX RESOLVER CHECKS PASSED\n");
    return 0;
}
