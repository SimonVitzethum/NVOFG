// RenderFX backend resolver (design.md §18.2): resolve a *compatible set* of backends
// across the enabled stages — never a per-stage argmax. Pure function of
// (capabilities, preference): deterministic, inspectable, and unit-testable without a GPU.
#include "renderfx.h"

#include <climits>
#include <cstring>

namespace {

const RfxBackendCaps* findCaps(const RfxCapabilities* caps, RfxStage s, RfxBackendId id) {
    for (uint32_t i = 0; i < caps->count; ++i)
        if (caps->backends[i].stage == s && caps->backends[i].id == id) return &caps->backends[i];
    return nullptr;
}

// Higher is better. Quality preference up-weights quality_tier; perf up-weights low cost.
// A disabled/NONE stage is neutral (0); any real supported backend scores positive, so
// an enabled stage prefers a real backend and only falls to NONE when constraints force it.
int stageScore(const RfxCapabilities* caps, RfxStage s, RfxBackendId id,
               const RfxPreference* pref) {
    if (id == RFX_BACKEND_NONE) return 0;
    const RfxBackendCaps* b = findCaps(caps, s, id);
    if (!b) return -1000;   // shouldn't happen (candidate came from the table)
    int q = (int)b->quality_tier;
    int c = (int)b->cost_hint;
    // Base 1000 so any valid real backend beats a disabled stage (NONE=0); the
    // policy-weighted delta only ranks real backends against each other.
    switch (pref->quality) {
        case RFX_QUALITY_QUALITY:  return 1000 + q * 2 - c;
        case RFX_QUALITY_PERF:     return 1000 + q - c * 2;
        default:                   return 1000 + q - c;
    }
}

// Candidate backends for a stage, respecting override / policy. Returns count; NONE is
// always index 0 for enabled stages without an override (graceful disable fallback).
uint32_t candidates(const RfxCapabilities* caps, const RfxPreference* pref, RfxStage s,
                    RfxBackendId out[RFX_BACKEND_COUNT]) {
    const bool enabled = (pref->stages_enabled & (1u << s)) != 0;
    if (!enabled) { out[0] = RFX_BACKEND_NONE; return 1; }

    const RfxBackendId ov = pref->override_backend[s];
    if (ov != RFX_BACKEND_COUNT) {
        // Strict override: only that backend, and only if supported.
        const RfxBackendCaps* b = findCaps(caps, s, ov);
        if (ov == RFX_BACKEND_NONE) { out[0] = RFX_BACKEND_NONE; return 1; }
        if (b && b->supported) { out[0] = ov; return 1; }
        return 0;  // override unsatisfiable -> no candidate -> resolution fails
    }

    uint32_t n = 0;
    out[n++] = RFX_BACKEND_NONE;  // fallback
    for (uint32_t i = 0; i < caps->count; ++i) {
        const RfxBackendCaps* b = &caps->backends[i];
        if (b->stage != s || !b->supported || b->id == RFX_BACKEND_NONE) continue;
        if (!pref->allow_proprietary && b->proprietary) continue;
        if (pref->vendor_pin != RFX_FAMILY_GENERIC && b->family != pref->vendor_pin) continue;
        out[n++] = b->id;
    }
    return n;
}

bool comboValid(const RfxCapabilities* caps, const RfxBackendId pick[RFX_STAGE_COUNT]) {
    // Cross-stage family constraints: a chosen backend requiring family F in stage S
    // needs stage S's chosen backend to be of family F (and not NONE).
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        if (pick[s] == RFX_BACKEND_NONE) continue;
        const RfxBackendCaps* b = findCaps(caps, (RfxStage)s, pick[s]);
        if (!b) return false;
        if (b->requires_family_stage != RFX_STAGE_COUNT) {
            RfxStage rs = b->requires_family_stage;
            const RfxBackendCaps* rb = findCaps(caps, rs, pick[rs]);
            if (!rb || pick[rs] == RFX_BACKEND_NONE || rb->family != b->requires_family)
                return false;
        }
    }
    return true;
}

}  // namespace

extern "C" RfxResult rfx_resolve(const RfxCapabilities* caps, const RfxPreference* pref,
                                 RfxSelection* out) {
    if (!caps || !pref || !out) return RFX_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));

    RfxBackendId cand[RFX_STAGE_COUNT][RFX_BACKEND_COUNT];
    uint32_t nc[RFX_STAGE_COUNT];
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        nc[s] = candidates(caps, pref, (RfxStage)s, cand[s]);
        if (nc[s] == 0) {  // an override was unsatisfiable
            out->valid = 0;
            return RFX_NO_VALID_SELECTION;
        }
    }

    // Brute-force the (tiny) product of candidate sets; keep the highest-scoring valid combo.
    RfxBackendId best[RFX_STAGE_COUNT] = {RFX_BACKEND_NONE, RFX_BACKEND_NONE, RFX_BACKEND_NONE};
    int bestScore = INT_MIN;
    bool found = false;
    for (uint32_t i = 0; i < nc[0]; ++i)
        for (uint32_t j = 0; j < nc[1]; ++j)
            for (uint32_t k = 0; k < nc[2]; ++k) {
                RfxBackendId pick[RFX_STAGE_COUNT] = {cand[0][i], cand[1][j], cand[2][k]};
                if (!comboValid(caps, pick)) continue;
                int total = stageScore(caps, RFX_STAGE_UPSCALING, pick[0], pref)
                          + stageScore(caps, RFX_STAGE_RAY_RECONSTRUCTION, pick[1], pref)
                          + stageScore(caps, RFX_STAGE_FRAME_GENERATION, pick[2], pref);
                if (total > bestScore) {
                    bestScore = total;
                    best[0] = pick[0]; best[1] = pick[1]; best[2] = pick[2];
                    found = true;
                }
            }

    if (!found) { out->valid = 0; return RFX_NO_VALID_SELECTION; }

    out->valid = 1;
    out->required_inputs = 0;
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        out->backend[s] = best[s];
        const RfxBackendCaps* b = findCaps(caps, (RfxStage)s, best[s]);
        if (best[s] == RFX_BACKEND_NONE) {
            out->reason[s] = (pref->stages_enabled & (1u << s)) ? "disabled: no compatible backend"
                                                                : "stage not requested";
        } else {
            out->reason[s] = (b && b->note) ? b->note : "selected by policy";
            if (b) out->required_inputs |= b->required_inputs;
        }
    }
    return RFX_OK;
}
