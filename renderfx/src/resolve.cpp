// RenderFX resolver (design.md §18): the four explicit concepts —
//   Capabilities + Preferences(intent) -> Policy -> Selection(explained).
// rfx_resolve is a PURE function: derive policy, resolve a *compatible set* of backends
// across stages (never per-stage argmax), and explain every choice. Headless-testable.
#include "renderfx.h"

#include <climits>
#include <cstring>

namespace {

const RfxBackendCaps* findCaps(const RfxCapabilities* caps, RfxStage s, RfxBackendId id) {
    for (uint32_t i = 0; i < caps->count; ++i)
        if (caps->backends[i].stage == s && caps->backends[i].id == id) return &caps->backends[i];
    return nullptr;
}

int stageScore(const RfxCapabilities* caps, RfxStage s, RfxBackendId id, const RfxPolicy* pol) {
    if (id == RFX_BACKEND_NONE) return 0;                 // disabled stage is neutral
    const RfxBackendCaps* b = findCaps(caps, s, id);
    if (!b) return -1000;
    // Base 1000 so any valid real backend beats a disabled stage; policy weights rank
    // real backends against each other.
    int sc = 1000 + pol->quality_weight * (int)b->quality_tier - pol->cost_weight * (int)b->cost_hint;
    if (pol->prefer_native && b->family == RFX_FAMILY_GENERIC) sc += 200;  // debug bias
    return sc;
}

// Candidate backends for a stage under (preference override + derived policy).
uint32_t candidates(const RfxCapabilities* caps, const RfxPreference* pref, const RfxPolicy* pol,
                    RfxStage s, RfxBackendId out[RFX_BACKEND_COUNT]) {
    const bool enabled = (pref->stages_enabled & (1u << s)) != 0;
    if (!enabled) { out[0] = RFX_BACKEND_NONE; return 1; }

    const RfxBackendId ov = pref->override_backend[s];
    if (ov != RFX_BACKEND_COUNT) {
        if (ov == RFX_BACKEND_NONE) { out[0] = RFX_BACKEND_NONE; return 1; }
        const RfxBackendCaps* b = findCaps(caps, s, ov);
        if (b && b->supported) { out[0] = ov; return 1; }
        return 0;  // override unsatisfiable -> no valid selection
    }

    uint32_t n = 0;
    out[n++] = RFX_BACKEND_NONE;
    for (uint32_t i = 0; i < caps->count; ++i) {
        const RfxBackendCaps* b = &caps->backends[i];
        if (b->stage != s || !b->supported || b->id == RFX_BACKEND_NONE) continue;
        if (pol->exclude_proprietary && b->proprietary) continue;
        if (pol->require_deterministic && !b->deterministic) continue;
        if (pol->vendor_pin != RFX_FAMILY_GENERIC && b->family != pol->vendor_pin) continue;
        out[n++] = b->id;
    }
    return n;
}

bool comboValid(const RfxCapabilities* caps, const RfxBackendId pick[RFX_STAGE_COUNT]) {
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        if (pick[s] == RFX_BACKEND_NONE) continue;
        const RfxBackendCaps* b = findCaps(caps, (RfxStage)s, pick[s]);
        if (!b) return false;
        if (b->requires_family_stage != RFX_STAGE_COUNT) {
            RfxStage rs = b->requires_family_stage;
            const RfxBackendCaps* rb = findCaps(caps, rs, pick[rs]);
            if (!rb || pick[rs] == RFX_BACKEND_NONE || rb->family != b->requires_family) return false;
        }
    }
    return true;
}

void addReason(RfxStageExplanation& e, RfxReason r) {
    for (uint32_t i = 0; i < e.reason_count; ++i) if (e.reasons[i] == r) return;
    if (e.reason_count < RFX_MAX_REASONS) e.reasons[e.reason_count++] = r;
}

// Explain one stage's choice against its candidate set and the whole table.
void explain(const RfxCapabilities* caps, const RfxPreference* pref, const RfxPolicy* pol,
             RfxStage s, RfxBackendId chosen, RfxStageExplanation& e) {
    std::memset(&e, 0, sizeof(e));
    e.backend = chosen;
    const bool enabled = (pref->stages_enabled & (1u << s)) != 0;

    if (chosen == RFX_BACKEND_NONE) {
        addReason(e, enabled ? RFX_REASON_DISABLED_NO_COMPATIBLE : RFX_REASON_STAGE_NOT_REQUESTED);
        return;
    }
    const RfxBackendCaps* b = findCaps(caps, s, chosen);
    addReason(e, RFX_REASON_SUPPORTED);
    if (pref->override_backend[s] == chosen) addReason(e, RFX_REASON_OVERRIDDEN);

    // quality context: is a higher-tier backend for this stage present but unavailable?
    uint32_t maxAll = 0, maxCand = 0;
    RfxBackendId cand[RFX_BACKEND_COUNT];
    uint32_t nc = candidates(caps, pref, pol, s, cand);
    for (uint32_t i = 0; i < caps->count; ++i)
        if (caps->backends[i].stage == s && caps->backends[i].quality_tier > maxAll)
            maxAll = caps->backends[i].quality_tier;
    for (uint32_t i = 0; i < nc; ++i) {
        const RfxBackendCaps* cb = findCaps(caps, s, cand[i]);
        if (cb && cb->quality_tier > maxCand) maxCand = cb->quality_tier;
    }
    if (b) {
        if (b->quality_tier >= maxCand) addReason(e, RFX_REASON_HIGHEST_QUALITY);
        if (b->quality_tier < maxAll)   addReason(e, RFX_REASON_ALTERNATIVE_UNAVAILABLE);
        addReason(e, b->proprietary ? RFX_REASON_PROPRIETARY_ALLOWED : RFX_REASON_OPEN_SOURCE);
        if (pol->vendor_pin != RFX_FAMILY_GENERIC && b->family == pol->vendor_pin)
            addReason(e, RFX_REASON_VENDOR_PINNED);
        if (pol->require_deterministic && b->deterministic) addReason(e, RFX_REASON_DETERMINISTIC);
        if (pol->prefer_native && b->family == RFX_FAMILY_GENERIC) addReason(e, RFX_REASON_PREFERRED_NATIVE);
        if (b->requires_family_stage != RFX_STAGE_COUNT) addReason(e, RFX_REASON_CONSTRAINT_SATISFIED);
        if (pol->cost_weight >= 2) addReason(e, RFX_REASON_LOWEST_COST);
    }
}

}  // namespace

extern "C" RfxResult rfx_derive_policy(const RfxPreference* pref, RfxPolicy* out) {
    if (!pref || !out) return RFX_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));
    int qw = 1, cw = 1;
    switch (pref->quality) {
        case RFX_QUALITY_QUALITY: qw = 2; cw = 1; break;
        case RFX_QUALITY_PERF:    qw = 1; cw = 2; break;
        default:                  qw = 1; cw = 1; break;
    }
    if (pref->prioritize_latency) cw += 1;
    if (pref->power_saver)        cw += 2;
    out->quality_weight = qw;
    out->cost_weight = cw;
    out->exclude_proprietary = (pref->open_source_only || !pref->allow_proprietary) ? 1 : 0;
    out->require_deterministic = (pref->deterministic || pref->debug) ? 1 : 0;
    out->vendor_pin = pref->vendor_pin;
    out->prefer_native = pref->debug ? 1 : 0;
    return RFX_OK;
}

extern "C" RfxResult rfx_resolve(const RfxCapabilities* caps, const RfxPreference* pref,
                                 RfxSelection* out) {
    if (!caps || !pref || !out) return RFX_INVALID_ARGUMENT;
    std::memset(out, 0, sizeof(*out));

    RfxPolicy pol;
    rfx_derive_policy(pref, &pol);

    RfxBackendId cand[RFX_STAGE_COUNT][RFX_BACKEND_COUNT];
    uint32_t nc[RFX_STAGE_COUNT];
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        nc[s] = candidates(caps, pref, &pol, (RfxStage)s, cand[s]);
        if (nc[s] == 0) { out->valid = 0; return RFX_NO_VALID_SELECTION; }
    }

    RfxBackendId best[RFX_STAGE_COUNT] = {RFX_BACKEND_NONE, RFX_BACKEND_NONE, RFX_BACKEND_NONE};
    int bestScore = INT_MIN;
    bool found = false;
    for (uint32_t i = 0; i < nc[0]; ++i)
        for (uint32_t j = 0; j < nc[1]; ++j)
            for (uint32_t k = 0; k < nc[2]; ++k) {
                RfxBackendId pick[RFX_STAGE_COUNT] = {cand[0][i], cand[1][j], cand[2][k]};
                if (!comboValid(caps, pick)) continue;
                int total = stageScore(caps, RFX_STAGE_UPSCALING, pick[0], &pol)
                          + stageScore(caps, RFX_STAGE_RAY_RECONSTRUCTION, pick[1], &pol)
                          + stageScore(caps, RFX_STAGE_FRAME_GENERATION, pick[2], &pol);
                if (total > bestScore) { bestScore = total; best[0] = pick[0]; best[1] = pick[1]; best[2] = pick[2]; found = true; }
            }

    if (!found) { out->valid = 0; return RFX_NO_VALID_SELECTION; }

    out->valid = 1;
    out->required_inputs = 0;
    for (int s = 0; s < RFX_STAGE_COUNT; ++s) {
        out->backend[s] = best[s];
        explain(caps, pref, &pol, (RfxStage)s, best[s], out->explanation[s]);
        if (best[s] != RFX_BACKEND_NONE) {
            const RfxBackendCaps* b = findCaps(caps, (RfxStage)s, best[s]);
            if (b) out->required_inputs |= b->required_inputs;
        }
    }
    return RFX_OK;
}
