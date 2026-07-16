// nvofg present pacer (design §7) — a CPU-side even-spacing scheduler. Touches no
// Vulkan; keeps pacing isolated from the core so it can never DEVICE_LOST it (§9).
#include "nvofg_pacer.h"

#include <chrono>
#include <new>
#include <thread>

namespace {
uint64_t now_ns() {
    using namespace std::chrono;
    return (uint64_t)duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
}
}  // namespace

struct NvofgPacer {
    uint64_t period_ns = 0;    // display period
    uint32_t multiplier = 2;
    float    flow_scale = 1.0f;
    NvofgQuality quality = NVOFG_QUALITY_BALANCED;
    uint32_t use_present_wait = 0;

    bool     started = false;
    uint64_t next_slot = 0;    // target time of the next slot to hand out

    void recompute_period(double hz) {
        period_ns = (hz > 0.0) ? (uint64_t)(1.0e9 / hz) : (uint64_t)(1.0e9 / 60.0);
    }
};

extern "C" {

NvofgPacer* nvofg_pacer_create(const NvofgPacerConfig* cfg) {
    if (!cfg) return nullptr;
    auto* p = new (std::nothrow) NvofgPacer();
    if (!p) return nullptr;
    p->recompute_period(cfg->display_hz);
    p->multiplier = cfg->multiplier ? cfg->multiplier : 2;
    p->use_present_wait = cfg->use_present_wait;
    return p;
}

void nvofg_pacer_destroy(NvofgPacer* p) { delete p; }

void nvofg_pacer_tune(NvofgPacer* p, uint32_t multiplier, float flow_scale, NvofgQuality quality) {
    if (!p) return;
    if (multiplier) p->multiplier = multiplier;
    p->flow_scale = flow_scale;
    p->quality = quality;
}

uint64_t nvofg_pacer_wait_slot(NvofgPacer* p) {
    if (!p) return 0;
    const uint64_t t = now_ns();
    if (!p->started) {
        p->started = true;
        p->next_slot = t + p->period_ns;
        return t;   // first present goes out immediately
    }
    uint64_t slot = p->next_slot;
    // Fallen behind by more than a slot -> re-baseline to avoid catch-up bursts.
    if (t > slot + p->period_ns) {
        slot = t;
    } else if (t < slot) {
        // Sleep most of the way, then short-spin for accuracy near the deadline.
        const uint64_t spin_margin = 300000;  // 0.3 ms
        if (slot - t > spin_margin) {
            std::this_thread::sleep_for(std::chrono::nanoseconds(slot - t - spin_margin));
        }
        while (now_ns() < slot) std::this_thread::yield();
    }
    p->next_slot = slot + p->period_ns;
    return slot;
}

uint64_t nvofg_pacer_next_slot_time(NvofgPacer* p) {
    if (!p) return 0;
    return p->started ? p->next_slot : now_ns();
}

void nvofg_pacer_reset(NvofgPacer* p) {
    if (p) { p->started = false; p->next_slot = 0; }
}

double nvofg_pacer_base_fps_cap(const NvofgPacer* p) {
    if (!p || p->period_ns == 0 || p->multiplier == 0) return 0.0;
    return (1.0e9 / (double)p->period_ns) / (double)p->multiplier;
}

float        nvofg_pacer_flow_scale(const NvofgPacer* p) { return p ? p->flow_scale : 1.0f; }
NvofgQuality nvofg_pacer_quality(const NvofgPacer* p)    { return p ? p->quality : NVOFG_QUALITY_BALANCED; }

}  // extern "C"
