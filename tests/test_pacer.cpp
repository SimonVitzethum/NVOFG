// Present-pacer scheduling test. No Vulkan; validates the even-spacing math,
// base-fps cap, re-baseline, and tuning. Timing tolerances are loose to stay
// non-flaky under scheduler jitter.
#include "nvofg_pacer.h"

#include <cstdint>
#include <cstdio>

#define REQUIRE(c, m)                                                            \
    do { if (!(c)) { std::fprintf(stderr, "FAIL: %s\n", m); return 1; }          \
         std::printf("ok: %s\n", m); } while (0)

int main() {
    NvofgPacerConfig cfg{};
    cfg.display_hz = 2000.0;   // 0.5 ms period -> fast, keeps the test short
    cfg.multiplier = 2;
    NvofgPacer* p = nvofg_pacer_create(&cfg);
    REQUIRE(p != nullptr, "pacer created");

    // base fps cap = display_hz / multiplier
    double cap = nvofg_pacer_base_fps_cap(p);
    REQUIRE(cap > 999.0 && cap < 1001.0, "base fps cap = 1000 (2000/2)");

    const uint64_t period = (uint64_t)(1.0e9 / 2000.0);

    // Slots come out monotonically increasing and roughly one period apart.
    uint64_t prev = nvofg_pacer_wait_slot(p);   // first: immediate baseline
    bool monotonic = true;
    double sum_spacing = 0.0;
    const int N = 8;
    for (int i = 0; i < N; ++i) {
        uint64_t s = nvofg_pacer_wait_slot(p);
        if (s <= prev) monotonic = false;
        sum_spacing += double(s - prev);
        prev = s;
    }
    REQUIRE(monotonic, "slot times strictly increasing");
    double avg = sum_spacing / N;
    // Generous window: scheduler sleep granularity varies, but should track period.
    REQUIRE(avg > 0.3 * period && avg < 4.0 * period, "avg spacing tracks the display period");

    // next_slot_time does not advance the schedule.
    uint64_t a = nvofg_pacer_next_slot_time(p);
    uint64_t b = nvofg_pacer_next_slot_time(p);
    REQUIRE(a == b, "next_slot_time is side-effect free");

    // tune updates the multiplier -> base cap changes.
    nvofg_pacer_tune(p, 4, 0.75f, NVOFG_QUALITY_HIGH);
    REQUIRE(nvofg_pacer_base_fps_cap(p) > 499.0 && nvofg_pacer_base_fps_cap(p) < 501.0,
            "tune multiplier=4 -> base cap 500");
    REQUIRE(nvofg_pacer_flow_scale(p) > 0.74f && nvofg_pacer_flow_scale(p) < 0.76f,
            "tune stores flow_scale");
    REQUIRE(nvofg_pacer_quality(p) == NVOFG_QUALITY_HIGH, "tune stores quality");

    // reset re-baselines (next wait returns immediately again).
    nvofg_pacer_reset(p);
    uint64_t t0 = nvofg_pacer_wait_slot(p);
    uint64_t t1 = nvofg_pacer_next_slot_time(p);
    REQUIRE(t1 > t0, "after reset, schedule re-baselines");

    nvofg_pacer_destroy(p);
    std::printf("ALL PACER CHECKS PASSED\n");
    return 0;
}
