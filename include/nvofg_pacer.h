/*
 * nvofg_pacer.h — optional present pacer (design §7).
 *
 * Producing interpolated frames is the core library's job; *scheduling* them onto
 * the swapchain with even spacing is this small, separate, opt-in helper. Engines
 * with their own frame scheduler ignore it entirely.
 *
 * The pacer is a CPU-side, monotonic-clock scheduler: it hands out evenly spaced
 * display "slots" (at display_hz) so a `[gen] real [gen] real …` sequence presents
 * smoothly instead of in bursts. It performs no Vulkan work itself — the app calls
 * nvofg_pacer_wait_slot() right before each vkQueuePresentKHR (real OR generated),
 * or reads nvofg_pacer_next_slot_time() to drive VK_KHR_present_wait.
 *
 * Kept deliberately outside the core (design §9): pacing bugs must never be able to
 * DEVICE_LOST the renderer, so this module touches no queues.
 *
 * Licensed under Apache-2.0 OR MIT.
 */
#ifndef NVOFG_PACER_H
#define NVOFG_PACER_H

#include <stdint.h>
#include "nvofg.h"   /* NvofgQuality */

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NvofgPacer NvofgPacer;

typedef struct NvofgPacerConfig {
    double   display_hz;       /* refresh rate, e.g. 120.0                          */
    uint32_t multiplier;       /* presented frames per real frame (2/3/4)           */
    uint32_t use_present_wait; /* 1 = caller drives VK_KHR_present_wait with the
                                  returned slot time; 0 = pacer sleeps on a
                                  monotonic clock. Informational; see the header. */
} NvofgPacerConfig;

NvofgPacer* nvofg_pacer_create(const NvofgPacerConfig* cfg);
void        nvofg_pacer_destroy(NvofgPacer* pacer);

/* Adjust at runtime (settings menu). flow_scale/quality are stored hints the app
 * may read back; multiplier changes the base-fps cap. */
void nvofg_pacer_tune(NvofgPacer* pacer, uint32_t multiplier, float flow_scale,
                      NvofgQuality quality);

/* Call immediately before each present (real or generated). Blocks until the next
 * evenly spaced display slot, then returns that slot's target time in nanoseconds
 * (monotonic clock). The first call after create/reset establishes the baseline
 * and returns immediately. If the caller has fallen behind by more than one slot,
 * the schedule re-baselines to now (no runaway catch-up bursts). */
uint64_t nvofg_pacer_wait_slot(NvofgPacer* pacer);

/* The next slot's target time (ns) without blocking — for VK_KHR_present_wait
 * callers who pass it as the target. Does not advance the schedule. */
uint64_t nvofg_pacer_next_slot_time(NvofgPacer* pacer);

/* Re-baseline the schedule (call on a hitch, pause, resize, or vsync change). */
void nvofg_pacer_reset(NvofgPacer* pacer);

/* Recommended base render fps cap = display_hz / multiplier. */
double nvofg_pacer_base_fps_cap(const NvofgPacer* pacer);

/* Stored tuning hints (for the app; e.g. to drive nvofg quality/flow scale). */
float        nvofg_pacer_flow_scale(const NvofgPacer* pacer);
NvofgQuality nvofg_pacer_quality(const NvofgPacer* pacer);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NVOFG_PACER_H */
