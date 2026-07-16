/*
 * nvofg.h — native Vulkan frame generation on NVIDIA Optical Flow (OFA).
 *
 * C ABI. All functions are `nvofg_*`, return `NvofgResult`, and never allocate on
 * the per-frame hot path. The library drives the OFA exclusively through the
 * standard `VK_NV_optical_flow` Vulkan extension (no dlopen, no proprietary SDK).
 *
 * Licensed under Apache-2.0 OR MIT.
 */
#ifndef NVOFG_H
#define NVOFG_H

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NVOFG_VERSION_MAJOR 0
#define NVOFG_VERSION_MINOR 1
#define NVOFG_VERSION_PATCH 0

/* -------------------------------------------------------------------------- */
/* Result & enums                                                             */
/* -------------------------------------------------------------------------- */

typedef enum NvofgResult {
    NVOFG_OK = 0,
    NVOFG_UNSUPPORTED,        /* no OFA / driver too old / non-NVIDIA -> run 1x   */
    NVOFG_INVALID_ARGUMENT,
    NVOFG_NOT_REGISTERED,     /* record_generate before required resources bound */
    NVOFG_DEVICE_LOST,
    NVOFG_OUT_OF_MEMORY,
    NVOFG_INTERNAL,
} NvofgResult;

typedef enum NvofgQuality {
    NVOFG_QUALITY_PERF = 0,
    NVOFG_QUALITY_BALANCED,
    NVOFG_QUALITY_HIGH,
} NvofgQuality;

/* Integration mode. AUTOMATIC (default) fully encapsulates queues, layout
 * transitions and OFA synchronisation: the app only tells nvofg when the color
 * frames are ready (via NvofgGenerateInfo.input_*) and waits on the returned
 * NvofgFrameSync before presenting. EXTERNAL_COMMANDS is reserved for a future
 * expert mode where an engine drives command buffers / sync itself. */
typedef enum NvofgMode {
    NVOFG_MODE_AUTOMATIC = 0,
    NVOFG_MODE_EXTERNAL_COMMANDS,   /* reserved (not yet implemented)          */
} NvofgMode;

/* Which synthesis back-end turns flow into pixels (see design: modular interpolator). */
typedef enum NvofgInterpolator {
    NVOFG_INTERP_WARP = 0,    /* classical occlusion-aware forward/backward warp   */
    NVOFG_INTERP_CNN,         /* reserved: small learned model (cooperative matrix) */
    NVOFG_INTERP_TRANSFORMER, /* reserved                                          */
} NvofgInterpolator;

/* Creation / feature flags. */
enum {
    NVOFG_FLAG_USE_DEPTH      = 1u << 0, /* aux depth registered & used            */
    NVOFG_FLAG_USE_MOTION     = 1u << 1, /* app motion vectors registered & used   */
    NVOFG_FLAG_BIDIRECTIONAL  = 1u << 2, /* compute N->N-1 flow too (occlusion)    */
    NVOFG_FLAG_HDR            = 1u << 3, /* color images are HDR/scRGB linear      */
    NVOFG_FLAG_USE_UI_MASK    = 1u << 4, /* UI mask registered & applied           */
    NVOFG_FLAG_USE_REACTIVE   = 1u << 5, /* reactive mask registered & used        */
    NVOFG_FLAG_USE_MATERIAL_ID= 1u << 6, /* material/object id map registered      */
};

/* -------------------------------------------------------------------------- */
/* Capabilities                                                               */
/* -------------------------------------------------------------------------- */

typedef struct NvofgCaps {
    uint32_t supported;             /* 0 = no usable OFA (all else undefined)      */
    uint32_t bidirectional;         /* OFA supports backward/bidirectional flow    */
    uint32_t cost_map;              /* OFA can emit a per-cell cost/confidence map */
    uint32_t global_flow;           /* OFA can emit a global-flow hint             */
    uint32_t min_grid_size;         /* finest supported flow grid (e.g. 1)         */
    uint32_t max_grid_size;         /* coarsest supported flow grid (e.g. 4)       */
    uint32_t max_width, max_height; /* max input resolution the OFA accepts        */
    uint32_t hint_support;          /* OFA accepts external motion-vector hints    */
} NvofgCaps;

/* -------------------------------------------------------------------------- */
/* Creation                                                                   */
/* -------------------------------------------------------------------------- */

typedef struct NvofgContext NvofgContext;   /* opaque */

typedef struct NvofgCreateInfo {
    VkInstance        instance;
    VkPhysicalDevice  physical_device;
    VkDevice          device;
    VkQueue           queue;              /* queue nvofg submits OFA work on       */
    uint32_t          queue_family_index;
    PFN_vkGetInstanceProcAddr gipa;       /* resolve VK fns against app's loader   */
    uint32_t          width, height;      /* full present resolution               */
    NvofgQuality      quality;
    NvofgInterpolator interpolator;       /* NVOFG_INTERP_WARP for v1              */
    NvofgMode         mode;               /* NVOFG_MODE_AUTOMATIC for v1           */
    uint32_t          flags;              /* NVOFG_FLAG_*                          */
} NvofgCreateInfo;

/* Query without committing (for a settings menu). Fills caps; returns
 * NVOFG_UNSUPPORTED (and supported=0) if the device has no usable OFA. */
NvofgResult nvofg_query_support(VkInstance instance, VkPhysicalDevice pd,
                                PFN_vkGetInstanceProcAddr gipa, NvofgCaps* out);

/* Instance/device extensions nvofg needs the app to enable. Returns a
 * NULL-terminated array of C strings owned by the library (valid for process
 * lifetime); safe to call before nvofg_create. */
const char* const* nvofg_required_device_extensions(uint32_t* out_count);

NvofgResult nvofg_create(const NvofgCreateInfo* info, NvofgContext** out);
void        nvofg_destroy(NvofgContext* ctx);

/* -------------------------------------------------------------------------- */
/* Resource registration (once, at swapchain create / resize)                 */
/* -------------------------------------------------------------------------- */

typedef struct NvofgImageDesc {
    VkImage      image;
    VkImageView  view;
    VkFormat     format;
    uint32_t     width, height;
} NvofgImageDesc;

NvofgResult nvofg_register_color (NvofgContext*, const NvofgImageDesc* prev,
                                                 const NvofgImageDesc* curr);

/* Any pointer may be NULL if the corresponding NVOFG_FLAG_* is not set. */
typedef struct NvofgAuxDesc {
    const NvofgImageDesc* depth;       /* linear or device depth (see reproj)      */
    const NvofgImageDesc* motion;      /* app motion vectors (RG16F, pixels)       */
    const NvofgImageDesc* ui_mask;     /* R8: 255 = UI, never interpolate          */
    const NvofgImageDesc* reactive;    /* R8: 255 = reactive (alpha/particles)     */
    const NvofgImageDesc* material_id; /* R8/R16: object/material id               */
} NvofgAuxDesc;

NvofgResult nvofg_register_aux   (NvofgContext*, const NvofgAuxDesc* aux);
NvofgResult nvofg_register_output(NvofgContext*, const NvofgImageDesc* interpolated);
void        nvofg_unregister_all (NvofgContext*);

/* -------------------------------------------------------------------------- */
/* Per-frame recording (hot path)                                             */
/* -------------------------------------------------------------------------- */

/* Timeline-semaphore point the app must wait on before presenting `interpolated`. */
typedef struct NvofgFrameSync {
    VkSemaphore semaphore;   /* timeline semaphore owned by nvofg */
    uint64_t    value;       /* value it reaches when the frame is ready */
} NvofgFrameSync;

typedef struct NvofgGenerateInfo {
    float           phase;           /* 0..1 position of generated frame (0.5=2x)  */
    float           reproj[16];      /* prevVP_unjittered * inverse(currVP), row-major */
    float           near_plane, far_plane;
    uint32_t        reset;           /* 1 on camera cut/teleport -> duplicate frame */

    /* AUTOMATIC mode: the app signals this timeline value when prev/curr color
     * (and any aux) are finished rendering; nvofg waits on it before prep.
     * Pass input_timeline = VK_NULL_HANDLE to skip the wait (already ordered). */
    VkSemaphore     input_timeline;
    uint64_t        input_value;
    VkImageLayout   prev_layout;     /* layout prev color is in when ready         */
    VkImageLayout   curr_layout;     /* layout curr color is in when ready         */

    /* EXTERNAL_COMMANDS mode only (reserved): app-provided cmd buffer to record
     * the compute stages into. Ignored in AUTOMATIC mode. */
    VkCommandBuffer cmd;
} NvofgGenerateInfo;

/* Runs prep -> OFA execute -> refine -> interpolate -> composite. In AUTOMATIC
 * mode nvofg submits the internal command buffers itself (timeline-chained across
 * the compute and optical-flow queues) and returns, in *out_sync, the timeline
 * point the app must wait on before presenting the interpolated image. No CPU
 * stall: the call only records+submits, it does not wait on the GPU. */
NvofgResult nvofg_record_generate(NvofgContext*, const NvofgGenerateInfo*,
                                  NvofgFrameSync* out_sync);

/* -------------------------------------------------------------------------- */
/* Introspection & debug                                                      */
/* -------------------------------------------------------------------------- */

NvofgResult nvofg_caps(NvofgContext*, NvofgCaps* out);
const char* nvofg_last_error(NvofgContext*);

/* Debug visualisation targets (design: visualise flow/confidence/occlusion). */
typedef enum NvofgDebugView {
    NVOFG_DEBUG_NONE = 0,
    NVOFG_DEBUG_FLOW_FWD,
    NVOFG_DEBUG_FLOW_BWD,
    NVOFG_DEBUG_CONFIDENCE,
    NVOFG_DEBUG_OCCLUSION,
    NVOFG_DEBUG_DISOCCLUSION,
} NvofgDebugView;

/* When set (and an output registered), record_generate also writes the chosen
 * visualisation into `target`. NVOFG_DEBUG_NONE disables it. */
NvofgResult nvofg_set_debug_view(NvofgContext*, NvofgDebugView view,
                                 const NvofgImageDesc* target /*nullable*/);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NVOFG_H */
