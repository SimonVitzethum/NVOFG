/*
 * renderfx.h — a Linux-first, Vulkan-first modular rendering-effects framework.
 *
 * RenderFX is a SEPARATE project that *composes* focused effect libraries (nvofg for
 * frame generation today; NGX/FSR/XeSS backends later) behind ONE API. It owns no
 * render graph (design.md §15.2): it exposes logical stages the engine's own graph
 * schedules, and records into the app's command buffers.
 *
 * Design principle (design.md §18): **unify the API, never the implementation.** The
 * app programs against stages + a shared Frame Context and asks RenderFX what is
 * supported; RenderFX resolves a compatible *set* of backends (respecting cross-stage
 * constraints) that the app may inspect and override. Vendor names appear only as
 * backend *identifiers* — no vendor API type ever leaks into this header.
 *
 * Licensed under Apache-2.0 OR MIT.
 */
#ifndef RENDERFX_H
#define RENDERFX_H

#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RFX_VERSION 1

typedef enum RfxResult {
    RFX_OK = 0,
    RFX_UNSUPPORTED,
    RFX_INVALID_ARGUMENT,
    RFX_NO_VALID_SELECTION,   /* no backend combination satisfies the constraints */
    RFX_NOT_REGISTERED,
    RFX_INTERNAL,
} RfxResult;

/* Logical pipeline stages (design.md §15.2). Present is the app's own concern. */
typedef enum RfxStage {
    RFX_STAGE_UPSCALING = 0,
    RFX_STAGE_RAY_RECONSTRUCTION,
    RFX_STAGE_FRAME_GENERATION,
    RFX_STAGE_COUNT,
} RfxStage;

/* Backend identifiers. Vendor names are identifiers only — the *interface* is uniform. */
typedef enum RfxBackendId {
    RFX_BACKEND_NONE = 0,     /* stage disabled                                         */
    RFX_BACKEND_NATIVE,       /* built-in pass-through / bilinear (no vendor tech)       */
    RFX_BACKEND_TEMPORAL,     /* built-in temporal upscaler/AA                           */
    RFX_BACKEND_DLSS_SR,      /* NGX DLSS Super Resolution                               */
    RFX_BACKEND_DLAA,         /* NGX DLAA                                                */
    RFX_BACKEND_DLSS_RR,      /* NGX Ray Reconstruction                                  */
    RFX_BACKEND_DLSS_FG,      /* NGX DLSS Frame Generation (Windows-gated today)         */
    RFX_BACKEND_FSR,          /* AMD FidelityFX Super Resolution                         */
    RFX_BACKEND_FSR_FG,       /* FSR Frame Generation                                    */
    RFX_BACKEND_XESS,         /* Intel XeSS                                              */
    RFX_BACKEND_NVOFG,        /* native OFA/shader frame generation (this project)       */
    RFX_BACKEND_SHADER_FG,    /* portable shader frame generation (nvofg Tier B)         */
    RFX_BACKEND_COUNT,
} RfxBackendId;

/* Vendor/algorithm families — used for cross-stage constraints (design.md §18.2). */
typedef enum RfxBackendFamily {
    RFX_FAMILY_GENERIC = 0,
    RFX_FAMILY_DLSS,
    RFX_FAMILY_FSR,
    RFX_FAMILY_XESS,
    RFX_FAMILY_NVOFG,
} RfxBackendFamily;

/* Frame Context input bits (design.md §15.1). A backend declares which it needs; the
 * app produces only the union the resolved selection requires. */
enum {
    RFX_INPUT_COLOR           = 1u << 0,
    RFX_INPUT_DEPTH           = 1u << 1,
    RFX_INPUT_MOTION          = 1u << 2,
    RFX_INPUT_MATERIAL_ID     = 1u << 3,
    RFX_INPUT_REACTIVE        = 1u << 4,
    RFX_INPUT_EXPOSURE        = 1u << 5,
    RFX_INPUT_JITTER          = 1u << 6,
    RFX_INPUT_REPROJ          = 1u << 7,
    RFX_INPUT_ROUGHNESS       = 1u << 8,
    RFX_INPUT_ALBEDO_DIFFUSE  = 1u << 9,
    RFX_INPUT_ALBEDO_SPECULAR = 1u << 10,
    RFX_INPUT_NORMALS         = 1u << 11,
};

typedef enum RfxQuality { RFX_QUALITY_PERF = 0, RFX_QUALITY_BALANCED, RFX_QUALITY_QUALITY } RfxQuality;

/* -------------------------------------------------------------------------- */
/* Shared Frame Context — the renderer produces these once; every stage reuses */
/* them. ABI-versioned (struct_size/version) so fields append without breaks.  */
/* -------------------------------------------------------------------------- */
typedef struct RfxImageDesc {
    VkImage     image;
    VkImageView view;
    VkFormat    format;
    uint32_t    width, height;
} RfxImageDesc;

typedef struct RfxFrameContext {
    uint32_t     struct_size;   /* = sizeof(RfxFrameContext) — ABI guard */
    uint32_t     version;       /* = RFX_VERSION                          */
    uint32_t     provided_inputs; /* RFX_INPUT_* the app actually filled  */
    uint32_t     _pad;

    RfxImageDesc color, depth, motion, material_id, reactive;
    RfxImageDesc roughness, albedo_diffuse, albedo_specular, normals; /* for RR (reserved) */

    float exposure;
    float jitter[2];
    float reproj[16];           /* prevVP * inverse(currVP), row-major */
    float near_plane, far_plane;
} RfxFrameContext;

/* -------------------------------------------------------------------------- */
/* Capability discovery (design.md §18.1) — an aggregation of each backend's    */
/* own cheap, side-effect-free probe.                                          */
/* -------------------------------------------------------------------------- */
typedef struct RfxBackendCaps {
    RfxBackendId     id;
    RfxStage         stage;
    RfxBackendFamily family;
    uint32_t         supported;        /* 1/0 on this device+build            */
    uint32_t         proprietary;      /* needs a vendor blob/model           */
    uint32_t         quality_tier;     /* rough 0..100 (higher = better)      */
    uint32_t         cost_hint;        /* rough relative GPU cost             */
    uint32_t         required_inputs;  /* RFX_INPUT_* mask                    */
    /* Cross-stage constraint: this backend requires a backend of `requires_family`
     * active in `requires_family_stage`. RFX_STAGE_COUNT = no constraint. */
    RfxStage         requires_family_stage;
    RfxBackendFamily requires_family;
    const char*      name;
    const char*      note;             /* human-readable (e.g. why unsupported) */
} RfxBackendCaps;

typedef struct RfxCapabilities {
    uint32_t       count;
    RfxBackendCaps backends[RFX_BACKEND_COUNT];
} RfxCapabilities;

/* -------------------------------------------------------------------------- */
/* Policy + resolution (design.md §18.1/§18.2). resolve is a PURE function of   */
/* (capabilities, preference) — deterministic, inspectable, headless-testable.  */
/* -------------------------------------------------------------------------- */
typedef struct RfxPreference {
    RfxQuality       quality;
    uint32_t         allow_proprietary;   /* 1 = may use NGX/blobs               */
    RfxBackendFamily vendor_pin;          /* RFX_FAMILY_GENERIC = no pin          */
    uint32_t         stages_enabled;      /* bitmask (1u << RfxStage)             */
    /* Per-stage explicit override; RFX_BACKEND_COUNT = auto-select. Pins a backend. */
    RfxBackendId     override_backend[RFX_STAGE_COUNT];
} RfxPreference;

typedef struct RfxSelection {
    uint32_t     valid;                        /* 1 = a compatible set was found  */
    RfxBackendId backend[RFX_STAGE_COUNT];     /* chosen per stage (NONE = off)   */
    const char*  reason[RFX_STAGE_COUNT];      /* why this backend                */
    uint32_t     required_inputs;              /* union the app must produce      */
} RfxSelection;

/* Resolve a compatible backend *set* across the enabled stages, honouring policy,
 * overrides, and cross-stage family constraints. Never picks per-stage argmax in
 * isolation. Returns RFX_NO_VALID_SELECTION (and valid=0) if nothing satisfies. */
RfxResult rfx_resolve(const RfxCapabilities* caps, const RfxPreference* pref,
                      RfxSelection* out);

/* -------------------------------------------------------------------------- */
/* Context: aggregates backends, drives execution. Composes nvofg for FG.       */
/* -------------------------------------------------------------------------- */
typedef struct RfxContext RfxContext;

typedef struct RfxCreateInfo {
    VkInstance        instance;
    VkPhysicalDevice  physical_device;
    VkDevice          device;
    VkQueue           queue;
    uint32_t          queue_family_index;
    VkQueue           of_queue;              /* optical-flow queue (frame gen)     */
    uint32_t          of_queue_family_index;
    PFN_vkGetInstanceProcAddr gipa;
    uint32_t          width, height;
} RfxCreateInfo;

RfxResult rfx_create(const RfxCreateInfo* info, RfxContext** out);
void      rfx_destroy(RfxContext*);

/* Fill `out` by probing every backend on this device+build (side-effect free). */
RfxResult rfx_query_capabilities(RfxContext*, RfxCapabilities* out);

/* Commit a resolved selection (creates the chosen backends' resources). */
RfxResult rfx_commit(RfxContext*, const RfxSelection*);

/* Frame Generation stage: record + submit via the selected FG backend (nvofg).
 * Returns the timeline point to wait on before presenting the generated frame. */
typedef struct RfxFrameSync { VkSemaphore semaphore; uint64_t value; } RfxFrameSync;

/* `fc->color` is the current final frame; `prev_color` is the previous one (the app
 * ping-pongs two stable targets). Depth/motion/reproj come from `fc`. Registered on
 * the first call (targets must stay stable). Delegates to the selected FG backend. */
RfxResult rfx_record_frame_generation(RfxContext*, const RfxFrameContext* fc,
                                      const RfxImageDesc* prev_color,
                                      const RfxImageDesc* output, float phase,
                                      VkImageLayout color_layout,
                                      const RfxFrameSync* input_ready /*nullable*/,
                                      RfxFrameSync* out_sync);

const char* rfx_backend_name(RfxBackendId);
const char* rfx_stage_name(RfxStage);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RENDERFX_H */
