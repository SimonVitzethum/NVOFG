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

/* Capability versioning (evolve without API breaks): the app checks these to know how
 * to interpret the capability schema and each backend's revision. New backend
 * generations (DLSS SR v2/v3, RR revisions, FSR/XeSS gens) bump backend_version and add
 * feature bits — never new boolean fields sprinkled through the API. */
#define RFX_API_VERSION                1
#define RFX_CAPABILITY_SCHEMA_VERSION  1

typedef enum RfxResult {
    RFX_OK = 0,
    RFX_UNSUPPORTED,
    RFX_INVALID_ARGUMENT,
    RFX_NO_VALID_SELECTION,   /* no backend combination satisfies the constraints */
    RFX_NOT_REGISTERED,
    RFX_INTERNAL,
} RfxResult;

/* Logical pipeline stages (design.md §15.2). Present is the app's own concern.
 * Values are STABLE across releases (never reordered) — safe to serialize. */
typedef enum RfxStage {
    RFX_STAGE_UPSCALING          = 0,
    RFX_STAGE_RAY_RECONSTRUCTION = 1,
    RFX_STAGE_FRAME_GENERATION   = 2,
    RFX_STAGE_COUNT              = 3,
} RfxStage;

/* Backend identifiers. Vendor names are identifiers only — the *interface* is uniform.
 * **Numeric values are a STABLE public ABI**: never renumbered or reordered, only
 * appended. Applications may serialize them (configs/presets/project files); prefer the
 * string form (rfx_backend_name / rfx_backend_from_name) for human-readable persistence.
 * IDs never depend on registration order. */
typedef enum RfxBackendId {
    RFX_BACKEND_NONE      = 0,   /* stage disabled                                   */
    RFX_BACKEND_NATIVE    = 1,   /* built-in pass-through / bilinear (no vendor tech) */
    RFX_BACKEND_TEMPORAL  = 2,   /* built-in temporal upscaler/AA                     */
    RFX_BACKEND_DLSS_SR   = 3,   /* NGX DLSS Super Resolution                         */
    RFX_BACKEND_DLAA      = 4,   /* NGX DLAA                                          */
    RFX_BACKEND_DLSS_RR   = 5,   /* NGX Ray Reconstruction                            */
    RFX_BACKEND_DLSS_FG   = 6,   /* NGX DLSS Frame Generation (Windows-gated today)   */
    RFX_BACKEND_FSR       = 7,   /* AMD FidelityFX Super Resolution                   */
    RFX_BACKEND_FSR_FG    = 8,   /* FSR Frame Generation                              */
    RFX_BACKEND_XESS      = 9,   /* Intel XeSS                                        */
    RFX_BACKEND_NVOFG     = 10,  /* native OFA frame generation (this project)        */
    RFX_BACKEND_SHADER_FG = 11,  /* portable shader frame generation (nvofg Tier B)   */
    RFX_BACKEND_COUNT     = 12,  /* keep last; only ever grows                        */
} RfxBackendId;

/* Vendor/algorithm families — used for cross-stage constraints (design.md §18.2). */
typedef enum RfxBackendFamily {
    RFX_FAMILY_GENERIC = 0,
    RFX_FAMILY_DLSS    = 1,
    RFX_FAMILY_FSR     = 2,
    RFX_FAMILY_XESS    = 3,
    RFX_FAMILY_NVOFG   = 4,
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

/* Backend feature bits (design.md §18) — a query-oriented alternative to a boolean
 * explosion. New capabilities become new bits; the ABI does not change. A 64-bit mask
 * is finite; growth beyond it relies on RFX_CAPABILITY_SCHEMA_VERSION + struct
 * extensibility. Query per backend (RfxBackendCaps.features) or per stage
 * (rfx_query_stage_features). */
enum {
    RFX_FEATURE_HDR            = 1ull << 0,  /* HDR / scRGB color path                */
    RFX_FEATURE_ASYNC          = 1ull << 1,  /* runs on a separate/async queue        */
    RFX_FEATURE_TEMPORAL_RESET = 1ull << 2,  /* honours a per-frame history reset     */
    RFX_FEATURE_REACTIVE_MASK  = 1ull << 3,  /* consumes a reactive mask              */
    RFX_FEATURE_MATERIAL_IDS   = 1ull << 4,  /* consumes material/object ids          */
    RFX_FEATURE_GBUFFER        = 1ull << 5,  /* consumes a GBuffer (albedo/normals/…) */
    RFX_FEATURE_TENSOR_BACKEND = 1ull << 6,  /* uses Tensor Cores (cooperative/CUDA)  */
    RFX_FEATURE_AUTO_EXPOSURE  = 1ull << 7,  /* handles exposure internally           */
    RFX_FEATURE_STATISTICS     = 1ull << 8,  /* populates RfxStatistics               */
    RFX_FEATURE_DYNAMIC_RES    = 1ull << 9,  /* supports dynamic resolution           */
    /* bits 10..63 reserved for future features */
};

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

    /* Camera matrices — required by DLSS Ray Reconstruction (NGX consumes them to
     * reproject the denoised GBuffer). Row-major. ABI-appended (guard with struct_size);
     * leave zero for stages that don't need them (FG / SR / DLAA use motion+jitter). */
    float world_to_view[16];
    float view_to_clip[16];
    float mv_scale[2];          /* motion-vector -> pixel-space scale (0 => {1,1}) */
} RfxFrameContext;

/* -------------------------------------------------------------------------- */
/* Capability discovery (design.md §18.1) — an aggregation of each backend's    */
/* own cheap, side-effect-free probe.                                          */
/* -------------------------------------------------------------------------- */
typedef struct RfxBackendCaps {
    RfxBackendId     id;
    RfxStage         stage;
    RfxBackendFamily family;
    uint32_t         backend_version;  /* revision of this backend (SR v2/v3, …) */
    uint64_t         features;         /* RFX_FEATURE_* bitmask                 */
    uint32_t         supported;        /* 1/0 on this device+build            */
    uint32_t         proprietary;      /* needs a vendor blob/model           */
    uint32_t         deterministic;    /* reproducible output (for debug/golden) */
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
    uint32_t       api_version;               /* = RFX_API_VERSION                */
    uint32_t       capability_schema_version; /* = RFX_CAPABILITY_SCHEMA_VERSION  */
    uint32_t       count;
    uint32_t       _pad;
    RfxBackendCaps backends[RFX_BACKEND_COUNT];
} RfxCapabilities;

/* Union of features across the supported backends of a stage (design.md §18 —
 * feature queries, not a boolean explosion). Pure. */
uint64_t rfx_stage_features(const RfxCapabilities* caps, RfxStage stage);

/* -------------------------------------------------------------------------- */
/* The four explicit concepts (design.md §18):                                 */
/*   Capabilities  (what the hardware/build can do)   -> RfxCapabilities        */
/*   Preferences   (the app's INTENT, never a pipeline) -> RfxPreference        */
/*   Policy        (the effective rules derived from intent) -> RfxPolicy        */
/*   Selection     (the resolved, explained pipeline) -> RfxSelection           */
/* The app expresses intent; RenderFX owns backend selection.                   */
/* -------------------------------------------------------------------------- */

/* Application INTENT. The app states goals; it never constructs a pipeline. */
typedef struct RfxPreference {
    RfxQuality       quality;             /* quality target                       */
    uint32_t         prioritize_latency;  /* 1 = favour low latency over quality  */
    uint32_t         allow_proprietary;   /* 1 = may use vendor blobs/models      */
    uint32_t         open_source_only;    /* 1 = only open/vendor-neutral backends*/
    uint32_t         deterministic;       /* 1 = only reproducible backends       */
    uint32_t         power_saver;         /* 1 = minimise GPU cost (battery)      */
    uint32_t         debug;               /* 1 = prefer reference/native, verbose */
    RfxBackendFamily vendor_pin;          /* RFX_FAMILY_GENERIC = no vendor pin   */
    uint32_t         stages_enabled;      /* bitmask (1u << RfxStage)             */
    RfxBackendId     override_backend[RFX_STAGE_COUNT]; /* RFX_BACKEND_COUNT = auto */
} RfxPreference;

/* Effective POLICY derived from the preferences — the concrete rules the resolver
 * applies. Exposed so the app/debugger can inspect *why* selection behaved as it did. */
typedef struct RfxPolicy {
    int              quality_weight;      /* scoring weight for quality_tier      */
    int              cost_weight;         /* scoring penalty for cost_hint        */
    uint32_t         exclude_proprietary; /* derived: open_source_only||!allow    */
    uint32_t         require_deterministic;
    RfxBackendFamily vendor_pin;
    uint32_t         prefer_native;       /* debug: bias toward native/reference  */
} RfxPolicy;

/* Derive the effective policy from intent (pure, inspectable). */
RfxResult rfx_derive_policy(const RfxPreference* pref, RfxPolicy* out);

/* Per-stage explanation: the chosen backend and the reasons for it (design.md §18 —
 * every selection is explainable). */
#define RFX_MAX_REASONS 6
typedef enum RfxReason {
    RFX_REASON_SUPPORTED = 0,          /* backend is available on this device+build */
    RFX_REASON_HIGHEST_QUALITY,        /* best quality among the valid candidates    */
    RFX_REASON_LOWEST_COST,            /* cheapest among candidates (perf/battery)   */
    RFX_REASON_PROPRIETARY_ALLOWED,    /* uses a vendor model; policy permitted it   */
    RFX_REASON_OPEN_SOURCE,            /* vendor-neutral / no proprietary blob       */
    RFX_REASON_VENDOR_PINNED,          /* matched the requested vendor pin           */
    RFX_REASON_PREFERRED_NATIVE,       /* chosen for debug/deterministic preference  */
    RFX_REASON_DETERMINISTIC,          /* reproducible output (policy required it)   */
    RFX_REASON_CONSTRAINT_SATISFIED,   /* compatible with another stage's backend    */
    RFX_REASON_ALTERNATIVE_UNAVAILABLE,/* a higher option was excluded/unsupported   */
    RFX_REASON_OVERRIDDEN,             /* pinned by the app                          */
    RFX_REASON_DISABLED_NO_COMPATIBLE, /* stage on, but nothing compatible -> None   */
    RFX_REASON_STAGE_NOT_REQUESTED,    /* stage not enabled                          */
} RfxReason;

typedef struct RfxStageExplanation {
    RfxBackendId backend;
    uint32_t     reason_count;
    RfxReason    reasons[RFX_MAX_REASONS];
} RfxStageExplanation;

typedef struct RfxSelection {
    uint32_t            valid;                        /* 1 = a compatible set found     */
    RfxBackendId        backend[RFX_STAGE_COUNT];     /* the resolved pipeline (by stage) */
    RfxStageExplanation explanation[RFX_STAGE_COUNT]; /* why each stage chose its backend */
    uint32_t            required_inputs;              /* union the app must produce     */
} RfxSelection;

/* Resolve a compatible backend *set* across the enabled stages: derives policy from
 * intent, honours overrides + cross-stage family constraints, and never picks per-stage
 * argmax. Fills the explanation for each stage. Returns RFX_NO_VALID_SELECTION (valid=0)
 * if nothing satisfies. Pure function of (capabilities, preference). */
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

/* Upscaling stage: record the *committed* upscaler into the app's command buffer
 * (RenderFX owns no graph). The source is `fc->color` (render resolution); `dst` is the
 * present-resolution target; both must be in GENERAL layout. Dispatches to the backend
 * chosen by rfx_commit — Native (bilinear) and Temporal (TAAU, uses fc->motion) are
 * functional today. Temporal keeps a RenderFX-owned history; set `fc` reset semantics
 * via `reset` on the first frame / camera cut through GenerateInfo-style flags is not
 * needed here — pass reset=1 for the first frame. */
RfxResult rfx_record_upscaling(RfxContext*, VkCommandBuffer cmd,
                               const RfxFrameContext* fc, const RfxImageDesc* dst,
                               uint32_t reset);

/* Ray Reconstruction stage: record the *committed* RR backend into the app's command
 * buffer. Consumes the shared Frame Context (color/motion/depth/roughness/diffuse+
 * specular albedo/normals/exposure/jitter/reproj). The only backend today is DLSS RR
 * (NGX). When NGX / the trained model is unavailable it returns RFX_UNSUPPORTED and the
 * rest of the pipeline is unaffected — never a special case elsewhere in the framework. */
RfxResult rfx_record_ray_reconstruction(RfxContext*, VkCommandBuffer cmd,
                                        const RfxFrameContext* fc, const RfxImageDesc* output);

/* Inputs a backend REQUIRES, as an RFX_INPUT_* bitmask (from the capability table). */
uint32_t rfx_backend_required_inputs(RfxBackendId backend);

/* Diagnostics: which inputs the committed backend for `stage` requires but the Frame
 * Context does not provide (RFX_INPUT_* bitmask; 0 = all satisfied). Invaluable for
 * validating RR's GBuffer inputs. */
uint32_t rfx_missing_inputs(RfxContext*, RfxStage stage, const RfxFrameContext* fc);

/* RR/GBuffer debug visualisation (design: validate RR inputs). Renders one Frame
 * Context channel into `output` (a vendor-neutral compute pass into the app's cmd
 * buffer). If the requested channel is absent from the Frame Context, the output is
 * flagged magenta — an immediate missing-input diagnostic. */
typedef enum RfxDebugView {
    RFX_DEBUG_NONE = 0,
    RFX_DEBUG_NORMALS,
    RFX_DEBUG_ROUGHNESS,
    RFX_DEBUG_DIFFUSE_ALBEDO,
    RFX_DEBUG_SPECULAR_ALBEDO,
    RFX_DEBUG_DEPTH,
    RFX_DEBUG_MOTION,
} RfxDebugView;

RfxResult rfx_record_debug_view(RfxContext*, VkCommandBuffer cmd, const RfxFrameContext* fc,
                                const RfxImageDesc* output, RfxDebugView view);

/* Union of features across the supported backends of the given stage on this context. */
uint64_t rfx_query_stage_features(RfxContext*, RfxStage);

/* -------------------------------------------------------------------------- */
/* Optional runtime statistics (design.md §18) — a lightweight container the    */
/* active backends *populate*, NOT a profiler. RenderFX owns no command buffers  */
/* (engine-owned graph), so GPU timings are provided by backends that expose     */
/* timestamps (RFX_FEATURE_STATISTICS); unmeasured fields are 0 (unknown).       */
/* -------------------------------------------------------------------------- */
typedef struct RfxStatistics {
    uint32_t schema_version;    /* = RFX_CAPABILITY_SCHEMA_VERSION */
    uint32_t _pad;
    uint64_t frames_generated;  /* interpolated frames produced (real counter) */
    /* Per-stage GPU time, microseconds; 0.0 = not measured by the active backend. */
    double   total_gpu_us;
    double   frame_gen_us;
    double   optical_flow_us;
    double   tensor_us;
    double   ray_recon_us;
    double   upscaling_us;
    double   backend_latency_us;
} RfxStatistics;

RfxResult rfx_get_statistics(RfxContext*, RfxStatistics* out);

const char* rfx_backend_name(RfxBackendId);   /* stable string id (for serialization) */
const char* rfx_stage_name(RfxStage);
const char* rfx_reason_text(RfxReason);       /* human-readable reason                 */

/* Stable string <-> id for serializing configs/presets/project files. Returns
 * RFX_BACKEND_NONE for an unknown name (check with a round-trip if needed). */
RfxBackendId rfx_backend_from_name(const char* name);

/* Pipeline inspection (design.md §18) — format the resolved pipeline + per-stage
 * reasons into `buf` (NUL-terminated, truncated to `cap`). Returns bytes written
 * (excluding NUL). Also available programmatically via RfxSelection.explanation[]. */
uint32_t rfx_format_pipeline(const RfxSelection* sel, char* buf, uint32_t cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RENDERFX_H */
