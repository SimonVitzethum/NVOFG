# RenderFX

A **Linux-first, Vulkan-first modular rendering-effects framework** that *composes*
focused effect libraries behind **one API** — the architecture argued for in
[`../design.md`](../design.md) §13–§18.

RenderFX is architecturally a **separate project** (it does not modify or absorb nvofg;
it calls nvofg's C ABI). It is kept in-tree here for convenience during bring-up.

## Principles (from design.md §13–§18)

- **Unify the API, never the implementation.** Apps program against *stages* and a shared
  *Frame Context*; which backend runs a stage is a runtime detail they may inspect and
  override — never a vendor API type in the public header.
- **No framework-owned render graph.** RenderFX exposes logical stages the engine's own
  render graph schedules; it records into the app's command buffers (Streamline-style).
- **Capability discovery + policy + inspectable/overridable selection**, aggregated from
  each backend's own cheap probe (nvofg already provides `nvofg_query_support`).
- **Backend dispatch is cross-stage set-resolution**, not per-stage argmax: it honours
  constraints/bundles (DLSS RR needs a DLSS upscaler; FSR-FG pairs with FSR; DLSS-G would
  bundle SR+FG) and reports conflicts.
- **Hardware independence:** vendor names appear only as backend *identifiers*.

## Stages & backends

```
Frame Context → Upscaling → Ray Reconstruction → Frame Generation → (app presents)
```

| Stage | Backends (identifiers) | Functional today |
|---|---|---|
| Upscaling | Native, Temporal, DLSS SR, DLAA, FSR, XeSS | Native (bilinear) |
| Ray Reconstruction | DLSS RR, None | reserved (needs NGX) |
| Frame Generation | **nvofg (OFA)**, **nvofg (shader/Tier B)**, DLSS FG, FSR FG | **nvofg** ✅ |

Only frame generation is functional now (via nvofg). The DLSS/FSR/XeSS entries are
**honest reserved slots** — they need external SDKs/models (design.md §16), and the
capability table already encodes their inputs and cross-stage constraints so they drop in
without an API change. If NVIDIA ever ships native Linux DLSS-G, it becomes another
`FrameGeneration` backend with no interface change.

## Usage sketch

```c
RfxContext* ctx; rfx_create(&info, &ctx);
RfxCapabilities caps; rfx_query_capabilities(ctx, &caps);   // side-effect free

RfxPreference pref = { .quality = RFX_QUALITY_QUALITY, .allow_proprietary = 1 };
pref.stages_enabled = (1u<<RFX_STAGE_FRAME_GENERATION);
for (int s=0;s<RFX_STAGE_COUNT;++s) pref.override_backend[s] = RFX_BACKEND_COUNT; // auto

RfxSelection sel; rfx_resolve(&caps, &pref, &sel);   // pure; inspect sel.backend[...]/reason[...]
rfx_commit(ctx, &sel);                                // creates the chosen backends (nvofg)
// per frame: rfx_record_frame_generation(ctx, &frame_context, &prev, &out, 0.5f, ...);
```

`rfx_resolve` is a **pure function** — see `tests/test_resolve.cpp` (headless) for the
policy, override, and cross-stage-constraint coverage.

## Build

Built as part of the nvofg CMake tree (`add_subdirectory(renderfx)`), linking `nvofg`.
Run the resolver test: `ctest -R renderfx`.
