# ADR 0006 — NGX-gated Ray Reconstruction & DLAA backends (RenderFX)

- **Status:** Accepted (architecture + diagnostics functional; NGX execution gated)
- **Date:** 2026-07-16
- **Relates to:** design.md §16 (DLSS strategy), §18 (stages/dispatch); RenderFX ROADMAP
  phase C.

## Context

The RenderFX Ray Reconstruction (RR) stage and the DLAA upscaling backend both derive
their value from **NVIDIA's trained models**, delivered via **NGX**. Per design.md §16's
hardware-block-vs-model asymmetry, there is no native reimplementation path: unlike frame
generation (OFA + warp), RR/DLAA cannot be recreated from Tensor Cores alone. So the only
correct strategy is **use official NGX where available; otherwise report unsupported**.

The NGX SDK and the DLSS-RR / DLAA model blobs are **not present in this project's build
environment**. This ADR records how RR and DLAA are integrated so they are *architecturally
complete and fully wired into the framework* today, and become *functional* by enabling one
build flag once the NGX SDK is present — with **no public API change**.

## Decision

1. **RR is a first-class, backend-driven stage**, identical in shape to Upscaling and
   Frame Generation: `rfx_commit` records the RR backend, `rfx_record_ray_reconstruction`
   dispatches on it. No special-casing anywhere else in the framework.
2. **DLAA is another backend of the Upscaling stage** (already in the capability table),
   selected by the existing policy/resolver, with detailed capabilities (features +
   required inputs). No new API surface.
3. **NGX execution is gated behind `-DRENDERFX_NGX`** (off by default; no SDK here). When
   off, `DLSS_RR` and `DLAA` report `supported = 0` in capability discovery, the resolver
   never selects them, and the dispatch site is a safety-net returning `RFX_UNSUPPORTED`.
   When on (SDK + runtime + model present), the same dispatch site calls NGX. The FFI
   integration mirrors RMC's `crate::ngx` real/stub pattern.
4. **The value that *is* buildable without NGX is delivered now**, vendor-neutrally:
   - **Detailed capabilities** — RR/DLAA declare their `features`
     (HDR/GBuffer/material-ids/tensor/temporal-reset) and `required_inputs`
     (roughness, diffuse+specular albedo, normals, …) in the capability table.
   - **Missing-input diagnostics** — `rfx_missing_inputs` / `rfx_backend_required_inputs`
     report exactly which shared-Frame-Context channels a backend needs but the app did
     not provide. This is the single most useful validation aid for RR's GBuffer inputs.
   - **GBuffer/normals/roughness/albedo debug visualisation** — a vendor-neutral compute
     pass (`rfx_record_debug_view`) that renders Frame Context channels, for validating RR
     inputs on any GPU.

## Consequences

### Positive
- RR completes the stage architecture: every major stage (Upscaling, RR, FG) is now
  backend-driven and dispatch-uniform — validating the design before adding more vendors.
- Honest: no reverse-engineering, no fabricated model. NGX is the only RR/DLAA path and it
  is used where available; otherwise `unsupported`, and the pipeline is unaffected.
- Future RR implementations (should a vendor-neutral one ever exist) drop into the same
  stage/interface with no API change — the stated goal.
- The genuinely useful, buildable parts (capabilities, diagnostics, visualisation) ship now
  and make future NGX bring-up far easier to validate.

### Negative / open
- RR/DLAA are **non-functional in this environment** (no NGX SDK/model) — by necessity, and
  exactly as specified ("report unsupported"). Turning them on is a build-flag + SDK step,
  not an architecture change.
- The NGX FFI itself (the `-DRENDERFX_NGX` real path) is a stub until the SDK is vendored;
  the call sites and capability wiring are in place.

## NGX bring-up finding (this environment)

The NGX SDK (`nvsdk_ngx` headers + `libnvsdk_ngx.a` + the `dlss`/`dlssd` feature blobs)
was vendored under `renderfx/vendor/ngx/` (local only; NVIDIA-licensed, git-ignored) and
the build path wired behind `-DRENDERFX_NGX`. A diagnostic probe (`src/ngx_probe.cpp`,
mirroring RMC's proven NGX FFI) **builds and links cleanly**, and at runtime NGX:

- **loads the DLSS Super Resolution + Ray Reconstruction snippets** and reports the RTX
  5070 as **Blackwell, supported** (the `nvngx_dlssd` log shows "Setting DLSS Base Cubins
  for standalone RR"); the model blobs are architecture-compatible.
- but `NVSDK_NGX_VULKAN_Init` returns **`0xBAD00005`** because NGX **requires a writable
  models directory and defaults to `/usr/share/nvidia/ngx`**, which it cannot create
  (permission denied — needs root). The `nvidia-ngx-conf.json` `ngx_models_path` override,
  `__NGX_DISABLE_UPDATER`, and `NGX_CUBIN_DISABLE_RESOURCE_CACHE` did not redirect it on
  this driver (610.43.03).

### Resolution (design.md §19) — NGX now initialises

The blocker was fixed per **design.md §19**. Two things were required together:
1. **A writable application-data path**, passed in BOTH the legacy `InApplicationDataPath`
   arg AND `NVSDK_NGX_FeatureCommonInfo.PathListInfo` — so NGX writes its cubin/model cache
   and logs there instead of the root-owned `/usr/share/nvidia/ngx`.
2. **A valid UUID project id** for `Init_with_ProjectID` — a non-UUID string is itself a
   `FAIL_InvalidParameter` (`0xBAD00005`) cause (§19: "don't conflate the two").

With both applied, the probe (`src/ngx_probe.cpp`) on the RTX 5070 / driver 610 prints:

```
NGX init result: 0x00000001 (ok)
  DLSS SR / DLAA available : 1
  DLSS Ray Reconstruction  : 1
```

**So DLSS Super Resolution, DLAA, and Ray Reconstruction are available on this machine.**
The NGX-gated backends are therefore genuinely functional here (build with
`-DRENDERFX_NGX`): `rfx_query_capabilities` marks them `supported`, and the existing
resolver/dispatch selects them. Where NGX cannot init (non-NVIDIA, no SDK, unwritable
path), they report `unsupported` and the pipeline falls back — the graceful-degradation
path (G6). The remaining work to make them *do* the denoise/upscale is the per-frame
`CreateFeature`/`EvaluateFeature` wiring consuming the shared Frame Context.

## Alternatives considered
- **Reimplement DLSS RR / DLAA natively** — rejected (design.md §16.1 asymmetry; infeasible
  and legally untenable to reuse NVIDIA weights).
- **Expose RR only when NGX is present (no stage otherwise)** — rejected: the stage +
  capabilities + diagnostics are valuable and testable without NGX, and a permanently-present
  stage keeps the API stable and future backends drop-in.
