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

## Alternatives considered
- **Reimplement DLSS RR / DLAA natively** — rejected (design.md §16.1 asymmetry; infeasible
  and legally untenable to reuse NVIDIA weights).
- **Expose RR only when NGX is present (no stage otherwise)** — rejected: the stage +
  capabilities + diagnostics are valuable and testable without NGX, and a permanently-present
  stage keeps the API stable and future backends drop-in.
