# RenderFX implementation roadmap

Long-term plan to grow RenderFX into a **Linux-first, Vulkan-first rendering
middleware** with a stable API and interchangeable backends. Worked in **small,
independently verifiable milestones** — each builds, passes tests, stays Vulkan-
validation-clean, and is committed separately. Priority order (design.md §8):
**correctness > image quality > architecture > maintainability > performance.**

## Architectural invariants (never regressed)
Linux-first · Vulkan-first · engine-owned render graph (RenderFX records into it, owns
none) · stable public ABI (versioned; feature bits not booleans) · capability-driven ·
explainable selection · backend-independent API (no vendor type in headers) · modular
plugins · renderer-owned resources · shared Frame Context.

## Environment reality (honest scoping)
This roadmap distinguishes what is **buildable + testable here** from what is **blocked**
on external SDKs, licensed model blobs, other agents, or a running engine. Blocked items
are specified so they drop into the existing stage interface **without API change** — that
is the whole point of the architecture. I will not fabricate integrations I cannot build
and verify.

## Per-technology backend strategy (design.md §7, §16)
For each technology, the objectively strongest approach for a Linux-first Vulkan
middleware, judged on quality/perf/latency/maintainability/Linux-fit/legality/support:

| Stage · Backend | Chosen strategy | Why it is strongest | Status here |
|---|---|---|---|
| Upscaling · **Native** | built-in Vulkan compute (bilinear→Lanczos) | vendor-neutral, zero deps, always available baseline | **buildable now (M1)** |
| Upscaling · **Temporal** | built-in TAAU compute | open, cross-vendor, reuses Frame Context (MV/depth/jitter/reproj) | buildable (M3) |
| Upscaling · **DLSS SR / DLAA** | **official NGX SDK** (Option A) | quality *is* the trained model — only NGX delivers it; official Linux support; legal; low maintenance (§16 asymmetry) | blocked (NGX SDK+blobs) |
| Upscaling · **FSR** | **open FidelityFX** shaders (MIT), ported to Slang | fully open, cross-vendor, no blob; legitimate vendor-neutral quality | buildable later (M5) |
| Upscaling · **XeSS** | **Intel XeSS SDK** (DP4a cross-vendor path) | open-ish, cross-vendor fallback; only real source of its model | blocked (XeSS SDK) |
| RayReconstruction · **DLSS RR** | **official NGX** (Option A) | neural denoiser; model-bound; no native equivalent (§16) | blocked (NGX) |
| RayReconstruction · vendor-neutral | reserve stage; research-scale | honest: a non-DLSS RR is a large ML effort | reserved |
| FrameGeneration · **nvofg** | **hardware-native OFA + shader Tier B** | FG *decomposes* (flow+interp) — the one case native reimpl. works (§16); already best-in-class native Linux FG | **done** |
| FrameGeneration · **learned (CNN)** | Vulkan cooperative-matrix default; **CUDA Tensor Cores** for large models (proven, ADR 0004) | Tensor-Core path proven; needs a trained model | blocked (model) |
| FrameGeneration · **DLSS FG** | NGX DLSS-G *if* native Linux appears | Windows-gated today; drop-in FG backend when available (no API change) | blocked (Windows-gated) |
| FrameGeneration · **FSR FG** | open FSR3 FG shaders, ported | open, cross-vendor | buildable later (M6) |
| all · **CPU reference** | portable CPU implementation | deterministic golden reference for tests (§15.3) | buildable (M2) |

## Milestones (small, verifiable)

**Phase A — make the framework multi-stage-real (no external deps):**
- **M1. Native upscaling backend** — Slang bilinear upscale recorded into the app's
  command buffer; headless GPU test verifies a known up-scale. *Makes the Upscaling stage
  functional; proves the pipeline beyond FG.*
- **M2. CPU reference frame-gen backend** — deterministic CPU warp; golden-image test vs
  the GPU path. *Backend axis + golden testing (§15.3).*
- **M3. Temporal upscaler backend** — TAAU compute using the Frame Context temporal
  inputs. *Second vendor-neutral upscaler; exercises jitter/reproj/MV.*
- **M4. nvofg statistics** — nvofg exposes OFA/compute GPU time via timestamp queries;
  RenderFX surfaces it in `RfxStatistics` (fills `RFX_FEATURE_STATISTICS`).

**Phase B — open vendor-neutral backends (vendored open shaders):**
- **M5. FSR upscaling** (EASU/RCAS) · **M6. FSR3 frame generation** — ported open shaders.

**Phase C — official vendor SDKs (blocked on SDK availability):**
- **M7. NGX upscaling** (DLSS SR/DLAA) · **M8. NGX Ray Reconstruction** ·
  **M9. XeSS upscaling**. Each a backend module behind the existing stage interface;
  capability probes report unsupported when the SDK/driver is absent.

**Phase D — learned FG & runtime (blocked on model / other agent / hardware):**
- **M10. Learned CNN interpolator** (cooperative-matrix + CUDA backend, ADR 0004) — needs
  a trained model. **M11. RMC present-loop** (other agent). **M12. dynamic backend
  selection** from `RfxStatistics`.

## Testing & validation strategy
- **Headless unit tests** for all pure logic (resolver, policy, capabilities) — already
  green (`renderfx_test_resolve`).
- **Headless GPU tests** per functional backend (like `nvofg_headless_interp`): synthetic
  input, known-answer verification, run under **`VK_LAYER_KHRONOS_validation` → 0 VUIDs**.
- **Golden reference** (M2 CPU backend) for deterministic image comparison.
- Every milestone: `cmake --build` clean, `ctest` green, validation clean.

## Commit plan
One milestone → one (or a few) small commits, each self-contained and green, with a
message stating what/why. Prefer many small commits. ADR for any decision with multiple
reasonable approaches (design.md §9).
