# Path B — Functional Phase Plan (native DLSS Frame Generation)

Status after 2026-07-17: **capability phase COMPLETE** — the native NGX host + `nvngx_dlssg.dll`
snippet report FrameGeneration as *fully supported* (`FeatureSupported=0x0`), byte-identical to the
Proton oracle, with no Wine/Proton at runtime. That closed the feasibility gate: the driver supports
FG **and** the native host↔snippet bridge works end to end for the capability query.

This document plans the **functional phase** — the four sub-stages that turn "FG is available" into
"FG actually generates frames" — and weighs it against the §21 own-model fallback.

## The four sub-stages, their distinct challenges, and honest effort

### 1. `Init_ProjectID` — internal state + adapter/model setup
- What it does: builds NGX-internal param/context structs, correlates the adapter, discovers the
  model directory, spins telemetry/worker threads, stores the `VkDevice` for later stages.
- **Current blocker:** SIGSEGV at `_nvngx+0xceee`, inside multi-call helper `0x18000ce40` — a
  versioned struct copy that reads a garbage count (`esi == low32(InGIPA)`) and dereferences a bad
  pointer. `0xce40` is hit many times before the crashing invocation, so this needs *per-invocation
  gdb isolation + back-tracing the value's origin*, not a single boundary diff.
- Also needs: a **real `VkDevice`** created with the FG-required queues/extensions (the harness
  currently makes a minimal device sufficient only for the requirements query).
- Class: same one-value/data fixes as LUID/DRS/module-path, but **several** of them.
- Effort: ~days.

### 2. `CreateFeature(FG)` — model weights + CUDA/Vulkan interop
- Loads the FG **model weights** (the snippet reads model files; under Proton they come from the
  driver's NGX model dir / OTA — availability + path on Linux is an open question to measure).
- Allocates scratch `VkDeviceMemory`; sets up the **CUDA context + Vulkan↔CUDA external-memory /
  external-semaphore interop** (the snippet is a CUDA compute module sharing images with Vulkan).
- Needs a real command buffer + queue.
- Effort: ~1–2 weeks. Risks: model-file availability, interop correctness.

### 3. `EvaluateFeature(FG)` — the actual frame generation  ⚠ highest risk
- Consumes the **NGX Parameter map**: prev/curr color, motion vectors, depth, camera params →
  produces the interpolated/extrapolated frame.
- **The float/SSE-ABI thunk (the pivotal technical risk):** the evaluate + CUDA-interop path passes
  **floats/doubles and by-value structs** across the MS-x64 ↔ SysV boundary. Our current `ms2sysv`
  register-shuffle handles the **INTEGER class only** (RCX/RDX/R8/R9 ↔ RDI/RSI/RDX/RCX) — which is
  exactly why the 48 vulkan/cuda imports so far worked (all handles/pointers/sizes). Evaluate will
  hit XMM args and struct-by-value, where MS-x64 and SysV genuinely differ (XMM0-3 vs XMM0-7 mixed
  with ints; ≤16-byte structs in regs vs hidden-pointer; varargs). Building a general (or
  per-signature) float-aware thunk is bounded but non-trivial (~1 week) and is a **prerequisite** for
  the compute path.
- Runs the NN on CUDA/Tensor cores, writes the generated frame to a `VkImage`.
- Effort: ~2–3 weeks.

### 4. Present / Pacing — `VK_NV_low_latency2` (Reflex)
- FG must **pace** the generated frame between real frames; without it the extra frame gives no
  smoothness and adds latency.
- `VK_NV_low_latency2` is a native Linux extension — this stage does NOT need the PE bridge, it is
  ordinary native Vulkan work, and it overlaps directly with nvofg's own present path.
- Effort: ~1–2 weeks.

**Total: ~4–8 weeks** of reverse-engineering + ABI work — matching the honest estimate from the
start, now measured rather than guessed.

## The float-ABI thunk decides the shape of the whole phase

The single most load-bearing unknown is **stage 3's float-ABI thunk**. Everything up to and including
`CreateFeature` is plausibly INTEGER-class (the imports we've bridged prove the pattern). The moment
the compute path passes floats/structs by value, the current thunk is insufficient and must be
generalized. Because it gates the *actual frame generation*, it should be **spiked first** (before
sinking weeks into Init/CreateFeature polish): a ~1-week standalone spike that drives one
float+struct-by-value MS-x64→SysV call correctly tells us whether Path B's endgame is tractable.

## Path B (finish native DLSS-G) vs §21 (own trained model)

| | Path B functional | §21 own model |
|---|---|---|
| Ships legally today | ❌ needs NVIDIA clearance | ✅ our weights |
| Native from day one | ⚠ PE loader + ABI thunks | ✅ pure native (coopmat) |
| Quality | NVIDIA's trained SOTA | below NVIDIA initially |
| Scales to future DLSS-G | ✅ free (drop in new snippet) | ❌ re-train |
| Remaining effort | ~4–8 wks RE + float-ABI | training compute (5080, ~100–200 h) + ML work |
| Biggest risk | float-ABI + model files | model quality + train time |

Both remain alive. They are not mutually exclusive: Path B proves what "good" looks like and can seed
the own model (targets, ablations); §21 is the shippable product.

## Recommendation — keep both, gated on one cheap spike

1. **Spike the float-ABI thunk (≈1 wk, bounded).** It is the one unknown that decides whether Path B's
   endgame works. If it drives a float+struct-by-value call cleanly → Path B is the higher-leverage
   choice (NVIDIA quality, scales for free) and worth the RE tail.
2. **Continue the Init grind in parallel** — it is a prerequisite for CreateFeature/Evaluate no matter
   what, and each fix is the same measured one-value shape. Next concrete step: isolate the crashing
   `0xce40` invocation (break at the `Init+0xd75b` call, single-step into the specific call whose
   `esi/r10` are garbage, back-trace the source field), fix, rerun.
3. **When the 5080 server lands, start §21 training** as the shippable fallback (Path B needs NVIDIA
   clearance to ship regardless), using Path B's output as the quality target.

Decision point: after the float-ABI spike + Init reaching `CreateFeature`, re-evaluate whether to
push Path B to a running FG or pivot effort to §21 for the shippable path.
