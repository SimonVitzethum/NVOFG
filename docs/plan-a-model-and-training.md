# Plan A — model design & training on the 5080 (`ki-pc-fisch-101`)

Concrete execution of design.md §21 now that a training GPU (RTX 5080, Blackwell, 16 GB) is
available over SSH. Companion to `docs/plan-a-native-framegen.md` (the roadmap) and design.md §21
(the design analysis).

## Parameter budget — the decision

**Inference latency on the deployment GPU (Blackwell, <2–3 ms @1080p), not training, is the binding
constraint.** The 5080 trains tens of millions of params in days; the question is what runs per-frame
in the FG budget. And cost is FLOPs/latency at resolution, not param count (STSS 0.4M → 1.8 ms vs
ExtraNet 1.1M → 17 ms, both TensorRT@1080p/3090 — architecture dominates).

| Params | Regime | @1080p | Real-time FG? |
|---|---|---|---|
| 0.3–1M | STSS/ExtraNet-lite (we supply flow/MV/depth) | <1–2 ms | ✅ sweet spot |
| 1–3M | efficient, partial reduced-res | ~2–3 ms | ✅ (measure on 5080) |
| 3–8M | reduced-res + heavy Tensor-core only | ~3–6 ms | ⚠️ 1080p/high-end only |
| 8–40M | pure-RGB VFI regime | 10–400 ms | ❌ offline only |

**Decision:**
- **v1 real-time (primary, extrapolation): target ~1M params (0.8–1.5M).** Measure headroom to ~3–4M
  on the actual 5080/5070 before committing more. Never >8M for the real-time path.
- **Optional offline/quality tier (interpolation, cutscenes / quality-ceiling reference): 5–15M.**
  Latency-insensitive; also a useful training data-point and upper bound. Not a product real-time path.

Rationale: because we *supply* motion (OFA flow + engine MV + depth), the net only does synthesis +
disocclusion inpainting + shading correction — which saturates near ~1M (§21.1: ~80–95% of DLSS-G at
that scale). Larger buys a few % at 10–50× the latency.

## Architecture (v1, ~1M params — §21.4)

Rendering-aware forward-warp + gated-conv fusion (SoftSplat / ExtraNet / ExtraSS lineage):
1. **Softmax-splatting warp (no learned flow):** forward-warp prev frame(s) to the target phase using
   the *given* OFA flow / engine MVs; resolve many-to-one collisions with **depth** as splat weight.
   Differentiable; contributes ~0 params (this is the whole size lever).
2. **Fusion net:** lightweight encoder–decoder with **gated convolutions** (~11 enc / 7 dec, ExtraNet
   sizing) + a weight-shared **history encoder** (moving shadows/reflections MVs miss); skip
   connections; **residual add of the warped RGB** (stabilises training, and = the identity model at
   init — matches the A1 scaffold's identity default). Output: corrected color + a blend/hole mask.
3. **Inputs:** warped color(s), fwd+bwd flow, occlusion/confidence (from nvofg's REFINE stage), depth,
   engine MV, the OFA-vs-MV disagreement map, a disocclusion mask, and the UI/reactive masks
   (HUD/particles never synthesized from geometry motion).
4. **Two modes, shared backbone:** **extrapolation primary** (past frames only → no added latency —
   §21.2); **interpolation fallback** (bidirectional OFA flow, cleaner disocclusion) for
   latency-insensitive/offline use.
5. **Precision:** train AMP fp16; deploy fp16 over `VK_KHR_cooperative_matrix` (Tensor cores), the
   path proven by `src/spike/cuda_tensor.cu` / `cuda_vk_interop.cu`; TensorRT optional above threshold.

## Training plan on `ki-pc-fisch-101` (RTX 5080)

**Note — Blackwell toolchain:** the 5080 is `sm_120`; needs a recent CUDA (≥12.6) + a PyTorch build
with Blackwell kernels (recent stable or nightly cu126+). First step is verifying `torch.cuda` runs a
matmul on the 5080 before anything else.

- **T0. Env + smoke test.** PyTorch (Blackwell-capable) + CUDA; confirm fp16 Tensor-core matmul runs
  on the 5080; pin versions in a lockfile.
- **T1. Data pipeline (A2).** (a) **Rendered GT (primary):** capture harness dumps color+MV+depth+
  UI/reactive at **2× target fps** so every other frame is GT (coordinate format with the RMC agent);
  tens of thousands of triplets across varied motion. (b) **Vimeo-90K/X4K bootstrap** for RGB
  synthesis pretraining (no aux — warm-start only). Dataloaders + augmentation (540p–1080p crops).
- **T2. Model impl.** SoftSplat warp + gated-conv fusion net (~1M), both modes; verify the
  identity-init (residual add, zeroed final conv) reproduces the classical warp exactly.
- **T3. Losses (§21.6).** Charbonnier/L1 + LPIPS/VGG + census/warping + light GAN (micro-detail) +
  temporal-stability; **UI/reactive masked out of all losses**; disocclusion regions up-weighted.
- **T4. Schedule.** Pretrain on Vimeo (synthesis) → fine-tune on rendered data with real MV/depth/
  masks. AMP fp16, largest batch the 16 GB fits (1080p crops). ~**days** for a competitive v1 (§21.7).
- **T5. Eval (A4 harness).** PSNR/SSIM/**LPIPS**/VMAF + temporal-stability vs GT, and vs the (private)
  Path B reference on a fixed clip set. Gate every checkpoint on the harness.
- **T6. Export (A5).** Trained weights → fp16 coopmat/WMMA tile layout (versioned header) that the
  `recordCnnRefine` backend loads. Then the A1 identity is replaced by the real model — no ABi change.
- **T7. Ablate to the param ceiling.** Sweep ~0.5M → ~4M on the 5080; measure quality (T5) vs
  measured 1080p/1440p/4K latency on a Blackwell deployment card; pick the v1 point.

## Environment status (`ki-pc-fisch-101`) — T0 DONE ✅

Verified 2026-07-17: RTX 5080 (GB203, **sm_120**, 16.6 GB) passed through to an Ubuntu 24.04 VM;
NVIDIA **open kernel driver 595.71.05** loaded (`/dev/nvidia*`, `libcuda.so.1` present). Python 3.12
venv at `~/plana` (pip bootstrapped via get-pip; system pip absent). **PyTorch 2.11.0+cu128**
installed; smoke test passes: `torch.cuda.is_available()=True`, device `RTX 5080`, fp16 Tensor-core
matmul ~**118 TFLOPS** (8192³ in 9.3 ms). Confirms the inference-budget headroom (~3–4× a 3090 → a
~1M net ≈ 1–1.5 ms @1080p, room to ~3–4M). Resource envelope for this box: GPU free, RAM < 80 GB,
CPU ≤ 20% (~3 cores) — so **dataloader `num_workers` ≤ 3**, GPU/AMP unconstrained.

## Immediate next steps

1. **T0 on `ki-pc-fisch-101`** — bring up a Blackwell-capable PyTorch and prove a Tensor-core matmul.
2. **A4 harness** (from the roadmap) in parallel — nothing is claimed without it.
3. **T2 model** with identity-init verified against the classical warp (ties to the A1 scaffold).
4. **T1 data** — Vimeo bootstrap immediately; rendered capture as the RMC format lands.

## Definition of "ready"

- **Trainable:** T0–T2 done, identity-init reproduces the warp, harness live.
- **v1 model:** ~1M extrapolation net, <2–3 ms @1080p on Blackwell, **visibly better than the
  classical warp** on the A4 harness. Ships with no NVIDIA code.
