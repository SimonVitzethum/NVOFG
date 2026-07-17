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
4. **Two modes, shared backbone — a latency/quality trade-off axis, not "primary = better":**
   **extrapolation** (past frames only → *no added latency*, the low-latency primary) is the
   **harder** learning task — it must guess newly-revealed disocclusion, direction changes,
   acceleration, fade-ins, for which no signal exists in past-only inputs. **Interpolation**
   (bidirectional OFA flow → both surrounding frames) is *easier and higher-quality* — the OFA bidir
   flow nearly solves it — but incurs the hold-back latency. We ship extrapolation primary **despite**
   its lower quality, to buy latency; both are trained and measured (A4, per-regime) from the start so
   the quality gap is known before the server run, not after.
5. **Precision:** train AMP fp16; deploy fp16 over `VK_KHR_cooperative_matrix` (Tensor cores), the
   path proven by `src/spike/cuda_tensor.cu` / `cuda_vk_interop.cu`; TensorRT optional above threshold.

## Frame-generation multiplier — target 6× from 60–100 fps, seamless 2×–6×

**Requirement:** the model must generate up to **5 intermediate frames (6×)** cleanly from a **60–100
fps base** (→ 360–600 fps for 240/360 Hz displays), and handle **any 2×–6×** with no problems.

Why this is the *right* target rather than 6× from a low base: intermediate-frame quality is bounded
by **motion per interval**, not by the multiplier itself. At a 60–100 fps base the inter-frame gap is
10–17 ms, so even the 5th intermediate spans little motion → the flow stays near-linear, disocclusion
gaps are small, and error-from-endpoint stays low. The same 6× from 20 fps (50 ms gap) would show
artifacts and a 50 ms hold-back; from 60 fps the hold-back is ~16 ms. (Industry tops out at DLSS 4
MFG = 3 generated / 4×; 6× is only sane in this high-base regime.)

**How one net serves 2×–6× with no retraining — phase-conditioned training (the key design choice):**
- The net takes the target **phase `t∈(0,1)`** as a conditioning input (already in `nvofg.h`:
  `NvofgGenerateInfo.phase` + `nvofg_record_warp(phase)` which reuses the computed flow and only
  re-warps at a new phase — cheap, no OFA re-run).
- **Train with random `t` per sample** (not fixed 0.5), drawn across (0,1), so the net learns
  continuous phase interpolation → at inference any N maps to phases `k/(N+1)` for k=1..N. 2× uses
  t=0.5; 6× uses t∈{1/6,…,5/6}; nothing about the weights changes.
- **Cost of N intermediates:** flow + fusion inputs once per real-frame pair, then the synthesis runs
  N times over the interval. At the 60–100 fps base each interval is long enough that 5× the ~1–1.5 ms
  synthesis fits comfortably; compute is not the limiter, motion-per-interval is.
- **Quality target by regime:** train/validate 2×–4× as the primary quality band (matches DLSS 4 MFG);
  6× is validated specifically on the **high-base (≥60 fps) / small-motion** clips where it holds.
  A4 reports quality vs multiplier so the "clean up to 6× from 60 fps" claim is measured, not assumed.
- **Extrapolation + high multiplier:** extrapolating 5 frames ahead is too speculative; the
  extrapolation mode targets **1–2 future frames**. High multipliers (up to 6×) are an
  **interpolation-mode** capability (the "between two frames" case) — matching the user's framing.

## Build status — Track 1 built + validated on the 5080 (before the real run)

All plumbing is green *before* the expensive run (identity-first discipline). In `training/`:
- **Harness** (`train.py`/`ctl.sh`): resumable, tmux-persistent (survives SSH loss), PAUSE/resume.
- **T2 model** (`model.py`): SoftSplat + gated-conv fusion, **1.023M**, phase-conditioned (2×–6×),
  interp+extrap, **bit-exact identity** (out == warp).
- **T1 data** (`data.py`): A2 capture format + `TripletDataset` (2×–6× real-GT phases) + a richer
  synthetic generator (disocclusion + shading) for pre-real-data validation.
- **Alignment gate** (`align.py`): sub-pixel offset test wired into startup — **aborts** on a
  misaligned capture (proven: aligned PASS, 0.4 px offset caught).
- **T3 losses** (`losses.py`): Charbonnier + LPIPS + census, UI/reactive-masked, disocclusion-weighted;
  optional temporal-stability.
- **A4 harness** (`eval.py`): PSNR/SSIM/LPIPS **per regime** (easy/shading/disocc), model-vs-warp.
- **A5 export** (`export.py`): model → versioned fp16 `.nvfgw` for the `recordCnnRefine` backend;
  round-trip bit-identical.

**Validated on the 5080 — a trained reference model, held-out A4:** a full reference run (interp, 40
synth clips, ~50k steps, GPU 95% / Tensor Cores) beats the classical warp on **held-out** clips
(different seeds) by a wide margin — `easy` PSNR **21.6 → 29.8 (+8.2)** / LPIPS 0.112 → 0.008,
`shading` PSNR **19.2 → 23.2 (+4.1)** / LPIPS 0.146 → 0.027. Exported to `.nvfgw` (1.023M).
(The 2500-step PoC already showed +2.3 / +1.6; the trained model widens it.) The whole approach is
validated end to end; the real *product* run is blocked only on the RMC rendered captures in the A2
format (+ a matching flow source).

**B4 inference backend proven:** the exported `.nvfgw` runs correctly in a standalone C++ **CPU
reference** (max_err 0.006 vs PyTorch) and a **CUDA backend on Tensor Cores** (im2col + cuBLAS fp16
GEMM / fp32 accum, nvcc sm_120, max_err 0.009, 1.6 ms/forward @32² after an async-pool fix from
31 ms). Remaining B4: wire it into the nvofg `recordCnnRefine` seam via Vulkan-CUDA interop + the
vendor-neutral coopmat port.

## Training plan on `ki-pc-fisch-101` (RTX 5080)

**Note — Blackwell toolchain:** the 5080 is `sm_120`; needs a recent CUDA (≥12.6) + a PyTorch build
with Blackwell kernels (recent stable or nightly cu126+). First step is verifying `torch.cuda` runs a
matmul on the 5080 before anything else.

- **T0. Env + smoke test.** PyTorch (Blackwell-capable) + CUDA; confirm fp16 Tensor-core matmul runs
  on the 5080; pin versions in a lockfile.
- **T1. Data pipeline (A2) — BEFORE the model; ships with an alignment gate.** (a) **Rendered GT
  (primary):** capture at **2× target fps** so every other frame is GT (color+MV+depth+UI/reactive);
  **first a sub-pixel alignment round-trip test** — warp captured N-1/N+1 to phase 0.5 with the known
  flow, assert it lands on capture-GT frame N to sub-pixel (identical jitter/MV-convention/HUD-mask);
  **must pass before mass capture** (a silent offset trains the net to compensate, not interpolate).
  Then tens of thousands of triplets across pans/fast-entities/transparency. (b) **Vimeo-90K/X4K
  bootstrap** for RGB synthesis warm-start (no aux). Dataloaders + augmentation (540p–1080p crops),
  `num_workers ≤ 3`.
- **T2. Model impl — phase-conditioned, bit-exact identity.** SoftSplat warp + gated-conv fusion net
  (~1M); **phase `t` as a conditioning input, trained with random t∈(0,1)** so one net serves any
  2×–6× (§ multiplier). Both modes (interp/extrap). **Identity-init assertion is BIT-EXACT:**
  zeroed final conv + residual add reproduces `classical_warp` exactly (to the warp's own fp16
  output, computed identically) — not "looks the same".
- **T3. Losses (§21.6).** Charbonnier/L1 + LPIPS/VGG + census/warping + light GAN (micro-detail) +
  temporal-stability; **UI/reactive masked out of all losses**; disocclusion regions up-weighted.
- **T4. Schedule.** Pretrain on Vimeo (synthesis) → fine-tune on rendered data with real MV/depth/
  masks. AMP fp16, largest batch the 16 GB fits (1080p crops). ~**days** for a competitive v1 (§21.7).
- **T5. Eval (A4 harness) — stratified, both modes, vs multiplier.** PSNR/SSIM/**LPIPS**/VMAF +
  temporal-stability vs GT, **reported per difficulty regime** (easy / disocclusion-heavy /
  shading-change) and **per mode** (interp/extrap) — an averaged metric is warp-dominated and hides
  the win. Also **quality vs multiplier** (2×…6×) so the "clean to 6× from 60 fps" claim is measured.
  Path B reference is a **read-only** comparison metric — never a training target. Gate every
  checkpoint.
- **T6. Export (A5).** Trained weights → fp16 coopmat/WMMA tile layout (versioned header) that the
  `recordCnnRefine` backend loads. Then the A1 identity is replaced by the real model — no ABI change.
- **T7. Ablate params + multiplier.** Sweep ~0.5M → ~4M vs measured 1080p/1440p/4K latency on a
  Blackwell deployment card; and sweep multiplier 2×–6× vs per-regime quality at 60–100 fps base; pick
  the v1 point and the safe max multiplier per base-fps band.

## Training runtime & resumability (connection-independent, pausable)

The run must **survive SSH disconnect** and **pause/resume with no loss**. Built as a harness *before*
the model (identity-first discipline), proven with a placeholder net, so T2 only drops in the real
model/data.

- **Connection-independent:** the trainer runs under **`tmux`** (session `plana`) — closing SSH leaves
  it running; re-attach anytime. Falls back to `nohup … & disown` if tmux is absent. My SSH sessions
  are only for launch/monitor, never the process parent.
- **Atomic checkpoints:** every `--ckpt-every` steps **and** on signal, save
  `{model, optimizer, AMP scaler, scheduler, step, epoch, dataloader position, CPU/CUDA/Python RNG
  states}` via temp-file + `os.replace` (atomic — a crash mid-write never corrupts). Keep `latest.pt`
  + rolling `ckpt_<step>.pt` + `best.pt`.
- **Auto-resume:** on start the trainer loads `latest.pt` and continues bit-for-bit (step, RNG,
  optimizer, scaler) — so restart-after-crash/reboot is a no-op beyond relaunch.
- **Clean pause (keeps the process, frees the GPU):** `touch PAUSE` → the loop checkpoints, then idles
  polling until `PAUSE` is removed; `rm PAUSE` resumes. No kill, no loss.
- **Graceful stop:** `SIGTERM`/`SIGUSR1`/`SIGINT` → checkpoint, exit; relaunch auto-resumes.
- **Control scripts:** `launch.sh` (start under tmux), `pause.sh` / `resume.sh` (PAUSE file),
  `stop.sh` (SIGTERM), `status.sh` (tail log + latest step). Run dir: `~/plana_runs/<run>/`.
- **Proven now:** the placeholder harness was run → checkpointed → killed → relaunched → resumed from
  the exact step, before any real model exists.

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
