# nvofg — native Vulkan frame generation on NVIDIA Optical Flow

`nvofg` is a small, **engine-agnostic frame-generation library** for **native Linux
Vulkan** applications. It synthesises interpolated frames *between* two rendered frames
to multiply presented framerate — the same class of feature as DLSS-G / FSR 3 FG — but:

- **native** — no Wine, no Proton, no Windows PE bridge;
- **no paid dependency** — no `Lossless.dll` (unlike lsfg-vk);
- **hardware-accelerated** — drives the GPU's dedicated Optical Flow Accelerator (OFA)
  through the standard **`VK_NV_optical_flow`** Vulkan extension, not a shader
  optical-flow approximation;
- **reusable** — a C ABI (`nvofg.h`) plus an idiomatic Rust crate any Vulkan engine can
  adopt.

> This is *our own* interpolation driven by NVIDIA's optical-flow **hardware**. It is
> **not** DLSS-G's learned model (which NVIDIA gates behind a Windows-only PE). What is
> native and usable is the OFA silicon and its Vulkan API — which is exactly what this
> library drives.

## Architecture at a glance

```
prev color ─┐
curr color ─┤  (+ optional depth, motion vectors, UI mask)
            ▼
  format prep ──▶ VK_NV_optical_flow execute ──▶ flow refine ──▶ interpolator ──▶ UI composite
   (compute)         (OFA hardware)               (compute)      (pluggable)       (graphics)
```

The interpolation stage is **pluggable** (`WarpInterpolator` today; CNN / Transformer
back-ends later) so the flow/pipeline machinery is decoupled from the synthesis method.

See [`design.md`](design.md) for the full specification and
[`docs/`](docs/) for architecture-decision records (ADRs) and technical analyses,
including the [CUDA-vs-Vulkan analysis](docs/analysis/cuda-vs-vulkan.md) and the
[Vulkan-native OFA decision](docs/adr/0001-vulkan-native-optical-flow.md).

## Status

Milestones (see `design.md` §10), all validation-clean on the reference machine:

- **M0** ✅ — native OFA path proven: `VK_NV_optical_flow` session, real execute, flow dump.
- **M1** ✅ — 2× core: prep → execute → refine → warp → blend through the C ABI.
- **M2** ✅ — aux quality: bidirectional flow + occlusion, motion-vector fusion (post +
  OFA hint), depth/reprojection disocclusion, UI + reactive + material masks, debug views.
- **M3** ✅ — API polish: `nvofg-sys` + safe `nvofg` Rust crate, `query_support`,
  `resize`, per-frame command ring.
- **M4** — integrate in RustMineClient.
- **M5** — optional present pacer + portable shader-flow fallback (Tier B).

## Requirements

- NVIDIA GPU with OFA (Turing or newer) and a driver exposing `VK_NV_optical_flow`
  (verified on RTX 5070 Laptop, driver 610.43.03).
- Vulkan 1.3+ headers/loader, a C++20 compiler, CMake ≥ 3.24.
- For the Rust wrapper: Rust 1.85+ and `ash`.

`nvofg` degrades gracefully: on a GPU without a usable OFA, `nvofg_create` returns
`NVOFG_UNSUPPORTED` and the application simply runs at 1×.

## Building

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
# run the support probe (M0):
./build/nvofg_probe
```

## License

Licensed under  [Apache-2.0](LICENSE-APACHE) and [MIT](LICENSE-MIT). Unless you explicitly state otherwise, any contribution you intentionally submit
for inclusion shall be dual-licensed as above, without additional terms.

`nvofg` links only against the Vulkan loader and the user's installed NVIDIA driver at
runtime; it ships **no** NVIDIA proprietary code or headers.
