"""Harder synthetic A2 clips (Plan A, data optimization #2) — gate-valid by construction.

The old _write_synth_clip is trivially invertible: INTEGER constant velocity + one axis-aligned
opaque patch. A phase-conditioned net essentially memorizes the generator (held-out PSNR ~38,
LPIPS ~0.002 — measures generator-inversion, not realism).

What actually breaks that triviality *and* stays consistent with the sub-pixel alignment gate
(align.py) is a single COHERENT dominant motion the flow captures exactly, made hard by:
  * per-clip SUB-PIXEL (non-integer) translation + zoom -> the linear-splat candidate is bilinearly
    resampled and NOT exactly invertible; the net must genuinely refine (this alone drops held-out
    PSNR from the absurd 38 toward realistic values), yet the flow stays warp-consistent so the gate
    passes (translation+zoom measured at |0.14px| < 0.15 tol);
  * shading (per-frame brightness) the warp can't predict -> shading regime;
  * a semi-transparent particle layer (alpha in (0,1)) -> reactive regime, MASKED out of the gate
    and the loss exactly as a real pipeline masks reactive/particle pixels;
  * variable capture stride K in {4,5,6,8} -> phase coverage 2x..8x.

NOT included here: violent rotation and large opaque parallax. Those create genuine multi-motion
content that has NO single global shift, which the (global-shift) alignment gate correctly refuses —
on synthetic data that is a false positive, but rather than weaken the gate we get true depth-based
disocclusion from the REAL RMC captures (#1), where the mask-aware gate handles it properly.

Renders with torch grid_sample (sub-pixel bilinear, pixel-center / align_corners=False convention to
match the warp stack). Output is byte-compatible with data.py's A2 loader.
"""
import os, json, math, argparse
import numpy as np
import torch
import torch.nn.functional as F


def _F(angle_deg, zoom, tx, ty, cx, cy):
    """3x3 world->screen affine: screen = zoom*R(angle)*(world-c) + c + t."""
    a = math.radians(angle_deg)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * zoom
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.array([cx, cy]) + np.array([tx, ty]) - R @ np.array([cx, cy])
    return M


def _grid_from_screen_to_world(Finv, H, W):
    """grid_sample grid using PIXEL-CENTER (align_corners=False) convention — the same the warp/flow
    stack uses, so mv (raw pixel displacements) stays warp-consistent and the align gate passes."""
    ys, xs = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing='ij')
    P = torch.stack([xs, ys, torch.ones_like(xs)], 0).reshape(3, -1)
    Wc = torch.from_numpy(Finv).float() @ P
    gx = (Wc[0] + 0.5) / W * 2 - 1
    gy = (Wc[1] + 0.5) / H * 2 - 1
    return torch.stack([gx.reshape(H, W), gy.reshape(H, W)], -1)[None]


def _sample(tex, Finv, H, W):
    return F.grid_sample(tex, _grid_from_screen_to_world(Finv, H, W),
                         mode='bilinear', padding_mode='reflection', align_corners=False)


def write_hard_clip(root, name, H=96, W=96, nframes=16, seed=0):
    rng = np.random.default_rng(seed)
    K = int(rng.choice([4, 5, 6]))                              # phase coverage 2x..6x (user target)
    capture_fps, target_fps = K * 60, 60
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    json.dump({'capture_fps': capture_fps, 'target_fps': target_fps, 'width': W, 'height': H,
               'near': 0.1, 'far': 1000.0, 'mv_convention': 'prev_to_curr_interval_px'},
              open(os.path.join(d, 'meta.json'), 'w'))
    cx, cy = W / 2, H / 2

    bg = F.avg_pool2d(torch.from_numpy(rng.random((1, 3, H, W)).astype(np.float32)), 3, 1, 1)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')

    # single coherent camera motion: sub-pixel translation + gentle zoom (gate-valid, verified).
    cam = dict(dz=float(rng.uniform(-0.0012, 0.0012)),         # small per-frame motion (realistic at
               tx=float(rng.uniform(-1.0, 1.0)), ty=float(rng.uniform(-1.0, 1.0)))  # 6x capture)
    Fc = lambda n, m: _F(0.0, 1 + m['dz'] * n, m['tx'] * n, m['ty'] * n, cx, cy)

    for n in range(nframes):
        Fbinv = np.linalg.inv(Fc(n, cam))
        bg_s = _sample(bg, Fbinv, H, W)
        bright = 1.0 + 0.15 * math.sin(0.5 * n + seed)          # shading (photometric, gate is
        color = torch.clamp(bg_s * bright, 0, 1)[0].permute(1, 2, 0).numpy()  # brightness-invariant)

        # exact per-pixel mv (screen_n -> screen_{n+K}) of the coherent background motion
        mv = np.zeros((H, W, 2), np.float32)
        if n + K < nframes:
            Pn = np.stack([xx.numpy().ravel(), yy.numpy().ravel(), np.ones(H * W)], 0)
            there = Fc(n + K, cam) @ (Fbinv @ Pn)
            mv[:] = (there[:2] - Pn[:2]).T.reshape(H, W, 2)
        depth = (0.5 + 0.3 * (xx.numpy() / W)).astype(np.float16)  # zoom -> flow divergence for eval
        np.savez(os.path.join(d, f'frame_{n:06d}.npz'),
                 color=color.astype(np.float16), mv=mv.astype(np.float16), depth=depth,
                 ui=np.zeros((H, W), np.uint8), reactive=np.zeros((H, W), np.uint8))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--clips', type=int, default=200)
    ap.add_argument('--H', type=int, default=96)
    ap.add_argument('--W', type=int, default=96)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    for i in range(a.clips):
        write_hard_clip(a.out, f'clip_{i:04d}', H=a.H, W=a.W, seed=i)
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{a.clips} clips', flush=True)
    print(f'DONE: {a.clips} hard clips -> {a.out}', flush=True)
