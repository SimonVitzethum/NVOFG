"""A2 capture format + triplet dataloader (Plan A T1).

CAPTURE FORMAT (what the RMC/Minecraft harness produces — coordinate on this):
  <root>/<clip>/frame_000000.npz, frame_000001.npz, ...   captured at capture_fps (>= 6x target)
  each .npz:  color  float16 [H,W,3]  linear (HDR ok)
              mv     float16 [H,W,2]  pixels, prev->this-frame convention
              depth  float16 [H,W]    linear (or NDC + near/far in meta)
              ui     uint8   [H,W]    255 = UI/HUD (never interpolate)
              reactive uint8 [H,W]    255 = reactive (alpha/particles)
  <root>/<clip>/meta.json: {capture_fps, target_fps, width, height, near, far, mv_convention}

Capturing at N x target fps (N>=6) means between two real-rate frames (stride K=capture/target)
there are K-1 GT in-betweens -> we train the phase-conditioned net on real GT at t=j/K for j=1..K-1,
which is exactly what lets one net serve 2x..6x. A training sample = (frame i, frame i+K) inputs,
frame i+j GT at phase t=j/K.

The alignment gate (align.py) MUST pass on a sample of the capture before mass training.
"""
import glob, json, os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _mv_to_flow(mv):  # (2,H,W) pixels prev->this; flow_fwd(prev->curr) accumulates; here per-interval
    return mv


class TripletDataset(Dataset):
    def __init__(self, root, target_fps=60, cache=False):
        # cache=False (default): np.load per access and let the OS page cache absorb repeats. The old
        # in-RAM cache duplicated the page cache in *anonymous shared memory* across persistent workers
        # and grew unbounded (tens of GB at 448x256) — off by default so RAM stays flat.
        self.cache = cache
        self.samples = []
        for clip in sorted(glob.glob(os.path.join(root, '*'))):
            meta_p = os.path.join(clip, 'meta.json')
            if not os.path.isdir(clip) or not os.path.exists(meta_p):
                continue
            meta = json.load(open(meta_p))
            K = max(1, round(meta['capture_fps'] / target_fps))
            frames = sorted(glob.glob(os.path.join(clip, 'frame_*.npz')))
            n = len(frames)
            for i in range(0, n - K):
                for j in range(1, K):                 # GT in-betweens -> phases j/K
                    self.samples.append((frames[i], frames[i + K], frames[i + j], j / K))
        self.K_note = "phase-conditioned: t=j/K covers 2x..Kx from one net"
        self._cache = {}

    def __len__(self):
        return len(self.samples)

    def _load(self, p):
        if not self.cache:
            d = np.load(p); return {k: d[k] for k in d.files}   # OS page cache handles repeats
        c = self._cache.get(p)
        if c is None:
            d = np.load(p); c = {k: d[k] for k in d.files}; self._cache[p] = c
        return c

    def __getitem__(self, k):
        pa, pb, pg, t = self.samples[k]
        a, b, g = self._load(pa), self._load(pb), self._load(pg)
        to = lambda x: torch.from_numpy(np.ascontiguousarray(x)).float()
        prev = to(a['color']).permute(2, 0, 1)
        curr = to(b['color']).permute(2, 0, 1)
        target = to(g['color']).permute(2, 0, 1)
        flow_fwd = to(b['mv']).permute(2, 0, 1)       # prev->curr over the interval (px)
        flow_bwd = -flow_fwd
        depth = to(a['depth'])[None]
        ui = to(a['ui'])[None] / 255.0
        reactive = to(a['reactive'])[None] / 255.0
        return {'prev': prev, 'curr': curr, 'target': target,
                'flow_fwd': flow_fwd, 'flow_bwd': flow_bwd,
                'depth': depth, 'ui': ui, 'reactive': reactive,
                't': torch.tensor([t], dtype=torch.float32)}


def make_loader(root, batch=4, target_fps=60, workers=3, cache=False):
    ds = TripletDataset(root, target_fps, cache=cache)
    return DataLoader(ds, batch_size=batch, shuffle=True, num_workers=min(workers, 3),
                      pin_memory=True, drop_last=True, persistent_workers=(workers > 0))


# --- self-test: write fake clips, load them, prove the pipeline -------------------------------
def _write_fake_clip(root, name, H=64, W=64, capture_fps=360, target_fps=60, nframes=14, coherent=True):
    """coherent=True: a constant-velocity sequence with the correct interval flow, so the alignment
    gate PASSES and training can run (mimics a well-aligned capture). coherent=False: independent
    random frames (a misaligned capture) -> the gate must FAIL."""
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    K = max(1, round(capture_fps / target_fps))
    json.dump({'capture_fps': capture_fps, 'target_fps': target_fps, 'width': W, 'height': H,
               'near': 0.1, 'far': 1000.0, 'mv_convention': 'prev_to_curr_interval_px'},
              open(os.path.join(d, 'meta.json'), 'w'))
    base = np.random.rand(H, W, 3).astype(np.float32)
    vx, vy = 1, -1                                             # px per capture-frame (integer -> exact)
    intmv = np.zeros((H, W, 2), np.float16); intmv[..., 0] = vx * K; intmv[..., 1] = vy * K
    for i in range(nframes):
        if coherent:
            color = np.roll(np.roll(base, vy * i, axis=0), vx * i, axis=1)
            mv = intmv                                        # constant velocity -> interval flow = K*v
        else:
            color = np.random.rand(H, W, 3).astype(np.float32); mv = np.zeros((H, W, 2), np.float16)
        np.savez(os.path.join(d, f'frame_{i:06d}.npz'),
                 color=color.astype(np.float16), mv=mv,
                 depth=np.random.rand(H, W).astype(np.float16),
                 ui=np.zeros((H, W), np.uint8), reactive=np.zeros((H, W), np.uint8))


def _write_synth_clip(root, name, H=96, W=96, capture_fps=360, target_fps=60, nframes=16, seed=0):
    """Richer synthetic clip WITH disocclusion (2-layer motion) + shading (brightness) — so the hard
    regimes exist and the learned model's per-regime gain over the warp is testable before real data.
    A foreground patch slides over a background at a different velocity -> reveals background (holes
    the warp can't fill); a per-frame brightness the warp can't predict -> shading regime."""
    rng = np.random.default_rng(seed)
    d = os.path.join(root, name); os.makedirs(d, exist_ok=True)
    K = max(1, round(capture_fps / target_fps))
    json.dump({'capture_fps': capture_fps, 'target_fps': target_fps, 'width': W, 'height': H,
               'near': 0.1, 'far': 1000.0, 'mv_convention': 'prev_to_curr_interval_px'},
              open(os.path.join(d, 'meta.json'), 'w'))
    bg = rng.random((H, W, 3)).astype(np.float32)
    fg = rng.random((H, W, 3)).astype(np.float32)
    sz = H // 3; fy0, fx0 = H // 3, W // 4
    m0 = np.zeros((H, W), np.float32); m0[fy0:fy0 + sz, fx0:fx0 + sz] = 1
    vfg = np.array([int(rng.integers(1, 3)), int(rng.integers(-2, 3))])   # px/frame (integer)
    for n in range(nframes):
        fs = vfg * n
        mn = np.roll(np.roll(m0, fs[1], axis=0), fs[0], axis=1)
        fgn = np.roll(np.roll(fg, fs[1], axis=0), fs[0], axis=1)
        color = bg * (1 - mn[..., None]) + fgn * mn[..., None]
        bright = 1.0 + 0.15 * np.sin(0.5 * n + seed)                       # shading the warp can't predict
        color = np.clip(color * bright, 0, 1)
        mv = np.zeros((H, W, 2), np.float16)                               # bg static; fg = K*vfg
        mv[mn > 0, 0] = K * vfg[0]; mv[mn > 0, 1] = K * vfg[1]
        depth = np.where(mn > 0, 0.3, 0.8).astype(np.float16)
        np.savez(os.path.join(d, f'frame_{n:06d}.npz'),
                 color=color.astype(np.float16), mv=mv, depth=depth,
                 ui=np.zeros((H, W), np.uint8), reactive=np.zeros((H, W), np.uint8))


if __name__ == '__main__':
    import tempfile
    root = tempfile.mkdtemp()
    _write_fake_clip(root, 'clip_0000')
    ld = make_loader(root, batch=2, target_fps=60, workers=0)
    print(f"dataset samples: {len(ld.dataset)} (stride K=6 -> 5 phases per interval)")
    b = next(iter(ld))
    shapes = {k: tuple(v.shape) for k, v in b.items()}
    print("batch keys/shapes:", shapes)
    ts = sorted(set(round(float(t), 3) for (_, _, _, t) in ld.dataset.samples))
    print("phases present (t):", ts)
    assert b['prev'].shape == (2, 3, 64, 64) and len(ts) == 5, "loader shape/phase mismatch"
    print("DATA OK: format + loader + phase coverage verified")
