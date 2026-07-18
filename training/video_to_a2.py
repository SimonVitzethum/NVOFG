"""video -> A2 dataset converter (Plan A #1, real-data path).

Turns high-frame-rate video into the A2 training format the harness already consumes, using ESTIMATED
optical flow (RAFT) as the motion field instead of engine motion vectors. This matches nvofg's actual
inference — the runtime warp uses the NVIDIA OFA / portable block-match flow, NOT engine MVs — so a
model trained on RAFT flow sees the same kind of motion field it will see in deployment.

Per clip it writes frame_000000.npz .. in the A2 layout:
  color float16 [H,W,3]   (linear-ish; we store the decoded sRGB, good enough for a first pass)
  mv    float16 [H,W,2]    RAFT flow(n -> n+K), px, prev->curr interval  == the training convention
  depth float16 [H,W]      0.5 (unknown from video; the model's depth channel is optional)
  ui/reactive uint8 [H,W]  0   (no HUD/particle labels from raw video)
plus meta.json {capture_fps, target_fps} so TripletDataset forms (i, i+K, i+j) phase samples.

Run the alignment gate (align.py) on the result before mass training — RAFT is good but large motion /
occlusion clips will (correctly) be filtered.

Usage:
  video_to_a2.py --input clip.mp4 --out ~/plana_data/real --height 256 --width 256 \
                 --capture-fps 120 --target-fps 60 --clip-len 24
  video_to_a2.py --selftest --out /tmp/v2a_test        # no video needed; proves flow+format+gate
"""
import argparse, json, math, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_RAFT = None
_TF = None


def _raft(dev):
    global _RAFT, _TF
    if _RAFT is None:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        w = Raft_Large_Weights.DEFAULT
        _RAFT = raft_large(weights=w, progress=False).eval().to(dev)
        _TF = w.transforms()
    return _RAFT, _TF


@torch.no_grad()
def flow_batch(a, b, dev):
    """a,b: [B,3,H,W] in [0,1]. Returns RAFT flow a->b, [B,2,H,W] px. H,W must be /8."""
    raft, tf = _raft(dev)
    a2, b2 = tf(a.to(dev), b.to(dev))
    return raft(a2, b2)[-1]


def _decode_video(path, H, W, max_frames):
    import av
    container = av.open(path)
    out = []
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='rgb24')                 # [h,w,3] uint8
        t = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        t = F.interpolate(t, size=(H, W), mode='area')
        out.append(t[0])
        if len(out) >= max_frames:
            break
    return torch.stack(out) if out else torch.empty(0)


def _synth_frames(H, W, n=24):
    """A translating low-pass noise field with a known constant velocity -> RAFT must recover ~(vx,vy)
    and the alignment gate must pass; proves the whole flow+format+gate path without a video file."""
    g = torch.Generator().manual_seed(0)
    vx, vy = 2, 1
    pad = max(vx, vy) * n + 8                                   # enough headroom for n steps
    base = F.avg_pool2d(torch.rand(1, 3, H + 2 * pad, W + 2 * pad, generator=g), 5, 1, 2)
    frames = []
    for i in range(n):
        frames.append(base[:, :, pad + vy * i:pad + vy * i + H, pad + vx * i:pad + vx * i + W][0])
    return torch.stack(frames)


def write_clip(frames, out_dir, name, K, capture_fps, target_fps, dev, batch=8):
    """frames: [N,3,H,W] in [0,1]. Writes A2 npz per frame with mv = flow(n->n+K)."""
    N, _, H, W = frames.shape
    d = os.path.join(out_dir, name); os.makedirs(d, exist_ok=True)
    json.dump({'capture_fps': capture_fps, 'target_fps': target_fps, 'width': W, 'height': H,
               'near': 0.1, 'far': 1000.0, 'mv_convention': 'prev_to_curr_interval_px'},
              open(os.path.join(d, 'meta.json'), 'w'))
    mv = np.zeros((N, H, W, 2), np.float16)
    for s in range(0, N - K, batch):
        idx = list(range(s, min(s + batch, N - K)))
        a = frames[idx]; b = frames[[i + K for i in idx]]
        fl = flow_batch(a, b, dev).permute(0, 2, 3, 1).cpu().numpy()   # [b,H,W,2] px
        for j, i in enumerate(idx):
            mv[i] = fl[j].astype(np.float16)
    for n in range(N):
        col = frames[n].permute(1, 2, 0).numpy().astype(np.float16)
        # compressed: smooth flow + constant depth/masks shrink a lot vs np.savez (real-data disk win)
        np.savez_compressed(os.path.join(d, f'frame_{n:06d}.npz'),
                 color=col, mv=mv[n], depth=np.full((H, W), 0.5, np.float16),
                 ui=np.zeros((H, W), np.uint8), reactive=np.zeros((H, W), np.uint8))
    return N


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', help='video file (mp4/mkv/...)')
    p.add_argument('--out', required=True)
    p.add_argument('--height', type=int, default=256)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--capture-fps', type=int, default=120)
    p.add_argument('--target-fps', type=int, default=60)
    p.add_argument('--clip-len', type=int, default=24)
    p.add_argument('--max-frames', type=int, default=100000)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    assert a.height % 8 == 0 and a.width % 8 == 0, "H,W must be multiples of 8 (RAFT)"
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    K = max(1, round(a.capture_fps / a.target_fps))
    os.makedirs(a.out, exist_ok=True)

    if a.selftest:
        frames = _synth_frames(a.height, a.width, max(a.clip_len, K + 4))
        nf = write_clip(frames, a.out, 'clip_0000', K, a.capture_fps, a.target_fps, dev)
        # verify with the alignment gate
        import data, align
        from model import build_model, forward_batch
        loader = data.make_loader(a.out, batch=8, workers=0)
        _, cand, target = forward_batch(build_model('interp').to(dev), next(iter(loader)), dev)
        ok, off, mag = align.alignment_gate(cand.float(), target.float())
        print(f"[selftest] wrote {nf} frames; RAFT flow mean|v|={np.abs(np.load(os.path.join(a.out,'clip_0000','frame_000000.npz'))['mv']).mean():.2f}px")
        print(f"[selftest] align gate: offset={off[0]:+.2f},{off[1]:+.2f} |{mag:.2f}| -> {'PASS' if ok else 'FAIL'}")
        print("SELFTEST OK" if ok else "SELFTEST GATE FAIL")
        return

    assert a.input, "--input required (or --selftest)"
    frames = _decode_video(a.input, a.height, a.width, a.max_frames)
    if frames.numel() == 0:
        print("no frames decoded"); return
    nclips = 0; total = 0
    for s in range(0, frames.shape[0] - K, a.clip_len):
        clip = frames[s:s + a.clip_len]
        if clip.shape[0] < K + 2:
            break
        total += write_clip(clip, a.out, f'clip_{nclips:04d}', K, a.capture_fps, a.target_fps, dev)
        nclips += 1
    print(f"DONE: {nclips} clips, {total} frames -> {a.out}  (run align.py before mass training)")


if __name__ == '__main__':
    main()
