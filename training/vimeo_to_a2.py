"""Vimeo-90k (triplet, HuggingFace parquet) -> A2 dataset (Plan A #1, real-data path).

Vimeo-90k triplet is THE video-frame-interpolation training standard: 73k real clips of diverse
natural motion, 448x256, 3 frames each (im1, im2, im3). We use im1+im3 as the input pair and im2 as
the ground-truth in-between (2x, phase t=0.5) — exactly a K=2 A2 sample. The motion field is ESTIMATED
with RAFT (im1 -> im3), matching nvofg's inference flow (OFA/block-match), not engine MVs.

HF layout (danjacobellis/vimeo90k_triplet): each parquet ROW is one frame {image:{bytes},path:imN.png,
label:triplet_idx}; every 3 consecutive rows form a triplet. We stream shards, decode, run RAFT on the
GPU, and write a 3-frame A2 clip per triplet (mv on frame0 = flow im1->im3). A motion filter drops
near-static triplets (no learning signal). Run align.py on a sample before mass training.

Usage:
  vimeo_to_a2.py --parquet-dir ~/vimeo_dl/data --out ~/plana_data/vimeo --min-motion 0.5 --max-clips 8000
"""
import argparse, glob, io, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_to_a2 import flow_batch   # shared RAFT wrapper


def _decode(png_bytes, dev):
    from PIL import Image
    img = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    a = np.asarray(img, np.float32) / 255.0                    # [H,W,3]
    return torch.from_numpy(a).permute(2, 0, 1)                # [3,H,W]


def iter_triplets(parquet_dir):
    import pyarrow.parquet as pq
    for pf in sorted(glob.glob(os.path.join(parquet_dir, '*.parquet'))):
        imgs = pq.read_table(pf, columns=['image']).to_pydict()['image']   # struct {bytes, path}
        for k in range(0, len(imgs) - 2, 3):
            if imgs[k]['path'] == 'im1.png' and imgs[k + 2]['path'] == 'im3.png':
                yield (imgs[k]['bytes'], imgs[k + 1]['bytes'], imgs[k + 2]['bytes'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parquet-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--capture-fps', type=int, default=60)
    p.add_argument('--target-fps', type=int, default=30)       # -> K=2 (2x, t=0.5)
    p.add_argument('--min-motion', type=float, default=0.5, help='drop triplets with mean|flow|<this px')
    p.add_argument('--max-clips', type=int, default=100000)
    p.add_argument('--batch', type=int, default=16)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(a.out, exist_ok=True)
    K = max(1, round(a.capture_fps / a.target_fps))

    buf, kept, seen, dropped = [], 0, 0, 0
    def flush():
        nonlocal kept, dropped
        if not buf:
            return
        im1 = torch.stack([b[0] for b in buf]); im3 = torch.stack([b[2] for b in buf])
        fl = flow_batch(im1, im3, dev).permute(0, 2, 3, 1).cpu().numpy()   # [B,H,W,2] px, im1->im3
        for j, (a1, a2, a3) in enumerate(buf):
            mag = float(np.abs(fl[j]).mean())
            if mag < a.min_motion:                             # near-static -> no learning signal
                dropped += 1; continue
            H, W = a1.shape[1], a1.shape[2]
            d = os.path.join(a.out, f'clip_{kept:06d}'); os.makedirs(d, exist_ok=True)
            json.dump({'capture_fps': a.capture_fps, 'target_fps': a.target_fps, 'width': W, 'height': H,
                       'near': 0.1, 'far': 1000.0, 'mv_convention': 'prev_to_curr_interval_px'},
                      open(os.path.join(d, 'meta.json'), 'w'))
            frames = [a1, a2, a3]
            for n in range(3):
                col = frames[n].permute(1, 2, 0).numpy().astype(np.float16)
                # A2 convention: the loader reads mv from the CURR frame (frames[i+K]); for a triplet
                # (i=0,K=2) that is frame index 2 (im3). Store flow(im1->im3) there, not on frame0.
                mv = fl[j].astype(np.float16) if n == 2 else np.zeros((H, W, 2), np.float16)
                np.savez_compressed(os.path.join(d, f'frame_{n:06d}.npz'),
                                    color=col, mv=mv, depth=np.full((H, W), 0.5, np.float16),
                                    ui=np.zeros((H, W), np.uint8), reactive=np.zeros((H, W), np.uint8))
            kept += 1
        buf.clear()

    for trip in iter_triplets(a.parquet_dir):
        if kept >= a.max_clips:
            break
        buf.append((_decode(trip[0], dev), _decode(trip[1], dev), _decode(trip[2], dev)))
        seen += 1
        if len(buf) >= a.batch:
            flush()
            if seen % 512 == 0:
                print(f'seen {seen}  kept {kept}  dropped(static) {dropped}', flush=True)
    flush()
    print(f'DONE: kept {kept} clips (K={K}, 2x), dropped {dropped} static -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
