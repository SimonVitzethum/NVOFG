"""Plan A A4 validation harness — stratified by difficulty regime, model vs warp, per mode/phase.

The point (per review): our classical warp is already best-in-class, so an *averaged* metric is
warp-dominated and hides the learned model's win, which lives ONLY in the hard regimes. So we bucket
each sample into easy / disocclusion / shading and report warp-vs-model *per regime* — you must see
the model win where it should, and at least tie on easy.
"""
import argparse, os, sys, collections
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, forward_batch
from losses import flow_disocc_weight
import data
import lpips


def psnr(x, y):
    mse = ((x - y) ** 2).mean(dim=[1, 2, 3]).clamp(min=1e-10)
    return 10 * torch.log10(1.0 / mse)


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    mx = F.avg_pool2d(x, 7, 1, 3); my = F.avg_pool2d(y, 7, 1, 3)
    sx = F.avg_pool2d(x * x, 7, 1, 3) - mx ** 2
    sy = F.avg_pool2d(y * y, 7, 1, 3) - my ** 2
    sxy = F.avg_pool2d(x * y, 7, 1, 3) - mx * my
    s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))
    return s.mean(dim=[1, 2, 3])


def regimes(batch, cand, target, dev):
    ff = batch['flow_fwd'].to(dev)
    disc = flow_disocc_weight(ff).mean(dim=[1, 2, 3])
    res = (cand - target).abs().mean(dim=[1, 2, 3])
    out = []
    for i in range(cand.shape[0]):
        out.append('disocc' if disc[i] > 0.15 else ('shading' if res[i] > 0.06 else 'easy'))
    return out


@torch.no_grad()
def evaluate(ckpt, data_dir, device='cuda', modes=('interp',)):
    lpfn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()
    loader = data.make_loader(data_dir, batch=8, workers=0)
    print(f"eval on {len(loader.dataset)} samples")
    for mode in modes:
        m = build_model(mode).to(device).eval()
        if ckpt and os.path.exists(ckpt):
            st = torch.load(ckpt, map_location='cpu', weights_only=False)
            m.load_state_dict(st['model']); print(f"[{mode}] loaded {ckpt} (step {st.get('step','?')})")
        else:
            print(f"[{mode}] no ckpt -> identity model (== warp), expect ~0 delta")
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        pht = collections.defaultdict(lambda: collections.defaultdict(list))  # per-phase (multiplier)
        for batch in loader:
            out, cand, target = forward_batch(m, batch, device)
            out = out.clamp(0, 1); cand = cand.clamp(0, 1); target = target.clamp(0, 1)
            reg = regimes(batch, cand, target, device)
            tv = batch['t'].reshape(-1)
            for i, r in enumerate(reg):
                o, c, t = out[i:i+1], cand[i:i+1], target[i:i+1]
                wp, mp = psnr(c, t).item(), psnr(o, t).item()
                agg[r]['warp_psnr'].append(wp); agg[r]['model_psnr'].append(mp)
                agg[r]['warp_lpips'].append(lpfn(c * 2 - 1, t * 2 - 1).item())
                agg[r]['model_lpips'].append(lpfn(o * 2 - 1, t * 2 - 1).item())
                key = round(float(tv[i]), 2)                       # phase t=j/K -> multiplier bucket
                pht[key]['warp_psnr'].append(wp); pht[key]['model_psnr'].append(mp)
        mean = lambda v: sum(v) / max(1, len(v))
        print(f"\n=== mode={mode} : model vs warp, per regime ===")
        print(f"{'regime':10} {'n':>4} {'PSNR warp→model':>22} {'LPIPS warp→model':>24}")
        for r in ('easy', 'shading', 'disocc'):
            if not agg[r]['warp_psnr']:
                continue
            n = len(agg[r]['warp_psnr'])
            wp, mp = mean(agg[r]['warp_psnr']), mean(agg[r]['model_psnr'])
            wl, ml = mean(agg[r]['warp_lpips']), mean(agg[r]['model_lpips'])
            print(f"{r:10} {n:>4}   {wp:6.2f} -> {mp:6.2f} ({mp-wp:+.2f})   {wl:6.3f} -> {ml:6.3f} ({ml-wl:+.3f})")
        # #5 per-phase uniformity: the phase-conditioned net must serve every multiplier (small t=near
        # a real frame .. large t=deep in-between) — a big spread means some multipliers are underserved.
        print(f"\n=== mode={mode} : per-phase (t=j/K), model vs warp — uniformity across 2x..Kx ===")
        print(f"{'t':>6} {'n':>5}  {'warp':>7} {'model':>7} {'gain':>7}")
        gains = []
        for key in sorted(pht):
            n = len(pht[key]['warp_psnr'])
            wp, mp = mean(pht[key]['warp_psnr']), mean(pht[key]['model_psnr'])
            gains.append(mp)
            print(f"{key:6.2f} {n:>5}  {wp:7.2f} {mp:7.2f} {mp-wp:+7.2f}")
        if gains:
            print(f"model PSNR spread across phases: {max(gains)-min(gains):.2f} dB "
                  f"(min {min(gains):.2f} @worst phase — lower spread = more uniform multiplier support)")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--modes', default='interp')
    a = p.parse_args()
    evaluate(a.ckpt, a.data_dir, modes=a.modes.split(','))
