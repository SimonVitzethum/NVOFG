"""Plan A model — SoftSplat forward-warp + gated-conv fusion (design.md §21.4).

- ~1M params (the size lever: we *supply* flow, so the net only does synthesis/inpainting).
- Phase-conditioned: takes t in (0,1) as a channel -> one net serves any 2x-6x multiplier.
- Residual on the warp + BIT-EXACT identity at init (zeroed final conv -> output == warp).
- Both modes: interpolation (prev+curr) and extrapolation (past-only) share the backbone.

T3 adds the real losses; T1 swaps the synthetic batch for rendered triplets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- differentiable forward splatting (softmax splatting, Niklaus & Liu 2020) ----------------
def forward_splat(feat, flow):
    """Scatter-add bilinear forward warp. feat [B,C,H,W], flow [B,2,H,W] (dx,dy px)."""
    B, C, H, W = feat.shape
    dev = feat.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    px = xx.float()[None] + flow[:, 0]
    py = yy.float()[None] + flow[:, 1]
    x0 = torch.floor(px); y0 = torch.floor(py)
    fx = px - x0; fy = py - y0
    out = torch.zeros(B, C, H * W, device=dev, dtype=feat.dtype)
    fl = feat.reshape(B, C, -1)
    for dx in (0, 1):
        for dy in (0, 1):
            xc = x0 + dx; yc = y0 + dy
            wx = fx if dx == 1 else (1 - fx)
            wy = fy if dy == 1 else (1 - fy)
            w = wx * wy
            valid = (xc >= 0) & (xc < W) & (yc >= 0) & (yc < H)
            idx = (yc.clamp(0, H - 1).long() * W + xc.clamp(0, W - 1).long())
            idx = idx.reshape(B, 1, -1).expand(B, C, -1)
            contrib = fl * (w * valid).reshape(B, 1, -1).to(fl.dtype)
            out.scatter_add_(2, idx, contrib)
    return out.reshape(B, C, H, W)


def softmax_splat(img, flow, metric):
    """Depth/importance-weighted forward warp: resolves many-to-one by exp(metric)."""
    w = torch.exp(metric.clamp(-20, 20))
    s = forward_splat(torch.cat([img * w, w], 1), flow)
    num, den = s[:, :-1], s[:, -1:]
    return num / (den + 1e-6)


# --- gated-conv fusion U-Net (ExtraNet-style) ------------------------------------------------
class GatedConv(nn.Module):
    def __init__(self, cin, cout, k=3, s=1):
        super().__init__()
        p = k // 2
        self.feat = nn.Conv2d(cin, cout, k, s, p)
        self.gate = nn.Conv2d(cin, cout, k, s, p)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.feat(x)) * torch.sigmoid(self.gate(x))


class FusionNet(nn.Module):
    """Encoder-decoder, gated convs, skip connections. Final conv zero-init -> zero residual."""
    def __init__(self, cin, base=24):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 3, base * 4
        self.e1 = nn.Sequential(GatedConv(cin, c1), GatedConv(c1, c1))
        self.e2 = nn.Sequential(GatedConv(c1, c2, s=2), GatedConv(c2, c2))
        self.e3 = nn.Sequential(GatedConv(c2, c3, s=2), GatedConv(c3, c3))
        self.e4 = nn.Sequential(GatedConv(c3, c4, s=2), GatedConv(c4, c4))
        self.d3 = nn.Sequential(GatedConv(c4 + c3, c3), GatedConv(c3, c3))
        self.d2 = nn.Sequential(GatedConv(c3 + c2, c2), GatedConv(c2, c2))
        self.d1 = nn.Sequential(GatedConv(c2 + c1, c1), GatedConv(c1, c1))
        self.out = nn.Conv2d(c1, 3, 3, 1, 1)          # residual (RGB correction)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)   # identity at init

    def _up(self, x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        s1 = self.e1(x); s2 = self.e2(s1); s3 = self.e3(s2); s4 = self.e4(s3)
        d3 = self.d3(torch.cat([self._up(s4, s3), s3], 1))
        d2 = self.d2(torch.cat([self._up(d3, s2), s2], 1))
        d1 = self.d1(torch.cat([self._up(d2, s1), s1], 1))
        return self.out(d1)


class FrameGenModel(nn.Module):
    """Warp candidate at phase t + learned residual. mode in {'interp','extrap'}."""
    def __init__(self, mode='interp', base=24):
        super().__init__()
        # fusion input: candidate(3)+prev(3)+curr(3)+flow(2)+t(1) = 12
        self.fusion = FusionNet(cin=12, base=base)
        self.mode = mode

    def warp_candidate(self, prev, curr, flow_fwd, flow_bwd, t):
        # flow_fwd: prev->curr (px). Warp prev forward by t; (interp) warp curr back by (1-t).
        m0 = torch.zeros_like(prev[:, :1])
        wp = softmax_splat(prev, flow_fwd * t, m0)
        if self.mode == 'interp':
            wc = softmax_splat(curr, flow_bwd * (1 - t), m0)
            cand = (1 - t) * wp + t * wc          # phase-weighted blend
        else:                                     # extrapolation: past only, project forward
            cand = wp
        return cand

    def forward(self, prev, curr, flow_fwd, flow_bwd, t):
        # t: [B,1,1,1]
        cand = self.warp_candidate(prev, curr, flow_fwd, flow_bwd, t)
        B, _, H, W = cand.shape
        tch = t.expand(B, 1, H, W)
        x = torch.cat([cand, prev, curr, flow_fwd, tch], 1)
        residual = self.fusion(x)
        return cand + residual, cand


def build_model(mode='interp'):
    return FrameGenModel(mode=mode)


def forward_batch(model, batch, device):
    """Run the model on a data.py batch dict -> (out, cand, target)."""
    prev = batch['prev'].to(device); curr = batch['curr'].to(device)
    ff = batch['flow_fwd'].to(device); fb = batch['flow_bwd'].to(device)
    t = batch['t'].to(device).view(-1, 1, 1, 1)
    target = batch['target'].to(device)
    out, cand = model(prev, curr, ff, fb, t)
    return out, cand, target


# --- synthetic self-supervised batch (T1 replaces with rendered triplets) --------------------
def _grid_warp(img, flow):
    B, C, H, W = img.shape
    dev = img.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    gx = (xx.float()[None] + flow[:, 0]) / (W - 1) * 2 - 1
    gy = (yy.float()[None] + flow[:, 1]) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(img, grid, align_corners=True, padding_mode='border')


def synth_batch(B, H, W, device):
    base = torch.rand(B, 3, H, W, device=device)
    fx = (torch.rand(B, 1, 1, 1, device=device) * 12 - 6)
    fy = (torch.rand(B, 1, 1, 1, device=device) * 12 - 6)
    flow_fwd = torch.cat([fx.expand(B, 1, H, W), fy.expand(B, 1, H, W)], 1)  # prev->curr
    flow_bwd = -flow_fwd
    t = torch.rand(B, 1, 1, 1, device=device)
    prev = base
    curr = _grid_warp(base, -flow_fwd)            # curr = base shifted by full flow
    target = _grid_warp(base, -flow_fwd * t)      # GT in-between at phase t
    return prev, curr, flow_fwd, flow_bwd, t, target


def train_step(model, device):
    prev, curr, ff, fb, t, target = synth_batch(4, 128, 128, device)
    out, _ = model(prev, curr, ff, fb, t)
    return F.l1_loss(out, target)


# --- bit-exact identity check ----------------------------------------------------------------
def identity_check(device='cuda'):
    m = build_model('interp').to(device).eval()
    with torch.no_grad():
        prev, curr, ff, fb, t, _ = synth_batch(2, 96, 96, device)
        out, cand = m(prev, curr, ff, fb, t)
        exact = torch.equal(out, cand)
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.3f}M | identity bit-exact (out==warp): {exact}")
    return exact, n


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    ok, n = identity_check(dev)
    assert ok, "identity NOT bit-exact"
