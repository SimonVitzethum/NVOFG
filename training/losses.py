"""Plan A losses (§21.6): Charbonnier + LPIPS + soft-census, UI/reactive masked out of ALL losses,
disocclusion regions up-weighted. Optional temporal-stability (two-phase flicker penalty).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


def _census(img, k=7):
    g = img.mean(1, keepdim=True)
    pad = k // 2
    patches = F.unfold(F.pad(g, [pad] * 4, mode='reflect'), k)      # [B,k*k,HW]
    center = patches[:, k * k // 2:k * k // 2 + 1]
    return torch.tanh(32.0 * (patches - center))                    # soft ternary signs


def flow_disocc_weight(flow_fwd):
    """Cheap disocclusion proxy: object-boundary flow discontinuity (up-weighted in the loss)."""
    gx = (flow_fwd[:, :, :, 1:] - flow_fwd[:, :, :, :-1]).abs().sum(1, keepdim=True)
    gy = (flow_fwd[:, :, 1:, :] - flow_fwd[:, :, :-1, :]).abs().sum(1, keepdim=True)
    gx = F.pad(gx, [0, 1, 0, 0]); gy = F.pad(gy, [0, 0, 0, 1])
    d = (gx + gy)
    return (d / (d.amax(dim=[1, 2, 3], keepdim=True) + 1e-6)).clamp(0, 1)


class CompositeLoss(nn.Module):
    def __init__(self, device, w_char=1.0, w_lpips=1.0, w_census=0.1, disocc_boost=3.0):
        super().__init__()
        self.lp = lpips.LPIPS(net='vgg', verbose=False).to(device).eval()
        for p in self.lp.parameters():
            p.requires_grad_(False)
        self.w = dict(char=w_char, lpips=w_lpips, census=w_census)
        self.boost = disocc_boost

    def forward(self, out, target, batch):
        dev = out.device
        B, C, H, W = out.shape
        ui = batch['ui'].to(dev); reactive = batch['reactive'].to(dev)
        keep = (1 - ui) * (1 - reactive)                            # 0 on HUD/particles
        disocc = flow_disocc_weight(batch['flow_fwd'].to(dev))
        w = keep * (1 + self.boost * disocc)                        # up-weight disocclusion
        wsum = w.sum().clamp(min=1.0)

        char = (torch.sqrt((out - target) ** 2 + 1e-6) * w).sum() / wsum / C
        m = keep
        lp = self.lp(out * m * 2 - 1, target * m * 2 - 1).mean()    # LPIPS on masked, [-1,1]
        cen = ((_census(out * keep) - _census(target * keep)).abs().mean())
        total = self.w['char'] * char + self.w['lpips'] * lp + self.w['census'] * cen
        return total, {'char': char.item(), 'lpips': lp.item(), 'census': cen.item()}


def temporal_pair_loss(model, batch, device, forward_batch, dt=0.08):
    """Flicker penalty: outputs at t and t+dt should change smoothly (no jitter). Doubles the
    forward cost -> enable only in the polish phase. Uses the same (prev,curr) at two phases."""
    b2 = dict(batch); b2['t'] = (batch['t'] + dt).clamp(0.02, 0.98)
    o1, _, _ = forward_batch(model, batch, device)
    o2, _, _ = forward_batch(model, b2, device)
    ui = batch['ui'].to(device); reactive = batch['reactive'].to(device)
    keep = (1 - ui) * (1 - reactive)
    # second difference proxy: change per dt should be bounded/smooth
    return ((o2 - o1).abs() * keep).mean()
