"""A2 sub-pixel alignment gate (Plan A T1) — the correctness test that MUST pass before mass capture.

A silent sub-pixel offset between the captured GT frame N and what the model reconstructs from
N-1/N+1 (mismatched jitter state, MV convention, or HUD handling) poisons the loss: the net learns
to compensate the offset instead of interpolating, and it only shows as bad generalisation. This
gate measures the *systematic* shift between the warp candidate and the capture-GT and fails if it
exceeds a sub-pixel threshold — run it on a sample of the capture before generating tens of
thousands of triplets.

Proven here on synthetic data: aligned data passes; an injected 0.4 px offset is caught.
"""
import torch
import torch.nn.functional as F


def _shift(img, dx, dy):
    """Sub-pixel translate img by (dx,dy) px via bilinear grid_sample."""
    B, C, H, W = img.shape
    dev = img.device
    yy, xx = torch.meshgrid(torch.arange(H, device=dev), torch.arange(W, device=dev), indexing='ij')
    gx = (xx.float() + dx) / (W - 1) * 2 - 1
    gy = (yy.float() + dy) / (H - 1) * 2 - 1
    grid = torch.stack([gx, gy], -1)[None].expand(B, H, W, 2)
    return F.grid_sample(img, grid, align_corners=True, padding_mode='border')


def measure_offset(cand, gt, valid=None, rng=1.5, step=0.1, border=8, trim=0.2):
    """Global sub-pixel shift d that best maps cand onto gt.

    A capture-convention offset shifts the WHOLE frame systematically; disocclusion holes, moving
    hard edges and reactive/particle pixels are LOCAL and BIASED (the candidate has the wrong content
    there no matter how well the capture is aligned), so a plain mean-L1 fit is pulled off by them.
    We measure the shift only on the reliably-reconstructable majority:
      * `valid` (optional [B,1,H,W] in {0,1}) drops known-invalid pixels (UI, reactive/particles);
      * a trimmed mean drops the worst `trim` fraction of the remaining residuals (disocclusion/edges).
    This is strictly more correct for real captures too — you never judged alignment on pixels that
    legitimately can't be reconstructed."""
    c = cand[..., border:-border, border:-border]
    v = None if valid is None else valid[..., border:-border, border:-border]
    # brightness-invariant: per-frame shading/exposure is MULTIPLICATIVE and would otherwise let the
    # best-fit shift move to compensate a brightness mismatch (a false offset). Normalize each frame
    # by its own mean so the measure sees only the geometric shift. Correct for real captures too
    # (exposure/auto-brightness changes between frames must not read as misalignment).
    cn = c / c.mean(dim=[2, 3], keepdim=True).clamp(min=1e-4)
    best = (0.0, 0.0); best_e = float('inf')
    ds = torch.arange(-rng, rng + 1e-6, step).tolist()
    for dy in ds:
        for dx in ds:
            g = _shift(gt, dx, dy)[..., border:-border, border:-border]
            g = g / g.mean(dim=[2, 3], keepdim=True).clamp(min=1e-4)
            err = (cn - g).abs().mean(1, keepdim=True)          # per-pixel, mean over channels
            if v is not None:
                err = err[v > 0.5]
            else:
                err = err.reshape(-1)
            if err.numel() == 0:
                continue
            if 0.0 < trim < 1.0 and err.numel() > 16:          # robust: keep the best (1-trim) fraction
                k = int(err.numel() * (1.0 - trim))
                err = torch.topk(err, k, largest=False).values
            e = err.mean().item()
            if e < best_e:
                best_e, best = e, (dx, dy)
    return best, best_e


def alignment_gate(cand, gt, valid=None, tol=0.15):
    """Returns (passed, offset_px, magnitude). Fail if the systematic shift exceeds `tol` px.
    `valid` optionally excludes UI/reactive pixels (see measure_offset)."""
    (dx, dy), _ = measure_offset(cand, gt, valid=valid)
    mag = (dx * dx + dy * dy) ** 0.5
    return mag <= tol, (dx, dy), mag


# --- self-test: prove the gate catches a sub-pixel offset -------------------------------------
if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    gt = torch.rand(1, 3, 128, 128, device=dev)

    # aligned: candidate == GT (plus mild local noise mimicking disocclusion) -> must PASS
    cand_ok = gt + 0.02 * torch.rand_like(gt)
    p, off, m = alignment_gate(cand_ok, gt)
    print(f"aligned      -> pass={p}  offset={off[0]:+.2f},{off[1]:+.2f}  |{m:.2f}px|")

    # misaligned: candidate is GT shifted by 0.4 px (a capture offset) -> must FAIL
    cand_off = _shift(gt, 0.4, 0.0) + 0.02 * torch.rand_like(gt)
    p2, off2, m2 = alignment_gate(cand_off, gt)
    print(f"0.4px offset -> pass={p2}  offset={off2[0]:+.2f},{off2[1]:+.2f}  |{m2:.2f}px|")

    assert p and not p2, "alignment gate does not discriminate!"
    print("GATE OK: passes aligned, catches sub-pixel offset")
