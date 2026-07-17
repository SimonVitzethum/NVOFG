"""Plan A A5 — export trained weights to the runtime format the recordCnnRefine backend loads.

Format `.nvfgw` (little-endian):
  magic   8 : b'NVFGW\x01\x00\x00'
  header  16: u32 version, u32 arch_base, u32 arch_cin, u32 mode (0=interp,1=extrap)
  u32 n_tensors
  per tensor: u32 name_len, name bytes, u32 ndim, u32 dims[ndim], fp16 data[prod(dims)]

Tensors are in model.state_dict() order (deterministic). The C++/coopmat backend reads them by name
into its conv weights. Deploys next to the app; the A1 identity backend is replaced by this + the
inference kernels (B4). Round-trip verified here (export -> reload -> bit-identical fp16 state).
"""
import argparse, os, struct, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, FrameGenModel

MAGIC = b'NVFGW\x01\x00\x00'


def export(ckpt, out_path, mode='interp', base=24):
    m = build_model(mode)
    if ckpt and os.path.exists(ckpt):
        st = torch.load(ckpt, map_location='cpu', weights_only=False)
        m.load_state_dict(st['model'])
    sd = m.state_dict()
    with open(out_path, 'wb') as f:
        f.write(MAGIC)
        f.write(struct.pack('<IIII', 1, base, 12, 0 if mode == 'interp' else 1))
        f.write(struct.pack('<I', len(sd)))
        for name, t in sd.items():
            t = t.detach().cpu().to(torch.float16).contiguous()
            nb = name.encode()
            f.write(struct.pack('<I', len(nb))); f.write(nb)
            f.write(struct.pack('<I', t.dim()))
            for d in t.shape:
                f.write(struct.pack('<I', d))
            f.write(t.numpy().tobytes())
    n = sum(p.numel() for p in m.parameters())
    print(f"exported {len(sd)} tensors ({n/1e6:.3f}M params) -> {out_path} ({os.path.getsize(out_path)} B)")
    return sd


def load_back(path):
    sd = {}
    with open(path, 'rb') as f:
        assert f.read(8) == MAGIC, "bad magic"
        version, base, cin, mode = struct.unpack('<IIII', f.read(16))
        (n,) = struct.unpack('<I', f.read(4))
        import numpy as np
        for _ in range(n):
            (nl,) = struct.unpack('<I', f.read(4)); name = f.read(nl).decode()
            (nd,) = struct.unpack('<I', f.read(4)); dims = struct.unpack(f'<{nd}I', f.read(4 * nd))
            cnt = 1
            for d in dims:
                cnt *= d
            arr = np.frombuffer(f.read(cnt * 2), dtype=np.float16).reshape(dims)
            sd[name] = torch.from_numpy(arr.copy())
    return sd, dict(version=version, base=base, cin=cin, mode=mode)


def verify(ckpt, mode='interp', base=24):
    tmp = '/tmp/_nvfgw_test.nvfgw'
    ref = export(ckpt, tmp, mode, base)
    got, hdr = load_back(tmp)
    ok = (set(got) == set(ref)) and all(torch.equal(got[k], ref[k].to(torch.float16)) for k in ref)
    # reload into a fresh model to confirm it actually loads
    m2 = build_model(mode)
    m2.load_state_dict({k: got[k].float() for k in got})
    os.remove(tmp)
    print(f"round-trip: {'OK (bit-identical fp16, loads into model)' if ok else 'FAILED'}  header={hdr}")
    return ok


def _save_t(path, t):
    import numpy as np
    t = t.detach().cpu().to(torch.float16).contiguous()
    with open(path, 'wb') as f:
        f.write(struct.pack('<I', t.dim()))
        for d in t.shape:
            f.write(struct.pack('<I', d))
        f.write(t.numpy().tobytes())


def export_golden(ckpt, prefix, mode='interp', base=24, H=32, W=32):
    """Golden test for the C++/GPU inference: a fixed fusion input x (12,H,W) and the fusion residual
    output y (3,H,W), so the backend's forward can be checked against PyTorch bit-close."""
    torch.manual_seed(0)
    m = build_model(mode)
    if ckpt and os.path.exists(ckpt):
        m.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    m.eval()
    x = torch.rand(1, 12, H, W)
    with torch.no_grad():
        y = m.fusion(x)
    _save_t(prefix + '_x.bin', x); _save_t(prefix + '_y.bin', y)
    print(f"golden: x{tuple(x.shape)} y{tuple(y.shape)} -> {prefix}_x.bin {prefix}_y.bin")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default=None)
    p.add_argument('--out', default='fg_v1.nvfgw')
    p.add_argument('--mode', default='interp')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--golden', default=None, help='prefix -> <prefix>.nvfgw + _x/_y.bin')
    a = p.parse_args()
    if a.verify:
        assert verify(a.ckpt, a.mode), "round-trip failed"
    elif a.golden:
        export(a.ckpt, a.golden + '.nvfgw', a.mode)
        export_golden(a.ckpt, a.golden, a.mode)
    else:
        export(a.ckpt, a.out, a.mode)
