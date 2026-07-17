#!/usr/bin/env python
"""Plan A resumable training harness (connection-independent, pausable).

T2 replaces `build_model` + `train_step` with the real SoftSplat+fusion net and rendered data.
Everything else — atomic checkpointing, auto-resume, pause file, signal-graceful stop — is the
provable-now plumbing. Run under tmux so SSH disconnect never kills it.
"""
import argparse, itertools, os, random, signal, sys, time
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model, train_step, forward_batch   # real model (T2)
import data, align                                          # loader + alignment gate (T1)
import losses                                                # §21.6 composite loss (T3)


def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic rename on the same filesystem


class Trainer:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.run_dir, exist_ok=True)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = build_model(a.mode).to(self.dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr)
        self.scaler = torch.amp.GradScaler(self.dev, enabled=(self.dev == 'cuda'))
        self.step = 0
        self.stop = False
        self.data_iter = None; self.criterion = None
        if a.data_dir:                                        # real rendered triplets (T1)
            loader = data.make_loader(a.data_dir, batch=a.batch, target_fps=a.target_fps, workers=3)
            self._gate(loader)                                # MUST pass before training on real data
            self.data_iter = itertools.cycle(loader)
            self.criterion = losses.CompositeLoss(self.dev)   # §21.6 (T3)
            self._log(f"[data] {len(loader.dataset)} triplets from {a.data_dir}")
        else:
            self._log("[data] no --data-dir -> synthetic self-supervised batch (pipeline validation)")
        self._resume()
        for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(s, self._on_signal)

    def _gate(self, loader):
        # alignment gate: the warp candidate must land on the capture-GT to sub-pixel, else the
        # capture has a silent offset that would poison the loss. Abort rather than train on it.
        self.model.eval()
        with torch.no_grad():
            _, cand, target = forward_batch(self.model, next(iter(loader)), self.dev)
            ok, off, mag = align.alignment_gate(cand.float(), target.float())
        self.model.train()
        self._log(f"[align-gate] offset={off[0]:+.2f},{off[1]:+.2f}px |{mag:.2f}| -> {'PASS' if ok else 'FAIL'}")
        if not ok:
            self._log("[align-gate] FAILED: capture is sub-pixel-misaligned; fix the harness before training")
            raise SystemExit(2)

    def _ckpt(self, name):
        return os.path.join(self.a.run_dir, name)

    def _resume(self):
        p = self._ckpt("latest.pt")
        if os.path.exists(p):
            # load to CPU: keeps RNG ByteTensors on CPU (set_rng_state requires that); model/opt
            # move to the device explicitly below.
            st = torch.load(p, map_location='cpu', weights_only=False)
            self.model.load_state_dict(st['model'])
            self.opt.load_state_dict(st['opt'])
            for s in self.opt.state.values():                     # optimizer moments -> device
                for k, v in s.items():
                    if isinstance(v, torch.Tensor):
                        s[k] = v.to(self.dev)
            self.scaler.load_state_dict(st['scaler']); self.step = st['step']
            torch.set_rng_state(st['rng_cpu']); random.setstate(st['rng_py'])
            if self.dev == 'cuda' and st.get('rng_cuda') is not None:
                torch.cuda.set_rng_state_all(st['rng_cuda'])
            self._log(f"[resume] continuing from step {self.step}")
        else:
            self._log("[resume] no checkpoint -> fresh start")

    def _save(self, snapshot=False):
        st = {'model': self.model.state_dict(), 'opt': self.opt.state_dict(),
              'scaler': self.scaler.state_dict(), 'step': self.step,
              'rng_cpu': torch.get_rng_state(), 'rng_py': random.getstate(),
              'rng_cuda': (torch.cuda.get_rng_state_all() if self.dev == 'cuda' else None)}
        atomic_save(st, self._ckpt("latest.pt"))
        if snapshot:
            atomic_save(st, self._ckpt(f"ckpt_{self.step:09d}.pt"))
        self._log(f"[ckpt] saved at step {self.step}")

    def _on_signal(self, sig, frame):
        self._log(f"[signal] {signal.Signals(sig).name} -> checkpoint & stop")
        self.stop = True

    def _paused(self):
        return os.path.exists(self._ckpt("PAUSE"))

    def _log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(self._ckpt("train.log"), "a") as f:
            f.write(line + "\n")

    def train(self):
        self.model.train()
        while self.step < self.a.max_steps and not self.stop:
            if self._paused():
                self._save(); self._log("[pause] PAUSE present -> idle (GPU freed)")
                while self._paused() and not self.stop:
                    time.sleep(3)
                if not self.stop:
                    self._log("[resume] PAUSE cleared -> continue")
                continue
            comps = None
            if self.data_iter is not None:                    # real triplets + §21.6 losses
                batch = next(self.data_iter)
                out, _, target = forward_batch(self.model, batch, self.dev)
                loss, comps = self.criterion(out, target, batch)
                if self.a.temporal > 0:                       # §21.6 temporal-stability (flicker)
                    tl = losses.temporal_pair_loss(self.model, batch, self.dev, forward_batch)
                    loss = loss + self.a.temporal * tl; comps['temporal'] = tl.item()
            else:
                loss = train_step(self.model, self.dev)       # synthetic pipeline validation
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt); self.scaler.update()
            self.step += 1
            if self.step % self.a.log_every == 0:
                extra = (" " + " ".join(f"{k}={v:.3f}" for k, v in comps.items())) if comps else ""
                self._log(f"[step {self.step}] loss {loss.item():.4f}{extra}")
            if self.step % self.a.ckpt_every == 0:
                self._save(snapshot=True)
        self._save()
        self._log(f"[done] step={self.step} stop={self.stop}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', default=os.path.expanduser('~/plana_runs/run1'))
    p.add_argument('--max-steps', type=int, default=1_000_000)
    p.add_argument('--ckpt-every', type=int, default=500)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--mode', default='interp', choices=['interp','extrap'])
    p.add_argument('--data-dir', default=None, help='rendered-triplet root (A2); omit -> synthetic')
    p.add_argument('--target-fps', type=int, default=60)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--temporal', type=float, default=0.0, help='temporal-stability weight (0=off)')
    Trainer(p.parse_args()).train()


if __name__ == '__main__':
    main()
