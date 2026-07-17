#!/usr/bin/env python
"""Plan A resumable training harness (connection-independent, pausable).

T2 replaces `build_model` + `train_step` with the real SoftSplat+fusion net and rendered data.
Everything else — atomic checkpointing, auto-resume, pause file, signal-graceful stop — is the
provable-now plumbing. Run under tmux so SSH disconnect never kills it.
"""
import argparse, glob, os, random, signal, time
import torch
import torch.nn as nn


def atomic_save(state, path):
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic rename on the same filesystem


# --- T2 replaces these two ---------------------------------------------------
def build_model():
    # placeholder ~ small conv net; the real ~1M SoftSplat+gated-conv fusion net lands in T2
    return nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                         nn.Conv2d(32, 3, 3, padding=1))

def train_step(model, device):
    time.sleep(0.003)                                      # placeholder throttle (real step is ~ms); remove in T2
    x = torch.randn(8, 3, 64, 64, device=device)          # placeholder batch (T2: real triplets)
    y = model(x)
    return (y - x).abs().mean()                            # placeholder loss (T2: §21.6 losses)
# -----------------------------------------------------------------------------


class Trainer:
    def __init__(self, a):
        self.a = a
        os.makedirs(a.run_dir, exist_ok=True)
        self.dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = build_model().to(self.dev)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=a.lr)
        self.scaler = torch.amp.GradScaler(self.dev, enabled=(self.dev == 'cuda'))
        self.step = 0
        self.stop = False
        self._resume()
        for s in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
            signal.signal(s, self._on_signal)

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
            loss = train_step(self.model, self.dev)
            self.opt.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt); self.scaler.update()
            self.step += 1
            if self.step % self.a.log_every == 0:
                self._log(f"[step {self.step}] loss {loss.item():.4f}")
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
    Trainer(p.parse_args()).train()


if __name__ == '__main__':
    main()
