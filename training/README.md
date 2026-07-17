# Plan A — training harness (connection-independent, pausable)

Resumable training runtime for the native learned FG model (docs/plan-a-model-and-training.md). Built
and **proven before the model** (identity-first discipline): `train.py` here is the harness with a
**placeholder** model/step; T2 replaces `build_model` + `train_step` with the real SoftSplat + gated-
conv fusion net and rendered data. Everything else — atomic checkpointing, auto-resume, pause, signal-
graceful stop — is the proven plumbing.

## Runtime properties (verified on the RTX 5080 `ki-pc-fisch-101`)

- **Connection-independent:** runs under `tmux` (session `plana_<run>`); closing SSH never kills it.
- **Atomic checkpoints:** every `--ckpt-every` steps + on signal, temp-file + `os.replace`, saving
  `{model, optimizer, AMP scaler, step, CPU/CUDA/Python RNG}` → `latest.pt` + `ckpt_<step>.pt`.
- **Auto-resume:** on start loads `latest.pt` and continues from the exact step/RNG/optimizer state.
- **Clean pause (exact step, frees the GPU):** `pause` → checkpoints at the current step and idles
  until `resume`. Proven: paused at the exact step, GPU freed, resumed cleanly.
- **Graceful stop:** `SIGTERM`/`SIGUSR1`/`SIGINT` → checkpoint + exit; relaunch auto-resumes (loses at
  most `--ckpt-every` steps).

## Usage (on the server)

```
~/plana_train/ctl.sh launch <run>    # start under tmux
~/plana_train/ctl.sh status <run>    # running/paused + latest step
~/plana_train/ctl.sh pause  <run>    # clean pause (exact-step checkpoint, GPU freed)
~/plana_train/ctl.sh resume <run>    # continue
~/plana_train/ctl.sh stop   <run>    # SIGTERM (checkpoint + exit); relaunch resumes
~/plana_train/ctl.sh attach <run>    # re-attach the tmux session
```

Run dir `~/plana_runs/<run>/`: `latest.pt`, `ckpt_*.pt`, `train.log`, `console.log`, `PAUSE` (control).
venv `~/plana` (PyTorch 2.11 cu128, Blackwell sm_120). Dataloader `num_workers ≤ 3` (box CPU ≤ 20%).
