#!/usr/bin/env bash
# Plan A training control (connection-independent via tmux; pausable via PAUSE file).
# usage: ctl.sh {launch|pause|resume|stop|status|attach} [run]
set -uo pipefail
RUN="${2:-run1}"
RUNDIR="$HOME/plana_runs/$RUN"
PY="$HOME/plana/bin/python"
TRAIN="$HOME/plana_train/train.py"
SESSION="plana_$RUN"
mkdir -p "$RUNDIR"

running() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "${1:-status}" in
  launch)
    if ! command -v tmux >/dev/null; then echo "tmux missing"; exit 1; fi
    if running; then echo "already running (tmux:$SESSION)"; exit 0; fi
    rm -f "$RUNDIR/PAUSE"
    EXTRA="${*:3}"                                  # extra train.py args after the run name
    [ -n "$EXTRA" ] && echo "$EXTRA" > "$RUNDIR/args"      # remember them for relaunch/resume
    EXTRA="$(cat "$RUNDIR/args" 2>/dev/null)"
    tmux new-session -d -s "$SESSION" "$PY -u $TRAIN --run-dir $RUNDIR $EXTRA >> $RUNDIR/console.log 2>&1"
    echo "launched tmux:$SESSION -> $RUNDIR  args:[$EXTRA]" ;;
  pause)   touch "$RUNDIR/PAUSE"; echo "PAUSE set (checkpoints, idles, frees GPU)" ;;
  resume)  rm -f "$RUNDIR/PAUSE"; echo "PAUSE cleared (continues)" ;;
  stop)    pkill -TERM -f "train.py --run-dir $RUNDIR" && echo "SIGTERM sent (checkpoints, exits; relaunch auto-resumes)" || echo "no process" ;;
  status)
    running && echo "process: RUNNING (tmux:$SESSION)" || echo "process: stopped"
    [ -f "$RUNDIR/PAUSE" ] && echo "state: PAUSED" || echo "state: active"
    echo "latest snapshot: $(ls -1 "$RUNDIR"/ckpt_*.pt 2>/dev/null | tail -1 | xargs -r basename)"
    echo "--- last log ---"; tail -4 "$RUNDIR/train.log" 2>/dev/null ;;
  attach)  tmux attach -t "$SESSION" ;;
  *) echo "usage: $0 {launch|pause|resume|stop|status|attach} [run]"; exit 1 ;;
esac
