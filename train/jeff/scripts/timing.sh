#!/bin/sh
# timing.sh — plan M6.3: jefftrain's seconds per optimizer step on gpubox, in the kit's own TIMING line.
#
# Plan M6.3, word for word: "jefftrain on the full train split, --max-steps 100 --batch 8, printing the same TIMING line
# as the kit's timing.sh. Compare with PyTorch's 0.143 s/step and the D§15 target (< 0.4 s/step). Write the number
# down. Being slower than PyTorch is expected for v1 and is not a failure. M8 is where speed gets work."
#
# So this script measures and reports; it does not fail on a slow trainer. What it does check is that a measurement
# happened: the run exited 0, the log prints the compute mode it was asked for (TF32 or exact fp32 — a stale env must
# not quietly time the other mode), and there are step lines to average. Anything else is a broken measurement.
#
# The TIMING line is tools/jeff_lora/cuda/timing.sh's line: its regex (`step N/100  X.XXs`), its steps-1-3 warm-up
# rule, its 28,645-row epoch extrapolation. That is what lets the two numbers be read side by side.
#
# It only orchestrates, on gpubox, inside ~/CLM-8B (plan M6's preamble): the full split is tokenised there by prep.py,
# and the .jtb files, the log and report.json stay in CLM-8B's git-ignored data/ (plan rule 5 — no real data in
# macuda). Nothing is written into macuda, and nothing here asks the card for anything: this is a gpubox run.
#
# Batch 8 only by default, because batch 8 is what design §15 sizes the target for (batch 16 pages WSL to host memory
# and drops to 3.6 s/step). BATCHES="8 16" measures both, each with its own prep and its own run directory.
#
# The full train split matters: jefftrain's step cost depends on the packed batch's token count, and the 4000-row
# sample the kit times is not the distribution the target is stated for. Tokenising those 28,645 rows is prep, and it
# is not part of the TIMING number (the step lines are).
#
# usage (from train/jeff, on the Mac; this script syncs and builds the tree itself):
#     export JT_REMOTE=macuda-train/wt-J34        # parallel tasks must not share one
#     sh scripts/timing.sh
#   env:
#     JT_REMOTE      build tree under the box's ~ (default macuda-train/jeff)
#     JT_SKIP_BUILD  1 = do not sync/build, run the binary that is already there
#     ARCH           the nv build's arch (default sm_120)
#     BATCHES        batch sizes to time (default 8 — design §15's batch)
#     MAX_STEPS      optimizer steps per timing run (default 100, the plan's)
#     DEV_ROWS       rows in the timing run's dev check (default 64, the kit's: the clock is not dominated by it)
#     EXACT          1 = JT_EXACT=1 TINYCUBLAS_TC=0; default 0, the TF32 build the D§15 target is stated for
#     SEED           prep.py's seed (default 0, the kit's)
#     RUN            run directory (default data/ovril_detect/runs/jefftrain-timing-<UTC stamp>, inside CLM-8B)
#     PY_REF         PyTorch's s/step for the comparison (default 0.143, design §15)
#     TARGET         the D§15 target in s/step (default 0.4; reported, not a failure — see above)
#     JT_ENV         "JT_A=1 JT_B=2" adds those settings to the remote jefftrain's environment (default: none).
#                    Every word is checked here, on the Mac, against ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ before it reaches
#                    the ssh command, and anything else is refused.
#     JT_EXPECT      '<regex>' (ERE): the run fails unless jefftrain's log has a matching line — the way the compute
#                    mode is checked below, so an environment toggle that silently did nothing cannot pass (plan rule 9).
# The GPU work runs under gpubox's exclusive lock, `flock ~/macuda-train/.gpu.lock`, like the other gpubox scripts.
set -eu
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
ARCH=${ARCH:-sm_120}
BATCHES=${BATCHES:-8}
MAX_STEPS=${MAX_STEPS:-100}
DEV_ROWS=${DEV_ROWS:-64}
EXACT=${EXACT:-0}
SEED=${SEED:-0}
PY_REF=${PY_REF:-0.143}
TARGET=${TARGET:-0.4}
JT_ENV=${JT_ENV:-} JT_EXPECT=${JT_EXPECT:-}
JEFF=$(cd "$(dirname "$0")/.." && pwd)

# JT_ENV is added to the remote jefftrain's environment, so it is validated here first: one word per setting, and only
# ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ gets through, so a word with a space, ';', '$' or a quote is refused.
check_env() {
  for w in $JT_ENV; do
    printf '%s\n' "$w" | grep -Eq '^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$' || {
      echo "timing: JT_ENV: '$w' is not JT_NAME=value" >&2
      echo "timing: every word must match ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*\$ — no spaces, ';', '\$' or quotes" >&2
      exit 2
    }
  done
}
sq() { printf '%s' "$1" | sed "s/'/'\\\\''/g"; }
check_env
[ "${JT_SKIP_BUILD:-0}" = 1 ] || sh "$JEFF/scripts/gpubox-build.sh"

ssh gpubox "if ! command -v flock >/dev/null 2>&1; then
    echo 'timing: flock is missing on gpubox; it is needed for ~/macuda-train/.gpu.lock' >&2
    exit 2
  fi
  exec flock ~/macuda-train/.gpu.lock env JT_REMOTE='$JT_REMOTE' BATCHES='$BATCHES' MAX_STEPS='$MAX_STEPS' \
  DEV_ROWS='$DEV_ROWS' EXACT='$EXACT' SEED='$SEED' PY_REF='$PY_REF' TARGET='$TARGET' RUN='${RUN:-}' \
  JT_ENV='$(sq "$JT_ENV")' JT_EXPECT='$(sq "$JT_EXPECT")' sh -s" <<'REMOTE'
set -eu
cd "$HOME/CLM-8B"                          # the real splits and the venv live here (plan M6's preamble)
. tools/jeff_lora/cuda/env.sh               # DATA, VENV, REMOTE_MODEL (and the HF/tokenizer env)
BIN="$HOME/$JT_REMOTE/build/nv/jefftrain"
PREP="$HOME/$JT_REMOTE/py/prep.py"
MODEL="$HOME/$REMOTE_MODEL"
PY=$VENV/bin/python
for f in "$BIN" "$PREP" "$MODEL/model.safetensors" "$DATA/train.jsonl" "$DATA/development.jsonl"; do
  [ -e "$f" ] || { echo "timing: missing $f" >&2; exit 2; }
done

RUN=${RUN:-$PWD/data/ovril_detect/runs/jefftrain-timing-$(date -u +%Y%m%d-%H%M%S)}
mkdir -p "$RUN"
ln -sfn "$RUN" "$PWD/data/ovril_detect/runs/jefftrain-timing-latest"   # notes and follow-ups find the newest run
if [ "$EXACT" = 1 ]; then EXACT_ENV="JT_EXACT=1 TINYCUBLAS_TC=0"; WANT="gemm: exact fp32"; else EXACT_ENV=""; WANT="gemm: tf32"; fi
echo "timing: run directory $RUN"
echo "timing: model $MODEL  data $DATA"
echo "timing: $MAX_STEPS steps, batch [$BATCHES], $DEV_ROWS dev rows, seed $SEED, compute '$WANT'"

rc=0
for b in $BATCHES; do
  d=$RUN/b$b
  mkdir -p "$d"
  echo "=== prep: the full train split at --batch $b (tokenising; not timed)"
  "$PY" "$PREP" --clm "$PWD" --model "$MODEL" --train "$DATA/train.jsonl" --dev "$DATA/development.jsonl" \
    --dev-rows "$DEV_ROWS" --targets soft --weights none --batch "$b" --accum 1 --epochs 1 --seed "$SEED" \
    --out-dir "$d/prep"
  echo "=== jefftrain: $MAX_STEPS steps at batch $b -> $d/out (log $d/jefftrain.log)"
  log=$d/jefftrain.log
  if env $EXACT_ENV $JT_ENV "$BIN" --model "$MODEL" --train "$d/prep/train.jtb" --dev "$d/prep/dev.jtb" \
       --output "$d/out" --batch "$b" --accum 1 --epochs 1 --max-steps "$MAX_STEPS" --targets soft \
       --weights none --lr 2e-4 --readout-lr 5e-6 --log-every 1 > "$log" 2>&1; then
    head -n 6 "$log" | sed 's/^/  | /'
    echo "  | ..."
    tail -n 3 "$log" | sed 's/^/  | /'
    grep -qxF "$WANT" "$log" || { echo "timing: batch $b: expected '$WANT' on stdout, got:" >&2
                                  sed 's/^/  | /' "$log" >&2; rc=1; }
    if [ -n "$JT_EXPECT" ] && ! grep -Eq -- "$JT_EXPECT" "$log"; then
      echo "timing: batch $b: JT_EXPECT '$JT_EXPECT' matched no line of $log" >&2
      echo "timing: batch $b: jefftrain did not run the way it was asked to" >&2
      rc=1
    fi
    "$PY" - "$log" "$b" "$d/out/report.json" "$PY_REF" "$TARGET" <<'PY' || rc=1
import json, re, sys

log, batch, report, py_ref, target = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
steps = [(int(n), float(t)) for n, t in re.findall(r"step (\d+)/\d+\s+([\d.]+)s", open(log).read())]
s = [t for n, t in steps if n > 3]   # steps 1-3 carry compile and allocator warm-up (tools/jeff_lora/cuda/timing.sh)
if not s:
    print(f"TIMING batch {batch}: no step lines to average (log {log})")
    raise SystemExit(1)
per = sum(s) / len(s)
print(f"TIMING batch {batch}: {per:.3f} s/step ({len(s)} samples) -> {batch / per:.1f} rows/s -> "
      f"one epoch of 28645 rows ~{28645 / batch * per / 60:.0f} min + dev checks")
try:
    rep = json.load(open(report))
    print(f"TIMING batch {batch}: peak {rep['peak_gb']:.2f} GB  {rep['steps']} step(s) at "
          f"{rep['rows_per_second']} rows/s")
except (OSError, KeyError, ValueError) as e:
    print(f"TIMING batch {batch}: no report.json numbers ({e})")
    raise SystemExit(1)
print(f"timing: batch {batch}: {per:.3f} s/step vs PyTorch {py_ref:.3f} ({per / py_ref:.2f}x), "
      f"D§15 target < {target:.3f} -> "
      + ("within target" if per < target
         else f"OVER target by {per - target:.3f} s/step (v1: expected; M8 is the speed milestone)"))
PY
  else
    sed 's/^/  | /' "$log" | tail -n 20
    echo "timing: batch $b exited non-zero (log $log)" >&2
    rc=1
  fi
done
echo "timing: logs and report.json in ~/CLM-8B/$RUN (runs/jefftrain-timing-latest)"
if [ "$rc" = 0 ]; then
  echo "timing: PASS — the TIMING numbers above are plan M6.3's, for Appendix B"
else
  echo "timing: FAIL — the measurement did not complete (see the lines above)"
fi
exit "$rc"
REMOTE
