#!/bin/sh
# parity-steps.sh — plan M6.2: 20 steps of the real trainer against CLM-8B's own train.py, on gpubox.
#
# The four pieces the plan names, in its order, in one run directory:
#
#   1. py/make_adapter.py writes the synthetic adapter A (design §10.1: random A, B*1e-2, the model's readout) that
#      both trainers load with --init-adapter, so the LoRA path is neither zero nor freshly attached;
#   2. py/prep.py tokenises the same rows with the same RNG recipe and writes train.jtb/dev.jtb (plan M5.3) — the C
#      trainer walks those batches, and train.py walks the .jsonl splits with the identical order;
#   3. py/trace_train_loss.py runs train.py in fp32 for 20 steps and records its per-row CE per optimizer step plus the
#      dev checks from report.json;
#   4. jefftrain runs the same 20 steps over the .jtb files with --log-every 1 --trace-loss steps.jsonl.
#
# Then py/compare_steps.py applies design §10.3.2's gate and prints the verdict line this script exits on.
#
# Everything reads real data, so this runs **on gpubox inside CLM-8B** (plan M6's preamble, rule 5): the outputs go to
# CLM-8B's own data/ovril_detect/runs/parity-steps-<stamp>/ and the trace stays there as steps.jsonl, which plan M7.3's
# card run compares against. Nothing is written into macuda except this tree's build/.
#
# The C side runs in exact mode by default (JT_EXACT=1 TINYCUBLAS_TC=0) because the Python side is true fp32 (torch's
# allow_tf32 for matmul is off) and because that is the mode the card checkpoints use; EXACT=0 gives the TF32 build for a
# looser, default-precision look at the same trace.
#
# usage (from train/jeff, on the Mac or on gpubox):
#     JT_REMOTE=macuda-train/wt-J33 sh scripts/parity-steps.sh
#   env: JT_REMOTE (parallel tasks must not share one), CLM, DATA, MODEL, RUNS, ADAPTER, RUN, EXACT=0, ARCH, MAX_STEPS,
#        BATCH, ACCUM, EPOCHS, TRAIN_ROWS, DEV_ROWS, EVAL_EVERY, SEED, LR, READOUT_LR, ADAPTER_SEED,
#        JT_ENV="JT_A=1 JT_B=2" (added to the C trainer's environment; every word is checked here, on the Mac, against
#        ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ before it reaches the ssh command and anything else is refused),
#        JT_EXPECT='<regex>' (ERE: the run fails unless jefftrain's log has a matching line, so an environment toggle
#        that silently did nothing cannot pass — plan rule 9). The C step's output goes to the screen as before and is
#        also kept in the run directory's c.log, which is the file JT_EXPECT is matched against.
# The GPU work (the C step and the Python trace, both on the card) runs under gpubox's exclusive lock,
# `flock ~/macuda-train/.gpu.lock`, like the other gpubox scripts.
# Nothing here asks the card for anything: this is a gpubox run (plan rule 4, M6's preamble).
set -eu
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
JEFF=$(cd "$(dirname "$0")/.." && pwd)
CLM=${CLM:-$HOME/CLM-8B}
JT_ENV=${JT_ENV:-} JT_EXPECT=${JT_EXPECT:-}

# JT_ENV is added to the C trainer's environment, so it is checked before anything is forwarded: one word per setting,
# and only ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ gets through, so a word with a space, ';', '$' or a quote is refused.
for w in $JT_ENV; do
  printf '%s\n' "$w" | grep -Eq '^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$' || {
    echo "parity-steps: JT_ENV: '$w' is not JT_NAME=value" >&2
    echo "parity-steps: every word must match ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*\$ — no spaces, ';', '\$' or quotes" >&2
    exit 2
  }
done

# Real data and CLM-8B's venv live on gpubox, so when this checkout is not next to them the run is handed over there.
# The tunables set here are forwarded, so a run started from the Mac gets the steps and paths it asked for.
# CLM is deliberately not forwarded: it is what decided to hop in the first place, and gpubox's ~/CLM-8B is the one.
if [ ! -f "$CLM/tools/jeff_lora/train.py" ]; then
  echo "parity-steps: no CLM-8B at $CLM; running on gpubox:~/$JT_REMOTE"
  envs="JT_REMOTE='$JT_REMOTE'"
  for name in DATA MODEL PY PYC RUNS ADAPTER RUN EXACT ARCH MAX_STEPS BATCH ACCUM EPOCHS \
              TRAIN_ROWS DEV_ROWS EVAL_EVERY SEED LR READOUT_LR ADAPTER_SEED JT_ENV JT_EXPECT; do
    eval "val=\${$name-}"
    [ -n "${val-}" ] || continue
    envs="$envs $name='$(printf '%s' "$val" | sed "s/'/'\\\\''/g")'"
  done
  exec ssh gpubox "cd ~/$JT_REMOTE && $envs sh scripts/parity-steps.sh"
fi

# The GPU lock every gpubox script shares, held for the whole run: the Python trace and the C step both use the card.
# The re-exec keeps this script's own interface, so nothing below has to care whether it took the lock itself.
if [ "${JT_LOCKED:-0}" != 1 ]; then
  command -v flock >/dev/null 2>&1 || {
    echo "parity-steps: flock is missing on gpubox; it is needed for ~/macuda-train/.gpu.lock" >&2
    exit 2
  }
  export JT_LOCKED=1
  exec flock ~/macuda-train/.gpu.lock sh "$0" "$@"
fi
unset JT_LOCKED

DATA=${DATA:-$CLM/data/ovril_detect/out}
MODEL=${MODEL:-$HOME/models/jeff-base}
PY=${PY:-$CLM/.venv-cuda/bin/python}
PYC=${PYC:-python3}
RUNS=${RUNS:-$CLM/data/ovril_detect/runs}
ARCH=${ARCH:-sm_120}
EXACT=${EXACT:-1}
BATCH=${BATCH:-8} ACCUM=${ACCUM:-1} EPOCHS=${EPOCHS:-1}
TRAIN_ROWS=${TRAIN_ROWS:-256} DEV_ROWS=${DEV_ROWS:-64}
MAX_STEPS=${MAX_STEPS:-20} EVAL_EVERY=${EVAL_EVERY:-20}
LR=${LR:-2e-4} READOUT_LR=${READOUT_LR:-5e-6} SEED=${SEED:-0} ADAPTER_SEED=${ADAPTER_SEED:-1}
RUN=${RUN:-$RUNS/parity-steps-$(date +%Y%m%d-%H%M%S)}
ADAPTER=${ADAPTER:-$RUN/adapter}

export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p "$RUN"
ln -sfn "$RUN" "$RUNS/parity-steps-latest"   # G-m73 and M7.3 find the newest trace without knowing the stamp
echo "parity-steps: run directory $RUN"
echo "parity-steps: clm $CLM  data $DATA  model $MODEL  exact $EXACT"
echo "parity-steps: $MAX_STEPS steps of batch $BATCH x accum $ACCUM, $TRAIN_ROWS train rows, $DEV_ROWS dev rows, seed $SEED"

# ---- build (it is a no-op when build/nv is current) ---------------------------------------------------------------
export PATH=/usr/local/cuda/bin:$PATH
echo "=== build: make nv ARCH=$ARCH"
if ! make -C "$JEFF" nv ARCH="$ARCH" >"$RUN/build.log" 2>&1; then
  tail -20 "$RUN/build.log"
  echo "parity-steps: FAIL — the build failed (full log $RUN/build.log)"
  exit 1
fi
tail -2 "$RUN/build.log"

# ---- 1. the adapter both sides load ------------------------------------------------------------------------------
if [ -f "$ADAPTER/adapter_model.safetensors" ]; then
  echo "=== adapter: reusing $ADAPTER"
else
  echo "=== adapter: py/make_adapter.py -> $ADAPTER"
  "$PY" "$JEFF/py/make_adapter.py" --clm "$CLM" --model "$MODEL" --seed "$ADAPTER_SEED" --out "$ADAPTER"
fi

# ---- 2. the rows, tokenised and batched by prep.py ---------------------------------------------------------------
echo "=== prep: py/prep.py -> $RUN/prep"
"$PY" "$JEFF/py/prep.py" --clm "$CLM" --model "$MODEL" \
  --train "$DATA/train.jsonl" --dev "$DATA/development.jsonl" --out-dir "$RUN/prep" \
  --batch "$BATCH" --accum "$ACCUM" --epochs "$EPOCHS" --train-rows "$TRAIN_ROWS" --dev-rows "$DEV_ROWS" \
  --targets soft --weights none --seed "$SEED"

# ---- 3. Python: train.py's per-step per-row CE -------------------------------------------------------------------
echo "=== python: py/trace_train_loss.py (train.py, fp32) -> $RUN/py.json"
t0=$(date +%s)
"$PY" "$JEFF/py/trace_train_loss.py" --clm "$CLM" --model "$MODEL" \
  --train "$DATA/train.jsonl" --dev "$DATA/development.jsonl" --init-adapter "$ADAPTER" \
  --out "$RUN/py-out" --trace "$RUN/py.json" \
  --batch "$BATCH" --accum "$ACCUM" --epochs "$EPOCHS" --train-rows "$TRAIN_ROWS" --dev-rows "$DEV_ROWS" \
  --max-steps "$MAX_STEPS" --eval-every "$EVAL_EVERY" --lr "$LR" --readout-lr "$READOUT_LR" \
  --device cuda --seed "$SEED" --targets soft --weights none
echo "=== python: $(( $(date +%s) - t0 ))s"

# ---- 4. C: the same steps through the same batches --------------------------------------------------------------
if [ "$EXACT" = 1 ]; then EXACT_ENV="JT_EXACT=1 TINYCUBLAS_TC=0"; else EXACT_ENV=""; fi
echo "=== c: jefftrain ($EXACT_ENV) -> $RUN/steps.jsonl"
t0=$(date +%s)
log=$RUN/c.log
crc=0
env $EXACT_ENV $JT_ENV "$JEFF/build/nv/jefftrain" --model "$MODEL" --init-adapter "$ADAPTER" \
  --train "$RUN/prep/train.jtb" --dev "$RUN/prep/dev.jtb" --output "$RUN/c" \
  --lr "$LR" --readout-lr "$READOUT_LR" --batch "$BATCH" --accum "$ACCUM" --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" --eval-every "$EVAL_EVERY" --targets soft --weights none \
  --log-every 1 --trace-loss "$RUN/steps.jsonl" > "$log" 2>&1 || crc=$?
cat "$log"
if [ "$crc" -ne 0 ]; then
  echo "parity-steps: jefftrain exited $crc (log $log)" >&2
  echo "parity-steps: FAIL — the C step did not finish"
  exit "$crc"
fi
if [ -n "$JT_EXPECT" ] && ! grep -Eq -- "$JT_EXPECT" "$log"; then
  echo "parity-steps: JT_EXPECT '$JT_EXPECT' matched no line of $log" >&2
  echo "parity-steps: FAIL — jefftrain did not run the way it was asked to"
  exit 1
fi
echo "=== c: $(( $(date +%s) - t0 ))s"

# ---- 5. design §10.3.2's gate -------------------------------------------------------------------------------------
echo "=== compare: py/compare_steps.py"
rc=0
"$PYC" "$JEFF/py/compare_steps.py" "$RUN/py.json" "$RUN/steps.jsonl" --c-report "$RUN/c/report.json" || rc=$?
echo "parity-steps: trace $RUN/steps.jsonl  (python $RUN/py.json, report $RUN/c/report.json)"
if [ "$rc" -eq 0 ]; then
  echo "parity-steps: PASS — M6.2's gate holds (see the table above)"
else
  echo "parity-steps: FAIL — M6.2's gate does not hold (exit $rc)"
fi
exit "$rc"
