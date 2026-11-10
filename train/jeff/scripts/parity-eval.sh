#!/bin/sh
# parity-eval.sh — plan M6.1, eval parity: jefftrain scores the 160 real test rows twice (the TF32 mode and the exact
# mode) and both prediction files go through CLM-8B's own tools/jeff_lora/cuda/compare_preds.py against the PyTorch
# port's cuda-parity-fp32 run, with the gate that port passed (max |dp| <= 0.05, argmax >= 158/160).
#
# This script only orchestrates on gpubox, inside ~/CLM-8B (plan rule 5: the real data, the .jtb and the predictions
# stay there). It never copies data into macuda and writes no row content anywhere.
#   JT_REMOTE=macuda-train/wt-J32 bash scripts/parity-eval.sh
#
# The binary and prep.py come from the same tree as the other gpubox scripts: ~/$JT_REMOTE/build/nv/jefftrain and
# ~/$JT_REMOTE/py/prep.py. The script syncs and builds that tree first (scripts/gpubox-build.sh), so the numbers are
# for the working copy it was called in; JT_SKIP_BUILD=1 runs an already synced binary.
#
# Env:
#   JT_REMOTE      build tree under the box's ~ (default macuda-train/jeff; parallel tasks need their own)
#   JT_SKIP_BUILD  1 = do not sync/build, just run (the binary must already be there)
#   JT_PARITY_TS   suffix for the run directory (default: UTC timestamp); outputs go to
#                  ~/CLM-8B/data/ovril_detect/runs/jefftrain-parity-<JT_PARITY_TS>/<tf32|exact>/
#   JT_REF_PREDS   the reference pred-test.jsonl (default: the newest runs/cuda-parity-fp32-*/pred-test.jsonl)
#   JT_MAX_DP      gate: max |dp| (default 0.05 — the plan's; do not loosen it)
#   JT_MIN_AGREE   gate: argmax agreement out of 160 (default 158 — likewise)
#   JT_ENV         "JT_A=1 JT_B=2" adds those settings to the remote jefftrain's environment (default: none). Every
#                  word is checked here, on the Mac, against ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ before it reaches the ssh
#                  command, and anything else is refused.
#   JT_EXPECT      '<regex>' (ERE): a mode fails unless jefftrain's log has a matching line — the way the compute
#                  mode is checked already, so an environment toggle that silently did nothing cannot pass (rule 9).
# The GPU work runs under gpubox's exclusive lock, `flock ~/macuda-train/.gpu.lock`, like the other gpubox scripts.
set -eu
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
MAXDP=${JT_MAX_DP:-0.05}
MINAGREE=${JT_MIN_AGREE:-158}
JT_ENV=${JT_ENV:-} JT_EXPECT=${JT_EXPECT:-}
JEFF=$(cd "$(dirname "$0")/.." && pwd)

# JT_ENV is added to the remote jefftrain's environment, so it is validated here first: one word per setting, and only
# ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ gets through, so a word with a space, ';', '$' or a quote is refused.
check_env() {
  for w in $JT_ENV; do
    printf '%s\n' "$w" | grep -Eq '^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$' || {
      echo "parity-eval: JT_ENV: '$w' is not JT_NAME=value" >&2
      echo "parity-eval: every word must match ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*\$ — no spaces, ';', '\$' or quotes" >&2
      exit 2
    }
  done
}
sq() { printf '%s' "$1" | sed "s/'/'\\\\''/g"; }
check_env
[ "${JT_SKIP_BUILD:-0}" = 1 ] || sh "$JEFF/scripts/gpubox-build.sh"

ssh gpubox "if ! command -v flock >/dev/null 2>&1; then
    echo 'parity-eval: flock is missing on gpubox; it is needed for ~/macuda-train/.gpu.lock' >&2
    exit 2
  fi
  exec flock ~/macuda-train/.gpu.lock env JT_REMOTE='$JT_REMOTE' MAXDP='$MAXDP' MINAGREE='$MINAGREE' \
  JT_PARITY_TS='${JT_PARITY_TS:-}' JT_REF_PREDS='${JT_REF_PREDS:-}' JT_ENV='$(sq "$JT_ENV")' \
  JT_EXPECT='$(sq "$JT_EXPECT")' bash -s" <<'REMOTE'
set -euo pipefail
cd "$HOME/CLM-8B"
. tools/jeff_lora/cuda/env.sh            # DATA, VENV, REMOTE_MODEL (and the HF/tokenizer env)
BIN="$HOME/$JT_REMOTE/build/nv/jefftrain"
PREP="$HOME/$JT_REMOTE/py/prep.py"
COMPARE=tools/jeff_lora/cuda/compare_preds.py
MODEL="$HOME/$REMOTE_MODEL"
PY=$VENV/bin/python
for f in "$BIN" "$PREP" "$COMPARE" "$MODEL/decision_config.json" "$DATA/test.jsonl"; do
  [ -e "$f" ] || { echo "parity-eval: missing $f" >&2; exit 2; }
done
ref=${JT_REF_PREDS:-}
if [ -z "$ref" ]; then                         # the PyTorch port's own fp32 eval run (plan M6.1)
  ref=$(ls -d data/ovril_detect/runs/cuda-parity-fp32-*/pred-test.jsonl 2>/dev/null | tail -n 1 || true)
fi
[ -n "$ref" ] && [ -f "$ref" ] || { echo "parity-eval: no reference; pass JT_REF_PREDS=<pred-test.jsonl>" >&2; exit 2; }

# The answer temperature is the checkpoint's own: load_decision_config knows where it lives, train.py's default is
# "--eval-temperature checkpoint", and the reference run used exactly that value.
T=$($PY -c 'import sys; from tools.jeff_lora.prompt import load_decision_config as l; print(repr(l(sys.argv[1])["temperature"]))' "$MODEL")

TS=${JT_PARITY_TS:-$(date -u +%Y%m%d-%H%M%S)}
OUT=data/ovril_detect/runs/jefftrain-parity-$TS
mkdir -p "$OUT"
echo "parity-eval: model $MODEL"
echo "parity-eval: T $T (decision_config.json)  ref $ref"
echo "parity-eval: prep.py --eval test=$DATA/test.jsonl --batch 4 -> $OUT"
$PY "$PREP" --clm "$PWD" --model "$MODEL" --eval "test=$DATA/test.jsonl" --batch 4 --out-dir "$OUT"

rc=0
for mode in tf32 exact; do
  want="gemm: tf32"; exact_env=""
  if [ "$mode" = exact ]; then want="gemm: exact fp32"; exact_env="JT_EXACT=1"; fi
  echo "parity-eval: jefftrain --eval-only, $mode build -> $OUT/$mode"
  log=$(mktemp)
  if env $exact_env $JT_ENV "$BIN" --model "$MODEL" --eval-only --eval "test=$OUT/test.jtb" --batch 4 \
       --eval-temperature "$T" --output "$OUT/$mode" > "$log" 2>&1; then
    sed 's/^/  | /' "$log"
    grep -qxF "$want" "$log" || { echo "parity-eval: $mode: expected '$want' on stdout, got:" >&2
                                   sed 's/^/  | /' "$log" >&2; rc=1; }
    if [ -n "$JT_EXPECT" ] && ! grep -Eq -- "$JT_EXPECT" "$log"; then
      echo "parity-eval: $mode: JT_EXPECT '$JT_EXPECT' matched no line of the log" >&2
      rc=1
    fi
    $PY "$COMPARE" "$OUT/$mode/pred-test.jsonl" "$ref" --max-dp "$MAXDP" --min-agree "$MINAGREE" \
      || { echo "parity-eval: $mode FAILED the gate (max |dp| <= $MAXDP, argmax >= $MINAGREE)" >&2; rc=1; }
  else
    sed 's/^/  | /' "$log"
    echo "parity-eval: $mode build exited non-zero" >&2; rc=1
  fi
  rm -f "$log"
done
echo "parity-eval: outputs in ~/CLM-8B/$OUT (no row content leaves the box)"
echo "parity-eval: $([ "$rc" = 0 ] && echo PASS || echo FAIL)  gate max |dp| <= $MAXDP, argmax >= $MINAGREE/160"
exit "$rc"
REMOTE
