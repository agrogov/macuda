#!/bin/sh
# ab.sh — A/B timing for jefftrain on gpubox: side A is the worktree's committed HEAD, side B is the worktree as it
# is, and the runs are interleaved A B A B ... so a box that drifts, warms up or shares its GPU cannot favour one side.
#
# The M8 follow-ups measure every later change with this file, so its two verdicts are the batch's rules:
#   * speed:   B at least 1.5% faster than A, and the gap more than 3x the larger side's run-to-run spread  -> WORTH: yes
#   * sameness: exact mode, both sides from the same prepared data, the 20-step steps.jsonl and both prediction files
#              byte-identical to the baseline's                                                         -> BITEXACT: yes
#
# What each side is:
#   A   `git archive HEAD train/jeff` into a fresh directory under $TMPDIR, synced and built by that copy's own
#       scripts/gpubox-sync.sh + gpubox-build.sh to ~/$JT_REMOTE-base on gpubox (the card's start commit, before its
#       edits — the script edits of a tooling card like this one do not change the binary, which is what makes the
#       smoke run below report "inside the spread");
#   B   this worktree, synced and built to ~/$JT_REMOTE by this tree's scripts/gpubox-build.sh.
# Both builds start from `rm -rf build` on the box, because `rsync -a` restores the source's mtime and make would
# otherwise keep a stale object (design §11.3's trap).
#
# Both sides run jefftrain inside ~/CLM-8B on the .jtb files J38-J40 timed (JT_AB_DATA, default
# data/ovril_detect/runs/parity-steps-latest: py/prep.py's 256 train rows / 64 dev rows at --batch 8 --accum 1 --seed 0,
# plus py/make_adapter.py's adapter), so a number here is comparable with the ones in notes/J38.md … notes/J40.md.
# A different --batch or --accum needs a JT_AB_DATA prepared for it (py/prep.py, the parity-steps.sh call).
#
# usage (from train/jeff, on the Mac):
#   export JT_REMOTE=macuda-train/wt-J43          # parallel tasks must not share one
#   sh scripts/ab.sh                              # 3 interleaved pairs of 20 steps, batch 8 x accum 1, TF32
#   sh scripts/ab.sh --steps 6 --pairs 1          # a smoke run
#   sh scripts/ab.sh --same                       # exact mode, one run per side, then cmp the three files
#   sh scripts/ab.sh --env-b 'JT_X=1' --expect-b '^ctx: JT_X'    # the toggle under test, and its startup line
#
#   --pairs P        interleaved A B pairs (default 3). A side's median is the median of its P run medians; its spread
#                    is max-min of them (rule 8's run-to-run spread)
#   --steps N        optimizer steps per run (default 20). Each run's median is over steps 3..N, because steps 1-3
#                    carry compile and allocator warm-up (plan M6.3's rule, tools/jeff_lora/cuda/timing.sh)
#   --batch B        jefftrain --batch (default 8, the default data's shape)
#   --accum A        jefftrain --accum (default 1, likewise)
#   --exact          JT_EXACT=1 TINYCUBLAS_TC=0 instead of TF32 (--same always runs exact)
#   --same           the bit-identical check: each side once, in exact mode, with --trace-loss and --eval dev=/test=,
#                    then cmp steps.jsonl, pred-dev.jsonl and pred-test.jsonl. Prints BITEXACT: yes only if all three
#                    are byte-identical (A is the baseline). jefftrain has no RNG, so "the same seed" is the same
#                    prepared data and adapter (py/prep.py --seed, py/make_adapter.py --seed) on both sides.
#   --env-a WORDS    extra JT_*=value settings for side A's jefftrain (default: none), e.g. 'JT_A=1 JT_B=2'
#   --env-b WORDS    the same for side B (the change under test)
#   --expect-a RE    fail a side-A run whose log has no line matching the ERE RE
#   --expect-b RE    the same for side B: this is where the toggle's one startup line is checked, so a toggle that
#                    silently did nothing cannot report WORTH: yes (batch rule 9's JT_EXPECT)
#   -h, --help       this text
#   Every --env word is checked on the Mac against ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$ before it goes anywhere near ssh,
#   so a word with a space, ';' or '$' is refused rather than run.
#
#   env: JT_REMOTE (default macuda-train/jeff; side A is ${JT_REMOTE}-base)
#        JT_AB_DATA (gpubox path, relative to ~/CLM-8B, holding prep/train.jtb, prep/dev.jtb and adapter/;
#                    default data/ovril_detect/runs/parity-steps-latest)
#        JT_AB_RUN  (run directory name under ~/CLM-8B/data/ovril_detect/runs/, default ab-<UTC stamp>)
#        JT_SKIP_BUILD=1 (use the binaries that are already on the box)
#        ARCH (default sm_120, the nv build's arch)
#
# Every GPU run is under gpubox's exclusive lock (`flock ~/macuda-train/.gpu.lock`, the same file the other gpubox
# scripts take); if flock is missing there the script fails and says so. Logs, traces and run directories stay in
# CLM-8B's git-ignored data/ on gpubox (plan rule 5): nothing is written into macuda and no row content leaves the box.
set -eu
JEFF=$(cd "$(dirname "$0")/.." && pwd)
ROOT=$(git -C "$JEFF" rev-parse --show-toplevel 2>/dev/null) || {
  echo "ab: $JEFF is not inside a git checkout" >&2; exit 2; }

usage() { sed -n '2,/^set -eu$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
bad() { echo "ab: $*" >&2; exit 2; }

# JT_*=value only: checked here, on the Mac, so nothing else can reach the remote jefftrain's environment (rule 2).
check_env() {  # $1 = the option's name, $2 = its words
  for w in $2; do
    printf '%s\n' "$w" | grep -Eq '^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*$' || {
      echo "ab: $1: '$w' is not JT_NAME=value" >&2
      echo "ab: every word must match ^JT_[A-Z0-9_]+=[A-Za-z0-9._-]*\$ — no spaces, ';', '\$' or quotes" >&2
      exit 2
    }
  done
}

# one word, quoted once for the remote shell (single quotes; the validators above keep quotes out of the values)
sq() { printf '%s' "$1" | sed "s/'/'\\\\''/g"; }

PAIRS=3 STEPS=20 BATCH=8 ACCUM=1
EXACT=0 SAME=0
ENV_A= ENV_B= EXP_A= EXP_B=
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
ARCH=${ARCH:-sm_120}
JT_AB_DATA=${JT_AB_DATA:-data/ovril_detect/runs/parity-steps-latest}
JT_AB_RUN=${JT_AB_RUN:-}
TMP=${TMPDIR:-/tmp}

while [ $# -gt 0 ]; do
  case $1 in
    --pairs|--steps|--batch|--accum|--env-a|--env-b|--expect-a|--expect-b)
      [ $# -ge 2 ] || bad "$1 wants a value"
      opt=$1 val=$2
      shift 2
      case $opt in
        --pairs)    PAIRS=$val ;;
        --steps)    STEPS=$val ;;
        --batch)    BATCH=$val ;;
        --accum)    ACCUM=$val ;;
        --env-a)    ENV_A=$val ;;
        --env-b)    ENV_B=$val ;;
        --expect-a) EXP_A=$val ;;
        --expect-b) EXP_B=$val ;;
      esac ;;
    --exact) EXACT=1; shift ;;
    --same)  SAME=1;  shift ;;
    -h|--help) usage; exit 0 ;;
    *) bad "unknown option '$1' (--help)" ;;
  esac
done

case $PAIRS in ''|*[!0-9]*) bad "--pairs wants a positive integer, got '$PAIRS'" ;; esac
case $STEPS in ''|*[!0-9]*) bad "--steps wants a positive integer, got '$STEPS'" ;; esac
case $BATCH in ''|*[!0-9]*) bad "--batch wants a positive integer, got '$BATCH'" ;; esac
case $ACCUM in ''|*[!0-9]*) bad "--accum wants a positive integer, got '$ACCUM'" ;; esac
[ "$PAIRS" -ge 1 ] || bad "--pairs must be at least 1"
[ "$STEPS" -ge 4 ] || bad "--steps must be at least 4 (the median starts at step 3)"
[ "$BATCH" -ge 1 ] || bad "--batch must be at least 1"
[ "$ACCUM" -ge 1 ] || bad "--accum must be at least 1"
check_env "--env-a" "$ENV_A"
check_env "--env-b" "$ENV_B"

if [ "$SAME" = 1 ]; then PAIRS=1; EXACT=1; fi   # --same is one run per side, in exact mode

A_REMOTE=${JT_REMOTE}-base
if [ "$SAME" = 1 ]; then
  MODE_TEXT="--same: one exact-mode run per side, cmp steps.jsonl + pred-dev.jsonl + pred-test.jsonl"
elif [ "$EXACT" = 1 ]; then MODE_TEXT="$PAIRS pair(s), exact fp32"
else MODE_TEXT="$PAIRS pair(s), TF32"
fi
echo "ab: side A = git archive HEAD ($(git -C "$ROOT" rev-parse --short HEAD)) -> gpubox:~/$A_REMOTE"
echo "ab: side B = the worktree $JEFF -> gpubox:~/$JT_REMOTE"
echo "ab: $STEPS steps, batch $BATCH x accum $ACCUM, $MODE_TEXT"
echo "ab: data ~/CLM-8B/$JT_AB_DATA  (prep/train.jtb, prep/dev.jtb, adapter/)"
[ -z "$ENV_A" ] || echo "ab: side A env: $ENV_A"
[ -z "$ENV_B" ] || echo "ab: side B env: $ENV_B"

if [ "${JT_SKIP_BUILD:-0}" = 1 ]; then
  echo "ab: JT_SKIP_BUILD=1: using the binaries already in ~/$A_REMOTE and ~/$JT_REMOTE"
else
  A_TMP=$TMP/ab-A-$(date -u +%Y%m%d-%H%M%S)
  rm -rf "$A_TMP"
  mkdir -p "$A_TMP"
  git -C "$ROOT" archive HEAD train/jeff | tar -x -C "$A_TMP"
  [ -f "$A_TMP/train/jeff/scripts/gpubox-build.sh" ] || bad "the archive has no train/jeff/scripts/gpubox-build.sh"
  echo "ab: side A source $A_TMP/train/jeff"
  ssh gpubox "rm -rf ~/$A_REMOTE/build"                       # design §11.3: never let make keep a stale object
  JT_REMOTE=$A_REMOTE ARCH=$ARCH sh "$A_TMP/train/jeff/scripts/gpubox-build.sh"
  ssh gpubox "rm -rf ~/$JT_REMOTE/build"
  JT_REMOTE=$JT_REMOTE ARCH=$ARCH sh "$JEFF/scripts/gpubox-build.sh"
fi

ssh gpubox "if ! command -v flock >/dev/null 2>&1; then
    echo 'ab: flock is missing on gpubox; it is needed for ~/macuda-train/.gpu.lock' >&2
    exit 2
  fi
  exec flock ~/macuda-train/.gpu.lock env JT_REMOTE_A='$A_REMOTE' JT_REMOTE_B='$(sq "$JT_REMOTE")' \
    JT_AB_DATA='$(sq "$JT_AB_DATA")' JT_AB_RUN='$(sq "$JT_AB_RUN")' PAIRS='$PAIRS' STEPS='$STEPS' \
    BATCH='$BATCH' ACCUM='$ACCUM' SAME='$SAME' EXACT='$EXACT' ENV_A='$(sq "$ENV_A")' ENV_B='$(sq "$ENV_B")' \
    EXP_A='$(sq "$EXP_A")' EXP_B='$(sq "$EXP_B")' sh -s" <<'REMOTE'
set -eu
cd "$HOME/CLM-8B"
. tools/jeff_lora/cuda/env.sh                # DATA, VENV, REMOTE_MODEL (and the HF/tokenizer env)
MODEL="$HOME/$REMOTE_MODEL"
PY=$VENV/bin/python
case $JT_AB_DATA in
  /*) D=$JT_AB_DATA ;;
  *)  D=$PWD/$JT_AB_DATA ;;
esac
BIN_A="$HOME/$JT_REMOTE_A/build/nv/jefftrain"
BIN_B="$HOME/$JT_REMOTE_B/build/nv/jefftrain"
RUNS=$PWD/data/ovril_detect/runs
RUN=${JT_AB_RUN:+$RUNS/$JT_AB_RUN}
if [ -z "${RUN:-}" ]; then RUN=$RUNS/ab-$(date -u +%Y%m%d-%H%M%S); fi
for f in "$BIN_A" "$BIN_B" "$MODEL/model.safetensors" "$D/prep/train.jtb" "$D/prep/dev.jtb" \
         "$D/adapter/adapter_model.safetensors"; do
  [ -e "$f" ] || { echo "ab: missing $f" >&2; exit 2; }
done
if [ "$SAME" = 1 ] || [ "$EXACT" = 1 ]; then
  EXACT_ENV="JT_EXACT=1 TINYCUBLAS_TC=0"; WANT="gemm: exact fp32"
else
  EXACT_ENV=""; WANT="gemm: tf32"
fi

# ---- the two numbers a run reports: its median s/step over steps 3..N and its peak memory --------------------------
median_of() {  # stdin: numbers -> their median
  sort -g | awk '{ v[NR] = $1 } END {
    if (NR == 0) { print "nan"; exit 0 }
    if (NR % 2) printf "%.4f\n", v[(NR + 1) / 2]
    else printf "%.4f\n", (v[NR / 2] + v[NR / 2 + 1]) / 2
  }'
}
spread_of() {  # stdin: numbers -> max minus min ("nan" with none)
  sort -g | awk 'NR == 1 { lo = $1 } { hi = $1 } END {
    if (NR == 0) { print "nan"; exit 0 }
    printf "%.4f\n", hi - lo
  }'
}
median_step() {  # $1 = log, $2 = steps -> the median s/step over steps 3..N of jefftrain's step lines
  awk -v n="$2" '
    /^ *step / {
      s = ""; t = 0; have = 0
      for (i = 1; i <= NF; i++) {
        if ($i ~ /^[0-9]+\/[0-9]+$/) { split($i, a, "/"); s = a[1] + 0 }
        if ($i ~ /^[0-9]+\.[0-9]+s$/) { t = substr($i, 1, length($i) - 1) + 0; have = 1 }
      }
      if (have && s >= 3 && s <= n) print t
    }' "$1" | median_of
}
peak_gb() {  # $1 = log -> jefftrain's own "peak <x> GB" line
  grep -oE 'peak [0-9.]+ GB' "$1" | tail -n 1 | awk '{ print $2 }'
}

run_side() {  # $1 side (A|B), $2 binary, $3 extra JT_*=value words, $4 JT_EXPECT regex, $5 run index
  side=$1 bin=$2 extra=$3 expect=$4 k=$5
  out=$RUN/$side-$k
  log=$out.log
  mkdir -p "$out"
  echo "=== $side run $k: $bin -> $out"
  t0=$(date +%s)
  set -- "$bin" --model "$MODEL" --init-adapter "$D/adapter" \
    --train "$D/prep/train.jtb" --dev "$D/prep/dev.jtb" --output "$out" \
    --lr 2e-4 --readout-lr 5e-6 --batch "$BATCH" --accum "$ACCUM" --epochs 1 \
    --max-steps "$STEPS" --eval-every "$STEPS" --targets soft --weights none --log-every 1 \
    --trace-loss "$out/steps.jsonl"
  if [ "$SAME" = 1 ]; then
    set -- "$@" --eval "dev=$D/prep/dev.jtb" --eval "test=$TEST_JTB"
  fi
  if ! env $EXACT_ENV $extra "$@" > "$log" 2>&1; then
    sed 's/^/  | /' "$log" | tail -n 25
    echo "ab: $side run $k exited non-zero (log $log)" >&2
    exit 1
  fi
  if ! grep -qxF "$WANT" "$log"; then
    echo "ab: $side run $k: expected '$WANT' on stdout, got:" >&2
    head -n 6 "$log" | sed 's/^/  | /' >&2
    exit 1
  fi
  if [ -n "$expect" ]; then
    if ! grep -Eq -- "$expect" "$log"; then
      case $side in A) lbl=--expect-a ;; *) lbl=--expect-b ;; esac
      echo "ab: $side run $k: $lbl '$expect' matched no line of the log:" >&2
      head -n 8 "$log" | sed 's/^/  | /' >&2
      echo "ab: $side run $k: a toggle that did nothing does not count as a run" >&2
      exit 1
    fi
    echo "ab: $side run $k: log matches /$expect/"
  fi
  med=$(median_step "$log" "$STEPS")
  peak=$(peak_gb "$log")
  if [ "$med" = nan ]; then
    sed 's/^/  | /' "$log" | tail -n 10
    echo "ab: $side run $k: no step lines to take a median from (log $log)" >&2
    exit 1
  fi
  echo "ab: $side run $k: median $med s/step over steps 3..$STEPS, peak ${peak:-?} GB, $(( $(date +%s) - t0 ))s wall"
  echo "$side $med" >> "$RUN/medians.txt"
}

echo "ab: data $D"
echo "ab: model $MODEL  compute '$WANT'"
mkdir -p "$RUN"
: > "$RUN/medians.txt"
echo "ab: run directory $RUN"

# --same needs the test split's predictions too. JT_AB_DATA carries train/dev; a test.jtb is made once, here, from
# CLM-8B's own test split, with the arguments plan M6.1's parity-eval.sh uses (prep.py --eval test --batch 4).
TEST_JTB=
if [ "$SAME" = 1 ]; then
  if [ -f "$D/prep/test.jtb" ]; then
    TEST_JTB=$D/prep/test.jtb
    echo "ab: test split $TEST_JTB"
  else
    echo "=== prep: py/prep.py --eval test=$DATA/test.jsonl --batch $BATCH -> $RUN/testprep"
    if ! "$PY" "$HOME/$JT_REMOTE_B/py/prep.py" --clm "$PWD" --model "$MODEL" \
         --eval "test=$DATA/test.jsonl" --batch "$BATCH" --accum "$ACCUM" --epochs 1 \
         --out-dir "$RUN/testprep" > "$RUN/testprep.log" 2>&1; then
      sed 's/^/  | /' "$RUN/testprep.log" | tail -n 20
      echo "ab: prep.py failed (log $RUN/testprep.log)" >&2
      exit 1
    fi
    grep '^prep:' "$RUN/testprep.log" | sed 's/^/  | /' || true
    TEST_JTB=$RUN/testprep/test.jtb
  fi
fi

i=1
while [ "$i" -le "$PAIRS" ]; do
  run_side A "$BIN_A" "$ENV_A" "$EXP_A" "$i"
  run_side B "$BIN_B" "$ENV_B" "$EXP_B" "$i"
  i=$((i + 1))
done

# ---- the verdict ---------------------------------------------------------------------------------------------------
if [ "$SAME" = 1 ]; then
  ok=1
  for f in steps.jsonl pred-dev.jsonl pred-test.jsonl; do
    if [ ! -f "$RUN/A-1/$f" ] || [ ! -f "$RUN/B-1/$f" ]; then
      echo "ab: $f is missing on one side" >&2
      ok=0
    elif cmp -s "$RUN/A-1/$f" "$RUN/B-1/$f"; then
      echo "ab: $f: byte-identical ($(wc -c < "$RUN/A-1/$f" | tr -d ' ') bytes)"
    else
      echo "ab: $f: DIFFERS (baseline $RUN/A-1/$f, change $RUN/B-1/$f)" >&2
      ok=0
    fi
  done
  if [ "$ok" = 1 ]; then
    echo "BITEXACT: yes"
  else
    echo "BITEXACT: no"
    exit 1
  fi
else
  na=$(awk '$1 == "A"' "$RUN/medians.txt" | wc -l | tr -d ' ')
  nb=$(awk '$1 == "B"' "$RUN/medians.txt" | wc -l | tr -d ' ')
  A_MED=$(awk '$1 == "A" { print $2 }' "$RUN/medians.txt" | median_of)
  B_MED=$(awk '$1 == "B" { print $2 }' "$RUN/medians.txt" | median_of)
  A_SPR=$(awk '$1 == "A" { print $2 }' "$RUN/medians.txt" | spread_of)
  B_SPR=$(awk '$1 == "B" { print $2 }' "$RUN/medians.txt" | spread_of)
  awk -v a="$A_MED" -v b="$B_MED" -v sa="$A_SPR" -v sb="$B_SPR" -v na="$na" -v nb="$nb" 'BEGIN {
    gap = a - b
    rel = (a > 0) ? 100 * gap / a : 0
    big = (sa > sb) ? sa : sb
    if (gap > 0 && big > 0) x = sprintf("the gap is %.1fx the larger spread (%.4f)", gap / big, big)
    else if (big == 0)    x = sprintf("the larger spread is 0.0000 over %s and %s run(s), gap %+.4f", na, nb, gap)
    else                  x = sprintf("the gap %+.4f is below the larger spread %.4f", gap, big)
    printf "ab: side A median %.4f s/step, spread %.4f (%s run(s))\n", a, sa, na
    printf "ab: side B median %.4f s/step, spread %.4f (%s run(s))\n", b, sb, nb
    if (rel >= 0) printf "ab: B is %.2f%% faster than A; %s\n", rel, x
    else          printf "ab: B is %.2f%% slower than A; %s\n", -rel, x
    printf "ab: worth needs B >= 1.5%% faster and gap > 3x the larger spread\n"
    printf "WORTH: %s\n", (rel >= 1.5 && gap > 3 * big) ? "yes" : "no"
  }'
fi
echo "ab: run directory $RUN (logs <side>-<k>.log, traces <side>-<k>/steps.jsonl)"
REMOTE
