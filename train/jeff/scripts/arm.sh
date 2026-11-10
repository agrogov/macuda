#!/bin/sh
# arm.sh — plan M6.4: one full soft arm trained by jefftrain, scored next to the Python arm (design §10.3.3).
#
# Plan M6.4: "scripts/arm.sh soft: the kit's arms.sh soft arm settings (1 epoch, batch 8, dev every 500 steps on all of
# development, eval calibration and test at T=1), with jefftrain doing the training. Then CLM-8B's fetch_and_score.sh
# path scores it next to the Python arm. Done when: D§10.3.3's gates hold (dev soft CE at the chosen step within 0.01
# of the Python soft arm, test soft CE within 0.02, score.py metrics inside the two-seed noise band). Report the table
# to the user."
#
# This script holds gpubox's GPU for about 20-30 minutes (3,581 steps of batch 8, plus three evals). That run is card
# G-m64 and asks the user first; running this script is that run, not a dry run.
#
# It only orchestrates, on gpubox, inside ~/CLM-8B (plan M6's preamble). The real splits, the .jtb files, the adapter,
# the predictions and the reports all stay in CLM-8B's git-ignored data/ (plan rule 5); nothing is written into macuda.
#
# Pieces, in the kit's order (tools/jeff_lora/cuda/arms.sh):
#   1. py/prep.py          — train (28,645 rows, the epoch order at --seed), dev (all 1,722 rows), calibration and test,
#                            all at --batch 8 so the trainer walks train.py's batches (design §2.3, §9.1);
#   2. py/make_adapter.py  — a fresh adapter to start from. jefftrain has no init of its own (src/lora.c zeroes A and B,
#                            and main.c does not fill them), so a run without --init-adapter would never move. This is
#                            design §10.1's fresh adapter: A random, B random * 1e-2. train.py starts with B = 0, so the
#                            first steps differ by a ~2e-2 LoRA update — far inside the 0.01/0.02 gate, and stated here
#                            so nobody has to guess;
#   3. jefftrain           — the arm's settings, dev checks every 500 steps on all of dev, then calibration and
#                            test at --eval-temperature 1, exactly arms.sh's call;
#   4. fit_temperature.py  — CLM-8B's own: fit T on calibration, write pred-test-cal.jsonl (arms.sh does the same);
#   5. score.py            — CLM-8B's own, on jefftrain's calibrated and T=1 predictions and on each Python arm run of
#                            the same arm, all with the same --test file;
#   6. the gates of design §10.3.3, below.
#
# Step 5 is the "fetch_and_score.sh path" without its bookkeeping: fetch_and_score.sh rsyncs a run named cuda-<arm>-<TS>
# to the Mac and overwrites bench/ovril_detect/predictions/jeff-lora-<arm>.jsonl and scores_lora_cuda.json, which is the
# Python arm's own record. We call the same score.py with the same NAME=PRED.jsonl arguments and keep its JSON in our
# own run directory instead, so the Python arm's files are reproduced, not replaced.
#
# The gates:
#   dev soft CE   jefftrain's report.json best_dev_soft_ce vs the Python run's, |d| <= 0.01   (design §10.3.3)
#   test soft CE  score.py's calibrated jev_soft_ce, |d| <= 0.02                              (design §10.3.3)
#   score.py metrics  every column of score.py's table, |d| <= the two-seed band: the spread of that same metric
#                 across the arm's Python runs (the same arm at --seed 0 and --seed 1, design §10.3.3's "noise"),
#                 floored at the measurement's own resolution: one test line = one card = 1/160 for ratios, 2 for
#                 counts. The floors keep a metric whose two seeds happen to agree exactly from demanding bit-equal
#                 predictions; they do not touch the two CE gates above, which are the design's numbers.
#                 THE COUNT FLOOR IS 2, NOT 1 (2026-11-10, the user's decision on G-m74). A count metric moves by two
#                 lines whenever a prediction lands within rounding distance of the threshold on either side of it,
#                 and a single line is below the resolution score.py's thresholds actually have: G-m74's card arms
#                 matched the Python arm on every scored metric — precision, recall, F1, kind accuracy and tp/fp/fn
#                 at both thresholds, some to the digit — and still missed th0.9.maybe_cards 5 vs 7, because two
#                 lines both sides classify identically sat on opposite sides of the 0.9 boundary after TF32 rounding.
#                 The floor was 1 line until this change; G-m64's gpubox arm passed under it, and G-m74's arms record
#                 what the 1-line floor said in notes/G-m74.md so the decision can be read back.
# The Python reference is the arm's own runs that are already on the box (data/ovril_detect/runs/cuda-<arm>-*, 2026-11-08
# for soft). The primary is the one whose report.json args.seed equals this run's --seed, so the row order matches;
# JT_PRIMARY_RUN names one explicitly, JT_REF_RUNS replaces the whole list.
#
# usage (from train/jeff, on the Mac; this script syncs and builds the tree itself):
#     export JT_REMOTE=macuda-train/wt-J34        # parallel tasks must not share one
#     sh scripts/arm.sh soft                      # plan M6.4's run (card G-m64: ask the user first)
#   env:
#     JT_REMOTE      build tree under the box's ~ (default macuda-train/jeff)
#     JT_SKIP_BUILD  1 = do not sync/build, run the binary that is already there
#     ARCH           the nv build's arch (default sm_120)
#     EXACT          1 = JT_EXACT=1 TINYCUBLAS_TC=0; default 0 (TF32), the build an arm is timed on
#     BATCH, ACCUM, EPOCHS, EVAL_EVERY, SEED, LR, READOUT_LR, ADAPTER_SEED   the arm's settings (kit defaults)
#     RUN            run directory (default data/ovril_detect/runs/jefftrain-arm-<arm>-<UTC stamp>)
#     JT_REF_RUNS    space-separated Python arm run directories for the comparison
#     JT_PRIMARY_RUN the one the two CE gates are quoted against (default: same --seed, else the newest)
#     DEV_TOL, TEST_TOL   the two CE gates (defaults 0.01 and 0.02 — the design's; do not loosen them)
set -eu
ARM=${1:-}
case $ARM in
  soft)     TARGETS=soft; WEIGHTS=none;     WEIGHT_CAP=4.0 ;;
  hard)     TARGETS=hard; WEIGHTS=none;     WEIGHT_CAP=4.0 ;;
  balanced) TARGETS=soft; WEIGHTS=balanced; WEIGHT_CAP=4.0 ;;
  *)
    echo "usage: sh scripts/arm.sh <soft|hard|balanced>" >&2
    echo "  plan M6.4's run is the soft arm: sh scripts/arm.sh soft" >&2
    echo "  It holds gpubox's GPU for about 20-30 min; that is card G-m64 and needs the user's OK first." >&2
    exit 2 ;;
esac
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
ARCH=${ARCH:-sm_120}
EXACT=${EXACT:-0}
BATCH=${BATCH:-8} ACCUM=${ACCUM:-1} EPOCHS=${EPOCHS:-1} EVAL_EVERY=${EVAL_EVERY:-500}
SEED=${SEED:-0} LR=${LR:-2e-4} READOUT_LR=${READOUT_LR:-5e-6} ADAPTER_SEED=${ADAPTER_SEED:-1}
DEV_TOL=${DEV_TOL:-0.01} TEST_TOL=${TEST_TOL:-0.02}
JEFF=$(cd "$(dirname "$0")/.." && pwd)
[ "${JT_SKIP_BUILD:-0}" = 1 ] || sh "$JEFF/scripts/gpubox-build.sh"

ssh gpubox "JT_REMOTE='$JT_REMOTE' ARM='$ARM' TARGETS='$TARGETS' WEIGHTS='$WEIGHTS' WEIGHT_CAP='$WEIGHT_CAP' \
  EXACT='$EXACT' BATCH='$BATCH' ACCUM='$ACCUM' EPOCHS='$EPOCHS' EVAL_EVERY='$EVAL_EVERY' SEED='$SEED' LR='$LR' \
  READOUT_LR='$READOUT_LR' ADAPTER_SEED='$ADAPTER_SEED' DEV_TOL='$DEV_TOL' TEST_TOL='$TEST_TOL' \
  RUN='${RUN:-}' JT_REF_RUNS='${JT_REF_RUNS:-}' JT_PRIMARY_RUN='${JT_PRIMARY_RUN:-}' sh -s" <<'REMOTE'
set -eu
cd "$HOME/CLM-8B"                          # the real splits, the venv and the other arms' runs live here
. tools/jeff_lora/cuda/env.sh               # DATA, VENV, REMOTE_MODEL (and the HF/tokenizer env)
BIN="$HOME/$JT_REMOTE/build/nv/jefftrain"
PREP="$HOME/$JT_REMOTE/py/prep.py"
MAKE_ADAPTER="$HOME/$JT_REMOTE/py/make_adapter.py"
FIT=tools/jeff_lora/cuda/fit_temperature.py
SCORE=bench/ovril_detect/score.py
MODEL="$HOME/$REMOTE_MODEL"
PY=$VENV/bin/python
RUNS=$PWD/data/ovril_detect/runs
for f in "$BIN" "$PREP" "$MAKE_ADAPTER" "$FIT" "$SCORE" "$MODEL/model.safetensors" \
         "$DATA/train.jsonl" "$DATA/development.jsonl" "$DATA/calibration.jsonl" "$DATA/test.jsonl"; do
  [ -e "$f" ] || { echo "arm: missing $f" >&2; exit 2; }
done

RUN=${RUN:-$RUNS/jefftrain-arm-$ARM-$(date -u +%Y%m%d-%H%M%S)}
mkdir -p "$RUN/prep" "$RUN/ref"
ln -sfn "$RUN" "$RUNS/jefftrain-arm-$ARM-latest"   # plan M7.4 and the notes find the newest run by this link
if [ "$EXACT" = 1 ]; then EXACT_ENV="JT_EXACT=1 TINYCUBLAS_TC=0"; WANT="gemm: exact fp32"; else EXACT_ENV=""; WANT="gemm: tf32"; fi
echo "arm: run directory $RUN"
echo "arm: $ARM arm: --targets $TARGETS --weights $WEIGHTS${WEIGHT_CAP:+ --weight-cap $WEIGHT_CAP}, $EPOCHS epoch(s),"
echo "     batch $BATCH x accum $ACCUM, dev every $EVAL_EVERY steps on all of development, seed $SEED, compute '$WANT'"
echo "arm: model $MODEL  data $DATA"

# ---- 1. the rows: train.py's own tokenisation and epoch order ------------------------------------------------------
echo "=== prep: py/prep.py (full train split, all dev, calibration and test) -> $RUN/prep"
"$PY" "$PREP" --clm "$PWD" --model "$MODEL" \
  --train "$DATA/train.jsonl" --dev "$DATA/development.jsonl" \
  --eval "calibration=$DATA/calibration.jsonl" --eval "test=$DATA/test.jsonl" \
  --targets "$TARGETS" --weights "$WEIGHTS" --weight-cap "$WEIGHT_CAP" \
  --batch "$BATCH" --accum "$ACCUM" --epochs "$EPOCHS" --seed "$SEED" --out-dir "$RUN/prep"

# ---- 2. the adapter the run starts from ----------------------------------------------------------------------------
echo "=== adapter: py/make_adapter.py (fresh: A random, B random * 1e-2) -> $RUN/adapter-init"
"$PY" "$MAKE_ADAPTER" --clm "$PWD" --model "$MODEL" --seed "$ADAPTER_SEED" --out "$RUN/adapter-init"

# ---- 3. the arm ----------------------------------------------------------------------------------------------------
echo "=== jefftrain: the $ARM arm -> $RUN/c"
log=$RUN/jefftrain.log
t0=$(date +%s)
env $EXACT_ENV "$BIN" --model "$MODEL" --init-adapter "$RUN/adapter-init" \
  --train "$RUN/prep/train.jtb" --dev "$RUN/prep/dev.jtb" --output "$RUN/c" \
  --batch "$BATCH" --accum "$ACCUM" --epochs "$EPOCHS" --eval-every "$EVAL_EVERY" \
  --eval "calibration=$RUN/prep/calibration.jtb" --eval "test=$RUN/prep/test.jtb" --eval-temperature 1 \
  --targets "$TARGETS" --weights "$WEIGHTS" --weight-cap "$WEIGHT_CAP" \
  --lr "$LR" --readout-lr "$READOUT_LR" --log-every 25 > "$log" 2>&1 || {
    sed 's/^/  | /' "$log" | tail -n 30
    echo "arm: jefftrain exited non-zero (log $log)" >&2
    echo "arm: FAIL — the $ARM arm did not finish"
    exit 1
  }
grep -qxF "$WANT" "$log" || { echo "arm: expected '$WANT' on stdout, got:" >&2
                                head -n 5 "$log" | sed 's/^/  | /' >&2; }
grep -E '^  dev @ step|^eval |^jefftrain:|^peak ' "$log" | head -n 40 | sed 's/^/  | /'
echo "  | $(grep -c '^  step ' "$log") step line(s) over $(( $(date +%s) - t0 ))s (full log $log)"

# ---- 4. the calibrated predictions, in the same way arms.sh does it ------------------------------------------------
echo "=== temperature: tools/jeff_lora/cuda/fit_temperature.py on calibration"
"$PY" "$FIT" "$RUN/c" --calibration "$DATA/calibration.jsonl"

# ---- 5. score.py next to the Python arm ----------------------------------------------------------------------------
echo "=== score: jefftrain's predictions (T=1 and calibrated), 160 test rows"
"$PY" "$SCORE" --test "$DATA/test.jsonl" \
  "jefftrain-$ARM-T1=$RUN/c/pred-test.jsonl" "jefftrain-$ARM=$RUN/c/pred-test-cal.jsonl" --json "$RUN/c/score.json"

refs=${JT_REF_RUNS:-}
if [ -z "$refs" ]; then
  refs=$(ls -d "$RUNS"/cuda-"$ARM"-* 2>/dev/null | sort || true)   # the kit's own runs of this arm, oldest first
fi
have=""
for d in $refs; do
  if [ -f "$d/report.json" ] && [ -f "$d/pred-test-cal.jsonl" ]; then
    have="$have $d"
  else
    echo "  (skipping $d: no report.json or pred-test-cal.jsonl)"
  fi
done
refs=$have
if [ -z "$refs" ]; then
  echo "arm: there is no Python '$ARM' run to compare against under $RUNS." >&2
  echo "arm: pass JT_REF_RUNS=\"<run dir> [<run dir> ...]\" (each needs report.json and pred-test-cal.jsonl)." >&2
  echo "arm: the run itself is done: $RUN"
  echo "arm: FAIL — no Python arm reference (this is not a gate failure; the comparison did not happen)"
  exit 2
fi
echo "=== score: the Python arm's calibrated predictions for the comparison"
for d in $refs; do
  n=$(basename "$d")
  "$PY" "$SCORE" --test "$DATA/test.jsonl" "py-$n=$d/pred-test-cal.jsonl" --json "$RUN/ref/$n.json" \
    > "$RUN/ref/$n.log" 2>&1 || { echo "arm: score.py failed on $d (log $RUN/ref/$n.log)" >&2; exit 2; }
  echo "  $n"
done

# ---- 6. design §10.3.3's gates -------------------------------------------------------------------------------------
echo "=== compare: design §10.3.3, jefftrain vs the Python $ARM arm"
"$PY" - "$RUN" "$ARM" "$SEED" "$DEV_TOL" "$TEST_TOL" "${JT_PRIMARY_RUN:-}" $refs <<'PY'
import json, os, sys

run, arm, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
dev_tol, test_tol = float(sys.argv[4]), float(sys.argv[5])
want_primary, ref_dirs = sys.argv[6], sys.argv[7:]

ours = json.load(open(os.path.join(run, "c", "score.json")))
our_rep = json.load(open(os.path.join(run, "c", "report.json")))
key = "jefftrain-" + arm
if key not in ours:
    print(f"arm: {run}/c/score.json has no '{key}' (it has {sorted(ours)})")
    raise SystemExit(2)
mine = ours[key]

def load(d):
    b = os.path.basename(d.rstrip("/"))
    sc = json.load(open(os.path.join(run, "ref", b + ".json")))
    name = next(k for k in sc if k.startswith("py-"))
    return b, sc[name], json.load(open(os.path.join(d, "report.json")))

refs = [load(d) for d in ref_dirs]
primary = refs[-1]
if want_primary:
    hit = [r for r in refs if r[0] == os.path.basename(want_primary.rstrip("/"))]
    if not hit:
        print(f"arm: JT_PRIMARY_RUN={want_primary} is not one of the runs scored ({[r[0] for r in refs]})")
        raise SystemExit(2)
    primary = hit[0]
else:  # the fairest reference is the Python run at the same --seed: same row order, same trajectory
    hit = [r for r in refs if r[2].get("args", {}).get("seed") == seed]
    if hit:
        primary = hit[-1]
pname, pscore, prep = primary

def num(d):
    return float(d) if d is not None else float("nan")

our_dev, py_dev = num(our_rep.get("best_dev_soft_ce")), num(prep.get("best_dev_soft_ce"))
d_dev, dev_ok = abs(our_dev - py_dev), abs(our_dev - py_dev) <= dev_tol
our_ce, py_ce = num(mine["jev_soft_ce"]), num(pscore["jev_soft_ce"])
d_ce, ce_ok = abs(our_ce - py_ce), abs(our_ce - py_ce) <= test_tol

print("=== the two CE gates of design §10.3.3")
print(f"  dev soft CE, all 1,722 development rows at the chosen step (report.json best_dev_soft_ce)")
print(f"    jefftrain {our_dev:.5f}   python {py_dev:.5f} ({pname})   |delta| {d_dev:.5f}   "
      f"gate {dev_tol}   {'OK' if dev_ok else 'OVER'}")
print(f"  test soft CE, calibrated (score.py jev_soft_ce on the 160 test rows)")
print(f"    jefftrain {our_ce:.5f}   python {py_ce:.5f} ({pname})   |delta| {d_ce:.5f}   "
      f"gate {test_tol}   {'OK' if ce_ok else 'OVER'}")
t1 = [num(r[2].get("eval_test", {}).get("soft_ce")) for r in refs]
print(f"  (context, not a gate: T=1 test soft CE jefftrain "
      f"{num(our_rep.get('eval_test', {}).get('soft_ce')):.5f}, python {num(prep.get('eval_test', {}).get('soft_ce')):.5f};"
      f" the Python arm's own seed-to-seed spread is {max(t1) - min(t1):.5f} at T=1 and "
      f"{max(num(r[1]['jev_soft_ce']) for r in refs) - min(num(r[1]['jev_soft_ce']) for r in refs):.5f} calibrated)")

METRICS = [  # (the JSON path as keys — "th0.5" is one key, not "th0" and "5" —, count or ratio)
    (("jev_argmax_agreement",), "ratio"),
    (("th0.5", "precision"), "ratio"), (("th0.5", "recall"), "ratio"), (("th0.5", "f1"), "ratio"),
    (("th0.5", "kind_acc"), "ratio"), (("th0.9", "precision"), "ratio"), (("th0.9", "recall"), "ratio"),
    (("th0.5", "maybe_cards"), "count"), (("th0.5", "tp"), "count"), (("th0.5", "fp"), "count"),
    (("th0.5", "fn"), "count"), (("th0.9", "maybe_cards"), "count"), (("th0.9", "tp"), "count"),
    (("th0.9", "fp"), "count"), (("th0.9", "fn"), "count"),
]

def leaf(d, keys):
    v = d
    for k in keys:
        v = v[k]
    return float(v)

print("=== score.py's metrics against the two-seed band")
print(f"| metric | jefftrain | python({pname}) | seed band | gate | |delta| | verdict |")
print("|---|---:|---:|---:|---:|---:|---|")
metrics_ok = True
for keys, kind in METRICS:
    vals = [leaf(r[1], keys) for r in refs]
    band = max(vals) - min(vals)
    # the resolution floors: a ratio moves in steps of one line (1/160); a count moves in cards, and one card is
    # below what score.py's thresholds can resolve, so a count may drift by 2 (see the header's 2026-11-10 note)
    gate = max(band, 2.0 if kind == "count" else 1.0 / 160.0)
    a, b = leaf(mine, keys), leaf(pscore, keys)
    delta, ok = abs(a - b), abs(a - b) <= gate
    metrics_ok = metrics_ok and ok
    f = (lambda v: f"{v:.0f}") if kind == "count" else (lambda v: f"{v:.3f}")
    print(f"| {'.'.join(keys)} | {f(a)} | {f(b)} | {f(band)} | {f(gate)} | {f(delta)} | {'OK' if ok else 'OVER'} |")

steps, secs = int(our_rep.get("steps", 0)), num(our_rep.get("train_seconds"))
print(f"=== cost: {secs:.0f} s over {steps} step(s) = {secs / max(1, steps):.3f} s/step, "
      f"{our_rep.get('rows_per_second')} rows/s, peak {our_rep.get('peak_gb')} GB")
print(f"=== references: {', '.join(r[0] for r in refs)} (primary {pname}); jefftrain report {run}/c/report.json")
ok = dev_ok and ce_ok and metrics_ok
print(f"arm: {'PASS' if ok else 'FAIL'} — design §10.3.3 over {arm}: dev soft CE |delta| {d_dev:.5f} "
      f"(gate {dev_tol}), test soft CE |delta| {d_ce:.5f} (gate {test_tol}), score.py metrics "
      + ("inside the two-seed band" if metrics_ok else "OUTSIDE the two-seed band (see the table)"))
raise SystemExit(0 if ok else 1)
PY
REMOTE
