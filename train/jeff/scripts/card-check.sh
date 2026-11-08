#!/bin/sh
# One card checkpoint; the user's OK covers all its steps. DRY=1 uses the null device: no preflight, no card.
# usage: sh scripts/card-check.sh <A|B> <m0|m1|m2|m3-fwd|m3|m4|m5>
# m4 and m5 also need JT_REF (ref.safetensors) and JT_DUMP_ARGS (--model/--init-adapter/--train flags, Mac paths).
set -u
WHO=${1:?A|B}; CP=${2:?checkpoint}
JEFF=$(cd "$(dirname "$0")/.." && pwd); B=$JEFF/build/shim
MACUDA=${MACUDA:-$HOME/Downloads/vfio/macuda}   # the main checkout: the built shim, tools/ and logs/ are there
PY=${JT_PY:-$HOME/Downloads/CLM-8B/.venv/bin/python}
EXACT="JT_EXACT=1 TINYCUBLAS_TC=0"
n=0; failed=0
finish() { echo "checkpoint $CP: $n steps, $failed failed"; [ "$failed" -eq 0 ] && exit 0; exit 1; }
stop() { echo "STOP: $1"; failed=$((failed+1)); finish; }
# step <env words, or -> <binary> [args...]. A step that never ran, crashed or was void ends the checkpoint.
step() {
  envs=$1; bin=$2; shift 2; [ "$envs" = - ] && envs=
  n=$((n+1)); echo "=== step $n: $envs $(basename "$bin") $*"
  tmp=$(mktemp); env $envs sh "$MACUDA/tools/nv_shim_step.sh" "$WHO" probe "$bin" "$@" > "$tmp" 2>&1; rc=$?; cat "$tmp"
  ran=0; grep -q '   exit [0-9]* after' "$tmp" && ran=1; rm -f "$tmp"
  [ "$ran" = 1 ] || stop "step $n never ran (rc $rc)"
  [ "$rc" -gt 128 ] && stop "step $n crashed (rc $rc): the card's state is unknown"
  [ "$rc" = 97 ] && stop "step $n was void: it did not have the card to itself"
  [ "$rc" = 0 ] || { echo "FAILED: step $n (rc $rc)"; failed=$((failed+1)); }
}
case $CP in
  m0) step - "$B/test_hello" ;;
  m1) step "$EXACT" "$B/test_gemm"; step - "$B/test_gemm" ;;   # exact, then TF32
  m2) step - "$B/test_kernels" all ;;
  m3-fwd) step - "$B/test_gdn" fwd ;;
  m3) step - "$B/test_gdn" all ;;
  m4|m5)
    : "${JT_REF:?path to ref.safetensors}"; : "${JT_DUMP_ARGS:?flags for the dump run}"
    out=$(mktemp -d)/$CP.safetensors; flag=--dump-forward; [ "$CP" = m5 ] && flag=--dump-backward
    step "$EXACT" "$B/jefftrain" $JT_DUMP_ARGS $flag "$out"
    [ "${DRY:-0}" = 1 ] || "$PY" "$JEFF/py/compare_layers.py" "$JT_REF" "$out" --atol-rel 1e-3 --grad-rel 5e-3 \
      || { echo "FAILED: compare_layers"; failed=$((failed+1)); } ;;
  *) echo "unknown checkpoint $CP"; exit 2 ;;
esac
finish
