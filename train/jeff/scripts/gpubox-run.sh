#!/bin/sh
# gpubox-run.sh — build on the Linux box, then run build/nv/<binary> there and stream its output.
#   bash scripts/gpubox-run.sh test_hello
#   bash scripts/gpubox-run.sh jefftrain --model ~/models/jeff-base ... 
#   JT_REMOTE=macuda-train/wt-J01 bash scripts/gpubox-run.sh test_hello
set -eu
[ $# -ge 1 ] || { echo "usage: sh scripts/gpubox-run.sh <binary> [args...]" >&2; exit 2; }
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
bin=$1; shift
sh "$(dirname "$0")/gpubox-build.sh"
# quote every argument once for the remote shell
cmd="cd ~/$JT_REMOTE && ./build/nv/$bin"
for a in "$@"; do
  q=$(printf '%s' "$a" | sed "s/'/'\\\\''/g")
  cmd="$cmd '$q'"
done
exec ssh gpubox "$cmd"
