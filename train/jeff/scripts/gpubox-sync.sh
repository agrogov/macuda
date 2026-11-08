#!/bin/sh
# gpubox-sync.sh — mirror train/jeff/ to the Linux box. No data, no weights, no build output.
#   JT_REMOTE=macuda-train/wt-J01 bash scripts/gpubox-sync.sh     # parallel tasks need separate directories
# The remote path is relative to ~, and --delete means two tasks must never share one.
set -eu
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
JEFF=$(cd "$(dirname "$0")/.." && pwd)
ssh gpubox "mkdir -p ~/$JT_REMOTE"        # rsync --delete will not create the parent of its target
rsync -a --delete --exclude build/ "$JEFF/" "gpubox:$JT_REMOTE/"
echo "synced $JEFF/ -> gpubox:~/$JT_REMOTE/"
