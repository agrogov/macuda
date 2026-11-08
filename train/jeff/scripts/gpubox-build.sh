#!/bin/sh
# gpubox-build.sh — sync, then `make nv` on the Linux box with real nvcc and cuBLAS.
#   ARCH=sm_120 bash scripts/gpubox-build.sh
# nvcc is not on a bare ssh PATH there, so it is added explicitly.
set -eu
JT_REMOTE=${JT_REMOTE:-macuda-train/jeff}
ARCH=${ARCH:-sm_120}
sh "$(dirname "$0")/gpubox-sync.sh"
ssh gpubox "export PATH=/usr/local/cuda/bin:\$PATH; cd ~/$JT_REMOTE && make nv ARCH=$ARCH"
