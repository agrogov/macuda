#!/bin/sh
# gemm.cubin FROM SOURCE, and the check that it is the committed one (G's review of the bf16 fix: Makefile:63 embeds a
# supplied cubin, so its provenance has to be a recipe anyone can re-run). Runs where nvcc is: a Linux box, or the
# nvidia/cuda:13.0.3-devel container cuda-shim/build/tinycc compiles device code with (the Mac has no nvcc).
#   sh build-gemm-cubin.sh <gemm.cu> <committed gemm.cubin> [out dir] [arch] [nvcc]
# Builds the cubin twice from the same source with the pinned toolkit and flags, requires the two builds to be
# byte-identical (a build that is not reproducible cannot vouch for a committed binary), then compares the result with
# the committed cubin. <arch> defaults to sm_120, the committed sm_120 cubin's own target; sm_86 builds the Ampere set
# Ada (sm_89) also runs. <nvcc> defaults to the pinned toolkit path, and NVCC overrides both (the container keeps nvcc at
# /usr/local/cuda/bin/nvcc).
set -eu
NVCC=${5:-${NVCC:-/usr/local/cuda-13.0/bin/nvcc}}
SRC=$1; REF=$2; OUT=${3:-$(mktemp -d)}; ARCH=${4:-sm_120}; mkdir -p "$OUT"
echo "nvcc:    $NVCC"
echo "toolkit: $($NVCC --version | tail -1)"
echo "arch:    $ARCH"
echo "source:  $(sha256sum "$SRC")"
$NVCC -cubin -arch=$ARCH -O3 -o "$OUT/a.cubin" "$SRC"
$NVCC -cubin -arch=$ARCH -O3 -o "$OUT/b.cubin" "$SRC"
a=$(sha256sum < "$OUT/a.cubin" | cut -c1-64); b=$(sha256sum < "$OUT/b.cubin" | cut -c1-64); r=$(sha256sum < "$REF" | cut -c1-64)
echo "build 1: $a"; echo "build 2: $b"; echo "committed: $r"
[ "$a" = "$b" ] || { echo "NOT REPRODUCIBLE: two builds of one source differ"; exit 1; }
[ "$a" = "$r" ] && echo "REPRODUCED: the committed gemm.cubin is this source built with this toolkit" || { echo "DIFFERS from the committed cubin"; exit 1; }
