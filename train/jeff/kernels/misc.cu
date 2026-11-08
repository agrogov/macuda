// kernels/misc.cu — plumbing kernels (design §8.1). Only the __global__ kernels and the extern "C" launch wrappers live
// here; the C host calls jt_fill() and never sees CUDA syntax (design §11.1). tinycc compiles this file twice over: the
// device half with nvcc, the host half with clang, and embeds the fatbin in the .o (see the shim's build/tinycc).
#include <cuda_runtime.h>
#include <stddef.h>

#include "misc.h"

// M0.2: the hello kernel. Grid-stride, so the launch shape never has to fit n (1M floats here, 248k-token tables later).
__global__ void jt_fill_kernel(float *x, float v, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] = v;
}

extern "C" void jt_fill(float *x, float v, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  const size_t want = (n + block - 1) / block;
  const int grid = want > 65535u ? 65535 : (int)want;   // a grid-stride kernel needs no more than this
  jt_fill_kernel<<<grid, block, 0, s>>>(x, v, n);
}
