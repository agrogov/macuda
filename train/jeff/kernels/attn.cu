// kernels/attn.cu — the causal attention softmax (plan M2.5, design §4.7, §8.1 row `attn_softmax_fwd/bwd`, §8.2).
//
// One block of 256 threads per (head, query t) row of the [heads][L][L] score matrix. The row is only s <= t, so the
// block makes two passes (max, then sum) over its live prefix, block-reduces each with shared memory, and writes the
// normalised probabilities — including the exact zeros above the diagonal.
//
// Reductions are plain fp32 tree reductions in shared memory, never atomicAdd: the order is fixed, so the same input
// gives the same bits on every run (design §8.1). The CPU oracle (ref/attn.c) accumulates in double, which is what a
// reference should do; the fp32 tree here is within the plan's 1e-5 forward / 1e-4 backward gates of it.
#include <cuda_runtime.h>

#include "attn.h"

#define JT_ATTN_BLOCK 256 // power of two: the tree below halves the width until one lane is left

// -FLT_MAX without <float.h>'s macro games in a device translation unit. A macro, not a __device__ global: there is no
// device symbol table to fill (design §8 forbids __constant__ too).
#define JT_ATTN_NEG (-3.4028234663852886e+38f)

// Block-wide max. `sh` must hold JT_ATTN_BLOCK floats; every thread must call it with the same blockDim.x.
static __device__ __forceinline__ float attn_block_max(float v, float *sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int off = (int)blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + off]);
    __syncthreads();
  }
  const float r = sh[0];
  __syncthreads(); // all threads have read sh[0] before the caller reuses the array
  return r;
}

// Block-wide sum, in the same fixed tree order.
static __device__ __forceinline__ float attn_block_sum(float v, float *sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int off = (int)blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) sh[threadIdx.x] += sh[threadIdx.x + off];
    __syncthreads();
  }
  const float r = sh[0];
  __syncthreads();
  return r;
}

// Grid-stride over the rows so that the launch never depends on heads*L fitting an int-sized grid.
__global__ void attn_softmax_fwd_kernel(const float *S, float *P, int L, int nrows, float scale) {
  __shared__ float sh[JT_ATTN_BLOCK];
  const int tid = (int)threadIdx.x;
  for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
    const int t = row % L; // row = h * L + t
    const size_t off = (size_t)row * (size_t)L;
    const float *sr = S + off;
    float *pr = P + off;

    float v = JT_ATTN_NEG;
    for (int s = tid; s <= t; s += (int)blockDim.x) v = fmaxf(v, sr[s] * scale);
    const float m = attn_block_max(v, sh);

    float acc = 0.0f;
    for (int s = tid; s <= t; s += (int)blockDim.x) acc += expf(sr[s] * scale - m);
    const float inv = 1.0f / attn_block_sum(acc, sh);

    for (int s = tid; s < L; s += (int)blockDim.x) pr[s] = (s <= t) ? expf(sr[s] * scale - m) * inv : 0.0f;
  }
}

__global__ void attn_softmax_bwd_kernel(const float *P, const float *dP, float *dS, int L, int nrows, float scale) {
  __shared__ float sh[JT_ATTN_BLOCK];
  const int tid = (int)threadIdx.x;
  for (int row = blockIdx.x; row < nrows; row += gridDim.x) {
    const int t = row % L;
    const size_t off = (size_t)row * (size_t)L;
    const float *pr = P + off;
    const float *dr = dP + off;
    float *orow = dS + off;

    float acc = 0.0f;
    for (int s = tid; s <= t; s += (int)blockDim.x) acc += pr[s] * dr[s];
    const float dot = attn_block_sum(acc, sh);

    for (int s = tid; s < L; s += (int)blockDim.x)
      orow[s] = (s <= t) ? scale * pr[s] * (dr[s] - dot) : 0.0f;
  }
}

// How many blocks to launch for `nrows` rows: one per row, capped (a grid-stride loop covers the rest).
static int attn_grid(int nrows) { return nrows < 65535 ? nrows : 65535; }

extern "C" void attn_softmax_fwd(const float *S, float *P, int L, int heads, float scale, cudaStream_t s) {
  if (L <= 0 || heads <= 0) return;
  const int nrows = heads * L;
  attn_softmax_fwd_kernel<<<attn_grid(nrows), JT_ATTN_BLOCK, 0, s>>>(S, P, L, nrows, scale);
}

extern "C" void attn_softmax_bwd(const float *P, const float *dP, float *dS, int L, int heads, float scale,
                                 cudaStream_t s) {
  if (L <= 0 || heads <= 0) return;
  const int nrows = heads * L;
  attn_softmax_bwd_kernel<<<attn_grid(nrows), JT_ATTN_BLOCK, 0, s>>>(P, dP, dS, L, nrows, scale);
}
