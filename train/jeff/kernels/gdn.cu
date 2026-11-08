// kernels/gdn.cu — the GDN (gated delta net) recurrence, forward (plan M3.2; design §3.6 the loop, §8.3 the kernel).
// Only the __global__ kernel and the extern "C" launch wrapper live here; the C side includes gdn.h and never sees CUDA
// syntax (design §11.1). The oracle is ref/gdn.c (plan M3.1) and test/test_gdn.c compares the two.
//
// Shape (design §8.3): grid (B, H, 4 column tiles), one 32-thread warp per block. Lane l owns value column
// j = tile*32 + l and keeps that column of the state, `float s[128]`, in registers for the whole row. A value column of S
// evolves independently in the forward (design §3.6: kv[j], delta[j] and o[t,j] only touch column j), so there is no
// cross-lane reduction anywhere: kv[j] and o[t,j] are this lane's own serial sums over the key index i.
//
// Registers. s[] stays in registers only while every index into it is a compile-time constant after unrolling, so the
// loops over i are `#pragma unroll` with JT_GDN_DIM = 128 a macro and no runtime index is ever applied to s[] (design
// §8.3, plan M3.2). Measured on gpubox (sm_120, `nvcc -O2 -arch=sm_120 -std=c++17 -Xptxas -v kernels/gdn.cu`):
//
//   gdn_fwd_kernel: 201 registers, 0 bytes spill stores, 0 bytes spill loads, 1024 bytes smem
//
// One warp per block keeps that inside the register file: 201 * 32 = 6432 of an SM's 65536 registers, i.e. 10 blocks per
// SM, and 0 spill is the gate the plan sets. The split of each 128-iteration loop into two halves is what buys the zero:
// as one unrolled block ptxas prefetches all 128 shared loads of the sum and needs 255 registers plus 68 bytes of spill
// stores (52 loads). Two 64-iteration halves with a __syncwarp() between them — a scheduling barrier, and every lane of
// the warp reaches it, so it is free of divergence — leave 201 registers and no spill. The sums still run i = 0, 1, ...
// 127 ascending, so the arithmetic is the same as one loop.
//
// Checkpoints (design §8.3): before the step of every token whose row-local index is a multiple of JT_GDN_SEG the block
// writes its 128 values of S to ckpt[b][h][tl/64][i][j]. Because it is written *before* the step, segment 0's slot is all
// zeros; the backward (plan M3.3) starts each segment from it instead of from token 0.
#include <cuda_runtime.h>
#include <stddef.h>

#include "gdn.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

__global__ void gdn_fwd_kernel(const float *q, const float *k, const float *v, const float *g, const float *beta,
                               const int *off, int H, int nseg, float *ckpt, float *o) {
  __shared__ float k_s[JT_GDN_DIM]; // the token's k[t,h,:], so every lane reads all 128 key entries
  __shared__ float q_s[JT_GDN_DIM]; // and q[t,h,:]
  const int b = (int)blockIdx.x;
  const int h = (int)blockIdx.y;
  const int lane = (int)threadIdx.x;
  const int j = (int)blockIdx.z * JT_GDN_TILE + lane; // this lane's value column
  const int t0 = off[b], t1 = off[b + 1];              // design §7: row b is [off[b], off[b+1])

  float s[JT_GDN_DIM]; // column j of S[i,j] for i = 0..127, live across the whole row
#pragma unroll
  for (int i = 0; i < JT_GDN_DIM; i++) s[i] = 0.0f; // S_0 = 0 at every row start, per head (design §3.6)

  for (int t = t0; t < t1; t++) {
    const size_t gi = (size_t)t * (size_t)H + (size_t)h; // token-major [N][H]
    const float *k_t = k + gi * JT_GDN_DIM;
    const float *q_t = q + gi * JT_GDN_DIM;
    // Each lane loads 4 of the 128 entries; the shared arrays are the warp's own, so __syncwarp() is the whole barrier.
#pragma unroll
    for (int c = 0; c < JT_GDN_DIM / JT_GDN_TILE; c++) {
      const int i = lane + c * JT_GDN_TILE;
      k_s[i] = k_t[i];
      q_s[i] = q_t[i];
    }
    __syncwarp(); // the stores are visible, and no lane is still reading the previous token's values

    const int tl = t - t0;
    if (ckpt != NULL && (tl & (JT_GDN_SEG - 1)) == 0) {
      float *c = ckpt + (((size_t)b * (size_t)H + (size_t)h) * (size_t)nseg + (size_t)(tl / JT_GDN_SEG)) * JT_GDN_CKPT_FLOATS;
#pragma unroll
      for (int i = 0; i < JT_GDN_DIM; i++) c[(size_t)i * JT_GDN_DIM + (size_t)j] = s[i]; // S_prev, i.e. before the step
    }

    const float e = expf(g[gi]); // decay of design §3.6: S = S * exp(g[t]), g <= 0
    const float bt = beta[gi];
    const size_t vi = gi * JT_GDN_DIM + (size_t)j;
    float kv = 0.0f;
#pragma unroll
    for (int i = 0; i < JT_GDN_DIM / 2; i++) {
      s[i] *= e;           // S = S * exp(g[t])
      kv += s[i] * k_s[i]; // kv[j] = sum_i S[i,j] * k[t,i]
    }
    __syncwarp(); // the register-pressure barrier explained above; the sum continues at i = 64
#pragma unroll
    for (int i = JT_GDN_DIM / 2; i < JT_GDN_DIM; i++) {
      s[i] *= e;
      kv += s[i] * k_s[i];
    }
    const float d = (v[vi] - kv) * bt; // delta[j] = (v[t,j] - kv[j]) * beta[t]
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < JT_GDN_DIM / 2; i++) {
      s[i] += k_s[i] * d;   // S[i,j] += k[t,i] * delta[j]
      acc += s[i] * q_s[i]; // o[t,j] = sum_i S[i,j] * q[t,i], with the S_t the line above just made
    }
    __syncwarp();
#pragma unroll
    for (int i = JT_GDN_DIM / 2; i < JT_GDN_DIM; i++) {
      s[i] += k_s[i] * d;
      acc += s[i] * q_s[i];
    }
    o[vi] = acc;
    __syncwarp(); // every lane is done with k_s/q_s before the next token overwrites them
  }
}

extern "C" void gdn_fwd(const float *q, const float *k, const float *v, const float *g, const float *beta, const int *off,
                        int B, int H, int nseg, float *ckpt, float *o, cudaStream_t s) {
  if (B <= 0 || H <= 0) return;
  if (q == NULL || k == NULL || v == NULL || g == NULL || beta == NULL || off == NULL || o == NULL) return;
  // A checkpoint buffer needs a segment count to be indexed with: without one, run the pass that computes o only rather
  // than write a buffer whose stride is not known. gdn_ckpt_segs() is the helper that produces a valid nseg.
  if (ckpt != NULL && nseg <= 0) ckpt = NULL;
  const dim3 grid((unsigned)B, (unsigned)H, JT_GDN_TILES);
  gdn_fwd_kernel<<<grid, JT_GDN_TILE, 0, s>>>(q, k, v, g, beta, off, H, nseg, ckpt, o);
  JT_SYNC_DEBUG;
}
