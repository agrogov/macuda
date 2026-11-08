// kernels/rope.cu — the RoPE kernel (plan M2.4; design §3.3 the rotation, §4.7 "then RoPE backward (the rotation
// transposed: same formula with s -> -s)", §7 positions, §8.1 "one thread per (token, head, j<32)").
//
// Only the __global__ kernel, the extern "C" launch wrappers and the host table builder live here; the C side includes
// rope.h and never sees CUDA syntax (design §11.1). ref/rope.c is the same arithmetic, and test/k_rope.c compares them.
//
// One launch covers the whole packed batch: block b is token b (grid.x = n — tokens do not fit grid.y's 65535, a 64-row
// batch of 1026-token rows is 65664), and the block's threads stride over the heads * 32 (head, j) pairs of that token.
// A thread reads x[j] and x[j + 32] of its head and writes both back, so the block's reads are 32 consecutive floats per
// warp twice over: no shared memory, no cross-thread reduction, nothing to synchronise.
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

#include "rope.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

// ssign is +1 for the forward and -1 for the backward: y[j] = a*c - b*s, y[j+32] = b*c + a*s with s already negated is
// the transposed rotation, so one kernel serves both directions and they cannot drift apart.
__global__ void jt_rope_kernel(float *x, const int *row_off, const int *row_of, int heads, int head_dim,
                               const float *cosT, const float *sinT, float ssign) {
  const int t = (int)blockIdx.x;
  const size_t p = (size_t)(t - row_off[row_of[t]]); // position inside the row, 0-based (design §7)
  const float *c = cosT + p * JT_ROPE_DIM;
  const float *s = sinT + p * JT_ROPE_DIM;

  float *xt = x + (size_t)t * (size_t)heads * (size_t)head_dim;
  const int pairs = heads * JT_ROPE_HALF; // (head, j) pairs of this token
  for (int idx = (int)threadIdx.x; idx < pairs; idx += (int)blockDim.x) {
    const int h = idx / JT_ROPE_HALF;
    const int j = idx - h * JT_ROPE_HALF;
    float *xh = xt + (size_t)h * (size_t)head_dim;
    const float a = xh[j], b = xh[j + JT_ROPE_HALF];
    const float sn = s[j] * ssign;
    xh[j] = a * c[j] - b * sn;
    xh[j + JT_ROPE_HALF] = b * c[j] + a * sn;
  }
}

void rope_tables(float *cos, float *sin, int lmax) {
  if (lmax <= 0) return;
  for (int k = 0; k < JT_ROPE_HALF; k++) {
    // fp32 throughout, in torch's order: 1.0 / (theta ** (arange(0, 64, 2).float() / 64)), theta = 1e7 (design §3.3).
    const float inv = 1.0f / powf(1e7f, (float)(2 * k) / (float)JT_ROPE_DIM);
    for (int p = 0; p < lmax; p++) {
      const float ang = (float)p * inv;
      const float c = cosf(ang), s = sinf(ang);
      cos[(size_t)p * JT_ROPE_DIM + k] = c;
      cos[(size_t)p * JT_ROPE_DIM + k + JT_ROPE_HALF] = c;
      sin[(size_t)p * JT_ROPE_DIM + k] = s;
      sin[(size_t)p * JT_ROPE_DIM + k + JT_ROPE_HALF] = s;
    }
  }
}

// One block per token, 256 threads; nothing to reduce, so blockDim only has to cover heads * 32 pairs when it can
// (the model's 8 heads * 32 = 256 exactly) and the stride loop covers any other head count.
static void rope_launch(float *x, const int *row_off, const int *row_of, int n, int heads, int head_dim,
                        const float *cos, const float *sin, float ssign, cudaStream_t s) {
  if (n <= 0 || head_dim < JT_ROPE_DIM) return;
  const int block = 256;
  jt_rope_kernel<<<n, block, 0, s>>>(x, row_off, row_of, heads, head_dim, cos, sin, ssign);
  JT_SYNC_DEBUG;
}

void rope_fwd(float *x, const int *row_off, const int *row_of, int n, int heads, int head_dim, const float *cos,
              const float *sin, cudaStream_t s) {
  rope_launch(x, row_off, row_of, n, heads, head_dim, cos, sin, 1.0f, s);
}

void rope_bwd(float *dx, const int *row_off, const int *row_of, int n, int heads, int head_dim, const float *cos,
              const float *sin, cudaStream_t s) {
  rope_launch(dx, row_off, row_of, n, heads, head_dim, cos, sin, -1.0f, s);
}
