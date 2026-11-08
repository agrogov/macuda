// kernels/head.cu — the decision head's softmax and per-row loss (plan M2.5, design §4.1).
//
// One block of 256 threads per row. The block finds the row's max (log-sum-exp, so a dead option's -1e9 does not poison
// the sum), the normaliser, then sum(t) and the loss, and finally writes p and dlogits. All four reductions are fixed
// fp32 tree reductions in shared memory — no atomicAdd, so the bits do not depend on scheduling (design §8.1). The CPU
// oracle (ref/head.c) accumulates in double; this is within the plan's 1e-5 forward gate of it.
#include <cuda_runtime.h>

#include "head.h"

#define JT_HEAD_BLOCK 256 // power of two; also wider than the widest real row (255 options)

#define JT_HEAD_NEG (-3.4028234663852886e+38f)

static __device__ __forceinline__ float head_block_max(float v, float *sh) {
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int off = (int)blockDim.x >> 1; off > 0; off >>= 1) {
    if ((int)threadIdx.x < off) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + off]);
    __syncthreads();
  }
  const float r = sh[0];
  __syncthreads();
  return r;
}

static __device__ __forceinline__ float head_block_sum(float v, float *sh) {
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

__global__ void ce_head_kernel(const float *logits, int n, const float *targets, const float *wrow, float wgroup, int B,
                               float *loss_out, float *dlogits, float *probs_out) {
  __shared__ float sh[JT_HEAD_BLOCK];
  const int tid = (int)threadIdx.x;
  const int step = (int)blockDim.x;
  for (int r = blockIdx.x; r < B; r += gridDim.x) {
    const size_t off = (size_t)r * (size_t)n;
    const float *lg = logits + off;
    const float *tg = targets + off;
    float *dg = dlogits != NULL ? dlogits + off : NULL;
    float *pg = probs_out != NULL ? probs_out + off : NULL;

    float v = JT_HEAD_NEG;
    for (int j = tid; j < n; j += step) v = fmaxf(v, lg[j]);
    const float m = head_block_max(v, sh);

    float acc = 0.0f;
    for (int j = tid; j < n; j += step) acc += expf(lg[j] - m);
    // log-sum-exp, so a dead option's log p is the finite (-1e9 - lse) torch's log_softmax gives it: 0 * that is 0,
    // while 0 * -inf would be NaN. Its exp() is exactly 0 either way.
    const float lse = m + logf(head_block_sum(acc, sh));

    float tacc = 0.0f, cacc = 0.0f;
    for (int j = tid; j < n; j += step) {
      tacc += tg[j];
      cacc -= tg[j] * (lg[j] - lse);
    }
    const float tsum = head_block_sum(tacc, sh);
    const float ce = head_block_sum(cacc, sh);
    const float w = wrow[r] / wgroup;

    for (int j = tid; j < n; j += step) {
      const float p = expf(lg[j] - lse);
      if (pg != NULL) pg[j] = p;
      if (dg != NULL) dg[j] = w * (p * tsum - tg[j]);
    }
    if (tid == 0) loss_out[r] = w * ce;
  }
}

extern "C" void ce_head(const float *logits, int n_opts, const float *targets, const float *wrow, float wgroup, int B,
                        float *loss_out, float *dlogits, float *probs_out, cudaStream_t s) {
  if (n_opts <= 0 || B <= 0) return;
  const int grid = B < 65535 ? B : 65535;
  ce_head_kernel<<<grid, JT_HEAD_BLOCK, 0, s>>>(logits, n_opts, targets, wrow, wgroup, B, loss_out, dlogits,
                                               probs_out);
}
