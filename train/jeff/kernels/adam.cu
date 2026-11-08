// kernels/adam.cu — the optimizer kernels (plan M2.6, design §2.2 and §8.1). Only the __global__ kernels and the
// extern "C" launch wrappers live here; each kernel is ref/adam.c's loop with the reduction spread over threads.
//
// The gradient norm is the one place where float addition order matters, so it is a fixed two-pass reduction: block b
// owns the arithmetic progression i = b, b+nblocks, b+2*nblocks, ... (nblocks = K_ADAM_BLOCKS = 1024, so the first pass
// has exactly 1024 blocks), reduces its slice with warp xor-butterflies and shared memory — the same shape as the norms
// in kernels/norm.cu — and writes one partial; the second pass is one block over the 1024 partials. There is no
// atomicAdd anywhere, so two runs of the same launch agree bit for bit, which is what lets a test compare it with a
// reference at all (design §8.1).
//
// The arithmetic of one Adam element is design §2.2 word for word:
//   m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;  p -= lr * (m/(1-b1^t)) / (sqrt(v/(1-b2^t)) + eps)
// with the bias corrections computed on the host in double and handed over as floats (plan M2.6). sqrtf, not __fsqrt_rn:
// the reference's r_sqrt is a correctly rounded double sqrt, and two ulp is far below any gate here.
#include <cuda_runtime.h>

#include "adam.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

#define ADAM_WARP 32

// Sum across a warp's lanes; every lane holds the total afterwards (xor butterflies, as kernels/norm.cu does).
static __device__ __forceinline__ float adam_warp_sum(float v) {
#pragma unroll
  for (int off = ADAM_WARP / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sum across the block: warp shuffles, the warps' partials through smem, then warp 0 adds those. smem needs ADAM_WARP
// floats. The trailing barrier leaves smem reusable.
static __device__ __forceinline__ float adam_block_sum(float v, float *smem) {
  const int lane = (int)(threadIdx.x & (ADAM_WARP - 1));
  const int warp = (int)(threadIdx.x / ADAM_WARP);
  const int nwarps = (int)((blockDim.x + ADAM_WARP - 1) / ADAM_WARP);
  v = adam_warp_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (threadIdx.x < (unsigned)nwarps) ? smem[threadIdx.x] : 0.0f;
    t = adam_warp_sum(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  const float out = smem[0];
  __syncthreads();
  return out;
}

// A grid-stride launch shape: at most 65535 blocks, 256 threads (a legal launch for any n).
static inline int adam_grid(int n) {
  const long g = ((long)n + K_ADAM_THREADS - 1) / K_ADAM_THREADS;
  return g > 65535 ? 65535 : (int)g;
}

// ---- sumsq: the two passes of the global gradient norm ---------------------------------------------------------------
// Block b sums the slice {b + m*nblocks}, its 256 threads taking every 256th element of that slice. That is exactly the
// element set ref_sumsq_partial() sums sequentially, so the partials are comparable one by one, not only their sum.
__global__ void sumsq_partial_kernel(const float *g, int n, float *partials) {
  __shared__ float smem[ADAM_WARP];
  const long nb = gridDim.x;
  const long step = nb * blockDim.x;
  float s = 0.0f;
  for (long i = (long)blockIdx.x + (long)threadIdx.x * nb; i < n; i += step) s += g[i] * g[i];
  s = adam_block_sum(s, smem);
  if (threadIdx.x == 0) partials[blockIdx.x] = s;
}

// One block, every thread striding over the partials by blockDim, then the same block reduction.
__global__ void sumsq_final_kernel(const float *partials, int nblocks, float *out) {
  __shared__ float smem[ADAM_WARP];
  float s = 0.0f;
  for (int i = (int)threadIdx.x; i < nblocks; i += (int)blockDim.x) s += partials[i];
  s = adam_block_sum(s, smem);
  if (threadIdx.x == 0) out[0] = s;
}

extern "C" void k_sumsq_partial(const float *g, int n, float *partials, int nblocks, cudaStream_t s) {
  if (n <= 0 || nblocks <= 0) return;
  if (nblocks > 65535) nblocks = 65535; // more blocks than partials can name would be a caller bug; the cap keeps it legal
  sumsq_partial_kernel<<<nblocks, K_ADAM_THREADS, 0, s>>>(g, n, partials);
  JT_SYNC_DEBUG;
}

extern "C" void k_sumsq_final(const float *partials, int nblocks, float *out, cudaStream_t s) {
  if (nblocks <= 0 || out == NULL) return;
  sumsq_final_kernel<<<1, K_ADAM_THREADS, 0, s>>>(partials, nblocks, out);
  JT_SYNC_DEBUG;
}

// ---- clip: g *= coef, the one launch train.py's clip_grad_norm_ becomes -----------------------------------------------
__global__ void clip_scale_kernel(float *g, int n, float coef) {
  const int stride = (int)(gridDim.x * blockDim.x);
  for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += stride) g[i] *= coef;
}

extern "C" void k_clip_scale(float *g, int n, float coef, cudaStream_t s) {
  if (n <= 0) return;
  clip_scale_kernel<<<adam_grid(n), K_ADAM_THREADS, 0, s>>>(g, n, coef);
  JT_SYNC_DEBUG;
}

// ---- the Adam step (design §2.2) --------------------------------------------------------------------------------------
__global__ void adamw_step_kernel(float *p, const float *g, float *m, float *v, int n, int lr_group_split, float lr_lora,
                                  float lr_readout, float b1, float b2, float eps, float bias1, float bias2) {
  const int stride = (int)(gridDim.x * blockDim.x);
  const float one_m_b1 = 1.0f - b1, one_m_b2 = 1.0f - b2;
  for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x); i < n; i += stride) {
    const float lr = i < lr_group_split ? lr_lora : lr_readout; // LoRA first, the readout last (design §8.1)
    const float gi = g[i];
    const float mi = b1 * m[i] + one_m_b1 * gi;
    const float vi = b2 * v[i] + one_m_b2 * gi * gi;
    m[i] = mi;
    v[i] = vi;
    const float mhat = mi / bias1;
    const float vhat = vi / bias2;
    p[i] -= lr * mhat / (sqrtf(vhat) + eps);
  }
}

extern "C" void k_adamw_step(float *p, const float *g, float *m, float *v, int n, int lr_group_split, float lr_lora,
                             float lr_readout, float b1, float b2, float eps, float bias1, float bias2,
                             cudaStream_t s) {
  if (n <= 0) return;
  adamw_step_kernel<<<adam_grid(n), K_ADAM_THREADS, 0, s>>>(p, g, m, v, n, lr_group_split, lr_lora, lr_readout, b1, b2,
                                                            eps, bias1, bias2);
  JT_SYNC_DEBUG;
}
