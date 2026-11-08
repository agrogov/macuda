// kernels/norm.cu — the norm kernels (plan M2.2; design §4.3 RMSNorm, §4.4 gated RMSNorm, §4.8 l2norm, §8.1 shapes).
// Only the __global__ kernels and the extern "C" launch wrappers live here. Each kernel is the row loop of the
// matching ref/norm.c with the reduction spread over threads; nothing else differs.
//
// Reductions (design §8.1): warp shuffles (o = 16,8,4,2,1) then shared memory across the block's warps. No atomicAdd
// anywhere — the sum's order is the tree's, which is fixed, so two runs of the same launch agree bit for bit.
//
// Launch shapes (design §8.1, plan M2.2):
//   rmsnorm_*        d = 1024 -> one 256-thread block per row (4 elements per thread, block reduce)
//                    d <= 256 -> one warp per row (4 elements per thread at d = 128, 8 at d = 256)
//   gated_rmsnorm_*  d = 128, one warp per row, 8 rows (warps) per 256-thread block
//   l2norm_*         the same as gated_rmsnorm: one warp per (token, head)
// A row is always one length-d vector: a token for the token-major norms, a (token, head) pair for the GDN ones.
#include <cuda_runtime.h>
#include <stddef.h>

#include "norm.h"
#include "util.h" // JT_SYNC_DEBUG (a no-op unless -DJT_DEBUG)

#define JT_WARP 32            // one warp per row for the 128/256-dim norms
#define JT_WARPS_PER_BLOCK 8  // 256 threads: eight rows per block (rows need not be a multiple of eight)

// Sum across a warp's 32 lanes; every lane holds the total afterwards (xor butterflies), so no broadcast is needed.
__device__ __forceinline__ float jt_warp_sum(float v) {
#pragma unroll
  for (int off = JT_WARP / 2; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

// Sum across the whole block: warp shuffles, the warps' partials through smem, then warp 0 adds those. smem must have
// JT_WARP floats. The trailing barrier means a second call in the same kernel can reuse the same smem.
__device__ __forceinline__ float jt_block_sum(float v, float *smem) {
  const int lane = (int)(threadIdx.x & (JT_WARP - 1));
  const int warp = (int)(threadIdx.x / JT_WARP);
  const int nwarps = (int)((blockDim.x + JT_WARP - 1) / JT_WARP);
  v = jt_warp_sum(v);
  if (lane == 0) smem[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float t = (threadIdx.x < (unsigned)nwarps) ? smem[threadIdx.x] : 0.0f;
    t = jt_warp_sum(t);
    if (threadIdx.x == 0) smem[0] = t;
  }
  __syncthreads();
  const float out = smem[0];
  __syncthreads();
  return out;
}

// (1 + w) for the zero-centred norms, w for the GDN gated norm (design §3.3 / §4.4).
__device__ __forceinline__ float jt_w_scale(float w, int zero_centred) {
  return zero_centred ? 1.0f + w : w;
}

// silu(z) = z * sigmoid(z) and its derivative sigmoid(z) * (1 + z * (1 - sigmoid(z))). expf, not __expf: these kernels
// are compared against the reference at 1e-4 and are not the bottleneck.
__device__ __forceinline__ float jt_silu(float z) { return z / (1.0f + expf(-z)); }
__device__ __forceinline__ float jt_dsilu(float z) {
  const float sig = 1.0f / (1.0f + expf(-z));
  return sig * (1.0f + z * (1.0f - sig));
}

// ---- RMSNorm (design §4.3) ------------------------------------------------------------------------------------------
__global__ void k_rmsnorm_fwd(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ y,
                              float *__restrict__ r, int rows, int d, int zero_centred) {
  __shared__ float smem[JT_WARP];
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  float *yr = y + base;
  float ss = 0.0f;
  for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) ss = fmaf(xr[i], xr[i], ss);
  const float rb = rsqrtf(jt_block_sum(ss, smem) / (float)d + JT_NORM_EPS_F);
  for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) yr[i] = xr[i] * rb * jt_w_scale(w[i], zero_centred);
  if (threadIdx.x == 0) r[row] = rb;
}

__global__ void k_rmsnorm_bwd(const float *__restrict__ x, const float *__restrict__ w,
                              const float *__restrict__ dy, const float *__restrict__ r, float *__restrict__ dx,
                              int rows, int d, int zero_centred) {
  __shared__ float smem[JT_WARP];
  const int row = (int)blockIdx.x;
  if (row >= rows) return;
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  const float *dr = dy + base;
  float *dxr = dx + base;
  float s = 0.0f; // sum of gy * x over the row, gy = dy * wp
  for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x)
    s = fmaf(dr[i] * jt_w_scale(w[i], zero_centred), xr[i], s);
  const float mean = jt_block_sum(s, smem) / (float)d;
  const float rb = r[row];
  for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) {
    const float gy = dr[i] * jt_w_scale(w[i], zero_centred);
    dxr[i] = rb * (gy - xr[i] * rb * rb * mean);
  }
}

// ---- Gated RMSNorm, one warp per (token, head) (design §4.4) --------------------------------------------------------
// Warps of a block take consecutive rows; a warp whose row is past the end returns before touching anything (there is no
// __syncthreads in this kernel, so an early warp return is safe).
__global__ void k_gated_rmsnorm_fwd(const float *__restrict__ x, const float *__restrict__ w,
                                    const float *__restrict__ z, float *__restrict__ y, float *__restrict__ r,
                                    int rows, int d, int zero_centred) {
  const int warps = (int)(blockDim.x / JT_WARP);
  const int row = (int)blockIdx.x * warps + (int)(threadIdx.x / JT_WARP);
  if (row >= rows) return;
  const int lane = (int)(threadIdx.x & (JT_WARP - 1));
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  const float *zr = z + base;
  float *yr = y + base;
  float ss = 0.0f;
  for (int i = lane; i < d; i += JT_WARP) ss = fmaf(xr[i], xr[i], ss);
  const float rb = rsqrtf(jt_warp_sum(ss) / (float)d + JT_NORM_EPS_F);
  for (int i = lane; i < d; i += JT_WARP)
    yr[i] = jt_w_scale(w[i], zero_centred) * (xr[i] * rb) * jt_silu(zr[i]);
  if (lane == 0) r[row] = rb;
}

__global__ void k_gated_rmsnorm_bwd(const float *__restrict__ x, const float *__restrict__ w,
                                    const float *__restrict__ z, const float *__restrict__ dy,
                                    const float *__restrict__ r, float *__restrict__ dx, float *__restrict__ dz,
                                    int rows, int d, int zero_centred) {
  const int warps = (int)(blockDim.x / JT_WARP);
  const int row = (int)blockIdx.x * warps + (int)(threadIdx.x / JT_WARP);
  if (row >= rows) return;
  const int lane = (int)(threadIdx.x & (JT_WARP - 1));
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  const float *zr = z + base;
  const float *dr = dy + base;
  float *dxr = dx + base;
  float *dzr = dz + base;
  const float rb = r[row];
  float s = 0.0f; // sum of gn * n over the row, gn = dy * wp * sg, n = x * r
  for (int i = lane; i < d; i += JT_WARP) {
    const float n = xr[i] * rb;
    s += dr[i] * jt_w_scale(w[i], zero_centred) * jt_silu(zr[i]) * n;
  }
  const float mean = jt_warp_sum(s) / (float)d;
  for (int i = lane; i < d; i += JT_WARP) {
    const float wp = jt_w_scale(w[i], zero_centred);
    const float n = xr[i] * rb;
    const float gn = dr[i] * wp * jt_silu(zr[i]);
    dxr[i] = rb * (gn - n * mean);
    dzr[i] = dr[i] * wp * n * jt_dsilu(zr[i]);
  }
}

// ---- L2 norm, one warp per (token, head) (design §3.3, §4.8) --------------------------------------------------------
__global__ void k_l2norm_fwd(const float *__restrict__ x, float post_scale, float *__restrict__ y,
                             float *__restrict__ r, int rows, int d) {
  const int warps = (int)(blockDim.x / JT_WARP);
  const int row = (int)blockIdx.x * warps + (int)(threadIdx.x / JT_WARP);
  if (row >= rows) return;
  const int lane = (int)(threadIdx.x & (JT_WARP - 1));
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  float *yr = y + base;
  float ss = 0.0f;
  for (int i = lane; i < d; i += JT_WARP) ss = fmaf(xr[i], xr[i], ss);
  const float rb = rsqrtf(jt_warp_sum(ss) + JT_NORM_EPS_F); // sum of squares, no mean (l.294)
  for (int i = lane; i < d; i += JT_WARP) yr[i] = xr[i] * rb * post_scale;
  if (lane == 0) r[row] = rb;
}

__global__ void k_l2norm_bwd(const float *__restrict__ x, const float *__restrict__ dy,
                             const float *__restrict__ r, float post_scale, float *__restrict__ dx, int rows, int d) {
  const int warps = (int)(blockDim.x / JT_WARP);
  const int row = (int)blockIdx.x * warps + (int)(threadIdx.x / JT_WARP);
  if (row >= rows) return;
  const int lane = (int)(threadIdx.x & (JT_WARP - 1));
  const size_t base = (size_t)row * (size_t)d;
  const float *xr = x + base;
  const float *dr = dy + base;
  float *dxr = dx + base;
  const float rb = r[row];
  float dot = 0.0f; // qn . dqn, qn = x * r, dqn = dy * post_scale
  for (int i = lane; i < d; i += JT_WARP) dot += (xr[i] * rb) * (dr[i] * post_scale);
  dot = jt_warp_sum(dot);
  for (int i = lane; i < d; i += JT_WARP) {
    const float qn = xr[i] * rb;
    const float dqn = dr[i] * post_scale;
    dxr[i] = rb * (dqn - qn * dot);
  }
}

// ---- launch wrappers ------------------------------------------------------------------------------------------------
// d <= 256 needs one warp per row (4 elements per thread at d = 128, 8 at d = 256); d = 1024 gets a 256-thread block
// and 4 elements per thread (plan M2.2). Both forms reduce through the same code, so the numbers are the same tree.
static inline int jt_rms_block(int d) { return d > 256 ? 256 : JT_WARP; }

extern "C" void rmsnorm_fwd(const float *x, const float *w, float *y, float *r, int rows, int d, int zero_centred,
                            cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  k_rmsnorm_fwd<<<rows, jt_rms_block(d), 0, s>>>(x, w, y, r, rows, d, zero_centred);
  JT_SYNC_DEBUG;
}

extern "C" void rmsnorm_bwd(const float *x, const float *w, const float *dy, const float *r, float *dx, int rows, int d,
                            int zero_centred, cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  k_rmsnorm_bwd<<<rows, jt_rms_block(d), 0, s>>>(x, w, dy, r, dx, rows, d, zero_centred);
  JT_SYNC_DEBUG;
}

// One warp per row, eight warps per block: the warps of the last block past the end just return, so the grid rounds up.
extern "C" void gated_rmsnorm_fwd(const float *x, const float *w, const float *z, float *y, float *r, int rows, int d,
                                  int zero_centred, cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  const int grid = (rows + JT_WARPS_PER_BLOCK - 1) / JT_WARPS_PER_BLOCK;
  k_gated_rmsnorm_fwd<<<grid, JT_WARPS_PER_BLOCK * JT_WARP, 0, s>>>(x, w, z, y, r, rows, d, zero_centred);
  JT_SYNC_DEBUG;
}

extern "C" void gated_rmsnorm_bwd(const float *x, const float *w, const float *z, const float *dy, const float *r,
                                  float *dx, float *dz, int rows, int d, int zero_centred, cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  const int grid = (rows + JT_WARPS_PER_BLOCK - 1) / JT_WARPS_PER_BLOCK;
  k_gated_rmsnorm_bwd<<<grid, JT_WARPS_PER_BLOCK * JT_WARP, 0, s>>>(x, w, z, dy, r, dx, dz, rows, d, zero_centred);
  JT_SYNC_DEBUG;
}

extern "C" void l2norm_fwd(const float *x, float post_scale, float *y, float *r, int rows, int d, cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  const int grid = (rows + JT_WARPS_PER_BLOCK - 1) / JT_WARPS_PER_BLOCK;
  k_l2norm_fwd<<<grid, JT_WARPS_PER_BLOCK * JT_WARP, 0, s>>>(x, post_scale, y, r, rows, d);
  JT_SYNC_DEBUG;
}

extern "C" void l2norm_bwd(const float *x, const float *dy, const float *r, float post_scale, float *dx, int rows, int d,
                           cudaStream_t s) {
  if (rows <= 0 || d <= 0) return;
  const int grid = (rows + JT_WARPS_PER_BLOCK - 1) / JT_WARPS_PER_BLOCK;
  k_l2norm_bwd<<<grid, JT_WARPS_PER_BLOCK * JT_WARP, 0, s>>>(x, dy, r, post_scale, dx, rows, d);
  JT_SYNC_DEBUG;
}
