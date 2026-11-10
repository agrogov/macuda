// kernels/misc.cu — the plumbing kernels (design §8.1) and their extern "C" launch wrappers. The host C calls jt_*()
// and never sees CUDA syntax (design §11.1). Every kernel here is a copy, a sum or a widen: no reduction over threads,
// no shuffles, no shared memory, so the arithmetic is exact and the GPU test's only real job is the layouts.
//
// Launch shapes follow design §8.1: grid-stride for the 1-D elementwise ops, one block per token for embed_gather, and
// one block per row (strided over the rows, so a large packed batch cannot overrun the grid's x limit) for the ops that
// walk a row's head structure. Every wrapper is a no-op for a zero dimension, so no caller has to special-case an empty
// row slice.
#include <cuda_runtime.h>
#include <stddef.h>

#include "misc.h"

// Grid size for a grid-stride launch over n elements: never more than 65535 blocks, because the kernel strides anyway.
static int jt_grid_stride(size_t n, int block) {
  size_t want = (n + (size_t)block - 1) / (size_t)block;
  return want > 65535u ? 65535 : (int)want;
}

// Grid size for a "one block per row" launch: those kernels stride over the rows as well, so this is only the cap.
static int jt_grid_rows(int rows) { return rows > 65535 ? 65535 : (rows > 0 ? rows : 0); }

// Grid size for a "one block per (head, token tile)" launch: the tiles are the x axis, the heads the y axis.
static int jt_grid_tiles(int rows) {
  const int tiles = (rows + JT_MISC_TILE - 1) / JT_MISC_TILE;
  return tiles > 65535 ? 65535 : (tiles > 0 ? tiles : 0);
}

// ---- the hello kernel (M0.2) ----------------------------------------------------------------------------------------
// Grid-stride, so the launch shape never has to fit n (1M floats here, 248k-token tables later).
__global__ void jt_fill_kernel(float *x, float v, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) x[i] = v;
}

extern "C" void jt_fill(float *x, float v, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  jt_fill_kernel<<<jt_grid_stride(n, block), block, 0, s>>>(x, v, n);
}

// ---- zero / add_inplace / scale / bf16_to_f32 (grid-stride, design §8.1) ---------------------------------------------

__global__ void jt_zero_kernel(float *x, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) x[i] = 0.0f;
}

__global__ void jt_add_inplace_kernel(float *y, const float *x, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) y[i] += x[i];
}

__global__ void jt_scale_kernel(float *x, float a, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) x[i] *= a;
}

// bf16 -> fp32 is the top 16 bits, nothing else: no rounding, no denormal handling (design §8.1).
__global__ void jt_bf16_to_f32_n_kernel(const unsigned short *src, float *dst, size_t n) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    dst[i] = __uint_as_float((unsigned int)src[i] << 16);
}

extern "C" void jt_zero(float *x, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  jt_zero_kernel<<<jt_grid_stride(n, block), block, 0, s>>>(x, n);
}

extern "C" void jt_add_inplace(float *y, const float *x, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  jt_add_inplace_kernel<<<jt_grid_stride(n, block), block, 0, s>>>(y, x, n);
}

extern "C" void jt_scale(float *x, float a, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  jt_scale_kernel<<<jt_grid_stride(n, block), block, 0, s>>>(x, a, n);
}

extern "C" void jt_bf16_to_f32_n(const unsigned short *src, float *dst, size_t n, cudaStream_t s) {
  if (n == 0) return;
  const int block = 256;
  jt_bf16_to_f32_n_kernel<<<jt_grid_stride(n, block), block, 0, s>>>(src, dst, n);
}

// ---- embed_gather: one block per token (design §8.1) ----------------------------------------------------------------
// Rows stay contiguous on both sides (that is what the GEMM consuming them wants) and no thread divides an index.
__global__ void jt_embed_gather_kernel(const int *ids, const unsigned short *table, float *out, int n_tok, int d) {
  for (int t = blockIdx.x; t < n_tok; t += gridDim.x) {
    const unsigned short *row = table + (size_t)ids[t] * (size_t)d;
    float *o = out + (size_t)t * (size_t)d;
    for (int j = threadIdx.x; j < d; j += blockDim.x) o[j] = __uint_as_float((unsigned int)row[j] << 16);
  }
}

extern "C" void jt_embed_gather(const int *ids, const unsigned short *table_bf16, float *out, int n_tok, int d,
                                cudaStream_t s) {
  if (n_tok <= 0 || d <= 0) return;
  const int block = 256;
  jt_embed_gather_kernel<<<jt_grid_rows(n_tok), block, 0, s>>>(ids, table_bf16, out, n_tok, d);
}

// ---- qgate_split / qgate_merge (design §3.5) ------------------------------------------------------------------------
// Per head h: qg columns [h*2*d, h*2*d+d) are the query block, [h*2*d+d, h*2*d+2d) the gate block. One block per row,
// the block walking the heads and the columns, so the index arithmetic is multiplication only.
__global__ void jt_qgate_split_kernel(const float *qg, float *q, float *gate, int rows, int heads, int head_dim) {
  const int cols = heads * head_dim;
  for (int t = blockIdx.x; t < rows; t += gridDim.x) {
    const float *src = qg + (size_t)t * (size_t)(2 * cols);
    float *qd = q + (size_t)t * (size_t)cols;
    float *gd = gate + (size_t)t * (size_t)cols;
    for (int h = 0; h < heads; h++) {
      const float *s = src + (size_t)h * 2 * (size_t)head_dim;
      float *qo = qd + (size_t)h * head_dim;
      float *go = gd + (size_t)h * head_dim;
      for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        qo[j] = s[j];
        go[j] = s[head_dim + j];
      }
    }
  }
}

__global__ void jt_qgate_merge_kernel(const float *q, const float *gate, float *qg, int rows, int heads, int head_dim) {
  const int cols = heads * head_dim;
  for (int t = blockIdx.x; t < rows; t += gridDim.x) {
    float *dst = qg + (size_t)t * (size_t)(2 * cols);
    const float *qd = q + (size_t)t * (size_t)cols;
    const float *gd = gate + (size_t)t * (size_t)cols;
    for (int h = 0; h < heads; h++) {
      float *d = dst + (size_t)h * 2 * (size_t)head_dim;
      const float *qo = qd + (size_t)h * head_dim;
      const float *go = gd + (size_t)h * head_dim;
      for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        d[j] = qo[j];
        d[head_dim + j] = go[j];
      }
    }
  }
}

extern "C" void jt_qgate_split(const float *qg, float *q, float *gate, int rows, int heads, int head_dim,
                               cudaStream_t s) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0) return;
  const int block = 256;
  jt_qgate_split_kernel<<<jt_grid_rows(rows), block, 0, s>>>(qg, q, gate, rows, heads, head_dim);
}

extern "C" void jt_qgate_merge(const float *q, const float *gate, float *qg, int rows, int heads, int head_dim,
                               cudaStream_t s) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0) return;
  const int block = 256;
  jt_qgate_merge_kernel<<<jt_grid_rows(rows), block, 0, s>>>(q, gate, qg, rows, heads, head_dim);
}

// ---- kv_expand / kv_reduce (design §3.5) ----------------------------------------------------------------------------
// Query head h = g*group + r reads KV head g = h/group. kv_expand copies group g's row into each of its heads;
// kv_reduce sums heads g*group .. g*group+group-1 in that order (the adjoint, and the deterministic order the plan's
// adjoint check pins down).
__global__ void jt_kv_expand_kernel(const float *kv, float *kve, int rows, int kv_heads, int group, int head_dim) {
  const int out_heads = kv_heads * group;
  const int tiles = (rows + JT_MISC_TILE - 1) / JT_MISC_TILE;
  for (int qh = blockIdx.y; qh < out_heads; qh += gridDim.y) {
    const int g = qh / group; // query head h = g*group + r, so its KV group is h/group
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int t0 = tile * JT_MISC_TILE;
      const int t1 = t0 + JT_MISC_TILE < rows ? t0 + JT_MISC_TILE : rows;
      for (int t = t0; t < t1; t++) {
        const float *s = kv + ((size_t)t * kv_heads + g) * (size_t)head_dim;
        float *d = kve + ((size_t)t * out_heads + qh) * (size_t)head_dim;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) d[j] = s[j];
      }
    }
  }
}

__global__ void jt_kv_reduce_kernel(const float *dkve, float *dkv, int rows, int kv_heads, int group, int head_dim) {
  const int kv_cols = kv_heads * head_dim;
  const int in_cols = kv_cols * group;
  for (int t = blockIdx.x; t < rows; t += gridDim.x) {
    const float *src = dkve + (size_t)t * (size_t)in_cols;
    float *dst = dkv + (size_t)t * (size_t)kv_cols;
    for (int g = 0; g < kv_heads; g++) {
      const float *base = src + (size_t)(g * group) * head_dim;
      float *d = dst + (size_t)g * head_dim;
      for (int j = threadIdx.x; j < head_dim; j += blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < group; r++) acc += base[(size_t)r * head_dim + j];
        d[j] = acc;
      }
    }
  }
}

extern "C" void jt_kv_expand(const float *kv, float *kve, int rows, int kv_heads, int group, int head_dim,
                             cudaStream_t s) {
  if (rows <= 0 || kv_heads <= 0 || group <= 0 || head_dim <= 0) return;
  const int block = 256;
  dim3 grid((unsigned)jt_grid_tiles(rows), (unsigned)jt_grid_rows(kv_heads * group));
  jt_kv_expand_kernel<<<grid, block, 0, s>>>(kv, kve, rows, kv_heads, group, head_dim);
}

extern "C" void jt_kv_reduce(const float *dkve, float *dkv, int rows, int kv_heads, int group, int head_dim,
                             cudaStream_t s) {
  if (rows <= 0 || kv_heads <= 0 || group <= 0 || head_dim <= 0) return;
  const int block = 256;
  jt_kv_reduce_kernel<<<jt_grid_rows(rows), block, 0, s>>>(dkve, dkv, rows, kv_heads, group, head_dim);
}

// ---- to_head_major / from_head_major (design §8.2) ------------------------------------------------------------------
// [rows, heads*head_dim] <-> [heads][rows][head_dim] for one row slice. One block per (head, token tile), so the launch
// covers the whole GPU even for the model's 8 heads; the old one-block-per-head launch gave 8 blocks and each walked
// the entire token axis alone. A block still walks whole rows with contiguous j, so every output element comes from
// exactly the same input element as before (pure copy, no reduction).
__global__ void jt_to_head_major_kernel(const float *x, float *y, int rows, int heads, int head_dim) {
  const int tiles = (rows + JT_MISC_TILE - 1) / JT_MISC_TILE;
  for (int h = blockIdx.y; h < heads; h += gridDim.y) {
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int t0 = tile * JT_MISC_TILE;
      const int t1 = t0 + JT_MISC_TILE < rows ? t0 + JT_MISC_TILE : rows;
      for (int t = t0; t < t1; t++) {
        const float *s = x + ((size_t)t * heads + h) * (size_t)head_dim;
        float *d = y + ((size_t)h * rows + t) * (size_t)head_dim;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) d[j] = s[j];
      }
    }
  }
}

__global__ void jt_from_head_major_kernel(const float *y, float *x, int rows, int heads, int head_dim) {
  const int tiles = (rows + JT_MISC_TILE - 1) / JT_MISC_TILE;
  for (int h = blockIdx.y; h < heads; h += gridDim.y) {
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
      const int t0 = tile * JT_MISC_TILE;
      const int t1 = t0 + JT_MISC_TILE < rows ? t0 + JT_MISC_TILE : rows;
      for (int t = t0; t < t1; t++) {
        const float *s = y + ((size_t)h * rows + t) * (size_t)head_dim;
        float *d = x + ((size_t)t * heads + h) * (size_t)head_dim;
        for (int j = threadIdx.x; j < head_dim; j += blockDim.x) d[j] = s[j];
      }
    }
  }
}

extern "C" void jt_to_head_major(const float *x, float *y, int rows, int heads, int head_dim, cudaStream_t s) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0) return;
  const int block = 256;
  dim3 grid((unsigned)jt_grid_tiles(rows), (unsigned)jt_grid_rows(heads));
  jt_to_head_major_kernel<<<grid, block, 0, s>>>(x, y, rows, heads, head_dim);
}

extern "C" void jt_from_head_major(const float *y, float *x, int rows, int heads, int head_dim, cudaStream_t s) {
  if (rows <= 0 || heads <= 0 || head_dim <= 0) return;
  const int block = 256;
  dim3 grid((unsigned)jt_grid_tiles(rows), (unsigned)jt_grid_rows(heads));
  jt_from_head_major_kernel<<<grid, block, 0, s>>>(y, x, rows, heads, head_dim);
}
