// ref/misc.c — the CPU references for kernels/misc.cu (plan M2.1), written first (plan rule 2). Every loop here is the
// literal statement of a layout rule from the design, in the order the GPU test's random inputs will exercise it:
//
//   zero / add_inplace / scale          residual bookkeeping (§8.1's last rows of the table)
//   bf16_to_f32                         the top 16 bits widen to a float, nothing else (§8.1)
//   embed_gather                        out[t,:] = float(table_bf16[ids[t],:]) (§8.1)
//   qgate_split / qgate_merge           the per-head interleave of §3.5, the layout mistake that "still trains"
//   kv_expand / kv_reduce               GQA group 4: forward copies a group into its 4 heads, backward sums them
//   to_head_major / from_head_major     [L, H*D] <-> [H][L][D] for one row slice (§8.2)
//
// Everything is `real` (ref/real.h): float for the comparison against the kernel, double under -DJT_REAL_DOUBLE.
#include <stdint.h>
#include <string.h>

#include "misc.h"

// bf16 is the top half of an fp32: widen by shifting into place. Written through a uint32 so that no arithmetic
// (rounding, denormal flush) can touch the bits on the way.
static real bf16_widen(unsigned short b) {
  uint32_t u = (uint32_t)b << 16;
  float f;
  memcpy(&f, &u, sizeof f);
  return REAL(f);
}

void ref_zero(real *x, size_t n) {
  for (size_t i = 0; i < n; i++) x[i] = REAL(0);
}

void ref_add_inplace(real *y, const real *x, size_t n) {
  for (size_t i = 0; i < n; i++) y[i] += x[i];
}

void ref_scale(real *x, real a, size_t n) {
  for (size_t i = 0; i < n; i++) x[i] *= a;
}

void ref_bf16_to_f32(const unsigned short *src, real *dst, size_t n) {
  for (size_t i = 0; i < n; i++) dst[i] = bf16_widen(src[i]);
}

void ref_embed_gather(const int *ids, const unsigned short *table_bf16, real *out, int n_tok, int d) {
  for (int t = 0; t < n_tok; t++) {
    const unsigned short *row = table_bf16 + (size_t)ids[t] * (size_t)d;
    real *o = out + (size_t)t * (size_t)d;
    for (int j = 0; j < d; j++) o[j] = bf16_widen(row[j]);
  }
}

void ref_qgate_split(const real *qg, real *q, real *gate, int rows, int heads, int head_dim) {
  const size_t cols = (size_t)heads * (size_t)head_dim;             // 2048
  const size_t qg_cols = 2 * cols;                                  // 4096
  for (int t = 0; t < rows; t++) {
    const real *src = qg + (size_t)t * qg_cols;
    real *qd = q + (size_t)t * cols;
    real *gd = gate + (size_t)t * cols;
    for (int h = 0; h < heads; h++) {
      const real *s = src + (size_t)h * 2 * (size_t)head_dim;       // head h's 512 columns
      for (int j = 0; j < head_dim; j++) {
        qd[(size_t)h * head_dim + j] = s[j];                        // [h*512 + j]
        gd[(size_t)h * head_dim + j] = s[head_dim + j];              // [h*512 + 256 + j]
      }
    }
  }
}

void ref_qgate_merge(const real *q, const real *gate, real *qg, int rows, int heads, int head_dim) {
  const size_t cols = (size_t)heads * (size_t)head_dim;
  const size_t qg_cols = 2 * cols;
  for (int t = 0; t < rows; t++) {
    real *dst = qg + (size_t)t * qg_cols;
    const real *qd = q + (size_t)t * cols;
    const real *gd = gate + (size_t)t * cols;
    for (int h = 0; h < heads; h++) {
      real *d = dst + (size_t)h * 2 * (size_t)head_dim;
      for (int j = 0; j < head_dim; j++) {
        d[j] = qd[(size_t)h * head_dim + j];
        d[head_dim + j] = gd[(size_t)h * head_dim + j];
      }
    }
  }
}

void ref_kv_expand(const real *kv, real *kve, int rows, int kv_heads, int group, int head_dim) {
  const size_t kv_cols = (size_t)kv_heads * (size_t)head_dim;        // 512
  const size_t out_cols = kv_cols * (size_t)group;                   // 2048
  for (int t = 0; t < rows; t++) {
    const real *src = kv + (size_t)t * kv_cols;
    real *dst = kve + (size_t)t * out_cols;
    for (int g = 0; g < kv_heads; g++)
      for (int r = 0; r < group; r++) {
        const real *s = src + (size_t)g * head_dim;
        real *d = dst + (size_t)(g * group + r) * head_dim;          // query head h = g*group + r
        for (int j = 0; j < head_dim; j++) d[j] = s[j];
      }
  }
}

void ref_kv_reduce(const real *dkve, real *dkv, int rows, int kv_heads, int group, int head_dim) {
  const size_t kv_cols = (size_t)kv_heads * (size_t)head_dim;
  const size_t in_cols = kv_cols * (size_t)group;
  for (int t = 0; t < rows; t++) {
    const real *src = dkve + (size_t)t * in_cols;
    real *dst = dkv + (size_t)t * kv_cols;
    for (int g = 0; g < kv_heads; g++)
      for (int j = 0; j < head_dim; j++) {
        real acc = REAL(0);
        for (int r = 0; r < group; r++)                              // heads 4g .. 4g+3, in that order
          acc += src[(size_t)(g * group + r) * head_dim + j];
        dst[(size_t)g * head_dim + j] = acc;
      }
  }
}

void ref_to_head_major(const real *x, real *y, int rows, int heads, int head_dim) {
  for (int h = 0; h < heads; h++)
    for (int t = 0; t < rows; t++) {
      const real *s = x + ((size_t)t * heads + h) * (size_t)head_dim;
      real *d = y + ((size_t)h * rows + t) * (size_t)head_dim;
      for (int j = 0; j < head_dim; j++) d[j] = s[j];
    }
}

void ref_from_head_major(const real *y, real *x, int rows, int heads, int head_dim) {
  for (int h = 0; h < heads; h++)
    for (int t = 0; t < rows; t++) {
      const real *s = y + ((size_t)h * rows + t) * (size_t)head_dim;
      real *d = x + ((size_t)t * heads + h) * (size_t)head_dim;
      for (int j = 0; j < head_dim; j++) d[j] = s[j];
    }
}
