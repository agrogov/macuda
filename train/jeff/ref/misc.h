// ref/misc.h — the CPU references for kernels/misc.cu (plan M2.1). One `ref_<name>` per `jt_<name>`, same arguments in
// the same order, the same maths, in `real` (ref/real.h): built as float for the comparison against the GPU, and as
// double under -DJT_REAL_DOUBLE for the gradchecks. None of these ops is differentiable on its own (they are copies,
// sums and gathers), so misc has no grad_*.c — every one of them is checked against the kernel bit for bit instead.
//
// No CUDA here: a reference is compiled by the host compiler alone (make test-cpu), and it is written *before* the
// kernel it describes (plan rule 2), so that the test compares two independent descriptions of the same layout and not
// one implementation with itself.
#ifndef JT_REF_MISC_H
#define JT_REF_MISC_H

#include <stddef.h>

#include "real.h"

// ---- 1-D elementwise ------------------------------------------------------------------------------------------------
// x[i] = 0 for every i < n.
void ref_zero(real *x, size_t n);
// y[i] += x[i]; x and y may be the same array.
void ref_add_inplace(real *y, const real *x, size_t n);
// x[i] *= a.
void ref_scale(real *x, real a, size_t n);
// dst[i] = float(bf16 src[i]): the top 16 bits, nothing else (design §8.1).
void ref_bf16_to_f32(const unsigned short *src, real *dst, size_t n);

// ---- embedding lookup -----------------------------------------------------------------------------------------------
// out[t, j] = float(table_bf16[ids[t], j]) for t < n_tok, j < d (d = 1024: design §3.2's embed_tokens).
void ref_embed_gather(const int *ids, const unsigned short *table_bf16, real *out, int n_tok, int d);

// ---- the attention q/gate interleave (design §3.5) ------------------------------------------------------------------
// q_proj's output is [rows, heads*2*head_dim], and per head h its columns [h*2*d, h*2*d + d) are the query block and
// [h*2*d + d, h*2*d + 2*d) the gate block. Split into two [rows, heads*head_dim] arrays, and merge back. It is NOT
// "first half = q, second half = gate".
void ref_qgate_split(const real *qg, real *q, real *gate, int rows, int heads, int head_dim);
void ref_qgate_merge(const real *q, const real *gate, real *qg, int rows, int heads, int head_dim);

// ---- the GQA expand / reduce (design §3.5) --------------------------------------------------------------------------
// kv is [rows, kv_heads*head_dim]; the expanded form is [rows, kv_heads*group*head_dim] with query head h reading group
// g = h / group (h = g*group + r, r < group), so the forward is a copy of group g's columns into each of its heads and
// the backward is the sum of those heads' gradients, accumulated r = 0, 1, ..., group-1 in that order.
void ref_kv_expand(const real *kv, real *kve, int rows, int kv_heads, int group, int head_dim);
void ref_kv_reduce(const real *dkve, real *dkv, int rows, int kv_heads, int group, int head_dim);

// ---- token-major <-> head-major, for one row slice (design §8.2) ----------------------------------------------------
// x is [rows, heads*head_dim] (token-major, what a GEMM writes); y is [heads][rows][head_dim] (head-major, what the
// per-head attention GEMMs and the GDN recurrence read). rows is that row's length L, not the packed token count.
void ref_to_head_major(const real *x, real *y, int rows, int heads, int head_dim);
void ref_from_head_major(const real *y, real *x, int rows, int heads, int head_dim);

#endif // JT_REF_MISC_H
