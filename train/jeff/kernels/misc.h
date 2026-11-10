// kernels/misc.h — the plumbing kernels (design §8.1). Host C includes this; every wrapper here is extern "C" so the C
// side never sees CUDA syntax (design §11.1). Each `jt_<name>` has a `ref_<name>` in ref/misc.h written first, a CUDA
// kernel of the same name in kernels/misc.cu, and a check in test/k_misc.c (plan M2.1).
//
// All of these are launches on the stream passed last (0 = the default stream) and all of them are no-ops when a
// dimension is 0. Sizes are in elements, never bytes.
#ifndef JT_MISC_H
#define JT_MISC_H

#include <stddef.h>
#include <cuda_runtime.h>

// The model's row layouts (design §3.1), so that model.c, layer_*.c and the tests agree and none of them has to
// recompute an offset. The wrappers take heads/head_dim as arguments; these are just the model's values of them.
#define JT_HIDDEN 1024                                          // hidden size D
#define JT_VOCAB 248320                                         // embed_tokens rows
#define JT_ATTN_HEADS 8                                         // full-attention query heads
#define JT_ATTN_HEAD_DIM 256                                    // attention head dim
#define JT_ATTN_KV_HEADS 2                                      // GQA: 2 KV heads ...
#define JT_ATTN_KV_GROUP 4                                      // ... each shared by 4 query heads (h/4)
#define JT_ATTN_Q_W (JT_ATTN_HEADS * JT_ATTN_HEAD_DIM)          // 2048  q (and gate, and o) columns
#define JT_ATTN_QG_W (2 * JT_ATTN_Q_W)                          // 4096  q_proj out: per head 256 query + 256 gate
#define JT_ATTN_KV_W (JT_ATTN_KV_HEADS * JT_ATTN_HEAD_DIM)      // 512   k_proj / v_proj columns
#define JT_ATTN_KV_EXPANDED_W (JT_ATTN_KV_W * JT_ATTN_KV_GROUP) // 2048  after kv_expand

// The head-major transposes and kv_expand walk the token axis in tiles of this many rows, one block per (head, tile),
// so their launch covers the whole GPU rather than one block per head. Exported so test/k_misc.c can exercise both
// sides of a tile boundary instead of a magic number.
#define JT_MISC_TILE 16

#ifdef __cplusplus
extern "C" {
#endif

// Launch on stream s (0 = default): x[i] = v for every i < n. A no-op when n == 0. (The M0 hello kernel.)
void jt_fill(float *x, float v, size_t n, cudaStream_t s);

// ---- 1-D elementwise ------------------------------------------------------------------------------------------------
void jt_zero(float *x, size_t n, cudaStream_t s);                       // x[i] = 0
void jt_add_inplace(float *y, const float *x, size_t n, cudaStream_t s); // y[i] += x[i]; y and x may be the same
void jt_scale(float *x, float a, size_t n, cudaStream_t s);             // x[i] *= a

// bf16 -> fp32 for a whole buffer (weight conversion at load, design §8.1). The `_n` suffix is because test/harness.h
// already owns the scalar helper jt_bf16_to_f32(unsigned short); this is the same widening, n elements at a time.
void jt_bf16_to_f32_n(const unsigned short *src, float *dst, size_t n, cudaStream_t s);

// ---- embedding lookup -----------------------------------------------------------------------------------------------
// out[t, j] = float(table_bf16[ids[t], j]), t < n_tok, j < d. The table is [vocab, d] bf16 and is never converted
// wholesale (248320 x 1024 would be 1 GB fp32); only the rows the batch uses are widened.
void jt_embed_gather(const int *ids, const unsigned short *table_bf16, float *out, int n_tok, int d, cudaStream_t s);

// ---- the attention q/gate interleave (design §3.5) ------------------------------------------------------------------
// qg is [rows, heads*2*head_dim]; per head h its columns [h*2*d, h*2*d+d) are the query block and [h*2*d+d, h*2*d+2d)
// the gate block. q and gate are [rows, heads*head_dim]. Getting this wrong still trains — only parity catches it.
void jt_qgate_split(const float *qg, float *q, float *gate, int rows, int heads, int head_dim, cudaStream_t s);
void jt_qgate_merge(const float *q, const float *gate, float *qg, int rows, int heads, int head_dim, cudaStream_t s);

// ---- the GQA expand / reduce (design §3.5) --------------------------------------------------------------------------
// kv is [rows, kv_heads*head_dim], kve is [rows, kv_heads*group*head_dim]; query head h = g*group + r reads kv head
// g. kv_reduce is the adjoint of kv_expand: it sums heads g*group .. g*group+group-1 in that order.
void jt_kv_expand(const float *kv, float *kve, int rows, int kv_heads, int group, int head_dim, cudaStream_t s);
void jt_kv_reduce(const float *dkve, float *dkv, int rows, int kv_heads, int group, int head_dim, cudaStream_t s);

// ---- token-major <-> head-major, for one row slice (design §8.2) ----------------------------------------------------
// x is [rows, heads*head_dim], y is [heads][rows][head_dim]. rows is one row's L, not the packed token count.
void jt_to_head_major(const float *x, float *y, int rows, int heads, int head_dim, cudaStream_t s);
void jt_from_head_major(const float *y, float *x, int rows, int heads, int head_dim, cudaStream_t s);

#ifdef __cplusplus
}
#endif

#endif // JT_MISC_H
