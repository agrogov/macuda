// src/layer_attn.c — the full-attention mixer: design §3.5 forward, §4.7 backward, on the kernels of plan M2.
// The forward is J22's (plan M4.3 item 4); the backward is J29's (plan M5.1 item 3) and reads the saves the forward
// below leaves, so the two halves of this file are one save layout and one set of kernels.
//
// The forward is design §3.5 in order, with the call each step uses:
//
//   a    = RMSNorm(x_in; input_layernorm)                     rmsnorm_fwd, r saved
//   qg   = q_proj(a)                                          linear_fwd [n,4096]
//   q, gate = qgate_split(qg)                                 jt_qgate_split: per head h, [h*512, h*512+256) is the
//                                                             query block and [h*512+256, h*512+512) the gate block —
//                                                             interleaved per head, NOT first 2048 / last 2048
//   k    = k_proj(a), v = v_proj(a)                           linear_fwd, [n,512] each
//   q = RMSNorm(q; q_norm), k = RMSNorm(k; k_norm)            rmsnorm_fwd with d = 256, zero-centred (rows = n*8 for q,
//                                                             n*2 for k)
//   q, k = RoPE(q, p), RoPE(k, p)                             rope_fwd with p = t - off[row_of[t]] (row-local, design §7)
//   per row: to_head_major, kv_expand (q, k, v)                 design §8.2's layout, one launch each per row — or, with
//                                                             JT_ATTN_BATCH=<cap> on, once for a chunk of rows (see
//                                                             "launch batching" below)
//   per row: S = q k^T (gemm_batched), attn_softmax_fwd(scale 1/16), PV (gemm_batched), then the chunk's
//            from_head_major                                     design §5, §8.2; one row's P at a time, bounded by c->lmax
//   o    = o * sigmoid(gate)                                  k_sigmoid_gate_fwd
//   out  = o_proj(o)                                          linear_fwd
//   x_out= x_in + out                                         the residual, design §4.6
//
// The backward is design §4.7 in the reverse of that order, with the call each step uses:
//
//   d_o_gated = dres through o_proj                             linear_bwd, dy = dres (nothing writes dres until the last
//                                                               line of the function), dx taken fresh
//   d_o, dgate = sigmoid_gate_bwd(d_o_gated, o_pre, gate)       the forward's gate is re-split out of the saved qg
//   per row (design §8.2's second block)                        dP = dO V^T, dV = P^T dO, attn_softmax_bwd(P, dP),
//                                                               dQ = dS K, dK = dS^T Q; P is recomputed from the saved q
//                                                               and k (only one row's P is ever kept), then from_head_major
//                                                               and kv_reduce collapse dk/dv to the 2 KV heads. With
//                                                               JT_ATTN_BATCH these five launches are per chunk, not per row
//   dq, dk = rope_bwd(...)                                      one launch each over the packed batch (positions are
//                                                               row-local, so the whole batch is in the transposed tables)
//   q_norm/k_norm backward                                      rmsnorm_bwd, d = 256, zero-centred (design §4.3)
//   dqg = qgate_merge(dq, dgate)                                the adjoint of qgate_split
//   da = q_proj/k_proj/v_proj backward, summed                   three linear_bwd into one buffer: dx is the same `a`
//                                                               (design §4.6: add to a gradient, never overwrite it)
//   dres += RMSNormBackward(da)                                 rmsnorm_bwd over all n tokens, then the residual
//
// The four projections accumulate dA/dB into the flat buffer (src/linear.h: a parameter gradient belongs to the step, not
// to the layer). Nothing here synchronises the stream; the caller does, as everywhere else in the tree.
//
// ---- what a JT_FWD_SAVE pass saves (design §6.2, §4.7) --------------------------------------------------------------
// The backward takes no activation pointers, so it re-derives them from the arena with attn_save_ptrs(), the same
// function the forward lays the block out with; that is the whole "save layout" and it is one function for that reason.
// The forward takes ONE block of attn_saves_floats(n) words (the arena is a bump allocator, so fifteen allocations of
// s1..s15 land exactly where one of their sum does) from the save half in JT_FWD_SAVE and from the work half in
// JT_FWD_ONLY (freed before the function returns); the arithmetic is identical either way. In the order the block holds
// them:
//
//   r_in   [n]         RMSNorm(input_layernorm)'s r          (4.3's forward save)
//   a      [n,1024]    its output, the input of the four projections (linear_bwd's x)
//   qg     [n,4096]    q_proj's output; the gate is re-split from it, so gate is not saved
//   q_pre  [n,2048]    the split query before q_norm          (rmsnorm_bwd's x)
//   r_q    [n,8]       q_norm's r
//   q      [n,2048]    query after q_norm and rope            (the attention GEMMs' Q)
//   k_raw  [n,512]     k_proj's output before k_norm          (rmsnorm_bwd's x)
//   r_k    [n,2]       k_norm's r
//   k      [n,512]     key after k_norm and rope              (the attention GEMMs' K)
//   v      [n,512]     v_proj's output, no norm and no rope    (the attention GEMMs' V)
//   o_pre  [n,2048]    the attention's output before the gate (sigmoid_gate_bwd's `o`)
//   u_qg/u_k/u_v/u_o [n,16]  the four linears' saved u = x A^T (design §4.2)
//
// That is 12,875 floats per token, well inside JT_ARENA_FLOATS_PER_TOKEN (98,304). P is never saved: the backward
// recomputes it row by row from q and k, which is why the arena carries 2 x 8 x lmax^2 floats for one row's P and dP.
//
// Where the backward finds them: the block starts at the arena's base, because layer_fwd() resets the save half before
// its mixer (src/layer.h) and a caller that invokes this piece directly resets it itself — the same convention head_bwd
// uses. Its own temporaries come from the work half and are given back before it returns (design §6.2).
#include "layer.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#include "act.h"    // k_sigmoid_gate_fwd, k_sigmoid_gate_bwd
#include "attn.h"   // attn_softmax_fwd, attn_softmax_bwd
#include "gemm.h"   // gemm_batched
#include "linear.h" // linear_fwd, linear_bwd
#include "misc.h"   // jt_qgate_split/merge, jt_kv_expand/reduce, jt_to/from_head_major, jt_add_inplace
#include "norm.h"   // rmsnorm_fwd, rmsnorm_bwd
#include "rope.h"   // rope_fwd, rope_bwd
#include "util.h"   // CK, JT_SYNC_DEBUG

// 256^-0.5 (design §3.5, §4.7): attn_softmax_fwd applies it to the raw GEMM scores, and its backward takes the same
// number. Exactly 1/16 in fp32, so it is written as one.
#define ATTN_SCALE 0.0625f

static int attn_fail(const char *msg) {
  fprintf(stderr, "layer_attn: %s\n", msg);
  return JT_LAYER_FAIL;
}

// The save layout (see the header comment). One layout, used by the fwd (which allocates the block) and by the bwd
// (which reads it back), so the two cannot disagree about which buffer holds what. The order is fixed and must not
// change.
typedef struct {
  float *r_in, *a, *qg, *q_pre, *r_q, *q, *k_raw, *r_k, *k, *v, *o_pre;
  float *u_qg, *u_k, *u_v, *u_o;
} AttnSave;

#define ATTN_NSAVE 15 // the fields of AttnSave, in the order they are handed out

// How many words the block takes: 12,875 per token (the header comment's table).
static size_t attn_saves_floats(int n) {
  const size_t t = (size_t)n;
  return t * ((size_t)1 + MT_HIDDEN + MT_Q_OUT + 3 * (size_t)MT_O_IN + MT_FULL_HEADS + MT_KV_HEADS +
              3 * (size_t)MT_KV_OUT + 4 * (size_t)MT_LORA_RANK);
}

// The pointers of a block of attn_saves_floats(n) words at `base`. Both callers go through here: the forward with the
// block it just took, the backward with the save pass's block (the arena's base, see the header comment).
static void attn_save_ptrs(int n, float *base, AttnSave *s) {
  const size_t t = (size_t)n;
  const size_t sz[ATTN_NSAVE] = {t,
                                 t * MT_HIDDEN,
                                 t * MT_Q_OUT,
                                 t * MT_O_IN,
                                 t * MT_FULL_HEADS,
                                 t * MT_O_IN,
                                 t * MT_KV_OUT,
                                 t * MT_KV_HEADS,
                                 t * MT_KV_OUT,
                                 t * MT_KV_OUT,
                                 t * MT_O_IN,
                                 t * MT_LORA_RANK,
                                 t * MT_LORA_RANK,
                                 t * MT_LORA_RANK,
                                 t * MT_LORA_RANK};
  float **dst[ATTN_NSAVE] = {&s->r_in, &s->a,   &s->qg,  &s->q_pre, &s->r_q, &s->q,   &s->k_raw, &s->r_k,
                             &s->k,    &s->v,   &s->o_pre, &s->u_qg, &s->u_k, &s->u_v, &s->u_o};
  float *p = base;
  for (int k = 0; k < ATTN_NSAVE; k++) {
    *dst[k] = p;
    p += sz[k];
  }
}

// ---- launch batching (plan M8 item 5) ------------------------------------------------------------------------------
// Everything around the per-row GEMMs is token-wise: to_head_major, kv_expand, from_head_major and kv_reduce each read and
// write one (token, head) pair, so those launches need not be per row. One launch covers a whole CHUNK of consecutive
// rows, laid out over the chunk's packed tokens: head block h of the chunk starts at h*ctok*256, so a row's slice of it
// starts at h*ctok*256 + (off - coff)*256. The per-row GEMMs then take a batch stride of ctok*256 instead of L*256 — the
// head blocks' rows are still 256 floats and the leading dimensions are unchanged, so the same values land in the same
// places, bit for bit; only the number of launches changes. Per layer that is 6 launches in the forward and 11 in the
// backward instead of 6*rows and 11*rows (rows = the batch's row count), which is the whole point on the shim, where a
// launch costs orders of magnitude more than it does on the card (plan M8 item 5; notes/J40.md has the counts).
//
// A chunk is capped at the toggle's value so the hoisted blocks cost the arena nothing new: a chunk holds at most
// max(cap, lmax) tokens either way (a single long row is a chunk of its own, which attn_chunk_max() sizes for), and the
// arena's per-token budget (JT_ARENA_FLOATS_PER_TOKEN, 98,304) covers 10,240 floats per chunk token in the forward (4
// head-major blocks + kv_expand's scratch) and 16,384 in the backward (7 blocks + scratch) with room to spare — the
// blocks are sized by min(that, n), the batch's own token count, so no cap can exceed the budget. The work half's own
// rule is unchanged: everything below comes from jt_work_alloc and is given back before returning.
//
// The switch is JT_ATTN_BATCH=<cap>, in the environment as JT_EXACT and JT_GDN_CHUNKED are: unset or 0 is the per-row
// path above, byte for byte (one row per chunk, so the same kernels are called with the same arguments in the same
// order); a positive value chunks consecutive rows up to <cap> packed tokens and prints one start-up line, which is what
// scripts/ab.sh's JT_EXPECT checks — a toggle that silently did nothing cannot pass it. The value is parsed once, in a
// function-scope static, and the one line is printed the first time either mixer runs (gemm_init()'s pattern).
static int attn_batch_cap(void) {
  static int cap = -1, told; // the toggle, and whether the start-up line has been printed
  if (cap < 0) {
    const char *e = getenv("JT_ATTN_BATCH");
    char *end = NULL;
    const long v = e != NULL ? strtol(e, &end, 10) : 0;
    cap = (e != NULL && end != e && v > 0) ? (int)(v > INT_MAX ? INT_MAX : v) : 0;
  }
  if (cap > 0 && !told) {
    printf("attn: launch batching, cap %d\n", cap);
    fflush(stdout);
    told = 1;
  }
  return cap;
}

// One chunk's attention buffers (design §8.2): [8][ctok][256] head-major for q, k, v and o, one token-major [ctok][2048]
// scratch for kv_expand's output, and the [8][lmax][lmax] scores and probabilities — those are one row's and reused by
// every row of every chunk, which is the arena's 2 x 8 x lmax^2 term. jt_work_alloc dies with the numbers if the arena is
// ever sized too small.
typedef struct {
  float *qh, *kh, *vh, *oh, *kv_tok, *sc, *p;
} AttnRow;

// The tokens the chunk blocks are sized for: the cap, unless a single row is longer than it (then that row is a chunk).
// The per-row path (cap <= 0) keeps the committed tree's c->lmax-sized blocks, so its allocation is unchanged too.
static int attn_chunk_max(const JtCtx *c, int n, int cap) {
  int m;
  if (cap <= 0) return c->lmax;
  m = cap < c->lmax ? c->lmax : cap;
  return m < n ? m : n;
}

static void attn_row_alloc(JtCtx *c, AttnRow *r, int ctok_max) {
  const size_t ct = (size_t)ctok_max;
  const size_t l = (size_t)c->lmax;
  const size_t hm = (size_t)MT_FULL_HEADS * ct * MT_HEAD_DIM;
  const size_t sq = (size_t)MT_FULL_HEADS * l * l;
  r->qh = jt_work_alloc(c, hm);
  r->kh = jt_work_alloc(c, hm);
  r->vh = jt_work_alloc(c, hm);
  r->oh = jt_work_alloc(c, hm);
  r->kv_tok = jt_work_alloc(c, ct * MT_O_IN);
  r->sc = jt_work_alloc(c, sq);
  r->p = jt_work_alloc(c, sq);
}

// The first row after the chunk that starts at r0: consecutive rows whose tokens fit the cap, but always at least one (a
// row longer than the cap is a chunk of its own, which attn_chunk_max() sized the blocks for). With the toggle off (cap <=
// 0) a chunk is exactly one row, which is the committed tree's per-row loop. Shared by both directions, so the forward and
// the backward chunk the batch the same way and a row's slice is the same in both.
static int attn_chunk_end(const JtBatch *b, int r0, int cap) {
  int r1 = r0, ctok = 0;
  while (r1 < b->rows) {
    const int L = b->off[r1 + 1] - b->off[r1];
    if (r1 > r0 && (cap <= 0 || ctok + L > cap)) break;
    ctok += L;
    r1++;
  }
  return r1;
}

// The chunk's layout pass, in the order design §8.2 lists it: q is transposed to head-major, and k and v each go through
// kv_expand (which copies KV group g to its 4 query heads h = g*group + r, group = h/4) before the same transpose
// (design §3.5); MT_KV_HEADS x group = MT_FULL_HEADS is the model's only consistent value.
static void attn_chunk_prep(JtCtx *c, const AttnSave *s, const AttnRow *r, int coff, int ctok) {
  const int group = MT_FULL_HEADS / MT_KV_HEADS;
  cudaStream_t st = c->stream;
  jt_to_head_major(s->q + (size_t)coff * MT_O_IN, r->qh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->k + (size_t)coff * MT_KV_OUT, r->kv_tok, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->kh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->v + (size_t)coff * MT_KV_OUT, r->kv_tok, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->vh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
}

// One row's slice [off, off+L) of design §8.2's first block, inside a chunk that starts at coff: the two GEMMs and the
// softmax, at the row's slice of the chunk's head-major blocks (the per-head step is ctok*256, not L*256).
static void attn_row(JtCtx *c, AttnRow *r, int coff, int off, int L, int ctok) {
  const long hm = (long)ctok * (long)MT_HEAD_DIM;
  const long sq = (long)L * (long)L;
  const size_t at = (size_t)(off - coff) * MT_HEAD_DIM; // the row's first token, in every head block
  const float *qh = r->qh + at, *kh = r->kh + at, *vh = r->vh + at;
  float *oh = r->oh + at;

  // Sc[h] = Q[h] K[h]^T  [L,L]: A is stored [M,K] = [L,256], B is stored [N,K] = [L,256] (hence tb = 1).
  gemm_batched(c->cublas, 0, 1, L, L, MT_HEAD_DIM, 1.0f, qh, hm, kh, hm, 0.0f, r->sc, sq, MT_FULL_HEADS);
  attn_softmax_fwd(r->sc, r->p, L, MT_FULL_HEADS, ATTN_SCALE, c->stream); // P[h] = causal_softmax(Sc[h] / 16)
  // O[h] = P[h] V[h]  [L,256]: A is stored [M,K] = [L,L], B is stored [K,N] = [L,256] (tb = 0).
  gemm_batched(c->cublas, 0, 0, L, MT_HEAD_DIM, L, 1.0f, r->p, sq, vh, hm, 0.0f, oh, hm, MT_FULL_HEADS);
}

// One chunk's attention backward buffers (design §8.2's second block): the forward's q/k/v head-major scratch again (the
// backward re-derives P from them), the cotangent dO head-major, the three cotangents dQ/dK/dV head-major, one token-major
// [ctok][2048] scratch for from_head_major and kv_reduce, and [8][lmax][lmax] for P, dP and dS. dP takes over the score
// buffer's words (`sc`) once P exists, so a row costs three L x L slabs, not four; jt_work_alloc still wants all of them
// live at the work half's bottom, which is why they are taken here in one place.
typedef struct {
  float *qh, *kh, *vh, *doh, *dqh, *dkh, *dvh, *kv_tok, *sc, *p, *dsc;
} AttnBwdRow;

static void attn_bwd_row_alloc(JtCtx *c, AttnBwdRow *r, int ctok_max) {
  const size_t ct = (size_t)ctok_max;
  const size_t l = (size_t)c->lmax;
  const size_t hm = (size_t)MT_FULL_HEADS * ct * MT_HEAD_DIM;
  const size_t sq = (size_t)MT_FULL_HEADS * l * l;
  r->qh = jt_work_alloc(c, hm);
  r->kh = jt_work_alloc(c, hm);
  r->vh = jt_work_alloc(c, hm);
  r->doh = jt_work_alloc(c, hm);
  r->dqh = jt_work_alloc(c, hm);
  r->dkh = jt_work_alloc(c, hm);
  r->dvh = jt_work_alloc(c, hm);
  r->kv_tok = jt_work_alloc(c, ct * MT_O_IN);
  r->sc = jt_work_alloc(c, sq);
  r->p = jt_work_alloc(c, sq);
  r->dsc = jt_work_alloc(c, sq);
}

// The chunk's layout pass: the same five token-wise ops as the forward's attn_chunk_prep(), plus the cotangent dO.
static void attn_chunk_prep_bwd(JtCtx *c, const AttnSave *s, const AttnBwdRow *r, const float *d_o, int coff, int ctok) {
  const int group = MT_FULL_HEADS / MT_KV_HEADS;
  cudaStream_t st = c->stream;
  jt_to_head_major(s->q + (size_t)coff * MT_O_IN, r->qh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->k + (size_t)coff * MT_KV_OUT, r->kv_tok, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->kh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->v + (size_t)coff * MT_KV_OUT, r->kv_tok, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->vh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_to_head_major(d_o + (size_t)coff * MT_O_IN, r->doh, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
}

// One row's slice [off, off+L) of design §8.2's backward, in the order the design lists it. The forward's P is
// recomputed from the saved q and k by the same two calls attn_row() makes (the score GEMM and attn_softmax_fwd), so the
// softmax backward differentiates the forward's own P; the four GEMMs are then the transposes of the forward's two
// (design §5), with attn_softmax_bwd folding in the 1/16 (design §4.7's `dq = c * sum dS k`). dQ, dK and dV stay
// head-major over the chunk; attn_chunk_tail() turns them token-major and kv_reduces dk/dv to the 2 KV heads.
static void attn_row_bwd(JtCtx *c, AttnBwdRow *r, int coff, int off, int L, int ctok) {
  const long hm = (long)ctok * (long)MT_HEAD_DIM;
  const long sq = (long)L * (long)L;
  const size_t at = (size_t)(off - coff) * MT_HEAD_DIM; // the row's first token, in every head block
  const float *qh = r->qh + at, *kh = r->kh + at, *vh = r->vh + at, *doh = r->doh + at;
  float *dqh = r->dqh + at, *dkh = r->dkh + at, *dvh = r->dvh + at;

  // P = causal_softmax(Q K^T / 16), the forward's two calls (design §8.2's first two lines).
  gemm_batched(c->cublas, 0, 1, L, L, MT_HEAD_DIM, 1.0f, qh, hm, kh, hm, 0.0f, r->sc, sq, MT_FULL_HEADS);
  attn_softmax_fwd(r->sc, r->p, L, MT_FULL_HEADS, ATTN_SCALE, c->stream);
  // dP = dO V^T (the same (ta=0, tb=1) shape as Sc, so it takes Sc's words over now that P exists).
  gemm_batched(c->cublas, 0, 1, L, L, MT_HEAD_DIM, 1.0f, doh, hm, vh, hm, 0.0f, r->sc, sq, MT_FULL_HEADS);
  // dV = P^T dO, the forward's PV GEMM with its operands swapped (ta = 1).
  gemm_batched(c->cublas, 1, 0, L, MT_HEAD_DIM, L, 1.0f, r->p, sq, doh, hm, 0.0f, dvh, hm, MT_FULL_HEADS);
  // dS = (1/16) P (dP - sum_s' P dP), exactly 0 above the diagonal (design §4.7, kernels/attn.h).
  attn_softmax_bwd(r->p, r->sc, r->dsc, L, MT_FULL_HEADS, ATTN_SCALE, c->stream);
  // dQ = dS K [L,256] and dK = dS^T Q [L,256].
  gemm_batched(c->cublas, 0, 0, L, MT_HEAD_DIM, L, 1.0f, r->dsc, sq, kh, hm, 0.0f, dqh, hm, MT_FULL_HEADS);
  gemm_batched(c->cublas, 1, 0, L, MT_HEAD_DIM, L, 1.0f, r->dsc, sq, qh, hm, 0.0f, dkh, hm, MT_FULL_HEADS);
}

// The chunk's way back to token-major: dQ is one from_head_major, and dV/dK go through from_head_major and kv_reduce,
// i.e. the 4 query heads of a group summed into its KV head (design §4.7's `summed over the 4 query heads of the group`).
// The two reductions share kv_tok, which is fine: both are on c->stream, in this order.
static void attn_chunk_tail(JtCtx *c, const AttnBwdRow *r, int ctok, float *dq, float *dk, float *dv) {
  const int group = MT_FULL_HEADS / MT_KV_HEADS;
  cudaStream_t st = c->stream;
  jt_from_head_major(r->dqh, dq, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_from_head_major(r->dvh, r->kv_tok, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_reduce(r->kv_tok, dv, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_from_head_major(r->dkh, r->kv_tok, ctok, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_reduce(r->kv_tok, dk, ctok, MT_KV_HEADS, group, MT_HEAD_DIM, st);
}

int layer_attn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                         int mode) {
  if (c == NULL || lw == NULL || x_in == NULL || x_out == NULL || b == NULL)
    return attn_fail("layer_attn_mixer_fwd: bad arguments");
  if (n != b->n) {
    fprintf(stderr, "layer_attn: layer_attn_mixer_fwd: n %d but the batch has %d tokens\n", n, b->n);
    return JT_LAYER_FAIL;
  }
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE) {
    fprintf(stderr, "layer_attn: layer_attn_mixer_fwd: mode %d, want JT_FWD_ONLY or JT_FWD_SAVE\n", mode);
    return JT_LAYER_FAIL;
  }
  if (n < 0) return attn_fail("layer_attn_mixer_fwd: negative n");
  if (n == 0) return JT_LAYER_OK;

  // The row geometry has to fit what the context was sized for: rope_fwd's tables cover 0..c->lmax-1 and the row buffers
  // are [8][lmax][...]. A longer row is a context sized for the wrong batch, not something to read past.
  for (int r = 0; r < b->rows; r++) {
    const int L = b->off[r + 1] - b->off[r];
    if (L < 0 || b->off[r + 1] > n) return attn_fail("layer_attn_mixer_fwd: the batch's off table is not a prefix sum");
    if (L > c->lmax) {
      fprintf(stderr, "layer_attn: layer_attn_mixer_fwd: row %d is %d tokens, the context was sized for %d (jt_ctx_alloc lmax)\n",
              r, L, c->lmax);
      return JT_LAYER_FAIL;
    }
  }

  const size_t mark = jt_work_mark(c); // every temporary below is given back at `done` (saves are not: JT_FWD_SAVE)
  const size_t saves_n = attn_saves_floats(n);
  float *saves = (mode == JT_FWD_SAVE) ? jt_save_alloc(c, saves_n) : jt_work_alloc(c, saves_n);
  AttnSave s;
  attn_save_ptrs(n, saves, &s);

  float *gate = jt_work_alloc(c, (size_t)n * MT_O_IN); // the split gate; re-derived from qg by the bwd, so not saved
  float *o_gated = jt_work_alloc(c, (size_t)n * MT_O_IN);
  float *out = jt_work_alloc(c, (size_t)n * MT_HIDDEN);
  AttnRow row;
  const int cap = attn_batch_cap(); // 0 = the per-row path; > 0 = a chunk of at most `cap` packed tokens
  attn_row_alloc(c, &row, attn_chunk_max(c, n, cap));

  // a = RMSNorm(x_in; input_layernorm); then the four projections off it.
  rmsnorm_fwd(x_in, lw->input_norm, s.a, s.r_in, n, MT_HIDDEN, 1, c->stream);
  int rc = linear_fwd(c, &lw->q_proj, s.a, n, s.qg, s.u_qg);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->k_proj, s.a, n, s.k_raw, s.u_k);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->v_proj, s.a, n, s.v, s.u_v);
  if (rc != JT_LAYER_OK) goto done; // linear_fwd printed why

  // q, gate = qgate_split(qg)  (per head interleaved, design §3.5)
  jt_qgate_split(s.qg, s.q_pre, gate, n, MT_FULL_HEADS, MT_HEAD_DIM, c->stream);

  // q = RMSNorm(q; q_norm) then RoPE; k = RMSNorm(k; k_norm) then RoPE. d = 256 per (token, head), zero-centred, and
  // the positions come from the batch (row-local, design §7) — q and k are [n][heads*256], so the norms see n*8 / n*2
  // rows of 256 and rope sees heads = 8 / 2.
  rmsnorm_fwd(s.q_pre, lw->q_norm, s.q, s.r_q, n * MT_FULL_HEADS, MT_HEAD_DIM, 1, c->stream);
  rope_fwd(s.q, b->off_dev, b->row_of, n, MT_FULL_HEADS, MT_HEAD_DIM, c->rope_cos, c->rope_sin, c->stream);
  rmsnorm_fwd(s.k_raw, lw->k_norm, s.k, s.r_k, n * MT_KV_HEADS, MT_HEAD_DIM, 1, c->stream);
  rope_fwd(s.k, b->off_dev, b->row_of, n, MT_KV_HEADS, MT_HEAD_DIM, c->rope_cos, c->rope_sin, c->stream);

  // The row chunks (attn_chunk_end): one layout pass per chunk, then the per-row GEMMs, then the chunk's way back to
  // token-major. With JT_ATTN_BATCH a batch of rows that fits the cap is one chunk, i.e. six launches per layer instead of
  // six per row; without it a chunk is one row and this is the per-row loop it replaced.
  for (int r0 = 0; r0 < b->rows;) {
    const int r1 = attn_chunk_end(b, r0, cap), coff = b->off[r0], cend = b->off[r1];
    if (cend > coff) {
      attn_chunk_prep(c, &s, &row, coff, cend - coff);
      for (int r = r0; r < r1; r++) {
        const int off = b->off[r], L = b->off[r + 1] - off;
        if (L > 0) attn_row(c, &row, coff, off, L, cend - coff);
      }
      jt_from_head_major(row.oh, s.o_pre + (size_t)coff * MT_O_IN, cend - coff, MT_FULL_HEADS, MT_HEAD_DIM, c->stream);
    }
    r0 = r1;
  }

  // o = o * sigmoid(gate) over the whole packed batch (the gate is laid out like o, [n][8*256]).
  k_sigmoid_gate_fwd(s.o_pre, gate, o_gated, n * MT_O_IN, c->stream);
  rc = linear_fwd(c, &lw->o_proj, o_gated, n, out, s.u_o);
  if (rc != JT_LAYER_OK) goto done;

  // x_out = x_in + out (design §4.6); x_out may alias x_in (src/layer.h's contract), so the copy is skipped then.
  if (x_out != x_in)
    CK(cudaMemcpyAsync(x_out, x_in, (size_t)n * MT_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
  jt_add_inplace(x_out, out, (size_t)n * MT_HIDDEN, c->stream);

done:
  jt_work_restore(c, mark);
  return rc;
}

int layer_attn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  if (c == NULL || lw == NULL || x_in == NULL || dres == NULL || b == NULL)
    return attn_fail("layer_attn_mixer_bwd: bad arguments");
  if (c->model == NULL || c->lora == NULL) return attn_fail("layer_attn_mixer_bwd: the context has no model or no LoRA buffer");
  if (n != b->n) {
    fprintf(stderr, "layer_attn: layer_attn_mixer_bwd: n %d but the batch has %d tokens\n", n, b->n);
    return JT_LAYER_FAIL;
  }
  if (n < 0) return attn_fail("layer_attn_mixer_bwd: negative n");
  if (n == 0) return JT_LAYER_OK;

  // The same row geometry the forward checked: rope's tables cover 0..c->lmax-1, the row buffers are [8][lmax][...], and
  // rope_bwd's positions come from the batch's row table.
  for (int r = 0; r < b->rows; r++) {
    const int L = b->off[r + 1] - b->off[r];
    if (L < 0 || b->off[r + 1] > n) return attn_fail("layer_attn_mixer_bwd: the batch's off table is not a prefix sum");
    if (L > c->lmax) {
      fprintf(stderr, "layer_attn: layer_attn_mixer_bwd: row %d is %d tokens, the context was sized for %d (jt_ctx_alloc lmax)\n",
              r, L, c->lmax);
      return JT_LAYER_FAIL;
    }
  }

  // The saves: the SAVE pass's block at the arena's base (the header comment). A context whose save half holds less than
  // this layer's block was never given a JT_FWD_SAVE pass of this layer.
  const size_t saves_n = attn_saves_floats(n);
  if (c->save_at < saves_n)
    return attn_fail("layer_attn_mixer_bwd: the arena holds no forward saves (was layer_attn_mixer_fwd run in JT_FWD_SAVE?)");
  AttnSave s;
  attn_save_ptrs(n, c->arena, &s);

  const size_t mark = jt_work_mark(c);
  const size_t n_q = (size_t)n * MT_O_IN;
  const size_t n_h = (size_t)n * MT_HIDDEN;
  float *o_gated = jt_work_alloc(c, n_q);     // o * sigmoid(gate): o_proj's x and sigmoid_gate_bwd's `o`
  float *d_o_gated = jt_work_alloc(c, n_q);   // o_proj's dx
  float *d_o = jt_work_alloc(c, n_q);         // the attention's cotangent (design §4.7's `do`)
  float *dgate = jt_work_alloc(c, n_q);       // the gate's cotangent, merged back into [n,4096] at the end
  float *gate = jt_work_alloc(c, n_q);        // re-split out of the saved qg (the gate itself is not saved)
  float *dqg = jt_work_alloc(c, (size_t)n * MT_Q_OUT); // q_proj's dx; the split's throwaway q half lands here first
  float *dq = jt_work_alloc(c, n_q);          // token-major dQ after the row loop, then rope_bwd's target
  float *dq_pre = jt_work_alloc(c, n_q);      // q_norm's dx (the merge's q half)
  float *dk = jt_work_alloc(c, (size_t)n * MT_KV_OUT);
  float *dk_raw = jt_work_alloc(c, (size_t)n * MT_KV_OUT); // k_norm's dx
  float *dv = jt_work_alloc(c, (size_t)n * MT_KV_OUT);
  float *da = jt_work_alloc(c, n_h);          // the three projections' shared input gradient (design §4.6)
  float *dxin = jt_work_alloc(c, n_h);        // the input norm's dx
  AttnBwdRow row;
  const int cap = attn_batch_cap(); // 0 = the per-row path; > 0 = a chunk of at most `cap` packed tokens
  attn_bwd_row_alloc(c, &row, attn_chunk_max(c, n, cap));

  // gate = qgate_split(qg)'s second half; q_pre is already saved, so the split's q half goes over dqg (which is only
  // filled by the merge at the end): one pass over [n,4096] instead of another 2048 floats per token in the saves.
  jt_qgate_split(s.qg, dqg, gate, n, MT_FULL_HEADS, MT_HEAD_DIM, c->stream);
  // o = o_pre * sigmoid(gate) again: it is both o_proj's x and the `o` of the gate's backward (design §4.7).
  k_sigmoid_gate_fwd(s.o_pre, gate, o_gated, n * MT_O_IN, c->stream);

  // 1. the output projection. dres is read here and not written until the last line, so it serves as dy directly —
  //    dx gets its own buffer (linear_bwd refuses dx == dy). dA/dB accumulate into the flat buffer.
  int rc = linear_bwd(c, &lw->o_proj, o_gated, s.u_o, dres, n, d_o_gated, 0);
  if (rc != JT_LAYER_OK) goto done; // linear_bwd printed why
  // 2. the output gate: do = dog sig(gate), dgate = dog o sig(gate) (1 - sig(gate)).
  k_sigmoid_gate_bwd(d_o_gated, s.o_pre, gate, d_o, dgate, n * MT_O_IN, c->stream);

  // 3. per chunk: design §8.2's second block, into the token-major dq/dk/dv (one layout pass and one way back per chunk;
  // with the toggle off a chunk is one row).
  for (int r0 = 0; r0 < b->rows;) {
    const int r1 = attn_chunk_end(b, r0, cap), coff = b->off[r0], cend = b->off[r1];
    if (cend > coff) {
      attn_chunk_prep_bwd(c, &s, &row, d_o, coff, cend - coff);
      for (int r = r0; r < r1; r++) {
        const int off = b->off[r], L = b->off[r + 1] - off;
        if (L > 0) attn_row_bwd(c, &row, coff, off, L, cend - coff);
      }
      attn_chunk_tail(c, &row, cend - coff, dq + (size_t)coff * MT_O_IN, dk + (size_t)coff * MT_KV_OUT,
                      dv + (size_t)coff * MT_KV_OUT);
    }
    r0 = r1;
  }

  // 4. RoPE backward (design §4.7: the rotation transposed) and the two norms (design §4.3, d = 256, zero-centred). Both
  //    are per (token, head) with row-local positions, so one launch covers the packed batch.
  rope_bwd(dq, b->off_dev, b->row_of, n, MT_FULL_HEADS, MT_HEAD_DIM, c->rope_cos, c->rope_sin, c->stream);
  rope_bwd(dk, b->off_dev, b->row_of, n, MT_KV_HEADS, MT_HEAD_DIM, c->rope_cos, c->rope_sin, c->stream);
  rmsnorm_bwd(s.q_pre, lw->q_norm, dq, s.r_q, dq_pre, n * MT_FULL_HEADS, MT_HEAD_DIM, 1, c->stream);
  rmsnorm_bwd(s.k_raw, lw->k_norm, dk, s.r_k, dk_raw, n * MT_KV_HEADS, MT_HEAD_DIM, 1, c->stream);

  // 5. the q/gate merge back into q_proj's [n,4096] layout, then the three projections off `a`. Every dx goes to the same
  //    buffer (`a` is the input of all three), so the first writes it and the other two add (design §4.6).
  jt_qgate_merge(dq_pre, dgate, dqg, n, MT_FULL_HEADS, MT_HEAD_DIM, c->stream);
  rc = linear_bwd(c, &lw->q_proj, s.a, s.u_qg, dqg, n, da, 0);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->k_proj, s.a, s.u_k, dk_raw, n, da, 1);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->v_proj, s.a, s.u_v, dv, n, da, 1);
  if (rc != JT_LAYER_OK) goto done;

  // 6. the input norm's backward and the residual (design §4.6): dres already holds dL/d(x_out), so this adds to it.
  rmsnorm_bwd(x_in, lw->input_norm, da, s.r_in, dxin, n, MT_HIDDEN, 1, c->stream);
  jt_add_inplace(dres, dxin, n_h, c->stream);

done:
  JT_SYNC_DEBUG; // a no-op unless -DJT_DEBUG=1, where it names a launch error at this line
  jt_work_restore(c, mark);
  return rc;
}
