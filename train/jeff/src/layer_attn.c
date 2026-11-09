// src/layer_attn.c — the full-attention mixer: design §3.5 forward, §4.7 backward, on the kernels of plan M2.
// This file is J22 (plan M4.3 item 4): layer_attn_mixer_fwd. layer_attn_mixer_bwd is still J18's stub — plan M5.1
// item 4 writes it, against the saves the forward below leaves.
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
//   per row: to_head_major, kv_expand, S = q k^T (gemm_batched), attn_softmax_fwd(scale 1/16), PV (gemm_batched),
//            from_head_major                                     design §5, §8.2; one row's P at a time, bounded by c->lmax
//   o    = o * sigmoid(gate)                                  k_sigmoid_gate_fwd
//   out  = o_proj(o)                                          linear_fwd
//   x_out= x_in + out                                         the residual, design §4.6
//
// The backward is design §4.7: o_proj, sigmoid_gate_bwd, then per row the four attention GEMMs and
// attn_softmax_bwd(P, dP) (P recomputed for the row, since only one row's P is ever kept), then rope_bwd, q_norm/k_norm
// rmsnorm_bwd (d = 256), qgate_merge back into [n,4096], kv_reduce for dk/dv (heads h/4), and finally the four
// linear_bwd calls accumulating dA/dB into the flat buffer.
//
// ---- what a JT_FWD_SAVE pass saves (design §6.2, §4.7) --------------------------------------------------------------
// The backward takes no activation pointers, so it re-derives them from the arena by running attn_saves() in the same
// order; that is the whole "save layout" and it is one function for that reason. In JT_FWD_SAVE the buffers come from
// the arena's save half, in JT_FWD_ONLY from its work half (freed before the function returns), and the arithmetic is
// identical either way:
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
#include "layer.h"

#include <stdio.h>

#include "act.h"    // k_sigmoid_gate_fwd
#include "attn.h"   // attn_softmax_fwd
#include "gemm.h"   // gemm_batched
#include "linear.h" // linear_fwd
#include "misc.h"   // jt_qgate_split, jt_kv_expand, jt_to_head_major, jt_from_head_major, jt_add_inplace
#include "norm.h"   // rmsnorm_fwd
#include "rope.h"   // rope_fwd
#include "util.h"   // CK

// 256^-0.5 (design §3.5, §4.7): attn_softmax_fwd applies it to the raw GEMM scores, and its backward takes the same
// number. Exactly 1/16 in fp32, so it is written as one.
#define ATTN_SCALE 0.0625f

static int attn_fail(const char *msg) {
  fprintf(stderr, "layer_attn: %s\n", msg);
  return JT_LAYER_FAIL;
}

// The save layout (see the header comment). One function, called by the fwd(SAVE) now and by the bwd later, so the two
// cannot disagree about which buffer holds what. `mode` picks the arena half; the order is fixed and must not change.
typedef struct {
  float *r_in, *a, *qg, *q_pre, *r_q, *q, *k_raw, *r_k, *k, *v, *o_pre;
  float *u_qg, *u_k, *u_v, *u_o;
} AttnSave;

static void attn_saves(JtCtx *c, int n, int mode, AttnSave *s) {
  float *(*take)(JtCtx *, size_t) = (mode == JT_FWD_SAVE) ? jt_save_alloc : jt_work_alloc;
  const size_t n_qg = (size_t)n * MT_Q_OUT;
  const size_t n_q = (size_t)n * MT_O_IN;
  const size_t n_kv = (size_t)n * MT_KV_OUT;
  const size_t n_u = (size_t)n * MT_LORA_RANK;
  s->r_in = take(c, (size_t)n);
  s->a = take(c, (size_t)n * MT_HIDDEN);
  s->qg = take(c, n_qg);
  s->q_pre = take(c, n_q);
  s->r_q = take(c, (size_t)n * MT_FULL_HEADS);
  s->q = take(c, n_q);
  s->k_raw = take(c, n_kv);
  s->r_k = take(c, (size_t)n * MT_KV_HEADS);
  s->k = take(c, n_kv);
  s->v = take(c, n_kv);
  s->o_pre = take(c, n_q);
  s->u_qg = take(c, n_u);
  s->u_k = take(c, n_u);
  s->u_v = take(c, n_u);
  s->u_o = take(c, n_u);
}

// One row's attention buffers (design §8.2): [8][lmax][256] head-major for q, k, v and o, one token-major [lmax][2048]
// scratch for kv_expand's output, and the [8][lmax][lmax] scores and probabilities. Sized once for c->lmax and reused by
// every row, so a batch costs the longest row, not the sum. That is the arena's extra 2 x 8 x lmax^2 term plus about
// 10 K floats per position; jt_work_alloc dies with the numbers if the arena is ever sized too small.
typedef struct {
  float *qh, *kh, *vh, *oh, *kv_tok, *sc, *p;
} AttnRow;

static void attn_row_alloc(JtCtx *c, AttnRow *r) {
  const size_t l = (size_t)c->lmax;
  const size_t hm = (size_t)MT_FULL_HEADS * l * MT_HEAD_DIM;
  const size_t sq = (size_t)MT_FULL_HEADS * l * l;
  r->qh = jt_work_alloc(c, hm);
  r->kh = jt_work_alloc(c, hm);
  r->vh = jt_work_alloc(c, hm);
  r->oh = jt_work_alloc(c, hm);
  r->kv_tok = jt_work_alloc(c, l * MT_O_IN);
  r->sc = jt_work_alloc(c, sq);
  r->p = jt_work_alloc(c, sq);
}

// One row's slice [off, off+L) of design §8.2, in the order the design lists it. q, k and v are token-major; the GEMMs
// want head-major per head, so to_head_major does the transpose and kv_expand copies each KV group to its 4 query heads
// (h = g*group + r, i.e. group = h/4) before the same transpose. o goes back to token-major, into the caller's o_pre.
static void attn_row(JtCtx *c, const AttnSave *s, AttnRow *r, int off, int L) {
  // h = g*group + r (design §3.5): the 8 query heads are 4 per KV head, so kv_expand copies group g to heads
  // [g*group, g*group+group). MT_KV_HEADS x group = MT_FULL_HEADS is the model's only consistent value.
  const int group = MT_FULL_HEADS / MT_KV_HEADS;
  const long hm = (long)L * (long)MT_HEAD_DIM;
  const long sq = (long)L * (long)L;
  cudaStream_t st = c->stream;

  jt_to_head_major(s->q + (size_t)off * MT_O_IN, r->qh, L, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->k + (size_t)off * MT_KV_OUT, r->kv_tok, L, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->kh, L, MT_FULL_HEADS, MT_HEAD_DIM, st);
  jt_kv_expand(s->v + (size_t)off * MT_KV_OUT, r->kv_tok, L, MT_KV_HEADS, group, MT_HEAD_DIM, st);
  jt_to_head_major(r->kv_tok, r->vh, L, MT_FULL_HEADS, MT_HEAD_DIM, st);

  // Sc[h] = Q[h] K[h]^T  [L,L]: A is stored [M,K] = [L,256], B is stored [N,K] = [L,256] (hence tb = 1).
  gemm_batched(c->cublas, 0, 1, L, L, MT_HEAD_DIM, 1.0f, r->qh, hm, r->kh, hm, 0.0f, r->sc, sq, MT_FULL_HEADS);
  attn_softmax_fwd(r->sc, r->p, L, MT_FULL_HEADS, ATTN_SCALE, st); // P[h] = causal_softmax(Sc[h] / 16)
  // O[h] = P[h] V[h]  [L,256]: A is stored [M,K] = [L,L], B is stored [K,N] = [L,256] (tb = 0).
  gemm_batched(c->cublas, 0, 0, L, MT_HEAD_DIM, L, 1.0f, r->p, sq, r->vh, hm, 0.0f, r->oh, hm, MT_FULL_HEADS);
  jt_from_head_major(r->oh, s->o_pre + (size_t)off * MT_O_IN, L, MT_FULL_HEADS, MT_HEAD_DIM, st);
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
  AttnSave s;
  attn_saves(c, n, mode, &s);

  float *gate = jt_work_alloc(c, (size_t)n * MT_O_IN); // the split gate; re-derived from qg by the bwd, so not saved
  float *o_gated = jt_work_alloc(c, (size_t)n * MT_O_IN);
  float *out = jt_work_alloc(c, (size_t)n * MT_HIDDEN);
  AttnRow row;
  attn_row_alloc(c, &row);

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

  for (int r = 0; r < b->rows; r++) {
    const int off = b->off[r], L = b->off[r + 1] - off;
    if (L > 0) attn_row(c, &s, &row, off, L);
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
  (void)c, (void)lw, (void)x_in, (void)dres, (void)n, (void)b;
  fprintf(stderr, "layer_attn: layer_attn_mixer_bwd: not implemented (plan M5.1)\n");
  return JT_LAYER_NOT_IMPL;
}
