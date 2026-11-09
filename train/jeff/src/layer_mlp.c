// src/layer_mlp.c — the post-attention norm and the MLP, shared by both layer kinds (design §3.4's last two lines, §4.5).
//
//   m    = RMSNorm(x_in; post_attention_layernorm)          rmsnorm_fwd, zero-centred, r saved for the bwd
//   h1   = gate(m), h2 = up(m)                              linear_fwd (LoRA on both), u = m A^T kept
//   act  = silu(h1) * h2                                    k_silu_mul_fwd
//   y    = down(act)                                        linear_fwd
//   x_out= x_in + y                                         the residual, design §4.6
//
// Design §3.4 and D§4.5 read exactly like that, and every call below is a function that already exists and is tested
// (kernels/norm.h, src/linear.h, kernels/act.h); this file is only their order, their buffers and their sizes. The MLP is
// per token — design §7: "every per-token op (linears, norms, MLP, GDN projections, embeddings) runs on all N tokens at
// once, ignoring row boundaries" — so the row table `b` is not read here at all.
//
// ---- the saves (JT_FWD_SAVE), in the order they are taken from the arena's save half -------------------------------
//
// A bwd reads all of these (and adds its own work temporaries from the back), so they are taken as ONE block in ONE order,
// in one place (saves_floats / saves_bind), because jt_save_alloc is a bump allocator with no names:
//
//   1  m         [n][1024]     the post-attention norm's output
//   2  r         [n]           RMSNorm's 1/sqrt(mean(x^2)+eps) (design §4.3)
//   3  u_gate    [n][16]       the gate's LoRA u (src/linear.h)
//   4  u_up      [n][16]       the up's
//   5  h1        [n][3584]     gate(m) — dh1 = dact * h2 * dsilu(h1) needs it
//   6  h2        [n][3584]     up(m)   — dh2 = dact * silu(h1) needs it
//   7  act       [n][3584]     silu(h1) * h2 — down's input
//   8  u_down    [n][16]       the down's LoRA u
//
// 11,825 floats per token, well inside JT_ARENA_FLOATS_PER_TOKEN (98304). In JT_FWD_ONLY nothing is saved: m, r, h1, h2
// and act come from the work half instead and are given back before returning (a bwd never follows that pass), and the
// three u's are linear_fwd's own work temporaries (u_save == NULL, src/linear.c).
//
// ---- the backward (plan M5.1 item 3, design §4.5) --------------------------------------------------------------------
//
//   dact = dout through down                     linear_bwd(down): dy = dres, dx = dact (fresh), dA/dB for down
//   dh1  = dact * h2 * dsilu(h1)                 k_silu_mul_bwd over all n * 3584 values
//   dh2  = dact * silu(h1)                       the same kernel's second output
//   dm   = (dh1 through gate) + (dh2 through up) linear_bwd(gate) into dm (fresh), then linear_bwd(up) with
//                                                accumulate_dx = 1; dA/dB for both
//   dres += RMSNormBackward(dm)                  rmsnorm_bwd into a work buffer, then jt_add_inplace into dres
//
// `dout` is dres on entry — dL/d(x_out), the running residual gradient that layer_fwd's `x_out = x_in + y` gives the
// down projection — and on return dres holds dL/d(x_in) = dL/d(x_out) + the norm's backward (design §4.6: accumulate, never
// overwrite). The two dm GEMMs are the only place the two paths meet: gate and up share the same input m, so their dx's
// add instead of overwrite.
//
// The temporaries (dact, dh1, dh2 [n][3584]; dm, dnorm [n][1024]) are 12,800 floats per token, taken from the arena's
// work half and given back before returning (design §6.2: no cudaMalloc inside a layer call). dres is read by the down's
// linear_bwd and written only at the very end, so no GEMM output aliases an input.
//
// ---- the residual and the frozen contract ----------------------------------------------------------------------------
//
// x_out may alias x_in (src/layer.h: layer_fwd passes the same buffer for the MLP's input and output in a JT_FWD_ONLY
// pass), so x_in is read exactly once — by the input norm, before anything writes x_out. The down projection writes a
// work temporary `y` and the residual is then two elementwise passes (`x_out = x_in`, `x_out += y`): no GEMM ever writes
// an output that still holds an input, and no temporary aliases another.
//
#include "layer.h"

#include "act.h"    // k_silu_mul_fwd (design §4.5's act = silu(h1) * h2)
#include "linear.h" // linear_fwd
#include "misc.h"   // jt_add_inplace
#include "norm.h"   // rmsnorm_fwd
#include "util.h"   // CK, JT_SYNC_DEBUG

#include <stdio.h>

// A failure names the piece, like src/linear.c's bad_args: the three linears carry their adapter stem (design §9.2), so
// a message points at the tensor and not only at this file.
static int bad_args(const char *fn, const char *why, const char *stem) {
  fprintf(stderr, "layer_mlp: %s: %s (%s)\n", fn, why, stem != NULL ? stem : "?");
  return JT_LAYER_FAIL;
}

// The forward's per-token buffers. In JT_FWD_SAVE they are the arena's saves (in the order above); in JT_FWD_ONLY they
// are work temporaries. The three u's are only materialised when they are saved (NULL means "linear_fwd, use your own").
typedef struct {
  float *m, *r, *h1, *h2, *act;
  float *u_gate, *u_up, *u_down;
} Acts;

// The save block's size in floats: the eight arrays of the header comment, in that order — m [n][1024], r [n], the three
// u's [n][16], h1 / h2 / act [n][3584] — i.e. 11,825 floats per token.
static size_t saves_floats(int n) {
  return (size_t)n * ((size_t)MT_HIDDEN + 1 + 3 * (size_t)MT_MLP_INTER + 3 * (size_t)LORA_RANK);
}

// Lay the eight saves out over one block. The fwd(SAVE) hands it the block jt_save_alloc returned and the bwd hands it the
// arena's base; one function, so the order and the sizes cannot drift apart between the two halves.
static void saves_bind(float *base, int n, Acts *a) {
  const size_t hidden_n = (size_t)n * MT_HIDDEN;
  const size_t inter_n = (size_t)n * MT_MLP_INTER;
  const size_t u_n = (size_t)n * LORA_RANK;
  float *p = base;
  a->m = p, p += hidden_n;
  a->r = p, p += (size_t)n;
  a->u_gate = p, p += u_n;
  a->u_up = p, p += u_n;
  a->h1 = p, p += inter_n;
  a->h2 = p, p += inter_n;
  a->act = p, p += inter_n;
  a->u_down = p, p += u_n;
}

// The forward's buffers: one save block in SAVE mode, work temporaries (and no u's: linear_fwd keeps its own) in ONLY mode.
static void acts_alloc(JtCtx *c, int save, int n, Acts *a) {
  if (save) {
    saves_bind(jt_save_alloc(c, saves_floats(n)), n, a);
    return;
  }
  const size_t hidden_n = (size_t)n * MT_HIDDEN;
  const size_t inter_n = (size_t)n * MT_MLP_INTER;
  a->m = jt_work_alloc(c, hidden_n);
  a->r = jt_work_alloc(c, (size_t)n);
  a->h1 = jt_work_alloc(c, inter_n);
  a->h2 = jt_work_alloc(c, inter_n);
  a->act = jt_work_alloc(c, inter_n);
  a->u_gate = a->u_up = a->u_down = NULL; // linear_fwd's own temporaries (src/linear.c)
}

// m -> h1/h2 -> act -> y -> the residual. The caller has allocated `a` and the work temporary `y`.
static int mlp_path(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, float *x_out, Acts *a, float *y) {
  const size_t hidden_n = (size_t)n * MT_HIDDEN;
  const size_t inter_n = (size_t)n * MT_MLP_INTER;

  rmsnorm_fwd(x_in, lw->post_attn_norm, a->m, a->r, n, MT_HIDDEN, 1, c->stream); // 1: zero-centred (design §3.3)
  int rc = linear_fwd(c, &lw->gate, a->m, n, a->h1, a->u_gate);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->up, a->m, n, a->h2, a->u_up);
  if (rc != JT_LAYER_OK) return rc;
  k_silu_mul_fwd(a->h1, a->h2, a->act, (int)inter_n, c->stream);
  rc = linear_fwd(c, &lw->down, a->act, n, y, a->u_down);
  if (rc != JT_LAYER_OK) return rc;

  if (x_out != x_in) // x_in was read once, by the norm above, so writing the residual now is safe in both aliasing cases
    CK(cudaMemcpyAsync(x_out, x_in, hidden_n * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
  jt_add_inplace(x_out, y, hidden_n, c->stream); // the residual (design §4.6)
  JT_SYNC_DEBUG; // a no-op unless -DJT_DEBUG=1, where it names a launch error at this line
  return JT_LAYER_OK;
}

int layer_mlp_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out, int mode) {
  (void)b; // per token only (design §7): no row boundary is crossed here
  if (c == NULL || lw == NULL || x_in == NULL || x_out == NULL) return bad_args("layer_mlp_fwd", "null argument", "?");
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE)
    return bad_args("layer_mlp_fwd", "mode must be JT_FWD_ONLY or JT_FWD_SAVE", "?");
  if (lw->post_attn_norm == NULL)
    return bad_args("layer_mlp_fwd", "no post_attention_layernorm weight", "post_attention_layernorm");
  if (lw->gate.W == NULL || lw->up.W == NULL || lw->down.W == NULL)
    return bad_args("layer_mlp_fwd", "an MLP linear has no frozen weight",
                    lw->gate.W == NULL ? lw->gate.stem : (lw->up.W == NULL ? lw->up.stem : lw->down.stem));
  if (n < 0) return bad_args("layer_mlp_fwd", "negative n", "?");
  if (n == 0) return JT_LAYER_OK; // no tokens: nothing to compute, and the buffers below would be zero-length

  const size_t mark = jt_work_mark(c); // everything taken from the work half here is given back before returning
  Acts a;
  acts_alloc(c, mode == JT_FWD_SAVE, n, &a);
  float *y = jt_work_alloc(c, (size_t)n * MT_HIDDEN); // down's output, before it is added to the residual
  const int rc = mlp_path(c, lw, x_in, n, x_out, &a, y);
  jt_work_restore(c, mark);
  return rc;
}

int layer_mlp_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  (void)b; // per token only (design §7), as in the forward
  if (c == NULL || lw == NULL || x_in == NULL || dres == NULL) return bad_args("layer_mlp_bwd", "null argument", "?");
  if (lw->post_attn_norm == NULL)
    return bad_args("layer_mlp_bwd", "no post_attention_layernorm weight", "post_attention_layernorm");
  if (lw->gate.W == NULL || lw->up.W == NULL || lw->down.W == NULL)
    return bad_args("layer_mlp_bwd", "an MLP linear has no frozen weight",
                    lw->gate.W == NULL ? lw->gate.stem : (lw->up.W == NULL ? lw->up.stem : lw->down.stem));
  if (n < 0) return bad_args("layer_mlp_bwd", "negative n", "?");
  if (n == 0) return JT_LAYER_OK; // no tokens: no gradient to propagate, and the kernels below would launch nothing
  // The bwd reads the fwd's saves; a block bigger than the whole arena can only mean "was layer_mlp_fwd run SAVE?"
  if (saves_floats(n) > c->arena_n)
    return bad_args("layer_mlp_bwd", "the MLP's saves are bigger than the arena (was layer_mlp_fwd run SAVE?)", "?");

  const size_t hidden_n = (size_t)n * MT_HIDDEN;
  const size_t inter_n = (size_t)n * MT_MLP_INTER;
  Acts s;
  saves_bind(c->arena, n, &s); // layer_mlp_fwd(SAVE)'s block, in its order from the arena's base

  const size_t mark = jt_work_mark(c);
  float *dact = jt_work_alloc(c, inter_n); // dL/d(act): taken first, so a test can read it at mark - inter_n
  float *dh1 = jt_work_alloc(c, inter_n);
  float *dh2 = jt_work_alloc(c, inter_n);
  float *dm = jt_work_alloc(c, hidden_n); // dL/d(m): the two projections' summed dx (design §4.5)
  float *dnorm = jt_work_alloc(c, hidden_n); // dL/d(x_in) from the norm alone, before it is added to dres

  // 1. dact = dout through down. dres is dL/d(x_out) and also the down's dy: dx = dact (fresh), never over dy.
  int rc = linear_bwd(c, &lw->down, s.act, s.u_down, dres, n, dact, 0);
  // 2. dh1 = dact * h2 * dsilu(h1), dh2 = dact * silu(h1) (design §4.5's two lines, one kernel).
  if (rc == JT_LAYER_OK) k_silu_mul_bwd(dact, s.h1, s.h2, dh1, dh2, (int)inter_n, c->stream);
  // 3. dm = (dh1 through gate) + (dh2 through up): the second writes with accumulate_dx = 1 (design §4.6's rule).
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->gate, s.m, s.u_gate, dh1, n, dm, 0);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->up, s.m, s.u_up, dh2, n, dm, 1);
  // 4. dres += the post-attention norm's backward: dL/d(x_in) = dL/d(x_out) + dnorm_backward(dm) (design §4.6).
  if (rc == JT_LAYER_OK) {
    rmsnorm_bwd(x_in, lw->post_attn_norm, dm, s.r, dnorm, n, MT_HIDDEN, 1, c->stream);
    jt_add_inplace(dres, dnorm, hidden_n, c->stream);
    JT_SYNC_DEBUG;
  }
  jt_work_restore(c, mark); // the temporaries are gone; the GEMMs and kernels that read them are ordered before reuse
  return rc;
}
