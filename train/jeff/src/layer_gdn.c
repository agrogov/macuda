// src/layer_gdn.c — the GDN mixer: design §3.6 forward (plan M4.3 item 3, DONE) and §4.8 backward (still the plan J18
// stub; plan M5.1 item 3 writes it).
//
// Plan J18 froze the interface (src/layer.h), so this file is the GDN layer's owner's and nothing else changes when it
// does. The forward is design §3.6 in order, one already-tested call per step:
//
//   a   = RMSNorm(x_in; input_layernorm)                    rmsnorm_fwd (zero-centred, r saved)
//   qkv = in_proj_qkv(a)  z = in_proj_z(a)  aa = in_proj_a(a)  bb = in_proj_b(a)
//                                                           linear_fwd x4 (a and b are frozen: lora == 0)
//   pre = conv(qkv); c = silu(pre)                          conv_fwd (row-local taps, design §7)
//   q | k | v = c as three contiguous [n][2048] slices      jt_to_head_major (see below)
//   q = l2norm(q) / sqrt(128);  k = l2norm(k)                l2norm_fwd, rows = n*16: one (token, head) pair each
//   beta = sigmoid(bb), g = -exp(A_log) softplus(aa+dt_bias) k_gdn_gates_fwd(lw->a_log, lw->dt_bias)
//   o   = gdn(q, k, v, g, beta) over the packed rows        gdn_fwd(b->off_dev, b->rows, 16, c->gdn_nseg, ...)
//   y   = gated_rmsnorm(o; gdn_norm.weight, gate = z)       gated_rmsnorm_fwd (w, NOT 1 + w: plan M2.2's pitfall)
//   out = out_proj(y)                                       linear_fwd
//   x_out = x_in + out                                      the residual, design §4.6
//
// ckpt is c->gdn_ckpt in JT_FWD_SAVE mode and NULL in JT_FWD_ONLY: the backward replays from the recompute pass's
// checkpoints (design §8.3), so a forward sweep keeps none. c->gdn_work is the backward's scratch, not this file's.
//
// Three decisions the frozen kernels forced, all visible below:
//
//  * The q/k/v split is a copy, not the plan's "no copy" view into the [n][6144] conv output. Neither l2norm_fwd nor
//    gdn_fwd takes a row stride (kernels/norm.h: rows are contiguous of length d; kernels/gdn.h: q is [N][H][DIM] with no
//    pitch), so a view with stride 6144 would be read as garbage from the second token on. One jt_to_head_major call
//    turns the token-major [n][3][2048] into exactly the three contiguous slices those kernels want, in one pass over the
//    3 N 2048 floats that the l2norm and the recurrence read anyway — the cheapest correct alternative to a new kernel,
//    which plan rule 4 keeps out of this file's reach.
//  * linear_fwd always overwrites (design §4.2's first GEMM has beta = 0), so it cannot accumulate the residual and
//    x_out may alias x_in (src/layer.h's contract): out_proj lands in a work buffer and the sum is written into x_out
//    last, after a copy of x_in when the two are distinct buffers.
//  * The work buffers are released with jt_work_restore before returning even though the kernels above are only
//    *ordered* on ctx->stream, not finished: that is the arena's rule (src/linear.c does the same), and a later call
//    reusing those words is queued behind them on the same stream.
//
// What JT_FWD_SAVE keeps is exactly what design §4.8 reads, and nothing else: a and the RMSNorm's r; the two adapted
// linears' u (in_proj_qkv, in_proj_z — in_proj_a / in_proj_b are frozen and have none); aa (k_gdn_gates_bwd's input) with
// beta and g; the conv's pre (for silu's derivative in conv_bwd); the post-silu, pre-l2norm q|k|v (ck) and the l2norm'd
// q|k (qkn) with their rq/rk (l2norm_bwd needs both forms, gdn_bwd wants the normalised one); z and o and the gated
// norm's ro; and y, the gated norm's output, which out_proj's linear_bwd takes as its x. bb, qkv, cv and out are not
// saved: design §4.8 derives everything they would say from beta, pre and y.
#include "layer.h"

#include <math.h>
#include <stdio.h>

#include "act.h"  // k_gdn_gates_fwd
#include "conv.h" // conv_fwd
#include "gdn.h"  // gdn_fwd, gdn_ckpt_floats, JT_GDN_*
#include "misc.h" // jt_to_head_major, jt_add_inplace, JT_HIDDEN
#include "norm.h" // rmsnorm_fwd, l2norm_fwd, gated_rmsnorm_fwd
#include "util.h" // CK

// The recurrence kernel bakes Dk = Dv = 128 in (kernels/gdn.h); the model's head_dim has to be that number, and the
// projections' widths follow from it (design §3.6: 16 heads x 128 = 2048 for q/k/v/z, 3 x 2048 = 6144 for qkv).
_Static_assert(MT_GDN_HEAD_DIM == JT_GDN_DIM, "the GDN kernel's head dim and the model's disagree");
#define GDN_HEADS MT_GDN_HEADS       // 16 key heads == value heads (no repeat_interleave, design §3.6)
#define GDN_W MT_GDN_Z_OUT           // 2048: q, k, v, z, o and y are all this wide
#define GDN_QKV_W MT_GDN_QKV_OUT     // 6144: in_proj_qkv's width and the conv's channel count

static int bad(const char *why) {
  fprintf(stderr, "layer_gdn: layer_gdn_mixer_fwd: %s\n", why);
  return JT_LAYER_FAIL;
}

// The forward's buffers, in design §3.6's order. `save` picks the arena half (see the header comment): a saved buffer
// comes from the front in JT_FWD_SAVE mode, everything else from the back as a temporary. The widths are floats per
// packed token, or per (token, head) for the norms' r arrays.
typedef struct {
  float *a, *ra;             // RMSNorm(x_in), and its r                       [1024], [1]
  float *uqkv, *uz;          // the adapted linears' u                          [16], [16]
  float *aa, *bb, *beta, *g; // the gate inputs and outputs                     [16] each
  float *qkv, *pre, *cv;     // in_proj_qkv, the conv's pre, and silu(pre)      [6144] each
  float *ck, *qkn;           // q|k|v post-silu, then the l2norm'd q|k          [3 x 2048], [2 x 2048]
  float *rq, *rk;            // the l2norm's r, per (token, head)              [16] each
  float *z, *y, *o;          // in_proj_z, the gated norm's out, gdn's o        [2048] each
  float *ro;                 // the gated norm's r, per (token, head)           [16]
  float *out;                // out_proj's y                                   [1024]
} GdnBufs;

static float *take(JtCtx *c, int save, size_t width, int n) {
  return save ? jt_save_alloc(c, width * (size_t)n) : jt_work_alloc(c, width * (size_t)n);
}

static void bufs_get(JtCtx *c, int save, int n, GdnBufs *f) {
  f->a = take(c, save, MT_HIDDEN, n);
  f->ra = take(c, save, 1, n);
  f->uqkv = take(c, save, LORA_RANK, n);
  f->uz = take(c, save, LORA_RANK, n);
  f->aa = take(c, save, GDN_HEADS, n);
  f->bb = take(c, 0, GDN_HEADS, n); // design §4.8's dbb comes from beta, so bb itself is not saved
  f->beta = take(c, save, GDN_HEADS, n);
  f->g = take(c, save, GDN_HEADS, n);
  f->qkv = take(c, 0, GDN_QKV_W, n); // the conv reads it and nothing upstream does again
  f->pre = take(c, save, GDN_QKV_W, n);
  f->cv = take(c, 0, GDN_QKV_W, n); // silu(pre): pre is the saved form (conv_bwd's derivative)
  f->ck = take(c, save, 3 * GDN_W, n);
  f->qkn = take(c, save, 2 * GDN_W, n);
  f->rq = take(c, save, GDN_HEADS, n);
  f->rk = take(c, save, GDN_HEADS, n);
  f->z = take(c, save, GDN_W, n);
  f->y = take(c, save, GDN_W, n);
  f->o = take(c, save, GDN_W, n);
  f->ro = take(c, save, GDN_HEADS, n);
  f->out = take(c, 0, MT_HIDDEN, n);
}

// The four input projections, in design §3.6's order. Only the x of an adapted linear's u is kept (JT_FWD_SAVE), and a
// frozen linear (lora == 0) is one GEMM with no u at all (design §2.1, §4.2).
static int project(JtCtx *c, const LayerWeights *lw, const GdnBufs *f, int n, int save) {
  int rc = linear_fwd(c, &lw->in_proj_qkv, f->a, n, f->qkv, save ? f->uqkv : NULL);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->in_proj_z, f->a, n, f->z, save ? f->uz : NULL);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->in_proj_a, f->a, n, f->aa, NULL);
  if (rc == JT_LAYER_OK) rc = linear_fwd(c, &lw->in_proj_b, f->a, n, f->bb, NULL);
  return rc;
}

int layer_gdn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                        int mode) {
  if (c == NULL || lw == NULL || x_in == NULL || x_out == NULL || b == NULL) return bad("null argument");
  if (n != b->n) return bad("n does not match the batch's token count");
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE) return bad("mode is neither JT_FWD_ONLY nor JT_FWD_SAVE");
  if (n == 0) return JT_LAYER_OK;

  const int save = (mode == JT_FWD_SAVE);
  const int nt = n * GDN_HEADS; // rows of every per-head norm: one (token, head) pair each (design §8.1)
  const size_t mark = jt_work_mark(c);
  GdnBufs f;
  bufs_get(c, save, n, &f);

  // a = RMSNorm(x_in; input_layernorm), zero-centred: y = x r (1 + w) (design §3.3).
  rmsnorm_fwd(x_in, lw->input_norm, f.a, f.ra, n, MT_HIDDEN, 1, c->stream);

  int rc = project(c, lw, &f, n, save);
  if (rc != JT_LAYER_OK) {
    jt_work_restore(c, mark);
    return rc;
  }

  // pre = conv(qkv), c = silu(pre): the causal depthwise conv of design §3.6, taps never crossing a row (§7).
  conv_fwd(f.qkv, lw->conv1d, b->row_of, b->off_dev, f.pre, f.cv, n, GDN_QKV_W, c->stream);

  // q | k | v = c[:,0:2048], c[:,2048:4096], c[:,4096:6144] as three contiguous [n][2048] slices (why a copy: the
  // header comment; jt_to_head_major's [heads][rows][head_dim] with heads = 3 is exactly that split).
  jt_to_head_major(f.cv, f.ck, n, 3, GDN_W, c->stream);
  const size_t n2 = (size_t)n * GDN_W;
  const float *k = f.ck + n2, *v = f.ck + 2 * n2;

  // l2norm per (token, head): q carries 1/sqrt(128), k does not (design §3.6), and the norm is a sum, not a mean
  // (design §3.3). The two r arrays are the backward's, so they are kept in SAVE mode.
  l2norm_fwd(f.ck, 1.0f / sqrtf((float)MT_GDN_HEAD_DIM), f.qkn, f.rq, nt, MT_GDN_HEAD_DIM, c->stream);
  l2norm_fwd(k, 1.0f, f.qkn + n2, f.rk, nt, MT_GDN_HEAD_DIM, c->stream);

  // beta = sigmoid(bb); g = -exp(A_log) * softplus(aa + dt_bias), fp32 (design §3.6). A_log, not neg_a: the kernel's
  // expression already has the minus sign.
  k_gdn_gates_fwd(f.bb, f.aa, lw->a_log, lw->dt_bias, f.beta, f.g, nt, GDN_HEADS, c->stream);

  // The recurrence, one grid over the packed rows (state restarts at each row's first token, design §7/§8.3).
  gdn_fwd(f.qkn, f.qkn + n2, v, f.g, f.beta, b->off_dev, b->rows, GDN_HEADS, c->gdn_nseg,
          save ? c->gdn_ckpt : NULL, f.o, c->stream);

  // y = GatedRMSNorm(o; gdn_norm.weight, gate = z): y = w (x r) silu(z), per head, d = 128 (design §3.3/§4.4).
  gated_rmsnorm_fwd(f.o, lw->gdn_norm, f.z, f.y, f.ro, nt, MT_GDN_HEAD_DIM, 0, c->stream);

  // out = out_proj(y), then x_out = x_in + out (design §4.6). x_out MAY alias x_in and linear_fwd overwrites, so the sum
  // is formed in x_out only after the copy it needs.
  rc = linear_fwd(c, &lw->out_proj, f.y, n, f.out, NULL);
  if (rc == JT_LAYER_OK) {
    if (x_out != x_in)
      CK(cudaMemcpyAsync(x_out, x_in, (size_t)n * MT_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    jt_add_inplace(x_out, f.out, (size_t)n * MT_HIDDEN, c->stream);
  }
  jt_work_restore(c, mark);
  return rc;
}

int layer_gdn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  (void)c, (void)lw, (void)x_in, (void)dres, (void)n, (void)b;
  fprintf(stderr, "layer_gdn: layer_gdn_mixer_bwd: not implemented (plan M5.1)\n");
  return JT_LAYER_NOT_IMPL;
}
