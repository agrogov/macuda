// src/layer_gdn.c — the GDN mixer: design §3.6 forward (plan M4.3 item 3, J21) and §4.8 backward (plan M5.1 item 3, J28).
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
// What JT_FWD_SAVE keeps is exactly what design §4.8 reads, and nothing else: a and the RMSNorm's r; the three adapted
// linears' u (in_proj_qkv, in_proj_z, out_proj — in_proj_a / in_proj_b are frozen and have none); aa (k_gdn_gates_bwd's
// input) with beta and g; the conv's pre (for silu's derivative in conv_bwd); the post-silu, pre-l2norm q|k|v (ck) and
// the l2norm'd q|k (qkn) with their rq/rk (l2norm_bwd needs both forms, gdn_bwd wants the normalised one); z and o and
// the gated norm's ro; and y, the gated norm's output, which out_proj's linear_bwd takes as its x. bb, qkv, cv and out
// are not saved: design §4.8 derives everything they would say from beta, pre and y. out_proj's u is saved too, because
// its linear_bwd needs it and the forward is the only place it exists (J28; the forward's numbers do not change).
//
// The backward is design §4.8 in reverse, one already-tested call per step (J28):
//
//   out_proj          dres (= dL/d(out)) -> dy = dL/d(y); dA/dB of out_proj into the flat buffer   linear_bwd
//   y = gnorm(o; z)   dy -> do = dL/d(o), dz = dL/d(z)                                            gated_rmsnorm_bwd
//   recurrence        (q, k, v, g, beta, do) -> dq, dk, dv, dg, dbeta, from the fwd's checkpoints  gdn_bwd
//   l2norm q and k    dq -> dqc (q carries 1/sqrt(128)), dk -> dkc, the two per-head norms          l2norm_bwd x2
//   cv = silu(pre)    dqc|dkc|dv -> dcv [n][6144], then dpre's transposed conv -> dqkv              jt_from_head_major, conv_bwd
//   gates             (dbeta, dg) -> dbb, daa, with beta and aa from the saves                     k_gdn_gates_bwd
//   four projections  dqkv / dz / dbb / daa -> da, the first writing and the other three adding    linear_bwd x4
//   input norm        a = RMSNorm(x_in) -> dres += input_layernorm's dx                            rmsnorm_bwd
//
// `dres` is the running residual gradient (design §4.6): it enters as dL/d(x_out) = dL/d(out), is read once as
// out_proj's dy, and receives the norm's contribution at the end. Nothing is overwritten and nothing is allocated: the
// temporaries come from the arena's work half and go back before returning. Every buffer below is per packed token
// (the mixer is token-wise except the conv and the recurrence, which take the row table), 29,760 floats per token — and
// the forward's saves are 23,697, both well inside JT_ARENA_FLOATS_PER_TOKEN's 98,304.
#include "layer.h"

#include <math.h>
#include <stdio.h>

#include "act.h"  // k_gdn_gates_fwd / k_gdn_gates_bwd
#include "conv.h" // conv_fwd / conv_bwd
#include "gdn.h"  // gdn_fwd / gdn_bwd, gdn_ckpt_floats, JT_GDN_*
#include "misc.h" // jt_to_head_major, jt_from_head_major, jt_add_inplace, JT_HIDDEN
#include "norm.h" // rmsnorm_fwd/bwd, l2norm_fwd/bwd, gated_rmsnorm_fwd/bwd
#include "util.h" // CK

// The recurrence kernel bakes Dk = Dv = 128 in (kernels/gdn.h); the model's head_dim has to be that number, and the
// projections' widths follow from it (design §3.6: 16 heads x 128 = 2048 for q/k/v/z, 3 x 2048 = 6144 for qkv).
_Static_assert(MT_GDN_HEAD_DIM == JT_GDN_DIM, "the GDN kernel's head dim and the model's disagree");
#define GDN_HEADS MT_GDN_HEADS       // 16 key heads == value heads (no repeat_interleave, design §3.6)
#define GDN_W MT_GDN_Z_OUT           // 2048: q, k, v, z, o and y are all this wide
#define GDN_QKV_W MT_GDN_QKV_OUT     // 6144: in_proj_qkv's width and the conv's channel count

static int bad(const char *fn, const char *why) {
  fprintf(stderr, "layer_gdn: %s: %s\n", fn, why);
  return JT_LAYER_FAIL;
}

// The forward's buffers, in design §3.6's order. `save` picks the arena half (see the header comment): a saved buffer
// comes from the front in JT_FWD_SAVE mode, everything else from the back as a temporary. The widths are floats per
// packed token, or per (token, head) for the norms' r arrays.
typedef struct {
  float *a, *ra;             // RMSNorm(x_in), and its r                       [1024], [1]
  float *uqkv, *uz, *uout;   // the adapted linears' u                          [16] each
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

// ---- the save layout, one list for both halves -----------------------------------------------------------------------
// The forward writes these buffers and the backward reads them, so their order and widths live in ONE place: `walk` steps
// through the layout and both directions call it (the forward from the front of the arena's save half, the backward from
// the arena's base, read-only). A permutation of this list is the one change that would silently pair a forward buffer
// with the wrong backward one, which is why it is a list and not two.
static void walk(float *base, size_t *off, int n, GdnBufs *f) {
  const size_t kh = GDN_HEADS, kw = GDN_W, kq = GDN_QKV_W;
  const struct {
    float **slot;
    size_t width;
  } fields[] = {
      {&f->a, MT_HIDDEN},    {&f->ra, 1},          {&f->uqkv, LORA_RANK}, {&f->uz, LORA_RANK},
      {&f->uout, LORA_RANK}, {&f->aa, kh},         {&f->beta, kh},        {&f->g, kh},
      {&f->pre, kq},         {&f->ck, 3 * kw},     {&f->qkn, 2 * kw},     {&f->rq, kh},
      {&f->rk, kh},          {&f->z, kw},          {&f->y, kw},           {&f->o, kw},
      {&f->ro, kh},
  };
  for (size_t i = 0; i < sizeof fields / sizeof fields[0]; i++) {
    if (base != NULL) *fields[i].slot = base + *off; // base == NULL: measure the layout, assign nothing
    *off += fields[i].width * (size_t)n;
  }
}

// The saved words for a batch of n tokens: what the forward takes in SAVE mode, and what the backward checks against the
// save cursor (a bwd that finds fewer has been called without a JT_FWD_SAVE forward of the same n).
static size_t saves_words(int n) {
  GdnBufs f;
  size_t off = 0;
  walk(NULL, &off, n, &f);
  return off;
}

// One block of `words` floats from the half `save` picks, so the bump cursors move once per call.
static float *take_words(JtCtx *c, int save, size_t words) {
  return save ? jt_save_alloc(c, words) : jt_work_alloc(c, words);
}

static void bufs_get(JtCtx *c, int save, int n, GdnBufs *f) {
  size_t off = 0;
  walk(take_words(c, save, saves_words(n)), &off, n, f);
  f->bb = take(c, 0, GDN_HEADS, n);  // design §4.8's dbb comes from beta, so bb itself is not saved
  f->qkv = take(c, 0, GDN_QKV_W, n); // the conv reads it and nothing upstream does again
  f->cv = take(c, 0, GDN_QKV_W, n);  // silu(pre): pre is the saved form (conv_bwd's derivative)
  f->out = take(c, 0, MT_HIDDEN, n);
}

// The backward's read of the save half: the same layout, walked from the arena's base without moving any cursor. The
// mixer's saves start at the base because layer_fwd() resets the save cursor in front of its SAVE pass (src/layer.h) and
// the mixer runs before the MLP.
static int saves_replay(JtCtx *c, int n, GdnBufs *f) {
  const size_t words = saves_words(n);
  if (words > c->save_at) return bad("layer_gdn_mixer_bwd", "the save half holds fewer floats than the mixer saves; did "
                                                             "layer_gdn_mixer_fwd run in JT_FWD_SAVE mode for this n?");
  size_t off = 0;
  walk(c->arena, &off, n, f);
  return JT_LAYER_OK;
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
  if (c == NULL || lw == NULL || x_in == NULL || x_out == NULL || b == NULL) return bad("layer_gdn_mixer_fwd", "null argument");
  if (n != b->n) return bad("layer_gdn_mixer_fwd", "n does not match the batch's token count");
  if (mode != JT_FWD_ONLY && mode != JT_FWD_SAVE)
    return bad("layer_gdn_mixer_fwd", "mode is neither JT_FWD_ONLY nor JT_FWD_SAVE");
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

  // The recurrence, one grid over the packed rows (state restarts at each row's first token, design §7/§8.3). The plan is
  // plan M8.2's chunked (WY-form) path: BLAS handle plus the backward's `sp` buffer as its arena, which gdn.h's
  // gdn_chunk_arena_floats() shows is four times bigger than the chunked path needs (it is the smallest scratch the
  // layer already owns). JT_GDN_CHUNKED=1 in the environment selects it (the default is the sequential kernels: gdn.h has
  // the measured reason); the two agree to rounding (1e-6 relative).
  gdn_chunk_plan plan;
  plan.blas = c->cublas;
  plan.arena = c->gdn_work.sp;
  plan.arena_n = gdn_bwd_sp_floats((size_t)b->rows, GDN_HEADS);
  gdn_fwd(f.qkn, f.qkn + n2, v, f.g, f.beta, b->off_dev, b->rows, GDN_HEADS, c->gdn_nseg,
          save ? c->gdn_ckpt : NULL, f.o, &plan, c->stream);

  // y = GatedRMSNorm(o; gdn_norm.weight, gate = z): y = w (x r) silu(z), per head, d = 128 (design §3.3/§4.4).
  gated_rmsnorm_fwd(f.o, lw->gdn_norm, f.z, f.y, f.ro, nt, MT_GDN_HEAD_DIM, 0, c->stream);

  // out = out_proj(y), then x_out = x_in + out (design §4.6). x_out MAY alias x_in and linear_fwd overwrites, so the sum
  // is formed in x_out only after the copy it needs. u_out is the backward's (out_proj is adapted, design §2.1).
  rc = linear_fwd(c, &lw->out_proj, f.y, n, f.out, save ? f.uout : NULL);
  if (rc == JT_LAYER_OK) {
    if (x_out != x_in)
      CK(cudaMemcpyAsync(x_out, x_in, (size_t)n * MT_HIDDEN * sizeof(float), cudaMemcpyDeviceToDevice, c->stream));
    jt_add_inplace(x_out, f.out, (size_t)n * MT_HIDDEN, c->stream);
  }
  jt_work_restore(c, mark);
  return rc;
}

// The backward's buffers, in the order design §4.8 needs them. All are work temporaries (the forward's saves are read
// where they lie), per packed token, and given back before this function returns.
typedef struct {
  float *dy;                     // dL/d(y): out_proj's dy, then the gated norm's dy            [1024]
  float *dout, *dz;              // dL/d(o) (gdn_bwd's dout) and dL/d(z)                        [2048] each
  float *dqh, *dch;              // dq|dk from gdn_bwd, then dq|dk|dv after the two l2norms    [2x2048], [3x2048]
  float *dcv, *dqkv;             // dL/d(cv) token-major (conv_bwd's dout), then dL/d(qkv)      [6144] each
  float *dg, *dbeta, *dbb, *daa; // the recurrence's and the gates' cotangents                   [16] each
  float *da;                     // dL/d(a): the four projections summed into                   [1024]
  float *dnorm;                  // the input norm's dx, added into dres                         [1024]
} GdnBwd;

static void bwd_get(JtCtx *c, int n, GdnBwd *g) {
  const size_t kw = GDN_W, kq = GDN_QKV_W, kh = GDN_HEADS;
  g->dy = take(c, 0, MT_HIDDEN, n);
  g->dout = take(c, 0, kw, n);
  g->dz = take(c, 0, kw, n);
  g->dqh = take(c, 0, 2 * kw, n);
  g->dch = take(c, 0, 3 * kw, n);
  g->dcv = take(c, 0, kq, n);
  g->dqkv = take(c, 0, kq, n);
  g->dg = take(c, 0, kh, n);
  g->dbeta = take(c, 0, kh, n);
  g->dbb = take(c, 0, kh, n);
  g->daa = take(c, 0, kh, n);
  g->da = take(c, 0, MT_HIDDEN, n);
  g->dnorm = take(c, 0, MT_HIDDEN, n);
}

// The four input projections' backward, design §4.8's last line: dqkv, dz, dbb and daa through in_proj_qkv, in_proj_z,
// in_proj_b and in_proj_a, summed into da. The first writes da and the other three add to it (linear_bwd's
// accumulate_dx = 1, design §4.6); in_proj_a / in_proj_b are frozen, so they write no parameter gradient and need no u.
static int project_bwd(JtCtx *c, const LayerWeights *lw, const GdnBufs *f, const GdnBwd *g, int n, float *da) {
  int rc = linear_bwd(c, &lw->in_proj_qkv, f->a, f->uqkv, g->dqkv, n, da, 0);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->in_proj_z, f->a, f->uz, g->dz, n, da, 1);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->in_proj_b, f->a, NULL, g->dbb, n, da, 1);
  if (rc == JT_LAYER_OK) rc = linear_bwd(c, &lw->in_proj_a, f->a, NULL, g->daa, n, da, 1);
  return rc;
}

int layer_gdn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  if (c == NULL || lw == NULL || x_in == NULL || dres == NULL || b == NULL)
    return bad("layer_gdn_mixer_bwd", "null argument");
  if (n != b->n) return bad("layer_gdn_mixer_bwd", "n does not match the batch's token count");
  if (n < 0) return bad("layer_gdn_mixer_bwd", "negative n");
  if (n == 0) return JT_LAYER_OK;
  if (c->model == NULL || c->lora == NULL || c->lora->grad == NULL)
    return bad("layer_gdn_mixer_bwd", "no model or gradient buffer");
  if (lw->input_norm == NULL || lw->conv1d == NULL || lw->gdn_norm == NULL)
    return bad("layer_gdn_mixer_bwd", "a frozen weight of the mixer is missing");

  const int nt = n * GDN_HEADS; // one (token, head) pair per row of every per-head norm (design §8.1)
  const size_t n2 = (size_t)n * GDN_W;
  const size_t mark = jt_work_mark(c);
  GdnBufs f;
  GdnBwd g;
  int rc = saves_replay(c, n, &f); // the forward's saves, read-only (design §6.2's recompute: layer_fwd(SAVE) ran first)
  if (rc != JT_LAYER_OK) {
    jt_work_restore(c, mark);
    return rc;
  }
  bwd_get(c, n, &g);

  // out: x_out = x_in + out_proj(y). dres is dL/d(x_out) = dL/d(out), so it is out_proj's dy and the linear's dx is the
  // gated norm's dy. Its dA/dB go into the flat gradient buffer (design §4.2).
  rc = linear_bwd(c, &lw->out_proj, f.y, f.uout, dres, n, g.dy, 0);

  // y = GatedRMSNorm(o; gdn_norm.weight, gate = z) -> do and dz (design §4.4), per (token, head), w NOT 1 + w.
  if (rc == JT_LAYER_OK)
    gated_rmsnorm_bwd(f.o, lw->gdn_norm, f.z, g.dy, f.ro, g.dout, g.dz, nt, MT_GDN_HEAD_DIM, 0, c->stream);

  // The recurrence (design §4.8's reverse loop, kernels/gdn.h): replayed from the forward's checkpoints, so it needs the
  // JT_FWD_SAVE pass, not a JT_FWD_ONLY one. dv has no tile reduction; dq, dk and the two gate gradients do, inside the
  // call. dq|dk go to the pre-l2norm (post-silu) gradient through l2norm_bwd below; dv is already one.
  if (rc == JT_LAYER_OK) {
    gdn_chunk_plan plan; // the forward's (gdn.h: the handle and the arena the chunked path needs)
    plan.blas = c->cublas;
    plan.arena = c->gdn_work.sp;
    plan.arena_n = gdn_bwd_sp_floats((size_t)b->rows, GDN_HEADS);
    gdn_bwd(f.qkn, f.qkn + n2, f.ck + 2 * n2, f.g, f.beta, g.dout, b->off_dev, b->rows, GDN_HEADS, n, c->gdn_nseg,
            c->gdn_ckpt, &c->gdn_work, g.dch + 2 * n2, g.dqh, g.dqh + n2, g.dg, g.dbeta, &plan, c->stream);
  }

  // q = l2norm(qc) / sqrt(128) and k = l2norm(kc): the two per-head norms' backward (design §4.8's second block). The
  // post_scale is the same one the forward passed (kernels/norm.h); dv needs no norm, so it is not touched.
  if (rc == JT_LAYER_OK) {
    l2norm_bwd(f.ck, g.dqh, f.rq, 1.0f / sqrtf((float)MT_GDN_HEAD_DIM), g.dch, nt, MT_GDN_HEAD_DIM, c->stream);
    l2norm_bwd(f.ck + n2, g.dqh + n2, f.rk, 1.0f, g.dch + n2, nt, MT_GDN_HEAD_DIM, c->stream);
    // dc = concat(dqc, dkc, dv) [n][6144]: the inverse of the forward's three-slice split (design §4.8's dc).
    jt_from_head_major(g.dch, g.dcv, n, 3, GDN_W, c->stream);
  }

  // cv = silu(pre), pre = conv(qkv): conv_bwd takes dL/d(cv) and applies silu's derivative at the saved pre itself, then
  // runs the transposed causal conv into dqkv (design §4.8; taps stay inside the token's row, design §7).
  if (rc == JT_LAYER_OK)
    conv_bwd(g.dcv, f.pre, lw->conv1d, b->row_of, b->off_dev, g.dqkv, n, GDN_QKV_W, c->stream);

  // beta = sigmoid(bb), g = -exp(A_log) softplus(aa + dt_bias): dbb and daa from dbeta and dg, with beta and aa from the
  // saves (design §4.8's last two lines before the projections).
  if (rc == JT_LAYER_OK)
    k_gdn_gates_bwd(g.dbeta, g.dg, f.beta, f.aa, lw->a_log, lw->dt_bias, g.dbb, g.daa, nt, GDN_HEADS, c->stream);

  if (rc == JT_LAYER_OK) rc = project_bwd(c, lw, &f, &g, n, g.da);

  // a = RMSNorm(x_in; input_layernorm): the last step is the zero-centred norm's backward, accumulated into the running
  // residual gradient (design §4.3, §4.6) — dres already holds dL/d(x_out)'s skip-connection part.
  if (rc == JT_LAYER_OK) {
    rmsnorm_bwd(x_in, lw->input_norm, g.da, f.ra, g.dnorm, n, MT_HIDDEN, 1, c->stream);
    jt_add_inplace(dres, g.dnorm, (size_t)n * MT_HIDDEN, c->stream);
  }
  jt_work_restore(c, mark);
  return rc;
}
