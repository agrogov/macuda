// src/layer_gdn.c — the GDN mixer: design §3.6 forward, §4.8 backward, on the kernels of plan M2 and M3.
// STUB: plan J18 freezes the interface (src/layer.h); the GDN layer's owner replaces this file, and neither the header nor
// src/model.c changes when it does (design §13).
//
// The forward is design §3.6 in order, with the call each step uses (all of them already written and tested):
//
//   a   = RMSNorm(x_in; input_layernorm)                          rmsnorm_fwd, r saved
//   qkv = in_proj_qkv(a)  z = in_proj_z(a)  bb = in_proj_b(a)  aa = in_proj_a(a)
//                                                                 linear_fwd x4 (a and b are frozen: lora == 0)
//   pre = conv(qkv); c = silu(pre)                                conv_fwd  (row-local taps, design §7)
//   q = c[:,0:2048], k = c[:,2048:4096], v = c[:,4096:6144]       views into c — no copy; l2norm takes the row stride
//   q = l2norm(q, post_scale 1/sqrt(128)); k = l2norm(k, 1)       l2norm_fwd
//   beta, g = sigmoid(bb), -exp(A_log)*softplus(aa+dt_bias)       k_gdn_gates_fwd(lw->a_log, lw->dt_bias)
//   o   = gdn(q, k, v, g, beta) over the packed rows               gdn_fwd(q, k, v, g, beta, b->off_dev, b->rows, 16,
//                                                                  c->gdn_nseg, ckpt, o, c->stream)
//   y   = gated_rmsnorm(o; gdn_norm.weight, gate = z)             gated_rmsnorm_fwd (NOT zero-centred)
//   out = out_proj(y)                                             linear_fwd
//   x_out = x_in + out                                            the residual, design §4.6
//
// ckpt is c->gdn_ckpt in JT_FWD_SAVE mode and NULL in JT_FWD_ONLY (the backward replays from the recompute pass's
// checkpoint, design §8.3). c->gdn_work is the backward's scratch (gdn_bwd's `work`; the buffers are shared by all 18
// GDN layers and sized once by jt_ctx_alloc).
//
// The backward is design §4.8 in its stated order: out_proj's linear_bwd, gated_rmsnorm_bwd (do and dz), the recurrence's
// gdn_bwd replayed from c->gdn_ckpt, then the l2norm backwards, the transposed conv (conv_bwd), silu's, the five input
// gradients, and the five linears' linear_bwd. The order of the recurrence's own steps is what §4.8's "the order matters"
// paragraph fixes: dk's first term and ddelta use dS before the k (x) dkv term is added, and dg uses dS after it.
#include "layer.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "layer_gdn: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

int layer_gdn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                        int mode) {
  (void)c, (void)lw, (void)x_in, (void)n, (void)b, (void)x_out, (void)mode;
  return not_implemented("layer_gdn_mixer_fwd");
}

int layer_gdn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  (void)c, (void)lw, (void)x_in, (void)dres, (void)n, (void)b;
  return not_implemented("layer_gdn_mixer_bwd");
}
