// src/head.c — the decision head: final norm, the last token of each row, the readout GEMM and the soft CE (design §4.1,
// §13). STUB: plan J18 freezes the interface (src/layer.h); the head's owner replaces this file, and neither the header
// nor src/model.c changes when it does.
//
// The forward is design §4.1 in order, with the call each step uses:
//
//   final = RMSNorm(x; norm)                                  rmsnorm_fwd, d = 1024, zero-centred, r saved
//   z     = final[off[b+1]-1] for each row b                  a gather of B rows (jt_fill / cudaMemcpyAsync per row, or a
//                                                             small copy loop): design §7's "the head reads token
//                                                             off[b+1]-1 for row b"
//   logits= z R^T, first n_opts readout rows                  gemm(0,1) B x n_opts x 1024 against
//                                                             ctx->lora->param + ctx->lora->readout_off
//   logits[b][j >= live[b]] = -1e9                            train.py's masked_fill, exactly
//   ce_head(logits, n_opts, targets, wrow, wgroup, B, ...)    k_head.h: per-row loss, dlogits and probs
//
// and the backward is the second half of design §4.1: dR += dlogits (x) z into ctx->lora->grad + readout_off, dz = R^T
// dlogits, dhN[off[b+1]-1] = dz and dhN[t] = 0 elsewhere, then the final norm's rmsnorm_bwd over all n tokens writes
// dres. Only the live option columns of the readout get a gradient — the dead ones are -1e9, so their dlogits is exactly
// 0, which is what dump_ref's grad.readout rows 5..254 show (notes/J14.md).
//
// head_fwd reads the loss back to the host once (*loss): the value is train.py's `(per * wt).sum() / gw`, i.e. the sum of
// ce_head's per-row losses, and a training step wants it on the host for its log/schedule. That makes head_fwd the one
// layer call that synchronises; pass loss == NULL to keep the step asynchronous.
#include "layer.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "head: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

int head_fwd(JtCtx *c, const float *x, int n, const JtBatch *b, const HeadBatch *hb, float *final, float *logits,
             float *loss, int mode) {
  (void)c, (void)x, (void)n, (void)b, (void)hb, (void)final, (void)logits, (void)loss, (void)mode;
  return not_implemented("head_fwd");
}

int head_bwd(JtCtx *c, const float *x, int n, const JtBatch *b, float *dres) {
  (void)c, (void)x, (void)n, (void)b, (void)dres;
  return not_implemented("head_bwd");
}
