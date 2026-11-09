// src/layer_mlp.c — the post-attention norm and the MLP, shared by both layer kinds (design §3.4's last two lines, §4.5).
// STUB: plan J18 freezes the interface (src/layer.h); the MLP's owner replaces this file, and neither the header nor
// src/model.c changes when it does (design §13).
//
// The real implementation is design §4.5, on top of functions that already exist and are tested:
//
//   m    = RMSNorm(x_in; post_attention_layernorm)          rmsnorm_fwd (kernels/norm.h)
//   h1   = gate(m), h2 = up(m)                              linear_fwd (LoRA on both)
//   act  = silu(h1) * h2                                    k_silu_mul_fwd (kernels/act.h)
//   y    = down(act)                                        linear_fwd
//   x_out= x_in + y                                         (the residual, design §4.6)
//
// and its backward is the same file's kernels in reverse: dact = down_bwd, then k_silu_mul_bwd gives dh1 and dh2, then
// gate_bwd and up_bwd accumulate into the flat buffer, dm = both dx's summed, then rmsnorm_bwd adds into dres.
//
// Saves (JT_FWD_SAVE): m, the three u's, h1, h2, act and RMSNorm's r — from the arena's save half, never cudaMalloc.
#include "layer.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "layer_mlp: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

int layer_mlp_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out, int mode) {
  (void)c, (void)lw, (void)x_in, (void)n, (void)b, (void)x_out, (void)mode;
  return not_implemented("layer_mlp_fwd");
}

int layer_mlp_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  (void)c, (void)lw, (void)x_in, (void)dres, (void)n, (void)b;
  return not_implemented("layer_mlp_bwd");
}
