// src/layer_attn.c — the full-attention mixer: design §3.5 forward, §4.7 backward, on the kernels of plan M2.
// STUB: plan J18 freezes the interface (src/layer.h); the attention layer's owner replaces this file, and neither the
// header nor src/model.c changes when it does (design §13).
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
#include "layer.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "layer_attn: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

int layer_attn_mixer_fwd(JtCtx *c, const LayerWeights *lw, const float *x_in, int n, const JtBatch *b, float *x_out,
                         int mode) {
  (void)c, (void)lw, (void)x_in, (void)n, (void)b, (void)x_out, (void)mode;
  return not_implemented("layer_attn_mixer_fwd");
}

int layer_attn_mixer_bwd(JtCtx *c, const LayerWeights *lw, const float *x_in, float *dres, int n, const JtBatch *b) {
  (void)c, (void)lw, (void)x_in, (void)dres, (void)n, (void)b;
  return not_implemented("layer_attn_mixer_bwd");
}
