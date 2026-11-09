// src/linear.c — the linear layer with LoRA: the only code that touches a frozen weight's memory or a LoRA factor
// (design §13, §4.2, §5). STUB: plan J18 writes the interface (src/linear.h) and this placeholder so that the layer
// files can be written in parallel against a fixed signature; plan M4.3 item 1 ("`linear_fwd`, saving `u` when a save
// pointer is given") replaces it with the GEMM chain design §5 lays out, and nothing else has to change.
//
// What the real implementation is, so that whoever writes it starts from the design rather than from here:
//
//   linear_fwd   y = x W^T with gemm(0,1) N x out x in; if lora, u = x A^T with gemm(0,1) N x rank x in (written to
//                u_save), then y += s u B^T with gemm(0,1) alpha = s beta = 1 (design §4.2's `y = x W^T + s (x A^T) B^T`).
//   linear_bwd   e = s (dy B) gemm(0,0) alpha = s N x rank x out; dx = dy W gemm(0,0) N x in x out and dx += e A
//                gemm(0,0) beta = 1 (or both into dx when accumulate_dx); dB += s dy^T u gemm(1,0) out x rank x N;
//                dA += e^T x gemm(1,0) rank x in x N. dA and dB land in ctx->lora->grad + a_off / b_off.
//
// The CPU oracle of the same order and shapes is ref/linear.c (ref_linear_fwd / ref_linear_bwd); test/gradcheck's `lora`
// group already passes against it, so the only thing left here is the GEMM sequence.
#include "linear.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "linear: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

int linear_fwd(JtCtx *ctx, const Linear *lin, const float *x, int n, float *y, float *u_save) {
  (void)ctx, (void)lin, (void)x, (void)n, (void)y, (void)u_save;
  return not_implemented("linear_fwd");
}

int linear_bwd(JtCtx *ctx, const Linear *lin, const float *x, const float *u, const float *dy, int n, float *dx,
               int accumulate_dx) {
  (void)ctx, (void)lin, (void)x, (void)u, (void)dy, (void)n, (void)dx, (void)accumulate_dx;
  return not_implemented("linear_bwd");
}
