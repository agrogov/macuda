// src/linear.c — the linear layer with LoRA: the only code that touches a frozen weight's memory or a LoRA factor
// (design §13, §4.2, §5).
//
//   linear_fwd   DONE (plan M4.3 item 1, J19): design §4.2's `y = x W^T + s (x A^T) B^T` as the three GEMMs design §5's
//                table lists, u = x A^T kept when the caller passes a save pointer.
//   linear_bwd   STILL THE J18 STUB: e = s (dy B) gemm(0,0) alpha = s N x rank x out; dx = dy W gemm(0,0) N x in x out
//                and dx += e A gemm(0,0) beta = 1 (or both into dx when accumulate_dx); dB += s dy^T u gemm(1,0)
//                out x rank x N; dA += e^T x gemm(1,0) rank x in x N. dA and dB land in ctx->lora->grad + a_off / b_off.
//
// The CPU oracle of the same order and shapes is ref/linear.c (ref_linear_fwd / ref_linear_bwd); test/gradcheck's `lora`
// group already passes against it, so the backward is only the GEMM sequence above.
//
// Two things the forward does that the design does not spell out, because they are consequences of its memory rule (no
// cudaMalloc inside a layer call, design §6.2) and of the frozen struct:
//
//  * u has to be materialised before it can be multiplied by B, so when the caller does not want it kept (u_save NULL,
//    i.e. a JT_FWD_ONLY pass) the temporary comes from the arena's work half and is given back before returning. It is
//    n * rank floats (67 K at N = 4200), nothing next to the arena.
//  * struct Linear carries no rank: every LoRA pair in this model is rank LORA_RANK (design §2.1, src/lora.h), which is
//    also what the flat buffer's table is built with.
#include "gemm.h"
#include "layer.h"
#include "linear.h"
#include "util.h"

#include <stdio.h>

static int not_implemented(const char *what) {
  fprintf(stderr, "linear: %s: not implemented (plan M4.3)\n", what);
  return JT_LAYER_NOT_IMPL;
}

// A bad call names the linear when it can: the stem is the adapter key (design §9.2), so a failure points at the tensor.
static int bad_args(const Linear *lin, const char *why) {
  fprintf(stderr, "linear: linear_fwd: %s (%s)\n", why, (lin != NULL && lin->stem != NULL) ? lin->stem : "?");
  return JT_LAYER_FAIL;
}

int linear_fwd(JtCtx *ctx, const Linear *lin, const float *x, int n, float *y, float *u_save) {
  if (ctx == NULL || lin == NULL || x == NULL || y == NULL) return bad_args(lin, "null argument");
  if (lin->in <= 0 || lin->out <= 0) return bad_args(lin, "in/out must be positive");
  if (n < 0) return bad_args(lin, "negative n");
  if (y == x) return bad_args(lin, "y may not alias x"); // design §13's Linear contract (src/linear.h)
  if (lin->lora && (ctx->lora == NULL || ctx->lora->param == NULL)) return bad_args(lin, "lora linear without a buffer");
  if (n == 0) return JT_LAYER_OK; // nothing to compute; the counters below would be fine, the GEMMs would not

  const int rank = LORA_RANK;
  if (lin->lora) { // the two factors are slices of the one flat parameter buffer (design §8.1)
    const size_t need_a = (size_t)rank * (size_t)lin->in, need_b = (size_t)lin->out * (size_t)rank;
    if (lin->a_off + need_a > ctx->lora->n_param || lin->b_off + need_b > ctx->lora->n_param)
      return bad_args(lin, "LoRA offsets outside the flat parameter buffer");
  }

  const size_t mark = jt_work_mark(ctx);
  // y = x W^T. The frozen weight is the same pointer the backward uses (design §5: no W^T is ever stored).
  gemm(ctx->cublas, 0, 1, n, lin->out, lin->in, 1.0f, x, lin->W, 0.0f, y);
  if (lin->lora) {
    const float *A = ctx->lora->param + lin->a_off; // [rank][in]
    const float *B = ctx->lora->param + lin->b_off; // [out][rank]
    float *u = (u_save != NULL) ? u_save : jt_work_alloc(ctx, (size_t)n * (size_t)rank);
    gemm(ctx->cublas, 0, 1, n, rank, lin->in, 1.0f, x, A, 0.0f, u);                              // u = x A^T
    gemm(ctx->cublas, 0, 1, n, lin->out, rank, MT_LORA_SCALE, u, B, 1.0f, y);                     // y += s u B^T
  }
  jt_work_restore(ctx, mark); // the temporary is gone; the GEMMs above are ordered before anything reuses those words
  return JT_LAYER_OK;
}

int linear_bwd(JtCtx *ctx, const Linear *lin, const float *x, const float *u, const float *dy, int n, float *dx,
               int accumulate_dx) {
  (void)ctx, (void)lin, (void)x, (void)u, (void)dy, (void)n, (void)dx, (void)accumulate_dx;
  return not_implemented("linear_bwd");
}
