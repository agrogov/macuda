// src/linear.c — the linear layer with LoRA: the only code that touches a frozen weight's memory or a LoRA factor
// (design §13, §4.2, §5).
//
//   linear_fwd   DONE (plan M4.3 item 1, J19): design §4.2's `y = x W^T + s (x A^T) B^T` as the three GEMMs design §5's
//                table lists, u = x A^T kept when the caller passes a save pointer.
//   linear_bwd   DONE (plan M5.1 item 2, J25): design §4.2's backward as the five GEMMs design §5's table lists —
//                e = s (dy B) gemm(0,0) alpha = s N x rank x out; dx = dy W gemm(0,0) N x in x out and dx += e A
//                gemm(0,0) beta = 1 (both with beta = 1 when accumulate_dx); dB += s dy^T u gemm(1,0) out x rank x N;
//                dA += e^T x gemm(1,0) rank x in x N, with e built once and reused (D§4.2's own optimisation). dA and dB
//                land in ctx->lora->grad + a_off / b_off.
//
// The CPU oracle of the same order and shapes is ref/linear.c (ref_linear_fwd / ref_linear_bwd); test/gradcheck's `lora`
// group already passes against it, so the backward is only the GEMM sequence above. test/test_linear.c is the GPU gate
// for it: a central-difference check of dx, dA and dB on a [37,64] -> [48] linear in exact mode (plan M5.1 item 2), and
// layer 23's three MLP LoRA pairs against dump_ref's `grad.{stem}.lora_A/B`.
//
// Two things the forward does that the design does not spell out, because they are consequences of its memory rule (no
// cudaMalloc inside a layer call, design §6.2) and of the frozen struct:
//
//  * u has to be materialised before it can be multiplied by B, so when the caller does not want it kept (u_save NULL,
//    i.e. a JT_FWD_ONLY pass) the temporary comes from the arena's work half and is given back before returning. It is
//    n * rank floats (67 K at N = 4200), nothing next to the arena.
//  * struct Linear carries no rank: every LoRA pair in this model is rank LORA_RANK (design §2.1, src/lora.h), which is
//    also what the flat buffer's table is built with.
//  * with JT_BF16_WEIGHTS=1 (plan M8.1b) a frozen weight's GEMMs are the only place the bf16 storage shows: the
//    activation operand is cast into the same work half and gemm_bf16 runs both operands bf16 (src/linear.c's
//    weight_gemm). LoRA A and B, u and e stay fp32, so those GEMMs do not change at all.
#include "gemm.h"
#include "layer.h"
#include "linear.h"
#include "misc.h" // kernels/misc.h: jt_f32_to_bf16_n (the bf16 activation operand) and jt_bf16_to_f32_n (the fallback)
#include "util.h"

#include <stdint.h>
#include <stdio.h>

// A bad call names the function and the linear: the stem is the adapter key (design §9.2), so a failure points at the
// tensor as well as at the line of the contract it broke.
static int bad_args(const Linear *lin, const char *fn, const char *why) {
  fprintf(stderr, "linear: %s: %s (%s)\n", fn, why, (lin != NULL && lin->stem != NULL) ? lin->stem : "?");
  return JT_LAYER_FAIL;
}

// The flat buffer's bounds for a LoRA linear (design §8.1): both factors must lie inside param, and the same words of
// grad must exist because the backward writes them. Returns 0 when the offsets are usable.
static int lora_offsets_ok(const JtCtx *ctx, const Linear *lin) {
  const size_t rank = LORA_RANK;
  const size_t need_a = rank * (size_t)lin->in, need_b = (size_t)lin->out * rank;
  return lin->a_off + need_a <= ctx->lora->n_param && lin->b_off + need_b <= ctx->lora->n_param;
}

// One GEMM of a frozen weight (plan M8.1b): the two calls design §5 gives a weight, `y = x W^T` and `dx = dy W`.
//
// With `lin->Wb` set the weight is bf16 on the device and stays that way: the activation operand (A here, always the
// matrix multiplied from the left) is cast into the arena's work half with J46's jt_f32_to_bf16_n, and gemm_bf16 runs
// both operands bf16 with fp32 output. The cast is per call, i.e. per GEMM, not per weight: 2 x n x in bf16 bytes for the
// forward instead of a widened 2 x out x in fp32 weight, which is the whole trade (design §6.1, notes/J36.md).
//
// Where design §6.1's alignment rule does not hold, gemm_bf16_ok says so and the call falls back to the fp32 path: the
// weight is widened into the same work half (jt_bf16_to_f32_n) and gemm() runs exactly as it always did. No shape of
// this model takes that branch — every width here is a multiple of 8 — it exists so a shape outside the rule stays
// correct rather than failing.
static void weight_gemm(JtCtx *c, const Linear *lin, int tb, int M, int N, int K, float alpha, const float *A,
                        float beta, float *C) {
  if (lin->Wb == NULL) { // the fp32 path, unchanged: the same call with the same arguments as before M8.1b
    gemm(c->cublas, 0, tb, M, N, K, alpha, A, lin->W, beta, C);
    return;
  }
  const size_t mark = jt_work_mark(c);
  const size_t na = (size_t)M * (size_t)K; // the activation's elements
  float *raw = jt_work_alloc(c, na / 2 + 8); // na bf16 = na/2 floats, + 8 for the alignment shift below
  unsigned short *ab = (unsigned short *)(((uintptr_t)raw + (uintptr_t)15) & ~(uintptr_t)15);
  if (gemm_bf16_ok(0, tb, M, N, K, ab, lin->Wb)) {
    jt_f32_to_bf16_n(A, ab, na, c->stream); // same stream as the GEMM: the cast is ordered before it
    gemm_bf16(c->cublas, 0, tb, M, N, K, alpha, ab, lin->Wb, beta, C);
    jt_work_restore(c, mark);
    return;
  }
  jt_work_restore(c, mark);
  float *wf = jt_work_alloc(c, (size_t)N * (size_t)K); // the weight as fp32: W is [out][in] in both GEMMs' terms
  jt_bf16_to_f32_n(lin->Wb, wf, (size_t)N * (size_t)K, c->stream);
  gemm(c->cublas, 0, tb, M, N, K, alpha, A, wf, beta, C);
  jt_work_restore(c, mark);
}

int linear_fwd(JtCtx *ctx, const Linear *lin, const float *x, int n, float *y, float *u_save) {
  if (ctx == NULL || lin == NULL || x == NULL || y == NULL) return bad_args(lin, "linear_fwd", "null argument");
  if (lin->in <= 0 || lin->out <= 0) return bad_args(lin, "linear_fwd", "in/out must be positive");
  if (n < 0) return bad_args(lin, "linear_fwd", "negative n");
  if (y == x) return bad_args(lin, "linear_fwd", "y may not alias x"); // design §13's Linear contract (src/linear.h)
  if (lin->lora && (ctx->lora == NULL || ctx->lora->param == NULL))
    return bad_args(lin, "linear_fwd", "lora linear without a buffer");
  if (lin->lora && !lora_offsets_ok(ctx, lin)) // the two factors are slices of the one flat buffer (design §8.1)
    return bad_args(lin, "linear_fwd", "LoRA offsets outside the flat parameter buffer");
  if (n == 0) return JT_LAYER_OK; // nothing to compute; the counters below would be fine, the GEMMs would not

  const int rank = LORA_RANK;
  const size_t mark = jt_work_mark(ctx);
  // y = x W^T. The frozen weight is the same pointer the backward uses (design §5: no W^T is ever stored), and the
  // bf16 storage (plan M8.1b) is the same weight, cast per call.
  weight_gemm(ctx, lin, 1, n, lin->out, lin->in, 1.0f, x, 0.0f, y);
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
  if (ctx == NULL || lin == NULL || x == NULL || dy == NULL) return bad_args(lin, "linear_bwd", "null argument");
  if (lin->in <= 0 || lin->out <= 0) return bad_args(lin, "linear_bwd", "in/out must be positive");
  if (n < 0) return bad_args(lin, "linear_bwd", "negative n");
  // dy is read by three of the five GEMMs, so a caller that writes dx over it would read a half-updated buffer.
  if (dx == dy) return bad_args(lin, "linear_bwd", "dx may not alias dy");
  if (lin->lora && (ctx->lora == NULL || ctx->lora->param == NULL || ctx->lora->grad == NULL))
    return bad_args(lin, "linear_bwd", "lora linear without a parameter buffer");
  if (lin->lora && u == NULL) // the forward's saved u is an input of two of the GEMMs (design §4.2)
    return bad_args(lin, "linear_bwd", "a LoRA linear needs the forward's saved u");
  if (lin->lora && !lora_offsets_ok(ctx, lin))
    return bad_args(lin, "linear_bwd", "LoRA offsets outside the flat parameter buffer");
  if (n == 0) return JT_LAYER_OK;
  // A frozen linear has no parameter gradient (design §4.2), so with no dx there is nothing at all to compute.
  if (!lin->lora && dx == NULL) return JT_LAYER_OK;

  const int rank = LORA_RANK;
  const size_t mark = jt_work_mark(ctx);
  const float beta = accumulate_dx ? 1.0f : 0.0f; // design §4.6: several paths may add into one residual gradient

  // e = s (dy B) [n, rank], built once and reused by dx and dA exactly as design §4.2 writes it.
  const float *A = NULL, *B = NULL;
  float *e = NULL;
  if (lin->lora) {
    A = ctx->lora->param + lin->a_off; // [rank][in]
    B = ctx->lora->param + lin->b_off; // [out][rank]
    e = jt_work_alloc(ctx, (size_t)n * (size_t)rank);
    gemm(ctx->cublas, 0, 0, n, rank, lin->out, MT_LORA_SCALE, dy, B, 0.0f, e);
  }

  if (dx != NULL) { // dx = dy W (+ e A), design §5's two gemm(0,0) rows. W is the weight the forward used.
    weight_gemm(ctx, lin, 0, n, lin->in, lin->out, 1.0f, dy, beta, dx);
    if (lin->lora) gemm(ctx->cublas, 0, 0, n, lin->in, rank, 1.0f, e, A, 1.0f, dx);
  }

  if (lin->lora) { // the parameter gradients, accumulated into the flat buffer (design §4.2, §8.1)
    float *dA = ctx->lora->grad + lin->a_off; // [rank][in]
    float *dB = ctx->lora->grad + lin->b_off; // [out][rank]
    gemm(ctx->cublas, 1, 0, lin->out, rank, n, MT_LORA_SCALE, dy, u, 1.0f, dB); // dB += s dy^T u
    gemm(ctx->cublas, 1, 0, rank, lin->in, n, 1.0f, e, x, 1.0f, dA);            // dA += e^T x
  }
  jt_work_restore(ctx, mark); // e is gone; the GEMMs that read it are ordered before anything reuses those words
  return JT_LAYER_OK;
}
