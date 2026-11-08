// ref/linear.h — the CPU oracle for a linear layer with LoRA (plan M2.6/M5, design §4.2). This is the op every adapted
// projection of the model is: the frozen checkpoint weight W plus a rank-16 pair A/B, D§2.1's `y = W x + scale * B (A x)`.
//
// Design §4.2, word for word (one row `x` shown; the reference works on N rows at once):
//
//   forward   y  = x W^T + s * (x A^T) B^T,  s = alpha/rank = 2.0, u = x A^T saved [N, rank]
//   backward  e  = s * (dy B)                   [N, rank]
//             dx = dy W + e A                   [N, in]
//             dB += s * dy^T u                  [out, rank]
//             dA += e^T x                       [rank, in]
//
// Layouts are the checkpoint's/torch's, all row-major: x [N,in], W [out,in], A [rank,in], B [out,rank], u [N,rank],
// y [N,out], dy [N,out], dx [N,in], dA [rank,in], dB [out,rank]. `s * (dy B)` is written `e` because the design writes it
// once and reuses it; that is also the order the arithmetic is done in.
//
// The gradient buffers accumulate (`+=`): a layer's dA/dB collect the contributions of every row and every accumulation
// step into the flat parameter buffer's gradient (design §8.1). dx is written, never accumulated: the caller adds it to
// the residual gradient (design §4.6). dA/dB may be NULL (a frozen linear: A and B NULL, see MtLinear.lora in
// src/model_table.h), and so may dx when only the parameter gradients are wanted.
//
// The GPU has no kernel of this name on purpose: in the trainer these are cuBLAS calls (design §5's gemm()), strung
// together by the linear_fwd/linear_bwd pair design §13 puts in src/model.c. This reference fixes the exact order and
// shape contract those calls must reproduce, and test/grad_lora.c's `gradcheck lora` is the plan's M2.0 gate for the A
// and B gradients (dx and the W-only path too, since they are part of the same formula).
#ifndef JT_REF_LINEAR_H
#define JT_REF_LINEAR_H

#include <stddef.h>

#include "real.h"

#ifdef __cplusplus
extern "C" {
#endif

// y[r][o] = sum_i x[r][i] W[o][i]  +  scale * sum_k u[r][k] B[o][k], with u[r][k] = sum_i x[r][i] A[k][i].
// `u` is written when it is not NULL (it is the forward's saved activation, D§4.2); pass NULL when only y is needed.
// A and B may both be NULL (or rank <= 0): a plain frozen linear, y = x W^T.
void ref_linear_fwd(const real *x, const real *W, const real *A, const real *B, real *y, real *u, int n, int in,
                    int out, int rank, real scale);

// The backward of ref_linear_fwd (design §4.2). `u` is the forward's saved u[N,rank]. dx is written; dA and dB are
// accumulated (`+=`) and may be NULL; A and B may be NULL, in which case only dx = dy W is computed.
void ref_linear_bwd(const real *x, const real *W, const real *u, const real *dy, const real *A, const real *B, real *dx,
                    real *dA, real *dB, int n, int in, int out, int rank, real scale);

#ifdef __cplusplus
}
#endif

#endif // JT_REF_LINEAR_H
