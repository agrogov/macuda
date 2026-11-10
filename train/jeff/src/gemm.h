// src/gemm.h — the one row-major GEMM wrapper every dense op goes through (design §5, plan M1.1).
//
// Both entry points compute the row-major C[M,N] = alpha * op(A) * op(B) + beta * C on top of cuBLAS, which is
// column-major, by issuing the transposed problem once: the row-major C[M,N] = A[M,K] B[K,N] is the column-major
// C^T[N,M] = B^T[N,K] A^T[K,M], so the cuBLAS "A" argument is our B and the cuBLAS "B" argument is our A (design §5).
//
// ta/tb say how the argument is *stored*: 0 = as stored (A is [M,K], B is [K,N]), 1 = transposed (A is [K,M], B is
// [N,K]). The leading dimension is always the stored row length. alpha and beta are values here; inside they are handed
// to cuBLAS as pointers, which is the default pointer mode.
#ifndef JT_GEMM_H
#define JT_GEMM_H

#include <cublas_v2.h>

// Decides the compute mode once (CUBLAS_COMPUTE_32F for JT_EXACT, else CUBLAS_COMPUTE_32F_FAST_TF32) and prints it:
// "gemm: exact fp32" or "gemm: tf32". Called from start-up; gemm()/gemm_batched() call it themselves, so the first
// user of either cannot forget it and the choice is still made exactly once.
void gemm_init(void);

// 1 in exact mode (JT_EXACT at compile time or in the environment), 0 for TF32. Tests pick their gate with it.
int gemm_is_exact(void);

// Row-major. C[M,N] = alpha * op(A) op(B) + beta * C.
void gemm(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C);

// The same, batch GEMMs side by side with element strides: matrix b of A lives at A + b*sA, of B at B + b*sB, of C at
// C + b*sC. The strides follow their own matrix, so the cuBLAS "A" argument (our B) gets sB (design §5, plan M1.1).
// Used for the attention heads, where one call covers the 8 query heads of one row.
void gemm_batched(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
                  const float *A, long sA, const float *B, long sB, float beta, float *C, long sC, int batch);

// ---- bf16 operands, fp32 output (plan M8.1b, design §6.1's alignment rules) -------------------------------------------
// The same row-major GEMM with A and B stored as bf16 and C fp32, through cublasGemmEx's CUDA_R_16BF operands and
// CUBLAS_COMPUTE_32F (fp32 accumulate). cuBLAS runs a GEMM only when both operands are one type, so a bf16 weight can
// only meet a bf16 activation — that is why the frozen-weight path casts the activation (kernels/misc.h's
// jt_f32_to_bf16_n) instead of leaving it fp32.
//
// Design §6.1's alignment rule, as libtinycublas states it (cublas.c:145): the contiguous extent and the leading
// dimension of each operand must be multiples of 8 elements and both bases 16-byte aligned. A shape outside that rule
// has no bf16 kernel on the shim (NOT_SUPPORTED) and is not the fast path on real cuBLAS either, so gemm_bf16_ok()
// answers the question first and the caller keeps the fp32 path when it says no. `A` is [M,K] / `B` is [K,N] (or their
// transposes, as ta/tb say — the same convention as gemm()), both bf16 as `unsigned short` (the tree's bf16 type).
int gemm_bf16_ok(int ta, int tb, int M, int N, int K, const void *A, const void *B);

// The bf16 GEMM itself. Precondition: gemm_bf16_ok() returned 1 for these shapes and pointers; it re-checks and stops
// with the numbers instead of handing cuBLAS a shape its bf16 kernels do not serve. Only gemm()'s non-batched form is
// mirrored here: the two frozen-weight GEMMs are the ones M8.1b changes, and neither is batched.
void gemm_bf16(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha, const unsigned short *A,
               const unsigned short *B, float beta, float *C);

#endif // JT_GEMM_H
