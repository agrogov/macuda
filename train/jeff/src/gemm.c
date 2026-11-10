// src/gemm.c — M1.1: the row-major cuBLAS wrapper (design §5). Every dense op in jefftrain is one of these two calls.
//
// Design §5's issue, once: the row-major C[M,N] = alpha op(A) op(B) + beta C is the column-major C^T[N,M] =
// alpha op(B)^T op(A)^T + beta C^T. So m = N, n = M, k = K, the cuBLAS "A" argument is our B (leading dim tb ? K : N,
// op tb ? T : N) and the cuBLAS "B" argument is our A (leading dim ta ? M : K, op ta ? T : N); C's leading dim is its
// row length, N. Getting these the wrong way round is the classic bug this wrapper exists to make impossible, so the
// argument order is written out once here and nowhere else.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "gemm.h"

// The compute mode. libtinycublas accepts both for fp32 A/B (cuda-shim/test/cublas_contract.c): f32 with
// COMPUTE_32F_FAST_TF32 serves on the tensor-core path, COMPUTE_32F on the plain fp32 one, which is what the gradchecks
// need. TINYCUBLAS_TC=0 (the shim's own knob) additionally forces the non-tensor-core path in either mode.
#ifndef JT_EXACT
#define JT_EXACT 0
#endif

static cublasComputeType_t jt_compute;
static int jt_compute_ready;

// Compile-time JT_EXACT wins (a build is either exact or it is not); otherwise the environment decides per run, which
// is how scripts/card-check.sh's `JT_EXACT=1` step and the TF32 step share one build (design §5, plan M1.1).
static int jt_exact(void) {
#if JT_EXACT
  return 1;
#else
  const char *e = getenv("JT_EXACT");
  return e != NULL && strcmp(e, "1") == 0;
#endif
}

void gemm_init(void) {
  int exact;
  if (jt_compute_ready) return;
  exact = jt_exact();
  jt_compute = exact ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
  jt_compute_ready = 1;
  printf("gemm: %s\n", exact ? "exact fp32" : "tf32");
  fflush(stdout);
}

int gemm_is_exact(void) {
  gemm_init();
  return jt_compute == CUBLAS_COMPUTE_32F;
}

static cublasComputeType_t jt_compute_get(void) {
  gemm_init();
  return jt_compute;
}

void gemm(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
          const float *A, const float *B, float beta, float *C) {
  CB(cublasGemmEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                  B, CUDA_R_32F, tb ? K : N, A, CUDA_R_32F, ta ? M : K, &beta, C, CUDA_R_32F, N,
                  jt_compute_get(), CUBLAS_GEMM_DEFAULT));
}

void gemm_batched(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha,
                  const float *A, long sA, const float *B, long sB, float beta, float *C, long sC, int batch) {
  // Same argument mapping as gemm(), plus the strides: the cuBLAS "A" argument is our B, so it takes sB.
  CB(cublasGemmStridedBatchedEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                                B, CUDA_R_32F, tb ? K : N, (long long)sB,
                                A, CUDA_R_32F, ta ? M : K, (long long)sA,
                                &beta, C, CUDA_R_32F, N, (long long)sC, batch,
                                jt_compute_get(), CUBLAS_GEMM_DEFAULT));
}

// ---- bf16 operands, fp32 output (plan M8.1b) ---------------------------------------------------------------------------
// gemm()'s argument mapping with both operands bf16 (gemm.h's section has the why). Two things differ from gemm():
// the compute type is always CUBLAS_COMPUTE_32F — a bf16 operand pair has no TF32 variant, the 8-bit mantissa is
// already there and the accumulate is fp32 in either mode — and the choice is not the caller's: the shape has to pass
// the alignment rule below first.
//
// Design §6.1's rule in this wrapper's own terms. cuBLAS sees m = N, n = M, k = K, its A is our B (leading dim
// tb ? K : N, op tb) and its B is our A (leading dim ta ? M : K, op ta). For a stored matrix the leading dimension IS
// the row length, i.e. the contiguous extent, so the rule is: both row lengths are multiples of 8 elements, and both
// bases are 16-byte aligned. (A batched call would also need the element strides so rounded; these two GEMMs are not
// batched, and gemm_bf16() below does not mirror gemm_batched().)
int gemm_bf16_ok(int ta, int tb, int M, int N, int K, const void *A, const void *B) {
  if (M <= 0 || N <= 0 || K <= 0) return 0;
  const int lda = tb ? K : N; // our B, as cuBLAS sees it
  const int ldb = ta ? M : K; // our A
  const int ea = tb ? K : N;  // cuBLAS's A: the extent its kernel walks contiguously
  const int eb = ta ? M : K;  // cuBLAS's B, likewise
  if ((lda % 8) != 0 || (ea % 8) != 0) return 0;
  if ((ldb % 8) != 0 || (eb % 8) != 0) return 0;
  if ((((uintptr_t)A | (uintptr_t)B) & (uintptr_t)15) != 0) return 0;
  return 1;
}

void gemm_bf16(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha, const unsigned short *A,
               const unsigned short *B, float beta, float *C) {
  if (!gemm_bf16_ok(ta, tb, M, N, K, A, B)) { // a caller that skipped the check gets the numbers, not a silent fp32 run
    fprintf(stderr, "gemm: bf16 ta=%d tb=%d M=%d N=%d K=%d ld %d/%d does not meet design §6.1's alignment rule\n", ta, tb, M,
            N, K, tb ? K : N, ta ? M : K);
    exit(1);
  }
  CB(cublasGemmEx(h, tb ? CUBLAS_OP_T : CUBLAS_OP_N, ta ? CUBLAS_OP_T : CUBLAS_OP_N, N, M, K, &alpha,
                  B, CUDA_R_16BF, tb ? K : N, A, CUDA_R_16BF, ta ? M : K, &beta, C, CUDA_R_32F, N,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
}
