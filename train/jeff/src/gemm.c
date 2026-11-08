// src/gemm.c — M1.1: the row-major cuBLAS wrapper (design §5). Every dense op in jefftrain is one of these two calls.
//
// Design §5's issue, once: the row-major C[M,N] = alpha op(A) op(B) + beta C is the column-major C^T[N,M] =
// alpha op(B)^T op(A)^T + beta C^T. So m = N, n = M, k = K, the cuBLAS "A" argument is our B (leading dim tb ? K : N,
// op tb ? T : N) and the cuBLAS "B" argument is our A (leading dim ta ? M : K, op ta ? T : N); C's leading dim is its
// row length, N. Getting these the wrong way round is the classic bug this wrapper exists to make impossible, so the
// argument order is written out once here and nowhere else.
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
