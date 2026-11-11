// The type contract of libtinycublas's Ex entry points (G's review of the bf16 fix, 2026-09-21), on the null device: A and
// B of one type, a compute type cuBLAS defines for that input type, and C as that compute type allows. Anything else is
// NOT_SUPPORTED. Before this, A=f16 with B=bf16 ran on the f16 kernel (and the reverse on the bf16 one), and every compute
// type except exactly COMPUTE_16F had its alpha/beta read as float, 64F and 32I included.
// Each row is checked through all three entry points: GemmEx, GemmStridedBatchedEx (batch 2) and GemmBatchedEx (2 pointers).
//
// The second half pins the f32 *kernel* choice (G-m8probe finding 1, 2026-11-11): an f32 GEMM used to take the TF32
// tensor-core kernel whatever the compute type said, which silently put ~6e-4 relative error into a caller asking for
// CUBLAS_COMPUTE_32F. The decision is queried through tinycublas_f32_uses_tf32 (the same card-free shape split-K's plan
// rows have), and the launch it causes is visible in the driver's "would launch" lines.
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
int tinycublas_f32_uses_tf32(cublasHandle_t h, cublasComputeType_t ct);
static int fails, rows;
#define CHECK(c, ...) do { if (!(c)) { fails++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } } while (0)
static cublasHandle_t h;
static void *A, *B, *C, **Ap, **Bp, **Cp;
static const float one_f = 1.f, zero_f = 0.f;
static const uint16_t one_h = 0x3c00, zero_h = 0;
static void row(const char *what, cudaDataType at, cudaDataType bt, cudaDataType ct, cublasComputeType_t cm, cublasStatus_t want) {
  const void *al = (cm == CUBLAS_COMPUTE_16F || cm == CUBLAS_COMPUTE_16F_PEDANTIC) ? (const void *)&one_h : (const void *)&one_f;
  const void *be = (cm == CUBLAS_COMPUTE_16F || cm == CUBLAS_COMPUTE_16F_PEDANTIC) ? (const void *)&zero_h : (const void *)&zero_f;
  cublasStatus_t s1 = cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, A, at, 32, B, bt, 32, be, C, ct, 32, cm, CUBLAS_GEMM_DEFAULT);
  cublasStatus_t s2 = cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, A, at, 32, 1024, B, bt, 32, 1024, be, C, ct, 32,
                                                 1024, 2, cm, CUBLAS_GEMM_DEFAULT);
  cublasStatus_t s3 = cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 32, 32, 32, al, (const void *const *)Ap, at, 32, (const void *const *)Bp, bt, 32,
                                          be, (void *const *)Cp, ct, 32, 2, cm, CUBLAS_GEMM_DEFAULT);
  rows++;
  int ok = s1 == want && s2 == want && s3 == want;
  printf("  %s %-48s GemmEx %d, StridedBatchedEx %d, BatchedEx %d (want %d)\n", ok ? "ok  " : "FAIL", what, s1, s2, s3, want);
  if (!ok) fails++;
}
int main(void) {
  CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate");
  CHECK(cudaMalloc(&A, 1 << 16) == cudaSuccess && cudaMalloc(&B, 1 << 16) == cudaSuccess && cudaMalloc(&C, 1 << 16) == cudaSuccess, "malloc");
  void *hp[2]; CHECK(cudaMalloc((void **)&Ap, 16) == cudaSuccess && cudaMalloc((void **)&Bp, 16) == cudaSuccess && cudaMalloc((void **)&Cp, 16) == cudaSuccess, "ptr arrays");
  hp[0] = A; hp[1] = (char *)A + 4096; cudaMemcpy(Ap, hp, 16, cudaMemcpyHostToDevice);
  hp[0] = B; hp[1] = (char *)B + 4096; cudaMemcpy(Bp, hp, 16, cudaMemcpyHostToDevice);
  hp[0] = C; hp[1] = (char *)C + 8192; cudaMemcpy(Cp, hp, 16, cudaMemcpyHostToDevice);
  printf("served (SUCCESS on the null device, where launches are reported, not executed):\n");
  row("bf16 x bf16 -> f32, COMPUTE_32F (ggml's bf16)", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("bf16 x bf16 -> bf16, COMPUTE_32F", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("f16 x f16 -> f16, COMPUTE_16F (ggml's f16)", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_SUCCESS);
  row("f16 x f16 -> f32, COMPUTE_32F", CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("f32 x f32 -> f32, COMPUTE_32F (jt's exact path)", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_SUCCESS);
  row("f32 x f32 -> f32, COMPUTE_32F_FAST_TF32", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_STATUS_SUCCESS);
  printf("refused (NOT_SUPPORTED = %d):\n", CUBLAS_STATUS_NOT_SUPPORTED);
  row("f16 x bf16 (mismatched A/B)", CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 x f16 (mismatched A/B)", CUDA_R_16BF, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 x f16 (mismatched A/B)", CUDA_R_32F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 with COMPUTE_16F", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f16 COMPUTE_16F with an f32 C", CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_16F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 with COMPUTE_64F", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_64F, CUBLAS_STATUS_NOT_SUPPORTED);
  row("f32 with COMPUTE_32I", CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32I, CUBLAS_STATUS_NOT_SUPPORTED);
  row("bf16 with COMPUTE_32F_FAST_TF32", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_STATUS_NOT_SUPPORTED);
  // Which kernel an f32 GEMM takes. ggml sets the math mode to TF32 and asks for COMPUTE_32F (common.cuh:1544 +
  // batched_mul_mat_traits<GGML_TYPE_F32>), so its path must stay on the tensor cores; jt's gemm.c asks for COMPUTE_32F
  // on a handle whose math mode is the default, and GDN's chunked recurrence does the same, so those must be exact.
  printf("f32 kernel choice (1 = TF32 tensor cores, 0 = exact fp32 kernel):\n");
  cublasHandle_t hd = NULL;
  CHECK(cublasCreate(&hd) == CUBLAS_STATUS_SUCCESS, "cublasCreate (default math mode)");
  struct { const char *what; cublasHandle_t hh; cublasComputeType_t cm; int want; } kr[] = {
    {"COMPUTE_32F, default math mode (jt gemm.c / GDN chunked)", h,  CUBLAS_COMPUTE_32F,               0},
    {"COMPUTE_32F_PEDANTIC, default math mode",                   h,  CUBLAS_COMPUTE_32F_PEDANTIC,      0},
    {"COMPUTE_32F, math mode TF32 (ggml)",                        hd, CUBLAS_COMPUTE_32F,               1},
    {"COMPUTE_32F_FAST_TF32, default math mode",                  h,  CUBLAS_COMPUTE_32F_FAST_TF32,     1},
  };
  CHECK(cublasSetMathMode(hd, CUBLAS_TF32_TENSOR_OP_MATH) == CUBLAS_STATUS_SUCCESS, "cublasSetMathMode TF32");
  for (size_t i = 0; i < sizeof kr / sizeof kr[0]; i++) {
    const int got = tinycublas_f32_uses_tf32(kr[i].hh, kr[i].cm);
    rows++;
    printf("  %s %-56s tf32 %d (want %d)\n", got == kr[i].want ? "ok  " : "FAIL", kr[i].what, got, kr[i].want);
    if (got != kr[i].want) fails++;
  }
  cublasDestroy(hd);
  printf("\n%d row(s), %d failure(s)\n", rows, fails);
  return fails ? 1 : 0;
}
