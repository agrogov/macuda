// test/test_gemm.c — M1.2: the row-major GEMM wrappers against a double-precision CPU triple loop (design §5, §10.1).
//
// What is checked: every (ta,tb) pair on ragged and large sizes, alpha in {1,2}, beta in {0,1} (beta=1 with C
// prefilled), and gemm_batched with batch 8 on the shapes the attention code will use. The metric is the plan's
// rel = max|got-want| / max(1e-30, max|want|); the gate is 1e-5 in exact (JT_EXACT) mode and 5e-3 under TF32 (plan M1.2).
// The gate is picked from gemm_is_exact(), not from an assumption, so one binary serves both card-check steps.
//
// On the shim's null device nothing computes: every call is still issued (that is the point — allocation ambiguity,
// argument marshalling and launch sequencing), the numbers are not checked and the run says so and exits 0, which is
// what `make test-null` needs (README.md).
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "util.h"
#include "gemm.h"

#define TOL_EXACT 1e-5
#define TOL_TF32 5e-3

static int verifying = 1; // 0 on the null device: issue the calls, skip the numbers
static int ncase, nfail;

// Deterministic inputs (plan M1.2: a fixed xorshift seed), uniform in [-1,1).
static uint32_t rng = 2463534242u;
static uint32_t xorshift(void) {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}
static float frand(void) { return (float)(((double)(xorshift() >> 8) * (1.0 / 16777216.0)) * 2.0 - 1.0); }

static void *xmalloc(size_t n) {
  void *p = malloc(n != 0 ? n : 1);
  if (p == NULL) {
    fprintf(stderr, "test_gemm: out of host memory (%zu bytes)\n", n);
    exit(1);
  }
  return p;
}

// The hardware path needs both variables (design §11.2); without them the shim is a null device.
static int has_card(void) {
#if defined(JT_SHIM)
  const char *hw = getenv("TINYNV_HW");
  const char *sock = getenv("TINYNV_SOCKET");
  return hw != NULL && hw[0] != '\0' && sock != NULL && sock[0] != '\0';
#else
  return 1;
#endif
}

// Stored layout -> contiguous row-major [M,K] / [K,N], so the reference loop has one access pattern whatever ta/tb is.
static void prep_a(const float *A, int ta, int M, int K, float *out) {
  for (int m = 0; m < M; m++)
    for (int k = 0; k < K; k++) out[(long)m * K + k] = ta ? A[(long)k * M + m] : A[(long)m * K + k];
}
static void prep_b(const float *B, int tb, int K, int N, float *out) {
  for (int k = 0; k < K; k++)
    for (int n = 0; n < N; n++) out[(long)k * N + n] = tb ? B[(long)n * K + k] : B[(long)k * N + n];
}

// The CPU reference: C[m,n] = alpha * sum_k A[m,k] B[k,n] + beta * Cin[m,n], double accumulator (plan M1.2).
static void ref_gemm(int M, int N, int K, double alpha, const float *A, const float *B, double beta,
                     const double *Cin, double *Cout) {
  double *acc = xmalloc((size_t)N * sizeof *acc);
  for (int m = 0; m < M; m++) {
    const float *arow = A + (long)m * K;
    for (int n = 0; n < N; n++) acc[n] = 0.0;
    for (int k = 0; k < K; k++) {
      const double a = (double)arow[k];
      const float *brow = B + (long)k * N;
      for (int n = 0; n < N; n++) acc[n] += a * (double)brow[n];
    }
    for (int n = 0; n < N; n++) Cout[(long)m * N + n] = alpha * acc[n] + beta * Cin[(long)m * N + n];
  }
  free(acc);
}

// rel = max|got-want| / max(1e-30, max|want|); a non-finite result is a failure, never a small error.
static double rel_err(const float *got, const double *want, long n) {
  double maxd = 0.0, maxw = 0.0;
  for (long i = 0; i < n; i++) {
    const double d = fabs((double)got[i] - want[i]);
    const double w = fabs(want[i]);
    if (!isfinite((double)got[i]) || !isfinite(d)) return INFINITY;
    if (d > maxd) maxd = d;
    if (w > maxw) maxw = w;
  }
  return maxd / (maxw > 1e-30 ? maxw : 1e-30);
}

static void report(const char *name, double rel) {
  const double tol = gemm_is_exact() ? TOL_EXACT : TOL_TF32;
  ncase++;
  if (rel <= tol) {
    printf("ok   %-56s rel=%.3g\n", name, rel);
  } else {
    nfail++;
    printf("FAIL %-56s rel=%.3g tol=%.0e\n", name, rel, tol);
  }
}

// One gemm() call: A [M,K] (or [K,M]), B [K,N] (or [N,K]), C [M,N] and C prefilled so beta=1 means something.
static void check_gemm(cublasHandle_t h, int ta, int tb, int M, int N, int K, float alpha, float beta) {
  const long na = (long)M * K, nb = (long)K * N, nc = (long)M * N;
  float *a = xmalloc((size_t)na * sizeof *a);
  float *b = xmalloc((size_t)nb * sizeof *b);
  float *c = xmalloc((size_t)nc * sizeof *c);
  for (long i = 0; i < na; i++) a[i] = frand();
  for (long i = 0; i < nb; i++) b[i] = frand();
  for (long i = 0; i < nc; i++) c[i] = frand();

  float *da = NULL, *db = NULL, *dc = NULL;
  CK(cudaMalloc((void **)&da, (size_t)na * sizeof *da));
  CK(cudaMalloc((void **)&db, (size_t)nb * sizeof *db));
  CK(cudaMalloc((void **)&dc, (size_t)nc * sizeof *dc));
  CK(cudaMemcpy(da, a, (size_t)na * sizeof *a, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b, (size_t)nb * sizeof *b, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c, (size_t)nc * sizeof *c, cudaMemcpyHostToDevice));

  gemm(h, ta, tb, M, N, K, alpha, da, db, beta, dc);

  double rel = 0.0;
  if (verifying) {
    float *ap = xmalloc((size_t)na * sizeof *ap);
    float *bp = xmalloc((size_t)nb * sizeof *bp);
    double *cin = xmalloc((size_t)nc * sizeof *cin);
    double *want = xmalloc((size_t)nc * sizeof *want);
    prep_a(a, ta, M, K, ap);
    prep_b(b, tb, K, N, bp);
    for (long i = 0; i < nc; i++) cin[i] = (double)c[i];
    ref_gemm(M, N, K, (double)alpha, ap, bp, (double)beta, cin, want);
    CK(cudaMemcpy(c, dc, (size_t)nc * sizeof *c, cudaMemcpyDeviceToHost));
    rel = rel_err(c, want, nc);
    free(ap);
    free(bp);
    free(cin);
    free(want);
  }

  CK(cudaFree(da));
  CK(cudaFree(db));
  CK(cudaFree(dc));
  if (verifying) {
    char name[96];
    snprintf(name, sizeof name, "gemm ta=%d tb=%d M=%d N=%d K=%d alpha=%g beta=%g", ta, tb, M, N, K, (double)alpha,
             (double)beta);
    report(name, rel);
  }
  free(a);
  free(b);
  free(c);
}

// One gemm_batched() call: `batch` independent GEMMs, strides = the stored matrix sizes (plan M1.1: the strides follow
// their matrices), so a wrong sA/sB/sC shows up as the second matrix onward being wrong.
static void check_batched(cublasHandle_t h, int ta, int tb, int M, int N, int K, int batch, float alpha, float beta) {
  const long na = (long)M * K, nb = (long)K * N, nc = (long)M * N;
  const long sA = na, sB = nb, sC = nc;
  float *a = xmalloc((size_t)(na * batch) * sizeof *a);
  float *b = xmalloc((size_t)(nb * batch) * sizeof *b);
  float *c = xmalloc((size_t)(nc * batch) * sizeof *c);
  for (long i = 0; i < na * batch; i++) a[i] = frand();
  for (long i = 0; i < nb * batch; i++) b[i] = frand();
  for (long i = 0; i < nc * batch; i++) c[i] = frand();

  float *da = NULL, *db = NULL, *dc = NULL;
  CK(cudaMalloc((void **)&da, (size_t)(na * batch) * sizeof *da));
  CK(cudaMalloc((void **)&db, (size_t)(nb * batch) * sizeof *db));
  CK(cudaMalloc((void **)&dc, (size_t)(nc * batch) * sizeof *dc));
  CK(cudaMemcpy(da, a, (size_t)(na * batch) * sizeof *a, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b, (size_t)(nb * batch) * sizeof *b, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c, (size_t)(nc * batch) * sizeof *c, cudaMemcpyHostToDevice));

  gemm_batched(h, ta, tb, M, N, K, alpha, da, sA, db, sB, beta, dc, sC, batch);

  double rel = 0.0;
  if (verifying) {
    float *ap = xmalloc((size_t)na * sizeof *ap);
    float *bp = xmalloc((size_t)nb * sizeof *bp);
    double *cin = xmalloc((size_t)nc * sizeof *cin);
    double *want = xmalloc((size_t)(nc * batch) * sizeof *want);
    for (int j = 0; j < batch; j++) {
      prep_a(a + (long)j * sA, ta, M, K, ap);
      prep_b(b + (long)j * sB, tb, K, N, bp);
      for (long i = 0; i < nc; i++) cin[i] = (double)c[(long)j * sC + i];
      ref_gemm(M, N, K, (double)alpha, ap, bp, (double)beta, cin, want + (long)j * sC);
    }
    CK(cudaMemcpy(c, dc, (size_t)(nc * batch) * sizeof *c, cudaMemcpyDeviceToHost));
    rel = rel_err(c, want, nc * batch);
    free(ap);
    free(bp);
    free(cin);
    free(want);
  }

  CK(cudaFree(da));
  CK(cudaFree(db));
  CK(cudaFree(dc));
  if (verifying) {
    char name[96];
    snprintf(name, sizeof name, "batched ta=%d tb=%d M=%d N=%d K=%d batch=%d alpha=%g beta=%g", ta, tb, M, N, K, batch,
             (double)alpha, (double)beta);
    report(name, rel);
  }
  free(a);
  free(b);
  free(c);
}

int main(void) {
  // Plan M1.2's sizes: the degenerate one, odd ones (7,13,5 and 130,17,1024 catch leading-dim/transpose mix-ups) and
  // the two real width-of-the-model ones (4200 rows = a full 8-row batch, 1024 = the hidden size, 16 = a LoRA rank).
  static const int sizes[][3] = {
      {1, 1, 1}, {7, 13, 5}, {64, 64, 64}, {130, 17, 1024}, {4200, 16, 1024}, {16, 1024, 4200},
  };
  static const float alphas[] = {1.0f, 2.0f};
  static const float betas[] = {0.0f, 1.0f};
  static const int lens[] = {37, 300};
  cublasHandle_t h;

  verifying = has_card();
  gemm_init();
  CB(cublasCreate(&h));

  for (size_t s = 0; s < sizeof sizes / sizeof sizes[0]; s++) {
    for (int ta = 0; ta <= 1; ta++)
      for (int tb = 0; tb <= 1; tb++)
        for (size_t i = 0; i < sizeof alphas / sizeof alphas[0]; i++)
          for (size_t j = 0; j < sizeof betas / sizeof betas[0]; j++)
            check_gemm(h, ta, tb, sizes[s][0], sizes[s][1], sizes[s][2], alphas[i], betas[j]);
  }

  // gemm_batched, batch 8 = the 8 query heads: M = L (rows of the packed row), N = K = 256 (the head dim), which is the
  // [L x 256] shape plan M1.2 names, in the three (ta,tb) pairs the attention code uses.
  for (size_t l = 0; l < sizeof lens / sizeof lens[0]; l++) {
    const int L = lens[l];
    static const int pairs[][2] = {{0, 1}, {0, 0}, {1, 0}};
    for (size_t p = 0; p < sizeof pairs / sizeof pairs[0]; p++)
      for (size_t i = 0; i < sizeof alphas / sizeof alphas[0]; i++)
        check_batched(h, pairs[p][0], pairs[p][1], L, 256, 256, 8, alphas[i], betas[i]);
    // ... plus the two per-head attention GEMMs themselves: scores Q[L,256] K^T (M=L, N=L, K=256) and P[L,L] V[L,256]
    // (M=L, N=256, K=L). These are the calls M1.2a asks the card about, so they are worth naming.
    check_batched(h, 0, 1, L, L, 256, 8, 1.0f, 0.0f);
    check_batched(h, 0, 0, L, 256, L, 8, 1.0f, 0.0f);
  }

  CB(cublasDestroy(h));

  if (!verifying) {
    printf("GEMM OK (null device: launch only)\n");
    return 0;
  }
  if (nfail != 0) {
    printf("GEMM FAILED %d/%d\n", ncase - nfail, ncase);
    return 1;
  }
  printf("GEMM OK %d/%d\n", ncase, ncase);
  return 0;
}
