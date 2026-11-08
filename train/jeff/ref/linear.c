// ref/linear.c — the CPU oracle for a linear layer with LoRA (plan M2.6/M5, design §4.2). See ref/linear.h for the
// equations and the layouts.
//
// Accumulation is in double even in the float build, as in ref/head.c and ref/attn.c: the reference is the oracle, so it
// rounds once at the end rather than once per term. The one place where the *order* matters (and is therefore fixed) is
// `e = s * (dy B)`, which the backward builds first and reuses for dx and dA exactly as design §4.2 says.
#include "linear.h"

#include <stdio.h>
#include <stdlib.h>

static real *alloc_real(size_t n, const char *what) {
  real *p = (real *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "ref_linear: out of memory (%s, %zu elements)\n", what, n);
    exit(1);
  }
  return p;
}

void ref_linear_fwd(const real *x, const real *W, const real *A, const real *B, real *y, real *u, int n, int in,
                    int out, int rank, real scale) {
  if (x == NULL || W == NULL || y == NULL || n <= 0 || in <= 0 || out <= 0) return;
  const int lora = A != NULL && B != NULL && rank > 0;
  const size_t nin = (size_t)in, nout = (size_t)out, nrk = lora ? (size_t)rank : 0;
  // A caller that only wants y still needs somewhere to put the saved u; a small heap buffer keeps the function
  // malloc-free in the common case where the trainer passes its own u.
  real *usave = u;
  if (lora && usave == NULL) usave = alloc_real((size_t)n * nrk, "u");

  for (int r = 0; r < n; r++) {
    const real *xr = x + (size_t)r * nin;
    real *yr = y + (size_t)r * nout;
    real *ur = lora ? usave + (size_t)r * nrk : NULL;
    for (size_t k = 0; k < nrk; k++) { // u = x A^T
      const real *ak = A + k * nin;
      double acc = 0.0;
      for (size_t i = 0; i < nin; i++) acc += (double)ak[i] * (double)xr[i];
      ur[k] = REAL(acc);
    }
    for (size_t o = 0; o < nout; o++) { // y = x W^T + s * u B^T
      const real *wo = W + o * nin;
      double acc = 0.0;
      for (size_t i = 0; i < nin; i++) acc += (double)wo[i] * (double)xr[i];
      if (lora) {
        const real *bo = B + o * nrk;
        for (size_t k = 0; k < nrk; k++) acc += (double)scale * (double)ur[k] * (double)bo[k];
      }
      yr[o] = REAL(acc);
    }
  }
  if (lora && u == NULL) free(usave);
}

void ref_linear_bwd(const real *x, const real *W, const real *u, const real *dy, const real *A, const real *B, real *dx,
                    real *dA, real *dB, int n, int in, int out, int rank, real scale) {
  if (x == NULL || W == NULL || dy == NULL || n <= 0 || in <= 0 || out <= 0) return;
  const int lora = A != NULL && B != NULL && u != NULL && rank > 0;
  const size_t nin = (size_t)in, nout = (size_t)out, nrk = lora ? (size_t)rank : 0;

  // e = s * (dy B) [n, rank], design §4.2: built once, then reused by dx and dA.
  real *e = lora ? alloc_real((size_t)n * nrk, "e") : NULL;
  if (lora) {
    for (int r = 0; r < n; r++) {
      const real *dyr = dy + (size_t)r * nout;
      real *er = e + (size_t)r * nrk;
      for (size_t k = 0; k < nrk; k++) {
        double acc = 0.0;
        for (size_t o = 0; o < nout; o++) acc += (double)dyr[o] * (double)B[o * nrk + k];
        er[k] = REAL((double)scale * acc);
      }
    }
  }

  if (dx != NULL) { // dx = dy W + e A
    for (int r = 0; r < n; r++) {
      const real *dyr = dy + (size_t)r * nout;
      real *dxr = dx + (size_t)r * nin;
      const real *er = lora ? e + (size_t)r * nrk : NULL;
      for (size_t i = 0; i < nin; i++) {
        double acc = 0.0;
        for (size_t o = 0; o < nout; o++) acc += (double)dyr[o] * (double)W[o * nin + i];
        if (lora)
          for (size_t k = 0; k < nrk; k++) acc += (double)er[k] * (double)A[k * nin + i];
        dxr[i] = REAL(acc);
      }
    }
  }

  if (lora && dA != NULL) { // dA += e^T x   [rank, in]
    for (size_t k = 0; k < nrk; k++) {
      for (size_t i = 0; i < nin; i++) {
        double acc = (double)dA[k * nin + i];
        for (int r = 0; r < n; r++) acc += (double)e[(size_t)r * nrk + k] * (double)x[(size_t)r * nin + i];
        dA[k * nin + i] = REAL(acc);
      }
    }
  }

  if (lora && dB != NULL) { // dB += s * dy^T u [out, rank]
    for (size_t o = 0; o < nout; o++) {
      for (size_t k = 0; k < nrk; k++) {
        double acc = (double)dB[o * nrk + k];
        for (int r = 0; r < n; r++)
          acc += (double)scale * (double)dy[(size_t)r * nout + o] * (double)u[(size_t)r * nrk + k];
        dB[o * nrk + k] = REAL(acc);
      }
    }
  }

  free(e);
}
