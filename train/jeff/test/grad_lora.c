// test/grad_lora.c — the `lora` group of gradcheck (plan M2.0/M2.6): the finite-difference check of design §4.2's linear
// with LoRA, through the CPU oracle in ref/linear.c. This is the M2.0 gate the plan asks for: "A and B in LoRA".
//
// What is differentiated (y = x W^T + s (x A^T) B^T, L = <w, y> with a random cotangent w for the output):
//
//   1. dA — x is the factor A [rank,in]; B, W and x are fixed. Analytic: e^T x (via ref_linear_bwd with dy = w).
//   2. dB — x is the factor B [out,rank]. Analytic: s * dy^T u, with u the forward's saved x A^T.
//   3. dx — x is the layer's input x [n,in]. Analytic: dy W + e A. dx is not in the plan's list of M2.0 inputs (it is
//           the residual's, D§4.6), but it is the same formula's third gradient and costs nothing to check here.
//   4. dx_frozen — A = B = NULL: the plain frozen linear's dx = dy W, the path every non-adapted projection takes.
//
// Two shapes, both deliberate: the first is asymmetric in three ways (n=3 rows, in=4, out=5, rank=2), so a transposed
// index anywhere in the reference shows up; the second makes rank > out (rank 3, out 2), which is legal for a rank-16
// adapter on the model's 255-row readout but not for any of its 3584/1024-row projections.
//
// A and B are both filled with random values: with B = 0 (torch's init) dA would be exactly 0 and the dA check would pass
// without testing anything. The gate is JT_TOL_GRAD (rel < 1e-6) in the double build, as the plan's M2.0 says; nothing
// here is loosened.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"
#include "../ref/linear.h"

typedef struct {
  int n, in, out, rank;
  real scale;
  const real *x, *W, *A, *B;
  real *u; // [n, rank] scratch: the forward's saved activation, rewritten by every closure call
} LoraCtx;

// x = the A factor.
static void f_by_A(void *v, const real *A, real *y) {
  const LoraCtx *c = (const LoraCtx *)v;
  ref_linear_fwd(c->x, c->W, A, c->B, y, c->u, c->n, c->in, c->out, c->rank, c->scale);
}
// x = the B factor.
static void f_by_B(void *v, const real *B, real *y) {
  const LoraCtx *c = (const LoraCtx *)v;
  ref_linear_fwd(c->x, c->W, c->A, B, y, c->u, c->n, c->in, c->out, c->rank, c->scale);
}
// x = the input.
static void f_by_x(void *v, const real *x, real *y) {
  const LoraCtx *c = (const LoraCtx *)v;
  ref_linear_fwd(x, c->W, c->A, c->B, y, c->u, c->n, c->in, c->out, c->rank, c->scale);
}
// x = the input, with no LoRA (A = B = NULL): the frozen path.
static void f_by_x_frozen(void *v, const real *x, real *y) {
  const LoraCtx *c = (const LoraCtx *)v;
  ref_linear_fwd(x, c->W, NULL, NULL, y, NULL, c->n, c->in, c->out, 0, c->scale);
}

static real *alloc_real(size_t n) {
  real *p = (real *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_lora: out of memory (%zu elements)\n", n);
    exit(1);
  }
  return p;
}

// One shape: random x/W/A/B, one cotangent w, then the three analytic gradients against the finite differences.
static void case_lora(const char *tag, int n, int in, int out, int rank, uint32_t seed) {
  const size_t nx = (size_t)n * (size_t)in, nW = (size_t)out * (size_t)in, nA = (size_t)rank * (size_t)in;
  const size_t nB = (size_t)out * (size_t)rank, nu = (size_t)n * (size_t)rank, ny = (size_t)n * (size_t)out;
  char name[128];

  real *x = alloc_real(nx), *W = alloc_real(nW), *A = alloc_real(nA), *B = alloc_real(nB);
  real *u = alloc_real(nu), *y = alloc_real(ny), *w = alloc_real(ny);
  real *dx = alloc_real(nx), *dA = alloc_real(nA), *dB = alloc_real(nB);
  const real scale = REAL(2.0); // alpha/rank = 32/16, design §2.1

  rnd_fill(x, nx, -1.0, 1.0, seed);
  rnd_fill(W, nW, -0.5, 0.5, seed + 1u);
  rnd_fill(A, nA, -0.5, 0.5, seed + 2u);
  rnd_fill(B, nB, -0.5, 0.5, seed + 3u);
  rnd_fill(w, ny, -1.0, 1.0, seed + 4u);

  LoraCtx c = {n, in, out, rank, scale, x, W, A, B, u};
  ref_linear_fwd(x, W, A, B, y, u, n, in, out, rank, scale); // the point the three checks differentiate at

  // The analytic gradients: dx written, dA/dB accumulated, so they start at zero.
  memset(dA, 0, nA * sizeof *dA);
  memset(dB, 0, nB * sizeof *dB);
  ref_linear_bwd(x, W, u, w, A, B, dx, dA, dB, n, in, out, rank, scale);

  snprintf(name, sizeof name, "lora/dA_%s", tag);
  jt_grad_check(name, f_by_A, &c, A, nA, dA, w, ny);
  snprintf(name, sizeof name, "lora/dB_%s", tag);
  jt_grad_check(name, f_by_B, &c, B, nB, dB, w, ny);
  snprintf(name, sizeof name, "lora/dx_%s", tag);
  jt_grad_check(name, f_by_x, &c, x, nx, dx, w, ny);

  // The frozen path: contract dA/dB stay untouched (NULL) and dx is dy W.
  memset(dx, 0, nx * sizeof *dx);
  ref_linear_bwd(x, W, NULL, w, NULL, NULL, dx, NULL, NULL, n, in, out, 0, scale);
  snprintf(name, sizeof name, "lora/dx_frozen_%s", tag);
  jt_grad_check(name, f_by_x_frozen, &c, x, nx, dx, w, ny);

  free(x);
  free(W);
  free(A);
  free(B);
  free(u);
  free(y);
  free(w);
  free(dx);
  free(dA);
  free(dB);
}

int grad_lora(void) {
  case_lora("n3_in4_out5_r2", 3, 4, 5, 2, 61u);
  case_lora("n2_in3_out2_r3", 2, 3, 2, 3, 71u);
  return jt_exit_code();
}
