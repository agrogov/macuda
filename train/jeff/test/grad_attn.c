// test/grad_attn.c — the `attn` group of gradcheck (plan M2.0/M2.5): the finite-difference check of
// ref_attn_softmax_bwd.
//
// What is differentiated: P = causal_softmax(scale * S) (design §4.7/§8.2, ref_attn_softmax_fwd), with a random
// cotangent dP standing in for dP[t,s] = do_t . v_s. So L(S) = <dP, P(S)> and the numeric gradient of L must be
// dS = scale * P * (dP - sum_{s<=t} P*dP) — which is exactly what ref_attn_softmax_bwd computes, masked entries
// included (they are 0 on both sides, because P is 0 above the diagonal).
//
// CPU only and double precision (the Makefile compiles this file and the refd_*.o build of ref/ with
// -DJT_REAL_DOUBLE); nothing here touches a device, so jt_grad_check is never skipped. The gate is JT_TOL_GRAD.
#include <stdio.h>
#include <stdlib.h>

#include "harness.h"
#include "../ref/attn.h"

typedef struct {
  int L, heads;
  real scale;
} AttnCtx;

// y = P, computed from x = S by the reference forward: the function the numeric gradient differentiates.
static void fwd_from_scores(void *v, const real *x, real *y) {
  const AttnCtx *c = (const AttnCtx *)v;
  ref_attn_softmax_fwd(x, y, c->L, c->heads, c->scale);
}

static real *alloc(size_t n) {
  real *p = (real *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_attn: out of memory (%zu elements)\n", n);
    exit(1);
  }
  return p;
}

// One (L, heads, scale): random S, random dP, the analytic dS from the reference backward, the numeric one from the
// harness. Every element of S is perturbed, so L is small on purpose (the gate does not depend on the size).
static void case_grad(const char *tag, int L, int heads, real scale, uint32_t seed) {
  const size_t n = (size_t)heads * (size_t)L * (size_t)L;
  real *S = alloc(n), *P = alloc(n), *dP = alloc(n), *dS = alloc(n);
  real *w = alloc(n);
  rnd_fill(S, n, -4.0, 4.0, seed);
  rnd_fill(dP, n, -2.0, 2.0, seed + 1u); // the cotangent: dP itself, so that L(S) = <dP, P(S)>
  for (size_t i = 0; i < n; i++) w[i] = dP[i];
  ref_attn_softmax_fwd(S, P, L, heads, scale);
  ref_attn_softmax_bwd(P, dP, dS, L, heads, scale);

  AttnCtx c = {L, heads, scale};
  char name[96];
  snprintf(name, sizeof name, "attn_bwd/%s", tag);
  jt_grad_check(name, fwd_from_scores, &c, S, n, dS, w, n);

  free(S);
  free(P);
  free(dP);
  free(dS);
  free(w);
}

int grad_attn(void) {
  const real c = REAL(1.0 / 16.0); // the attention scale (design §4.7)
  // L=1 and a single head would make every gradient exactly 0 (one option, P=1), which checks nothing; L=3 with one
  // head still covers the first row's all-zero row, and the others cover a real triangular pattern.
  case_grad("L3_h1", 3, 1, c, 11u);
  case_grad("L5_h2", 5, 2, c, 12u);
  case_grad("L7_h2_scale1", 7, 2, REAL(1.0), 13u); // a scale other than 1/16, applied inside both directions
  return jt_exit_code();
}
