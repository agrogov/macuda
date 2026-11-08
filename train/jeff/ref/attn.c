// ref/attn.c — the CPU oracle for attn_softmax_fwd/bwd (plan M2.5, design §4.7 and §8.1/§8.2).
//
// Written literally from the design's formulas, in the layout ref/attn.h documents. Sums are accumulated in double even
// in the float build: a row of up to 1026 exponentials summed in fp32 picks up an order-dependent error of its own, and
// the reference is the oracle the kernel is judged against (the same rule ref/real.h's r_* wrappers follow — one
// rounding, at the end).
#include "attn.h"

#include <math.h>
#include <stddef.h>

// The scores of one row: scale applied, max over the live (s <= t) entries, exponentiate, normalise, and write 0 above
// the diagonal.
static void row_fwd(const real *sr, real *pr, int L, int t, double c) {
  double m = c * (double)sr[0]; // s = 0 is live for every t >= 0
  for (int s = 1; s <= t; s++) {
    double x = c * (double)sr[s];
    if (x > m) m = x;
  }
  double sum = 0.0;
  for (int s = 0; s <= t; s++) sum += exp(c * (double)sr[s] - m);
  for (int s = 0; s <= t; s++) pr[s] = REAL(exp(c * (double)sr[s] - m) / sum);
  for (int s = t + 1; s < L; s++) pr[s] = REAL(0);
}

// One row of the backward: dot = sum_{s<=t} P*dP, then scale * P * (dP - dot); 0 above the diagonal (P is 0 there, so
// the formula would give 0 anyway — writing it explicitly keeps a masked entry exactly 0 whatever dP holds).
static void row_bwd(const real *pr, const real *dr, real *orow, int L, int t, double c) {
  double dot = 0.0;
  for (int s = 0; s <= t; s++) dot += (double)pr[s] * (double)dr[s];
  for (int s = 0; s <= t; s++) orow[s] = REAL(c * (double)pr[s] * ((double)dr[s] - dot));
  for (int s = t + 1; s < L; s++) orow[s] = REAL(0);
}

void ref_attn_softmax_fwd(const real *S, real *P, int L, int heads, real scale) {
  if (S == NULL || P == NULL || L <= 0 || heads <= 0) return;
  const double c = (double)scale;
  const size_t plane = (size_t)L * (size_t)L;
  for (int h = 0; h < heads; h++) {
    const real *Sh = S + (size_t)h * plane;
    real *Ph = P + (size_t)h * plane;
    for (int t = 0; t < L; t++) row_fwd(Sh + (size_t)t * (size_t)L, Ph + (size_t)t * (size_t)L, L, t, c);
  }
}

void ref_attn_softmax_bwd(const real *P, const real *dP, real *dS, int L, int heads, real scale) {
  if (P == NULL || dP == NULL || dS == NULL || L <= 0 || heads <= 0) return;
  const double c = (double)scale;
  const size_t plane = (size_t)L * (size_t)L;
  for (int h = 0; h < heads; h++) {
    const real *Ph = P + (size_t)h * plane;
    const real *Dh = dP + (size_t)h * plane;
    real *Oh = dS + (size_t)h * plane;
    for (int t = 0; t < L; t++) {
      const size_t off = (size_t)t * (size_t)L;
      row_bwd(Ph + off, Dh + off, Oh + off, L, t, c);
    }
  }
}
