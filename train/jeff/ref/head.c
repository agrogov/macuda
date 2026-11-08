// ref/head.c — the CPU oracle for the decision head (plan M2.5, design §4.1 and CLM-8B's tools/jeff_lora/train.py).
//
// The per-row core is train.py's `soft_ce` written out, with the two gradients design §4.1 gives (dlogits and the
// readout's dR / dz). Accumulation is in double even in the float build, for the reason ref/attn.c gives: the reference
// is the oracle, so it rounds once, at the end.
#include "head.h"

#include <math.h>
#include <stddef.h>

// One row: softmax over n options, the weighted loss and the dlogits. Writes loss always, p and dg when they are not
// NULL.
static void row_ce(const real *lg, const real *tg, int n, double w, real *loss, real *dg, real *pg) {
  double m = (double)lg[0];
  for (int j = 1; j < n; j++) {
    if ((double)lg[j] > m) m = (double)lg[j];
  }
  // log-sum-exp, so that a dead option's log p stays the finite number (-1e9 - lse) it is in torch's log_softmax:
  // 0 * that is 0, while 0 * -inf would be NaN. exp() of a dead option's score underflows to exactly 0 either way.
  double sum = 0.0;
  for (int j = 0; j < n; j++) sum += exp((double)lg[j] - m);
  const double lse = m + log(sum);
  double tsum = 0.0, ce = 0.0;
  for (int j = 0; j < n; j++) tsum += (double)tg[j];
  for (int j = 0; j < n; j++) {
    const double p = exp((double)lg[j] - lse);
    ce -= (double)tg[j] * ((double)lg[j] - lse);
    if (dg != NULL) dg[j] = REAL(w * (p * tsum - (double)tg[j]));
    if (pg != NULL) pg[j] = REAL(p);
  }
  *loss = REAL(w * ce);
}

void ref_ce_head(const real *logits, int n_opts, const real *targets, const real *wrow, real wgroup, int B,
                 real *loss_out, real *dlogits, real *probs_out) {
  if (logits == NULL || targets == NULL || wrow == NULL || loss_out == NULL) return;
  if (n_opts <= 0 || B <= 0) return;
  const size_t n = (size_t)n_opts;
  const double W = (double)wgroup;
  for (int r = 0; r < B; r++) {
    const double w = (double)wrow[r] / W;
    row_ce(logits + (size_t)r * n, targets + (size_t)r * n, n_opts, w, loss_out + r,
           dlogits != NULL ? dlogits + (size_t)r * n : NULL, probs_out != NULL ? probs_out + (size_t)r * n : NULL);
  }
}

void ref_head_readout_fwd(const real *z, const real *R, const unsigned char *live, int B, int n, int d, real *logits) {
  if (z == NULL || R == NULL || logits == NULL || B <= 0 || n <= 0 || d <= 0) return;
  for (int r = 0; r < B; r++) {
    for (int j = 0; j < n; j++) {
      const size_t rj = (size_t)r * (size_t)n + (size_t)j;
      if (live != NULL && !live[rj]) {
        logits[rj] = REAL(REF_HEAD_DEAD);
        continue;
      }
      const real *zr = z + (size_t)r * (size_t)d;
      const real *Rj = R + (size_t)j * (size_t)d;
      double acc = 0.0;
      for (int k = 0; k < d; k++) acc += (double)Rj[k] * (double)zr[k];
      logits[rj] = REAL(acc);
    }
  }
}

void ref_head_readout_bwd(const real *z, const real *R, const real *dlogits, int B, int n, int d, real *dz,
                          real *dR) {
  if (z == NULL || R == NULL || dlogits == NULL || dz == NULL || dR == NULL) return;
  if (B <= 0 || n <= 0 || d <= 0) return;
  for (int r = 0; r < B; r++) {
    const real *dl = dlogits + (size_t)r * (size_t)n;
    for (int k = 0; k < d; k++) {
      double acc = 0.0;
      for (int j = 0; j < n; j++) acc += (double)R[(size_t)j * (size_t)d + (size_t)k] * (double)dl[j];
      dz[(size_t)r * (size_t)d + (size_t)k] = REAL(acc);
    }
  }
  for (int j = 0; j < n; j++) {
    for (int k = 0; k < d; k++) {
      double acc = 0.0;
      for (int r = 0; r < B; r++) acc += (double)dlogits[(size_t)r * (size_t)n + (size_t)j] * (double)z[(size_t)r * (size_t)d + (size_t)k];
      dR[(size_t)j * (size_t)d + (size_t)k] += REAL(acc);
    }
  }
}
