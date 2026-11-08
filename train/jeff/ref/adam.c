// ref/adam.c — the CPU oracle for plan M2.6 (design §2.2, §8.1). See ref/adam.h for the equations.
//
// The float build of this file is what test/k_adam.c compares the GPU kernels against; the double build
// (-DJT_REAL_DOUBLE) is the truth for the coming step-parity work. Each element update stays in `real` arithmetic, so the
// float build rounds in the same places the device does (one multiply, one add, one divide, one sqrt), which is what makes
// a 1e-6 comparison meaningful; sqrt goes through r_sqrt (double, rounded once), so it can differ from the device's
// sqrtf by one ulp and nothing more.
#include "adam.h"

#include <stdlib.h>

void ref_sumsq_partial(const real *g, size_t n, real *partials, int nblocks) {
  if (g == NULL || partials == NULL || nblocks <= 0) return;
  for (int b = 0; b < nblocks; b++) {
    real s = REAL(0);
    for (size_t i = (size_t)b; i < n; i += (size_t)nblocks) s = REAL((double)s + (double)g[i] * (double)g[i]);
    partials[b] = s;
  }
}

void ref_sumsq_final(const real *partials, int nblocks, real *out) {
  if (partials == NULL || out == NULL || nblocks <= 0) {
    if (out != NULL) *out = REAL(0);
    return;
  }
  real s = REAL(0);
  for (int b = 0; b < nblocks; b++) s = REAL((double)s + (double)partials[b]);
  *out = s;
}

void ref_sumsq(const real *g, size_t n, int nblocks, real *out) {
  if (out == NULL) return;
  real *partials = (real *)malloc((size_t)(nblocks > 0 ? nblocks : 1) * sizeof *partials);
  if (partials == NULL) {
    *out = REAL(0);
    return;
  }
  ref_sumsq_partial(g, n, partials, nblocks);
  ref_sumsq_final(partials, nblocks, out);
  free(partials);
}

real ref_clip_coef(real norm) {
  real coef = REAL(1.0 / ((double)norm + REF_ADAM_CLIP_EPS));
  return (double)coef < REF_ADAM_MAX_NORM ? coef : REAL(REF_ADAM_MAX_NORM);
}

void ref_clip_scale(real *g, size_t n, real coef) {
  if (g == NULL) return;
  for (size_t i = 0; i < n; i++) g[i] = REAL((double)g[i] * (double)coef);
}

void ref_adam_bias_corr(double b1, double b2, int step_t, real *bias1, real *bias2) {
  const double t = (double)(step_t > 0 ? step_t : 1);
  if (bias1 != NULL) *bias1 = REAL(1.0 - pow(b1, t));
  if (bias2 != NULL) *bias2 = REAL(1.0 - pow(b2, t));
}

void ref_adamw_step(real *p, const real *g, real *m, real *v, size_t n, size_t lr_group_split, real lr_lora,
                    real lr_readout, real b1, real b2, real eps, real bias1, real bias2) {
  if (p == NULL || g == NULL || m == NULL || v == NULL) return;
  const double one_m_b1 = 1.0 - (double)b1;
  const double one_m_b2 = 1.0 - (double)b2;
  const double bc1 = (double)bias1;
  const double bc2 = (double)bias2;
  for (size_t i = 0; i < n; i++) {
    // The lr group is a range of the flat buffer: LoRA first, the readout last (design §8.1, plan M2.6).
    const real lr = i < lr_group_split ? lr_lora : lr_readout;
    const real mi = REAL((double)b1 * (double)m[i] + one_m_b1 * (double)g[i]);
    const real vi = REAL((double)b2 * (double)v[i] + one_m_b2 * (double)g[i] * (double)g[i]);
    m[i] = mi;
    v[i] = vi;
    const real mhat = REAL((double)mi / bc1);
    const real vhat = REAL((double)vi / bc2);
    const real denom = REAL((double)r_sqrt(vhat) + (double)eps);
    p[i] = REAL((double)p[i] - (double)lr * (double)mhat / (double)denom);
  }
}
