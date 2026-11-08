// ref/act.c — the CPU references for plan M2.3 (design §3.3, §4.5, §4.7, §4.8). See ref/act.h for the equations.
//
// The float build of this file is the truth test/k_act.c compares the GPU kernels against; the double build
// (-DJT_REAL_DOUBLE, linked into gradcheck) is the truth test/grad_act.c finite-differences. Every scalar helper
// evaluates in double and rounds back once, at a `real` boundary; the array references then stay in `real` arithmetic so
// the float build rounds in the same places the device does (two multiplies, not one double multiply).
#include "act.h"

real ref_sigmoid(real u) { return REAL(1.0 / (1.0 + exp(-(double)u))); }

real ref_silu(real u) { return REAL((double)u * (double)ref_sigmoid(u)); }

real ref_dsilu(real u) {
  real s = ref_sigmoid(u);
  return REAL((double)s * (1.0 + (double)u * (1.0 - (double)s)));
}

// torch's softplus (beta = 1, threshold = 20): the linear branch above 20 is exactly u, matching the derivative of 1.
real ref_softplus(real u) { return (double)u > 20.0 ? u : REAL(log1p(exp((double)u))); }

real ref_dsoftplus(real u) { return (double)u > 20.0 ? REAL(1.0) : ref_sigmoid(u); }

void ref_silu_mul_fwd(const real *h1, const real *h2, real *act, int n) {
  for (int i = 0; i < n; i++) act[i] = REAL((double)ref_silu(h1[i]) * (double)h2[i]);
}

void ref_silu_mul_bwd(const real *dact, const real *h1, const real *h2, real *dh1, real *dh2, int n) {
  for (int i = 0; i < n; i++) {
    real silu = ref_silu(h1[i]);
    real dsilu = ref_dsilu(h1[i]);
    // dact * h2 * dsilu(h1) and dact * silu(h1), the association order the kernel uses.
    dh1[i] = REAL((double)dact[i] * (double)h2[i] * (double)dsilu);
    dh2[i] = REAL((double)dact[i] * (double)silu);
  }
}

void ref_sigmoid_gate_fwd(const real *o, const real *gate, real *out, int n) {
  for (int i = 0; i < n; i++) out[i] = REAL((double)o[i] * (double)ref_sigmoid(gate[i]));
}

void ref_sigmoid_gate_bwd(const real *dout, const real *o, const real *gate, real *d_o, real *dgate, int n) {
  for (int i = 0; i < n; i++) {
    real s = ref_sigmoid(gate[i]);
    d_o[i] = REAL((double)dout[i] * (double)s);
    dgate[i] = REAL((double)dout[i] * (double)o[i] * (double)s * (1.0 - (double)s));
  }
}

void ref_gdn_gates_fwd(const real *bb, const real *aa, const real *A_log, const real *dt_bias, real *beta, real *g,
                       int n, int heads) {
  for (int i = 0; i < n; i++) {
    int h = i % heads;
    real decay = REAL(-exp((double)A_log[h]));
    beta[i] = ref_sigmoid(bb[i]);
    g[i] = REAL((double)decay * (double)ref_softplus(REAL((double)aa[i] + (double)dt_bias[h])));
  }
}

void ref_gdn_gates_bwd(const real *dbeta, const real *dg, const real *beta, const real *aa, const real *A_log,
                       const real *dt_bias, real *dbb, real *daa, int n, int heads) {
  for (int i = 0; i < n; i++) {
    int h = i % heads;
    real decay = REAL(-exp((double)A_log[h]));
    real b = beta[i];
    real u = REAL((double)aa[i] + (double)dt_bias[h]);
    dbb[i] = REAL((double)dbeta[i] * (double)b * (1.0 - (double)b));
    daa[i] = REAL((double)dg[i] * (double)decay * (double)ref_dsoftplus(u));
  }
}
