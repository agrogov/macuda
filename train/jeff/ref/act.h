// ref/act.h — the CPU references for the activation and gate kernels (plan M2.3, design §3.3, §4.5, §4.7, §4.8).
//
// Written before their kernels (plan rule 2): every `ref_x` here is the oracle the GPU test compares `k_x` against, and
// every `ref_x_bwd` is also what test/grad_act.c finite-differences in double precision. `real` is float by default and
// double under -DJT_REAL_DOUBLE (ref/real.h), so nothing here has to know which build it is in.
//
// The three ops, word for word from the design:
//   * MLP gating (D§4.5):  act = silu(h1) * h2,      dh1 = dact * h2 * dsilu(h1),   dh2 = dact * silu(h1)
//   * attention output gate (D§4.7's last lines): out = o * sig(gate), do = dout * sig(gate),
//     dgate = dout * o * sig(gate) * (1 - sig(gate))
//   * GDN gates (D§4.8's last lines): beta = sig(bb), g = -exp(A_log) * softplus(aa + dt_bias),
//     dbb = dbeta * beta * (1 - beta), daa = dg * (-exp(A_log)) * dsoftplus(aa + dt_bias)
//
// softplus is torch's: u > 20 ? u : log1p(exp(u)), with the matching derivative (1 on the linear branch). A test keeps
// its inputs away from 20 (plan M2.3), which only means the gate does not hinge on the branch.
//
// bb, aa, beta, g are [n] token-major: entry i belongs to head i % heads, and A_log/dt_bias are [heads]. bf16 weights
// are converted by the loader, so every pointer here is fp32 (or double in the gradcheck build).
#ifndef JT_REF_ACT_H
#define JT_REF_ACT_H

#include "real.h"

// The scalar pieces, so that ref/conv.c (silu after the conv) and later refs can share one definition. Each computes in
// double and rounds back once, which is what ref/real.h's r_* wrappers do; the array references below then work in
// `real` so the float build rounds where the GPU rounds.
real ref_sigmoid(real u);   // 1 / (1 + exp(-u))
real ref_silu(real u);      // u * sigmoid(u)
real ref_dsilu(real u);     // sigmoid(u) * (1 + u * (1 - sigmoid(u)))
real ref_softplus(real u);  // u > 20 ? u : log1p(exp(u))
real ref_dsoftplus(real u); // u > 20 ? 1 : sigmoid(u)

// act[i] = silu(h1[i]) * h2[i]                                    (MLP, D§4.5)
void ref_silu_mul_fwd(const real *h1, const real *h2, real *act, int n);
// dh1[i] = dact[i] * h2[i] * dsilu(h1[i]);  dh2[i] = dact[i] * silu(h1[i])
void ref_silu_mul_bwd(const real *dact, const real *h1, const real *h2, real *dh1, real *dh2, int n);

// out[i] = o[i] * sigmoid(gate[i])                                 (attention output gate, D§4.7)
void ref_sigmoid_gate_fwd(const real *o, const real *gate, real *out, int n);
// dout -> do[i] = dout[i] * sigmoid(gate[i]);  dgate[i] = dout[i] * o[i] * sigmoid(gate[i]) * (1 - sigmoid(gate[i]))
void ref_sigmoid_gate_bwd(const real *dout, const real *o, const real *gate, real *d_o, real *dgate, int n);

// beta[i] = sigmoid(bb[i]);  g[i] = -exp(A_log[i % heads]) * softplus(aa[i] + dt_bias[i % heads])   (D§3.6, D§4.8)
// n is the number of (token, head) entries; heads is the model's 16 (a test may shrink it).
void ref_gdn_gates_fwd(const real *bb, const real *aa, const real *A_log, const real *dt_bias, real *beta, real *g,
                       int n, int heads);
// dbb[i] = dbeta[i] * beta[i] * (1 - beta[i]);  daa[i] = dg[i] * (-exp(A_log[i % heads])) * dsigmoid(aa + dt_bias)
// beta is the forward's saved value (torch's sigmoid backward uses the saved output, not a recomputation).
void ref_gdn_gates_bwd(const real *dbeta, const real *dg, const real *beta, const real *aa, const real *A_log,
                       const real *dt_bias, real *dbb, real *daa, int n, int heads);

#endif // JT_REF_ACT_H
