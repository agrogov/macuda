// test/grad_act.c — the `act` group of gradcheck (plan M2.3, and the M2.0 numeric core in test/harness.h).
//
// Double precision, central difference h = 1e-6, gate rel < 1e-6, every element of every differentiable input:
//
//   silu_mul_bwd     d h1 and d h2
//   sigmoid_gate_bwd d o and d gate
//   gdn_gates_bwd    d aa through g = -exp(A_log) * softplus(aa + dt_bias), and d bb through beta = sigmoid(bb)
//
// A_log and dt_bias are frozen checkpoint values (design §4.8: no gradient is needed for them), so they are constants of
// the closures. Inputs stay well inside the softplus linear branch (u = aa + dt_bias within [-4.5, 4.5], far from the
// threshold 20), which is the plan's M2.3 caveat; the branch itself is covered value-wise by test/k_act.c.
//
// Each closure writes the whole output of one reference forward, so for the gates the cotangent is the concatenation
// [dbeta; dg] of length 2n and the closure writes [beta; g] the same way.
#include <stdio.h>

#include "harness.h"

#include "../ref/act.h" // ref_*_fwd / ref_*_bwd, double under -DJT_REAL_DOUBLE

enum { NMAX = 256, HMAX = 16 };

// ---- silu_mul ---------------------------------------------------------------------------------------------------------
typedef struct {
  const real *a, *b;
  int n;
} pair_ctx;

static void silu_as_h1(void *v, const real *x, real *y) {
  pair_ctx *c = v;
  ref_silu_mul_fwd(x, c->b, y, c->n);
}
static void silu_as_h2(void *v, const real *x, real *y) {
  pair_ctx *c = v;
  ref_silu_mul_fwd(c->a, x, y, c->n);
}

static void check_silu_mul(int n, uint32_t seed) {
  char name[64];
  real h1[NMAX], h2[NMAX], w[NMAX], dh1[NMAX], dh2[NMAX];
  pair_ctx c = {h1, h2, n};

  rnd_fill(h1, n, -3.0, 3.0, seed);
  rnd_fill(h2, n, -2.0, 2.0, seed ^ 0x5bd1e995u);
  rnd_fill(w, n, -1.5, 1.5, seed ^ 0x1b873593u);
  ref_silu_mul_bwd(w, h1, h2, dh1, dh2, n);

  snprintf(name, sizeof name, "silu_mul/d_h1 n=%d", n);
  jt_grad_check(name, silu_as_h1, &c, h1, (size_t)n, dh1, w, (size_t)n);
  snprintf(name, sizeof name, "silu_mul/d_h2 n=%d", n);
  jt_grad_check(name, silu_as_h2, &c, h2, (size_t)n, dh2, w, (size_t)n);
}

// ---- sigmoid_gate -----------------------------------------------------------------------------------------------------
static void gate_as_o(void *v, const real *x, real *y) {
  pair_ctx *c = v;
  ref_sigmoid_gate_fwd(x, c->b, y, c->n);
}
static void gate_as_gate(void *v, const real *x, real *y) {
  pair_ctx *c = v;
  ref_sigmoid_gate_fwd(c->a, x, y, c->n);
}

static void check_sigmoid_gate(int n, uint32_t seed) {
  char name[64];
  real o[NMAX], gate[NMAX], w[NMAX], d_o[NMAX], dgate[NMAX];
  pair_ctx c = {o, gate, n};

  rnd_fill(o, n, -2.0, 2.0, seed);
  rnd_fill(gate, n, -3.0, 3.0, seed ^ 0x9e3779b1u);
  rnd_fill(w, n, -1.5, 1.5, seed ^ 0x85ebca6bu);
  ref_sigmoid_gate_bwd(w, o, gate, d_o, dgate, n);

  snprintf(name, sizeof name, "sigmoid_gate/d_o n=%d", n);
  jt_grad_check(name, gate_as_o, &c, o, (size_t)n, d_o, w, (size_t)n);
  snprintf(name, sizeof name, "sigmoid_gate/d_gate n=%d", n);
  jt_grad_check(name, gate_as_gate, &c, gate, (size_t)n, dgate, w, (size_t)n);
}

// ---- gdn_gates --------------------------------------------------------------------------------------------------------
typedef struct {
  const real *bb, *aa, *A_log, *dt_bias;
  int n, heads;
} gates_ctx;

static void gates_as_aa(void *v, const real *x, real *y) {
  gates_ctx *c = v;
  ref_gdn_gates_fwd(c->bb, x, c->A_log, c->dt_bias, y, y + c->n, c->n, c->heads); // y = [beta; g]
}
static void gates_as_bb(void *v, const real *x, real *y) {
  gates_ctx *c = v;
  ref_gdn_gates_fwd(x, c->aa, c->A_log, c->dt_bias, y, y + c->n, c->n, c->heads);
}

static void check_gdn_gates(int n, int heads, uint32_t seed) {
  char name[64];
  real bb[NMAX], aa[NMAX], A_log[HMAX], dt_bias[HMAX];
  real beta[NMAX], g[NMAX], w[2 * NMAX], dbb[NMAX], daa[NMAX];
  gates_ctx c = {bb, aa, A_log, dt_bias, n, heads};

  rnd_fill(bb, n, -3.0, 3.0, seed);
  rnd_fill(aa, n, -4.0, 4.0, seed ^ 0x7f4a7c15u);
  rnd_fill(A_log, heads, -1.5, 1.5, seed ^ 0x2545f491u);
  rnd_fill(dt_bias, heads, -0.5, 0.5, seed ^ 0x94d049bbu);
  rnd_fill(w, 2 * n, -1.5, 1.5, seed ^ 0x632be5abu); // the cotangent [dbeta; dg]
  ref_gdn_gates_fwd(bb, aa, A_log, dt_bias, beta, g, n, heads);
  ref_gdn_gates_bwd(w, w + n, beta, aa, A_log, dt_bias, dbb, daa, n, heads);

  snprintf(name, sizeof name, "gdn_gates/d_aa n=%d heads=%d", n, heads);
  jt_grad_check(name, gates_as_aa, &c, aa, (size_t)n, daa, w, (size_t)(2 * n));
  snprintf(name, sizeof name, "gdn_gates/d_bb n=%d heads=%d", n, heads);
  jt_grad_check(name, gates_as_bb, &c, bb, (size_t)n, dbb, w, (size_t)(2 * n));
}

int grad_act(void) {
  check_silu_mul(17, 0x11u);  // not a multiple of the device's block or warp
  check_silu_mul(256, 0x12u); // exactly NMAX
  check_sigmoid_gate(13, 0x21u);
  check_sigmoid_gate(256, 0x22u);
  check_gdn_gates(12, 3, 0x31u);  // uneven heads, so i % heads really varies
  check_gdn_gates(48, 16, 0x32u); // the model's 16 heads
  return jt_exit_code();
}
