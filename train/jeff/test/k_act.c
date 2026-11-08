// test/k_act.c — the `act` group of test_kernels (plan M2.3): silu_mul_fwd/bwd, sigmoid_gate_fwd/bwd and
// gdn_gates_fwd/bwd against their CPU references in ref/act.c.
//
// The pattern is the plan M2.0 one: random inputs with rnd_fill, the reference (float build) on the host, the kernel on
// the device with the same inputs, and CHECK_REL with the plan's gates — 1e-5 forward, 1e-4 backward (both sides are
// fp32, so only the libm functions and the association order differ; nothing here reduces). The backward runs take the
// forward's saved beta for the GDN gates, exactly as the trainer will.
//
// Shapes: the plan's two packed batches, {1, 37, 130} = 168 tokens and {520, 670, 1026, 300} = 2516 tokens, at the real
// widths (3584 for the MLP's act, 2048 for the attention gate, 16 heads for the GDN gates). The full batch at a smaller
// width only keeps the allocations modest; an elementwise kernel cannot tell a width from any other count. The last
// gate case runs aa + dt_bias past softplus's threshold 20, so the kernel's u > 20 branch is really taken.
//
// On the null device every launch happens and every value check prints "(launch only)" (harness.h); the group still
// returns JT_OK, which is what make test-null needs.
#include <stdio.h>
#include <stdlib.h>

#include "harness.h"

#include "../kernels/act.h" // k_* launch wrappers
#include "../ref/act.h"     // ref_* CPU truth

static float *hvec(size_t n) {
  float *p = (float *)malloc((n ? n : 1) * sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_act: out of memory (%zu floats)\n", n);
    exit(1);
  }
  return p;
}

static float *dvec(size_t n) { return (float *)dev_alloc((n ? n : 1) * sizeof(float)); }

static float *up(const float *host, size_t n) { return dev_upload(host, n); }

// ---- silu_mul: act = silu(h1) * h2, dh1 = dact*h2*dsilu(h1), dh2 = dact*silu(h1) (D§4.5) -----------------------------
static void case_silu_mul(int n, uint32_t seed) {
  char name[96];
  size_t nf = (size_t)n;
  float *h1 = hvec(nf), *h2 = hvec(nf), *dact = hvec(nf);
  float *act = hvec(nf), *dh1 = hvec(nf), *dh2 = hvec(nf);
  float *want_act = hvec(nf), *want_dh1 = hvec(nf), *want_dh2 = hvec(nf);

  rnd_fill(h1, nf, -4.0, 4.0, seed);
  rnd_fill(h2, nf, -3.0, 3.0, seed ^ 0x5bd1e995u);
  rnd_fill(dact, nf, -2.0, 2.0, seed ^ 0x1b873593u);
  ref_silu_mul_fwd(h1, h2, want_act, n);
  ref_silu_mul_bwd(dact, h1, h2, want_dh1, want_dh2, n);

  float *d_h1 = up(h1, nf), *d_h2 = up(h2, nf), *d_dact = up(dact, nf);
  float *d_act = dvec(nf), *d_dh1 = dvec(nf), *d_dh2 = dvec(nf);

  k_silu_mul_fwd(d_h1, d_h2, d_act, n, 0);
  k_silu_mul_bwd(d_dact, d_h1, d_h2, d_dh1, d_dh2, n, 0);
  dev_download(d_act, act, nf);
  dev_download(d_dh1, dh1, nf);
  dev_download(d_dh2, dh2, nf);

  snprintf(name, sizeof name, "silu_mul_fwd n=%d", n);
  CHECK_REL(name, act, want_act, nf, JT_TOL_FWD);
  snprintf(name, sizeof name, "silu_mul_bwd/dh1 n=%d", n);
  CHECK_REL(name, dh1, want_dh1, nf, JT_TOL_BWD);
  snprintf(name, sizeof name, "silu_mul_bwd/dh2 n=%d", n);
  CHECK_REL(name, dh2, want_dh2, nf, JT_TOL_BWD);

  dev_free(d_h1), dev_free(d_h2), dev_free(d_dact), dev_free(d_act), dev_free(d_dh1), dev_free(d_dh2);
  free(h1), free(h2), free(dact), free(act), free(dh1), free(dh2), free(want_act), free(want_dh1), free(want_dh2);
}

// ---- sigmoid_gate: out = o * sigmoid(gate), d_o = dout*s, dgate = dout*o*s*(1-s) (D§4.7) -----------------------------
static void case_sigmoid_gate(int n, uint32_t seed) {
  char name[96];
  size_t nf = (size_t)n;
  float *o = hvec(nf), *gate = hvec(nf), *dout = hvec(nf);
  float *out = hvec(nf), *d_o = hvec(nf), *dgate = hvec(nf);
  float *want_out = hvec(nf), *want_do = hvec(nf), *want_dgate = hvec(nf);

  rnd_fill(o, nf, -2.0, 2.0, seed);
  rnd_fill(gate, nf, -6.0, 6.0, seed ^ 0x9e3779b1u);
  rnd_fill(dout, nf, -2.0, 2.0, seed ^ 0x85ebca6bu);
  ref_sigmoid_gate_fwd(o, gate, want_out, n);
  ref_sigmoid_gate_bwd(dout, o, gate, want_do, want_dgate, n);

  float *d_o_in = up(o, nf), *d_gate = up(gate, nf), *d_dout = up(dout, nf);
  float *d_out = dvec(nf), *d_do = dvec(nf), *d_dgate = dvec(nf);

  k_sigmoid_gate_fwd(d_o_in, d_gate, d_out, n, 0);
  k_sigmoid_gate_bwd(d_dout, d_o_in, d_gate, d_do, d_dgate, n, 0);
  dev_download(d_out, out, nf);
  dev_download(d_do, d_o, nf);
  dev_download(d_dgate, dgate, nf);

  snprintf(name, sizeof name, "sigmoid_gate_fwd n=%d", n);
  CHECK_REL(name, out, want_out, nf, JT_TOL_FWD);
  snprintf(name, sizeof name, "sigmoid_gate_bwd/d_o n=%d", n);
  CHECK_REL(name, d_o, want_do, nf, JT_TOL_BWD);
  snprintf(name, sizeof name, "sigmoid_gate_bwd/dgate n=%d", n);
  CHECK_REL(name, dgate, want_dgate, nf, JT_TOL_BWD);

  dev_free(d_o_in), dev_free(d_gate), dev_free(d_dout), dev_free(d_out), dev_free(d_do), dev_free(d_dgate);
  free(o), free(gate), free(dout), free(out), free(d_o), free(dgate), free(want_out), free(want_do), free(want_dgate);
}

// ---- gdn_gates: beta = sigmoid(bb), g = -exp(A_log)*softplus(aa+dt_bias), and dbb, daa (D§4.8) -----------------------
// aalo/aahi bound aa (and so aa + dt_bias, since |dt_bias| <= 0.5): the ordinary cases stay inside [-4.5, 4.5], the last
// one crosses softplus's threshold at 20.
static void case_gdn_gates(int n, int heads, double aalo, double aahi, uint32_t seed) {
  char name[96];
  size_t nf = (size_t)n, hf = (size_t)heads;
  float *bb = hvec(nf), *aa = hvec(nf), *dbeta = hvec(nf), *dg = hvec(nf);
  float *beta = hvec(nf), *g = hvec(nf), *dbb = hvec(nf), *daa = hvec(nf);
  float *want_beta = hvec(nf), *want_g = hvec(nf), *want_dbb = hvec(nf), *want_daa = hvec(nf);
  float *A_log = hvec(hf), *dt_bias = hvec(hf);

  rnd_fill(bb, nf, -3.0, 3.0, seed);
  rnd_fill(aa, nf, aalo, aahi, seed ^ 0x7f4a7c15u);
  rnd_fill(dbeta, nf, -1.5, 1.5, seed ^ 0x2545f491u);
  rnd_fill(dg, nf, -1.5, 1.5, seed ^ 0x94d049bbu);
  rnd_fill(A_log, hf, -1.5, 1.5, seed ^ 0x632be5abu);
  rnd_fill(dt_bias, hf, -0.5, 0.5, seed ^ 0x27d4eb2fu);

  ref_gdn_gates_fwd(bb, aa, A_log, dt_bias, want_beta, want_g, n, heads);
  ref_gdn_gates_bwd(dbeta, dg, want_beta, aa, A_log, dt_bias, want_dbb, want_daa, n, heads);

  float *d_bb = up(bb, nf), *d_aa = up(aa, nf), *d_A = up(A_log, hf), *d_dt = up(dt_bias, hf);
  float *d_dbeta = up(dbeta, nf), *d_dg = up(dg, nf);
  float *d_beta = dvec(nf), *d_g = dvec(nf), *d_dbb = dvec(nf), *d_daa = dvec(nf);

  k_gdn_gates_fwd(d_bb, d_aa, d_A, d_dt, d_beta, d_g, n, heads, 0);
  k_gdn_gates_bwd(d_dbeta, d_dg, d_beta, d_aa, d_A, d_dt, d_dbb, d_daa, n, heads, 0);
  dev_download(d_beta, beta, nf);
  dev_download(d_g, g, nf);
  dev_download(d_dbb, dbb, nf);
  dev_download(d_daa, daa, nf);

  snprintf(name, sizeof name, "gdn_gates_fwd/beta n=%d heads=%d", n, heads);
  CHECK_REL(name, beta, want_beta, nf, JT_TOL_FWD);
  snprintf(name, sizeof name, "gdn_gates_fwd/g n=%d heads=%d", n, heads);
  CHECK_REL(name, g, want_g, nf, JT_TOL_FWD);
  snprintf(name, sizeof name, "gdn_gates_bwd/dbb n=%d heads=%d", n, heads);
  CHECK_REL(name, dbb, want_dbb, nf, JT_TOL_BWD);
  snprintf(name, sizeof name, "gdn_gates_bwd/daa n=%d heads=%d", n, heads);
  CHECK_REL(name, daa, want_daa, nf, JT_TOL_BWD);

  dev_free(d_bb), dev_free(d_aa), dev_free(d_A), dev_free(d_dt);
  dev_free(d_dbeta), dev_free(d_dg), dev_free(d_beta), dev_free(d_g), dev_free(d_dbb), dev_free(d_daa);
  free(bb), free(aa), free(dbeta), free(dg), free(beta), free(g), free(dbb), free(daa);
  free(want_beta), free(want_g), free(want_dbb), free(want_daa), free(A_log), free(dt_bias);
}

int k_act(void) {
  int t3 = 0, t4 = 0;
  const int *l3 = jt_lens_ragged3(), *l4 = jt_lens_ragged4();
  for (int b = 0; b < JT_RAGGED3_ROWS; b++) t3 += l3[b];
  for (int b = 0; b < JT_RAGGED4_ROWS; b++) t4 += l4[b];

  // The MLP's act, [tokens, 3584].
  case_silu_mul(t3 * 3584, 0x101u);
  case_silu_mul(t4 * 512, 0x102u);
  // The attention output gate, [tokens, 2048].
  case_sigmoid_gate(t3 * 2048, 0x201u);
  case_sigmoid_gate(t4 * 512, 0x202u);
  // The GDN gates, [tokens, 16], plus a shrunk head count (the ref and the kernel must agree on i % heads).
  case_gdn_gates(t3 * 16, 16, -4.0, 4.0, 0x301u);
  case_gdn_gates(t4 * 16, 16, -4.0, 4.0, 0x302u);
  case_gdn_gates(1000, 3, -4.0, 4.0, 0x303u);
  // aa + dt_bias crosses softplus's threshold: u <= 20 everywhere above, u up to 100 here.
  case_gdn_gates(1024, 16, -6.0, 100.0, 0x304u);

  return jt_exit_code();
}
