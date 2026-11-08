// test/grad_gdn.c — the `gdn` group of gradcheck (plan M3.1; design §3.6 forward, §4.8 backward, §8.3's "gdn_ref's
// backward vs finite differences in double precision").
//
// Double precision, central difference h = 1e-6, gate rel < 1e-6 (test/harness.h), every element of every input:
//
//   ref_gdn_bwd   the five input gradients of the GDN recurrence — dq, dk, dv, dg, dbeta — against
//                 d/d(input) <dout, ref_gdn_fwd(input)>.  ref_gdn_fwd is the closure, so the check is the adjoint of
//                 the forward the kernel will be compared against, not of a hand-written formula.
//
// Plan M3.1 fixes the shape: H = 2, Dk = Dv = 8, B = 2 rows of lengths {5, 3}, g in [-1, 0] (decays are <= 1) and beta
// in [0, 1]. The rows are ragged and start from S = 0 each, so the state reset, the row-local token indexing and the
// fact that dS is 0 again at the next row's end are all exercised. A second, smaller case ({1, 4, 2} with H = 1 and
// Dk = 4 != Dv = 3) covers a one-token row and a non-square state, which no other shape in this group does. A third
// ({70, 3}, H = 2, Dk = Dv = 8) is a row long enough to cross a checkpoint segment (ref/gdn.c keeps one every 64
// tokens, design §8.3), so the backward's replay of a segment is checked by finite differences too, not only by the
// other two cases' single-segment rows.
//
// dq, dk, dv, dg and dbeta are all single-token terms that the backward writes (not accumulates) at the token they
// belong to; dS is what is carried. A swapped i/j or a state update taken before its decay shows up in one of the five.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"

#include "../ref/gdn.h" // ref_gdn_fwd / ref_gdn_bwd, double under -DJT_REAL_DOUBLE

enum { JT_G_MAX_TOK = 96, JT_G_MAX_ROWS = 4 };

// Zeroed and never NULL: every allocation below is small and fixed, and a failed one ends the run with a message.
static real *rnew(size_t n) {
  real *p = (real *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "grad_gdn: out of memory (%zu reals)\n", n);
    exit(1);
  }
  return p;
}

typedef struct {
  int B, H, Dk, Dv, N;
  int off[JT_G_MAX_ROWS + 1];
  // The unperturbed inputs. One of them is replaced by the closure's x; the others stay as they are.
  const real *q, *k, *v, *g, *beta;
} gdn_ctx;

// <dout, o> as a function of one input, the others held at their unperturbed values (harness.h's jt_fd_fn).
static void gdn_as_o_q(void *vv, const real *x, real *y) {
  gdn_ctx *c = (gdn_ctx *)vv;
  ref_gdn_fwd(c->B, c->off, c->H, c->Dk, c->Dv, x, c->k, c->v, c->g, c->beta, y);
}
static void gdn_as_o_k(void *vv, const real *x, real *y) {
  gdn_ctx *c = (gdn_ctx *)vv;
  ref_gdn_fwd(c->B, c->off, c->H, c->Dk, c->Dv, c->q, x, c->v, c->g, c->beta, y);
}
static void gdn_as_o_v(void *vv, const real *x, real *y) {
  gdn_ctx *c = (gdn_ctx *)vv;
  ref_gdn_fwd(c->B, c->off, c->H, c->Dk, c->Dv, c->q, c->k, x, c->g, c->beta, y);
}
static void gdn_as_o_g(void *vv, const real *x, real *y) {
  gdn_ctx *c = (gdn_ctx *)vv;
  ref_gdn_fwd(c->B, c->off, c->H, c->Dk, c->Dv, c->q, c->k, c->v, x, c->beta, y);
}
static void gdn_as_o_beta(void *vv, const real *x, real *y) {
  gdn_ctx *c = (gdn_ctx *)vv;
  ref_gdn_fwd(c->B, c->off, c->H, c->Dk, c->Dv, c->q, c->k, c->v, c->g, x, y);
}

// One fixture: random q/k/v in [-1,1], g in [-1,0], beta in [0,1], a random cotangent, the analytic gradient from
// ref_gdn_bwd and the numeric one from the harness core, for each of the five inputs.
static void check_gdn(const char *tag, const int *lens, int nrows, int H, int Dk, int Dv, uint32_t seed) {
  gdn_ctx c;
  char name[128];

  c.B = nrows;
  c.H = H;
  c.Dk = Dk;
  c.Dv = Dv;
  c.N = 0;
  c.off[0] = 0;
  for (int b = 0; b < nrows; b++) {
    c.N += lens[b];
    c.off[b + 1] = c.N;
  }
  if (c.N > JT_G_MAX_TOK || nrows > JT_G_MAX_ROWS) {
    printf("grad_gdn: %s is too large for the fixture\n", tag);
    jt_fail_inc();
    return;
  }

  const size_t nqk = (size_t)c.N * (size_t)H * (size_t)Dk; // q, k, dq, dk
  const size_t nv = (size_t)c.N * (size_t)H * (size_t)Dv;  // v, o, dv, dout
  const size_t ng = (size_t)c.N * (size_t)H;               // g, beta, dg, dbeta

  real *q = rnew(nqk), *k = rnew(nqk), *dq = rnew(nqk), *dk = rnew(nqk);
  real *v = rnew(nv), *o = rnew(nv), *dv = rnew(nv), *dout = rnew(nv);
  real *g = rnew(ng), *beta = rnew(ng), *dg = rnew(ng), *dbeta = rnew(ng);

  rnd_fill(q, nqk, -1.0, 1.0, seed);
  rnd_fill(k, nqk, -1.0, 1.0, seed ^ 0x5bd1e995u);
  rnd_fill(v, nv, -1.0, 1.0, seed ^ 0x1b873593u);
  rnd_fill(g, ng, -1.0, 0.0, seed ^ 0x2545f491u);       // decays exp(g) in [0.37, 1]
  rnd_fill(beta, ng, 0.0, 1.0, seed ^ 0x68e31da4u);
  rnd_fill(dout, nv, -1.0, 1.0, seed ^ 0x0f6d5a37u);    // the cotangent (the output weight w below)

  c.q = q, c.k = k, c.v = v, c.g = g, c.beta = beta;
  ref_gdn_fwd(c.B, c.off, c.H, c.Dk, c.Dv, q, k, v, g, beta, o);
  ref_gdn_bwd(c.B, c.off, c.H, c.Dk, c.Dv, q, k, v, g, beta, dout, dq, dk, dv, dg, dbeta);

  snprintf(name, sizeof name, "gdn/dq %s N=%d H=%d Dk=%d Dv=%d", tag, c.N, H, Dk, Dv);
  jt_grad_check(name, gdn_as_o_q, &c, q, nqk, dq, dout, nv);
  snprintf(name, sizeof name, "gdn/dk %s N=%d H=%d Dk=%d Dv=%d", tag, c.N, H, Dk, Dv);
  jt_grad_check(name, gdn_as_o_k, &c, k, nqk, dk, dout, nv);
  snprintf(name, sizeof name, "gdn/dv %s N=%d H=%d Dk=%d Dv=%d", tag, c.N, H, Dk, Dv);
  jt_grad_check(name, gdn_as_o_v, &c, v, nv, dv, dout, nv);
  snprintf(name, sizeof name, "gdn/dg %s N=%d H=%d Dk=%d Dv=%d", tag, c.N, H, Dk, Dv);
  jt_grad_check(name, gdn_as_o_g, &c, g, ng, dg, dout, nv);
  snprintf(name, sizeof name, "gdn/dbeta %s N=%d H=%d Dk=%d Dv=%d", tag, c.N, H, Dk, Dv);
  jt_grad_check(name, gdn_as_o_beta, &c, beta, ng, dbeta, dout, nv);

  free(q), free(k), free(dq), free(dk);
  free(v), free(o), free(dv), free(dout);
  free(g), free(beta), free(dg), free(dbeta);
}

int grad_gdn(void) {
  static const int lens_plan[2] = {5, 3};    // plan M3.1: B = 2, lengths {5, 3}
  static const int lens_small[3] = {1, 4, 2}; // an L = 1 row, three rows, a non-square state
  static const int lens_ckpt[2] = {70, 3};   // 70 > 64: one row spans two checkpoint segments

  check_gdn("plan", lens_plan, 2, 2, 8, 8, 0x9e3779b9u);
  check_gdn("small", lens_small, 3, 1, 4, 3, 0x7f4a7c15u);
  check_gdn("ckpt", lens_ckpt, 2, 2, 8, 8, 0xc2b2ae35u);
  return jt_exit_code();
}
