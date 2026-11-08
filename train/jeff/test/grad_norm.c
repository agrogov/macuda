// test/grad_norm.c — the `norm` group of gradcheck (plan M2.0/M2.2). CPU only, double precision (-DJT_REAL_DOUBLE).
//
// Plan M2.2: "gradcheck has passed for all three backwards" — rmsnorm_bwd, gated_rmsnorm_bwd (both differentiable
// inputs: the x that is normalised and the gate z) and l2norm_bwd. The numeric core is test/harness.h's jt_grad_check:
// L(x) = <w, f(x)>, central difference with h = 1e-6 over every element, gate rel < 1e-6. The inputs here are the
// references' own shapes (design §4.3/§4.4/§4.8), shrunk so that differencing every element stays cheap; the shapes
// used on the device are test/k_norm.c's business.
//
// What is not checked, and why: the norm weights are frozen in this model (design §2.1/§4.3), so there is no dw. The
// zero_centred flag is exercised both ways for both RMSNorm forms — a norm that multiplies by w where the model
// needs (1 + w) (or the other way round) is exactly the mistake plan M2.2 warns about, and it is invisible to a
// forward-only comparison if the wrong flag is used on both sides.
#include <stdio.h>

#include "harness.h"
#include "../ref/norm.h"

#define GR_MAXD 1024
#define GR_MAXROWS 2
#define GR_MAXN (GR_MAXROWS * GR_MAXD)

// Everything a forward closure needs: the inputs that are held fixed, the dimensions, and the row scalars r that the
// forward saves and the analytic backward consumes.
typedef struct {
  const real *x, *w, *z;
  int rows, d, zc;
  real post_scale;
  real r[GR_MAXROWS];
} NormCtx;

// One closure per (op, differentiable input); each is just the reference forward with the fixed inputs filled in.
static void fx_rmsnorm(void *v, const real *x, real *y) {
  NormCtx *c = (NormCtx *)v;
  ref_rmsnorm_fwd(x, c->w, y, c->r, c->rows, c->d, c->zc);
}
static void fx_gated_x(void *v, const real *x, real *y) {
  NormCtx *c = (NormCtx *)v;
  ref_gated_rmsnorm_fwd(x, c->w, c->z, y, c->r, c->rows, c->d, c->zc);
}
static void fx_gated_z(void *v, const real *z, real *y) {
  NormCtx *c = (NormCtx *)v;
  ref_gated_rmsnorm_fwd(c->x, c->w, z, y, c->r, c->rows, c->d, c->zc);
}
static void fx_l2norm(void *v, const real *x, real *y) {
  NormCtx *c = (NormCtx *)v;
  ref_l2norm_fwd(x, c->post_scale, y, c->r, c->rows, c->d);
}

// ---- RMSNorm: d/dx (design §4.3) ------------------------------------------------------------------------------------
static void case_rmsnorm(const char *name, int rows, int d, int zc, uint32_t seed) {
  real w[GR_MAXD], x[GR_MAXN], dy[GR_MAXN], y[GR_MAXN], gx[GR_MAXN];
  NormCtx c;
  size_t n = (size_t)rows * (size_t)d;
  rnd_fill(w, d, -0.6, 0.6, seed);
  rnd_fill(x, n, -1.0, 1.0, seed + 1);
  rnd_fill(dy, n, -1.0, 1.0, seed + 2);
  c.x = x;
  c.w = w;
  c.z = NULL;
  c.rows = rows;
  c.d = d;
  c.zc = zc;
  c.post_scale = REAL(0);
  ref_rmsnorm_fwd(x, w, y, c.r, rows, d, zc);            // the saved r the backward takes
  ref_rmsnorm_bwd(x, w, dy, c.r, gx, rows, d, zc);       // analytic
  jt_grad_check(name, fx_rmsnorm, &c, x, n, gx, dy, n);  // numeric vs analytic
}

// ---- Gated RMSNorm: d/dx and d/dz (design §4.4) ---------------------------------------------------------------------
static void case_gated_rmsnorm(const char *name, int rows, int d, int zc, uint32_t seed) {
  real w[GR_MAXD], x[GR_MAXN], z[GR_MAXN], dy[GR_MAXN], y[GR_MAXN], gx[GR_MAXN], gz[GR_MAXN];
  NormCtx c;
  size_t n = (size_t)rows * (size_t)d;
  char sub[160];
  rnd_fill(w, d, -0.6, 0.6, seed);
  rnd_fill(x, n, -1.0, 1.0, seed + 1);
  rnd_fill(z, n, -2.5, 2.5, seed + 2); // silu is smooth everywhere; this just covers both branches of it
  rnd_fill(dy, n, -1.0, 1.0, seed + 3);
  c.x = x;
  c.w = w;
  c.z = z;
  c.rows = rows;
  c.d = d;
  c.zc = zc;
  c.post_scale = REAL(0);
  ref_gated_rmsnorm_fwd(x, w, z, y, c.r, rows, d, zc);
  ref_gated_rmsnorm_bwd(x, w, z, dy, c.r, gx, gz, rows, d, zc);
  snprintf(sub, sizeof sub, "%s/d_x", name);
  jt_grad_check(sub, fx_gated_x, &c, x, n, gx, dy, n);
  snprintf(sub, sizeof sub, "%s/d_z", name);
  jt_grad_check(sub, fx_gated_z, &c, z, n, gz, dy, n);
}

// ---- L2 norm: d/dx, both post_scales the model uses (design §3.6/§4.8) ----------------------------------------------
static void case_l2norm(const char *name, int rows, int d, real post_scale, uint32_t seed) {
  real x[GR_MAXN], dy[GR_MAXN], y[GR_MAXN], gx[GR_MAXN];
  NormCtx c;
  size_t n = (size_t)rows * (size_t)d;
  rnd_fill(x, n, -1.0, 1.0, seed);
  rnd_fill(dy, n, -1.0, 1.0, seed + 1);
  c.x = x;
  c.w = NULL;
  c.z = NULL;
  c.rows = rows;
  c.d = d;
  c.zc = 0;
  c.post_scale = post_scale;
  ref_l2norm_fwd(x, post_scale, y, c.r, rows, d);
  ref_l2norm_bwd(x, dy, c.r, post_scale, gx, rows, d);
  jt_grad_check(name, fx_l2norm, &c, x, n, gx, dy, n);
}

int grad_norm(void) {
  // The model's shapes are d = 1024 (final norm, one row per token: a small number of rows here), d = 256 (q_norm /
  // k_norm) and d = 128 (per GDN head). Small rows/d for the rest: the gate does not depend on the size, and every
  // element is differenced either way.
  case_rmsnorm("rmsnorm_1024_zc1/d_x", 1, 1024, 1, 0x5eed01u);
  case_rmsnorm("rmsnorm_zc1/d_x", 2, 7, 1, 0x5eed02u);
  case_rmsnorm("rmsnorm_zc0/d_x", 2, 7, 0, 0x5eed03u);
  case_rmsnorm("rmsnorm_256_zc1/d_x", 2, 256, 1, 0x5eed04u);
  case_gated_rmsnorm("gated_zc0", 2, 128, 0, 0x5eed05u); // the model's GDN gated norm: multiply by w
  case_gated_rmsnorm("gated_zc1", 2, 128, 1, 0x5eed06u); // the same kernel with (1 + w)
  case_gated_rmsnorm("gated_small_zc0", 2, 7, 0, 0x5eed07u);
  case_l2norm("l2norm_d128_c128/d_x", 2, 128, (real)(1.0 / 11.31370849898476), 0x5eed08u); // 1/sqrt(128)
  case_l2norm("l2norm_d128_1/d_x", 2, 128, 1, 0x5eed09u);                                  // k: no extra scale
  case_l2norm("l2norm_small_1/d_x", 2, 7, 1, 0x5eed0au);
  return jt_exit_code();
}
