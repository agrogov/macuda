// test/k_norm.c — the `norm` group of test_kernels (plan M2.0/M2.2). GPU vs the CPU reference, fp32 both sides.
//
// Plan M2.2: rmsnorm_fwd/bwd with zero_centred both ways (the model's (1 + w) and the GDN gated norm's w), the gated
// RMSNorm's forward and both of its gradients, and l2norm_fwd/bwd with the 1/sqrt(128) scale as post_scale. Shapes are
// the model's (d = 1024 final norm, d = 256 q_norm/k_norm, d = 128 per GDN head with rows = tokens * 16) at the two
// ragged batch sizes of plan M2.0 — 168 and 2516 tokens — plus a single token and a row count that is not a multiple of
// the eight warps a block runs, so the partial-block path is exercised too.
//
// Gates: forward JT_TOL_FWD (1e-5), backward JT_TOL_BWD (1e-4). Both sides are fp32 and the only difference is the
// order of the row sum (the kernel's warp tree vs the reference's sequential double accumulation), so the measured
// error is a few 1e-7 and neither number is anywhere near being touched. On the null device CHECK_REL prints
// "launch only" and checks nothing, which is what make test-null needs.
#include <stdio.h>
#include <stdlib.h>

#include "harness.h"
#include "norm.h"        // kernels/norm.h — the launch wrappers
#include "../ref/norm.h" // ref/norm.h — the CPU references (same basename, other directory)

static float *fnew(size_t n) {
  float *p = (float *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_norm: out of memory (%zu floats)\n", n);
    exit(1);
  }
  return p;
}

// ---- RMSNorm (design §4.3): one block per row for d = 1024, one warp per row for d <= 256 --------------------------
static void case_rmsnorm(int rows, int d, int zc, uint32_t seed) {
  const size_t n = (size_t)rows * (size_t)d;
  char tag[96], nm[128];

  float *x = fnew(n), *w = fnew((size_t)d), *dy = fnew(n);
  float *y_ref = fnew(n), *r_ref = fnew((size_t)rows), *dx_ref = fnew(n);
  rnd_fill(x, n, -1.0, 1.0, seed);
  rnd_fill(w, d, -0.6, 0.6, seed + 1);
  rnd_fill(dy, n, -1.0, 1.0, seed + 2);
  ref_rmsnorm_fwd(x, w, y_ref, r_ref, rows, d, zc);
  ref_rmsnorm_bwd(x, w, dy, r_ref, dx_ref, rows, d, zc);

  float *xd = dev_upload(x, n), *wd = dev_upload(w, (size_t)d), *dyd = dev_upload(dy, n);
  float *yd = (float *)dev_alloc(n * sizeof(float)), *rd = (float *)dev_alloc((size_t)rows * sizeof(float));
  float *dxd = (float *)dev_alloc(n * sizeof(float));
  rmsnorm_fwd(xd, wd, yd, rd, rows, d, zc, 0);
  rmsnorm_bwd(xd, wd, dyd, rd, dxd, rows, d, zc, 0);

  float *got_y = fnew(n), *got_r = fnew((size_t)rows), *got_dx = fnew(n);
  dev_download(yd, got_y, n);
  dev_download(rd, got_r, (size_t)rows);
  dev_download(dxd, got_dx, n);

  snprintf(tag, sizeof tag, "rmsnorm rows=%d d=%d zc=%d", rows, d, zc);
  snprintf(nm, sizeof nm, "%s y", tag);
  CHECK_REL(nm, got_y, y_ref, n, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s r", tag);
  CHECK_REL(nm, got_r, r_ref, (size_t)rows, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s dx", tag);
  CHECK_REL(nm, got_dx, dx_ref, n, JT_TOL_BWD);

  dev_free(xd);
  dev_free(wd);
  dev_free(dyd);
  dev_free(yd);
  dev_free(rd);
  dev_free(dxd);
  free(x);
  free(w);
  free(dy);
  free(y_ref);
  free(r_ref);
  free(dx_ref);
  free(got_y);
  free(got_r);
  free(got_dx);
}

// ---- Gated RMSNorm (design §4.4): one warp per (token, head) -------------------------------------------------------
static void case_gated_rmsnorm(int rows, int d, int zc, uint32_t seed) {
  const size_t n = (size_t)rows * (size_t)d;
  char tag[96], nm[128];

  float *x = fnew(n), *w = fnew((size_t)d), *z = fnew(n), *dy = fnew(n);
  float *y_ref = fnew(n), *r_ref = fnew((size_t)rows), *dx_ref = fnew(n), *dz_ref = fnew(n);
  rnd_fill(x, n, -1.0, 1.0, seed);
  rnd_fill(w, d, -0.6, 0.6, seed + 1);
  rnd_fill(z, n, -2.5, 2.5, seed + 2);
  rnd_fill(dy, n, -1.0, 1.0, seed + 3);
  ref_gated_rmsnorm_fwd(x, w, z, y_ref, r_ref, rows, d, zc);
  ref_gated_rmsnorm_bwd(x, w, z, dy, r_ref, dx_ref, dz_ref, rows, d, zc);

  float *xd = dev_upload(x, n), *wd = dev_upload(w, (size_t)d), *zd = dev_upload(z, n), *dyd = dev_upload(dy, n);
  float *yd = (float *)dev_alloc(n * sizeof(float)), *rd = (float *)dev_alloc((size_t)rows * sizeof(float));
  float *dxd = (float *)dev_alloc(n * sizeof(float)), *dzd = (float *)dev_alloc(n * sizeof(float));
  gated_rmsnorm_fwd(xd, wd, zd, yd, rd, rows, d, zc, 0);
  gated_rmsnorm_bwd(xd, wd, zd, dyd, rd, dxd, dzd, rows, d, zc, 0);

  float *got_y = fnew(n), *got_r = fnew((size_t)rows), *got_dx = fnew(n), *got_dz = fnew(n);
  dev_download(yd, got_y, n);
  dev_download(rd, got_r, (size_t)rows);
  dev_download(dxd, got_dx, n);
  dev_download(dzd, got_dz, n);

  snprintf(tag, sizeof tag, "gated_rmsnorm rows=%d d=%d zc=%d", rows, d, zc);
  snprintf(nm, sizeof nm, "%s y", tag);
  CHECK_REL(nm, got_y, y_ref, n, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s r", tag);
  CHECK_REL(nm, got_r, r_ref, (size_t)rows, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s dx", tag);
  CHECK_REL(nm, got_dx, dx_ref, n, JT_TOL_BWD);
  snprintf(nm, sizeof nm, "%s dz", tag);
  CHECK_REL(nm, got_dz, dz_ref, n, JT_TOL_BWD);

  dev_free(xd);
  dev_free(wd);
  dev_free(zd);
  dev_free(dyd);
  dev_free(yd);
  dev_free(rd);
  dev_free(dxd);
  dev_free(dzd);
  free(x);
  free(w);
  free(z);
  free(dy);
  free(y_ref);
  free(r_ref);
  free(dx_ref);
  free(dz_ref);
  free(got_y);
  free(got_r);
  free(got_dx);
  free(got_dz);
}

// ---- L2 norm (design §3.3/§4.8): one warp per (token, head) --------------------------------------------------------
static void case_l2norm(int rows, int d, float post_scale, uint32_t seed) {
  const size_t n = (size_t)rows * (size_t)d;
  char tag[96], nm[128];

  float *x = fnew(n), *dy = fnew(n), *y_ref = fnew(n), *r_ref = fnew((size_t)rows), *dx_ref = fnew(n);
  rnd_fill(x, n, -1.0, 1.0, seed);
  rnd_fill(dy, n, -1.0, 1.0, seed + 1);
  ref_l2norm_fwd(x, post_scale, y_ref, r_ref, rows, d);
  ref_l2norm_bwd(x, dy, r_ref, post_scale, dx_ref, rows, d);

  float *xd = dev_upload(x, n), *dyd = dev_upload(dy, n);
  float *yd = (float *)dev_alloc(n * sizeof(float)), *rd = (float *)dev_alloc((size_t)rows * sizeof(float));
  float *dxd = (float *)dev_alloc(n * sizeof(float));
  l2norm_fwd(xd, post_scale, yd, rd, rows, d, 0);
  l2norm_bwd(xd, dyd, rd, post_scale, dxd, rows, d, 0);

  float *got_y = fnew(n), *got_r = fnew((size_t)rows), *got_dx = fnew(n);
  dev_download(yd, got_y, n);
  dev_download(rd, got_r, (size_t)rows);
  dev_download(dxd, got_dx, n);

  snprintf(tag, sizeof tag, "l2norm rows=%d d=%d scale=%g", rows, d, (double)post_scale);
  snprintf(nm, sizeof nm, "%s y", tag);
  CHECK_REL(nm, got_y, y_ref, n, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s r", tag);
  CHECK_REL(nm, got_r, r_ref, (size_t)rows, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "%s dx", tag);
  CHECK_REL(nm, got_dx, dx_ref, n, JT_TOL_BWD);

  dev_free(xd);
  dev_free(dyd);
  dev_free(yd);
  dev_free(rd);
  dev_free(dxd);
  free(x);
  free(dy);
  free(y_ref);
  free(r_ref);
  free(dx_ref);
  free(got_y);
  free(got_r);
  free(got_dx);
}

int k_norm(void) {
  // The two ragged token counts of plan M2.0: 168 and 2516 tokens.
  int t168 = 0, t2516 = 0;
  const int *lens3 = jt_lens_ragged3(), *lens4 = jt_lens_ragged4();
  for (int b = 0; b < JT_RAGGED3_ROWS; b++) t168 += lens3[b];
  for (int b = 0; b < JT_RAGGED4_ROWS; b++) t2516 += lens4[b];

  // d = 1024: the final norm, one 256-thread block per row (zero_centred both ways, one row, and the full batch).
  case_rmsnorm(t2516, 1024, 1, 0x5eed11u);
  case_rmsnorm(t2516, 1024, 0, 0x5eed12u);
  case_rmsnorm(t168, 1024, 1, 0x5eed13u);
  case_rmsnorm(1, 1024, 1, 0x5eed14u);
  // d = 256: q_norm/k_norm, one warp per row (8 elements per thread).
  case_rmsnorm(t2516, 256, 1, 0x5eed15u);
  case_rmsnorm(t2516, 256, 0, 0x5eed16u);
  case_rmsnorm(3, 256, 1, 0x5eed17u);
  // d = 128 per GDN head: rows = tokens * 16, one warp per row (both batches, and 13 rows so that the last
  // block's remaining seven warps have nothing to do).
  case_gated_rmsnorm(t2516 * 16, 128, 0, 0x5eed18u);
  case_gated_rmsnorm(t168 * 16, 128, 0, 0x5eed19u);
  case_gated_rmsnorm(t168 * 16, 128, 1, 0x5eed1eu);
  case_gated_rmsnorm(13, 128, 0, 0x5eed1au);
  // l2norm at the model's scale: q gets 1/sqrt(128), k gets 1 (both are the same kernel with post_scale).
  case_l2norm(t2516 * 16, 128, 0.08838834764831845f, 0x5eed1bu);
  case_l2norm(t168 * 16, 128, 1.0f, 0x5eed1cu);
  case_l2norm(3, 128, 1.0f, 0x5eed1du);
  return jt_exit_code();
}
