// test/k_rope.c — the `rope` group of test_kernels (plan M2.0/M2.4). GPU vs the CPU reference, fp32 both sides.
//
// Plan M2.4: rope_tables (built on the host in fp32, [lmax][64], then uploaded) and rope_fwd/bwd in place on
// [n][heads * 256], with the position t - row_off[row_of[t]] and only the first 64 dims of each head rotating.
//
// Checked here:
//   * the table builder against ref_rope_tables, at lmax = 1, 2, 130 and 1026 (the longest real row) — a one-ulp
//     difference in inv_freq would grow to ~1e-4 in cos at p = 1026, so this is a real check and not a formality;
//   * forward and backward against the references over the plan's two ragged batches (168 and 2516 tokens) plus a
//     single token, a head_dim 64 variant (the whole head rotates, no pass-through) and 5 heads (heads * 32 = 160 pairs,
//     so the 256-thread block runs partly idle);
//   * position 0 at every row's first token, where c = 1 and s = 0 make the forward the identity. A packed index used
//     as a position instead of t - off[row_of[t]] would rotate those tokens and show up here immediately.
//
// The device is fed the same table the reference is fed (rope_tables' host copy), so a difference between the two
// builders cannot hide inside the forward comparison. Gates: JT_TOL_FWD (1e-5) forward, JT_TOL_BWD (1e-4) backward, and
// the rotation itself is exact at p = 0 (CHECK_ABS with 1e-9). On the null device every CHECK_* prints "launch only".
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "harness.h"
#include "rope.h"        // kernels/rope.h — rope_tables, rope_fwd, rope_bwd
#include "../ref/rope.h" // ref/rope.h — the CPU references (same basename, other directory)

static float *fnew(size_t n) {
  float *p = (float *)calloc(n ? n : 1, sizeof *p);
  if (p == NULL) {
    fprintf(stderr, "k_rope: out of memory (%zu floats)\n", n);
    exit(1);
  }
  return p;
}

// ---- the tables (design §3.3) ---------------------------------------------------------------------------------------
static void case_tables(int lmax) {
  const size_t nt = (size_t)lmax * JT_ROPE_DIM;
  char nm[96];
  float *cos_k = fnew(nt), *sin_k = fnew(nt), *cos_r = fnew(nt), *sin_r = fnew(nt);

  rope_tables(cos_k, sin_k, lmax);      // the tables the model uploads
  ref_rope_tables(cos_r, sin_r, lmax);  // the reference's

  snprintf(nm, sizeof nm, "rope_tables lmax=%d cos", lmax);
  CHECK_REL(nm, cos_k, cos_r, nt, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "rope_tables lmax=%d sin", lmax);
  CHECK_REL(nm, sin_k, sin_r, nt, JT_TOL_FWD);

  free(cos_k), free(sin_k), free(cos_r), free(sin_r);
}

// ---- forward and backward on one packed batch ------------------------------------------------------------------------
static void case_rope(const char *tag, const int *lens, int nrows, int heads, int head_dim, uint32_t seed) {
  enum { JT_MAXTOK = 8192 }; // the longest fixture is the plan's 4-row batch: 520 + 670 + 1026 + 300 = 2516 tokens
  int off[JT_RAGGED4_ROWS + 1], row_of[JT_MAXTOK];
  int n = 0;
  for (int b = 0; b < nrows; b++) n += lens[b];
  if (nrows > JT_RAGGED4_ROWS || n > JT_MAXTOK || head_dim < JT_ROPE_DIM) {
    printf("k_rope: %s does not fit the fixture\n", tag);
    jt_fail_inc();
    return;
  }
  n = jt_ragged(lens, nrows, off, row_of);

  int lmax = 1;
  for (int b = 0; b < nrows; b++)
    if (lens[b] > lmax) lmax = lens[b];

  const size_t width = (size_t)heads * (size_t)head_dim;
  const size_t nv = (size_t)n * width;
  const size_t nt = (size_t)lmax * JT_ROPE_DIM;
  char tag2[64], nm[128];

  float *cos_h = fnew(nt), *sin_h = fnew(nt);
  rope_tables(cos_h, sin_h, lmax);

  float *x = fnew(nv), *dy = fnew(nv);
  float *y_ref = fnew(nv), *dx_ref = fnew(nv);
  rnd_fill(x, nv, -1.5, 1.5, seed);
  rnd_fill(dy, nv, -1.5, 1.5, seed ^ 0x9e3779b9u);

  memcpy(y_ref, x, nv * sizeof *y_ref);
  ref_rope_fwd(y_ref, off, row_of, n, heads, head_dim, cos_h, sin_h);
  memcpy(dx_ref, dy, nv * sizeof *dx_ref);
  ref_rope_bwd(dx_ref, off, row_of, n, heads, head_dim, cos_h, sin_h);

  float *xd = dev_upload(x, nv), *dyd = dev_upload(dy, nv);
  float *cosd = dev_upload(cos_h, nt), *sind = dev_upload(sin_h, nt);
  int *offd = (int *)dev_upload_bytes(off, (size_t)(nrows + 1) * sizeof *offd);
  int *rowd = (int *)dev_upload_bytes(row_of, (size_t)n * sizeof *rowd);
  rope_fwd(xd, offd, rowd, n, heads, head_dim, cosd, sind, 0);
  rope_bwd(dyd, offd, rowd, n, heads, head_dim, cosd, sind, 0);

  float *got_y = fnew(nv), *got_dx = fnew(nv);
  dev_download(xd, got_y, nv);
  dev_download(dyd, got_dx, nv);

  snprintf(tag2, sizeof tag2, "%s n=%d heads=%d d=%d", tag, n, heads, head_dim);
  snprintf(nm, sizeof nm, "rope_fwd %s", tag2);
  CHECK_REL(nm, got_y, y_ref, nv, JT_TOL_FWD);
  snprintf(nm, sizeof nm, "rope_bwd %s", tag2);
  CHECK_REL(nm, got_dx, dx_ref, nv, JT_TOL_BWD);

  // p = 0 at each row's first token: the rotation is the identity there (cos[0] = 1, sin[0] = 0), so y must be x bit
  // for bit. This is the check that the position is the row-local index and not the packed one (design §7).
  float *st_got = fnew((size_t)nrows * JT_ROPE_DIM), *st_want = fnew((size_t)nrows * JT_ROPE_DIM);
  for (int b = 0; b < nrows; b++) {
    const size_t base = (size_t)off[b] * width; // head 0 of the row's first token
    for (int j = 0; j < JT_ROPE_DIM; j++) {
      st_got[(size_t)b * JT_ROPE_DIM + j] = got_y[base + (size_t)j];
      st_want[(size_t)b * JT_ROPE_DIM + j] = x[base + (size_t)j];
    }
  }
  snprintf(nm, sizeof nm, "rope_fwd %s p=0 row starts", tag2);
  CHECK_ABS(nm, st_got, st_want, (size_t)nrows * JT_ROPE_DIM, 1e-9);

  dev_free(xd), dev_free(dyd), dev_free(cosd), dev_free(sind), dev_free(offd), dev_free(rowd);
  free(cos_h), free(sin_h), free(x), free(dy), free(y_ref), free(dx_ref);
  free(got_y), free(got_dx), free(st_got), free(st_want);
}

int k_rope(void) {
  static const int one[1] = {1};
  static const int small[3] = {1, 2, 3};

  case_tables(1);
  case_tables(2);
  case_tables(130);
  case_tables(1026);

  case_rope("ragged3", jt_lens_ragged3(), JT_RAGGED3_ROWS, 8, 256, 0x51ed270bu);
  case_rope("ragged4", jt_lens_ragged4(), JT_RAGGED4_ROWS, 8, 256, 0x1b873593u);
  case_rope("single", one, 1, 1, 256, 0x2545f491u);
  case_rope("d64", small, 3, 3, 64, 0x68e31da4u);
  case_rope("heads5", small, 3, 5, 256, 0x9e3779b1u);
  return jt_exit_code();
}
